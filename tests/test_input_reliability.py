"""Exercise real dispatch wrappers without injecting desktop input."""

import ctypes
import unittest
from unittest.mock import Mock, patch

from app.gesture_engine import GestureResult
from app.input_simulator import INPUT, INPUT_MOUSE, INPUT_KEYBOARD, InputSimulator, _send_input


class NativeDispatchTests(unittest.TestCase):
    def test_acceptance_comes_from_sendinput_not_void_legacy_apis(self):
        for kind in (INPUT_MOUSE, INPUT_KEYBOARD):
            for accepted in (0, 1):
                with self.subTest(kind=kind, accepted=accepted):
                    api = Mock()
                    api.SendInput.return_value = accepted
                    inp = INPUT()
                    inp.type = kind
                    with patch('app.input_simulator._user32', api):
                        self.assertEqual(_send_input(inp), accepted == 1)
                    api.SendInput.assert_called_once()
                    self.assertEqual(api.SendInput.call_args.args[2], ctypes.sizeof(INPUT))
                    api.mouse_event.assert_not_called()
                    api.keybd_event.assert_not_called()

    def test_sendinput_exception_is_failure_without_unchecked_retry(self):
        api = Mock()
        api.SendInput.side_effect = OSError('injection failed')
        with patch('app.input_simulator._user32', api):
            self.assertFalse(_send_input(INPUT()))
        api.mouse_event.assert_not_called()
        api.keybd_event.assert_not_called()


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.focus = 123
        for name, kwargs in (
            ('_get_foreground_handle', {'side_effect': lambda: self.focus}),
            ('_get_foreground_window_title', {'return_value': 'Browser'}),
            ('_ensure_cursor_over_target', {'return_value': True}),
        ):
            p = patch('app.input_simulator.' + name, **kwargs)
            p.start()
            self.addCleanup(p.stop)
        p = patch('app.window_manager.activate_target_browser', return_value=True)
        self.activate = p.start()
        self.addCleanup(p.stop)

    def test_late_failure_replaces_success_and_partial_distance_remains_undoable(self):
        with patch.object(InputSimulator, '_send_wheel_delta', side_effect=[True, False, True]) as send:
            sim = InputSimulator('scroll')
            action = sim.execute_gesture(GestureResult('SWIPE_UP'))
            self.assertTrue(action['success'])
            self.assertFalse(sim.update_scroll(sim._wheel.started + 1))
            self.assertFalse(action['success'])
            self.assertEqual(action['status'], 'interrupted')
            self.assertEqual(action['wheel_units'], 120)
            self.assertEqual(sim.total_actions, 0)
            self.assertIs(sim.pop_scroll_error(), action)
            self.assertIsNone(sim.pop_scroll_error())
            sim.update_scroll(sim._wheel.started + 2)
            self.assertEqual(sim.total_actions, 0)
            self.assertTrue(sim.undo_action(action))
            self.assertEqual(send.call_args.args[0], -120)

    def test_focus_change_stops_continuation_and_undo(self):
        with patch.object(InputSimulator, '_send_wheel_delta', return_value=True) as send:
            sim = InputSimulator('scroll')
            action = sim.execute_gesture(GestureResult('SWIPE_UP'))
            self.focus = 456
            self.assertFalse(sim.update_scroll(sim._wheel.started + 0.1))
            self.assertFalse(action['success'])
            self.assertFalse(sim.undo_action(action))
            send.assert_called_once()
            self.assertIn('focused window changed', action['error'])

    def test_no_foreground_target_never_sends_or_activates_another_app(self):
        self.focus = None
        with patch('app.input_simulator._send_input') as send:
            sim = InputSimulator('scroll')
            self.assertFalse(sim.execute_gesture(GestureResult('SWIPE_UP'))['success'])
            self.assertFalse(sim.get_stats()['target_ready'])
            send.assert_not_called()
            self.activate.assert_not_called()

    def test_hud_never_redirects_to_an_arbitrary_browser(self):
        with patch('app.input_simulator._get_foreground_window_title', return_value='HandScroller (Privacy Mode)'), patch('app.input_simulator._send_input') as send:
            sim = InputSimulator('scroll')
            self.assertFalse(sim.execute_gesture(GestureResult('SWIPE_UP'))['success'])
            self.assertFalse(sim.get_stats()['target_ready'])
            send.assert_not_called()
            self.activate.assert_not_called()

    def test_console_documentation_in_browser_is_a_valid_target(self):
        with patch('app.input_simulator._get_foreground_window_title', return_value='PowerShell Terminal Bash manual - Google Chrome'), patch('app.input_simulator._send_input', return_value=True):
            sim = InputSimulator('scroll')
            self.assertTrue(sim.execute_gesture(GestureResult('SWIPE_UP'))['success'])
            self.assertTrue(sim.get_stats()['target_ready'])
            self.activate.assert_not_called()

    def test_completed_wheel_action_keeps_exact_distance(self):
        with patch.object(InputSimulator, '_send_wheel_delta', return_value=True):
            sim = InputSimulator('scroll')
            action = sim.execute_gesture(GestureResult('SWIPE_UP'))
            self.assertTrue(sim.update_scroll(sim._wheel.started + 1))
            self.assertTrue(action['success'])
            self.assertEqual(action['status'], 'completed')
            self.assertEqual(action['wheel_units'], sim.scroll_amount * 120)
            self.assertEqual(sim.total_actions, 1)
            self.assertIsNone(sim.pop_scroll_error())


class TargetAndFeedbackTests(unittest.TestCase):
    def test_window_class_overrides_document_title(self):
        from app.window_manager import target_error
        api = Mock()
        for window_class, title, blocked in (
            ('Chrome_WidgetWin_1', 'PowerShell', False),
            ('MozillaWindowClass', 'Terminal documentation', False),
            ('ConsoleWindowClass', 'python main.py', True),
            ('CASCADIA_HOSTING_WINDOW_CLASS', 'my arbitrary shell tab', True),
        ):
            def class_name(hwnd, buffer, size):
                buffer.value = window_class
                return len(window_class)
            api.GetClassNameW.side_effect = class_name
            with self.subTest(window_class=window_class), patch('app.window_manager._user32', api):
                self.assertEqual(target_error(123, title) is not None, blocked)

    def cursor_api(self, hovered):
        api = Mock()
        api.GetForegroundWindow.return_value = 123
        def rect(hwnd, pointer):
            pointer._obj.left, pointer._obj.top = 0, 0
            pointer._obj.right, pointer._obj.bottom = 1000, 800
            return 1
        def point(pointer):
            pointer._obj.x, pointer._obj.y = 500, 400
            return 1
        api.GetWindowRect.side_effect = rect
        api.GetCursorPos.side_effect = point
        api.WindowFromPoint.return_value = hovered
        api.GetAncestor.side_effect = lambda window, flags: window
        return api

    def test_overlapping_window_is_not_mistaken_for_target_body(self):
        from app.input_simulator import _ensure_cursor_over_target
        api = self.cursor_api(456)
        with patch('app.input_simulator._user32', api):
            self.assertFalse(_ensure_cursor_over_target(123))
        api.SetCursorPos.assert_not_called()

    def test_pointer_already_over_target_stays_put(self):
        from app.input_simulator import _ensure_cursor_over_target
        api = self.cursor_api(123)
        with patch('app.input_simulator._user32', api):
            self.assertTrue(_ensure_cursor_over_target(123))
        api.SetCursorPos.assert_not_called()

    def test_explicit_test_page_open_has_no_focus_worker(self):
        from app.learning_curve import open_test_page
        with patch('app.learning_curve.webbrowser.open', return_value=True) as browser, patch('threading.Thread') as worker:
            self.assertTrue(open_test_page())
            self.assertTrue(browser.call_args.args[0].startswith('file:///'))
            worker.assert_not_called()
        with patch('app.learning_curve.webbrowser.open', return_value=False):
            self.assertFalse(open_test_page())

    def test_late_error_replaces_hud_feedback_once(self):
        from main import update_scroll_feedback
        sim, overlay = Mock(), Mock()
        action = {'gesture': 'SWIPE_UP', 'description': 'Scroll interrupted', 'error': 'Focus changed'}
        sim.pop_scroll_error.side_effect = [action, None]
        update_scroll_feedback(sim, overlay)
        update_scroll_feedback(sim, overlay)
        overlay.trigger_gesture_flash.assert_called_once_with('SWIPE_UP', 'Scroll interrupted', success=False)

    def test_calibrated_startup_does_not_open_browser(self):
        import main
        from app.learning_curve import UserProfile
        with patch('sys.argv', ['main.py', '--no-preview']), \
             patch('main.setup_camera', return_value=Mock()), \
             patch('main.GestureEngine') as engine, patch('main.LowLightEnhancer'), \
             patch('main.InputSimulator') as simulator, patch('main.UIOverlay'), \
             patch('main.load_profile', return_value=UserProfile(calibrated=True)), \
             patch('main.LatestFrameCamera') as camera, \
             patch('main.open_test_page') as browser, \
             patch('main.cv2.destroyAllWindows'):
            engine.return_value.sensitivity = 1.0
            simulator.return_value.pop_scroll_error.return_value = None
            camera.return_value.read.return_value = (False, None)
            main.main()
            browser.assert_not_called()


if __name__ == '__main__':
    unittest.main()
