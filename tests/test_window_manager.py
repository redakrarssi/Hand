"""
Unit tests for Window & Focus Manager in HandScroller.
Verifies terminal console detection, HUD exclusion, browser prioritization,
and focus switching mechanisms.
"""

import unittest
from unittest.mock import patch, MagicMock

from app.window_manager import (
    is_console_title,
    is_hud_title,
    find_target_browser_hwnd,
    ensure_target_active,
)


class WindowManagerTests(unittest.TestCase):
    def test_console_detection(self):
        consoles = [
            'Windows PowerShell',
            'Administrator: Windows PowerShell',
            'cmd.exe',
            'C:\\WINDOWS\\system32\\cmd.exe - run.bat',
            'Command Prompt',
            'Windows Terminal',
            'bash',
            'conhost.exe',
        ]
        for title in consoles:
            self.assertTrue(is_console_title(title), f"Failed to identify console: {title}")

        browsers = [
            '🖐️ HandScroller Interactive Test Playground - Google Chrome',
            'TikTok - Make Your Day - Microsoft Edge',
            'YouTube Shorts - Firefox',
            'Reddit - Dive into anything',
            'Wikipedia, the free encyclopedia',
        ]
        for title in browsers:
            self.assertFalse(is_console_title(title), f"False positive console detection: {title}")

    def test_hud_detection(self):
        huds = [
            'HandScroller',
            'HandScroller (Privacy Mode)',
            '✋ HandScroller',
            '✋ HandScroller (Privacy Mode)',
            'âœ‹ HandScroller (Privacy Mode)',
        ]
        for title in huds:
            self.assertTrue(is_hud_title(title), f"Failed to identify HUD: {title}")

        non_huds = [
            'HandScroller Interactive Test Playground - Google Chrome',
            'HandScroller Documentation - Edge',
            'Visual Studio Code',
        ]
        for title in non_huds:
            self.assertFalse(is_hud_title(title), f"False positive HUD detection: {title}")

    @patch('app.window_manager._user32')
    def test_target_browser_selection_priority(self, mock_user32):
        # Mock EnumWindows to simulate several open windows
        windows = [
            (101, 'C:\\WINDOWS\\system32\\cmd.exe'),
            (102, 'HandScroller (Privacy Mode)'),
            (103, 'Program Manager'),
            (104, 'Google Chrome - New Tab'),
            (105, '🖐️ HandScroller Interactive Test Playground - Google Chrome'),
            (106, 'TikTok - Microsoft Edge'),
        ]

        mock_user32.IsWindowVisible.return_value = True

        def fake_enum_windows(callback, lparam):
            for hwnd, title in windows:
                with patch('app.window_manager.get_window_title', return_value=title):
                    callback(hwnd, lparam)
            return True

        mock_user32.EnumWindows.side_effect = fake_enum_windows

        best = find_target_browser_hwnd()
        # Should pick HWND 105 (the exact test playground) with score 100
        self.assertEqual(best, 105)

    @patch('app.window_manager.get_foreground_title', return_value='Administrator: Windows PowerShell')
    @patch('app.window_manager.activate_target_browser', return_value=True)
    def test_ensure_target_active_redirects_from_console(self, mock_activate, mock_title):
        self.assertTrue(ensure_target_active())
        mock_activate.assert_called_once()

    @patch('app.window_manager.get_foreground_title', return_value='HandScroller (Privacy Mode)')
    @patch('app.window_manager.activate_target_browser', return_value=True)
    def test_ensure_target_active_redirects_from_hud(self, mock_activate, mock_title):
        self.assertTrue(ensure_target_active())
        mock_activate.assert_called_once()

    @patch('app.window_manager.get_foreground_title', return_value='TikTok - Google Chrome')
    @patch('app.window_manager.activate_target_browser')
    def test_ensure_target_active_leaves_valid_browser_alone(self, mock_activate, mock_title):
        self.assertTrue(ensure_target_active())
        mock_activate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
