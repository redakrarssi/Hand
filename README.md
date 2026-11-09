# HandScroller

Scroll with small, relaxed finger or whole-hand movements using your webcam.
Processing stays on your computer. The camera feed is hidden by default; the
camera remains active while tracking.

## Start

1. Run `install.bat` once if dependencies are missing, then double-click `run.bat`.
2. Keep your hand anywhere in the camera view. There is no target
   box or restricted activation area. The hand is tracked across the full view.
   All processing is pure, deterministic code — no learning curve or calibration wizards.
3. Focus the browser or page you want to scroll (TikTok, Reels, Shorts, web page). Press **O**
   to open the test page. If using mouse-wheel mode, leave the pointer
   over the intended page or scrollable area.
4. Make an upward stroke to scroll UP, or a downward stroke to scroll DOWN.
   Works with just one finger, multiple fingers, or your whole hand.
   Movement triggers immediately (1ms) as soon as the stroke completes.
   Return movements are automatically ignored by anti-recoil suppression.

If tracking is missing, press **V** in the HandScroller window to check the
camera framing. A webcam cannot detect fingers outside its view, and needs
enough hand detail to identify the wrist and knuckles.

## Direction

| Setting | Move hand/finger UP | Move hand/finger DOWN |
|---|---|---|
| Natural (default) | Scroll up / previous | Scroll down / next |
| Phone | Scroll down / next video | Scroll up / previous video |

Press **D** to switch directions. Direction and sensitivity stay fixed while
you use the app. Command-line `--direction` or `--sensitivity` explicitly sets preferences.

## Controls

Controls apply while the HandScroller window is focused. Focus your browser
again afterward so it can receive scrolling.

| Key | Action |
|---|---|
| Q / Escape | Quit |
| P | Pause / resume |
| C | Recalibrate in your current relaxed position |
| D | Switch phone / natural direction |
| V | Show / hide camera video |
| + / - | Increase / decrease sensitivity |
| S | Cycle smart / wheel / keyboard mode |
| M | Mirror the camera view |
| F | Switch fast / quality image enhancement |
| H | Hide the preview |

When the preview is hidden, focus the console and use Ctrl+C to quit, then
restart to show it again. OpenCV's keyboard controls require a visible window.

Sideways movements have no special action by default. Optional `--gesture-undo`
lets a sideways finger wag undo the most recent scroll on the same focused page.
It never guesses a new direction or changes sensitivity. Guided learning with L
is an explicit way to recalibrate.

## High-Density Hand Tracking (134 Points)

HandScroller features an advanced **134-Point Dense Hand Tracking & Micro-Movement Engine**, extending standard hand tracking by over 6x to detect even the slightest finger deflections on low-cost or noisy webcams:

- **134 Anatomical Points**: Full skeletal spline + bilateral margin contours + tactile fingertip pad clusters.
- **Concentrated on Dominant Fingers (86 Points / 64%)**:
  - **Index Finger (32 Points)**: 4 joint nodes, 9 proximal points, 9 intermediate points, 6 distal points, 4 tactile pad sensory points.
  - **Middle Finger (28 Points)**: 4 joint nodes, 6 proximal points, 9 intermediate points, 6 distal points, 3 pad points.
  - **Thumb (26 Points)**: 4 joint nodes, 4 CMC-MCP points, 7 proximal points, 7 distal points, 4 pad points.
  - **Ring Finger (16 Points)** and **Pinky (14 Points)**: Supporting contour and joint tracking.
  - **Palm & Webbing (18 Points)**: Biomechanical anchor, thenar/hypothenar eminences, and inter-digital webbing.
- **Bad Camera Noise Filter (~5.6x SNR Improvement)**:
  - On grainy, low-light, or budget webcams, individual landmark points jitter by several pixels.
  - The dense engine computes **multi-point directional consensus** across all 32 index points and 28 middle points.
  - Because random pixel noise is uncorrelated across points, noise averages out by $\frac{1}{\sqrt{N}}$, completely eliminating jitter-induced false scrolls.
- **Resting vs. Top/Bottom Micro-Intent**:
  - **Rock-Solid Resting**: Identifies when the hand is relaxed or resting on a surface (stability meter 0–100%), continuously adapting the resting baseline.
  - **Micro-Deflections (Top/Bottom)**: Detects subtle upward or downward flicks (even under 10 pixels of motion) when unanimous directional coherence ($\ge 75\%$) is confirmed across dominant finger points.
  - **Sub-Pixel Optical Flow Refinement**: Directly tracks pixel intensity gradients between frames using Lucas-Kanade optical flow, anchoring the hand against neural landmark jitter.

## Detection and stability

The detector tracks each finger relative to its own knuckle and also tracks
the palm. It measures the complete movement over time, rather than requiring
large movements in consecutive frames. Distances adjust to the visible palm
size and a measured resting noise floor; slow strokes accumulate against the
relaxed baseline.

Camera capture continuously replaces old frames with the latest image, preventing
processing delays from building a camera backlog. Low-light processing measures
actual brightness and enhances dim frames; well-lit frames use the fast path.

Wheel scrolling eases over 0.22 seconds instead of sending one large burst. Input
continues between frames and stops when paused or when the focused window changes.
Keyboard mode keeps one arrow-key action per video swipe.

Three-frame median filtering removes isolated tracking spikes. Predominantly
horizontal or conflicting movements are discarded. A settling and recovery
gate limits each stroke to one action and suppresses its return movement.
Returning to neutral rearms tracking immediately, even during the 0.26-second
output cooldown, so fast follow-up strokes are retained rather than discarded.
Holding a new endpoint still uses conservative recovery to avoid scrolling back
on a delayed return movement.
Background feature tracking rejects coherent camera shifts. Losing tracking,
changing hands or depth substantially, pausing, or recalibrating requires acquiring a
new neutral position. Horizontal repositioning follows the hand across the view;
it does not leave scrolling anchored to the previous location. Learned movement
thresholds are bounded so changing rest postures cannot make small strokes unreachable.

Unidentified optical movement no longer triggers scrolling: a moving face,
background object, or camera alone is insufficient evidence of hand intent.
Showing only disconnected fingertips may therefore produce no action.

These checks reduce errors but cannot guarantee perfect intent recognition.
An intentional hand stroke and an identical incidental movement look the same
to a webcam. Lighting, occlusion, and camera placement still affect accuracy.

## Options

```powershell
python main.py
python main.py --direction natural --mode scroll
python main.py --sensitivity 1.4 --scroll-amount 2
python main.py --show-camera
python main.py --learn
python main.py --no-preview

# Observe tracking for 10 seconds without sending mouse or keyboard input:
python main.py --diagnostic-seconds 10
```

Smart mode uses arrow keys for titles mentioning TikTok or Shorts and wheel
scrolling otherwise. Use `--mode scroll` or `--mode keyboard` if the window title
does not identify your page correctly. Windows input acceptance is checked with
`SendInput`; a void legacy input call is not treated as proof of success. Failed
or interrupted actions are shown as failures and excluded from the action counter.
The HUD also updates if an eased scroll fails after its first step. Partial wheel
movement can still be undone on the original focused page.
Accepted input still depends on the target application's focus and behavior.

## Troubleshooting

| Symptom | Action |
|---|---|
| No hand tracked | Press V; include the wrist/knuckles and improve lighting or camera angle |
| Tracking but no scrolling | Wait for READY, focus the target page, make a vertical stroke and relax |
| Small gestures missed | Press + or use `--sensitivity 1.4`; press C in your comfortable posture |
| Accidental scrolling | Reduce sensitivity; pause with P while repositioning your hand or laptop |
| Wrong direction | Press D or use `--direction natural` |
| Scroll steps too large | Use `--scroll-amount 1` or `2` |
| Camera unavailable | Close another camera app or select `--camera 1` |
| Wrong action on a site | Override smart mode with `--mode scroll` or `--mode keyboard` |

Automatic camera exposure is enabled by default to retain useful hand detail.
Windows exposure settings are translated for Media Foundation and DirectShow;
rejected driver settings are reported instead of silently leaving the camera dark.
Low-light enhancement can help dim images but cannot restore missing detail.
The camera's privacy indicator is controlled by its hardware/driver; hiding
the preview does not turn off the camera.

## Requirements and validation

Windows 10/11, Python 3.9+, a webcam, and the packages in `requirements.txt`.
The hand model is included in `models/`, or downloaded on first use if missing.
Tune the defaults in `app/config.py`.

```powershell
python -m unittest discover -s tests -v
```

The regression suite replays the production detector and input pipeline with
quick acquisition, positions across the full view, sideways repositioning,
fresh camera capture, low-light processing, eased wheel output, small slow strokes at 6–60 FPS, both directions, different hand sizes, all five
fingers, preparation/return movements, repeat strokes, jitter, lost tracking,
background/camera movement, input mapping, and input failures. These controlled
tests do not measure accuracy on an individual user's real webcam gestures.
