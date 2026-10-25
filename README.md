# Interceptor HUD

Real-time drone detection and interception targeting from a single webcam.

The program detects and tracks the target (a small drone) with a YOLO26 nano model trained on Roboflow (Ultralytics) and ByteTrack, estimates its 3D position and velocity in meters using a calibrated focal length, solves for the time an interceptor launched from the camera would need to reach it, and draws a targeting HUD with a lead line to the predicted aim point. It runs on the CPU with OpenCV and can also process recorded video files.

## Requirements

- Windows 11 (tested), with a built-in or USB webcam
- Python 3.14 (tested with 3.14.7)
- No GPU needed; everything runs on the CPU

The Python packages are listed with tested versions in [`requirements.txt`](requirements.txt): `ultralytics`, `opencv-python`, `numpy` and `lap` (used by the ByteTrack tracker). PyTorch is installed automatically as a dependency of `ultralytics`.

## Setup

Create a virtual environment and install the packages into it:

```powershell
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The trained model, `PavoWeights.pt` (about 5 MB, one class: `Drone-Interceptor`), is not in Git (`*.pt` files are ignored): copy it into the project folder before the first run. The program never downloads it.

## Running

Live webcam:

```powershell
.venv\Scripts\python.exe main.py
```

A recorded video file:

```powershell
.venv\Scripts\python.exe main.py --video clip.mp4
```

In video mode, timestamps come from the file itself, so velocities are correct even though processing is slower than real time. The program exits by itself at the end of the file and prints a summary (frames processed, frames with a tracked target, distinct track IDs, processing speed and the current settings). Add `--loop` to start the video again at the end instead.

Other options: `--model PATH` uses a different YOLO model file (relative paths are taken from the project folder), and `--screen N` puts the window on monitor N (`--screen` alone lists the monitors). Run with `--help` for the full list.

### Demo mode

Double-click **`demo.bat`** (or run it from any terminal; nothing needs activating). It runs `main.py --demo` with the project's own Python. If the webcam fails, double-click **`demo_backup.bat`**: it replays `demo_clip.mp4` from the project folder in a loop (`--demo --video demo_clip.mp4 --loop`). Extra options are passed on, e.g. `demo.bat --screen 2`.

Demo mode:

- Runs pre-flight checks before the window opens, printing one line per problem: `FAIL` (model file missing, webcam or video won't open, no frames) stops the program; `WARN` (not calibrated, resolution doesn't match the calibration) doesn't.
- Never downloads anything: the model file must already be in the project folder, and the launchers set `YOLO_OFFLINE=1` so Ultralytics skips its internet check.
- Warms up the model before the window opens and prints the startup time, e.g. `Ready in 10.5 s (Python + imports 4.5 s, checks + camera 1.4 s, model 0.1 s, warm-up 4.2 s, window 0.3 s)`. The first launch after a reboot is slower.
- Starts full screen with the debug overlay off and the key legend showing for 8 seconds.
- Needs `q` pressed twice (within 3 seconds) to quit, so one stray key press can't end it.
- Keeps the laptop from sleeping or turning the screen off.
- Keeps the window up if the webcam disconnects, showing "Camera lost: reconnecting..." and trying again every second.
- Saves the details of any crash to `demo_error.log` and prints one line instead of a traceback.

**On the projector:** the simplest setup is Windows' **Duplicate** mode (Win+P), which shows the same picture on the laptop and the projector. With **Extend**, use `demo.bat --screen 2` (check the number with `--screen` first), or drag the window onto the projector and press `f`. Full screen always fills the monitor the window is on, at that monitor's own resolution. Keys only work while the HUD window has focus; click it if they seem dead.

**Before the demo:** plug in the charger and set the power mode to Best performance; turn on Do not disturb; close Zoom, Teams and any browser tab that might use the camera; turn off Windows Studio Effects; bring the drone and a tape measure (to recalibrate at 1.00 m if needed); copy `demo_clip.mp4` into the project folder; and do one full launch at the venue before going on.

### Recording test clips

```powershell
.venv\Scripts\python.exe record.py clip.mp4
```

Press `q` to stop. `record.py` records at the webcam's default resolution (the same one `main.py` uses, so the calibration stays valid) and stores the frame rate the webcam actually delivered, so the clip's timestamps are correct. It won't overwrite an existing file unless you add `--overwrite`. Video files are ignored by Git.

## Controls

| Key | Action |
| --- | --- |
| `c` | Start calibration; press again to cancel (webcam only) |
| `+` or `=` | Increase the interceptor speed by 0.5 m/s |
| `-` or `_` | Decrease the interceptor speed by 0.5 m/s (not below 0) |
| `r` | Reset the lock, so the most confident target is picked again |
| `d` | Show or hide the debug overlay (hidden at start) |
| `h` | Show or hide the key legend (shown for the first 8 seconds) |
| `f` | Switch between full screen and a window |
| `q` or the window's X button | Quit (in demo mode, press `q` twice) |

## Calibration

Distances come from how wide the drone's box is in the image (`SIZE_AXIS` in `tracker.py`), which needs the camera's focal length in pixels. It's measured once per camera and resolution:

1. Hold the drone exactly 1.00 m from the webcam lens, facing the camera, fully in view, with nothing else in the frame that looks like it.
2. Press `c`. The program averages the box width over 30 frames (skipping frames with no target, two targets, or a box touching the left or right edge) and computes `f = w × D / S`, where `S = 0.097 m` is the drone's real width.
3. The result is saved to `calibration.json` (ignored by Git, since it belongs to one camera) and loaded automatically from then on.

Calibrate again with the drone before trusting any distance: a `calibration.json` made earlier with the bottle used a different object size.

The drone is 97 × 97 mm and about 30 mm tall. The width is used because it spans about three times as many pixels as the height, so a pixel of noise matters less. A square body seen at an angle looks up to 1.4 times wider, so keep it facing the camera when calibrating. To measure the height instead, set `SIZE_AXIS = "height"` and `TARGET_SIZE_M = 0.030` in `tracker.py`.

Recalibrate after changing the camera, the resolution or the YOLO model. If the camera has an automatic framing or zoom feature (such as Windows Studio Effects), turn it off. The program warns if the frames don't match the resolution the calibration was made at.

## The HUD

The HUD is drawn by `hud.py` for an audience. The camera frame is scaled up before anything is drawn, so text stays sharp: in full screen to the screen's own resolution, otherwise to 1920×1080 (`DISPLAY_SIZE`). The panel goes in a strip on the right instead of covering the video (a 4:3 webcam leaves one on a 16:9 screen; on 16:10 the video shrinks slightly to make room).

### Lock state

The banner at the top shows the lock state, and the target gets corner brackets in the same color:

| State | Meaning |
| --- | --- |
| `SEARCHING` (cyan) | No tracked target. An untracked detection gets thin cyan brackets |
| `LOCKED DRONE-INTERCEPTOR #7` (green) | Following track 7. A miss shorter than 0.2 s (`LOST_GRACE_S`) keeps the lock, so one dropped detection doesn't flash the banner |
| `LOST DRONE-INTERCEPTOR #7` (red, blinking) | The target disappeared; its last position blinks until 2 s (`LOST_HOLD_S`) after it was last seen |

### Panel rows

| Row | Meaning |
| --- | --- |
| `DISTANCE` | Distance to the target along the camera's view direction (its depth Z), in meters, from the smoothed box height |
| `BEARING` | Angle from the camera's view direction, in whole degrees left/right and up/down |
| `INTERCEPT IN` | Time for the interceptor to reach the target, in seconds, from the intercept solve |
| `CONFIDENCE` | YOLO's confidence for the target |
| `FPS` | Frames processed per second |
| `MODE` | `LIVE` (webcam), `VIDEO` (`--video`) or `CALIBRATING 12/30` |

### Debug overlay

Press `d` for the engineering values. Positions use camera coordinates: the origin is the camera lens, **X** is positive to the right, **Y** positive up, and **Z** positive straight out of the camera.

| Row | Meaning |
| --- | --- |
| `h raw` / `h smooth` | The target's box height this frame, and its smoothed value, in pixels |
| `POS` | The target's 3D position X / Y / Z, in meters |
| `VEL` | Its 3D velocity X / Y / Z, in m/s (a negative Z velocity means it's approaching) |
| `AIM` | The 3D aim point X / Y / Z, in meters: where the target will be when the interceptor arrives |
| `BRG` | Horizontal / vertical bearing, in degrees (right and up positive) |
| `SPEED` | Interceptor speed, in m/s (change it with `+` and `-`; each change is also shown on the status line) |
| `f` | Calibrated focal length, and the resolution it was made at |
| `LEAD` | Whether the lead point is on screen |
| `capture`, `detect + math`, `hud draw` | Smoothed time per stage, in milliseconds |

Every detection also gets a `#7 0.85` label (track ID and confidence).

### Status line

A value shows `--` when it can't be computed, and the yellow status line under the banner explains why:

| Status line | Meaning |
| --- | --- |
| `Not calibrated: hold Drone-Interceptor at 1.00 m, press c` | No focal length yet |
| `Drone-interceptor at edge: distance unreliable` | The box touches the left or right edge, so it may be cut off |
| `Speed is 0: press + to speed up` | An interceptor speed of 0 never arrives |
| `Measuring velocity...` | Waiting for the first 5 measurements of a newly tracked target |
| `No intercept: target too fast` | The target is at least as fast as the interceptor and moving away |
| `Aim point at/behind camera` | The aim point is closer than 0.20 m, so it can't be drawn |
| `Lead point off screen` | The aim point is outside the frame |
| `Calibrating 12/30: size 87` | Calibration progress, with the box width in pixels (also `no Drone-Interceptor`, `2 Drone-Interceptors` or `at edge` when a frame is skipped) |
| `Saved: f 615.3 px` / `Calibration cancelled` | Calibration finished or was cancelled (shown for 3 seconds) |
| `Calibration only works with the webcam` | `c` was pressed in video mode |
| `Warning: 320x240, calibrated at 640x480` | The frames don't match the calibration's resolution, so distances will be wrong |
| `Interceptor 5.5 m/s` | The speed was changed with `+` or `-` (shown for 3 seconds) |
| `Lock reset: reacquiring` | `r` was pressed |
| `Press q again to quit` | `q` was pressed once in demo mode |
| `Camera lost: reconnecting...` | The webcam stopped sending frames; the last frame stays up until it's back |

### On-screen graphics

| Graphic | Meaning |
| --- | --- |
| Corner brackets with `DRONE-INTERCEPTOR #7` | The target, in the lock state's color |
| Thin gray corner brackets | Other detections |
| Red crosshair at the bottom center | Where both lines start: a screen convention for the launch point (the math puts the interceptor at the camera itself) |
| Yellow line and dot | Target line to the target's current center |
| Magenta line and reticle | Lead line to the projected aim point; a hollow circle at the frame edge means the point is off screen |

The HUD follows one target: it stays locked on the same track ID while it's visible, and otherwise picks the most confident detection. The target class is `TARGET_CLASS` in `tracker.py`; changing it also changes every label and message. Switching to another object also needs its real size in `TARGET_SIZE_M` (and `SIZE_AXIS`) and a new calibration.

## How it works

`tracker.py` detects and tracks the target and measures its 3D position and velocity, `intercept.py` solves the intercept, and `main.py` connects them to the camera, the keys and the HUD. Each frame runs these steps for the target. Pixel offsets `dx` and `dy` are measured from the image center (right and up positive), `w` is the box width in pixels (the side set by `SIZE_AXIS`), `S` is the target's real size along that side and `f` is the focal length in pixels.

**Calibration** (similar triangles: the image is `w` pixels wide at distance `f`, the target is `S` meters wide at distance `D`):

```text
f = w × D / S
```

**Distance** (depth), using the box width smoothed with an exponential moving average (`w_smooth = 0.3 × w + 0.7 × w_smooth`):

```text
Z = f × S / w
```

**3D position and bearing:**

```text
X = dx × Z / f
Y = dy × Z / f
bearing = atan(offset / f)        (once for dx, once for dy)
```

**Position and velocity:** a Kalman filter, run separately for X, Y and Z. Each axis tracks a position and a velocity and assumes the target moves at a roughly constant velocity. Each frame it predicts forward by the real time since the last measurement, then blends in the new raw position, trusting it less when the target is far (the depth error grows with the square of the distance: `σ_Z = Z² × σ_w / (f × S)`). The velocity is shown after 5 measurements. Timestamps come from the clock (webcam) or from the file (video).

```text
predict:  p = p + v·dt          (the uncertainty grows by the allowed acceleration)
update:   p = p + K·(measured − p),  v = v + K_v·(measured − p)
```

The filter starts over when the target's track ID changes or the target is missing for more than 0.5 s.

**Intercept:** the interceptor starts at the camera and flies in a straight line at speed `s`. It meets the target (position `P`, velocity `V`) when its flight distance equals the target's distance at that time:

```text
|P + V·t| = s·t
(V·V − s²) t² + 2 (P·V) t + P·P = 0
```

The smallest positive root is the intercept time, and the aim point is `P + V·t`. When the interceptor is faster than the target there's always exactly one positive root. A still target gives `t = |P| / s`.

**Projection** of the aim point back onto the screen (`cx`, `cy` is the image center; pixel y grows downward, so Y is flipped):

```text
pixel_x = cx + f × X / Z
pixel_y = cy − f × Y / Z
```

## Settings

The main settings are constants at the top of the module that uses them:

| Constant | Value | File | Meaning |
| --- | --- | --- | --- |
| `MODEL_PATH` | `PavoWeights.pt` | `tracker.py` | The YOLO model (override with `--model`) |
| `TARGET_CLASS` | `Drone-Interceptor` | `tracker.py` | The model's class to track |
| `CONF_THRESHOLD` | 0.01 | `tracker.py` | Minimum YOLO confidence. ByteTrack starts new tracks only at 0.03 or above and uses weaker detections (down to 0.01) to keep existing tracks. Those two limits are set in `bytetrack_low.yaml` (the Ultralytics defaults are 0.25 and 0.1) |
| `IMGSZ` | 640 | `tracker.py` | Image size YOLO works at: smaller is faster but less accurate for small or far targets (must be a multiple of 32) |
| `SIZE_AXIS` | `"width"` | `tracker.py` | Which side of the box gives the distance: `"width"` or `"height"` |
| `TARGET_SIZE_M` | 0.097 | `tracker.py` | Real size of the target along `SIZE_AXIS`, in meters (0.030 for the height) |
| `KNOWN_DISTANCE_M` | 1.00 | `tracker.py` | Calibration distance, in meters |
| `ACCEL_SIGMA_MPS2` | 1.0 | `tracker.py` | Kalman process noise: how much the velocity may change per second (bigger follows turns faster but is noisier) |
| `CENTER_SIGMA_PX`, `SIZE_SIGMA_PX` | 1.0, 1.5 | `tracker.py` | How much the box center and width jitter, in pixels (bigger trusts each measurement less: smoother but slower) |
| `WARMUP_UPDATES` | 5 | `tracker.py` | Measurements before the velocity is shown |
| `MIN_AIM_Z_M` | 0.20 | `main.py` | Aim points closer than this aren't drawn |
| `DEFAULT_SPEED_MPS` | 5.0 | `intercept.py` | Starting interceptor speed, in m/s |
| `SPEED_STEP_MPS` | 0.5 | `intercept.py` | Change per `+` or `-` press, in m/s |

The HUD's settings (display size, text sizes, colors, lock timing, `DEBUG_DEFAULT`) are constants at the top of `hud.py`.

To compare settings fairly, record a few clips with `record.py` and run each setting on the same clips with `--video`; the summary at the end shows how often the target was tracked and how many track IDs it got.

## Known limitations

- **Target size:** distance assumes the drone's real width is 0.097 m. A different drone gives a proportionally wrong distance.
- **Box width:** turning (a square body looks up to 1.4 times wider at an angle), a hand covering part of the drone, motion blur and how tightly YOLO fits the box all change `w`, and therefore the distance. Boxes touching the left or right edge are excluded. At 2 m the drone is only about 29 px wide, so one pixel is about 3.4% of the distance.
- **Range:** depth error grows with the square of the distance (about ±2 cm at 1 m, about ±9 cm at 2 m). The practical range is about 0.2 to 3 m.
- **Simple camera model:** no lens distortion correction, and the image center is assumed to be the optical center. The calibration is only valid for this camera at 640×480, with automatic framing turned off.
- **Fixed camera:** if the camera moves, the whole scene appears to move, and that shows up as target velocity.
- **Straight-line motion only:** the intercept assumes a constant velocity, so a turning target's path isn't predicted. The filter follows turns within a few frames but lags a little, and the velocity shows `--` for a target's first 5 measurements.
- **Idealized interceptor:** it starts at the camera, reaches full speed instantly, flies straight and has no launch delay. The processing delay (about 60 to 80 ms per frame) isn't compensated, so positions are slightly out of date.
- **Lead line:** it starts at the crosshair as a screen convention, not along the real flight path. Aim points closer than 0.20 m or behind the camera aren't drawn, and off-screen ones are cut at the frame edge.
- **Tracking:** IDs can switch when targets cross; a target needs one detection at 0.03 confidence or above before it's tracked (low thresholds also let more false detections through, so raise them in `bytetrack_low.yaml` if other objects get a box); calibration needs exactly one target in view.
- **Performance:** CPU only, about 11 to 19 FPS on the development laptop. GPU acceleration through PyTorch (CUDA) needs an NVIDIA card.
- **Video mode:** footage must come from the same camera at the same resolution as the calibration.
- **Demo mode:** `r` and each `--loop` restart only reset this program's lock; ByteTrack keeps its own tracks, so track IDs keep counting up. A webcam that freezes without reporting an error (rather than disconnecting) isn't detected.

## Files

| File | Purpose |
| --- | --- |
| `main.py` | The program: camera or video input, the main loop, the keys and demo mode |
| `hud.py` | All HUD drawing |
| `screen.py` | Windows helpers: monitors, full screen, keep-awake, startup time |
| `demo.bat` / `demo_backup.bat` | Demo launchers (live webcam / looping backup clip) |
| `demo_clip.mp4` | The backup clip (record it with `record.py`; not in Git) |
| `demo_error.log` | Crash details from demo mode, if any (not in Git) |
| `tracker.py` | Detection, track ID lock, calibration, Kalman-filtered 3D position and velocity |
| `intercept.py` | Intercept time, aim point and bearing angles; pure math, no camera or drawing |
| `bytetrack_low.yaml` | ByteTrack settings with lowered confidence thresholds, used by `tracker.py` |
| `record.py` | Records test clips from the webcam |
| `requirements.txt` | Python packages with tested versions |
| `NOTES.md` | Development notes and answers for each milestone |
| `calibration.json` | Saved calibration (created by pressing `c`; not in Git) |
| `PavoWeights.pt` | The trained drone model (copy it in; not in Git) |
