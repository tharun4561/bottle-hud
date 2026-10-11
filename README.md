# Bottle Interceptor HUD

Real-time water bottle detection and interception targeting from a single webcam.

The program detects and tracks a bottle with YOLO26 (Ultralytics) and ByteTrack, estimates its 3D position and velocity in meters using a calibrated focal length, solves for the time an interceptor launched from the camera would need to reach it, and draws a targeting HUD with a lead line to the predicted aim point. It runs on the CPU with OpenCV and can also process recorded video files.

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

The YOLO26 nano model (`yolo26n.pt`, about 5 MB) downloads automatically the first time the program runs.

## Running

Live webcam:

```powershell
.venv\Scripts\python.exe main.py
```

A recorded video file:

```powershell
.venv\Scripts\python.exe main.py --video clip.mp4
```

In video mode, timestamps come from the file itself, so velocities are correct even though processing is slower than real time. The program exits by itself at the end of the file and prints a summary (frames processed, frames with a tracked bottle, distinct track IDs, processing speed and the current settings).

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
| `q` or the window's X button | Quit |

## Calibration

Distances come from the bottle's height in the image, which needs the camera's focal length in pixels. It's measured once per camera and resolution:

1. Stand the bottle upright exactly 1.00 m from the webcam lens, fully in view, with nothing else in the frame that looks like a bottle.
2. Press `c`. The program averages the box height over 30 frames (skipping frames with no bottle, two bottles, or a box touching the frame edge) and computes `f = h × D / H`.
3. The result is saved to `calibration.json` (ignored by Git, since it belongs to one camera) and loaded automatically from then on.

This project's calibration: bottle height **H = 0.142 m**, distance **D = 1.00 m**, average box height **h ≈ 87.4 px**, giving a focal length of **f = 615.3 px** at **640×480**.

Recalibrate after changing the camera, the resolution or the YOLO model. If the camera has an automatic framing or zoom feature (such as Windows Studio Effects), turn it off. The program warns if the frames don't match the resolution the calibration was made at.

## The HUD

### Panel rows

Positions use camera coordinates: the origin is the camera lens, **X** is positive to the right, **Y** positive up, and **Z** positive straight out of the camera.

| Row | Meaning |
| --- | --- |
| `DIST` | Distance to the bottle along the camera's view direction (its depth Z), in meters, from the smoothed box height |
| `POS` | The bottle's 3D position X / Y / Z, in meters |
| `TIME` | Time for the interceptor to reach the bottle, in seconds, from the intercept solve |
| `AIM` | The 3D aim point X / Y / Z, in meters: where the bottle will be when the interceptor arrives |
| `BRG H` | Horizontal bearing: angle from the camera's view direction, in degrees (right is positive) |
| `BRG V` | Vertical bearing, in degrees (up is positive) |
| `VEL` | The bottle's 3D velocity X / Y / Z, in m/s (a negative Z velocity means it's approaching) |
| `CONF` | YOLO's confidence for the target bottle, from 0 to 1 |
| `SPEED` | Interceptor speed, in m/s (change it with `+` and `-`) |
| `FPS` | Frames processed per second |

A value shows `--` when it can't be computed, and the yellow status line under the panel explains why:

| Status line | Meaning |
| --- | --- |
| `Not calibrated: hold bottle at 1.00 m, press c` | No focal length yet |
| `Bottle at edge: distance unreliable` | The box touches the top or bottom edge, so it may be cut off |
| `Speed is 0: press + to speed up` | An interceptor speed of 0 never arrives |
| `Measuring velocity...` | Collecting the first 10 positions of a newly tracked bottle |
| `No intercept: target too fast` | The bottle is at least as fast as the interceptor and moving away |
| `Aim point at/behind camera` | The aim point is closer than 0.20 m, so it can't be drawn |
| `Lead point off screen` | The aim point is outside the frame |
| `Calibrating 12/30: h 87` | Calibration progress (also `no bottle`, `2 bottles` or `at edge` when a frame is skipped) |
| `Saved: f 615.3 px` / `Calibration cancelled` | Calibration finished or was cancelled (shown for 3 seconds) |
| `Calibration only works with the webcam` | `c` was pressed in video mode |
| `Warning: 320x240, calibrated at 640x480` | The frames don't match the calibration's resolution, so distances will be wrong |

### On-screen graphics

| Graphic | Meaning |
| --- | --- |
| Green box with a `#7 0.85` label | A detected bottle: track ID and confidence (just the confidence if it isn't tracked yet) |
| Red crosshair at the bottom center | Where both lines start: a screen convention for the launch point (the math puts the interceptor at the camera itself) |
| Yellow line and dot | Target line to the bottle's current center |
| Magenta line and dot | Lead line to the projected aim point; a hollow circle at the frame edge means the point is off screen |

The HUD follows one target bottle: it stays locked on the same track ID while it's visible, and otherwise picks the most confident bottle.

## How it works

Each frame, the program detects and tracks bottles, then runs these steps for the target bottle. Pixel offsets `dx` and `dy` are measured from the image center (right and up positive), `h` is the box height in pixels and `f` is the focal length in pixels.

**Calibration** (similar triangles: the image is `h` pixels tall at distance `f`, the bottle is `H` meters tall at distance `D`):

```text
f = h × D / H
```

**Distance** (depth), using the box height smoothed with an exponential moving average (`h_smooth = 0.3 × h + 0.7 × h_smooth`):

```text
Z = f × H / h
```

**3D position and bearing:**

```text
X = dx × Z / f
Y = dy × Z / f
bearing = atan(offset / f)        (once for dx, once for dy)
```

**Velocity:** a least-squares straight-line fit through the last 10 positions, separately for X, Y and Z. The slope is the velocity. Timestamps are shifted to start at 0 for numerical precision, and they come from the clock (webcam) or from the file (video).

```text
v = Σ (t − t̄)(p − p̄) / Σ (t − t̄)²
```

**Intercept:** the interceptor starts at the camera and flies in a straight line at speed `s`. It meets the bottle (position `P`, velocity `V`) when its flight distance equals the bottle's distance at that time:

```text
|P + V·t| = s·t
(V·V − s²) t² + 2 (P·V) t + P·P = 0
```

The smallest positive root is the intercept time, and the aim point is `P + V·t`. When the interceptor is faster than the bottle there's always exactly one positive root. A still bottle gives `t = |P| / s`.

**Projection** of the aim point back onto the screen (`cx`, `cy` is the image center; pixel y grows downward, so Y is flipped):

```text
pixel_x = cx + f × X / Z
pixel_y = cy − f × Y / Z
```

## Settings

The main settings are constants at the top of `main.py`:

| Constant | Value | Meaning |
| --- | --- | --- |
| `CONF_THRESHOLD` | 0.1 | Minimum YOLO confidence. ByteTrack starts new tracks only at 0.25 or above and uses weaker detections (down to 0.1) to keep existing tracks |
| `IMGSZ` | 640 | Image size YOLO works at: smaller is faster but less accurate for small or far bottles (must be a multiple of 32) |
| `BOTTLE_HEIGHT_M` | 0.142 | Real height of the bottle, in meters |
| `KNOWN_DISTANCE_M` | 1.00 | Calibration distance, in meters |
| `VELOCITY_WINDOW` | 10 | Positions per velocity fit: more is steadier but slower to react |
| `MIN_AIM_Z_M` | 0.20 | Aim points closer than this aren't drawn |
| `DEFAULT_SPEED_MPS` | 5.0 | Starting interceptor speed, in m/s |
| `SPEED_STEP_MPS` | 0.5 | Change per `+` or `-` press, in m/s |

To compare settings fairly, record a few clips with `record.py` and run each setting on the same clips with `--video`; the summary at the end shows how often the bottle was tracked and how many track IDs it got.

## Known limitations

- **Bottle size:** distance assumes a 0.142 m bottle. A different bottle gives a proportionally wrong distance, and YOLO's "bottle" class accepts any bottle.
- **Box height:** tilt, rotation, a hand covering part of the bottle, motion blur and how tightly YOLO fits the box all change `h`, and therefore the distance. Boxes touching the top or bottom edge are excluded.
- **Range:** depth error grows with the square of the distance (about ±2 cm at 1 m, about ±9 cm at 2 m). The practical range is about 0.2 to 3 m.
- **Simple camera model:** no lens distortion correction, and the image center is assumed to be the optical center. The calibration is only valid for this camera at 640×480, with automatic framing turned off.
- **Fixed camera:** if the camera moves, the whole scene appears to move, and that shows up as bottle velocity.
- **Straight-line motion only:** no acceleration or gravity, so a thrown bottle's arc isn't predicted. The velocity reflects the last 10 samples (about 0.5 to 0.9 s), so it lags behind changes, and it shows `--` for a bottle's first 10 samples.
- **Idealized interceptor:** it starts at the camera, reaches full speed instantly, flies straight and has no launch delay. The processing delay (about 60 to 80 ms per frame) isn't compensated, so positions are slightly out of date.
- **Lead line:** it starts at the crosshair as a screen convention, not along the real flight path. Aim points closer than 0.20 m or behind the camera aren't drawn, and off-screen ones are cut at the frame edge.
- **Tracking:** IDs can switch when bottles cross; a bottle needs one detection at 0.25 confidence or above before it's tracked; calibration needs exactly one bottle in view.
- **Performance:** CPU only, about 11 to 19 FPS on the development laptop. GPU acceleration through PyTorch (CUDA) needs an NVIDIA card.
- **Video mode:** footage must come from the same camera at the same resolution as the calibration.

## Files

| File | Purpose |
| --- | --- |
| `main.py` | The HUD program |
| `record.py` | Records test clips from the webcam |
| `requirements.txt` | Python packages with tested versions |
| `NOTES.md` | Development notes and answers for each milestone |
| `calibration.json` | Saved calibration (created by pressing `c`; not in Git) |
| `yolo26n.pt` | YOLO26 nano model (downloaded on first run; not in Git) |
