"""Bottle Interceptor HUD.

Detects and tracks a water bottle with YOLO, measures its 3D position and
velocity from a single calibrated webcam, solves for the intercept time of
an interceptor launched from the camera, and draws a targeting HUD with a
lead line to the predicted aim point.

Usage:
    .venv\\Scripts\\python.exe main.py                     (webcam)
    .venv\\Scripts\\python.exe main.py --video clip.mp4    (recorded file)
    .venv\\Scripts\\python.exe main.py --demo              (full screen demo)
Run with --help for all the options; demo.bat and demo_backup.bat start
the demo with a double-click.

Press c to calibrate (press c again to cancel; webcam only).
Press + / - to change the interceptor speed, r to reset the lock.
Press d for the debug overlay, h for the key legend, f for full screen.
Press q (or close the window) to quit; twice in demo mode.

The drawing is in hud.py and the Windows screen helpers in screen.py; this
file does the detection, tracking and math.
"""

import argparse  # reading command-line options like --video
import json  # reading and writing calibration.json
import math  # math functions for the bearing formula and the intercept
import sys  # used to exit the program with an error code
import time  # high-precision clock for FPS and webcam timestamps
import traceback  # saving crash details to a file in demo mode
from collections import deque  # a list that's quick to trim from the front
from datetime import datetime  # timestamp saved with the calibration
from pathlib import Path  # building file paths that work on any OS

import cv2  # OpenCV: camera access, windows, and image drawing
import numpy as np  # a black frame for the model warm-up
from ultralytics import YOLO  # Ultralytics: loads and runs YOLO models

import hud  # all the HUD drawing
import screen  # monitors, full screen and keep-awake on Windows

CAMERA_INDEX = 0  # 0 = the first camera Windows finds (the built-in webcam)
# Windows' DirectShow opens the webcam in about 1 s; the default backend
# (Media Foundation) took about 9 s here. The default is still tried if
# DirectShow can't open the camera.
CAMERA_BACKEND = cv2.CAP_DSHOW
WINDOW_NAME = "Interceptor HUD"  # the title shown on the window
WINDOW_SIZE = (1280, 720)  # starting window size; the HUD itself is bigger

# Demo mode.
QUIT_CONFIRM_S = 3.0  # in demo mode, q must be pressed twice within this
RECONNECT_INTERVAL_S = 1.0  # how often to try reopening a lost webcam
# Crash details in demo mode, next to main.py.
ERROR_LOG_PATH = Path(__file__).parent / "demo_error.log"

MODEL_PATH = "yolo26n.pt"  # YOLO26 nano; downloaded automatically on first run
TARGET_CLASS = "bottle"  # the only class we want to show
CONF_THRESHOLD = 0.1  # hide detections the model is less sure about than this
IMGSZ = 640  # image size YOLO works at; smaller is faster but sees less detail
# Tracker that gives each bottle an ID that stays the same across frames.
# ByteTrack is lighter than the default BoT-SORT, which spends extra time
# compensating for a moving camera (ours doesn't move).
TRACKER_CONFIG = "bytetrack.yaml"

# Lead point: the 3D aim point projected onto the screen.
# Aim points closer than this (Z, in meters) aren't drawn. Projecting divides
# by Z, so points at or behind the camera can't be drawn, and the whole
# bottle only fits in the frame from about 0.18 m away anyway.
MIN_AIM_Z_M = 0.20

# 3D velocity (m/s) from a least-squares fit over the last N positions.
VELOCITY_WINDOW = 10  # samples per fit; more = steadier but slower to react
LOST_TIMEOUT_S = 0.5  # missing longer than this (s) resets the history

# Status line.
STATUS_MESSAGE_SECONDS = 3.0  # how long "Saved"/"Cancelled" messages stay up

# Calibration.
BOTTLE_HEIGHT_M = 0.142  # real height of the bottle, in meters
KNOWN_DISTANCE_M = 1.00  # distance from the webcam lens to the bottle, in meters
CALIBRATION_FRAMES = 30  # how many frames of h to average
CALIBRATION_FILE = "calibration.json"
EDGE_MARGIN = 5  # boxes this close (px) to the top/bottom edge may be cut off

# Keep calibration.json next to main.py, no matter which folder we run from.
CALIBRATION_PATH = Path(__file__).parent / CALIBRATION_FILE

# Interceptor.
DEFAULT_SPEED_MPS = 5.0  # starting interceptor speed, in meters per second
SPEED_STEP_MPS = 0.5  # how much each + or - press changes the speed

# Smoothing (exponential moving averages). Between 0 and 1: smaller is
# steadier but slower to react, larger is quicker but jumpier.
SMOOTHING_ALPHA = 0.3  # for the bottle's box height h
FPS_SMOOTHING = 0.1  # for the frame time (FPS)
# YOLO's first run is a slow warm-up (seconds, not milliseconds). Leave it out
# of the FPS, or the smoothing would take a long time to forget it.
WARMUP_FRAMES = 1


def compute_focal_length(h_px, known_distance_m, bottle_height_m):
    """Return the camera's focal length in pixels.

    h_px: the bottle's box height in the image, in pixels
    known_distance_m: distance from the camera to the bottle, in meters
    bottle_height_m: the bottle's real height, in meters
    """
    f = (h_px * known_distance_m) / bottle_height_m
    return f


def compute_distance(h_px, focal_length_px, bottle_height_m):
    """Return the distance from the camera to the bottle, in meters.

    h_px: the bottle's box height in the image, in pixels
    focal_length_px: the calibrated focal length, in pixels
    bottle_height_m: the bottle's real height, in meters
    """
    d = (bottle_height_m * focal_length_px) / h_px
    return d


def compute_bearing(offset_px, focal_length_px):
    """Return the bearing angle to the bottle along one axis, in degrees.

    Used for both directions: the caller passes the horizontal offset for
    the horizontal bearing and the vertical offset for the vertical one.

    offset_px: signed offset of the bottle's center from the image center
        along one axis, in pixels (right = positive horizontally,
        up = positive vertically; the caller already handles the sign)
    focal_length_px: the calibrated focal length, in pixels

    Returns degrees with the same sign as offset_px (0 when centered).
    """
    theta_rad = math.atan(offset_px / focal_length_px)
    return math.degrees(theta_rad)


def compute_3d_position(offset_x_px, offset_y_px, distance_m, focal_length_px):
    """Return the bottle's 3D position relative to the camera, in meters.

    Coordinate system: the origin is at the camera lens. X is positive to
    the right, Y is positive up, and Z is positive straight out of the
    camera (the direction it's pointing).

    offset_x_px: horizontal offset of the bottle's center from the image
        center, in pixels (right = positive)
    offset_y_px: vertical offset of the bottle's center from the image
        center, in pixels (up = positive)
    distance_m: the bottle's distance along the camera's view direction
        (its depth), in meters
    focal_length_px: the calibrated focal length, in pixels

    Returns (X, Y, Z) in meters.
    """
    X = offset_x_px * distance_m / focal_length_px
    Y = offset_y_px * distance_m / focal_length_px
    Z = distance_m
    return X, Y, Z


def project_to_pixels(X, Y, Z, focal_length_px, center_x, center_y):
    """Return where a 3D point appears on screen, in pixels.

    The reverse of compute_3d_position: it takes a point in camera
    coordinates and finds the pixel it lands on.

    X, Y, Z: the point in meters, camera coordinates (X right, Y up, Z away
        from the camera). The caller only calls this with Z above
        MIN_AIM_Z_M, so Z is never 0 or negative.
    focal_length_px: the calibrated focal length, in pixels
    center_x, center_y: the image center, in pixels

    Returns (pixel_x, pixel_y) as floats, in image coordinates: pixel x grows
    to the right and pixel y grows DOWNWARD, so "up" (positive Y) has to be
    flipped.
    """
    offset_x = focal_length_px * X / Z
    offset_y = focal_length_px * Y / Z
    pixel_x = center_x + offset_x
    pixel_y = center_y - offset_y
    return pixel_x, pixel_y


def fit_velocity(times_s, positions_m):
    """Return the velocity along one axis from a least-squares line fit.

    Fits the straight line that comes closest to all the (time, position)
    points at once, and returns its slope. Called once per axis (X, Y, Z).

    times_s: sample times in seconds, already shifted so they're small
        numbers (the caller subtracts the first timestamp)
    positions_m: positions along one axis at those times, in meters (same
        length as times_s)

    Returns the slope in meters per second (0.0 if all the times are equal).
    """
    n = len(times_s)
    t_mean = sum(times_s) / n
    p_mean = sum(positions_m) / n

    numerator = 0.0
    denominator = 0.0
    for t, p in zip(times_s, positions_m):
        numerator += (t - t_mean) * (p - p_mean)
        denominator += (t - t_mean) ** 2

    if denominator == 0:
        return 0.0
    return numerator / denominator


def solve_intercept_time(P, V, speed_mps):
    """Return the earliest time the interceptor can reach the moving bottle.

    The interceptor starts at the camera (the origin) and flies in a straight
    line at a constant speed. The bottle moves in a straight line at a
    constant velocity. The intercept is when the interceptor's flight
    distance (speed * t) equals the bottle's distance from the start point
    at that time.

    P: the bottle's position now, (x, y, z) in meters, measured from the
        interceptor's start point (X right, Y up, Z away from the camera)
    V: the bottle's velocity, (vx, vy, vz) in meters per second, same axes
    speed_mps: interceptor speed in meters per second (the caller only
        calls this with a speed above 0)

    Note: when the interceptor speed equals the bottle's speed, the t^2
    term of the equation vanishes, so it has to be solved as a straight
    line instead (dividing by that term would crash).

    Returns the smallest positive t in seconds, or None if there is no
    valid intercept.
    """
    # Step 1: dot products
    VV = V[0]*V[0] + V[1]*V[1] + V[2]*V[2]   # V·V, bottle speed squared
    PV = P[0]*V[0] + P[1]*V[1] + P[2]*V[2]   # P·V
    PP = P[0]*P[0] + P[1]*P[1] + P[2]*P[2]   # P·P, distance squared

    # Step 2: quadratic coefficients for a*t^2 + b*t + c = 0
    a = VV - speed_mps ** 2
    b = 2 * PV
    c = PP

    # Step 3: if a is ~0, the t^2 term vanishes; solve b*t + c = 0
    if abs(a) < 1e-9:
        if b == 0:
            return None
        t = -c / b
        return t if t > 0 else None

    # Step 4: negative discriminant means no real roots, no intercept
    disc = b * b - 4 * a * c
    if disc < 0:
        return None

    # Step 5: both roots from the quadratic formula
    root = math.sqrt(disc)
    t1 = (-b - root) / (2 * a)
    t2 = (-b + root) / (2 * a)

    # Step 6: smallest positive root, or None
    positive = [t for t in (t1, t2) if t > 0]
    if not positive:
        return None
    return min(positive)


def signed(value, decimals):
    # Format a number with a + or - sign, e.g. signed(0.1234, 2) = "+0.12".
    # Rounding first and adding 0.0 turns a tiny negative like -0.001 into
    # +0.00 instead of "-0.00" (Python keeps the minus sign on a rounded 0).
    return f"{round(value, decimals) + 0.0:+.{decimals}f}"


def touches_edge(y1, y2, frame_height):
    # A box touching the top or bottom edge is probably cut off, which makes
    # h too small (and the distance too large).
    return y1 <= EDGE_MARGIN or y2 >= frame_height - EDGE_MARGIN


def ema(old, new, alpha):
    # Exponential moving average: mix a fraction (alpha) of the new value
    # into the running value, so one noisy value can only move it a little.
    # With no running value yet, start from the new value.
    if old is None:
        return new
    return alpha * new + (1 - alpha) * old


def smooth_h(smoothed_h, new_h):
    # Smooth the bottle's box height with SMOOTHING_ALPHA.
    return ema(smoothed_h, new_h, SMOOTHING_ALPHA)


def choose_target(bottles, locked_id):
    # Pick the bottle to measure and aim at. Each bottle is
    # (x1, y1, x2, y2, conf, track_id). If the bottle we were already
    # following (locked_id) is still visible, keep it, even if another one
    # is more confident right now. Otherwise take the most confident one.
    if not bottles:
        return None
    if locked_id is not None:
        for bottle in bottles:
            if bottle[5] == locked_id:
                return bottle
    return max(bottles, key=lambda b: b[4])  # index 4 is conf


def measurement_values(target, frame_height, focal_length, speed, smoothed_h):
    # Work out the distance for the panel. target is the bottle as
    # (x1, y1, x2, y2, conf, track_id), or None if there's no bottle.
    # smoothed_h is the smoothed box height, or None if there isn't one (then
    # the raw h of target is used).
    # Returns (distance_text, distance, reason). distance is the number in
    # meters, or None. reason explains a "--" for the status line (including
    # a speed of 0, which blocks the intercept), or is None when there's
    # nothing to explain.
    if focal_length is None:
        return "--", None, (f"Not calibrated: hold {TARGET_CLASS} at "
                            f"{KNOWN_DISTANCE_M:.2f} m, press c")
    if target is None:
        return "--", None, None

    x1, y1, x2, y2 = target[:4]
    if touches_edge(y1, y2, frame_height):
        return "--", None, (f"{TARGET_CLASS.capitalize()} at edge: "
                            "distance unreliable")

    # Prefer the smoothed h; fall back to this frame's raw h.
    h = smoothed_h if smoothed_h is not None else y2 - y1

    # compute_distance divides by h, so a 0-pixel box would crash it.
    if h <= 0:
        return "--", None, None

    distance = compute_distance(h, focal_length, BOTTLE_HEIGHT_M)
    distance_text = f"{distance:.2f} m"

    # With a speed of 0 the interceptor never arrives, so there's no
    # intercept; say why here.
    if speed <= 0:
        return distance_text, distance, "Speed is 0: press + to speed up"

    return distance_text, distance, None


def box_center(target):
    # The middle of the box: halfway between the left and right edges, and
    # halfway between the top and bottom edges.
    x1, y1, x2, y2 = target[:4]
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def center_offsets(target, frame):
    # Offset of the bottle's (raw) box center from the image center, in
    # pixels, as (dx, dy) with right = positive and up = positive. Shared by
    # the bearings and the 3D position so they always agree.

    # The image center is where the camera points (its optical axis).
    height, width = frame.shape[:2]
    cx = width / 2
    cy = height / 2

    bx, by = box_center(target)

    # Image x grows to the right, so right is already positive.
    dx = bx - cx
    # Image y grows DOWNWARD, so subtract the other way round to make up
    # positive.
    dy = cy - by
    return dx, dy


def bearing_values(target, frame, focal_length):
    # Work out the horizontal and vertical bearing angles for the panel.
    # Returns (bearing_h, bearing_v) in degrees (right and up positive), or
    # None. The reason for a None (not calibrated) already comes from
    # measurement_values.
    if target is None or focal_length is None:
        return None

    dx, dy = center_offsets(target, frame)

    bearing_h = compute_bearing(dx, focal_length)
    bearing_v = compute_bearing(dy, focal_length)
    return bearing_h, bearing_v


def format_xyz(point):
    # Format a 3D point in meters for the panel, like "+0.10 / -0.03 / 1.02 m".
    # X and Y always show a sign; Z only shows one if it's negative (behind
    # the camera), since it's normally positive.
    X, Y, Z = point
    return f"{signed(X, 2)} / {signed(Y, 2)} / {Z:.2f} m"


def position_values(target, frame, focal_length, distance):
    # Work out the 3D position from the raw box center and the (smoothed)
    # distance. Returns (text, position): position is (X, Y, Z) in meters,
    # or None, with text "--", when there's no distance (no bottle, not
    # calibrated, or at the edge).
    if target is None or focal_length is None or distance is None:
        return "--", None

    dx, dy = center_offsets(target, frame)
    position = compute_3d_position(dx, dy, distance, focal_length)
    return format_xyz(position), position


def intercept_values(position, velocity_3d, speed):
    # Work out the intercept time and 3D aim point for the panel.
    # Returns (time_text, aim_text, aim, reason). aim is (X, Y, Z) in meters,
    # or None when there's no intercept. reason explains a "--" for the
    # status line, or is None if there's nothing to add (no position or
    # speed 0 is already explained by measurement_values).
    if position is None or speed <= 0:
        return "--", "--", None, None
    if velocity_3d is None:
        return "--", "--", None, "Measuring velocity..."

    t = solve_intercept_time(position, velocity_3d, speed)
    if t is None:
        # With one positive root whenever the interceptor is faster, None
        # only happens when the bottle is at least as fast as it.
        return "--", "--", None, "No intercept: target too fast"

    # Aim point: where the bottle will be after t seconds, P + V*t.
    aim = tuple(p + v * t for p, v in zip(position, velocity_3d))
    return f"{t:.2f} s", format_xyz(aim), aim, None


def load_calibration():
    # Returns (focal_length, frame_size). focal_length is None if there's no
    # usable calibration. frame_size is (width, height) of the frames it was
    # made with, or None if the file doesn't say (files saved before the
    # size was recorded).

    # No file yet means the camera hasn't been calibrated.
    if not CALIBRATION_PATH.exists():
        return None, None

    # If the file is broken (bad JSON, missing field, not a number), treat it
    # as "not calibrated" instead of crashing.
    try:
        with open(CALIBRATION_PATH) as f:
            data = json.load(f)
        focal_length = float(data["focal_length_px"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        print(f"WARN: could not read {CALIBRATION_PATH}")
        return None, None

    # The frame size is optional, so a missing or odd value just means
    # "unknown" rather than a broken file.
    try:
        frame_size = (int(data["frame_width"]), int(data["frame_height"]))
    except (KeyError, TypeError, ValueError):
        frame_size = None
    return focal_length, frame_size


def save_calibration(focal_length, avg_h, frame_size):
    # Save the result together with the values used to get it, so the file
    # explains itself later. frame_size is (width, height) in pixels: the
    # focal length in pixels is only valid at that resolution.
    data = {
        "focal_length_px": focal_length,
        "avg_box_height_px": avg_h,
        "known_distance_m": KNOWN_DISTANCE_M,
        "bottle_height_m": BOTTLE_HEIGHT_M,
        "frames_averaged": CALIBRATION_FRAMES,
        "frame_width": frame_size[0],
        "frame_height": frame_size[1],
        "calibrated_at": datetime.now().isoformat(timespec="seconds"),
    }
    # indent=2 puts each field on its own line so it's easy to read.
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(data, f, indent=2)


def parse_args(argv):
    # Read the command-line options. argv is the list of options (None means
    # "use the ones the program was started with").
    parser = argparse.ArgumentParser(
        description="Bottle detection and interceptor targeting HUD.")
    parser.add_argument(
        "--video", metavar="PATH",
        help="run on a recorded video file instead of the webcam")
    parser.add_argument(
        "--loop", action="store_true",
        help="with --video: start the video again when it ends")
    parser.add_argument(
        "--demo", action="store_true",
        help="demo mode: full screen, q twice to quit, keep the laptop "
             "awake, never download the model")
    parser.add_argument(
        "--model", metavar="PATH", default=MODEL_PATH,
        help=f"YOLO model file (default: {MODEL_PATH} in the project folder)")
    parser.add_argument(
        "--screen", metavar="N", type=int, nargs="?", const=0,
        help="put the window on monitor N (1 = the main one); without N, "
             "list the monitors")
    args = parser.parse_args(argv)
    if args.loop and args.video is None:
        parser.error("--loop only works with --video")
    return args


def model_file(path):
    # A relative model path is taken from main.py's folder (like
    # calibration.json), so the model is found no matter which folder the
    # program is started from.
    path = Path(path)
    return path if path.is_absolute() else Path(__file__).parent / path


def key_legend(video_mode, demo):
    # The keys for the on-screen legend, as (key, action) pairs, matching
    # what the key handling in main() does.
    keys = [] if video_mode else [("c", "calibrate")]
    keys += [
        ("+ / -", "interceptor speed"),
        ("r", "reset lock"),
        ("d", "debug overlay"),
        ("f", "full screen"),
        ("h", "show/hide keys"),
        ("q", "quit (press twice)" if demo else "quit"),
    ]
    return keys


def open_camera():
    # Open the webcam: DirectShow first (fast on Windows), then the default
    # backend. Returns the capture, or None if neither works.
    for backend in (CAMERA_BACKEND, cv2.CAP_ANY):
        cap = cv2.VideoCapture(CAMERA_INDEX, backend)
        if cap.isOpened():
            return cap
        cap.release()
    return None


def preflight(args, model_path):
    """Check everything the program needs before the window opens.

    Prints one line per problem ("FAIL: ..." or "WARN: ...") and exits if
    anything essential is missing, so nothing goes wrong later in front of
    an audience. Returns (cap, first_frame, focal_length, calibration_size,
    size_warning): the opened camera or video and its first frame, the
    calibration (None, None if there isn't one), and a status line message
    if the frame size doesn't match the calibration (or None).
    """
    failures = 0
    video_mode = args.video is not None

    if not model_path.is_file():
        if args.demo:
            print(f"FAIL: model file not found: {model_path} (demo mode never "
                  "downloads it; copy it into the project folder)")
            failures += 1
        else:
            print(f"WARN: model file not found: {model_path}; Ultralytics "
                  "will try to download it")

    focal_length, calibration_size = load_calibration()
    if focal_length is None:
        print(f"WARN: not calibrated: hold the {TARGET_CLASS} at "
              f"{KNOWN_DISTANCE_M:.2f} m and press c")

    # The camera or video file, and its first frame (needed to check the
    # frame size).
    cap = None
    frame = None
    if video_mode:
        if not Path(args.video).is_file():
            print(f"FAIL: video file not found: {args.video}")
            failures += 1
        else:
            cap = cv2.VideoCapture(args.video)
            if not cap.isOpened():
                print(f"FAIL: could not open the video file: {args.video}")
                failures += 1
    else:
        cap = open_camera()
        if cap is None:
            print("FAIL: could not open the webcam (close Zoom, Teams or "
                  "browser tabs using it, and check Settings > Privacy > "
                  "Camera); the backup is demo_backup.bat")
            failures += 1

    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            frame = None
            source = "video file" if video_mode else "webcam"
            print(f"FAIL: the {source} opened but gives no frames")
            failures += 1

    # The focal length in pixels only fits the resolution it was
    # calibrated at.
    size_warning = None
    if frame is not None and calibration_size is not None:
        frame_size = (frame.shape[1], frame.shape[0])  # (width, height)
        if frame_size != calibration_size:
            size_warning = (f"Warning: {frame_size[0]}x{frame_size[1]}, "
                            f"calibrated at {calibration_size[0]}x"
                            f"{calibration_size[1]}")
            print(f"WARN: frames are {frame_size[0]}x{frame_size[1]} but the "
                  f"calibration was made at {calibration_size[0]}x"
                  f"{calibration_size[1]}; distances will be wrong until you "
                  "recalibrate")

    if failures:
        if cap is not None:
            cap.release()
        print(f"Pre-flight check failed ({failures} problem"
              f"{'s' if failures > 1 else ''}); not starting.")
        sys.exit(1)

    return cap, frame, focal_length, calibration_size, size_warning


class QuitKey:
    """q quits at once, or in demo mode only when pressed twice within
    QUIT_CONFIRM_S, so one stray key press can't end the demo."""

    def __init__(self, confirm):
        self.confirm = confirm  # True: needs a second press
        self.first_press = None  # when q was pressed first (perf_counter)

    def press(self):
        # Returns True to quit now, False to wait for a second press.
        now = time.perf_counter()
        if (not self.confirm or (self.first_press is not None
                                 and now - self.first_press < QUIT_CONFIRM_S)):
            return True
        self.first_press = now
        return False


def toggle_fullscreen(display, monitor=None):
    # Switch the window between full screen and normal. In full screen the
    # HUD is drawn at the screen's own size, so it's shown pixel for pixel.
    # monitor: the screen to fill, or None for the one the window is on.
    if screen.is_fullscreen(WINDOW_NAME):
        screen.set_fullscreen(WINDOW_NAME, False)
        display.set_display_size(hud.DISPLAY_SIZE)
    else:
        size = screen.set_fullscreen(WINDOW_NAME, True, monitor)
        if size is not None:
            display.set_display_size(size)


def handle_view_keys(key, display):
    # The keys that only change the view, the same in every situation:
    # d (debug overlay), h (key legend) and f (full screen).
    if key == ord("d"):
        display.toggle_debug()
    if key == ord("h"):
        display.toggle_legend()
    if key == ord("f"):
        toggle_fullscreen(display)


def window_closed():
    # True if the user closed the window with the X button.
    return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1


def reconnect_camera(display, last_frame, quit_key):
    """Keep trying to reopen a lost webcam, with the window still working.

    Shows the last frame with a "camera lost" message and tries the camera
    again every RECONNECT_INTERVAL_S. Returns (cap, frame) once it's back,
    or (None, None) if the user quits or closes the window meanwhile.
    """
    print("Camera lost; trying to reconnect (q to quit).")
    status = "Camera lost: reconnecting..."
    next_try = time.perf_counter()
    while True:
        if time.perf_counter() >= next_try:
            cap = open_camera()
            if cap is not None:
                ret, frame = cap.read()
                if ret:
                    print("Camera reconnected.")
                    return cap, frame
                cap.release()
            next_try = time.perf_counter() + RECONNECT_INTERVAL_S

        image = display.render(
            last_frame, state=hud.SEARCHING, box=None, track_id=None,
            conf=None, others=[], lead=None, distance_text="--",
            bearing=None, intercept_text="--", fps=None, mode="LIVE",
            status=status, debug_rows=[("camera", "reconnecting")],
            timings=[])
        cv2.imshow(WINDOW_NAME, image)
        key = cv2.waitKey(50) & 0xFF

        if key == ord("q"):
            if quit_key.press():
                return None, None
            status = "Camera lost: press q again to quit"
        handle_view_keys(key, display)
        if window_closed():
            return None, None


def video_time(cap, last_t):
    # Timestamp of the frame just read from a video file, in seconds, from
    # the file's own timing (not the clock), so velocities are right no
    # matter how fast or slow the frames are processed. last_t is the
    # previous frame's timestamp, or None for the first frame.
    t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    if last_t is None or t > last_t:
        return t

    # Some files report a time that doesn't move forward. Fall back to one
    # frame interval after the previous frame (30 FPS if the file doesn't
    # say its frame rate).
    fps = cap.get(cv2.CAP_PROP_FPS)
    return last_t + 1 / (fps if fps > 0 else 30)


def print_video_summary(stats):
    # Print a short summary after a video run, to compare settings.
    frames = stats["frames"]
    tracked = stats["tracked_frames"]
    percent = 100 * tracked / frames if frames else 0
    ids = sorted(stats["track_ids"])

    # Processing speed, leaving out YOLO's slow warm-up frame(s).
    timed_frames = frames - WARMUP_FRAMES
    if stats["timing_start"] is not None and timed_frames > 0:
        seconds = time.perf_counter() - stats["timing_start"]
        fps_text = f"{timed_frames / seconds:.1f} FPS"
    else:
        fps_text = "--"

    rows = [
        ("Frames processed:", f"{frames}"),
        ("Frames with tracked bottle:", f"{tracked} ({percent:.0f}%)"),
        ("Distinct bottle IDs seen:", f"{len(ids)} {ids if ids else ''}"),
        ("Video length:", f"{stats['video_seconds']:.1f} s"),
        ("Processing speed:", fps_text),
        ("Settings:", f"CONF_THRESHOLD = {CONF_THRESHOLD}, IMGSZ = {IMGSZ}"),
    ]
    print()
    print("Video summary")
    for label, value in rows:
        print(f"  {label:<29}{value}")  # :<29 pads labels to line up


def report_startup(launch_s, main_start, camera_done, model_done,
                   warmup_done):
    # Print how long it took from launch to the first frame on screen, and
    # where the time went. launch_s is how long Python and the imports took
    # before main() started (None if unknown); the rest are perf_counter
    # times of each step's end.
    now = time.perf_counter()
    parts = [("checks + camera", camera_done - main_start),
             ("model", model_done - camera_done),
             ("warm-up", warmup_done - model_done),
             ("window", now - warmup_done)]
    total = now - main_start
    if launch_s is not None:
        parts.insert(0, ("Python + imports", launch_s))
        total += launch_s
    details = ", ".join(f"{name} {seconds:.1f} s" for name, seconds in parts)
    print(f"Ready in {total:.1f} s ({details})")


def main(argv=None):
    # How long Python and the imports took, for the startup report.
    launch_s = screen.seconds_since_launch()
    main_start = time.perf_counter()

    # Before any window exists, so full screen uses real pixels.
    screen.make_dpi_aware()

    args = parse_args(argv)
    video_mode = args.video is not None

    # --screen: which monitor the window goes on (None = wherever Windows
    # puts it). --screen without a number just lists the monitors.
    monitor = None
    if args.screen is not None:
        monitors = screen.list_monitors()
        if not 1 <= args.screen <= len(monitors):
            if args.screen != 0:
                print(f"Error: there is no screen {args.screen}.")
            for i, (x, y, w, h, primary) in enumerate(monitors, start=1):
                print(f"Screen {i}: {w}x{h} at ({x}, {y})"
                      f"{' (main)' if primary else ''}")
            sys.exit(0 if args.screen == 0 else 1)
        monitor = monitors[args.screen - 1]

    # Pre-flight checks: model file, calibration, camera or video file.
    # Also opens the camera (or file) and reads its first frame.
    model_path = model_file(args.model)
    cap, first_frame, focal_length, calibration_size, size_warning = (
        preflight(args, model_path))
    camera_done = time.perf_counter()

    # Load the model once, before the loop (loading is slow).
    try:
        model = YOLO(str(model_path))
    except Exception as error:  # any problem with the file
        print(f"FAIL: could not load the model {model_path}: {error}")
        cap.release()
        sys.exit(1)
    model_done = time.perf_counter()

    # model.names maps each class ID to its name, e.g. {39: 'bottle', ...}.
    # Look up the ID for our target class instead of hardcoding the number.
    bottle_id = None
    for class_id, name in model.names.items():
        if name == TARGET_CLASS:
            bottle_id = class_id
            break

    if bottle_id is None:
        print(f"FAIL: the model has no class named '{TARGET_CLASS}'.")
        cap.release()
        sys.exit(1)

    # Warm-up: YOLO's first run takes seconds (PyTorch sets itself up), so
    # do it now, on a black frame the size of the camera's, instead of
    # freezing the first live frame. Same settings as the tracking call.
    model.track(np.zeros_like(first_frame), persist=True,
                tracker=TRACKER_CONFIG, conf=CONF_THRESHOLD, imgsz=IMGSZ,
                verbose=False)
    warmup_done = time.perf_counter()

    height, width = first_frame.shape[:2]
    if video_mode:
        source = f"video {args.video} {width}x{height}"
    else:
        source = f"camera {width}x{height} ({cap.getBackendName()})"
    if focal_length is None:
        calibration_text = "not calibrated"
    else:
        calibration_text = f"f {focal_length:.1f} px"
    print(f"OK: {source}, model {model_path.name}, {calibration_text}")

    # Calibration state.
    calibrating = False  # True while we're collecting samples
    samples = []  # box heights (h) collected so far
    message = ""  # last calibration result or progress, shown on the panel
    message_time = 0.0  # when message was set (time.perf_counter() seconds)

    speed = DEFAULT_SPEED_MPS  # interceptor speed, changed with + and -
    smoothed_h = None  # running EMA of h; None until there's a bottle to track

    # The HUD, and the lock state it shows (SEARCHING / LOCKED / LOST).
    keys = key_legend(video_mode, args.demo)
    display = hud.Hud(TARGET_CLASS, keys=keys)
    lock = hud.LockIndicator()
    quit_key = QuitKey(confirm=args.demo)

    # Tracking and velocity state.
    locked_id = None  # track ID of the bottle we're following
    history_id = None  # the track ID the history belongs to
    last_seen = None  # when the target was last seen (perf_counter seconds)
    # (time, X, Y, Z) of the target in meters, raw (unsmoothed); maxlen drops
    # the oldest sample automatically, so it always holds the last N.
    history_3d = deque(maxlen=VELOCITY_WINDOW)

    def reset_lock():
        # Forget the target we're following, so the most confident one is
        # picked again on the next frame (the r key, and each video loop).
        # nonlocal: these are main()'s variables, not new ones.
        nonlocal locked_id, history_id, last_seen, smoothed_h
        locked_id = None
        history_id = None
        last_seen = None
        smoothed_h = None
        history_3d.clear()
        lock.reset()

    # FPS state.
    frame_time = None  # smoothed seconds per loop pass; None until measured
    last_loop_start = None  # when the previous loop pass started
    frame_count = 0  # loop passes so far

    frame_t = None  # timestamp of the current frame, in seconds
    # For a video: the file's own time of the current frame, and how much
    # to add to it so time keeps counting up when --loop starts it again.
    file_t = None
    loop_offset = 0.0
    pending_frame = first_frame  # read by the pre-flight check, not used yet
    last_frame = first_frame  # shown while the webcam reconnects

    # Counts for the end-of-video summary. timing_start is when the first
    # frame after the warm-up began processing.
    stats = {"frames": 0, "tracked_frames": 0, "track_ids": set(),
             "video_seconds": 0.0, "timing_start": None}

    print("Keys: " + ", ".join(f"{key} {action}" for key, action in keys))

    # A resizable window (needed for full screen), starting at WINDOW_SIZE,
    # on the chosen monitor if there is one. Demo mode starts full screen.
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_NAME, *WINDOW_SIZE)
    if monitor is not None:
        cv2.moveWindow(WINDOW_NAME, monitor[0] + 40, monitor[1] + 40)
    if args.demo:
        toggle_fullscreen(display, monitor)
        screen.keep_awake(True)

    # The frame size warning, if any, goes on the status line now (not
    # before the warm-up, or it would be gone before the window opens).
    if size_warning:
        message = size_warning
        message_time = time.perf_counter()
    ready_reported = False

    # try/finally makes sure the cleanup below runs no matter how the loop ends.
    try:
        while True:
            # perf_counter() is a high-precision clock in seconds. The time
            # between the starts of two loop passes is one frame's time.
            loop_start = time.perf_counter()
            frame_count += 1
            # The gap since the previous pass covers that pass, so skip it
            # while the previous pass was a warm-up pass.
            if frame_count > WARMUP_FRAMES + 1:
                frame_time = ema(frame_time, loop_start - last_loop_start,
                                 FPS_SMOOTHING)
            last_loop_start = loop_start
            if frame_count == WARMUP_FRAMES + 1:
                stats["timing_start"] = loop_start

            # Grab one frame (one still picture) from the camera or file.
            # ret is True if it worked; frame is the image itself. The very
            # first frame was already read by the pre-flight check.
            if pending_frame is not None:
                ret, frame = True, pending_frame
                pending_frame = None
            else:
                ret, frame = cap.read()

            # --loop: at the end of the video, start it again. Time keeps
            # counting up from where this pass ended, so the velocity and
            # lock timing never see it go backwards, and the lock starts
            # over, since the scene jumps.
            if not ret and video_mode and args.loop:
                fps = cap.get(cv2.CAP_PROP_FPS)
                loop_offset = frame_t + 1 / (fps if fps > 0 else 30)
                file_t = None
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                reset_lock()
                ret, frame = cap.read()

            # A failed read is the normal end of a video file. For the
            # webcam it's an error (e.g. camera unplugged): keep trying to
            # reconnect, with the window still up, instead of quitting.
            if not ret:
                if video_mode:
                    print("End of video.")
                    break
                cap.release()
                cap, frame = reconnect_camera(display, last_frame, quit_key)
                if cap is None:
                    break
                # The time spent reconnecting isn't a slow frame.
                last_loop_start = time.perf_counter()

            # When this frame was taken. Used for the velocity. For a video
            # file that's the file's own frame time; for the webcam it's now
            # (close enough: right after reading, before the slow YOLO step).
            if video_mode:
                file_t = video_time(cap, file_t)
                frame_t = loop_offset + file_t
                stats["video_seconds"] = frame_t
            else:
                frame_t = time.perf_counter()
            stats["frames"] += 1
            capture_done = time.perf_counter()  # for the debug stage timings

            # Kept to show while the webcam reconnects, if it's lost.
            last_frame = frame

            # Run YOLO in track mode on this frame: it detects, then matches
            # each box to the boxes of earlier frames and gives it an ID.
            # persist=True keeps the tracker's memory between calls. conf=
            # drops anything below our threshold (the model's own default is
            # 0.25). imgsz= is the size YOLO shrinks the frame to before
            # detecting. verbose=False stops it printing a line for every
            # frame. It returns one result per image; we gave it one image,
            # so take the first.
            result = model.track(frame, persist=True, tracker=TRACKER_CONFIG,
                                 conf=CONF_THRESHOLD, imgsz=IMGSZ,
                                 verbose=False)[0]

            # Every bottle found in this frame, as
            # (x1, y1, x2, y2, conf, track_id). Labels are drawn later, after
            # the panel, so they can avoid it.
            bottles = []

            # result.boxes holds one entry per detected object.
            for box in result.boxes:
                # Class ID and confidence come back as tensors; int() and
                # float() turn them into plain Python numbers.
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # Skip anything that isn't a bottle.
                if cls_id != bottle_id:
                    continue

                # The track ID, or None if this box isn't part of a track yet.
                track_id = int(box.id[0]) if box.id is not None else None

                # xyxy = top-left (x1, y1) and bottom-right (x2, y2) corners,
                # in pixels of our frame. OpenCV needs whole numbers.
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                bottles.append((x1, y1, x2, y2, conf, track_id))

                if track_id is not None:
                    stats["track_ids"].add(track_id)

            # The bottle we measure and aim at (same one for everything).
            target = choose_target(bottles, locked_id)
            if target is not None:
                locked_id = target[5]
                if target[5] is not None:
                    stats["tracked_frames"] += 1

            # frame.shape[0] is the frame's height in pixels.
            frame_height = frame.shape[0]

            # Smooth h only while there's exactly one bottle, fully in view.
            # Otherwise start over: with two bottles the target can switch
            # between them, and a cut-off box has a wrong h, so neither should
            # be mixed into the running value.
            if len(bottles) == 1 and not touches_edge(target[1], target[3],
                                                      frame_height):
                smoothed_h = smooth_h(smoothed_h, target[3] - target[1])
            else:
                smoothed_h = None

            # 3D position history. Start over if the target is a different
            # track, or if it was missing for too long. A short miss is fine:
            # the timestamps just show a bigger gap.
            missing_too_long = (last_seen is not None
                                and frame_t - last_seen > LOST_TIMEOUT_S)
            if target is not None:
                if target[5] != history_id or missing_too_long:
                    history_3d.clear()
                    history_id = target[5]
                last_seen = frame_t
            elif missing_too_long:
                history_3d.clear()

            # Distance for the target.
            distance_text, distance, reason = measurement_values(
                target, frame_height, focal_length, speed, smoothed_h)

            # Bearing angles for the target, from its raw box center.
            bearing = bearing_values(target, frame, focal_length)

            # 3D position: raw box center (like the bearings) and the
            # smoothed distance (like DIST), so all three readouts agree.
            pos_text, position = position_values(target, frame, focal_length,
                                                 distance)

            # 3D velocity sample: this frame's RAW position. The fit does the
            # smoothing itself, so feeding it the smoothed distance would
            # smooth twice and add lag. Only tracked bottles, and only frames
            # with a distance (skips "--" frames, e.g. bottle at the edge,
            # without resetting).
            if distance is not None and target[5] is not None:
                raw_h = target[3] - target[1]
                if raw_h > 0:
                    raw_distance = compute_distance(raw_h, focal_length,
                                                    BOTTLE_HEIGHT_M)
                    dx, dy = center_offsets(target, frame)
                    X, Y, Z = compute_3d_position(dx, dy, raw_distance,
                                                  focal_length)
                    history_3d.append((frame_t, X, Y, Z))

            # Fit once the window is full, so the noise level is always the
            # same. Times are shifted to start at 0 to keep the numbers small
            # (better floating-point precision).
            velocity_3d = None  # (vx, vy, vz) in m/s, or None
            if len(history_3d) == VELOCITY_WINDOW:
                t_first = history_3d[0][0]
                times = [s[0] - t_first for s in history_3d]
                xs = [s[1] for s in history_3d]
                ys = [s[2] for s in history_3d]
                zs = [s[3] for s in history_3d]
                velocity_3d = (fit_velocity(times, xs),
                               fit_velocity(times, ys),
                               fit_velocity(times, zs))

            # True intercept time and 3D aim point, from the position shown
            # in POS and the fitted velocity.
            time_text, aim_text, aim, intercept_reason = intercept_values(
                position, velocity_3d, speed)

            # Lead point: the 3D aim point projected onto the screen. Points
            # at or behind the camera (or too close to it) can't be drawn.
            lead = None
            aim_reason = None
            if aim is not None:
                if aim[2] <= MIN_AIM_Z_M:
                    aim_reason = "Aim point at/behind camera"
                else:
                    # The same image center the 3D position is measured from.
                    frame_h, frame_w = frame.shape[:2]
                    lead = project_to_pixels(aim[0], aim[1], aim[2],
                                             focal_length,
                                             frame_w / 2, frame_h / 2)
            compute_done = time.perf_counter()  # for the debug stage timings

            # While calibrating, collect one h per frame, but only from frames
            # we can trust.
            if calibrating:
                if len(bottles) == 0:
                    status = f"no {TARGET_CLASS}"
                elif len(bottles) > 1:
                    # Can't tell which box is the bottle at the known distance.
                    status = f"{len(bottles)} {TARGET_CLASS}s"
                else:
                    x1, y1, x2, y2 = bottles[0][:4]
                    if touches_edge(y1, y2, frame_height):
                        status = "at edge"
                    else:
                        samples.append(y2 - y1)
                        status = f"h {y2 - y1}"

                # Enough samples: average them and compute the focal length.
                if len(samples) >= CALIBRATION_FRAMES:
                    calibrating = False
                    avg_h = sum(samples) / len(samples)
                    focal_length = compute_focal_length(
                        avg_h, KNOWN_DISTANCE_M, BOTTLE_HEIGHT_M)
                    calibration_size = (frame.shape[1], frame.shape[0])
                    save_calibration(focal_length, avg_h, calibration_size)
                    message = f"Saved: f {focal_length:.1f} px"
                    print(f"Calibrated: avg h = {avg_h:.1f} px, "
                          f"focal length = {focal_length:.1f} px")
                else:
                    message = (f"Calibrating {len(samples)}/"
                               f"{CALIBRATION_FRAMES}: {status}")
                message_time = time.perf_counter()

            # Whether the lead point lands inside the frame (the HUD cuts an
            # off-screen lead line at the edge).
            frame_h, frame_w = frame.shape[:2]
            lead_on_screen = (lead is None
                              or (0 <= round(lead[0]) < frame_w
                                  and 0 <= round(lead[1]) < frame_h))

            # The status line: a calibration message wins while calibrating
            # or for a few seconds after it was set; then the reason for any
            # "--" distance values; then the reason there's no intercept;
            # then an aim point that can't be drawn; then an off-screen lead
            # point (or nothing).
            message_is_fresh = (time.perf_counter() - message_time
                                < STATUS_MESSAGE_SECONDS)
            if message and (calibrating or message_is_fresh):
                status_line = message
            elif reason:
                status_line = reason
            elif intercept_reason:
                status_line = intercept_reason
            elif aim_reason:
                status_line = aim_reason
            elif not lead_on_screen:
                status_line = "Lead point off screen"
            else:
                status_line = None

            # 3D velocity: right, up, and away from the camera are positive
            # (so a negative Z velocity means the bottle is approaching).
            if velocity_3d is None:
                vel_text = "--"
            else:
                vx, vy, vz = velocity_3d
                vel_text = (f"{signed(vx, 2)} / {signed(vy, 2)} / "
                            f"{signed(vz, 2)} m/s")

            # Debug overlay values (hidden unless d is pressed).
            if bearing is None:
                bearing_text = "--"
            else:
                bearing_text = (f"{signed(bearing[0], 1)} / "
                                f"{signed(bearing[1], 1)} deg")
            if focal_length is None:
                f_text = "--"
            elif calibration_size is None:
                f_text = f"{focal_length:.1f} px"
            else:
                f_text = (f"{focal_length:.1f} px at {calibration_size[0]}x"
                          f"{calibration_size[1]}")
            debug_rows = [
                ("h raw", f"{target[3] - target[1]} px"
                          if target is not None else "--"),
                ("h smooth", f"{smoothed_h:.1f} px"
                             if smoothed_h is not None else "--"),
                ("POS", pos_text),
                ("VEL", vel_text),
                ("AIM", aim_text),
                ("BRG", bearing_text),
                ("SPEED", f"{speed:.1f} m/s"),
                ("f", f_text),
            ]

            if calibrating:
                mode = f"CALIBRATING {len(samples)}/{CALIBRATION_FRAMES}"
            else:
                mode = "VIDEO" if video_mode else "LIVE"

            # Lock state: the HUD only gets plain values, never the bottle
            # tuples, so a different tracker only changes these calls.
            if target is None:
                state, lock_box, lock_id, lock_conf = lock.update(
                    None, None, None, frame_t)
            else:
                state, lock_box, lock_id, lock_conf = lock.update(
                    target[:4], target[5], target[4], frame_t)
            others = [(b[:4], b[5], b[4]) for b in bottles if b is not target]

            # The measurements belong to target. While the lock indicator
            # holds on to a target that just disappeared, they'd describe a
            # different box (or nothing), so leave them out of the HUD.
            if target is None or target[5] != lock_id:
                distance_text, bearing, time_text, lead = "--", None, "--", None

            # Draw the HUD on a scaled-up copy; the frame itself stays clean.
            image = display.render(
                frame, state=state, box=lock_box, track_id=lock_id,
                conf=lock_conf, others=others, lead=lead, distance_text=distance_text,
                bearing=bearing, intercept_text=time_text,
                fps=1 / frame_time if frame_time else None, mode=mode,
                status=status_line, debug_rows=debug_rows,
                timings=[("capture", capture_done - loop_start),
                         ("detect + math", compute_done - capture_done)])

            # Hand the image to the window. Nothing is drawn until waitKey runs.
            cv2.imshow(WINDOW_NAME, image)

            # Let the window process its events (actually draw the frame) and
            # wait up to 1 ms for a key press. & 0xFF keeps only the lowest
            # 8 bits of the key code so the comparison works on every platform.
            key = cv2.waitKey(1) & 0xFF

            # Startup time, once the first frame is on screen.
            if not ready_reported:
                ready_reported = True
                report_startup(launch_s, main_start, camera_done, model_done,
                               warmup_done)

            # ord('q') is the key code for the letter q. In demo mode it has
            # to be pressed twice.
            if key == ord("q"):
                if quit_key.press():
                    break
                message = "Press q again to quit"
                message_time = time.perf_counter()

            # c starts calibration, or cancels it if it's already running.
            # Webcam only: a video may have a different resolution, and its
            # result would overwrite the webcam's calibration.
            if key == ord("c"):
                if video_mode:
                    message = "Calibration only works with the webcam"
                elif calibrating:
                    calibrating = False
                    message = "Calibration cancelled"
                else:
                    calibrating = True
                    samples = []
                    message = f"Calibrating 0/{CALIBRATION_FRAMES}"
                message_time = time.perf_counter()

            # + speeds up and - slows down. Each key counts with or without
            # Shift: the +/= key gives "+" or "=", and the -/_ key gives "-"
            # or "_" (Shift is easy to still be holding after typing "+").
            # round() stops tiny decimal errors building up; max() keeps the
            # speed from going below 0.
            # The speed isn't on the panel, so show it on the status line.
            if key in (ord("+"), ord("=")):
                speed = round(speed + SPEED_STEP_MPS, 2)
                message = f"Interceptor {speed:.1f} m/s"
                message_time = time.perf_counter()
            if key in (ord("-"), ord("_")):
                speed = max(0.0, round(speed - SPEED_STEP_MPS, 2))
                message = f"Interceptor {speed:.1f} m/s"
                message_time = time.perf_counter()

            # r drops the lock, so the tracker's most confident target is
            # picked again (for when it's following the wrong object).
            if key == ord("r"):
                reset_lock()
                message = "Lock reset: reacquiring"
                message_time = time.perf_counter()

            # d (debug overlay), h (key legend) and f (full screen).
            handle_view_keys(key, display)

            # Stop if the user closed the window with the X button.
            if window_closed():
                break
    finally:
        # Turn the camera off (or close the file) so other programs can use
        # it. cap is None if the webcam was lost and never came back.
        if cap is not None:
            cap.release()
        # Close every window OpenCV opened, and let the laptop sleep again.
        cv2.destroyAllWindows()
        screen.keep_awake(False)

    if video_mode:
        print_video_summary(stats)


# Only run main() when this file is run directly (not when it's imported).
if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # In demo mode a crash shouldn't fill the screen with a traceback:
        # save the details to a file and print one line instead.
        if "--demo" not in sys.argv[1:]:
            raise
        with open(ERROR_LOG_PATH, "w") as f:
            traceback.print_exc(file=f)
        print(f"Error: {type(error).__name__}: {error} "
              f"(details in {ERROR_LOG_PATH})")
        sys.exit(1)
