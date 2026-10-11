"""Bottle Interceptor HUD.

Detects and tracks a water bottle with YOLO, measures its 3D position and
velocity from a single calibrated webcam, solves for the intercept time of
an interceptor launched from the camera, and draws a targeting HUD with a
lead line to the predicted aim point.

Usage:
    .venv\\Scripts\\python.exe main.py                     (webcam)
    .venv\\Scripts\\python.exe main.py --video clip.mp4    (recorded file)

Press c to calibrate (press c again to cancel; webcam only).
Press + / - to change the interceptor speed.
Press q (or close the window) to quit.
"""

import argparse  # reading command-line options like --video
import json  # reading and writing calibration.json
import math  # math functions for the bearing formula and the intercept
import sys  # used to exit the program with an error code
import time  # high-precision clock for FPS and webcam timestamps
from collections import deque  # a list that's quick to trim from the front
from datetime import datetime  # timestamp saved with the calibration
from pathlib import Path  # building file paths that work on any OS

import cv2  # OpenCV: camera access, windows, and image drawing
from ultralytics import YOLO  # Ultralytics: loads and runs YOLO models

CAMERA_INDEX = 0  # 0 = the first camera Windows finds (the built-in webcam)
WINDOW_NAME = "Bottle HUD"  # the title shown on the window

MODEL_PATH = "yolo26n.pt"  # YOLO26 nano; downloaded automatically on first run
TARGET_CLASS = "bottle"  # the only class we want to show
CONF_THRESHOLD = 0.1  # hide detections the model is less sure about than this
IMGSZ = 640  # image size YOLO works at; smaller is faster but sees less detail
# Tracker that gives each bottle an ID that stays the same across frames.
# ByteTrack is lighter than the default BoT-SORT, which spends extra time
# compensating for a moving camera (ours doesn't move).
TRACKER_CONFIG = "bytetrack.yaml"
BOX_COLOR = (0, 255, 0)  # green; OpenCV colors are (blue, green, red)
TEXT_COLOR = (0, 0, 0)  # black text on the green label background

FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE = 0.5  # box labels like "#7 0.85"
LABEL_FONT_THICKNESS = 1

# Crosshair that marks the origin point (bottom center of the frame).
CROSSHAIR_SIZE = 20  # length of each arm from the center, in pixels
CROSSHAIR_COLOR = (0, 0, 255)  # red, so it stands out from the green box
CROSSHAIR_THICKNESS = 2  # line thickness in pixels
CROSSHAIR_BOTTOM_MARGIN = 40  # distance from the bottom edge (keep > SIZE)

# Target line from the crosshair to the bottle's center.
LINE_COLOR = (0, 255, 255)  # yellow
LINE_THICKNESS = 2  # line thickness in pixels
TARGET_DOT_RADIUS = 4  # filled dot at the bottle's center, in pixels

# Lead line from the crosshair to the 3D aim point, projected onto the screen.
LEAD_COLOR = (255, 0, 255)  # magenta
LEAD_THICKNESS = 2  # line thickness in pixels
# Aim points closer than this (Z, in meters) aren't drawn. Projecting divides
# by Z, so points at or behind the camera can't be drawn, and the whole
# bottle only fits in the frame from about 0.18 m away anyway.
MIN_AIM_Z_M = 0.20

# 3D velocity (m/s) from a least-squares fit over the last N positions.
VELOCITY_WINDOW = 10  # samples per fit; more = steadier but slower to react
LOST_TIMEOUT_S = 0.5  # missing longer than this (s) resets the history

# HUD panel.
PANEL_X = 10  # left edge of the panel, in pixels
PANEL_Y = 10  # top edge of the panel, in pixels
PANEL_WIDTH = 260  # minimum width; grows only if a line needs more room
PANEL_PADDING = 8  # space between the panel's edge and the text
PANEL_VALUE_OFFSET = 70  # where the value column starts, from the text's left
PANEL_ALPHA = 0.55  # background opacity: 0 = invisible, 1 = solid
PANEL_BG_COLOR = (0, 0, 0)  # black background
PANEL_LABEL_COLOR = (180, 180, 180)  # gray labels
PANEL_TEXT_COLOR = (255, 255, 255)  # white values
PANEL_STATUS_COLOR = (0, 255, 255)  # yellow status line
PANEL_FONT_SCALE = 0.5
PANEL_FONT_THICKNESS = 1
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
        return "--", None, (f"Not calibrated: hold bottle at "
                            f"{KNOWN_DISTANCE_M:.2f} m, press c")
    if target is None:
        return "--", None, None

    x1, y1, x2, y2 = target[:4]
    if touches_edge(y1, y2, frame_height):
        return "--", None, "Bottle at edge: distance unreliable"

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
    # Work out the horizontal and vertical bearing values for the panel.
    # Returns (horizontal_text, vertical_text). The reason for a "--" (not
    # calibrated) already comes from measurement_values.
    if target is None or focal_length is None:
        return "--", "--"

    dx, dy = center_offsets(target, frame)

    bearing_h = compute_bearing(dx, focal_length)
    bearing_v = compute_bearing(dy, focal_length)

    # :+.1f always shows the sign (+ or -) and one decimal place. "deg"
    # because OpenCV's fonts can't draw the degree symbol.
    return f"{bearing_h:+.1f} deg", f"{bearing_v:+.1f} deg"


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


def draw_target_line(frame, origin, target):
    # Line from the crosshair to the bottle's center, plus a dot at the end.
    # OpenCV needs whole-number pixel positions, so round the center.
    bx, by = box_center(target)
    center = (round(bx), round(by))
    cv2.line(frame, origin, center, LINE_COLOR, LINE_THICKNESS)
    # Thickness -1 means a filled circle.
    cv2.circle(frame, center, TARGET_DOT_RADIUS, LINE_COLOR, -1)


def draw_lead_line(frame, origin, lead):
    # Line from the crosshair to the aim point on screen (lead is its
    # (x, y) in pixels). Returns True if the point is on screen, False if
    # it's off screen.
    height, width = frame.shape[:2]
    end = (round(lead[0]), round(lead[1]))

    if 0 <= end[0] < width and 0 <= end[1] < height:
        cv2.line(frame, origin, end, LEAD_COLOR, LEAD_THICKNESS)
        cv2.circle(frame, end, TARGET_DOT_RADIUS, LEAD_COLOR, -1)
        return True

    # Off screen: cut the line where it leaves the frame, keeping its
    # direction. clipLine returns whether any part is inside the rectangle
    # (x, y, width, height), and the two ends of the part that is.
    inside, start, edge = cv2.clipLine((0, 0, width, height), origin, end)
    if inside:
        cv2.line(frame, start, edge, LEAD_COLOR, LEAD_THICKNESS)
        # A hollow circle at the edge means "the point is beyond here".
        cv2.circle(frame, edge, TARGET_DOT_RADIUS + 2, LEAD_COLOR, 2)
    return False


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
        print(f"Warning: could not read {CALIBRATION_PATH}; press c to recalibrate.")
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


def box_label(conf, track_id):
    # Short label for a box, like "#7 0.85" (track ID and confidence), or
    # just "0.85" for a box that isn't part of a track yet.
    if track_id is None:
        return f"{conf:.2f}"
    return f"#{track_id} {conf:.2f}"


def rects_overlap(a, b):
    # True if two rectangles (x1, y1, x2, y2) overlap at all.
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def draw_label(frame, box, label, avoid):
    # Draw a box's label on a green background. box is (x1, y1, x2, y2);
    # avoid is a rectangle (x1, y1, x2, y2) the label must not cover (the
    # HUD panel).

    # Measure the text, so the background fits it. text_w/text_h are its
    # width and height in pixels; baseline is the extra space below the line
    # that letters like "g" hang into.
    (text_w, text_h), baseline = cv2.getTextSize(label, FONT, LABEL_FONT_SCALE,
                                                 LABEL_FONT_THICKNESS)
    label_w = text_w + 8  # 4 px of padding on each side
    label_h = text_h + baseline + 6

    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = box

    # Line up with the box's left edge, but shift left if needed so the
    # label never runs off the right edge of the frame.
    left = max(0, min(x1, frame_w - label_w))

    # Try above the box, then below it, then inside its top, then (for a box
    # hidden behind the panel) just below the panel. Use the first spot that
    # fits in the frame and doesn't cover the panel; if none does, fall back
    # to inside the top of the box.
    top = max(0, min(y1, frame_h - label_h))
    for candidate in (y1 - label_h, y2, y1, avoid[3] + 2):
        rect = (left, candidate, left + label_w, candidate + label_h)
        if (candidate >= 0 and candidate + label_h <= frame_h
                and not rects_overlap(rect, avoid)):
            top = candidate
            break

    # Filled green rectangle (thickness -1), then the text on top of it.
    # putText's position is the bottom-left of the text.
    cv2.rectangle(frame, (left, top), (left + label_w, top + label_h),
                  BOX_COLOR, -1)
    cv2.putText(frame, label, (left + 4, top + text_h + 3), FONT,
                LABEL_FONT_SCALE, TEXT_COLOR, LABEL_FONT_THICKNESS)


def panel_text_width(text):
    # Width of a piece of text in the panel's font, in pixels.
    (text_w, _), _ = cv2.getTextSize(text, FONT, PANEL_FONT_SCALE,
                                     PANEL_FONT_THICKNESS)
    return text_w


def draw_panel(frame, rows, status):
    # Draw the HUD panel: a semi-transparent box with one (label, value) row
    # per line, plus an optional status line at the bottom. Returns the
    # panel's rectangle (x1, y1, x2, y2), so box labels can avoid it.

    # Measure the line height from a fixed sample ("(" and "g" are the tallest
    # and lowest characters), so every line is the same height.
    (_, text_h), baseline = cv2.getTextSize("(Ag)", FONT, PANEL_FONT_SCALE,
                                            PANEL_FONT_THICKNESS)
    line_h = text_h + baseline + 6  # 6 px between lines

    # Width: at least PANEL_WIDTH, wider only if a line needs it.
    needed = max(PANEL_VALUE_OFFSET + panel_text_width(value)
                 for label, value in rows)
    if status:
        needed = max(needed, panel_text_width(status))
    width = max(PANEL_WIDTH, needed + 2 * PANEL_PADDING)

    line_count = len(rows) + (1 if status else 0)
    height = line_count * line_h + 2 * PANEL_PADDING

    # Panel corners, kept inside the frame.
    frame_h, frame_w = frame.shape[:2]
    x1, y1 = PANEL_X, PANEL_Y
    x2 = min(x1 + width, frame_w)
    y2 = min(y1 + height, frame_h)

    # Semi-transparent background, blended only inside the panel's area.
    # frame[y1:y2, x1:x2] is that rectangle (rows first, then columns). It's
    # a view into the frame, so writing into it changes the frame itself.
    region = frame[y1:y2, x1:x2]
    overlay = region.copy()
    overlay[:] = PANEL_BG_COLOR  # fill the copy with the background color
    # region = overlay * alpha + region * (1 - alpha)
    cv2.addWeighted(overlay, PANEL_ALPHA, region, 1 - PANEL_ALPHA, 0,
                    dst=region)

    # Text, fully opaque on top of the background. putText's position is the
    # bottom-left of the text, hence "+ text_h".
    text_x = x1 + PANEL_PADDING
    for i, (label, value) in enumerate(rows):
        text_y = y1 + PANEL_PADDING + i * line_h + text_h
        cv2.putText(frame, label, (text_x, text_y), FONT, PANEL_FONT_SCALE,
                    PANEL_LABEL_COLOR, PANEL_FONT_THICKNESS)
        cv2.putText(frame, value, (text_x + PANEL_VALUE_OFFSET, text_y), FONT,
                    PANEL_FONT_SCALE, PANEL_TEXT_COLOR, PANEL_FONT_THICKNESS)

    if status:
        text_y = y1 + PANEL_PADDING + len(rows) * line_h + text_h
        cv2.putText(frame, status, (text_x, text_y), FONT, PANEL_FONT_SCALE,
                    PANEL_STATUS_COLOR, PANEL_FONT_THICKNESS)

    return (x1, y1, x2, y2)


def get_origin(frame):
    # frame.shape is (height, width, colors): rows first, then columns.
    height, width = frame.shape[:2]

    # Horizontal center. // is whole-number division (pixels are whole numbers).
    x = width // 2

    # y = 0 is the TOP of the image and grows downward, so subtracting from
    # the height moves the point up from the bottom edge.
    y = height - CROSSHAIR_BOTTOM_MARGIN

    return (x, y)


def draw_crosshair(frame, origin):
    x, y = origin

    # Horizontal arm: from left of the center to right of it.
    cv2.line(frame, (x - CROSSHAIR_SIZE, y), (x + CROSSHAIR_SIZE, y),
             CROSSHAIR_COLOR, CROSSHAIR_THICKNESS)

    # Vertical arm: from above the center to below it.
    cv2.line(frame, (x, y - CROSSHAIR_SIZE), (x, y + CROSSHAIR_SIZE),
             CROSSHAIR_COLOR, CROSSHAIR_THICKNESS)


def parse_args(argv):
    # Read the command-line options. argv is the list of options (None means
    # "use the ones the program was started with").
    parser = argparse.ArgumentParser(
        description="Bottle detection and interceptor targeting HUD.")
    parser.add_argument(
        "--video", metavar="PATH",
        help="run on a recorded video file instead of the webcam")
    return parser.parse_args(argv)


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


def main(argv=None):
    args = parse_args(argv)
    video_mode = args.video is not None

    # Check the video file before the slow model load, so a typo fails fast.
    if video_mode and not Path(args.video).is_file():
        print(f"Error: video file not found: {args.video}")
        sys.exit(1)

    # Load the model once, before the loop (loading is slow).
    model = YOLO(MODEL_PATH)

    # model.names maps each class ID to its name, e.g. {39: 'bottle', ...}.
    # Look up the ID for our target class instead of hardcoding the number.
    bottle_id = None
    for class_id, name in model.names.items():
        if name == TARGET_CLASS:
            bottle_id = class_id
            break

    if bottle_id is None:
        print(f"Error: the model has no class named '{TARGET_CLASS}'.")
        sys.exit(1)

    # Load a saved focal length (None if we haven't calibrated yet) and the
    # frame size it was made with (None if the file doesn't say).
    focal_length, calibration_size = load_calibration()
    if focal_length is None:
        print("Not calibrated yet. Hold the bottle at "
              f"{KNOWN_DISTANCE_M:.2f} m and press c.")
    else:
        print(f"Loaded focal length: {focal_length:.1f} px")

    # Calibration state.
    calibrating = False  # True while we're collecting samples
    samples = []  # box heights (h) collected so far
    message = ""  # last calibration result or progress, shown on the panel
    message_time = 0.0  # when message was set (time.perf_counter() seconds)

    speed = DEFAULT_SPEED_MPS  # interceptor speed, changed with + and -
    smoothed_h = None  # running EMA of h; None until there's a bottle to track

    # Tracking and velocity state.
    locked_id = None  # track ID of the bottle we're following
    history_id = None  # the track ID the history belongs to
    last_seen = None  # when the target was last seen (perf_counter seconds)
    # (time, X, Y, Z) of the target in meters, raw (unsmoothed); maxlen drops
    # the oldest sample automatically, so it always holds the last N.
    history_3d = deque(maxlen=VELOCITY_WINDOW)

    # FPS state.
    frame_time = None  # smoothed seconds per loop pass; None until measured
    last_loop_start = None  # when the previous loop pass started
    frame_count = 0  # loop passes so far

    frame_t = None  # timestamp of the current frame, in seconds
    size_checked = False  # whether the frame size was compared yet

    # Counts for the end-of-video summary. timing_start is when the first
    # frame after the warm-up began processing.
    stats = {"frames": 0, "tracked_frames": 0, "track_ids": set(),
             "video_seconds": 0.0, "timing_start": None}

    # Open the video file, or connect to the webcam. Either way this returns
    # a "capture" object we read frames from.
    if video_mode:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)

    # VideoCapture does not raise an error if it fails, so we must check.
    if not cap.isOpened():
        if video_mode:
            print(f"Error: could not open the video file: {args.video}")
        else:
            print("Error: could not open the webcam.")
            print("Check that no other app (Zoom, Teams, browser) is using it,")
            print("and that Windows camera privacy settings allow desktop apps.")
        sys.exit(1)  # exit code 1 means "something went wrong"

    if video_mode:
        print(f"Playing {args.video}. Press +/- to change speed, q to quit.")
    else:
        print("Webcam opened. Press c to calibrate, +/- to change speed, "
              "q to quit.")

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
            # ret is True if it worked; frame is the image itself.
            ret, frame = cap.read()

            # A failed read is the normal end of a video file, but an error
            # for the webcam (e.g. camera unplugged). Stop either way.
            if not ret:
                if video_mode:
                    print("End of video.")
                else:
                    print("Error: could not read a frame from the webcam.")
                break

            # When this frame was taken. Used for the velocity. For a video
            # file that's the file's own frame time; for the webcam it's now
            # (close enough: right after reading, before the slow YOLO step).
            if video_mode:
                frame_t = video_time(cap, frame_t)
                stats["video_seconds"] = frame_t
            else:
                frame_t = time.perf_counter()
            stats["frames"] += 1

            # The focal length in pixels only fits the resolution it was
            # calibrated at, so warn once if this frame size is different.
            if not size_checked:
                size_checked = True
                frame_size = (frame.shape[1], frame.shape[0])  # (width, height)
                if (focal_length is not None and calibration_size is not None
                        and frame_size != calibration_size):
                    message = (f"Warning: {frame_size[0]}x{frame_size[1]}, "
                               f"calibrated at {calibration_size[0]}x"
                               f"{calibration_size[1]}")
                    message_time = time.perf_counter()
                    print(f"Warning: frames are {frame_size[0]}x"
                          f"{frame_size[1]} but the calibration was made at "
                          f"{calibration_size[0]}x{calibration_size[1]}. "
                          "Distances and velocities will be wrong; "
                          "recalibrate at this resolution.")

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

                # The box: top-left corner, bottom-right corner, color,
                # line thickness.
                cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

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
            bearing_h_text, bearing_v_text = bearing_values(
                target, frame, focal_length)

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

            # While calibrating, collect one h per frame, but only from frames
            # we can trust.
            if calibrating:
                if len(bottles) == 0:
                    status = "no bottle"
                elif len(bottles) > 1:
                    # Can't tell which box is the bottle at the known distance.
                    status = f"{len(bottles)} bottles"
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

            origin = get_origin(frame)

            # The target line only needs a bottle, not calibration.
            if target is not None:
                draw_target_line(frame, origin, target)

            # The lead line goes on top of the target line, so when the
            # bottle is still and they overlap, the lead line shows.
            # Off-screen points keep the clipLine / hollow circle handling.
            lead_on_screen = True
            if lead is not None:
                lead_on_screen = draw_lead_line(frame, origin, lead)

            # Draw the origin crosshair after detection (so YOLO only ever sees
            # the clean frame) and after the lines (so it sits on top of them).
            draw_crosshair(frame, origin)

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

            # Panel values.
            conf_text = f"{target[4]:.2f}" if target is not None else "--"
            fps_text = f"{1 / frame_time:.0f}" if frame_time else "--"
            # 3D velocity: right, up, and away from the camera are positive
            # (so a negative Z velocity means the bottle is approaching).
            if velocity_3d is None:
                vel_text = "--"
            else:
                vx, vy, vz = velocity_3d
                vel_text = (f"{signed(vx, 2)} / {signed(vy, 2)} / "
                            f"{signed(vz, 2)} m/s")

            # The panel goes on top of everything except the box labels.
            panel_rect = draw_panel(frame, [
                ("DIST", distance_text),
                ("POS", pos_text),
                ("TIME", time_text),
                ("AIM", aim_text),
                ("BRG H", bearing_h_text),
                ("BRG V", bearing_v_text),
                ("VEL", vel_text),
                ("CONF", conf_text),
                ("SPEED", f"{speed:.1f} m/s  (+/-)"),
                ("FPS", fps_text),
            ], status_line)

            # Box labels last, placed so they don't cover the panel or run
            # off the frame.
            for bottle in bottles:
                draw_label(frame, bottle[:4], box_label(bottle[4], bottle[5]),
                           panel_rect)

            # Hand the frame to the window. Nothing is drawn until waitKey runs.
            cv2.imshow(WINDOW_NAME, frame)

            # Let the window process its events (actually draw the frame) and
            # wait up to 1 ms for a key press. & 0xFF keeps only the lowest
            # 8 bits of the key code so the comparison works on every platform.
            key = cv2.waitKey(1) & 0xFF

            # ord('q') is the key code for the letter q.
            if key == ord("q"):
                break

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
            if key in (ord("+"), ord("=")):
                speed = round(speed + SPEED_STEP_MPS, 2)
            if key in (ord("-"), ord("_")):
                speed = max(0.0, round(speed - SPEED_STEP_MPS, 2))

            # Stop if the user closed the window with the X button.
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        # Turn the camera off (or close the file) so other programs can use it.
        cap.release()
        # Close every window OpenCV opened.
        cv2.destroyAllWindows()

    if video_mode:
        print_video_summary(stats)


# Only run main() when this file is run directly (not when it's imported).
if __name__ == "__main__":
    main()
