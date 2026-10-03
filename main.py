"""Milestone 7: detect bottles with YOLO, mark a bottom-center origin,
calibrate the camera's focal length, and show distance, time to hit, bearing
angles, confidence, speed and FPS in a semi-transparent HUD panel.

Press c to calibrate (press c again to cancel).
Press + / - to change the interceptor speed.
Press q (or close the window) to quit.
"""

import json  # reading and writing calibration.json
import math  # math functions for the bearing formula
import sys  # used to exit the program with an error code
import time  # high-precision clock for FPS
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
BOX_COLOR = (0, 255, 0)  # green; OpenCV colors are (blue, green, red)
TEXT_COLOR = (0, 0, 0)  # black text on the green label background

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2

# Crosshair that marks the origin point (bottom center of the frame).
CROSSHAIR_SIZE = 20  # length of each arm from the center, in pixels
CROSSHAIR_COLOR = (0, 0, 255)  # red, so it stands out from the green box
CROSSHAIR_THICKNESS = 2  # line thickness in pixels
CROSSHAIR_BOTTOM_MARGIN = 40  # distance from the bottom edge (keep > SIZE)

# Target line from the crosshair to the bottle's center.
LINE_COLOR = (0, 255, 255)  # yellow
LINE_THICKNESS = 2  # line thickness in pixels
TARGET_DOT_RADIUS = 4  # filled dot at the bottle's center, in pixels

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


def compute_time_to_hit(distance_m, speed_mps):
    """Return the time for the interceptor to reach the bottle, in seconds.

    distance_m: distance to the bottle, in meters
    speed_mps: interceptor speed, in meters per second (must be > 0)
    """
    t = distance_m / speed_mps
    return t


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


def measurement_values(best, frame_height, focal_length, speed, smoothed_h):
    # Work out the distance and time-to-hit values for the panel. best is the
    # most confident bottle as (x1, y1, x2, y2, conf), or None if there's no
    # bottle. smoothed_h is the smoothed box height, or None if there isn't
    # one (then the raw h of best is used).
    # Returns (distance_text, time_text, reason). reason explains a "--"
    # for the status line, or is None when there's nothing to explain.
    if focal_length is None:
        return "--", "--", (f"Not calibrated: hold bottle at "
                            f"{KNOWN_DISTANCE_M:.2f} m, press c")
    if best is None:
        return "--", "--", None

    x1, y1, x2, y2, conf = best
    if touches_edge(y1, y2, frame_height):
        return "--", "--", "Bottle at edge: distance unreliable"

    # Prefer the smoothed h; fall back to this frame's raw h.
    h = smoothed_h if smoothed_h is not None else y2 - y1

    # compute_distance divides by h, so a 0-pixel box would crash it.
    if h <= 0:
        return "--", "--", None

    distance = compute_distance(h, focal_length, BOTTLE_HEIGHT_M)
    distance_text = f"{distance:.2f} m"

    # Dividing by a speed of 0 would crash, and the interceptor would never
    # arrive anyway, so don't call compute_time_to_hit at all.
    if speed <= 0:
        return distance_text, "--", "Speed is 0: press + to speed up"

    time_to_hit = compute_time_to_hit(distance, speed)
    return distance_text, f"{time_to_hit:.2f} s", None


def box_center(best):
    # The middle of the box: halfway between the left and right edges, and
    # halfway between the top and bottom edges.
    x1, y1, x2, y2, conf = best
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def bearing_values(best, frame, focal_length):
    # Work out the horizontal and vertical bearing values for the panel.
    # Returns (horizontal_text, vertical_text). The reason for a "--" (not
    # calibrated) already comes from measurement_values.
    if best is None or focal_length is None:
        return "--", "--"

    # The image center is where the camera points (its optical axis).
    height, width = frame.shape[:2]
    cx = width / 2
    cy = height / 2

    bx, by = box_center(best)

    # Image x grows to the right, so right is already positive.
    dx = bx - cx
    # Image y grows DOWNWARD, so subtract the other way round to make up
    # positive.
    dy = cy - by

    bearing_h = compute_bearing(dx, focal_length)
    bearing_v = compute_bearing(dy, focal_length)

    # :+.1f always shows the sign (+ or -) and one decimal place. "deg"
    # because OpenCV's fonts can't draw the degree symbol.
    return f"{bearing_h:+.1f} deg", f"{bearing_v:+.1f} deg"


def draw_target_line(frame, origin, best):
    # Line from the crosshair to the bottle's center, plus a dot at the end.
    # OpenCV needs whole-number pixel positions, so round the center.
    bx, by = box_center(best)
    center = (round(bx), round(by))
    cv2.line(frame, origin, center, LINE_COLOR, LINE_THICKNESS)
    # Thickness -1 means a filled circle.
    cv2.circle(frame, center, TARGET_DOT_RADIUS, LINE_COLOR, -1)


def load_calibration():
    # No file yet means the camera hasn't been calibrated.
    if not CALIBRATION_PATH.exists():
        return None

    # If the file is broken (bad JSON, missing field, not a number), treat it
    # as "not calibrated" instead of crashing.
    try:
        with open(CALIBRATION_PATH) as f:
            data = json.load(f)
        return float(data["focal_length_px"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        print(f"Warning: could not read {CALIBRATION_PATH}; press c to recalibrate.")
        return None


def save_calibration(focal_length, avg_h):
    # Save the result together with the values used to get it, so the file
    # explains itself later.
    data = {
        "focal_length_px": focal_length,
        "avg_box_height_px": avg_h,
        "known_distance_m": KNOWN_DISTANCE_M,
        "bottle_height_m": BOTTLE_HEIGHT_M,
        "frames_averaged": CALIBRATION_FRAMES,
        "calibrated_at": datetime.now().isoformat(timespec="seconds"),
    }
    # indent=2 puts each field on its own line so it's easy to read.
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(data, f, indent=2)


def draw_detection(frame, x1, y1, x2, y2, label):
    # Draw the box: top-left corner, bottom-right corner, color, line thickness.
    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

    # Measure how big the text will be, so the label background fits it.
    # text_w/text_h are its width and height in pixels; baseline is the extra
    # space below the line that letters like "g" hang into.
    (text_w, text_h), baseline = cv2.getTextSize(label, FONT, FONT_SCALE,
                                                 FONT_THICKNESS)
    label_h = text_h + baseline + 8  # 8 px of padding

    # Put the label on top of the box. If there's no room above (the box
    # touches the top of the screen), put it just inside the box instead.
    label_top = y1 - label_h if y1 >= label_h else y1

    # Filled green rectangle behind the text (thickness -1 means "filled"),
    # so the text is readable on any background.
    cv2.rectangle(frame, (x1, label_top), (x1 + text_w + 8, label_top + label_h),
                  BOX_COLOR, -1)

    # Draw the text on the background. putText's position is the bottom-left
    # of the text, so move down by the text height plus padding.
    cv2.putText(frame, label, (x1 + 4, label_top + text_h + 4), FONT,
                FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)


def panel_text_width(text):
    # Width of a piece of text in the panel's font, in pixels.
    (text_w, _), _ = cv2.getTextSize(text, FONT, PANEL_FONT_SCALE,
                                     PANEL_FONT_THICKNESS)
    return text_w


def draw_panel(frame, rows, status):
    # Draw the HUD panel: a semi-transparent box with one (label, value) row
    # per line, plus an optional status line at the bottom.

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


def main():
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

    # Load a saved focal length, or None if we haven't calibrated yet.
    focal_length = load_calibration()
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

    # FPS state.
    frame_time = None  # smoothed seconds per loop pass; None until measured
    last_loop_start = None  # when the previous loop pass started
    frame_count = 0  # loop passes so far

    # Connect to the webcam. This returns a "capture" object we read frames from.
    cap = cv2.VideoCapture(CAMERA_INDEX)

    # VideoCapture does not raise an error if it fails, so we must check.
    if not cap.isOpened():
        print("Error: could not open the webcam.")
        print("Check that no other app (Zoom, Teams, browser) is using it,")
        print("and that Windows camera privacy settings allow desktop apps.")
        sys.exit(1)  # exit code 1 means "something went wrong"

    print("Webcam opened. Press c to calibrate, +/- to change speed, q to quit.")

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

            # Grab one frame (one still picture) from the camera.
            # ret is True if it worked; frame is the image itself.
            ret, frame = cap.read()

            # If the read failed (e.g. camera unplugged), stop instead of crashing.
            if not ret:
                print("Error: could not read a frame from the webcam.")
                break

            # Run YOLO on this frame. conf= drops anything below our threshold
            # (the model's own default is 0.25). imgsz= is the size YOLO
            # shrinks the frame to before detecting. verbose=False stops it
            # printing a line for every frame. It returns one result per
            # image; we gave it one image, so take the first.
            result = model(frame, conf=CONF_THRESHOLD, imgsz=IMGSZ,
                           verbose=False)[0]

            # Every bottle found in this frame, as (x1, y1, x2, y2, conf).
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

                # xyxy = top-left (x1, y1) and bottom-right (x2, y2) corners,
                # in pixels of our frame. OpenCV needs whole numbers.
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                bottles.append((x1, y1, x2, y2, conf))

                # Label like "bottle, 0.75" (confidence to 2 decimal places).
                label = f"{model.names[cls_id]}, {conf:.2f}"
                draw_detection(frame, x1, y1, x2, y2, label)

            # The most confident bottle (index 4 is conf), or None.
            best = max(bottles, key=lambda b: b[4]) if bottles else None

            # frame.shape[0] is the frame's height in pixels.
            frame_height = frame.shape[0]

            # Smooth h only while there's exactly one bottle, fully in view.
            # Otherwise start over: with two bottles the "best" one can switch
            # between them, and a cut-off box has a wrong h, so neither should
            # be mixed into the running value.
            if len(bottles) == 1 and not touches_edge(best[1], best[3],
                                                      frame_height):
                smoothed_h = smooth_h(smoothed_h, best[3] - best[1])
            else:
                smoothed_h = None

            # Distance and time to hit for the same bottle.
            distance_text, time_text, reason = measurement_values(
                best, frame_height, focal_length, speed, smoothed_h)

            # Bearing angles for the same bottle, from its raw box center.
            bearing_h_text, bearing_v_text = bearing_values(
                best, frame, focal_length)

            # While calibrating, collect one h per frame, but only from frames
            # we can trust.
            if calibrating:
                if len(bottles) == 0:
                    status = "no bottle"
                elif len(bottles) > 1:
                    # Can't tell which box is the bottle at the known distance.
                    status = f"{len(bottles)} bottles"
                else:
                    x1, y1, x2, y2, conf = bottles[0]
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
                    save_calibration(focal_length, avg_h)
                    message = f"Saved: f {focal_length:.1f} px"
                    print(f"Calibrated: avg h = {avg_h:.1f} px, "
                          f"focal length = {focal_length:.1f} px")
                else:
                    message = (f"Calibrating {len(samples)}/"
                               f"{CALIBRATION_FRAMES}: {status}")
                message_time = time.perf_counter()

            # The status line: a calibration message wins while calibrating
            # or for a few seconds after it was set; otherwise the reason for
            # any "--" values (or nothing).
            message_is_fresh = (time.perf_counter() - message_time
                                < STATUS_MESSAGE_SECONDS)
            if message and (calibrating or message_is_fresh):
                status_line = message
            else:
                status_line = reason

            # Panel values.
            conf_text = f"{best[4]:.2f}" if best is not None else "--"
            fps_text = f"{1 / frame_time:.0f}" if frame_time else "--"

            origin = get_origin(frame)

            # The target line only needs a bottle, not calibration.
            if best is not None:
                draw_target_line(frame, origin, best)

            # Draw the origin crosshair after detection (so YOLO only ever sees
            # the clean frame) and after the line (so it sits on top of it).
            draw_crosshair(frame, origin)

            # The panel goes on last, on top of everything else.
            draw_panel(frame, [
                ("DIST", distance_text),
                ("TIME", time_text),
                ("BRG H", bearing_h_text),
                ("BRG V", bearing_v_text),
                ("CONF", conf_text),
                ("SPEED", f"{speed:.1f} m/s  (+/-)"),
                ("FPS", fps_text),
            ], status_line)

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
            if key == ord("c"):
                if calibrating:
                    calibrating = False
                    message = "Calibration cancelled"
                else:
                    calibrating = True
                    samples = []
                    message = f"Calibrating 0/{CALIBRATION_FRAMES}"
                message_time = time.perf_counter()

            # + (or =, the same key without Shift) speeds up; - slows down.
            # round() stops tiny decimal errors building up; max() keeps the
            # speed from going below 0.
            if key in (ord("+"), ord("=")):
                speed = round(speed + SPEED_STEP_MPS, 2)
            if key == ord("-"):
                speed = max(0.0, round(speed - SPEED_STEP_MPS, 2))

            # Stop if the user closed the window with the X button.
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        # Turn the camera off so other programs can use it.
        cap.release()
        # Close every window OpenCV opened.
        cv2.destroyAllWindows()


# Only run main() when this file is run directly (not when it's imported).
if __name__ == "__main__":
    main()
