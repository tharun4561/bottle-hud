"""Milestone 5: detect bottles with YOLO, mark a bottom-center origin,
calibrate the camera's focal length, and show live distance and time to hit.

Press c to calibrate (press c again to cancel).
Press + / - to change the interceptor speed.
Press q (or close the window) to quit.
"""

import json  # reading and writing calibration.json
import sys  # used to exit the program with an error code
from datetime import datetime  # timestamp saved with the calibration
from pathlib import Path  # building file paths that work on any OS

import cv2  # OpenCV: camera access, windows, and image drawing
from ultralytics import YOLO  # Ultralytics: loads and runs YOLO models

CAMERA_INDEX = 0  # 0 = the first camera Windows finds (the built-in webcam)
WINDOW_NAME = "Bottle HUD"  # the title shown on the window

MODEL_PATH = "yolo26n.pt"  # YOLO26 nano; downloaded automatically on first run
TARGET_CLASS = "bottle"  # the only class we want to show
CONF_THRESHOLD = 0.3  # hide detections the model is less sure about than this
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

# Status text in the top-left corner.
HUD_BG_COLOR = (0, 0, 0)  # black background
HUD_TEXT_COLOR = (255, 255, 255)  # white text

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

# Smoothing of h (exponential moving average). Between 0 and 1: smaller is
# steadier but slower to react, larger is quicker but jumpier.
SMOOTHING_ALPHA = 0.3


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


def touches_edge(y1, y2, frame_height):
    # A box touching the top or bottom edge is probably cut off, which makes
    # h too small (and the distance too large).
    return y1 <= EDGE_MARGIN or y2 >= frame_height - EDGE_MARGIN


def smooth_h(smoothed_h, new_h):
    # Exponential moving average: mix a fraction (SMOOTHING_ALPHA) of the new
    # value into the running value. One noisy frame can only move it a little.
    # With no running value yet, start from the new value.
    if smoothed_h is None:
        return new_h
    return SMOOTHING_ALPHA * new_h + (1 - SMOOTHING_ALPHA) * smoothed_h


def measurement_texts(best, frame_height, focal_length, speed, smoothed_h):
    # Work out the distance and time-to-hit lines for the HUD. best is the
    # most confident bottle as (x1, y1, x2, y2, conf), or None if there's no
    # bottle. smoothed_h is the smoothed box height, or None if there isn't
    # one (then the raw h of best is used). Returns (distance_text, time_text).
    if best is None:
        return "Distance: --", "Time to hit: --"
    if focal_length is None:
        return "Distance: -- (not calibrated)", "Time to hit: --"

    x1, y1, x2, y2, conf = best
    if touches_edge(y1, y2, frame_height):
        return "Distance: -- (bottle at edge)", "Time to hit: --"

    # Prefer the smoothed h; fall back to this frame's raw h.
    h = smoothed_h if smoothed_h is not None else y2 - y1

    # compute_distance divides by h, so a 0-pixel box would crash it.
    if h <= 0:
        return "Distance: --", "Time to hit: --"

    try:
        distance = compute_distance(h, focal_length, BOTTLE_HEIGHT_M)
    except NotImplementedError:
        return "Distance: not implemented", "Time to hit: --"
    distance_text = f"Distance: {distance:.2f} m"

    # Dividing by a speed of 0 would crash, and the interceptor would never
    # arrive anyway, so don't call compute_time_to_hit at all.
    if speed <= 0:
        return distance_text, "Time to hit: -- (speed 0)"

    try:
        time_to_hit = compute_time_to_hit(distance, speed)
    except NotImplementedError:
        return distance_text, "Time to hit: not implemented"
    return distance_text, f"Time to hit: {time_to_hit:.2f} s"


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


def draw_hud_text(frame, text, line):
    # Draw one line of status text in the top-left corner. line 0 is the top
    # line, line 1 goes under it, and so on.
    # Measure the line height from a fixed sample ("(" and "g" are the tallest
    # and lowest characters), so every line is the same height no matter what
    # text it holds, and the lines never overlap.
    (_, text_h), baseline = cv2.getTextSize("(Ag)", FONT, FONT_SCALE,
                                            FONT_THICKNESS)
    (text_w, _), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
    line_h = text_h + baseline + 8  # 8 px of padding
    top = 10 + line * (line_h + 4)  # 10 px from the top, 4 px between lines

    # Filled background, then the text on top of it (same idea as the label).
    cv2.rectangle(frame, (10, top), (10 + text_w + 8, top + line_h),
                  HUD_BG_COLOR, -1)
    cv2.putText(frame, text, (14, top + text_h + 4), FONT, FONT_SCALE,
                HUD_TEXT_COLOR, FONT_THICKNESS)


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
    message = ""  # last calibration result or problem, shown on screen

    speed = DEFAULT_SPEED_MPS  # interceptor speed, changed with + and -
    smoothed_h = None  # running EMA of h; None until there's a bottle to track

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
            # Grab one frame (one still picture) from the camera.
            # ret is True if it worked; frame is the image itself.
            ret, frame = cap.read()

            # If the read failed (e.g. camera unplugged), stop instead of crashing.
            if not ret:
                print("Error: could not read a frame from the webcam.")
                break

            # Run YOLO on this frame. conf= drops anything below our threshold
            # (the model's own default is 0.25). verbose=False stops it
            # printing a line for every frame. It returns one result per
            # image; we gave it one image, so take the first.
            result = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]

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

            # Show the raw h, plus the smoothed one when there is one.
            if best is None:
                h_text = "h: --"
            elif smoothed_h is None:
                h_text = f"h: {best[3] - best[1]} px"  # h = y2 - y1
            else:
                h_text = f"h: {best[3] - best[1]} px (smoothed {smoothed_h:.1f})"

            # Distance and time to hit for the same bottle.
            distance_text, time_text = measurement_texts(
                best, frame_height, focal_length, speed, smoothed_h)

            # While calibrating, collect one h per frame, but only from frames
            # we can trust.
            if calibrating:
                if len(bottles) == 0:
                    status = "no bottle, waiting"
                elif len(bottles) > 1:
                    # Can't tell which box is the bottle at the known distance.
                    status = f"{len(bottles)} bottles, waiting"
                else:
                    x1, y1, x2, y2, conf = bottles[0]
                    if touches_edge(y1, y2, frame_height):
                        status = "bottle at edge, waiting"
                    else:
                        samples.append(y2 - y1)
                        status = "collecting"

                # Enough samples: average them and compute the focal length.
                if len(samples) >= CALIBRATION_FRAMES:
                    calibrating = False
                    avg_h = sum(samples) / len(samples)
                    try:
                        focal_length = compute_focal_length(
                            avg_h, KNOWN_DISTANCE_M, BOTTLE_HEIGHT_M)
                        save_calibration(focal_length, avg_h)
                        message = f"Saved: avg h {avg_h:.1f} px"
                        print(f"Calibrated: avg h = {avg_h:.1f} px, "
                              f"focal length = {focal_length:.1f} px")
                    except NotImplementedError:
                        message = "compute_focal_length not implemented yet"
                        print(f"Calibration: avg h = {avg_h:.1f} px, but "
                              "compute_focal_length is not implemented yet.")
                else:
                    message = (f"Calibrating... {len(samples)}/"
                               f"{CALIBRATION_FRAMES} ({status})")

            # Status lines in the top-left corner.
            draw_hud_text(frame, h_text, 0)
            if focal_length is None:
                draw_hud_text(frame, "Not calibrated: hold bottle at "
                              f"{KNOWN_DISTANCE_M:.2f} m and press c", 1)
            else:
                draw_hud_text(frame, f"Focal length: {focal_length:.1f} px", 1)
            draw_hud_text(frame, f"Speed: {speed:.1f} m/s (+/-)", 2)
            draw_hud_text(frame, distance_text, 3)
            draw_hud_text(frame, time_text, 4)
            if message:
                draw_hud_text(frame, message, 5)

            # Draw the origin crosshair after detection (so YOLO only ever sees
            # the clean frame) and last (so nothing else covers it).
            origin = get_origin(frame)
            draw_crosshair(frame, origin)

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
                    message = f"Calibrating... 0/{CALIBRATION_FRAMES}"

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
