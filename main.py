"""Milestone 2: detect bottles in the live webcam feed with YOLO.

Press q (or close the window) to quit.
"""

import sys  # used to exit the program with an error code

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

    # Connect to the webcam. This returns a "capture" object we read frames from.
    cap = cv2.VideoCapture(CAMERA_INDEX)

    # VideoCapture does not raise an error if it fails, so we must check.
    if not cap.isOpened():
        print("Error: could not open the webcam.")
        print("Check that no other app (Zoom, Teams, browser) is using it,")
        print("and that Windows camera privacy settings allow desktop apps.")
        sys.exit(1)  # exit code 1 means "something went wrong"

    print("Webcam opened. Press q in the video window to quit.")

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

                # Label like "bottle, 0.75" (confidence to 2 decimal places).
                label = f"{model.names[cls_id]}, {conf:.2f}"
                draw_detection(frame, x1, y1, x2, y2, label)

            # Hand the frame to the window. Nothing is drawn until waitKey runs.
            cv2.imshow(WINDOW_NAME, frame)

            # Let the window process its events (actually draw the frame) and
            # wait up to 1 ms for a key press. & 0xFF keeps only the lowest
            # 8 bits of the key code so the comparison works on every platform.
            key = cv2.waitKey(1) & 0xFF

            # ord('q') is the key code for the letter q.
            if key == ord("q"):
                break

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
