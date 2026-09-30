"""Milestone 1: show the live webcam feed in a window. Press q to quit."""

import sys  # used to exit the program with an error code

import cv2  # OpenCV: camera access, windows, and image drawing

CAMERA_INDEX = 0  # 0 = the first camera Windows finds (the built-in webcam)
WINDOW_NAME = "Bottle HUD"  # the title shown on the window


def main():
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
