"""Record a test clip from the webcam for main.py --video.

Records at the webcam's default resolution, the same one main.py uses, so
the calibrated focal length stays valid for the clip.

Usage:
    .venv\\Scripts\\python.exe record.py clip.mp4
    .venv\\Scripts\\python.exe record.py clip.mp4 --overwrite

Press q (or close the window) to stop recording.
"""

import argparse  # reading command-line options
import sys  # used to exit the program with an error code
import time  # measuring the webcam's real frame rate
from pathlib import Path  # file names and checking whether they exist

import cv2  # OpenCV: camera access, windows, and video files

CAMERA_INDEX = 0  # same camera as main.py
WINDOW_NAME = "Recording"
WARMUP_SECONDS = 1.5  # frames to throw away first: webcams start out slow
CODEC = "mp4v"  # a codec OpenCV can always write to .mp4 files


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record a webcam clip for main.py --video.")
    parser.add_argument("output", help="file to save, e.g. clip.mp4")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace the file if it already exists")
    return parser.parse_args()


def rewrite_with_fps(source, output, fps):
    # Copy every frame of source into a new file that stores the given frame
    # rate. Returns the number of frames copied.
    reader = cv2.VideoCapture(str(source))
    width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*CODEC),
                             fps, (width, height))
    copied = 0
    while True:
        ret, frame = reader.read()
        if not ret:
            break
        writer.write(frame)
        copied += 1
    reader.release()
    writer.release()
    return copied


def main():
    args = parse_args()
    output = Path(args.output)
    # Recorded here first, then rewritten to output with the real frame rate.
    temp = output.with_name(output.stem + ".recording" + output.suffix)

    # Don't silently replace an existing recording.
    if output.exists() and not args.overwrite:
        print(f"Error: {output} already exists. Use --overwrite to replace it.")
        sys.exit(1)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: could not open the webcam.")
        sys.exit(1)

    # Throw away the first frames: right after opening, a webcam delivers
    # frames slowly while its exposure settles.
    print("Starting the webcam...")
    warmup_start = time.perf_counter()
    while time.perf_counter() - warmup_start < WARMUP_SECONDS:
        ret, frame = cap.read()
        if not ret:
            print("Error: could not read from the webcam.")
            cap.release()
            sys.exit(1)

    height, width = frame.shape[:2]
    nominal_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = cv2.VideoWriter(str(temp), cv2.VideoWriter_fourcc(*CODEC),
                             nominal_fps, (width, height))
    if not writer.isOpened():
        print(f"Error: could not create {temp}.")
        cap.release()
        sys.exit(1)

    print(f"Recording {width}x{height} to {output}. Press q to stop.")
    frames = 0
    start = time.perf_counter()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: could not read from the webcam.")
                break
            writer.write(frame)  # save the clean frame first...
            frames += 1

            # ...then draw the preview's red dot and elapsed time on it, so
            # they never end up in the file.
            elapsed = time.perf_counter() - start
            cv2.circle(frame, (20, 20), 8, (0, 0, 255), -1)
            cv2.putText(frame, f"REC {elapsed:.1f} s", (36, 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
        elapsed = time.perf_counter() - start
    finally:
        writer.release()
        cap.release()
        cv2.destroyAllWindows()

    if frames == 0:
        temp.unlink(missing_ok=True)
        print("Nothing was recorded.")
        sys.exit(1)

    # The webcam doesn't always deliver the frame rate it reports (dim light
    # slows it down, and slow frames get dropped), and main.py takes its
    # timestamps from the rate stored in the file. So store the real average
    # rate: rewrite the clip with frames / seconds.
    actual_fps = frames / elapsed
    print(f"Recorded {frames} frames in {elapsed:.1f} s "
          f"({actual_fps:.1f} FPS). Saving...")
    copied = rewrite_with_fps(temp, output, actual_fps)
    temp.unlink()
    print(f"Saved {copied} frames at {actual_fps:.1f} FPS to {output}.")


if __name__ == "__main__":
    main()
