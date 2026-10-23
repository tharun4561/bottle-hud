"""Bottle tracker: detection, track ID lock, calibration, 3D position and velocity.

Tracker.update() takes a camera frame and its timestamp and returns a Track:
the detected bottles, the target bottle being followed, its distance, its 3D
position and its 3D velocity. Nothing here draws on the frame, and nothing
here needs to know about the interceptor.

Positions use camera coordinates: the origin is the camera lens, X is
positive to the right, Y positive up, and Z positive straight out of the
camera. Everything is in meters unless a name says px.
"""

import json  # reading and writing calibration.json
import sys  # used to exit the program with an error code
from collections import deque  # a list that's quick to trim from the front
from datetime import datetime  # timestamp saved with the calibration
from pathlib import Path  # building file paths that work on any OS
from typing import NamedTuple, Optional

from ultralytics import YOLO  # Ultralytics: loads and runs YOLO models

MODEL_PATH = "yolo26n.pt"  # YOLO26 nano; downloaded automatically on first run
TARGET_CLASS = "bottle"  # the only class we want to track
CONF_THRESHOLD = 0.05  # hide detections the model is less sure about than this
IMGSZ = 640  # image size YOLO works at; smaller is faster but sees less detail
# Tracker that gives each bottle an ID that stays the same across frames.
# ByteTrack is lighter than the default BoT-SORT, which spends extra time
# compensating for a moving camera (ours doesn't move). The settings file is
# the Ultralytics ByteTrack default with lower confidence thresholds: the
# tracker only starts a track (and so only shows a bottle) at its own
# new_track_thresh, so CONF_THRESHOLD alone can't make it show weaker ones.
TRACKER_CONFIG = str(Path(__file__).parent / "bytetrack_low.yaml")

# 3D velocity (m/s) from a least-squares fit over the last N positions.
VELOCITY_WINDOW = 10  # samples per fit; more = steadier but slower to react
LOST_TIMEOUT_S = 0.5  # missing longer than this (s) resets the history

# Calibration.
BOTTLE_HEIGHT_M = 0.142  # real height of the bottle, in meters
KNOWN_DISTANCE_M = 1.00  # distance from the webcam lens to the bottle, in meters
CALIBRATION_FRAMES = 30  # how many frames of h to average
CALIBRATION_FILE = "calibration.json"
EDGE_MARGIN = 5  # boxes this close (px) to the top/bottom edge may be cut off

# Keep calibration.json next to this file, no matter which folder we run from.
CALIBRATION_PATH = Path(__file__).parent / CALIBRATION_FILE

# Smoothing (exponential moving average) for the bottle's box height h.
# Between 0 and 1: smaller is steadier but slower to react, larger is quicker
# but jumpier.
SMOOTHING_ALPHA = 0.3


class Bottle(NamedTuple):
    """One detected bottle. Indexable like a plain tuple, in this order."""
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    track_id: Optional[int]  # None if the box isn't part of a track yet


class Track(NamedTuple):
    """Everything the tracker knows about one frame.

    bottles: every bottle found in the frame
    target: the bottle being followed (one of bottles), or None
    distance: the target's smoothed distance along the view direction in
        meters, or None
    reason: why distance is None (not calibrated, bottle at the edge), or None
    offsets: (dx, dy) of the target's raw box center from the image center,
        in pixels (right and up positive), or None when there is no target
    position: the target's 3D (X, Y, Z) in meters, or None
    velocity: the target's 3D (vx, vy, vz) in m/s, or None while measuring
    message: calibration progress or result set during this frame, or None
    focal_length: the focal length (px) this frame was measured with, or
        None. A calibration that finishes on this frame only applies from
        the next one, so this can differ from Tracker.focal_length.
    """
    bottles: list
    target: Optional[Bottle]
    distance: Optional[float]
    reason: Optional[str]
    offsets: Optional[tuple]
    position: Optional[tuple]
    velocity: Optional[tuple]
    message: Optional[str]
    focal_length: Optional[float]


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


def measure_distance(target, frame_height, focal_length, smoothed_h):
    # Work out the distance for the target. target is the bottle as
    # (x1, y1, x2, y2, conf, track_id), or None if there's no bottle.
    # smoothed_h is the smoothed box height, or None if there isn't one (then
    # the raw h of target is used).
    # Returns (distance, reason). distance is in meters, or None. reason
    # explains a missing distance for the status line, or is None when
    # there's nothing to explain.
    if focal_length is None:
        return None, (f"Not calibrated: hold bottle at "
                      f"{KNOWN_DISTANCE_M:.2f} m, press c")
    if target is None:
        return None, None

    x1, y1, x2, y2 = target[:4]
    if touches_edge(y1, y2, frame_height):
        return None, "Bottle at edge: distance unreliable"

    # Prefer the smoothed h; fall back to this frame's raw h.
    h = smoothed_h if smoothed_h is not None else y2 - y1

    # compute_distance divides by h, so a 0-pixel box would crash it.
    if h <= 0:
        return None, None

    return compute_distance(h, focal_length, BOTTLE_HEIGHT_M), None


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


class Tracker:
    """Detects and tracks bottles, and measures the target bottle in 3D."""

    def __init__(self):
        # Load the model once (loading is slow).
        self.model = YOLO(MODEL_PATH)

        # model.names maps each class ID to its name, e.g. {39: 'bottle', ...}.
        # Look up the ID for our target class instead of hardcoding the number.
        self.bottle_id = None
        for class_id, name in self.model.names.items():
            if name == TARGET_CLASS:
                self.bottle_id = class_id
                break

        if self.bottle_id is None:
            print(f"Error: the model has no class named '{TARGET_CLASS}'.")
            sys.exit(1)

        # Load a saved focal length (None if we haven't calibrated yet) and
        # the frame size it was made with (None if the file doesn't say).
        self.focal_length, self.calibration_size = load_calibration()
        if self.focal_length is None:
            print("Not calibrated yet. Hold the bottle at "
                  f"{KNOWN_DISTANCE_M:.2f} m and press c.")
        else:
            print(f"Loaded focal length: {self.focal_length:.1f} px")

        # Calibration state.
        self.calibrating = False  # True while we're collecting samples
        self.samples = []  # box heights (h) collected so far

        # Tracking and velocity state.
        self.smoothed_h = None  # running EMA of h; None until there's a bottle
        self.locked_id = None  # track ID of the bottle we're following
        self.history_id = None  # the track ID the history belongs to
        self.last_seen = None  # when the target was last seen (seconds)
        # (time, X, Y, Z) of the target in meters, raw (unsmoothed); maxlen
        # drops the oldest sample automatically, so it always holds the last N.
        self.history_3d = deque(maxlen=VELOCITY_WINDOW)

    def toggle_calibration(self):
        """Start calibration, or cancel it if it's already running.

        Returns the message to show on the status line.
        """
        if self.calibrating:
            self.calibrating = False
            return "Calibration cancelled"
        self.calibrating = True
        self.samples = []
        return f"Calibrating 0/{CALIBRATION_FRAMES}"

    def update(self, frame, frame_t):
        """Process one frame taken at frame_t seconds and return a Track."""
        # Run YOLO in track mode on this frame: it detects, then matches
        # each box to the boxes of earlier frames and gives it an ID.
        # persist=True keeps the tracker's memory between calls. conf=
        # drops anything below our threshold (the model's own default is
        # 0.25). imgsz= is the size YOLO shrinks the frame to before
        # detecting. verbose=False stops it printing a line for every
        # frame. It returns one result per image; we gave it one image,
        # so take the first.
        result = self.model.track(frame, persist=True, tracker=TRACKER_CONFIG,
                                  conf=CONF_THRESHOLD, imgsz=IMGSZ,
                                  verbose=False)[0]

        # Every bottle found in this frame.
        bottles = []

        # result.boxes holds one entry per detected object.
        for box in result.boxes:
            # Class ID and confidence come back as tensors; int() and
            # float() turn them into plain Python numbers.
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # Skip anything that isn't a bottle.
            if cls_id != self.bottle_id:
                continue

            # The track ID, or None if this box isn't part of a track yet.
            track_id = int(box.id[0]) if box.id is not None else None

            # xyxy = top-left (x1, y1) and bottom-right (x2, y2) corners,
            # in pixels of our frame. OpenCV needs whole numbers.
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            bottles.append(Bottle(x1, y1, x2, y2, conf, track_id))

        # The bottle we measure and aim at (same one for everything).
        target = choose_target(bottles, self.locked_id)
        if target is not None:
            self.locked_id = target[5]

        # frame.shape[0] is the frame's height in pixels.
        frame_height = frame.shape[0]

        # Smooth h only while there's exactly one bottle, fully in view.
        # Otherwise start over: with two bottles the target can switch
        # between them, and a cut-off box has a wrong h, so neither should
        # be mixed into the running value.
        if len(bottles) == 1 and not touches_edge(target[1], target[3],
                                                  frame_height):
            self.smoothed_h = smooth_h(self.smoothed_h, target[3] - target[1])
        else:
            self.smoothed_h = None

        # 3D position history. Start over if the target is a different
        # track, or if it was missing for too long. A short miss is fine:
        # the timestamps just show a bigger gap.
        missing_too_long = (self.last_seen is not None
                            and frame_t - self.last_seen > LOST_TIMEOUT_S)
        if target is not None:
            if target[5] != self.history_id or missing_too_long:
                self.history_3d.clear()
                self.history_id = target[5]
            self.last_seen = frame_t
        elif missing_too_long:
            self.history_3d.clear()

        # Distance for the target.
        distance, reason = measure_distance(target, frame_height,
                                            self.focal_length, self.smoothed_h)

        # Offset of the raw box center from the image center. The bearings
        # and the 3D position both use it, so they always agree.
        offsets = center_offsets(target, frame) if target is not None else None

        # 3D position: raw box center and the (smoothed) distance, so all
        # the readouts agree. None when there's no distance (no bottle, not
        # calibrated, or at the edge).
        position = None
        if target is not None and self.focal_length is not None \
                and distance is not None:
            position = compute_3d_position(offsets[0], offsets[1], distance,
                                           self.focal_length)

        # 3D velocity sample: this frame's RAW position. The fit does the
        # smoothing itself, so feeding it the smoothed distance would
        # smooth twice and add lag. Only tracked bottles, and only frames
        # with a distance (skips "--" frames, e.g. bottle at the edge,
        # without resetting).
        if distance is not None and target[5] is not None:
            raw_h = target[3] - target[1]
            if raw_h > 0:
                raw_distance = compute_distance(raw_h, self.focal_length,
                                                BOTTLE_HEIGHT_M)
                dx, dy = offsets
                X, Y, Z = compute_3d_position(dx, dy, raw_distance,
                                              self.focal_length)
                self.history_3d.append((frame_t, X, Y, Z))

        # Fit once the window is full, so the noise level is always the
        # same. Times are shifted to start at 0 to keep the numbers small
        # (better floating-point precision).
        velocity = None  # (vx, vy, vz) in m/s, or None
        if len(self.history_3d) == VELOCITY_WINDOW:
            t_first = self.history_3d[0][0]
            times = [s[0] - t_first for s in self.history_3d]
            xs = [s[1] for s in self.history_3d]
            ys = [s[2] for s in self.history_3d]
            zs = [s[3] for s in self.history_3d]
            velocity = (fit_velocity(times, xs),
                        fit_velocity(times, ys),
                        fit_velocity(times, zs))

        # While calibrating, collect one h per frame, but only from frames
        # we can trust. This comes after the measurements above, so a frame
        # that finishes the calibration is still measured with the old
        # focal length, and the new one applies from the next frame.
        measured_with = self.focal_length
        message = self._calibration_step(bottles, frame, frame_height)

        return Track(bottles, target, distance, reason, offsets, position,
                     velocity, message, measured_with)

    def _calibration_step(self, bottles, frame, frame_height):
        # Returns the message to show for this frame, or None when not
        # calibrating.
        if not self.calibrating:
            return None

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
                self.samples.append(y2 - y1)
                status = f"h {y2 - y1}"

        # Enough samples: average them and compute the focal length.
        if len(self.samples) >= CALIBRATION_FRAMES:
            self.calibrating = False
            avg_h = sum(self.samples) / len(self.samples)
            self.focal_length = compute_focal_length(
                avg_h, KNOWN_DISTANCE_M, BOTTLE_HEIGHT_M)
            self.calibration_size = (frame.shape[1], frame.shape[0])
            save_calibration(self.focal_length, avg_h, self.calibration_size)
            print(f"Calibrated: avg h = {avg_h:.1f} px, "
                  f"focal length = {self.focal_length:.1f} px")
            return f"Saved: f {self.focal_length:.1f} px"
        return f"Calibrating {len(self.samples)}/{CALIBRATION_FRAMES}: {status}"
