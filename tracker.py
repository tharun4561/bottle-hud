"""Target tracker: detection, track ID lock, calibration, Kalman-filtered 3D
position and velocity.

Tracker.update() takes a camera frame and its timestamp and returns a Track:
the detected targets, the target being followed, its distance, its 3D
position and its 3D velocity (both from a Kalman filter). Nothing here draws
on the frame, and nothing here needs to know about the interceptor.

Positions use camera coordinates: the origin is the camera lens, X is
positive to the right, Y positive up, and Z positive straight out of the
camera. Everything is in meters unless a name says px.
"""

import json  # reading and writing calibration.json
from datetime import datetime  # timestamp saved with the calibration
from pathlib import Path  # building file paths that work on any OS
from typing import NamedTuple, Optional

import numpy as np  # a black frame for the warm-up
from ultralytics import YOLO  # Ultralytics: loads and runs YOLO models

# YOLO nano trained on Roboflow to find the drone (starting from yolo26n.pt).
MODEL_PATH = str(Path(__file__).parent / "PavoWeights.pt")
TARGET_CLASS = "Drone-Interceptor"  # the model's only class
CONF_THRESHOLD = 0.15  # hide detections the model is less sure about than this
IMGSZ = 640  # image size YOLO works at; smaller is faster but sees less detail
# Tracker that gives each bottle an ID that stays the same across frames.
# ByteTrack is lighter than the default BoT-SORT, which spends extra time
# compensating for a moving camera (ours doesn't move). The settings file is
# the Ultralytics ByteTrack default with lower confidence thresholds: the
# tracker only starts a track (and so only shows a bottle) at its own
# new_track_thresh, so CONF_THRESHOLD alone can't make it show weaker ones.
TRACKER_CONFIG = str(Path(__file__).parent / "bytetrack_low.yaml")

LOST_TIMEOUT_S = 0.5  # missing longer than this (s) resets the filter

# Kalman filter for the target's 3D position and velocity. The bottle is
# assumed to move at a roughly constant velocity; ACCEL_SIGMA_MPS2 is how much
# that velocity may change per second (bigger = follows turns faster but is
# noisier). The two pixel noise values say how much the box jitters: they set
# how much each new measurement is trusted (bigger = smoother but slower).
ACCEL_SIGMA_MPS2 = 1.0  # process noise: typical acceleration, in m/s^2
CENTER_SIGMA_PX = 1.0  # noise in the box center position, in pixels
SIZE_SIGMA_PX = 1.5  # noise in the measured box side (see SIZE_AXIS), in pixels
INITIAL_VELOCITY_SIGMA_MPS = 3.0  # how unsure the first velocity guess (0) is
WARMUP_UPDATES = 5  # measurements before the velocity is reported

# Calibration.
# Distance comes from how big one side of the box looks: Z = f * size / pixels.
# The drone is 97 x 97 mm and about 30 mm tall. Its width is the better ruler:
# it's about 3x more pixels than the height, so a pixel of noise matters less.
# A square body seen at an angle looks up to 1.4x wider, so keep it facing the
# camera when calibrating. To use the height instead, set "height" and 0.030.
SIZE_AXIS = "width"  # which side of the box to measure: "width" or "height"
TARGET_SIZE_M = 0.097  # real size of the target along SIZE_AXIS, in meters
KNOWN_DISTANCE_M = 1.00  # distance from the webcam lens to the target, in meters
CALIBRATION_FRAMES = 30  # how many frames of h to average
CALIBRATION_FILE = "calibration.json"
EDGE_MARGIN = 5  # boxes this close (px) to the top/bottom edge may be cut off

# Keep calibration.json next to this file, no matter which folder we run from.
CALIBRATION_PATH = Path(__file__).parent / CALIBRATION_FILE

# Smoothing (exponential moving average) for the measured box side.
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
    position: the target's 3D (X, Y, Z) in meters, filtered by the Kalman
        filter (the raw box position for a bottle that isn't tracked yet), or
        None
    velocity: the target's 3D (vx, vy, vz) in m/s from the Kalman filter, or
        None while it's warming up
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


def compute_focal_length(size_px, known_distance_m, target_size_m):
    """Return the camera's focal length in pixels.

    size_px: the target's box side (SIZE_AXIS) in the image, in pixels
    known_distance_m: distance from the camera to the target, in meters
    target_size_m: the target's real size along that side, in meters
    """
    f = (size_px * known_distance_m) / target_size_m
    return f


def compute_distance(size_px, focal_length_px, target_size_m):
    """Return the distance from the camera to the target, in meters.

    size_px: the target's box side (SIZE_AXIS) in the image, in pixels
    focal_length_px: the calibrated focal length, in pixels
    target_size_m: the target's real size along that side, in meters
    """
    d = (target_size_m * focal_length_px) / size_px
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


def measurement_sigmas(distance_m, focal_length_px):
    """Return the expected noise of a raw 3D position, in meters.

    A pixel of jitter is worth more meters the farther away the bottle is.
    The sideways position X = dx * Z / f moves by Z / f meters per pixel.
    The depth Z = f * S / s (S the real size, s its pixels) is worse: one
    pixel of s changes it by Z^2 / (f * S), so its noise grows with the
    square of the distance.

    Returns (sigma_x, sigma_y, sigma_z).
    """
    sideways = distance_m * CENTER_SIGMA_PX / focal_length_px
    depth = (distance_m ** 2 * SIZE_SIGMA_PX
             / (focal_length_px * TARGET_SIZE_M))
    return sideways, sideways, depth


class AxisFilter:
    """Kalman filter for one axis, with a constant-velocity model.

    The state is the position p and velocity v. The covariance P is the 2x2
    matrix [[pp, pv], [pv, vv]] (stored as three numbers because it's
    symmetric): pp is how unsure the position is, vv how unsure the velocity
    is, and pv how much their errors go together.
    """

    def __init__(self, position, position_var):
        self.p = position
        self.v = 0.0  # no idea yet; the large vv below says so
        self.pp = position_var
        self.pv = 0.0
        self.vv = INITIAL_VELOCITY_SIGMA_MPS ** 2

    def predict(self, dt):
        """Move the state forward dt seconds (position += velocity * dt)."""
        # P = F P F^T + Q with F = [[1, dt], [0, 1]]. Q is the noise from the
        # velocity being allowed to change by ACCEL_SIGMA_MPS2 each second.
        q = ACCEL_SIGMA_MPS2 ** 2
        self.p += self.v * dt
        self.pp += (2 * dt * self.pv + dt * dt * self.vv
                    + q * dt ** 4 / 4)
        self.pv += dt * self.vv + q * dt ** 3 / 2
        self.vv += q * dt ** 2

    def update(self, measured, measured_var):
        """Blend in a position measurement with variance measured_var."""
        # Gain: how much of the surprise (measured - predicted) to believe.
        # Near 1 when the measurement is more certain than the prediction,
        # near 0 when it's less. The velocity gets a share through pv.
        gain_p = self.pp / (self.pp + measured_var)
        gain_v = self.pv / (self.pp + measured_var)
        surprise = measured - self.p
        self.p += gain_p * surprise
        self.v += gain_v * surprise
        # P = (I - K H) P with H = [1, 0]: a measurement only ever shrinks
        # the uncertainty.
        self.vv -= gain_v * self.pv
        self.pv *= 1 - gain_p
        self.pp *= 1 - gain_p


class BottleFilter:
    """Kalman filter for the target's 3D position and velocity.

    X, Y and Z are filtered separately: the model and the noise are the
    same on each axis, so three small filters give the same answer as one
    big one and are much easier to read.
    """

    def __init__(self):
        self.axes = None  # three AxisFilters, or None until the first update
        self.last_t = None  # time of the last measurement, in seconds
        self.updates = 0  # measurements used so far

    def reset(self):
        """Forget everything (a different bottle, or a long gap)."""
        self.axes = None
        self.last_t = None
        self.updates = 0

    def update(self, t, position, sigmas):
        """Use the raw position (X, Y, Z) measured at time t, in seconds.

        sigmas: the expected noise of each coordinate, in meters (see
        measurement_sigmas).
        """
        if self.axes is None:
            self.axes = [AxisFilter(p, s ** 2)
                         for p, s in zip(position, sigmas)]
        else:
            # The real time since the last measurement, so a slow frame or a
            # missed detection just means a longer prediction step.
            dt = max(t - self.last_t, 0.0)
            for axis, p, s in zip(self.axes, position, sigmas):
                axis.predict(dt)
                axis.update(p, s ** 2)
        self.last_t = t
        self.updates += 1

    @property
    def position(self):
        """Filtered (X, Y, Z) in meters, or None before the first update."""
        if self.axes is None:
            return None
        return tuple(axis.p for axis in self.axes)

    @property
    def velocity(self):
        """Filtered (vx, vy, vz) in m/s, or None while it's warming up."""
        if self.axes is None or self.updates < WARMUP_UPDATES:
            return None
        return tuple(axis.v for axis in self.axes)


def box_size(box):
    # The pixel length of the box side we measure distance with.
    x1, y1, x2, y2 = box[:4]
    return x2 - x1 if SIZE_AXIS == "width" else y2 - y1


def touches_edge(box, frame_width, frame_height):
    # A box touching an edge that cuts across the measured side (left or right
    # for the width, top or bottom for the height) is probably cut off, which
    # makes the size too small (and the distance too large).
    x1, y1, x2, y2 = box[:4]
    if SIZE_AXIS == "width":
        return x1 <= EDGE_MARGIN or x2 >= frame_width - EDGE_MARGIN
    return y1 <= EDGE_MARGIN or y2 >= frame_height - EDGE_MARGIN


def ema(old, new, alpha):
    # Exponential moving average: mix a fraction (alpha) of the new value
    # into the running value, so one noisy value can only move it a little.
    # With no running value yet, start from the new value.
    if old is None:
        return new
    return alpha * new + (1 - alpha) * old


def smooth_size(smoothed, new_size):
    # Smooth the measured box side with SMOOTHING_ALPHA.
    return ema(smoothed, new_size, SMOOTHING_ALPHA)


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


def measure_distance(target, frame_width, frame_height, focal_length,
                     smoothed_size):
    # Work out the distance for the target. target is the box as
    # (x1, y1, x2, y2, conf, track_id), or None if there's no target.
    # smoothed_size is the smoothed box side, or None if there isn't one (then
    # the raw size of target is used).
    # Returns (distance, reason). distance is in meters, or None. reason
    # explains a missing distance for the status line, or is None when
    # there's nothing to explain.
    if focal_length is None:
        return None, (f"Not calibrated: hold {TARGET_CLASS} at "
                      f"{KNOWN_DISTANCE_M:.2f} m, press c")
    if target is None:
        return None, None

    if touches_edge(target, frame_width, frame_height):
        return None, (f"{TARGET_CLASS.capitalize()} at edge: "
                      "distance unreliable")

    # Prefer the smoothed size; fall back to this frame's raw size.
    size = smoothed_size if smoothed_size is not None else box_size(target)

    # compute_distance divides by size, so a 0-pixel box would crash it.
    if size <= 0:
        return None, None

    return compute_distance(size, focal_length, TARGET_SIZE_M), None


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


def save_calibration(focal_length, avg_size, frame_size):
    # Save the result together with the values used to get it, so the file
    # explains itself later. frame_size is (width, height) in pixels: the
    # focal length in pixels is only valid at that resolution.
    data = {
        "focal_length_px": focal_length,
        "avg_box_size_px": avg_size,
        "size_axis": SIZE_AXIS,
        "known_distance_m": KNOWN_DISTANCE_M,
        "target_size_m": TARGET_SIZE_M,
        "frames_averaged": CALIBRATION_FRAMES,
        "frame_width": frame_size[0],
        "frame_height": frame_size[1],
        "calibrated_at": datetime.now().isoformat(timespec="seconds"),
    }
    # indent=2 puts each field on its own line so it's easy to read.
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(data, f, indent=2)


class Tracker:
    """Detects and tracks the target, and measures it in 3D."""

    def __init__(self, model_path=MODEL_PATH, calibration=None):
        """Load the model and the calibration.

        model_path: the YOLO model file
        calibration: (focal_length, frame_size) if the caller already loaded
            it with load_calibration(), or None to load it here

        Raises ValueError if the model has no TARGET_CLASS class. A model
        file that can't be loaded raises whatever YOLO raises.
        """
        # Load the model once (loading is slow).
        self.model = YOLO(str(model_path))

        # model.names maps each class ID to its name, e.g. {0: 'Drone-Interceptor'}.
        # Look up the ID for our target class instead of hardcoding the number.
        self.bottle_id = None
        for class_id, name in self.model.names.items():
            if name == TARGET_CLASS:
                self.bottle_id = class_id
                break

        if self.bottle_id is None:
            raise ValueError(f"the model has no class named '{TARGET_CLASS}'")

        # The saved focal length (None if we haven't calibrated yet) and the
        # frame size it was made with (None if the file doesn't say).
        if calibration is None:
            calibration = load_calibration()
        self.focal_length, self.calibration_size = calibration

        # Calibration state.
        self.calibrating = False  # True while we're collecting samples
        self.samples = []  # box sizes (px) collected so far

        # Tracking and velocity state.
        self.smoothed_size = None  # running EMA of the size; None until a target
        self.locked_id = None  # track ID of the bottle we're following
        self.history_id = None  # the track ID the filter belongs to
        self.last_seen = None  # when the target was last seen (seconds)
        # Filters the target's raw 3D positions into a position and velocity.
        self.filter = BottleFilter()

    def warm_up(self, frame):
        """Run the model once on a black frame the size of frame.

        YOLO's first run takes seconds (PyTorch sets itself up), so do it
        before the loop instead of freezing the first live frame. It uses the
        same settings as update(), and doesn't change what the tracker knows
        about the target.
        """
        self.model.track(np.zeros_like(frame), persist=True,
                         tracker=TRACKER_CONFIG, conf=CONF_THRESHOLD,
                         imgsz=IMGSZ, verbose=False)

    def reset(self):
        """Forget the target being followed, so the most confident one is
        picked again on the next frame (for when it's following the wrong
        object, or the video starts over)."""
        self.locked_id = None
        self.history_id = None
        self.last_seen = None
        self.smoothed_size = None
        self.filter.reset()

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

        # frame.shape is (height, width, channels) in pixels.
        frame_height, frame_width = frame.shape[:2]

        # Smooth the size only while there's exactly one target, fully in
        # view. Otherwise start over: with two targets the one we follow can
        # switch between them, and a cut-off box has a wrong size, so neither
        # should be mixed into the running value.
        if len(bottles) == 1 and not touches_edge(target, frame_width,
                                                  frame_height):
            self.smoothed_size = smooth_size(self.smoothed_size,
                                             box_size(target))
        else:
            self.smoothed_size = None

        # Start the filter over if the target is a different track, or if it
        # was missing for too long. A short miss is fine: the filter just
        # predicts across the bigger time gap.
        missing_too_long = (self.last_seen is not None
                            and frame_t - self.last_seen > LOST_TIMEOUT_S)
        if target is not None:
            if target[5] != self.history_id or missing_too_long:
                self.filter.reset()
                self.history_id = target[5]
            self.last_seen = frame_t
        elif missing_too_long:
            self.filter.reset()

        # Distance for the target.
        distance, reason = measure_distance(target, frame_width, frame_height,
                                            self.focal_length,
                                            self.smoothed_size)

        # Offset of the raw box center from the image center. The bearings
        # and the 3D position both use it, so they always agree.
        offsets = center_offsets(target, frame) if target is not None else None

        # Feed the filter this frame's RAW position. The filter does the
        # smoothing itself, so feeding it the smoothed distance would smooth
        # twice and add lag. Only tracked targets, and only frames with a
        # distance (skips "--" frames, e.g. target at the edge, without
        # resetting).
        filtered = False  # whether the filter got a measurement this frame
        if distance is not None and target[5] is not None:
            raw_size = box_size(target)
            if raw_size > 0:
                raw_distance = compute_distance(raw_size, self.focal_length,
                                                TARGET_SIZE_M)
                dx, dy = offsets
                raw_position = compute_3d_position(dx, dy, raw_distance,
                                                   self.focal_length)
                self.filter.update(frame_t, raw_position,
                                   measurement_sigmas(raw_distance,
                                                      self.focal_length))
                filtered = True

        # 3D position: the filtered one when the filter was updated this
        # frame. Otherwise (a target that isn't tracked yet) the raw box
        # center with the smoothed distance. None when there's no distance
        # (no target, not calibrated, or at the edge).
        position = None
        if filtered:
            position = self.filter.position
        elif target is not None and self.focal_length is not None \
                and distance is not None:
            position = compute_3d_position(offsets[0], offsets[1], distance,
                                           self.focal_length)

        # 3D velocity in m/s: None for the first few measurements.
        velocity = self.filter.velocity if filtered else None

        # While calibrating, collect one size per frame, but only from frames
        # we can trust. This comes after the measurements above, so a frame
        # that finishes the calibration is still measured with the old
        # focal length, and the new one applies from the next frame.
        measured_with = self.focal_length
        message = self._calibration_step(bottles, frame)

        return Track(bottles, target, distance, reason, offsets, position,
                     velocity, message, measured_with)

    def _calibration_step(self, bottles, frame):
        # Returns the message to show for this frame, or None when not
        # calibrating.
        if not self.calibrating:
            return None

        if len(bottles) == 0:
            status = f"no {TARGET_CLASS}"
        elif len(bottles) > 1:
            # Can't tell which box is the target at the known distance.
            status = f"{len(bottles)} {TARGET_CLASS}s"
        else:
            frame_height, frame_width = frame.shape[:2]
            if touches_edge(bottles[0], frame_width, frame_height):
                status = "at edge"
            else:
                size = box_size(bottles[0])
                self.samples.append(size)
                status = f"size {size}"

        # Enough samples: average them and compute the focal length.
        if len(self.samples) >= CALIBRATION_FRAMES:
            self.calibrating = False
            avg_size = sum(self.samples) / len(self.samples)
            self.focal_length = compute_focal_length(
                avg_size, KNOWN_DISTANCE_M, TARGET_SIZE_M)
            self.calibration_size = (frame.shape[1], frame.shape[0])
            save_calibration(self.focal_length, avg_size,
                             self.calibration_size)
            print(f"Calibrated: avg size = {avg_size:.1f} px, "
                  f"focal length = {self.focal_length:.1f} px")
            return f"Saved: f {self.focal_length:.1f} px"
        return f"Calibrating {len(self.samples)}/{CALIBRATION_FRAMES}: {status}"
