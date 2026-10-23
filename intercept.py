"""Intercept math: when and where an interceptor can meet the bottle.

Pure math with no camera, OpenCV or drawing code, so it can be tested with
plain numbers. The interceptor starts at the camera (the origin) and flies in
a straight line at a constant speed; the bottle moves in a straight line at a
constant velocity. Positions are (X, Y, Z) in meters (X right, Y up, Z away
from the camera) and velocities are in meters per second.
"""

import math  # math functions for the bearing formula and the intercept
from typing import NamedTuple, Optional

DEFAULT_SPEED_MPS = 5.0  # starting interceptor speed, in meters per second
SPEED_STEP_MPS = 0.5  # how much each + or - press changes the speed


class Intercept(NamedTuple):
    """Result of solve().

    time_s: seconds until the interceptor reaches the bottle, or None
    aim: (X, Y, Z) in meters where the bottle will be at that time, or None
    reason: why there is no solution (shown on the status line), or None when
        there is one or when there is nothing to explain
    """
    time_s: Optional[float]
    aim: Optional[tuple]
    reason: Optional[str]


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


def bearing_angles(offsets_px, focal_length_px):
    """Return (horizontal, vertical) bearings in degrees for a pixel offset pair.

    offsets_px: (dx, dy) of the bottle's center from the image center, in
        pixels (right = positive, up = positive)
    focal_length_px: the calibrated focal length, in pixels
    """
    dx, dy = offsets_px
    return (compute_bearing(dx, focal_length_px),
            compute_bearing(dy, focal_length_px))


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


def solve(position, velocity, speed_mps):
    """Work out the intercept time and 3D aim point.

    position: the bottle's (X, Y, Z) in meters, or None when it isn't known
        (no bottle, not calibrated, or at the edge; the tracker already
        explains those)
    velocity: the bottle's (vx, vy, vz) in m/s, or None while it is still
        being measured
    speed_mps: interceptor speed in meters per second

    Returns an Intercept.
    """
    if position is None:
        return Intercept(None, None, None)
    # With a speed of 0 the interceptor never arrives, so there's no
    # intercept; say why.
    if speed_mps <= 0:
        return Intercept(None, None, "Speed is 0: press + to speed up")
    if velocity is None:
        return Intercept(None, None, "Measuring velocity...")

    t = solve_intercept_time(position, velocity, speed_mps)
    if t is None:
        # With one positive root whenever the interceptor is faster, None
        # only happens when the bottle is at least as fast as it.
        return Intercept(None, None, "No intercept: target too fast")

    # Aim point: where the bottle will be after t seconds, P + V*t.
    aim = tuple(p + v * t for p, v in zip(position, velocity))
    return Intercept(t, aim, None)
