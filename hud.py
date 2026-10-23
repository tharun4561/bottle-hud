"""Heads-up display for the interceptor demo.

All the drawing lives here: the lock banner, corner brackets around the
target, the target and lead lines, the launcher crosshair, the side panel,
the status line and the debug overlay. main.py does the detection, tracking
and math, then hands the results to Hud.render() as plain values (numbers,
strings and boxes), so it doesn't matter which code produced them.

The camera frame is scaled up to DISPLAY_SIZE first and everything is drawn
at that size, so text and lines stay sharp in full screen. Sizes are
fractions of the display height, so the HUD keeps its proportions at any
DISPLAY_SIZE.
"""

import time  # high-precision clock for blinking and the draw timing

import cv2  # OpenCV: resizing and drawing

# Lock states, returned by LockIndicator.update() and passed to Hud.render().
SEARCHING = "SEARCHING"  # no tracked target
LOCKED = "LOCKED"  # following a tracked target
LOST = "LOST"  # the tracked target disappeared a moment ago

# A target that disappears stays LOCKED this long (seconds), so one missed
# detection doesn't make the banner flash. Then it shows as LOST until
# LOST_HOLD_S after it was last seen, and then the HUD goes back to SEARCHING.
LOST_GRACE_S = 0.2
LOST_HOLD_S = 2.0

DEBUG_DEFAULT = False  # whether the debug overlay starts switched on

# Display.
DISPLAY_SIZE = (1920, 1080)  # (width, height) to draw at; match the screen
# If scaling the frame up leaves a strip beside it at least this wide (as a
# fraction of the display width), the panel goes in the strip instead of
# covering the video. A 4:3 webcam on a 16:9 screen leaves 25%.
MIN_STRIP_FRACTION = 0.18
OVERLAY_PANEL_WIDTH = 0.24  # panel width when it has to cover the video
MARGIN = 0.02  # gap between HUD elements and the edges

# Text heights, as fractions of the display height (0.04 = 43 px at 1080).
FONT = cv2.FONT_HERSHEY_DUPLEX  # double-stroked, so bolder than SIMPLEX
BANNER_TEXT = 0.055  # lock state banner
VALUE_TEXT = 0.04  # panel values
STATUS_TEXT = 0.028  # status line
LABEL_TEXT = 0.022  # panel labels and the target's name
DEBUG_TEXT = 0.02  # debug overlay and box labels
OUTLINE_EXTRA = 3  # extra thickness of the black outline around text

# Line sizes, as fractions of the display height.
LINE_THICKNESS = 0.003  # target and lead lines (3 px at 1080)
BRACKET_THICKNESS = 0.005  # corner brackets around the target
CANDIDATE_THICKNESS = 0.002  # corner brackets around other detections
BRACKET_LENGTH = 0.25  # bracket arm length, as a fraction of the box's shorter side
MIN_BRACKET_LENGTH = 0.015
DOT_RADIUS = 0.006  # dot at the end of the target line
AIM_RADIUS = 0.014  # aim reticle at the lead point
CROSSHAIR_SIZE = 0.025  # length of each crosshair arm from the center
CROSSHAIR_BOTTOM_MARGIN = 0.08  # crosshair distance from the video's bottom edge

# Colors are (blue, green, red).
LOCKED_COLOR = (0, 230, 0)  # green
SEARCHING_COLOR = (255, 255, 0)  # cyan
LOST_COLOR = (0, 0, 255)  # red
STATE_COLORS = {SEARCHING: SEARCHING_COLOR, LOCKED: LOCKED_COLOR,
                LOST: LOST_COLOR}
CANDIDATE_COLOR = (170, 170, 170)  # gray: detections that aren't the target
TARGET_LINE_COLOR = (0, 255, 255)  # yellow
LEAD_COLOR = (255, 0, 255)  # magenta
CROSSHAIR_COLOR = (0, 0, 255)  # red
STATUS_COLOR = (0, 255, 255)  # yellow
LABEL_COLOR = (170, 170, 170)  # gray panel labels
VALUE_COLOR = (255, 255, 255)  # white panel values
MISSING_COLOR = (110, 110, 110)  # dim gray for "--" values
BLACK = (0, 0, 0)

# Backgrounds: opacity from 0 (invisible) to 1 (solid).
PANEL_ALPHA = 0.8
BANNER_ALPHA = 0.6
DEBUG_ALPHA = 0.65

BLINK_PERIOD_S = 0.5  # one on/off cycle of the LOST blink, in seconds
TIMING_SMOOTHING = 0.1  # EMA factor for the stage timings in the debug overlay


class LockIndicator:
    """Decides the lock state, and which box to show it on.

    A target with a track ID is LOCKED. When it disappears it stays LOCKED
    (at its last position) for LOST_GRACE_S, then shows as LOST until
    LOST_HOLD_S, then the state goes back to SEARCHING.
    """

    def __init__(self):
        self.last = None  # (box, track_id, conf) of the last locked target
        self.last_seen = None  # when it was last seen, in seconds

    def update(self, box, track_id, conf, t):
        """Return (state, box, track_id, conf) to pass to Hud.render().

        box: this frame's target as (x1, y1, x2, y2) in frame pixels, or None
        track_id: its track ID, or None if it isn't tracked (yet)
        conf: its detection confidence from 0 to 1, or None
        t: this frame's time in seconds (a video file's own time, so a
            recording behaves the same however fast it's processed)

        The returned box is the last locked one while the target is missing.
        """
        if box is not None and track_id is not None:
            self.last = (box, track_id, conf)
            self.last_seen = t
            return LOCKED, box, track_id, conf

        if self.last is not None:
            missing = t - self.last_seen
            last_box, last_id, last_conf = self.last
            if missing < LOST_GRACE_S:
                return LOCKED, last_box, last_id, last_conf
            if missing < LOST_HOLD_S:
                return LOST, last_box, last_id, None
            self.last = None

        # Nothing locked. An untracked detection is still shown, as a candidate.
        return SEARCHING, box, track_id, conf


def px(fraction):
    # A size given as a fraction of the display height, in whole pixels
    # (at least 1).
    return max(1, round(fraction * DISPLAY_SIZE[1]))


def text_style(fraction):
    # Font scale and stroke thickness for text whose capital letters are
    # this fraction of the display height tall. Thicker strokes for bigger
    # text keep it bold enough to read on a projector.
    (_, base_h), _ = cv2.getTextSize("0", FONT, 1.0, 1)
    height = fraction * DISPLAY_SIZE[1]
    return height / base_h, max(2, round(height / 12))


def cap_height(style):
    # Height of a capital letter (or digit) in this style, in pixels.
    scale, thickness = style
    (_, h), _ = cv2.getTextSize("0", FONT, scale, thickness)
    return h


def degree_radius(style):
    # Radius of a hand-drawn degree mark (OpenCV's fonts have no "°").
    return max(2, round(cap_height(style) * 0.16))


def degree_advance(style):
    # How far a degree mark moves the next character along, in pixels.
    return 2 * degree_radius(style) + style[1] + 2


def text_width(text, style):
    # Width of text in pixels, counting each "°" as a hand-drawn mark.
    scale, thickness = style
    parts = text.split("°")
    width = (len(parts) - 1) * degree_advance(style)
    for part in parts:
        (w, _), _ = cv2.getTextSize(part, FONT, scale, thickness)
        width += w
    return width


def fit_style(text, style, max_width):
    # The style, shrunk if needed so the text fits in max_width pixels.
    width = text_width(text, style)
    if width <= max_width:
        return style
    factor = max_width / width
    scale, thickness = style
    return scale * factor, max(1, round(thickness * factor))


def put_text(img, text, org, style, color, outline=False):
    # Draw text with its bottom-left corner at org, drawing each "°" as a
    # small circle. outline=True first draws the same text thicker in
    # black, so it stays readable on a bright background.
    scale, thickness = style
    r = degree_radius(style)
    top = org[1] - cap_height(style)
    passes = [(BLACK, thickness + OUTLINE_EXTRA)] if outline else []
    passes.append((color, thickness))
    for pass_color, pass_thickness in passes:
        x = org[0]
        for i, part in enumerate(text.split("°")):
            if i > 0:
                center = (x + r + thickness // 2 + 1, top + r)
                cv2.circle(img, center, r, pass_color,
                           max(1, pass_thickness // 2), cv2.LINE_AA)
                x += degree_advance(style)
            cv2.putText(img, part, (x, org[1]), FONT, scale, pass_color,
                        pass_thickness, cv2.LINE_AA)
            # Advance by the normal width, so the outline lines up.
            (w, _), _ = cv2.getTextSize(part, FONT, scale, thickness)
            x += w


def blend_rect(img, rect, color, alpha):
    # Fill a rectangle (x1, y1, x2, y2) with a see-through color. Only the
    # part inside the image is changed.
    height, width = img.shape[:2]
    x1, y1 = max(0, rect[0]), max(0, rect[1])
    x2, y2 = min(width, rect[2]), min(height, rect[3])
    if x2 <= x1 or y2 <= y1:
        return
    # img[y1:y2, x1:x2] is a view into the image, so writing into it
    # changes the image itself.
    region = img[y1:y2, x1:x2]
    overlay = region.copy()
    overlay[:] = color
    cv2.addWeighted(overlay, alpha, region, 1 - alpha, 0, dst=region)


def rects_overlap(a, b):
    # True if two rectangles (x1, y1, x2, y2) overlap at all.
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def draw_brackets(img, rect, color, thickness):
    # Corner brackets around a rectangle (x1, y1, x2, y2): an L at each
    # corner instead of a full box, so the object itself stays visible.
    x1, y1, x2, y2 = rect
    arm = max(px(MIN_BRACKET_LENGTH),
              round(BRACKET_LENGTH * min(x2 - x1, y2 - y1)))
    # Each corner, and which way its two arms point (+1 or -1 in x and y).
    for x, y, sx, sy in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                         (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (x, y), (x + sx * arm, y), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x, y), (x, y + sy * arm), color, thickness, cv2.LINE_AA)


def side_text(degrees, positive, negative):
    # One bearing for the audience, like "12° R": whole degrees and a side
    # instead of a sign.
    d = round(degrees)
    if d == 0:
        return "0°"
    return f"{abs(d)}° {positive if d > 0 else negative}"


def bearing_text(bearing):
    # Both bearings, like "12° R  3° U", or "--".
    if bearing is None:
        return "--"
    return f"{side_text(bearing[0], 'R', 'L')}  {side_text(bearing[1], 'U', 'D')}"


def box_label(track_id, conf):
    # Debug label for a box, like "#7 0.85" (track ID and confidence).
    parts = []
    if track_id is not None:
        parts.append(f"#{track_id}")
    if conf is not None:
        parts.append(f"{conf:.2f}")
    return " ".join(parts) or "?"


class Hud:
    """Draws the HUD. Holds the settings that last between frames (the debug
    toggle and the smoothed stage timings)."""

    def __init__(self, target_name):
        # target_name: the class being tracked, e.g. "bottle", shown on the
        # banner and next to the target.
        self.target_name = target_name.upper()
        self.show_debug = DEBUG_DEFAULT
        self.timings = {}  # smoothed stage timings in seconds, by name
        self.draw_time = None  # how long the previous render() took

        self.banner_style = text_style(BANNER_TEXT)
        self.value_style = text_style(VALUE_TEXT)
        self.status_style = text_style(STATUS_TEXT)
        self.label_style = text_style(LABEL_TEXT)
        self.debug_style = text_style(DEBUG_TEXT)

        # Set by layout() for each frame.
        self.scale = 1.0  # display pixels per frame pixel
        self.video = (0, 0, 0, 0)  # the video's rectangle on the display
        self.strip = None  # the empty strip beside the video, or None

    def toggle_debug(self):
        self.show_debug = not self.show_debug

    def render(self, frame, *, state, box, track_id, conf, others, lead,
               distance_text, bearing, intercept_text, fps, mode, status,
               debug_rows, timings):
        """Return a new image, DISPLAY_SIZE big, with the frame and the HUD.

        frame is not changed. Positions and boxes are in frame pixels.

        state: SEARCHING, LOCKED or LOST (from LockIndicator.update())
        box: the target as (x1, y1, x2, y2), or None
        track_id, conf: the target's track ID and confidence (0 to 1), or None
        others: the other detections, as a list of (box, track_id, conf)
        lead: the aim point on screen as (x, y), or None
        distance_text, intercept_text: panel values, already formatted
        bearing: (horizontal, vertical) in degrees (right and up positive),
            or None
        fps: frames per second, or None
        mode: the MODE value, e.g. "LIVE"
        status: the status line, or None
        debug_rows: (label, value) text pairs for the debug overlay
        timings: (name, seconds) pairs of stage timings for the debug overlay
        """
        start = time.perf_counter()
        canvas = self.layout(frame)

        # LOST blinks: on for the first half of each period, off for the second.
        blink_on = time.perf_counter() % BLINK_PERIOD_S < BLINK_PERIOD_S / 2

        # The panel first: when it has to cover the video, the target
        # graphics are drawn over it, so the target never disappears.
        conf_text = f"{conf * 100:.0f}%" if conf is not None else "--"
        fps_text = f"{fps:.0f}" if fps else "--"
        avoid = [self.draw_panel(canvas, [
            ("DISTANCE", distance_text),
            ("BEARING", bearing_text(bearing)),
            ("INTERCEPT IN", intercept_text),
            ("CONFIDENCE", conf_text),
            ("FPS", fps_text),
            ("MODE", mode),
        ])]

        for other_box, _, _ in others:
            draw_brackets(canvas, self.to_rect(other_box), CANDIDATE_COLOR,
                          px(CANDIDATE_THICKNESS))

        origin = self.origin()
        lead_on_screen = True
        if box is not None:
            target_rect = self.to_rect(box)
            if state == SEARCHING:
                # A detection that isn't tracked yet: a candidate, not a lock.
                draw_brackets(canvas, target_rect, SEARCHING_COLOR,
                              px(CANDIDATE_THICKNESS))
            else:
                if state == LOCKED:
                    self.draw_target_line(canvas, origin, target_rect)
                if state == LOCKED or blink_on:
                    color = STATE_COLORS[state]
                    draw_brackets(canvas, target_rect, color,
                                  px(BRACKET_THICKNESS))
                    avoid.append(self.draw_target_name(canvas, target_rect,
                                                       track_id, color))

        # The lead line goes on top of the target line, so when the target
        # is still and they overlap, the lead line shows.
        if lead is not None:
            lead_on_screen = self.draw_lead_line(canvas, origin, lead)

        # The crosshair on top of both lines.
        self.draw_crosshair(canvas, origin)

        banner_rect = self.draw_banner(canvas, state, track_id, blink_on)
        avoid.append(banner_rect)
        if status:
            avoid.append(self.draw_status(canvas, status, banner_rect[3]))

        # Smooth the stage timings even while the overlay is hidden, so
        # they're settled when it's switched on.
        if self.draw_time is not None:
            timings = list(timings) + [("hud draw", self.draw_time)]
        for name, seconds in timings:
            old = self.timings.get(name)
            self.timings[name] = (seconds if old is None else
                                  TIMING_SMOOTHING * seconds
                                  + (1 - TIMING_SMOOTHING) * old)

        if self.show_debug:
            rows = list(debug_rows)
            rows.append(("LEAD", "on screen" if lead_on_screen else "off screen"))
            for name, seconds in self.timings.items():
                rows.append((name, f"{seconds * 1000:.1f} ms"))
            rows.append(("keys", "c calibrate  +/- speed  d debug  "
                                 "f full screen  q quit"))
            avoid.append(self.draw_debug(canvas, rows))
            # Box labels last, so nothing covers them.
            if box is not None:
                self.draw_box_label(canvas, self.to_rect(box),
                                    box_label(track_id, conf), avoid)
            for other_box, other_id, other_conf in others:
                self.draw_box_label(canvas, self.to_rect(other_box),
                                    box_label(other_id, other_conf), avoid)

        self.draw_time = time.perf_counter() - start
        return canvas

    def layout(self, frame):
        # Scale the frame up to fit DISPLAY_SIZE and return it on a black
        # canvas of that size. Records where the video landed.
        disp_w, disp_h = DISPLAY_SIZE
        frame_h, frame_w = frame.shape[:2]
        self.scale = min(disp_w / frame_w, disp_h / frame_h)
        video_w = round(frame_w * self.scale)
        video_h = round(frame_h * self.scale)

        # With a wide enough strip left over, put the video on the left and
        # keep the strip for the panel; otherwise center the video.
        if disp_w - video_w >= MIN_STRIP_FRACTION * disp_w:
            left = 0
            self.strip = (video_w, 0, disp_w, disp_h)
        else:
            left = (disp_w - video_w) // 2
            self.strip = None
        top = (disp_h - video_h) // 2
        self.video = (left, top, left + video_w, top + video_h)

        # INTER_LINEAR is a smooth, fast upscale; the HUD is drawn
        # afterwards, so its text and lines aren't blurred by it.
        scaled = cv2.resize(frame, (video_w, video_h),
                            interpolation=cv2.INTER_LINEAR)
        return cv2.copyMakeBorder(scaled, top, disp_h - video_h - top, left,
                                  disp_w - video_w - left,
                                  cv2.BORDER_CONSTANT, value=BLACK)

    def to_display(self, x, y):
        # A point in frame pixels, in display pixels (whole numbers).
        return (round(self.video[0] + x * self.scale),
                round(self.video[1] + y * self.scale))

    def to_rect(self, box):
        # A box (x1, y1, x2, y2) in frame pixels, in display pixels.
        return self.to_display(box[0], box[1]) + self.to_display(box[2], box[3])

    def origin(self):
        # The launcher crosshair: bottom center of the video, where both
        # lines start.
        x1, _, x2, y2 = self.video
        return ((x1 + x2) // 2, y2 - px(CROSSHAIR_BOTTOM_MARGIN))

    def draw_crosshair(self, img, origin):
        x, y = origin
        size = px(CROSSHAIR_SIZE)
        thickness = px(LINE_THICKNESS)
        cv2.line(img, (x - size, y), (x + size, y), CROSSHAIR_COLOR,
                 thickness, cv2.LINE_AA)
        cv2.line(img, (x, y - size), (x, y + size), CROSSHAIR_COLOR,
                 thickness, cv2.LINE_AA)

    def draw_target_line(self, img, origin, rect):
        # Line from the crosshair to the target's center, plus a dot there.
        center = ((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2)
        cv2.line(img, origin, center, TARGET_LINE_COLOR, px(LINE_THICKNESS),
                 cv2.LINE_AA)
        cv2.circle(img, center, px(DOT_RADIUS), TARGET_LINE_COLOR, -1,
                   cv2.LINE_AA)

    def draw_lead_line(self, img, origin, lead):
        # Line from the crosshair to the aim point, with an aim reticle
        # there. Returns True if the point is on the video, False if not.
        end = self.to_display(lead[0], lead[1])
        x1, y1, x2, y2 = self.video
        thickness = px(LINE_THICKNESS)

        if x1 <= end[0] < x2 and y1 <= end[1] < y2:
            cv2.line(img, origin, end, LEAD_COLOR, thickness, cv2.LINE_AA)
            # Reticle: a circle with a cross through it reads as "aim here".
            r = px(AIM_RADIUS)
            arm = round(r * 1.6)
            cv2.circle(img, end, r, LEAD_COLOR, thickness, cv2.LINE_AA)
            cv2.line(img, (end[0] - arm, end[1]), (end[0] + arm, end[1]),
                     LEAD_COLOR, max(1, thickness - 1), cv2.LINE_AA)
            cv2.line(img, (end[0], end[1] - arm), (end[0], end[1] + arm),
                     LEAD_COLOR, max(1, thickness - 1), cv2.LINE_AA)
            return True

        # Off the video: cut the line where it leaves it, keeping its
        # direction. clipLine returns whether any part is inside the
        # rectangle (x, y, width, height), and the two ends of that part.
        inside, start, edge = cv2.clipLine((x1, y1, x2 - x1, y2 - y1),
                                           origin, end)
        if inside:
            cv2.line(img, start, edge, LEAD_COLOR, thickness, cv2.LINE_AA)
            # A hollow circle at the edge means "the point is beyond here".
            cv2.circle(img, edge, px(DOT_RADIUS) + 3, LEAD_COLOR, thickness,
                       cv2.LINE_AA)
        return False

    def draw_target_name(self, img, rect, track_id, color):
        # The target's name, like "BOTTLE #7", above its brackets (or below
        # them if there's no room above). Returns the text's rectangle.
        text = self.target_name
        if track_id is not None:
            text += f" #{track_id}"
        style = self.label_style
        gap = px(0.01)
        y = rect[1] - gap
        if y - cap_height(style) < self.video[1]:
            y = rect[3] + gap + cap_height(style)
        width = text_width(text, style)
        x = min(max(rect[0], self.video[0]), self.video[2] - width)
        put_text(img, text, (x, y), style, color, outline=True)
        return (x, y - cap_height(style) - OUTLINE_EXTRA, x + width,
                y + OUTLINE_EXTRA)

    def draw_banner(self, img, state, track_id, blink_on):
        # The lock state in big letters, top center of the video, with a
        # colored lamp before it and a border in the same color. Returns the
        # banner's rectangle.
        text = state
        if state != SEARCHING:
            text += f"  {self.target_name}"
            if track_id is not None:
                text += f" #{track_id}"
        color = STATE_COLORS[state]

        margin = px(MARGIN)
        max_w = (self.video[2] - self.video[0]) - 4 * margin
        style = fit_style(text, self.banner_style, max_w)
        text_h = cap_height(style)
        lamp_r = text_h // 3
        pad = text_h // 2
        width = 2 * lamp_r + pad + text_width(text, style) + 2 * pad
        x1 = (self.video[0] + self.video[2] - width) // 2
        y1 = self.video[1] + margin
        rect = (x1, y1, x1 + width, y1 + text_h + 2 * pad)

        blend_rect(img, rect, BLACK, BANNER_ALPHA)
        cv2.rectangle(img, rect[:2], rect[2:], color, px(0.003), cv2.LINE_AA)
        lamp = (x1 + pad + lamp_r, y1 + pad + text_h // 2)
        # The lamp blinks when LOST; the word itself stays steady to read.
        if state != LOST or blink_on:
            cv2.circle(img, lamp, lamp_r, color, -1, cv2.LINE_AA)
        put_text(img, text, (x1 + pad + 2 * lamp_r + pad, y1 + pad + text_h),
                 style, color)
        return rect

    def draw_status(self, img, status, below):
        # The status line, centered under the banner (whose bottom is at y =
        # below). Returns its rectangle.
        margin = px(MARGIN)
        max_w = (self.video[2] - self.video[0]) - 4 * margin
        style = fit_style(status, self.status_style, max_w)
        text_h = cap_height(style)
        pad = text_h // 2
        width = text_width(status, style) + 2 * pad
        x1 = (self.video[0] + self.video[2] - width) // 2
        y1 = below + margin // 2
        rect = (x1, y1, x1 + width, y1 + text_h + 2 * pad)
        blend_rect(img, rect, BLACK, BANNER_ALPHA)
        put_text(img, status, (x1 + pad, y1 + pad + text_h), style,
                 STATUS_COLOR)
        return rect

    def draw_panel(self, img, rows):
        # The side panel: a small gray label above a big white value, for
        # each (label, value) row. Goes in the strip beside the video if
        # there is one, otherwise top right on the video over a dark
        # background. Returns the rectangle it covers on the video (empty in
        # the strip, since there's nothing to avoid there).
        margin = px(MARGIN)
        label_h = cap_height(self.label_style)
        value_h = cap_height(self.value_style)
        label_gap = label_h // 2  # between a label and its value
        row_gap = label_h  # between rows
        row_h = label_h + label_gap + value_h + row_gap
        height = len(rows) * row_h - row_gap + 2 * margin

        if self.strip is not None:
            x1, y1, x2, _ = self.strip
            x1 += margin  # the strip is black already; just keep off the video
            covered = (0, 0, 0, 0)
        else:
            x2 = self.video[2] - margin
            x1 = x2 - round(OVERLAY_PANEL_WIDTH * DISPLAY_SIZE[0])
            y1 = self.video[1] + margin
            covered = (x1, y1, x2, y1 + height)
            blend_rect(img, covered, BLACK, PANEL_ALPHA)

        text_x = x1 + margin
        max_w = x2 - margin - text_x
        y = y1 + margin
        for label, value in rows:
            put_text(img, label, (text_x, y + label_h), self.label_style,
                     LABEL_COLOR)
            y += label_h + label_gap
            style = fit_style(value, self.value_style, max_w)
            color = MISSING_COLOR if value == "--" else VALUE_COLOR
            put_text(img, value, (text_x, y + value_h), style, color)
            y += value_h + row_gap
        return covered

    def draw_debug(self, img, rows):
        # The debug overlay: small (label, value) rows, bottom left of the
        # video, over a dark background. Returns its rectangle.
        style = self.debug_style
        text_h = cap_height(style)
        line_h = round(text_h * 1.6)
        pad = text_h
        value_x = max(text_width(label, style) for label, _ in rows) + pad
        width = value_x + max(text_width(value, style) for _, value in rows)
        width += 2 * pad
        height = len(rows) * line_h + 2 * pad - (line_h - text_h)

        margin = px(MARGIN)
        x1 = self.video[0] + margin
        y2 = self.video[3] - margin
        rect = (x1, y2 - height, x1 + width, y2)
        blend_rect(img, rect, BLACK, DEBUG_ALPHA)

        y = rect[1] + pad + text_h
        for label, value in rows:
            put_text(img, label, (x1 + pad, y), style, LABEL_COLOR)
            put_text(img, value, (x1 + pad + value_x, y), style, VALUE_COLOR)
            y += line_h
        return rect

    def draw_box_label(self, img, rect, label, avoid):
        # A debug label like "#7 0.85" for a box (display pixels), on a gray
        # background. Placed above the box, else below it, else inside its
        # top: the first spot that stays on the video and covers none of
        # the avoid rectangles wins (inside the top if none does).
        style = self.debug_style
        text_h = cap_height(style)
        label_w = text_width(label, style) + 8  # 4 px of padding on each side
        label_h = text_h + 10
        vx1, vy1, vx2, vy2 = self.video
        x1, y1, x2, y2 = rect
        left = max(vx1, min(x1, vx2 - label_w))

        top = max(vy1, min(y1, vy2 - label_h))
        for candidate in (y1 - label_h, y2, y1):
            spot = (left, candidate, left + label_w, candidate + label_h)
            if (candidate >= vy1 and candidate + label_h <= vy2
                    and not any(rects_overlap(spot, a) for a in avoid)):
                top = candidate
                break

        cv2.rectangle(img, (left, top), (left + label_w, top + label_h),
                      CANDIDATE_COLOR, -1)
        put_text(img, label, (left + 4, top + 5 + text_h), style, BLACK)
