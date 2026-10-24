"""Windows screen helpers for the demo.

Things OpenCV can't do by itself: list the monitors, put the window full
screen on a chosen monitor, keep the laptop from sleeping, and tell how long
ago the program started. They talk to Windows directly through ctypes
(part of Python), so nothing extra needs installing. On other systems they
fall back to doing nothing (or to OpenCV's own full screen).
"""

import ctypes  # calling Windows functions directly
import sys  # checking which operating system we're on
import time  # the current time, for the startup measurement
from ctypes import wintypes  # Windows data types (RECT, DWORD, ...)

import cv2  # OpenCV: the fallback full screen on other systems

IS_WINDOWS = sys.platform == "win32"

# Windows constants, from the Windows API documentation.
MONITORINFOF_PRIMARY = 1  # flag set on the primary monitor
MONITOR_DEFAULTTONEAREST = 2  # "the monitor closest to this window"
GWL_STYLE = -16  # which window setting Get/SetWindowLong reads: the style
WS_POPUP = 0x80000000  # a window with no border or title bar
WS_VISIBLE = 0x10000000
HWND_TOP = 0
SWP_FRAMECHANGED = 0x0020  # redraw the frame after the style changed
SWP_SHOWWINDOW = 0x0040
ES_CONTINUOUS = 0x80000000  # keep this setting until it's changed again
ES_SYSTEM_REQUIRED = 0x00000001  # don't sleep
ES_DISPLAY_REQUIRED = 0x00000002  # don't turn the screen off
# FILETIME counts 100-nanosecond steps since 1601; Unix time counts seconds
# since 1970. This many steps lie between the two starting points.
FILETIME_UNIX_OFFSET = 116444736000000000

# Saved (style, rectangle) of windows we put in full screen, by window
# handle, so full screen can be undone.
_saved_windows = {}

# Tell ctypes the argument and result types of the Windows functions we
# call, so window handles (64-bit on 64-bit Windows) are passed correctly.
if IS_WINDOWS:
    _user32 = ctypes.windll.user32
    _user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
    _user32.FindWindowW.restype = wintypes.HWND
    _user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
    _user32.GetWindowLongW.restype = ctypes.c_long
    _user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int,
                                       ctypes.c_long]
    _user32.GetWindowRect.argtypes = [wintypes.HWND,
                                      ctypes.POINTER(wintypes.RECT)]
    _user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND,
                                     ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int, wintypes.UINT]
    _user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
    _user32.MonitorFromWindow.restype = wintypes.HANDLE
    _user32.GetMonitorInfoW.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
    _kernel32 = ctypes.windll.kernel32
    _kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    _kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [
        ctypes.POINTER(wintypes.FILETIME)] * 4


class MONITORINFOEXW(ctypes.Structure):
    # Windows' description of one monitor (see GetMonitorInfoW).
    _fields_ = [("cbSize", wintypes.DWORD),
                ("rcMonitor", wintypes.RECT),  # its area on the desktop
                ("rcWork", wintypes.RECT),  # the same minus the taskbar
                ("dwFlags", wintypes.DWORD),
                ("szDevice", wintypes.WCHAR * 32)]


def make_dpi_aware():
    # Tell Windows we handle display scaling (125%, 150%) ourselves. Without
    # this, Windows reports a smaller screen and stretches our window, which
    # blurs the HUD. Must run before any window is created.
    if not IS_WINDOWS:
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per monitor
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()  # older Windows
        except (AttributeError, OSError):
            pass


def monitor_info(handle):
    # (x, y, width, height, primary) of one monitor, from its handle.
    info = MONITORINFOEXW()
    info.cbSize = ctypes.sizeof(MONITORINFOEXW)
    ctypes.windll.user32.GetMonitorInfoW(handle, ctypes.byref(info))
    r = info.rcMonitor
    primary = bool(info.dwFlags & MONITORINFOF_PRIMARY)
    return (r.left, r.top, r.right - r.left, r.bottom - r.top, primary)


def list_monitors():
    # Every monitor as (x, y, width, height, primary), in pixels on the
    # desktop. The primary monitor comes first, then the others from left
    # to right, so "screen 1" is always the main one. Empty if unknown.
    if not IS_WINDOWS:
        return []
    found = []

    # EnumDisplayMonitors calls this function once per monitor.
    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HANDLE, wintypes.HDC,
                        ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)
    def callback(handle, hdc, rect, data):
        found.append(monitor_info(handle))
        return True  # keep going

    ctypes.windll.user32.EnumDisplayMonitors(None, None, callback, 0)
    # not primary sorts False (0) first, so the primary monitor leads.
    return sorted(found, key=lambda m: (not m[4], m[0], m[1]))


def find_window(title):
    # Windows' handle for the OpenCV window with this title (0 if none).
    return ctypes.windll.user32.FindWindowW(None, title) or 0


def is_fullscreen(title):
    if not IS_WINDOWS:
        return cv2.getWindowProperty(title, cv2.WND_PROP_FULLSCREEN) == 1
    return find_window(title) in _saved_windows


def set_fullscreen(title, on, monitor=None):
    """Put the window full screen (on=True) or back to normal (on=False).

    monitor: (x, y, width, height, primary) to fill, or None for the monitor
    the window is on now (so dragging it to the projector and pressing f
    works). Returns the (width, height) it now fills in full screen, or None.

    OpenCV's own full screen isn't used on Windows because it may always
    pick the main monitor; this removes the window's border and sizes it to
    exactly the chosen monitor.
    """
    if not IS_WINDOWS:
        mode = cv2.WINDOW_FULLSCREEN if on else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, mode)
        return None

    user32 = ctypes.windll.user32
    hwnd = find_window(title)
    if not hwnd:
        return None

    if not on:
        if hwnd in _saved_windows:
            style, r = _saved_windows.pop(hwnd)
            user32.SetWindowLongW(hwnd, GWL_STYLE, style)
            user32.SetWindowPos(hwnd, HWND_TOP, r.left, r.top,
                                r.right - r.left, r.bottom - r.top,
                                SWP_FRAMECHANGED | SWP_SHOWWINDOW)
        return None

    # Remember the normal style and position the first time only.
    if hwnd not in _saved_windows:
        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        _saved_windows[hwnd] = (user32.GetWindowLongW(hwnd, GWL_STYLE), rect)

    if monitor is None:
        handle = user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
        monitor = monitor_info(handle)
    x, y, width, height, _ = monitor

    # WS_POPUP is above the range of a signed 32-bit number, so convert it
    # to the negative number with the same bits.
    style = WS_POPUP | WS_VISIBLE
    user32.SetWindowLongW(hwnd, GWL_STYLE, style - (1 << 32))
    user32.SetWindowPos(hwnd, HWND_TOP, x, y, width, height,
                        SWP_FRAMECHANGED | SWP_SHOWWINDOW)
    return width, height


def keep_awake(on):
    # While on, stop Windows from sleeping or turning the screen off (a
    # demo with nobody touching the laptop looks idle to Windows).
    if not IS_WINDOWS:
        return
    flags = ES_CONTINUOUS
    if on:
        flags |= ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
    ctypes.windll.kernel32.SetThreadExecutionState(flags)


def seconds_since_launch():
    # How long ago this process started, in seconds, including Python's own
    # startup and the imports. None if it can't be found out.
    if not IS_WINDOWS:
        return None
    kernel32 = ctypes.windll.kernel32
    created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
    ok = kernel32.GetProcessTimes(kernel32.GetCurrentProcess(),
                                  ctypes.byref(created), ctypes.byref(exited),
                                  ctypes.byref(kernel), ctypes.byref(user))
    if not ok:
        return None
    steps = (created.dwHighDateTime << 32) | created.dwLowDateTime
    started = (steps - FILETIME_UNIX_OFFSET) / 10_000_000
    return time.time() - started
