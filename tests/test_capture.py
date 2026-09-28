"""Capture and window tests. These talk to the real desktop: a real mss grab,
a real Win32 window created here, moved and closed for real.

    python test_capture.py

No mocks on the interesting paths. A mock that agrees with the code proves
nothing about coordinates, which is the only thing this layer gets wrong, so
the window below is a real top-level Win32 window with a real client area, a
real message loop, and a real WM_PAINT that fills it with a pattern the test
can recognise. It appears on screen (top-most, tool window) for the length of
the run and closes itself at the end.

Two things this pins down that mocks would not:

  * `origin` really is the client rect's top-left, proven by a magenta marker
    the window paints at client pixel (0, 0) - so a crop that lands on the
    marker landed in the right place to the pixel.
  * `crop()` is inherited unchanged and the origin is subtracted inside it, so
    a region stored in absolute screen coordinates works against a window
    frame with no conversion. Checked by comparing a window crop against a
    whole-monitor crop of the same absolute rect, pixel for pixel.
"""

from __future__ import annotations
import _boot  # noqa: F401,E402  - repo root on sys.path, cwd at root

import ctypes
import os
import queue
import sys
import threading
import time
import warnings
from ctypes import wintypes

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Qt is not used here, but the warning filter costs nothing and this file is
# often run next to the GUI tests.
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.*=false")
for _stream in (sys.stdout, sys.stderr):  # titles carry emoji; never die on them
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

import pokervision.regions as rc  # noqa: E402
from pokervision import windows as win  # noqa: E402  (also sets DPI awareness)
from pokervision.capture import Capture, WindowCapture  # noqa: E402
from pokervision.config import Config  # noqa: E402

FAILURES: list[str] = []


def check(name, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}: got {got}"
          + ("" if ok else f", want {want}"))
    if not ok:
        FAILURES.append(name)
    return ok


def check_pixels(name, got, want):
    """Same as check, for frames: prints the worst pixel that differs."""
    if not isinstance(got, np.ndarray) or not isinstance(want, np.ndarray):
        return check(name, got, want)
    if got.shape != want.shape:
        ok = False
        detail = f"got shape {got.shape}, want {want.shape}"
    else:
        diff = np.abs(got.astype(int) - want.astype(int))
        ok = bool((diff == 0).all())
        detail = f"{int((diff.max(axis=2) > 0).sum())} px differ, " \
                 f"max channel diff {int(diff.max())}"
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}"
          + ("" if ok else f": {detail}"))
    if not ok:
        FAILURES.append(name)
    return ok


def check_true(name, got, why=""):
    ok = bool(got)
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}"
          + ("" if ok else f": {why or 'expected true, got ' + repr(got)}"))
    if not ok:
        FAILURES.append(name)
    return ok


def check_raises(name, exc_type, fn, *args, **kwargs):
    try:
        result = fn(*args, **kwargs)
    except exc_type as exc:
        print(f"  ok    {name}: raised {type(exc).__name__}: {exc}")
        return True
    except Exception as exc:  # noqa: BLE001 - the point is the exact type
        print(f"  FAIL  {name}: raised {type(exc).__name__} ({exc}), "
              f"want {exc_type.__name__}")
        FAILURES.append(name)
        return False
    print(f"  FAIL  {name}: returned {result!r}, want {exc_type.__name__}")
    FAILURES.append(name)
    return False


# --------------------------------------------------------------------------
# a throwaway Win32 window, on its own thread because a top-level window
# belongs to the thread that created it and needs that thread to pump messages.
# --------------------------------------------------------------------------

WS_OVERLAPPEDWINDOW = 0x00CF0000
WS_EX_TOPMOST = 0x00000008
WS_EX_TOOLWINDOW = 0x00000080
SWP_SHOWWINDOW = 0x0040
SW_MINIMIZE = 2
SW_RESTORE = 9
WM_CLOSE = 0x0010
WM_DESTROY = 0x0002
WM_PAINT = 0x000F
WM_ERASEBKGND = 0x0014
WM_GETMINMAXINFO = 0x0024
PS_SOLID = 0
HWND_TOPMOST = ctypes.c_void_p(-1)

# Pattern colours. Three coprime periods (7, 11) plus 5x5 blocks on 13x17, so a
# crop that is off by even a whole period shows up as different pixels.
BG = 0x1C1814      # BGR of (20, 24, 28)
GRID_X = 0x3AFAAA  # BGR of (250, 250, 60)
GRID_Y = 0xFA3A3A  # BGR of (60, 250, 250)
BLOCK = 0x3A3AFA  # BGR of (250, 60, 60)
MARKER = 0xFF00FF  # magenta, painted at client (0, 0)
MARKER_SIZE = 12
MARK_RGB = np.array([MARKER & 0xFF, (MARKER >> 8) & 0xFF,
                     (MARKER >> 16) & 0xFF], np.uint8)


class _RECT(ctypes.Structure):
    _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG),
                ("right", wintypes.LONG), ("bottom", wintypes.LONG)]


class _POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class _MINMAXINFO(ctypes.Structure):
    _fields_ = [("ptReserved", _POINT), ("ptMaxSize", _POINT),
                ("ptMaxPosition", _POINT), ("ptMinTrackSize", _POINT),
                ("ptMaxTrackSize", _POINT)]


class _PAINTSTRUCT(ctypes.Structure):
    _fields_ = [("hdc", wintypes.HDC), ("fErase", wintypes.BOOL),
                ("rcPaint", _RECT), ("fRestore", wintypes.BOOL),
                ("fIncUpdate", wintypes.BOOL),
                ("rgbReserved", ctypes.c_byte * 32)]


class _MSG(ctypes.Structure):
    _fields_ = [("hwnd", wintypes.HWND), ("message", wintypes.UINT),
                ("wParam", wintypes.WPARAM), ("lParam", wintypes.LPARAM),
                ("time", wintypes.DWORD), ("pt", _POINT)]


_WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT,
                              ctypes.c_size_t, ctypes.c_ssize_t)


class _WNDCLASSEXW(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.UINT), ("style", wintypes.UINT),
                ("lpfnWndProc", _WNDPROC), ("cbClsExtra", ctypes.c_int),
                ("cbWndExtra", ctypes.c_int), ("hInstance", wintypes.HINSTANCE),
                ("hIcon", wintypes.HICON), ("hCursor", wintypes.HANDLE),
                ("hbrBackground", wintypes.HBRUSH),
                ("lpszMenuName", wintypes.LPCWSTR),
                ("lpszClassName", wintypes.LPCWSTR), ("hIconSm", wintypes.HICON)]


def _bgr(r, g, b):
    return r | (g << 8) | (b << 16)


def _wnd_proc(hwnd, msg, wp, lp):
    if msg == WM_ERASEBKGND:
        return 1
    if msg == WM_GETMINMAXINFO:
        # let the test shrink this window to nothing, so the zero-size client
        # path can be reached on a real window and not only on a fake rect
        mmi = ctypes.cast(lp, ctypes.POINTER(_MINMAXINFO)).contents
        mmi.ptMinTrackSize = _POINT(0, 0)
        mmi.ptMaxTrackSize = _POINT(4000, 4000)
        return 0
    if msg == WM_PAINT:
        ps = _PAINTSTRUCT()
        hdc = user32.BeginPaint(hwnd, ctypes.byref(ps))
        r = _RECT()
        user32.GetClientRect(hwnd, ctypes.byref(r))
        w, h = int(r.right), int(r.bottom)
        brush = gdi32.CreateSolidBrush(BG)
        user32.FillRect(hdc, ctypes.byref(_RECT(0, 0, w, h)), brush)
        gdi32.DeleteObject(brush)
        pen = gdi32.CreatePen(PS_SOLID, 1, GRID_X)
        old = gdi32.SelectObject(hdc, pen)
        x = 0
        while x < w:
            gdi32.MoveToEx(hdc, x, 0, None)
            gdi32.LineTo(hdc, x, h)
            x += 7
        gdi32.SelectObject(hdc, old)
        gdi32.DeleteObject(pen)
        pen = gdi32.CreatePen(PS_SOLID, 1, GRID_Y)
        old = gdi32.SelectObject(hdc, pen)
        y = 0
        while y < h:
            gdi32.MoveToEx(hdc, 0, y, None)
            gdi32.LineTo(hdc, w, y)
            y += 11
        gdi32.SelectObject(hdc, old)
        gdi32.DeleteObject(pen)
        for bx in range(0, w, 13):
            for by in range(0, h, 17):
                brush = gdi32.CreateSolidBrush(BLOCK)
                user32.FillRect(hdc, ctypes.byref(
                    _RECT(bx, by, bx + 5, by + 5)), brush)
                gdi32.DeleteObject(brush)
        brush = gdi32.CreateSolidBrush(MARKER)
        user32.FillRect(hdc, ctypes.byref(
            _RECT(0, 0, MARKER_SIZE, MARKER_SIZE)), brush)
        gdi32.DeleteObject(brush)
        user32.EndPaint(hwnd, ctypes.byref(ps))
        return 0
    if msg == WM_DESTROY:
        user32.PostQuitMessage(0)
        return 0
    return user32.DefWindowProcW(hwnd, msg, wp, lp)


# kept alive for the process: ctypes does not own the function pointer
_WNDPROC_REF = _WNDPROC(_wnd_proc)

_CLASS = "PokerVisionCaptureTestWindow"
_registered = False


def _register_class():
    global _registered
    if _registered:
        return
    wc = _WNDCLASSEXW()
    wc.cbSize = ctypes.sizeof(_WNDCLASSEXW)
    wc.lpfnWndProc = _WNDPROC_REF
    wc.hInstance = kernel32.GetModuleHandleW(None)
    wc.lpszClassName = _CLASS
    if not user32.RegisterClassExW(ctypes.byref(wc)):
        raise ctypes.WinError(ctypes.get_last_error())
    _registered = True


def _set_prototypes():
    user32.RegisterClassExW.argtypes = [ctypes.POINTER(_WNDCLASSEXW)]
    user32.RegisterClassExW.restype = wintypes.ATOM
    user32.CreateWindowExW.restype = wintypes.HWND
    user32.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR,
                                       wintypes.LPCWSTR, wintypes.DWORD,
                                       ctypes.c_int, ctypes.c_int,
                                       ctypes.c_int, ctypes.c_int,
                                       wintypes.HWND, wintypes.HANDLE,
                                       wintypes.HINSTANCE, ctypes.c_void_p]
    user32.DefWindowProcW.restype = ctypes.c_ssize_t
    user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT,
                                      ctypes.c_size_t, ctypes.c_ssize_t]
    user32.GetMessageW.restype = ctypes.c_int
    user32.GetMessageW.argtypes = [ctypes.POINTER(_MSG), wintypes.HWND,
                                   ctypes.c_uint, ctypes.c_uint]
    user32.TranslateMessage.argtypes = [ctypes.POINTER(_MSG)]
    user32.DispatchMessageW.argtypes = [ctypes.POINTER(_MSG)]
    user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT,
                                    ctypes.c_size_t, ctypes.c_ssize_t]
    user32.SetWindowPos.argtypes = [wintypes.HWND, ctypes.c_void_p,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, wintypes.UINT]
    user32.SetWindowPos.restype = wintypes.BOOL
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.InvalidateRect.argtypes = [wintypes.HWND, ctypes.c_void_p,
                                      wintypes.BOOL]
    user32.UpdateWindow.argtypes = [wintypes.HWND]
    user32.IsWindow.restype = wintypes.BOOL
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsIconic.restype = wintypes.BOOL
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.GetClientRect.restype = wintypes.BOOL
    user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(_RECT)]
    user32.BeginPaint.restype = wintypes.HDC
    user32.BeginPaint.argtypes = [wintypes.HWND, ctypes.POINTER(_PAINTSTRUCT)]
    user32.EndPaint.restype = wintypes.BOOL
    user32.EndPaint.argtypes = [wintypes.HWND, ctypes.POINTER(_PAINTSTRUCT)]
    user32.FillRect.restype = ctypes.c_int
    user32.FillRect.argtypes = [wintypes.HDC, ctypes.POINTER(_RECT),
                                wintypes.HBRUSH]
    gdi32.CreateSolidBrush.restype = wintypes.HBRUSH
    gdi32.CreateSolidBrush.argtypes = [wintypes.COLORREF]
    gdi32.CreatePen.restype = wintypes.HPEN
    gdi32.CreatePen.argtypes = [ctypes.c_int, ctypes.c_int, wintypes.COLORREF]
    gdi32.SelectObject.restype = wintypes.HANDLE
    gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HANDLE]
    gdi32.DeleteObject.restype = wintypes.BOOL
    gdi32.DeleteObject.argtypes = [wintypes.HANDLE]
    gdi32.MoveToEx.restype = wintypes.BOOL
    gdi32.MoveToEx.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int,
                               ctypes.c_void_p]
    gdi32.LineTo.restype = wintypes.BOOL
    gdi32.LineTo.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
    kernel32.GetModuleHandleW.restype = wintypes.HMODULE
    kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]


_set_prototypes()


class TestWindow:
    """One real top-level window, owned by a thread that pumps its messages.

    `proc.kill()` is not a way to close a window - a Store-app window outlives
    its process, and a "window closed" test that relied on it passed for the
    wrong reason. So the window is closed by posting WM_CLOSE and the message
    loop actually exiting, and every phase waits for the handle to be dead
    before it asserts on it.
    """

    def __init__(self, title, x, y, w, h, cls=_CLASS, style=WS_OVERLAPPEDWINDOW):
        _register_class()
        self._box: queue.Queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._run, args=(title, x, y, w, h, cls, style), daemon=True)
        self._thread.start()
        kind, self.hwnd = self._box.get(timeout=10)
        if kind != "hwnd" or not self.hwnd:
            raise RuntimeError(f"CreateWindowExW failed for {title!r}")

    def _run(self, title, x, y, w, h, cls, style):
        hinst = kernel32.GetModuleHandleW(None)
        hwnd = user32.CreateWindowExW(
            WS_EX_TOPMOST | WS_EX_TOOLWINDOW, cls, title, style,
            x, y, w, h, None, None, hinst, None)
        if not hwnd:
            self._box.put(("error", 0))
            return
        # top-most and shown before the test is told the handle exists, so
        # nothing is on top of the client area when the first frame is grabbed
        user32.SetWindowPos(hwnd, HWND_TOPMOST, x, y, w, h, SWP_SHOWWINDOW)
        self._box.put(("hwnd", hwnd))
        msg = _MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
        self._box.put(("gone", hwnd))

    def move(self, x, y, w, h, settle=0.30):
        """SetWindowPos, then force a repaint and let the thread pump it."""
        user32.SetWindowPos(self.hwnd, HWND_TOPMOST, x, y, w, h,
                            SWP_SHOWWINDOW)
        user32.InvalidateRect(self.hwnd, None, True)
        user32.UpdateWindow(self.hwnd)
        time.sleep(settle)

    def minimise(self, settle=0.30):
        user32.ShowWindow(self.hwnd, SW_MINIMIZE)
        time.sleep(settle)

    def restore(self, settle=0.30):
        user32.ShowWindow(self.hwnd, SW_RESTORE)
        time.sleep(settle)

    def alive(self) -> bool:
        return bool(user32.IsWindow(self.hwnd))

    def iconic(self) -> bool:
        return bool(user32.IsIconic(self.hwnd))

    def close(self, timeout=5.0) -> bool:
        """Close for real, then wait until the handle is genuinely dead."""
        if not self.alive():
            return True
        user32.PostMessageW(self.hwnd, WM_CLOSE, 0, 0)
        self._thread.join(timeout=timeout)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and self.alive():
            time.sleep(0.02)
        return not self.alive()


# --------------------------------------------------------------------------

def marker_at(frame) -> bool:
    """True when the frame's own (0, 0) pixel is the marker we painted at the
    client's (0, 0). That is the proof that origin == client top-left: if the
    frame were shifted by even one pixel, or the grab had been clamped, the
    marker would not be in the corner."""
    return frame[0, 0].tolist() == MARK_RGB.tolist()


cfg = Config.load()
print(f"config: monitor {cfg.monitor}, {len(cfg.board_slots)} board slots, "
      f"match_score {cfg.match_score}")

desktop = Capture(cfg)
d = desktop.rect
print(f"mss monitor {cfg.monitor}: {d}  (frame grab for the whole desktop "
      f"used below)")

MARGIN = 60
w1 = min(520, d["width"] - 2 * MARGIN)
h1 = min(400, d["height"] - 2 * MARGIN)
print(f"  info  desktop is {d['width']}x{d['height']}; throwaway window "
      f"{w1}x{h1} at ({d['left'] + MARGIN},{d['top'] + MARGIN})")

print("\n--- 1. Capture on monitor 0 ---")
check("frame height matches the monitor height",
      desktop.get_frame().shape[0], d["height"])
check("frame width matches the monitor width",
      desktop.get_frame().shape[1], d["width"])
check("frame is 3-channel BGR", desktop.get_frame().shape[2], 3)
check("origin is the monitor top-left", desktop.origin,
      (d["left"], d["top"]))
check("rect is the monitor it reports", desktop.rect, d)

print("\n--- 8. list_windows ---")
found = win.list_windows()
check_true("list_windows is non-empty", found, "no candidate windows at all")
small = [w for w in found if w.client[2] <= 0 or w.client[3] <= 0]
check_true("every entry has a positive client w/h", not small,
           f"{[(w.hwnd, w.client) for w in small]}")
areas = [w.area for w in found]
check("sorted by descending client area", areas,
      sorted(areas, reverse=True))
check_true("titles are non-empty", all(w.title.strip() for w in found))
check_true("hwnds are unique", len({w.hwnd for w in found}) == len(found))
print(f"  info  {len(found)} windows, largest is "
      f"{win.console_safe(found[0].title)!r} at {found[0].client}")

print("\n--- 7. console_safe survives a real title ---")
nasty = "Monopoly\u200b Poker \U0001f3b4\nsecond line\U0001f600\u200b\u200d"
safe = win.console_safe(nasty)
print(f"  info  in : {nasty!r}")
print(f"  info  out: {safe!r}")
check_true("no newline survives", "\n" not in safe, repr(safe))
check_true("no zero-width space survives", "\u200b" not in safe, repr(safe))
check_true("something printable comes back", bool(safe.strip()))
check_true("encodes to the console's own encoding without error",
           safe.encode(sys.stdout.encoding or "ascii", "replace").decode(
               sys.stdout.encoding or "ascii", "replace") == safe)
check("a plain title passes through unchanged", win.console_safe("Poker"), "Poker")
check("a real window title is printable",
      isinstance(win.console_safe(found[0].title), str), True)

if w1 < 200 or h1 < 160:
    print(f"  FAIL  desktop too small to host the throwaway window "
          f"({d['width']}x{d['height']}) - every window test below needs one")
    FAILURES.append("throwaway window")
else:
    probe = TestWindow("pokervision capture test", d["left"] + MARGIN,
                       d["top"] + MARGIN, w1, h1)
    info = win.inspect(probe.hwnd)
    check_true("throwaway window is live and inspectable", info is not None)
    print(f"  info  hwnd 0x{probe.hwnd:X} window rect {info.rect} "
          f"client {info.client} - chrome excluded")

    print("\n--- 2. WindowCapture frame geometry ---")
    wcap = WindowCapture(cfg, probe.hwnd, info.title)
    wframe = wcap.get_frame()
    l, t, cw, ch = info.client
    check("frame height is the reported client height", wframe.shape[0], ch)
    check("frame width is the reported client width", wframe.shape[1], cw)
    check("origin is the client rect's top-left", wcap.origin, (l, t))
    check("rect is the client rect", wcap.rect,
          {"left": l, "top": t, "width": cw, "height": ch})
    check("client is smaller than the window (chrome excluded)",
          cw < info.rect[2] and ch < info.rect[3], True)
    check_true("the frame starts on the marker painted at client (0, 0)",
               marker_at(wframe),
               f"got {wframe[0, 0].tolist()}, want magenta "
               f"{MARK_RGB.tolist()}")
    check_true("WindowCapture is a Capture (same surface downstream)",
               isinstance(wcap, Capture), type(wcap).__name__)

    print("\n--- 3. crop() is inherited, so absolute rects become "
          "client-relative ---")
    # well inside the window, so both grabs read the same static pixels
    r = rc.Rect(l + 40, t + 30, 140, 100)
    mframe = desktop.get_frame()
    from_window = wcap.crop(wframe, r)
    from_monitor = desktop.crop(mframe, r)
    check("crop of an absolute rect has the rect's size", from_window.shape,
          (r.h, r.w, 3))
    check_pixels("window crop is pixel-identical to the monitor crop of the "
                 "same absolute rect", from_window, from_monitor)
    check_true("that crop is not a flat colour (a wrong offset would show)",
               float(from_window.std()) > 1.0, "the window content is uniform")
    ox, oy = wcap.origin
    check_pixels("the crop is the client-relative slice of the frame",
                 from_window,
                 wframe[r.y - oy:r.bottom - oy, r.x - ox:r.right - ox])
    check_pixels("a rect at the origin itself crops out the marker",
                 wcap.crop(wframe, rc.Rect(ox, oy, MARKER_SIZE, MARKER_SIZE)),
                 np.full((MARKER_SIZE, MARKER_SIZE, 3), MARK_RGB, np.uint8))
    later = rc.Rect(l + 200, t + 150, 60, 40)
    check_pixels("a second, different rect maps through the same offset",
                 wcap.crop(wframe, later),
                 wframe[later.y - oy:later.bottom - oy,
                        later.x - ox:later.right - ox])
    check_pixels("and matches the monitor crop there too",
                 wcap.crop(wframe, later), desktop.crop(mframe, later))
    over = rc.Rect(l + cw - 30, t + ch - 20, 400, 400)
    edge = wcap.crop(wframe, over)
    check("a rect hanging off the client is clipped to the frame, not padded",
           edge.shape, (20, 30, 3))
    off_table = rc.Rect(l + cw + 50, t + ch + 50, 40, 40)
    check("a rect entirely outside the client comes back empty",
           wcap.crop(wframe, off_table).shape, (0, 0, 3))

    print("\n--- 4. move and resize: the next frame re-queries ---")
    x2 = d["left"] + MARGIN + 80
    y2 = d["top"] + MARGIN + 60
    w2 = min(w1 + 120, d["left"] + d["width"] - MARGIN - x2)
    h2 = min(h1 + 90, d["top"] + d["height"] - MARGIN - y2)
    print(f"  info  SetWindowPos -> ({x2},{y2}) {w2}x{h2}")
    before = wcap.client
    probe.move(x2, y2, w2, h2)
    moved = wcap.get_frame()
    fresh = win.inspect(probe.hwnd)
    check("the new frame has the new width", moved.shape[1],
          fresh.client[2])
    check("the new frame has the new height", moved.shape[0],
          fresh.client[3])
    check("the new client rect is what Win32 says now", wcap.client,
          fresh.client)
    check("the new origin is the new client top-left", wcap.origin,
          fresh.client[:2])
    check("origin really moved", wcap.origin != before[:2], True)
    check_true("and the marker is still at frame (0, 0) after the move",
               marker_at(moved), f"got {moved[0, 0].tolist()}")
    mframe2 = desktop.get_frame()
    r2 = rc.Rect(wcap.origin[0] + 40, wcap.origin[1] + 30, 140, 100)
    check_pixels("crop still matches the monitor crop at the new origin",
                 wcap.crop(moved, r2), desktop.crop(mframe2, r2))

    print("\n--- 6. minimised / zero-size client raises ---")
    probe.minimise()
    check_true("the window really is minimised", probe.iconic())
    check_raises("get_frame on a minimised window", win.WindowLost,
                 wcap.get_frame)
    check_raises("building a new capture on a minimised window", win.WindowLost,
                 WindowCapture, cfg, probe.hwnd, info.title)
    probe.restore()
    check_true("and it comes back", win.inspect(probe.hwnd) is not None)

    # the same guard without depending on how a shell reports a minimised
    # window's client area: squeeze this one until there is no client at all
    probe.move(d["left"] + MARGIN, d["top"] + MARGIN, 1, 1)
    check_true("a window squeezed to nothing has no usable client",
               win.inspect(probe.hwnd) is None,
               f"inspect says {win.inspect(probe.hwnd).client}"
               if win.inspect(probe.hwnd) else "")
    check_raises("get_frame on a window with no client area", win.WindowLost,
                 wcap.get_frame)
    probe.move(x2, y2, w2, h2)
    check("and it captures again once it has room", wcap.get_frame().shape[0],
          wcap.client[3])

    print("\n--- 5. a dead handle fails loudly, never silently ---")
    got = wcap.get_frame()  # a live frame first, so the failure is not vacuous
    check_true("still capturing before the close", got.shape[1] > 0)
    check_true("the window is really closed", probe.close())
    check("IsWindow is now false", probe.alive(), False)
    check_raises("get_frame on a closed window", win.WindowLost, wcap.get_frame)
    check_raises("constructing a capture on a closed handle", win.WindowLost,
                 WindowCapture, cfg, probe.hwnd, info.title)
    check_raises("and client_rect on it too", win.WindowLost,
                 win.client_rect, probe.hwnd, info.pid, info.title, False)
    try:
        wcap.get_frame()
    except win.WindowLost as exc:
        print(f"  info  the message says: {exc}")
        check_true("the message names the dead handle",
                   f"{probe.hwnd:X}" in str(exc).upper(), str(exc))

    # a window with no client area at all, to reach the zero-size branch
    # without relying on how a particular shell treats a minimised window
    zero = TestWindow("pokervision zero client", d["left"] + 8, d["top"] + 8,
                      0, 0, cls="STATIC", style=0)
    check("a chrome-less 0x0 window has no usable client",
          win.inspect(zero.hwnd), None)
    check_raises("constructing a capture on a zero-size client",
                 win.WindowLost, WindowCapture, cfg, zero.hwnd, None)
    zero.close()

    print("\n--- a negative client top is grabbed, and that is correct ---")
    # a maximised window reports its client area starting above the desktop
    # origin; mss handles it, so this is measured and pinned, not "fixed"
    if d["height"] > 200:
        high = TestWindow("pokervision over the top", d["left"] + 120,
                          d["top"] - 40, 360, 260)
        hinfo = win.inspect(high.hwnd)
        hcap = WindowCapture(cfg, high.hwnd, hinfo.title)
        check("its client rect really does start above the desktop",
              hcap.origin[1] < 0, True)
        hframe = hcap.get_frame()
        check("and the frame comes back at the full client size", hframe.shape,
              (hcap.client[3], hcap.client[2], 3))
        hcap.close()
        high.close()
    else:
        print("  info  desktop too short to place a window over the top edge")

desktop.close()
try:
    wcap.close()
except Exception:  # noqa: BLE001 - already closed with the window
    pass

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURES: {FAILURES}")
    raise SystemExit(1)
print("all capture and window tests passed")
