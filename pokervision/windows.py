"""Windows-only window enumeration and client-rect lookup, via ctypes.

No third-party dependency: overlay.py already uses ctypes.WinDLL for its
click-through styles, so this follows the same pattern rather than adding
pygetwindow or pywin32.

DPI awareness matters here. Win32 reports window geometry in virtualised
coordinates unless the process opted in, while mss reports physical pixels. On
a display not at 100% scaling the two disagree and every region lands in the
wrong place, so awareness is set before any rect is read.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from dataclasses import dataclass

user32 = ctypes.WinDLL("user32", use_last_error=True)


class WindowLost(RuntimeError):
    """The selected window is gone, or its handle has been reused."""


class _RECT(ctypes.Structure):
    _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG),
                ("right", wintypes.LONG), ("bottom", wintypes.LONG)]


class _POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


_WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(_RECT)]
user32.GetClientRect.restype = wintypes.BOOL
user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(_POINT)]
user32.ClientToScreen.restype = wintypes.BOOL
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetWindowTextW.restype = ctypes.c_int
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.GetWindowTextLengthW.restype = ctypes.c_int
user32.IsWindow.argtypes = [wintypes.HWND]
user32.IsWindow.restype = wintypes.BOOL
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.IsWindowVisible.restype = wintypes.BOOL
user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(_RECT)]
user32.GetWindowRect.restype = wintypes.BOOL
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND,
                                            ctypes.POINTER(wintypes.DWORD)]
user32.GetWindowThreadProcessId.restype = wintypes.DWORD


def console_safe(text: str) -> str:
    """Make arbitrary window text printable on the current console.

    Titles are whatever the app put there, so they arrive with emoji, zero-width
    spaces and embedded newlines, and a cp1252 console raises on the first of
    those - which took the picker down before it listed anything.
    """
    cleaned = "".join(ch if ch.isprintable() else " " for ch in text)
    enc = getattr(sys.stdout, "encoding", None) or "ascii"
    return cleaned.encode(enc, "replace").decode(enc, "replace").strip()


def _ensure_dpi_aware() -> None:
    try:
        user32.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        user32.SetProcessDpiAwarenessContext.restype = wintypes.BOOL
        if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.WinDLL("shcore").SetProcessDpiAwareness(2)
        return
    except (AttributeError, OSError):
        pass
    try:
        user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


_ensure_dpi_aware()


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str
    rect: tuple[int, int, int, int]
    client: tuple[int, int, int, int]
    pid: int = 0

    @property
    def area(self) -> int:
        return self.client[2] * self.client[3]

    @property
    def key(self) -> str:
        """Stable-ish identity for storing a selection against this window.

        Browser titles carry the page name and the app name and change per tab,
        so the middle is the only part worth keying on. Exact title is kept as
        a secondary match.
        """
        parts = [p.strip() for p in self.title.replace("\u2014", "-").split("-")]
        middle = max(parts, key=len) if parts else self.title
        return middle or self.title

    def describe(self) -> str:
        l, t, w, h = self.client
        return (f"hwnd=0x{self.hwnd:X}  client={w}x{h}+{l}+{t}  "
                f"title={console_safe(self.title)!r}")


def title_of(hwnd: int) -> str:
    n = user32.GetWindowTextLengthW(hwnd)
    if n <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def process_id(hwnd: int) -> int:
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return int(pid.value)


def _client_rect(hwnd: int) -> tuple[int, int, int, int]:
    r = _RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(r)):
        raise ctypes.WinError(ctypes.get_last_error())
    origin = _POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
        raise ctypes.WinError(ctypes.get_last_error())
    return origin.x, origin.y, r.right - r.left, r.bottom - r.top


def _window_rect(hwnd: int) -> tuple[int, int, int, int]:
    r = _RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(r)):
        return 0, 0, 0, 0
    return r.left, r.top, r.right - r.left, r.bottom - r.top


def inspect(hwnd: int) -> WindowInfo | None:
    """Current geometry for one handle, or None if it is no longer a window."""
    if not user32.IsWindow(hwnd):
        return None
    try:
        client = _client_rect(hwnd)
    except OSError:
        return None
    if client[2] <= 0 or client[3] <= 0:
        return None
    return WindowInfo(int(hwnd), title_of(hwnd), _window_rect(hwnd), client,
                      process_id(hwnd))


def list_windows(include_hidden: bool = False) -> list[WindowInfo]:
    """Every top-level window that could plausibly be a game client.

    Largest first, so the game window is usually choice 1 and a mis-click lands
    on something obviously wrong rather than something subtly wrong.
    """
    found: list[WindowInfo] = []

    def collect(hwnd, _lparam):
        try:
            if not include_hidden and not user32.IsWindowVisible(hwnd):
                return True
            t = title_of(hwnd)
            if not t.strip():
                return True
            try:
                client = _client_rect(hwnd)
            except OSError:
                return True
            if client[2] < 64 or client[3] < 64:
                return True
            found.append(WindowInfo(int(hwnd), t, _window_rect(hwnd), client,
                                    process_id(hwnd)))
        except Exception:
            pass
        return True

    user32.EnumWindows(_WNDENUMPROC(collect), 0)
    found.sort(key=lambda w: (-w.area, w.title.lower()))
    return found


def client_rect(hwnd: int, expect_pid: int | None = None,
                expect_title: str | None = None,
                strict_title: bool = False) -> tuple[int, int, int, int]:
    """Re-read the client rect in screen coordinates.

    Called every frame, so this is the move/resize path as well as the
    stale-handle check. Two signals are authoritative and both are fatal: a
    dead handle, and a handle now owned by a different process. A changed title
    is not fatal by default, because real apps retitle - browsers on
    notification counts, Notepad as you type - and treating that as death
    produced a false "window closed" on the very first test.
    """
    if not user32.IsWindow(hwnd):
        raise WindowLost(f"window 0x{hwnd:X} no longer exists - it was closed.")
    if expect_pid is not None and process_id(hwnd) != expect_pid:
        raise WindowLost(
            f"window 0x{hwnd:X} now belongs to a different process - the handle "
            f"was reused. Re-select the window."
        )
    if strict_title and expect_title is not None and title_of(hwnd) != expect_title:
        raise WindowLost(
            f"window 0x{hwnd:X} is now titled "
            f"{console_safe(title_of(hwnd))!r} - strict title matching is on."
        )
    try:
        rect = _client_rect(hwnd)
    except OSError as exc:
        raise WindowLost(f"could not read the client rect of 0x{hwnd:X}: {exc}") from exc
    if rect[2] <= 0 or rect[3] <= 0:
        raise WindowLost(
            f"window 0x{hwnd:X} reports a {rect[2]}x{rect[3]} client area - it is "
            f"minimised or collapsed. Restore it and re-run."
        )
    return rect


def format_picker(windows: list[WindowInfo]) -> str:
    if not windows:
        return "  (no candidate windows found)"
    lines = []
    for i, w in enumerate(windows, 1):
        l, t, cw, ch = w.client
        chrome = (w.rect[2] * w.rect[3]) - (cw * ch)
        lines.append(f"  [{i:2d}] hwnd=0x{w.hwnd:<8X} pid {w.pid:<6d} "
                     f"client {cw:5d}x{ch:<5d} at ({l:5d},{t:5d})  "
                     f"chrome {chrome:6d}px  {console_safe(w.title)}")
    return "\n".join(lines)
