from __future__ import annotations

import cv2
import numpy as np

from . import regions as rc
from .config import Config


class Capture:
    def __init__(self, cfg: Config):
        import mss

        self.cfg = cfg
        self.sct = mss.mss()
        # kept on the instance, not just read from cfg: describe_source()
        # reports what is being captured and runs on the monitor path too,
        # which is every blank answer, EOF, Ctrl-C and --monitor
        self.monitor = min(cfg.monitor, len(self.sct.monitors) - 1)
        mon = self.sct.monitors[self.monitor]
        self.origin = (mon["left"], mon["top"])
        self.rect = {"left": mon["left"], "top": mon["top"],
                     "width": mon["width"], "height": mon["height"]}

    @staticmethod
    def offline() -> "Capture":
        c = Capture.__new__(Capture)
        c.monitor = 0
        c.origin = (0, 0)
        c.rect = {"left": 0, "top": 0, "width": 0, "height": 0}
        return c

    def get_frame(self) -> np.ndarray:
        raw = np.asarray(self.sct.grab(self.rect))
        return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

    def crop(self, frame: np.ndarray, r: rc.Rect) -> np.ndarray:
        ox, oy = self.origin
        x0, y0 = max(0, r.x - ox), max(0, r.y - oy)
        x1 = min(frame.shape[1], r.right - ox)
        y1 = min(frame.shape[0], r.bottom - oy)
        if x1 <= x0 or y1 <= y0:
            return np.empty((0, 0, 3), np.uint8)
        crop = frame[y0:y1, x0:x1]
        if not r.angle:
            return crop
        h, w = crop.shape[:2]
        # positive angle = counter-clockwise, about the crop's own centre
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), r.angle, 1.0)
        return cv2.warpAffine(
            crop, matrix, (w, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def close(self) -> None:
        self.sct.close()


class WindowCapture(Capture):
    """Capture from one window's CLIENT rect instead of a whole monitor.

    Same `get_frame()` / `crop()` surface as `Capture`, so nothing downstream
    knows the difference. `origin` is the client rect's top-left, which makes
    every region client-relative for free. The client rect is re-read every
    frame, so moving or resizing the window needs no reconfiguration, and a
    window that closes raises rather than quietly capturing whatever is now at
    those coordinates.

    A client rect that starts *above* the desktop origin is fine and is not
    special-cased: a maximised window reports one (measured at `(0, -9)` here),
    and mss grabs it correctly at full size. An earlier version refused such a
    rect on the assumption that mss could not, which would have broken exactly
    the maximised-browser case that matters most. Measured, not assumed.
    """

    def __init__(self, cfg: Config, hwnd: int, title: str | None = None,
                 strict_title: bool = False):
        import mss

        from . import windows as win

        info = win.inspect(hwnd)
        if info is None:
            raise win.WindowLost(
                f"window 0x{hwnd:X} is not a live window with a usable client "
                f"area. Re-select it."
            )
        self.cfg = cfg
        self.hwnd = int(hwnd)
        self.title = title
        self.pid = info.pid
        self.strict_title = strict_title
        self.monitor = cfg.monitor
        self.sct = mss.mss()
        self._publish(win.client_rect(self.hwnd, self.pid, self.title,
                                      self.strict_title))

    def _publish(self, client: tuple[int, int, int, int]) -> None:
        left, top, width, height = client
        self.client = client
        self.origin = (left, top)
        self.rect = {"left": left, "top": top,
                     "width": width, "height": height}

    def get_frame(self) -> np.ndarray:
        from . import windows as win

        # re-query every frame: both the move/resize path and the stale check
        self._publish(win.client_rect(self.hwnd, self.pid, self.title,
                                      self.strict_title))
        try:
            raw = np.asarray(self.sct.grab(self.rect))
        except Exception as exc:
            raise win.WindowLost(
                f"could not grab the client rect {self.rect} of window "
                f"0x{self.hwnd:X} (now at {self.origin}): {exc}"
            ) from exc
        return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
