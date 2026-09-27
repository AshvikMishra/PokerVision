from __future__ import annotations

import cv2
import numpy as np

import regions_config as rc
from .config import Config


class Capture:
    def __init__(self, cfg: Config):
        import mss

        self.cfg = cfg
        self.sct = mss.mss()
        mon = self.sct.monitors[min(cfg.monitor, len(self.sct.monitors) - 1)]
        self.origin = (mon["left"], mon["top"])
        self.rect = {"left": mon["left"], "top": mon["top"],
                     "width": mon["width"], "height": mon["height"]}

    @staticmethod
    def offline() -> "Capture":
        c = Capture.__new__(Capture)
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
