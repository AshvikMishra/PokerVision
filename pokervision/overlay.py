from __future__ import annotations

import os

os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")

import ctypes
import sys

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QColor, QFont, QPen
from PySide6.QtWidgets import QApplication, QWidget

from . import regions as rc
from .config import Config
from .capture import Capture, WindowCapture
from .perception import Card, CardReader, GlyphBank
from .state import Reading, read_frame

GREY = QColor(110, 110, 110)
YELLOW = QColor(255, 210, 0)
GREEN = QColor(60, 220, 60)
WHITE = QColor(255, 255, 255)


class RectTracker:
    """EMA over each slot's own detected rect. Rendering only: perception and
    decisions still key off the raw read."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self._smoothed: dict[str, tuple[float, float, float, float]] = {}

    def update(self, where: str, rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x, y, w, h = rect
        if where not in self._smoothed:
            self._smoothed[where] = (x, y, w, h)
        else:
            ox, oy, ow, oh = self._smoothed[where]
            a = self.alpha
            self._smoothed[where] = (ox + a * (x - ox), oy + a * (y - oy),
                                     ow + a * (w - ow), oh + a * (h - oh))
        return tuple(int(round(v)) for v in self._smoothed[where])

    def drop(self, where: str) -> None:
        self._smoothed.pop(where, None)


def _harden_click_through(win_id: int) -> None:
    if os.name != "nt":
        return
    GWL_EXSTYLE, WS_EX_LAYERED, WS_EX_TRANSPARENT = -20, 0x00080000, 0x00000020
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    style = user32.GetWindowLongW(win_id, GWL_EXSTYLE)
    user32.SetWindowLongW(win_id, GWL_EXSTYLE, style | WS_EX_LAYERED | WS_EX_TRANSPARENT)


class OverlayWindow(QWidget):
    def __init__(self, cfg: Config, cap: Capture, reader: CardReader):
        super().__init__()
        self.cfg, self.cap, self.reader = cfg, cap, reader
        self.reading: Reading | None = None
        self.paused = False
        self.tracker = RectTracker()

        r = cap.rect
        self.setGeometry(r["left"], r["top"], r["width"], r["height"])
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_ShowWithoutActivating)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(int(1000 / max(1, cfg.fps)))

    def showEvent(self, event):
        super().showEvent(event)
        _harden_click_through(int(self.winId()))

    def tick(self):
        if self.paused:
            return
        frame = self.cap.get_frame()
        self.reading = read_frame(self.reader, frame, self.cap, self.cfg)
        self.update()

    def _local(self, r: rc.Rect):
        ox, oy = self.cap.origin
        return int(r.x - ox), int(r.y - oy), int(r.w), int(r.h)

    def _box(self, p: QPainter, r: rc.Rect, colour: QColor, label="", sub="") -> None:
        self._box_xywh(p, *self._local(r), colour, label, sub)

    def _box_xywh(self, p: QPainter, x: int, y: int, w: int, h: int,
                  colour: QColor, label="", sub="") -> None:
        p.setPen(QPen(colour, 2)); p.drawRect(x, y, w, h)
        if label:
            p.setFont(QFont("Segoe UI", 11, QFont.Bold)); p.setPen(QPen(colour))
            ty = y - 8 if y > 20 else y + h + 18
            p.drawText(x, ty, label)
            p.setFont(QFont("Segoe UI", 8)); p.setPen(QPen(QColor(220, 220, 220)))
            for i, line in enumerate(sub.splitlines()):
                p.drawText(x, ty + 16 + 13 * i, line)

    def _card(self, p: QPainter, where: str, search_r: rc.Rect, c: Card | None) -> None:
        if c is None:
            self.tracker.drop(where)
            self._box(p, search_r, GREY)
            return
        ox, oy = self.cap.origin
        raw = (c.rect.x - ox, c.rect.y - oy, c.rect.w, c.rect.h)
        x, y, w, h = self.tracker.update(where, raw)
        label = c.code or f"{c.rank or '?'}{c.suit or '?'}"
        sub = f"r{c.rank_score:.2f}/{c.rank_margin:.2f}\ns{c.suit_score:.2f}/{c.suit_margin:.2f}"
        self._box_xywh(p, x, y, w, h, GREEN if c.code else YELLOW, label, sub)

    def _status_panel(self, p: QPainter) -> None:
        """What was read, and how much of it can be trusted.

        The green/amber split is on the read, not on a verdict: amber means
        the read is incomplete, which on this calibrated path means a glyph did
        not match. Nothing here recommends an action.
        """
        r = self.reading
        if r is None:
            return
        ok = r.complete
        colour = GREEN if ok else YELLOW
        problems = r.problems
        h = 56 if not problems else min(150, 56 + 16 * len(problems))
        w = min(660, max(380, self.width() - 24))
        x, y = 12, min(52, max(12, self.height() - h - 12))
        p.setBrush(QColor(20, 18, 16, 210)); p.setPen(QPen(colour, 2))
        p.drawRoundedRect(x, y, w, h, 6, 6)
        p.setFont(QFont("Segoe UI", 12, QFont.Bold)); p.setPen(QPen(colour))
        p.drawText(x + 14, y + 24,
                   "read complete" if ok else "read incomplete")
        p.setFont(QFont("Segoe UI", 9)); p.setPen(QPen(WHITE))
        p.drawText(x + 150, y + 24, r.describe())
        for i, line in enumerate(problems[:5]):
            p.setPen(QPen(YELLOW))
            p.drawText(x + 150, y + 44 + 16 * i, line[:86])

    def paintEvent(self, event):
        if self.reading is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        for i, r in enumerate(self.cfg.board_slots):
            self._card(p, f"board{i}", r,
                       next((c for c in self.reading.board if c.slot == i), None))
        for i, r in enumerate(self.cfg.hero_slots):
            self._card(p, f"hole{i}", r,
                       next((c for c in self.reading.hole if c.slot == i), None))
        for label, r, value in (("pot", self.cfg.pot, self.reading.pot),
                                ("stack", self.cfg.hero_stack,
                                 self.reading.hero_stack)):
            if r is not None:
                txt = f"{label} {value:,}" if value is not None else f"{label} ?"
                self._box(p, r, GREEN if value is not None else YELLOW, txt)
        p.setFont(QFont("Segoe UI", 12, QFont.Bold)); p.setPen(QPen(WHITE))
        p.drawText(12, 28,
                   self.reading.describe() + f"   [{self.reading.street}]")
        self._status_panel(p)


def cmd_overlay_gui(cfg: Config, window: str | None = None,
                    monitor: int | None = None) -> int:
    if not cfg.templates.exists():
        print("No learned glyphs yet. Run:  "
              "python -m pokervision learn --images unmarked-images")
        return 1
    from .cli import pick_capture, describe_source
    cap = pick_capture(cfg, window, monitor)
    print(f"source  : {describe_source(cap)}")
    reader = CardReader(cfg, GlyphBank(cfg.templates, cfg.match_score).load())
    app = QApplication(sys.argv)
    win = OverlayWindow(cfg, cap, reader)
    win.show()

    try:
        import keyboard
        keyboard.add_hotkey("space", lambda: setattr(win, "paused", not win.paused))
        keyboard.add_hotkey("esc", app.quit)
        print("Space = pause/resume   Esc = quit  (work even while you're clicking the table)")
    except ImportError:
        print("pip install keyboard for pause/quit hotkeys that work through the "
              "click-through overlay. Ctrl+C in this terminal still quits.")

    try:
        app.exec()
    finally:
        cap.close()
    return 0
