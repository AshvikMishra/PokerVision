"""The two-button GUI: pick a window, drag the game area, watch it read.

Flow:
    Selection      -> choose the game window, then drag a box around the table
                      in a snapshot of it, and the live read starts
    Pause / Resume -> freeze and restart the live read, so a selection can be
                      checked without closing anything

There is no second stage. This project reads cards; it does not decide what to
do with them.

The dragged box is stored as a fraction of the window's client rect, keyed by
window title, so it survives moving and resizing the window and each window
keeps its own. Saved areas are listed in the main window, can be reused with
one click, and can be forgotten individually or all at once.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from . import regions as rc
from . import windows as win
from .capture import WindowCapture
from .config import Config
from .detect import CardFinder, group_rows
from .perception import GlyphBank

REPO = Path(__file__).resolve().parent.parent
SELECTIONS = REPO / "config" / "selections.json"

# a box this small is a mis-click, not a selection
MIN_DRAG = 20


def load_selections() -> dict:
    if not SELECTIONS.exists():
        return {}
    try:
        data = json.loads(SELECTIONS.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_selections(data: dict) -> None:
    SELECTIONS.parent.mkdir(parents=True, exist_ok=True)
    if data:
        SELECTIONS.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    elif SELECTIONS.exists():
        SELECTIONS.unlink()


def to_qimage(frame: np.ndarray):
    from PySide6.QtGui import QImage
    if frame is None or frame.size == 0:
        return None
    h, w = frame.shape[:2]
    return QImage(frame.data, w, h, 3 * w, QImage.Format_RGB888).rgbSwapped()


def _build_gui(cfg: Config) -> int:
    from PySide6.QtCore import QEvent, QRect, QSize, Qt, QTimer
    from PySide6.QtGui import QFont, QPixmap
    from PySide6.QtWidgets import (QApplication, QDialog, QGroupBox, QHBoxLayout,
                                   QLabel, QListWidget, QListWidgetItem,
                                   QMainWindow, QMessageBox, QPushButton,
                                   QRubberBand, QVBoxLayout, QWidget)

    def fit_pixmap(image, avail_w: int, avail_h: int) -> QPixmap:
        """Scale a frame down to fit the space, never up."""
        pm = QPixmap.fromImage(to_qimage(image))
        if pm.width() <= 0 or avail_w <= 0 or avail_h <= 0:
            return pm
        s = min(avail_w / pm.width(), avail_h / pm.height(), 1.0)
        if s >= 1.0:
            return pm
        return pm.scaled(int(pm.width() * s), int(pm.height() * s),
                         Qt.KeepAspectRatio, Qt.SmoothTransformation)

    class WindowPicker(QDialog):
        """Numbered list of top-level windows, largest first."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Select the game window")
            self.resize(900, 540)
            self.wins = win.list_windows()
            lay = QVBoxLayout(self)
            hint = QLabel("Pick the window running the game, then OK. Its client "
                          "area is used, so browser chrome and title bars are "
                          "excluded.")
            hint.setWordWrap(True)
            lay.addWidget(hint)
            self.list = QListWidget()
            saved = load_selections()
            for i, w in enumerate(self.wins, 1):
                l, t, cw, ch = w.client
                chrome = (w.rect[2] * w.rect[3]) - (cw * ch)
                mark = "  [saved]" if w.key in saved else ""
                item = QListWidgetItem(
                    f"[{i:2d}]  {cw}x{ch} at ({l},{t})   chrome {chrome}px{mark}\n"
                    f"       {win.console_safe(w.title)}")
                self.list.addItem(item)
            if self.list.count():
                self.list.setCurrentRow(0)
            lay.addWidget(self.list, 1)
            row = QHBoxLayout()
            cancel = QPushButton("Cancel")
            cancel.clicked.connect(self.reject)
            ok = QPushButton("OK")
            ok.clicked.connect(self.accept)
            row.addStretch(1)
            row.addWidget(cancel)
            row.addWidget(ok)
            lay.addLayout(row)

        def chosen(self):
            r = self.list.currentRow()
            return self.wins[r] if 0 <= r < len(self.wins) else None

    class RegionPicker(QDialog):
        """A snapshot of the client area, scaled to fit the screen, draggable.

        Robust about where the drag happens. The event filter is installed on
        the dialog as well as on the label, because a press that lands beside
        the image - on the hint text, on the border - is a press the user
        reasonably expects to start a drag, and it used to be dropped silently.
        The mouse is grabbed on press so a drag that wanders outside the image
        still completes. And the band stays up after release instead of
        vanishing, with the resulting client-pixel rect spelled out, so it is
        obvious what was actually captured.
        """

        def __init__(self, frame: np.ndarray, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Drag around the game area")
            self.frame = frame
            self.rect = None
            self._from = None
            lay = QVBoxLayout(self)
            hint = QLabel("Drag a box around the table and its cards. The box is "
                          "kept as a fraction of this window, so moving or "
                          "resizing the window later is fine.")
            hint.setWordWrap(True)
            lay.addWidget(hint)
            self.label = QLabel()
            self.label.setAlignment(Qt.AlignCenter)
            self.label.setStyleSheet("background:#141110;")
            self.label.installEventFilter(self)
            lay.addWidget(self.label, 1)
            self.band = QRubberBand(QRubberBand.Rectangle, self.label)
            self.readout = QLabel("nothing selected yet")
            self.readout.setAlignment(Qt.AlignCenter)
            lay.addWidget(self.readout)
            row = QHBoxLayout()
            cancel = QPushButton("Cancel")
            cancel.clicked.connect(self.reject)
            self.ok = QPushButton("Use this box")
            self.ok.setEnabled(False)
            self.ok.setDefault(True)
            # clicked(bool) into a no-arg slot: tolerate either arity
            self.ok.clicked.connect(lambda *a: self._accept())
            row.addStretch(1)
            row.addWidget(cancel)
            row.addWidget(self.ok)
            lay.addLayout(row)
            self.installEventFilter(self)
            self._fit_to_screen()

        def _fit_to_screen(self):
            """Scale the snapshot to the screen and size the dialog to it.

            Fitted to the screen rather than to the dialog, deliberately: if
            the layout resizes the label after the pixmap is set, the picture is
            stretched and the drag-to-pixel mapping quietly stops being uniform.
            Pinning the label to the pixmap keeps one widget pixel equal to one
            pixmap pixel, so the two scales stay equal. The pixmap is still a
            downscale of the client, so `_scale` converts by that factor -
            see the note there.
            """
            screen = QApplication.screenAt(self.cursor().pos()) or \
                QApplication.primaryScreen()
            avail = screen.availableGeometry() if screen else None
            aw = (int(avail.width() * 0.94) - 60) if avail else 1000
            ah = (int(avail.height() * 0.94) - 210) if avail else 700
            w, h = self.frame.shape[1], self.frame.shape[0]
            s = min(aw / max(1, w), ah / max(1, h), 1.0)
            pm = QPixmap.fromImage(to_qimage(self.frame)).scaled(
                max(1, int(w * s)), max(1, int(h * s)),
                Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label.setFixedSize(pm.size())
            self.label.setPixmap(pm)
            self.resize(pm.width() + 60, pm.height() + 210)

        def _scale(self) -> tuple[float, float]:
            """Label pixels per CLIENT pixel.

            The label is pinned to the pixmap, so one widget pixel is one
            *pixmap* pixel and the label/pixmap ratio is always 1.0. That is not
            the number the picker needs. The pixmap is the client snapshot
            scaled down to fit the screen while `frame` is still the unscaled
            client, so the conversion is the downscale itself. Dividing by 1.0
            made every box on a window bigger than the screen come out a
            fraction of its true size, in the wrong place - which is exactly
            the "the selection does not take" report.
            """
            fh, fw = self.frame.shape[:2]
            sx = self.label.width() / max(1, fw)
            sy = self.label.height() / max(1, fh)
            return (sx or 1.0, sy or 1.0)

        def _point(self, obj, pos):
            """Event position -> label pixels.

            A position is already relative to the widget that received the
            event, so only a press that arrived on the dialog itself needs
            translating.
            """
            return pos if obj is self.label else self.label.mapFrom(self, pos)

        def _apply(self, at):
            # Qt's two-point QRect is inclusive on both edges, so normalizing a
            # right-to-left drag lands a pixel high and a pixel short. Order
            # the corners and treat width as an exclusive right edge, which is
            # what the client conversion below assumes.
            x0, x1 = sorted((self._from.x(), at.x()))
            y0, y1 = sorted((self._from.y(), at.y()))
            r = QRect(x0, y0, x1 - x0, y1 - y0)
            self.band.setGeometry(r)
            return r

        def _to_client(self, r) -> "rc.Rect | None":
            sx, sy = self._scale()
            fh, fw = self.frame.shape[:2]
            x0 = max(0, int(round(r.x() / sx)))
            y0 = max(0, int(round(r.y() / sy)))
            x1 = min(fw, int(round((r.x() + r.width()) / sx)))
            y1 = min(fh, int(round((r.y() + r.height()) / sy)))
            if x1 - x0 >= MIN_DRAG and y1 - y0 >= MIN_DRAG:
                return rc.Rect(x=x0, y=y0, w=x1 - x0, h=y1 - y0)
            return None

        def eventFilter(self, obj, e):
            if obj not in (self, self.label):
                return False
            kind = e.type()
            if kind == QEvent.MouseButtonPress and e.button() == Qt.LeftButton:
                self._from = self._point(obj, e.position().toPoint())
                self.band.setGeometry(QRect(self._from, QSize()))
                self.band.show()
                self.grabMouse()
                return True
            if kind == QEvent.MouseMove and self._from is not None:
                self._apply(self._point(obj, e.position().toPoint()))
                return True
            if kind == QEvent.MouseButtonRelease:
                self._finish(obj, e.position().toPoint())
                return True
            return False

        def _finish(self, obj, at):
            if self._from is None:
                return
            r = self._apply(self._point(obj, at))
            self._from = None
            self.releaseMouse()
            got = self._to_client(r)
            self.rect = got
            if got is None:
                self.readout.setText(
                    "box too small - drag a bigger area over the table")
                self.ok.setEnabled(False)
            else:
                fh, fw = self.frame.shape[:2]
                self.readout.setText(
                    f"selected {got.w}x{got.h} at ({got.x},{got.y}) in the "
                    f"window   {got.w * 100 // max(1, fw)}% wide, "
                    f"{got.h * 100 // max(1, fh)}% tall "
                    f"({fw}x{fh} client)")
                self.ok.setEnabled(True)

        def _accept(self):
            if self.rect is None:
                QMessageBox.information(
                    self, "Nothing selected",
                    "Drag a box over the table first, then press Use this box.")
                return
            self.accept()

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.cfg = cfg
            self.bank = GlyphBank(cfg.templates, cfg.match_score).load()
            self.finder = CardFinder(cfg, self.bank) if self.bank.ready else None
            self.cap = None
            self.info = None
            self.region = None
            self.dead = False
            self.setWindowTitle("PokerVision")
            self.resize(1150, 900)

            central = QWidget()
            lay = QVBoxLayout(central)
            row = QHBoxLayout()
            self.btn_select = QPushButton("Selection")
            self.btn_select.clicked.connect(self.do_selection)
            self.btn_pause = QPushButton("Pause")
            self.btn_pause.setEnabled(False)
            self.btn_pause.setToolTip(
                "Freeze the live read so a selection can be checked.")
            self.btn_pause.clicked.connect(self.toggle_pause)
            row.addWidget(self.btn_select)
            row.addWidget(self.btn_pause)
            row.addStretch(1)
            lay.addLayout(row)

            self.status = QLabel("No window selected. Press Selection.")
            self.status.setFont(QFont("Segoe UI", 10, QFont.Bold))
            lay.addWidget(self.status)

            saved_box = QGroupBox("Saved game areas")
            saved_lay = QHBoxLayout(saved_box)
            self.saved_list = QListWidget()
            self.saved_list.setMaximumHeight(120)
            saved_lay.addWidget(self.saved_list, 1)
            srow = QVBoxLayout()
            self.btn_use = QPushButton("Use")
            self.btn_use.clicked.connect(self.use_saved)
            self.btn_forget = QPushButton("Forget")
            self.btn_forget.clicked.connect(self.forget_saved)
            self.btn_forget_all = QPushButton("Forget all")
            self.btn_forget_all.clicked.connect(self.forget_all)
            for b in (self.btn_use, self.btn_forget, self.btn_forget_all):
                srow.addWidget(b)
            srow.addStretch(1)
            saved_lay.addLayout(srow)
            lay.addWidget(saved_box)

            self.view = QLabel("nothing captured yet")
            self.view.setAlignment(Qt.AlignCenter)
            self.view.setStyleSheet("background:#141110; color:#8a8a8a;")
            self.view.setMinimumHeight(240)
            lay.addWidget(self.view, 1)
            self.verdict = QLabel("no read yet")
            self.verdict.setFont(QFont("Segoe UI", 11, QFont.Bold))
            self.verdict.setWordWrap(True)
            lay.addWidget(self.verdict)
            self.setCentralWidget(central)

            self.timer = QTimer(self)
            self.timer.timeout.connect(self.tick)
            self.refresh_saved()

        # -- saved areas --------------------------------------------------
        def refresh_saved(self):
            self.saved_list.clear()
            data = load_selections()
            if not data:
                item = QListWidgetItem("(none saved yet - press Selection)")
                item.setFlags(Qt.NoItemFlags)
                self.saved_list.addItem(item)
                return
            for key, rec in data.items():
                fw, fh = max(1e-6, rec.get("fw", 0)), max(1e-6, rec.get("fh", 0))
                item = QListWidgetItem(
                    f"{key}  -  {fw * 100:.0f}% x {fh * 100:.0f}% of the window "
                    f"at ({rec.get('fx', 0) * 100:.0f}%, "
                    f"{rec.get('fy', 0) * 100:.0f}%)")
                item.setData(Qt.UserRole, key)
                self.saved_list.addItem(item)
            # select the first row: Use and Forget act on the selected row, and
            # leaving nothing selected made both buttons look broken
            if self.saved_list.count():
                self.saved_list.setCurrentRow(0)

        def _selected_key(self):
            item = self.saved_list.currentItem()
            return item.data(Qt.UserRole) if item is not None else None

        def forget_saved(self):
            key = self._selected_key()
            if not key:
                return
            data = load_selections()
            data.pop(key, None)
            save_selections(data)
            self.refresh_saved()
            if self.info is not None and self.info.key == key:
                self.timer.stop()
                self.close_capture()
                self.region = None
                self.status.setText(f"Forgot the saved area for {key}.")
                self.view.setPixmap(QPixmap())
                self.view.setText("nothing captured yet")

        def forget_all(self):
            if not load_selections():
                return
            if QMessageBox.question(self, "Forget all",
                                    "Delete every saved game area?") != \
                    QMessageBox.Yes:
                return
            save_selections({})
            self.refresh_saved()
            self.timer.stop()
            self.close_capture()
            self.region = None
            self.status.setText("All saved areas forgotten.")
            self.view.setPixmap(QPixmap())
            self.view.setText("nothing captured yet")

        def use_saved(self):
            key = self._selected_key()
            if not key:
                QMessageBox.information(
                    self, "No saved area",
                    "There are no saved game areas yet. Press Selection to "
                    "pick a window and drag one.")
                return
            info = self.find_window(key)
            if info is None:
                QMessageBox.warning(
                    self, "Window not open",
                    f"No open window looks like {key!r}. Open it, then press "
                    f"Use again - or press Selection to pick it by hand.")
                return
            self.status.setText(f"Opening {key} ...")
            self.activate(info, reuse_key=key)

        def find_window(self, key: str):
            exact = [w for w in win.list_windows() if w.key == key]
            if exact:
                return max(exact, key=lambda w: w.area)
            loose = [w for w in win.list_windows() if key.lower() in w.key.lower()]
            return max(loose, key=lambda w: w.area) if loose else None

        # -- pause / resume -----------------------------------------------
        def toggle_pause(self):
            """Freeze or restart the live read without losing the selection."""
            if self.dead or self.region is None:
                return
            if self.timer.isActive():
                self.timer.stop()
                self.btn_pause.setText("Resume")
                self.verdict.setText("paused - the image above is the last read")
            else:
                self.timer.start(max(1, self.cfg.fps))
                self.btn_pause.setText("Pause")

        # -- selection ----------------------------------------------------
        def do_selection(self):
            self.timer.stop()
            if self.finder is None:
                self.status.setText("No learned glyphs yet. Run: "
                                    "python -m pokervision learn")
                return
            dlg = WindowPicker(self)
            if dlg.exec() != QDialog.Accepted:
                return
            info = dlg.chosen()
            if info is None:
                return
            saved = load_selections().get(info.key)
            if saved:
                box = QMessageBox(self)
                box.setWindowTitle("Already saved")
                box.setText(f"An area is already saved for {info.key!r}.")
                box.setInformativeText("Use the saved area, or drag a new one?")
                use = box.addButton("Use saved", QMessageBox.AcceptRole)
                fresh = box.addButton("Drag a new one", QMessageBox.RejectRole)
                box.addButton("Cancel", QMessageBox.RejectRole)
                box.exec()
                clicked = box.clickedButton()
                if clicked is use:
                    self.activate(info, reuse_key=info.key)
                elif clicked is fresh:
                    self.activate(info, reuse_key=None)
                return
            self.activate(info, reuse_key=None)

        def activate(self, info, reuse_key):
            self.close_capture()
            try:
                cap = WindowCapture(self.cfg, info.hwnd, info.title)
                frame = cap.get_frame()
            except win.WindowLost as exc:
                self.status.setText(str(exc))
                return
            self.cap, self.info = cap, info
            if reuse_key:
                region = self._region_from_saved(frame, reuse_key)
                if region is None:
                    self.close_capture()
                    self.status.setText("The saved area no longer fits this window.")
                    return
                self.region = region
            else:
                picker = RegionPicker(frame, self)
                if picker.exec() != QDialog.Accepted or picker.rect is None:
                    self.close_capture()
                    self.status.setText("Selection cancelled.")
                    return
                self.region = picker.rect
                self._remember(frame)
            r = self.region
            l, t, cw, ch = self.cap.client
            self.status.setText(
                f"capturing window 0x{info.hwnd:X}  client {cw}x{ch} at "
                f"({l},{t})   |   game area {r.w}x{r.h} at ({r.x},{r.y}) "
                f"within it")
            self.dead = False
            self.btn_pause.setEnabled(True)
            self.btn_pause.setText("Pause")
            self.timer.start(max(1, self.cfg.fps))

        def _region_from_saved(self, frame, key: str):
            rec = load_selections().get(key)
            if not rec:
                return None
            fh, fw = frame.shape[:2]
            x0 = int(rec.get("fx", 0) * fw)
            y0 = int(rec.get("fy", 0) * fh)
            x1 = min(fw, x0 + int(rec.get("fw", 0) * fw))
            y1 = min(fh, y0 + int(rec.get("fh", 0) * fh))
            if x1 - x0 < MIN_DRAG or y1 - y0 < MIN_DRAG:
                return None
            return rc.Rect(x=x0, y=y0, w=x1 - x0, h=y1 - y0)

        def _remember(self, frame: np.ndarray):
            r = self.region
            fh, fw = frame.shape[:2]
            data = load_selections()
            data[self.info.key] = {
                "title": self.info.title,
                "fx": r.x / max(1, fw), "fy": r.y / max(1, fh),
                "fw": r.w / max(1, fw), "fh": r.h / max(1, fh),
            }
            save_selections(data)
            self.refresh_saved()

        def close_capture(self):
            if self.cap is not None:
                try:
                    self.cap.close()
                except Exception:
                    pass
            self.cap = None
            self.info = None

        # -- live loop ----------------------------------------------------
        def tick(self):
            if (self.dead or self.cap is None or self.region is None
                    or self.finder is None):
                return
            try:
                full = self.cap.get_frame()
            except win.WindowLost as exc:
                self.dead = True
                self.timer.stop()
                self.status.setText(f"STOPPED: {exc}")
                self.close_capture()
                self.verdict.setText("capture stopped")
                self.btn_pause.setEnabled(False)
                return
            r = self.region
            fh, fw = full.shape[:2]
            x0, y0 = max(0, r.x), max(0, r.y)
            x1, y1 = min(fw, r.right), min(fh, r.bottom)
            if x1 <= x0 or y1 <= y0:
                return
            sub = full[y0:y1, x0:x1]
            found = self.finder.find(sub)
            self._show(sub, found)

        def _show(self, sub, found):
            vis = sub.copy()
            board, hole = group_rows(found)
            board_rects = {c.rect for c in board}
            for c in found:
                x, y, w, h = c.rect.x, c.rect.y, c.rect.w, c.rect.h
                colour = (60, 220, 60) if c.rect in board_rects else (255, 190, 40)
                cv2.rectangle(vis, (x, y), (x + w, y + h), colour, 2)
                label = c.code or f"{c.rank or '?'}{c.suit or '?'}"
                cv2.putText(vis, label, (x, max(12, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
            pm = fit_pixmap(vis, self.view.width() - 8, self.view.height() - 8)
            if pm.width():
                self.view.setText("")
                self.view.setPixmap(pm)
            bc = " ".join(c.code for c in board) or "-"
            hc = " ".join(c.code for c in hole) or "-"
            title = win.console_safe(self.info.title) if self.info else ""
            self.status.setText(
                f"{title}    board [{bc}]    hole [{hc}]    "
                f"{len(found)} card{'s' if len(found) != 1 else ''}")
            self._show_confidence(found, board, hole)

        def _show_confidence(self, found, board, hole):
            """What is and is not trustworthy about the read above.

            Every card `CardFinder` returns is already named - `_read` returns
            None rather than a half-named card, so there is no "unnamed" state
            to report. What is left is the *row split*: `group_rows` decides
            which row of at most two cards is the hero's, and that is a
            geometric guess about layout, not a measurement. So the panel says
            how many cards were read and whether the split could be made at
            all, and says nothing it has not actually measured.
            """
            if not found:
                self.verdict.setStyleSheet("color: #bebebe;")
                self.verdict.setText(
                    "nothing found in the selected area - is the game area "
                    "right, and is the table in view?")
                self.verdict.setToolTip("")
                return
            if not hole:
                self.verdict.setStyleSheet("color: #ffd200;")
                self.verdict.setText(
                    f"read {len(found)} cards, all named, but no row of at "
                    f"most two - the hero's cards were not found")
                self.verdict.setToolTip(
                    "group_rows takes the lowest row of one or two cards as "
                    "the hole cards. With every row holding more than two, it "
                    "cannot tell which row is the hero's.")
                return
            partial = len(hole) != 2
            self.verdict.setStyleSheet(
                f"color: {'#ffd200' if partial else '#3cdc3c'};")
            self.verdict.setText(
                f"read {len(found)} card{'s' if len(found) != 1 else ''}, "
                f"all named"
                + (f" - {len(hole)}/2 hole cards, a partial read"
                   if partial else ""))
            self.verdict.setToolTip(
                "board: " + (" ".join(c.code for c in board) or "none")
                + "\nhole: " + " ".join(c.code for c in hole)
                + "\n\nread: " + "\n      ".join(
                    f"{c.code} at {c.rect.x},{c.rect.y}"
                    + (f", deskewed {c.angle:g} deg" if c.angle else "")
                    for c in found))

        def closeEvent(self, e):
            self.timer.stop()
            self.close_capture()
            super().closeEvent(e)


    app = QApplication.instance() or QApplication(sys.argv)
    w = MainWindow()
    w.show()
    return app.exec()


def run_gui(cfg: Config) -> int:
    if not cfg.templates.exists():
        print("No learned glyphs. Run:  "
              "python -m pokervision learn --images unmarked-images")
        return 1
    win.list_windows()  # also forces DPI awareness before any rect is read
    return _build_gui(cfg)
