"""GUI tests: the drag box, the saved areas, and what the display shows.

Headless, on Qt's offscreen platform, so no poker game is needed:

    python test_gui.py

`pokervision.gui` builds `RegionPicker` and `MainWindow` as closures inside
`_build_gui`, so they are reached the way a debugger would reach them:
intercept `QApplication.exec` while `_build_gui` is on the stack, harvest
`QApplication.topLevelWidgets()`, and pull the classes out of a bound method's
closure cells. The real `CardFinder` is replaced by a stub that hands back a
hand the test wrote down, so nothing here depends on what happens to be on
screen. Only the capture-source checks need a real window, and they say so and
fall back to a fake capture if this machine has none.

`config/selections.json` is the saved game areas for this machine, so it is
deleted at the start and at the end: a test that leaves a test area behind is a
test that has changed the user's machine.
"""

from __future__ import annotations
import _boot  # noqa: F401,E402  - repo root on sys.path, cwd at root

import os

# Before anything pulls in Qt: the whole run is offscreen, so no window is ever
# created on the desktop and no mouse is ever really grabbed.
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import contextlib  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402

warnings.simplefilter("ignore", DeprecationWarning)

import numpy as np  # noqa: E402
from PySide6.QtCore import (QEvent, QPoint, QPointF, QRect, Qt, QtMsgType,  # noqa: E402
                            QTimer, qInstallMessageHandler)
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import (QApplication, QDialog, QMessageBox,  # noqa: E402
                               QPushButton)

import pokervision.regions as rc  # noqa: E402
import pokervision.windows as win  # noqa: E402
from pokervision import gui  # noqa: E402
from pokervision.capture import WindowCapture  # noqa: E402
from pokervision.config import Config  # noqa: E402
from pokervision.detect import Detected, group_rows  # noqa: E402

SELECTIONS = gui.SELECTIONS

FAILURES = []


def check(name, got, want, tol=None):
    if tol is None:
        ok = got == want
    else:
        ok = got is not None and abs(got - want) <= tol
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}: got {_brief(got)}"
          + ("" if ok else f", want {_brief(want)}"))
    if not ok:
        FAILURES.append(name)
    return ok


def check_near(name, got, want, tol):
    """The same, for a rect or a pair of pixels."""
    ok = (got is not None and len(got) == len(want)
          and all(abs(g - w) <= tol for g, w in zip(got, want)))
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}: got {_brief(got)}"
          + ("" if ok else f", want {_brief(want)} (+-{tol})"))
    if not ok:
        FAILURES.append(name)
    return ok


def info(text):
    print(f"  info  {text}")


def _brief(v, n=96):
    s = repr(v)
    return s if len(s) <= n else s[:n - 3] + "..."


# Qt's own noise: the offscreen platform cannot grab the mouse, cannot
# propagate size hints and has no font directory. All three are expected here
# and none of them say anything about whether the GUI works.
def _qt_message(mode, _context, message):
    if mode >= QtMsgType.QtCriticalMsg:
        print(f"  qt    {getattr(QtMsgType(mode), 'name', mode)}: {message}")


qInstallMessageHandler(_qt_message)

# ---------------------------------------------------------------- harness


def closure_class(fn, name):
    """A class defined in a closure, reached the way a debugger would."""
    for cell in (fn.__closure__ or ()):
        value = cell.cell_contents
        if isinstance(value, type) and value.__name__ == name:
            return value
    return None


def build_gui(cfg):
    """Run `_build_gui` just far enough to construct the main window."""
    found = {}
    real_exec = QApplication.exec

    def fake_exec(self=None):
        for widget in QApplication.topLevelWidgets():
            if type(widget).__name__ == "MainWindow":
                found["mw"] = widget
        return 0

    QApplication.exec = fake_exec
    try:
        gui._build_gui(cfg)
    finally:
        QApplication.exec = real_exec
    return found["mw"]


def pump(app, rounds=3):
    for _ in range(rounds):
        app.processEvents()


def synthetic(h, w):
    """A frame with enough structure that it is obviously not one flat colour."""
    frame = np.zeros((h, w, 3), np.uint8)
    frame[:, :, 0] = np.linspace(20, 200, w, dtype=np.uint8)[None, :]
    frame[:, :, 1] = np.linspace(200, 20, h, dtype=np.uint8)[:, None]
    frame[:, :, 2] = 90
    frame[h // 4:max(h // 4 + 1, h // 2),
          w // 4:max(w // 4 + 1, w // 2)] = (240, 240, 240)
    return frame


def detected(codes, x0, y, w, h, gap=0):
    """Real `Detected` cards: `rank`, `suit`, `rect`. `angle` stays 0."""
    return [Detected(rank=code[0], suit=code[1],
                     rect=rc.Rect(x=x0 + i * (w + gap), y=y, w=w, h=h))
            for i, code in enumerate(codes)]


class StubFinder:
    """Stands in for CardFinder: returns a hand the test wrote down."""

    def __init__(self, cards):
        self.cards = list(cards)
        self.calls = 0
        self.sizes = []

    def find(self, frame):
        self.calls += 1
        self.sizes.append(None if frame is None else frame.shape[:2])
        return list(self.cards)


class FakeCap:
    """A `WindowCapture` stand-in with a frame and a client rect on paper."""

    def __init__(self, frame, client, hwnd=0):
        self.frame = frame
        self.client = client
        self.hwnd = hwnd
        self.origin = (client[0], client[1])
        self.rect = {"left": client[0], "top": client[1],
                     "width": client[2], "height": client[3]}
        self.reads = 0
        self.closed = False

    def get_frame(self):
        self.reads += 1
        return self.frame

    def close(self):
        self.closed = True


@contextlib.contextmanager
def fake_capture(frame, client):
    """Point the GUI's capture at a frame of our choosing."""
    real = gui.WindowCapture

    def factory(_cfg, hwnd, _title=None, **_kw):
        return FakeCap(frame, client, hwnd)

    gui.WindowCapture = factory
    try:
        yield
    finally:
        gui.WindowCapture = real


@contextlib.contextmanager
def only_window(info):
    """Make the window list hold exactly one window.

    The window picker enumerates what is really on the desktop, so without
    this the end-to-end flow would depend on which window happened to be
    largest, and on a snapshot big enough to be scaled down - which is a
    different bug (see section 2) and would hide this flow behind it.
    """
    real = win.list_windows
    win.list_windows = lambda include_hidden=False: [info]
    try:
        yield
    finally:
        win.list_windows = real


def readout_percent(text):
    """The `66% wide, 60% tall` the readout carries, if it has them.

    It used to read `(66% x 4000% of the window)`: the second number came from
    `frame.shape[2]`, which is the channel count, and the literal `%` after it
    was missing. The wording is now width-percent then height-percent.
    """
    m = re.search(r"(\d+)% wide,\s*(\d+)% tall", text)
    return (int(m.group(1)), int(m.group(2))) if m else None


def button_press(dlg, text):
    """Click a push button by its label, the way a user would."""
    for child in dlg.findChildren(QPushButton):
        if text.lower() in child.text().lower():
            child.click()
            return True
    return False


def rows_text(list_widget):
    return " | ".join(list_widget.item(i).text()
                      for i in range(list_widget.count()))


class Drag:
    """press / move / release over a RegionPicker.

    Every point is given in LABEL pixels - the coordinate the user thinks in,
    and the one `picker.rect` is derived from - and translated into whichever
    widget the event is delivered to. That is what makes the awkward cases
    testable at all: a release past the edge of the image is just a label point
    past the edge, and a press on the dialog's margin is just a label point
    before the top-left corner.

    The press defaults to the label and the rest of the drag to the dialog,
    because `RegionPicker` grabs the mouse on the dialog: in a real drag every
    event after the press arrives at the grabber, whatever is underneath.
    """

    PRESS = QEvent.Type.MouseButtonPress
    MOVE = QEvent.Type.MouseMove
    RELEASE = QEvent.Type.MouseButtonRelease

    def __init__(self, app, picker):
        self.app = app
        self.picker = picker

    def _send(self, widget, kind, pt, button, buttons):
        if widget is self.picker.label:
            local = QPoint(int(pt[0]), int(pt[1]))
        else:
            local = widget.mapFromGlobal(
                self.picker.label.mapToGlobal(QPoint(int(pt[0]), int(pt[1]))))
        self.app.sendEvent(widget, QMouseEvent(
            kind, QPointF(local), QPointF(local), button, buttons,
            Qt.KeyboardModifier.NoModifier))

    def press(self, pt, on="label"):
        self._send(self.picker if on == "dialog" else self.picker.label,
                   self.PRESS, pt, Qt.MouseButton.LeftButton,
                   Qt.MouseButton.LeftButton)

    def move(self, pt, to="dialog"):
        self._send(self.picker if to == "dialog" else self.picker.label,
                   self.MOVE, pt, Qt.MouseButton.NoButton,
                   Qt.MouseButton.LeftButton)

    def release(self, pt, to="dialog"):
        self._send(self.picker if to == "dialog" else self.picker.label,
                   self.RELEASE, pt, Qt.MouseButton.LeftButton,
                   Qt.MouseButton.NoButton)

    def drag(self, a, b, on="label", to="dialog", via=None):
        self.press(a, on=on)
        if via is not None:
            self.move(via, to=to)
        self.move(b, to=to)
        self.release(b, to=to)


class ModalPilot:
    """Plays the GUI's own modal dialogs from a timer inside their event loop.

    `QDialog.exec()` spins a nested event loop, so the only way to test a flow
    the GUI drives itself is to act from inside that loop. The pilot looks at
    whatever dialog is up, matches it by class name, and runs the script
    registered for it. A watchdog closes anything still up after a few seconds
    and records it, so a mis-scripted flow fails instead of hanging.
    """

    LIMIT = 240  # 15ms apart, so ~3.6s of nothing happening

    def __init__(self, app):
        self.app = app
        self.plan = {}
        self.done = set()
        self.messages = []
        self.forced = []
        self.answer = "ok"
        self.armed = False
        self._ticks = 0
        self.timer = QTimer()
        self.timer.setInterval(15)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    def reset(self):
        self.done.clear()
        self.messages.clear()
        self.forced.clear()
        self._ticks = 0

    def arm(self, answer="ok"):
        self.reset()
        self.answer = answer
        self.armed = True

    def disarm(self):
        self.armed = False

    def on(self, name, fn):
        self.plan[name] = fn
        return self

    def saw(self, name):
        return name in self.done

    def said(self):
        return [title for title, _ in self.messages[:1]]

    def _tick(self):
        if not self.armed:
            return
        dlg = QApplication.activeModalWidget()
        if dlg is None:
            for widget in QApplication.topLevelWidgets():
                if (isinstance(widget, QDialog) and widget.isVisible()
                        and type(widget).__name__ in self.plan):
                    dlg = widget
                    break
        if dlg is None:
            self._ticks = 0
            return
        name = type(dlg).__name__
        if name in self.done:
            return
        if name == "QMessageBox":
            self.messages.append((dlg.windowTitle(), dlg.text()))
            button_press(dlg, self.answer)
            return
        fn = self.plan.get(name)
        if fn is not None and fn(dlg):
            self.done.add(name)
        self._ticks += 1
        if self._ticks > self.LIMIT:
            self.forced.append(name)
            dlg.reject()


# ---------------------------------------------------------------- set-up

print("--- set-up: the offscreen app, the main window, the closure classes ---")
if SELECTIONS.exists():
    SELECTIONS.unlink()
info(f"cleared {SELECTIONS}")

cfg = Config.load()
app = QApplication.instance() or QApplication(sys.argv)
mw = build_gui(cfg)
MainWindow = type(mw)
RegionPicker = closure_class(mw.activate, "RegionPicker")
WindowPicker = closure_class(mw.do_selection, "WindowPicker")
check("MainWindow built", mw is not None, True)
check("RegionPicker reached through the closure of activate",
      RegionPicker is not None, True)
check("WindowPicker reached through the closure of do_selection",
      WindowPicker is not None, True)
check("the learned glyphs are loaded, so the GUI is not in its dead state",
      mw.finder is not None, True)
check("Pause is disabled before anything is selected", mw.btn_pause.isEnabled(),
      False)
check("and Selection is not", mw.btn_select.isEnabled(), True)
check("config/selections.json is gone to begin with", SELECTIONS.exists(), False)
pilot = ModalPilot(app)
pump(app)

# A snapshot small enough to be shown 1:1, so one label pixel is one client
# pixel. A downscaled snapshot is section 2's problem.
FRAME = synthetic(200, 300)
FH, FW = FRAME.shape[:2]


def new_picker(frame=FRAME):
    picker = RegionPicker(frame, mw)
    picker.show()
    pump(app)
    return picker


# ---------------------------------------------------------------- 1. drags
print("\n--- 1. the drag, in every shape it can take ---")
p = new_picker()
drag = Drag(app, p)
check("the label is pinned to the pixmap",
      (p.label.width(), p.label.height()),
      (p.label.pixmap().width(), p.label.pixmap().height()))
check("a small snapshot is not scaled down", p._scale(), (1.0, 1.0))
check("nothing selected yet", p.rect, None)
check("the readout says so", p.readout.text(), "nothing selected yet")
check("Use this box is disabled until a box exists", p.ok.isEnabled(), False)

drag.drag((40, 30), (240, 150), via=(150, 90))
pump(app)
check("A  press+move+release inside the image", p.rect,
      rc.Rect(40, 30, 200, 120))
check("A  ok is enabled", p.ok.isEnabled(), True)
check("A  the readout names the rect",
      "200x120 at (40,30)" in p.readout.text(), True)
check("A  the readout prints two percentages", p.readout.text().count("%"), 2)
check("A  and both are of the window", readout_percent(p.readout.text()),
      (200 * 100 // FW, 120 * 100 // FH))
check("A  the band is still up after the release", p.band.isVisible(), True)
check("A  the band is where the drag went", p.band.geometry(),
      QRect(40, 30, 200, 120))

drag.drag((40, 30), (240, 150), to="label")
pump(app)
check("A2 the same drag delivered to the label instead", p.rect,
      rc.Rect(40, 30, 200, 120))
check("A2 the band is still up", p.band.isVisible(), True)

drag.drag((20, 20), (900, 700))
pump(app)
check("B  press inside, release outside the image", p.rect,
      rc.Rect(20, 20, FW - 20, FH - 20))
check("B  clamped to the client, not to the image",
      p.rect.right <= FW and p.rect.bottom <= FH, True)
check("B  ok is enabled", p.ok.isEnabled(), True)
check("B  the band is still up", p.band.isVisible(), True)

drag.drag((60, 40), (160, 140), on="dialog")
pump(app)
check("C  a press that arrives on the dialog, not the label", p.rect,
      rc.Rect(60, 40, 100, 100))
check("C  ok is enabled", p.ok.isEnabled(), True)
check("C  the band is still up", p.band.isVisible(), True)

# beside the image: the hint text sits above it, so a press there is a label
# point with a negative y. The top of the box clamps to the top of the window
# and the bottom stays where the pointer was let go.
margin = QPoint(p.label.x() + 30, 4)
lp = p.label.mapFrom(p, margin)
info(f"a press on the hint text, {margin.x()},{margin.y()} in the dialog, "
     f"is label pixel {lp.x()},{lp.y()} - above the image")
drag.press((lp.x(), lp.y()), on="dialog")
drag.move((100, 60))
drag.release((200, 100))
pump(app)
check("D  a press beside the image still starts a drag", p.rect is not None, True)
check("D  and comes out inside the window",
      0 <= p.rect.x and 0 <= p.rect.y and p.rect.right <= FW
      and p.rect.bottom <= FH, True)
check("D  ok is enabled", p.ok.isEnabled(), True)
check("D  the band is still up", p.band.isVisible(), True)
info(f"   the box came out {p.rect}")

drag.drag((100, 100), (119, 119))
pump(app)
check("E  a 19x19 drag is below MIN_DRAG and is rejected", p.rect, None)
check("E  ok goes back to disabled", p.ok.isEnabled(), False)
check("E  the readout says why", "too small" in p.readout.text().lower(), True)
check("E  the band follows even a rejected drag", p.band.geometry(),
      QRect(100, 100, 19, 19))
check("E  the band does not vanish on release", p.band.isVisible(), True)

drag.drag((100, 100), (120, 120))
pump(app)
check("F  exactly MIN_DRAG is accepted", p.rect, rc.Rect(100, 100, 20, 20))
check("F  ok is enabled again", p.ok.isEnabled(), True)

drag.drag((50, 50), (250, 170))
pump(app)
check("G  a good drag after a bad one", p.rect, rc.Rect(50, 50, 200, 120))
check("G  ok is enabled", p.ok.isEnabled(), True)
drag.drag((200, 150), (205, 155))
pump(app)
check("G  a too-small drag after a good one is rejected", p.rect, None)
check("G  and puts ok back to disabled", p.ok.isEnabled(), False)
check("G  the band is still up", p.band.isVisible(), True)
p.close()

# ---------------------------------------------------------------- 2. mapping
print("\n--- 2. the mapping from the drag to client pixels ---")
p = new_picker()
drag = Drag(app, p)
drag.drag((10, 15), (110, 195))
pump(app)
check("a drag from (10,15) to (110,195) is Rect(10,15,100,180)", p.rect,
      rc.Rect(10, 15, 100, 180))
drag.drag((110, 195), (10, 15))
pump(app)
check("the same box dragged the other way round", p.rect,
      rc.Rect(10, 15, 100, 180))
drag.drag((0, 0), (FW, FH))
pump(app)
check("a drag over the whole image is the whole client rect", p.rect,
      rc.Rect(0, 0, FW, FH))
drag.drag((10, 10), (10_000, 10_000))
pump(app)
check("a release far outside is clamped to the client", p.rect,
      rc.Rect(10, 10, FW - 10, FH - 10))
p.close()

# a snapshot too big for the screen, so the dialog scales it down - which is
# what every real game window does
BIG = synthetic(1080, 1920)
BFH, BFW = BIG.shape[:2]
p = new_picker(BIG)
lw, lh = p.label.width(), p.label.height()
info(f"a {BFW}x{BFH} client is shown as a {lw}x{lh} label and _scale() "
     f"reports {p._scale()}")
check("this snapshot really was scaled down", lw < BFW and lh < BFH, True)
drag = Drag(app, p)
a, b = (lw // 4, lh // 4), (lw * 3 // 4, lh * 3 // 4)
drag.drag(a, b)
pump(app)
check_near("a drag over the middle quarter lands on the middle quarter of the "
           "window", (p.rect.x, p.rect.y),
           (a[0] * BFW // lw, a[1] * BFH // lh), 2)
check_near("... and is the right size in client pixels", (p.rect.w, p.rect.h),
           ((b[0] - a[0]) * BFW // lw, (b[1] - a[1]) * BFH // lh), 2)
check("... and the readout's fractions are of the unscaled window",
      readout_percent(p.readout.text()),
      (p.rect.w * 100 // BFW, p.rect.h * 100 // BFH))
p.close()

# ---------------------------------------------------------------- 3. accept
print("\n--- 3. the button that accepts the box ---")
p = new_picker()
drag = Drag(app, p)
drag.drag((30, 25), (230, 145))
pump(app)
check("the drag gave a box", p.rect, rc.Rect(30, 25, 200, 120))
check("ok is enabled", p.ok.isEnabled(), True)
check("the dialog has not accepted anything yet", int(p.result()),
      int(QDialog.DialogCode.Rejected))
p.ok.click()
pump(app)
check("clicking Use this box accepts the dialog", int(p.result()),
      int(QDialog.DialogCode.Accepted))
check("the rect survives the accept", p.rect, rc.Rect(30, 25, 200, 120))
check("and the dialog is down", p.isVisible(), False)
p.close()

# ... and with nothing dragged the click must not accept. It pops a message
# box, so the pilot is armed to dismiss it rather than leaving it on screen.
p = new_picker()
pilot.on("QMessageBox", lambda dlg: False)
check("Use is inert until something is dragged", p.ok.isEnabled(), False)
pilot.arm()
p.ok.click()
pump(app)
check("clicking it while inert does not accept", int(p.result()),
      int(QDialog.DialogCode.Rejected))
check("and pops nothing at all", pilot.said(), [])
check("nothing was selected", p.rect, None)
pilot.disarm()
p.close()

# ---------------------------------------------------------------- 4. saved
print("\n--- 4. saved game areas survive a restart ---")
if SELECTIONS.exists():
    SELECTIONS.unlink()

real = None
for candidate in win.list_windows():
    try:
        cap = WindowCapture(cfg, candidate.hwnd, candidate.title)
        frame = cap.get_frame()
        cap.close()
    except Exception:  # a window that cannot be grabbed is not usable
        continue
    if frame.size and frame.shape[0] > 200 and frame.shape[1] > 200:
        real = (candidate, frame)
        break

if real is None:
    info("no grabbable window on this machine: the capture checks use a fake "
         "capture instead")
    INFO = win.WindowInfo(0x5A17, "Poker Bot Test Table - synthetic",
                          (0, 0, 400, 340), (0, 0, 400, 300), 4242)
    RFRAME = synthetic(300, 400)
else:
    INFO, RFRAME = real
    l, t, cw, ch = INFO.client
    info(f"using the real window 0x{INFO.hwnd:X} "
         f"{win.console_safe(INFO.title)[:40]!r}, client {cw}x{ch} at ({l},{t})")


def open_ctx():
    return contextlib.nullcontext() if real else fake_capture(RFRAME, INFO.client)


RFH, RFW = RFRAME.shape[:2]
REGION = rc.Rect(x=RFW // 20, y=RFH // 10, w=RFW // 2, h=RFH // 3)
mw.cap = FakeCap(RFRAME, INFO.client, INFO.hwnd)
mw.info = INFO
mw.region = REGION
mw._remember(RFRAME)

record = gui.load_selections().get(INFO.key, {})
check("the selection is keyed by the window", record.get("title"), INFO.title)
check("stored as a fraction of the width", record.get("fx"), REGION.x / RFW,
      tol=1e-12)
check("stored as a fraction of the height", record.get("fy"), REGION.y / RFH,
      tol=1e-12)
check("stored as a fraction of the width", record.get("fw"), REGION.w / RFW,
      tol=1e-12)
check("stored as a fraction of the height", record.get("fh"), REGION.h / RFH,
      tol=1e-12)
check("and it is the file on disk", SELECTIONS.exists(), True)
check("the list shows one row", mw.saved_list.count(), 1)
row = mw.saved_list.currentItem()
check("a row is selected without being asked", row is not None, True)
check("the row is the right one", row.data(Qt.UserRole), INFO.key)
check("the row names the window and its fractions",
      INFO.key in row.text() and "of the window" in row.text()
      and "%" in row.text(), True)

fresh = MainWindow()
check("a freshly built MainWindow re-reads the file", fresh.saved_list.count(), 1)
check("and selects it", fresh._selected_key(), INFO.key)
check("but has opened nothing yet", fresh.cap, None)
check("and has no region yet", fresh.region, None)

with open_ctx():
    fresh.btn_use.click()
use_status, use_region, use_cap = fresh.status.text(), fresh.region, fresh.cap
use_timer = fresh.timer.isActive()
use_pause = fresh.btn_pause.isEnabled()
fresh.timer.stop()
pump(app)

check("Use reopens the window rather than the monitor",
      use_cap is not None and isinstance(use_cap, gui.WindowCapture), True)
check("the window it opened is the one that was saved", fresh.info.key, INFO.key)
check("the capture is anchored to the client rect, not to the monitor",
      (use_cap.origin, use_cap.rect["left"], use_cap.rect["top"],
       use_cap.rect["width"], use_cap.rect["height"]),
      ((use_cap.client[0], use_cap.client[1]), use_cap.client[0],
       use_cap.client[1], use_cap.client[2], use_cap.client[3]))
check("and the frame loop is running", use_timer, True)
# Pause is the second button now: inert until `activate` has something live to
# pause, which is exactly what `use_saved` has just done.
check("... and the read can be paused, so Pause is live", use_pause, True)
# the saved fractions are the contract between the JSON and the window, so the
# area has to come back in the window's own pixels and stay put when that
# window is resized
saved_fractions = (record["fx"], record["fy"], record["fw"], record["fh"])
check_near("the region comes back as the saved fraction of that window",
           (use_region.x / use_cap.client[2], use_region.y / use_cap.client[3],
            use_region.w / use_cap.client[2], use_region.h / use_cap.client[3]),
           saved_fractions, 0.002)
for grown in (1.25, 0.75):
    with open_ctx():
        fresh.btn_use.click()
        grown_region, grown_client = fresh.region, fresh.cap.client
    fresh.timer.stop()
    check_near(f"the same area in a window {int(grown * 100)}% the size",
               (grown_region.x / grown_client[2],
                grown_region.y / grown_client[3],
                grown_region.w / grown_client[2],
                grown_region.h / grown_client[3]),
               saved_fractions, 0.002)

fresh.btn_forget.click()
check("Forget removes the file", SELECTIONS.exists(), False)
check("and empties the list", "none saved yet" in rows_text(fresh.saved_list),
      True)
check("the capture is closed", fresh.cap, None)
check("the region is gone", fresh.region, None)
check("the frame loop is stopped", fresh.timer.isActive(), False)
check("and it says which area it forgot", INFO.key in fresh.status.text(), True)

# forget all, which asks first
fresh.cap, fresh.info, fresh.region = (FakeCap(RFRAME, INFO.client, INFO.hwnd),
                                       INFO, REGION)
fresh._remember(RFRAME)
check("the area is back on disk for the next case", SELECTIONS.exists(), True)
pilot.arm(answer="yes")
fresh.btn_forget_all.click()
pump(app)
check("Forget all asks before deleting", pilot.said(), ["Forget all"])
check("Forget all removes the file", SELECTIONS.exists(), False)
check("Forget all empties the list",
      "none saved yet" in rows_text(fresh.saved_list), True)
check("Forget all closes the capture", fresh.cap, None)
check("Forget all stops the frame loop", fresh.timer.isActive(), False)
check("and nothing had to be force-closed", pilot.forced, [])
pilot.disarm()
fresh.close()

# ------------------------------------------------- 5. what the display shows
print("\n--- 5. what the display shows for a read ---")
DFRAME = synthetic(900, 1000)
DREGION = rc.Rect(20, 20, 960, 860)
show = MainWindow()
show.show()
pump(app)
# The panel under the image is a confidence readout, not advice: it reports what
# was measured - how many cards were named, and whether `group_rows` could make
# the row split at all. There is no advisor to install and nothing is simulated,
# so every case here costs nothing and there is no money in it.
show.cap = FakeCap(DFRAME, (0, 0, 1000, 900), 0x5A17)
show.info = INFO


def read(cards, label):
    """One tick, and everything it was supposed to put on screen."""
    show.finder = StubFinder(cards)
    show.region = DREGION
    show.tick()
    st, vd, tip = show.status.text(), show.verdict.text(), show.verdict.toolTip()
    rows = group_rows(cards)
    info(f"{label}: status {st[:88]!r}")
    info(f"{label}: verdict {vd[:88]!r}")
    return st, vd, tip, rows


# The board is one row high up, the hole cards a lower row. `group_rows`
# clusters by y-centre within `ROW_TOLERANCE * median_height` and takes the
# lowest row of at most two as the hero's, so these two rows are the only
# layout the whole section depends on.
board5 = detected(["5s", "Ac", "7s", "Ad", "8h"], 140, 40, 100, 130, gap=40)
board3 = detected(["5s", "Ac", "7s"], 200, 40, 100, 130, gap=40)
hole2 = detected(["9h", "Ah"], 380, 540, 100, 150, gap=40)
hole1 = detected(["9h"], 420, 540, 100, 150)
# five cards, all on the same y at the same height, and nothing below them: one
# row of five, so no row holds at most two and the hero's cards cannot be
# picked out of it
norow = board5

st, vd, tip, rows = read(hole2 + board5, "river")
check("the crop handed to the finder is the game area", show.finder.sizes[0],
      (DREGION.h, DREGION.w))
check("the status names the window", win.console_safe(INFO.title) in st, True)
check("the status spells out the board", "board [5s Ac 7s Ad 8h]" in st, True)
check("the status spells out the hole", "hole [9h Ah]" in st, True)
check("the status counts the cards", "7 cards" in st, True)
check("the row split puts the low pair in the hole cards",
      [c.code for c in rows[1]], ["9h", "Ah"])
check("... and the five above in the board, left to right",
      [c.code for c in rows[0]], ["5s", "Ac", "7s", "Ad", "8h"])
check("a complete read counts its cards", vd, "read 7 cards, all named")
check("... and says every card was named", "all named" in vd, True)
check("... and does not call it partial", "partial" in vd, False)
check("the tooltip lists the board it was given", "board: 5s Ac 7s Ad 8h" in tip,
      True)
check("... and the hole cards", "hole: 9h Ah" in tip, True)
check("... and where each one was read", "9h at 380,540" in tip, True)
check("the boxes were drawn on the view",
      show.view.pixmap() is not None and not show.view.pixmap().isNull(), True)

st, vd, tip, rows = read(hole2 + board3, "flop")
check("the status counts a flop", "5 cards" in st, True)
check("a flop is still a complete read", vd, "read 5 cards, all named")
check("and still not a partial one", "partial" in vd, False)

st, vd, tip, rows = read(hole2, "preflop")
check("a preflop read shows no board", "board [-]" in st, True)
check("... and two cards", "hole [9h Ah]" in st, True)
check("two named hole cards is a complete read, so no partial wording",
      "partial" in vd, False)
check("... and no 1/2 wording either", "1/2 hole cards" in vd, False)
check("... and it says so positively", vd, "read 2 cards, all named")

st, vd, tip, rows = read(hole1, "one hole card")
check("a partial read shows what it did read", "hole [9h]" in st, True)
check("and counts one card", "1 card" in st, True)
check("... in the singular", "1 cards" in st, False)
check("... and no board", "board [-]" in st, True)
check("one hole card is a partial read", vd,
      "read 1 card, all named - 1/2 hole cards, a partial read")
check("the reason names the shortfall", "1/2 hole cards" in vd, True)
check("and calls it a partial read", "partial read" in vd, True)
check("the tooltip still says which card it was", "hole: 9h" in tip, True)

st, vd, tip, rows = read(norow, "one row of five")
# asserted against `group_rows` rather than assumed, because the whole point of
# this case is the split failing, and the split is geometry
check("one row of five leaves no hero row to find", rows[1], [])
check("... so the board is all of them",
      [c.code for c in rows[0]], ["5s", "Ac", "7s", "Ad", "8h"])
check("a row split that cannot be made says so", vd,
      "read 5 cards, all named, but no row of at most two - the hero's "
      "cards were not found")
check("... and says the cards themselves were fine", "all named" in vd, True)
check("the tooltip explains the rule it could not apply",
      "group_rows" in tip and "more than two" in tip, True)

st, vd, tip, rows = read([], "nothing found")
check("an empty read shows no cards at all",
      "board [-]    hole [-]    0 cards" in st, True)
check("an empty read says nothing was found", "nothing found" in vd, True)
check("... and claims nothing about a row split", "hero" in vd, False)
check("... and attaches no explanation it cannot back up", tip, "")
check("the finder was asked exactly once", show.finder.calls, 1)
show.timer.stop()
show.close()

# ------------------------------------------------ 6. the capture source, named
print("\n--- 6. the status says what is being captured ---")
# the text `Use` left behind, read before any frame had a chance to overwrite it
l, t, cw, ch = use_cap.client
check("the window handle is named", f"0x{use_cap.hwnd:X}" in use_status, True)
check("the client rect is named", f"{cw}x{ch}" in use_status, True)
check("and where it sits on the desktop", f"({l},{t})" in use_status, True)
check("the game area inside it is named",
      f"{use_region.w}x{use_region.h} at ({use_region.x},{use_region.y})"
      in use_status, True)
check("so a user cannot mistake the monitor for the window",
      "client" in use_status and "game area" in use_status, True)
info(f"status: {use_status}")

# ------------------------------------------- 7. the Selection button, end to end
print("\n--- 7. pressing Selection, all the way to a running frame loop ---")
SELECTIONS.unlink(missing_ok=True)
SINFO = win.WindowInfo(0x5A17, "Poker Bot Test Table - synthetic",
                       (0, 0, 400, 340), (0, 0, 400, 300), 4242)
SFRAME = synthetic(300, 400)


def choose_window(dlg):
    return button_press(dlg, "OK")


def drag_box(dlg):
    inner = Drag(app, dlg)
    inner.drag((30, 25), (230, 145), on="dialog")
    pump(app)
    dlg.ok.click()
    return True


pilot.on("WindowPicker", choose_window)
pilot.on("RegionPicker", drag_box)
pilot.arm()
# one window in the list, and a frame of its client's size, so the whole flow
# is the same every run and needs nothing real on the screen
with only_window(SINFO), fake_capture(SFRAME, SINFO.client):
    mw.btn_select.click()
    sel_timer = mw.timer.isActive()
    sel_pause = (mw.btn_pause.isEnabled(), mw.btn_pause.text())
sel_status, sel_region = mw.status.text(), mw.region
sel_saved = gui.load_selections()
mw.timer.stop()
pilot.disarm()
check("the window list was opened and answered", pilot.saw("WindowPicker"), True)
check("the drag dialog was opened and answered", pilot.saw("RegionPicker"), True)
check("nothing had to be force-closed", pilot.forced, [])
check("the dragged box was kept", sel_region, rc.Rect(30, 25, 200, 120))
check("and saved against the chosen window",
      sel_saved.get(SINFO.key, {}).get("title"), SINFO.title)
check("as a fraction of that window", sel_saved[SINFO.key]["fw"], 0.5, tol=1e-12)
check("and a fraction of its height", sel_saved[SINFO.key]["fh"], 0.4, tol=1e-12)
check("the list shows it", mw.saved_list.count(), 1)
check("and the frame loop started", sel_timer, True)
check("... and Pause is now live and reads Pause, having been pressed as a "
      "fresh selection", sel_pause, (True, "Pause"))
check("the status names the handle, the client and the game area",
      f"0x{SINFO.hwnd:X}" in sel_status
      and "client 400x300 at (0,0)" in sel_status
      and "game area 200x120 at (30,25)" in sel_status, True)
info(f"status: {sel_status}")

# a window that already has an area is offered back, not silently re-dragged
with only_window(SINFO), fake_capture(SFRAME, SINFO.client):
    pilot.arm(answer="use saved")
    mw.btn_select.click()
    reuse_region, reuse_timer = mw.region, mw.timer.isActive()
mw.timer.stop()
pilot.disarm()
check("a window that already has an area says so", pilot.said(),
      ["Already saved"])
# the same box, to within the float round trip through the saved fractions
check_near("and the saved area was used",
           (reuse_region.x, reuse_region.y, reuse_region.w, reuse_region.h),
           (sel_region.x, sel_region.y, sel_region.w, sel_region.h), 1)
check("with the frame loop running", reuse_timer, True)

# --------------------------------------------------------------- tidy up
print("\n--- tidy up ---")
for widget in QApplication.topLevelWidgets():
    if isinstance(widget, QDialog):
        widget.reject()
    elif isinstance(widget, QMessageBox):
        widget.close()
mw.timer.stop()
if SELECTIONS.exists():
    SELECTIONS.unlink()
check("config/selections.json is gone again", SELECTIONS.exists(), False)
info(f"deleted {SELECTIONS}")

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURES: {FAILURES}")
    raise SystemExit(1)
print("all GUI tests passed")
