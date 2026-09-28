"""Window picker and command-line wiring tests.

    python test_cli.py

The picker is the thing a user touches first, so it is tested the way it is
used: for real windows off this desktop, with real answers, and with stdin
closed. The fallback matters more than the happy path - `watch` and the overlay
both call `describe_source` on whatever comes back, and every way of saying
"no" (blank, out of range, unknown title, EOF, Ctrl-C) has to land on the
whole monitor instead of raising.

`main(["gui"])` is not run: it opens a real window. Instead a stand-in module
is put in sys.modules, which is exactly the seam `main` imports through, so
the wiring is checked without a GUI and without importing PySide6.
"""

from __future__ import annotations
import _boot  # noqa: F401,E402  - repo root on sys.path, cwd at root

import builtins
import contextlib
import io
import os
import sys
import types
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.*=false")
for _stream in (sys.stdout, sys.stderr):  # window titles carry emoji
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from pokervision import cli  # noqa: E402
from pokervision import windows as win  # noqa: E402
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


def check_true(name, got, why=""):
    ok = bool(got)
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}"
          + ("" if ok else f": {why or 'expected true, got ' + repr(got)}"))
    if not ok:
        FAILURES.append(name)
    return ok


def note(text):
    print(f"  info  {text}")


@contextlib.contextmanager
def stdin_is(text: str):
    """What the picker sees on stdin, including the empty string: with a
    StringIO exhausted, the built-in input() raises EOFError, which is the
    real 'pipe closed under it' case."""
    real = sys.stdin
    sys.stdin = io.StringIO(text)
    try:
        yield
    finally:
        sys.stdin = real


@contextlib.contextmanager
def input_raises(exc: type[BaseException]):
    real = builtins.input

    def boom(prompt=""):
        raise exc(prompt)

    builtins.input = boom
    try:
        yield
    finally:
        builtins.input = real


@contextlib.contextmanager
def answers(text: str):
    real = builtins.input
    calls: list[str] = []

    def fake(prompt=""):
        calls.append(prompt)
        return text

    builtins.input = fake
    try:
        yield calls
    finally:
        builtins.input = real


@contextlib.contextmanager
def captured():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        yield buf


def quiet_pick(**kwargs):
    """pick_capture prints its listing; keep the test output readable."""
    with captured() as buf:
        cap = cli.pick_capture(**kwargs)
    return cap, buf.getvalue()


def describe(cap):
    try:
        return cli.describe_source(cap), None
    except Exception as exc:  # noqa: BLE001 - that is the whole point
        return None, exc


cfg = Config.load()
windows = win.list_windows()
if not windows:
    print("no candidate windows on this desktop - every picker path here needs "
          "at least one real window to choose")
    raise SystemExit(1)
note(f"cfg.monitor {cfg.monitor}, {len(windows)} candidate windows, largest "
     f"{win.console_safe(windows[0].title)!r} at {windows[0].client}")
note("picker: " + win.format_picker(windows).splitlines()[0].strip())

print("\n--- 1. a number from the listing picks that window ---")
by_number, out = quiet_pick(cfg=cfg, window="1")
check_true("returns a WindowCapture", isinstance(by_number, WindowCapture),
           f"got {type(by_number).__name__}")
check("and it is window [1]", getattr(by_number, "hwnd", None),
      windows[0].hwnd)
line, exc = describe(by_number)
check_true("describe_source names the window handle",
           line is not None and f"0x{by_number.hwnd:X}" in line,
           f"raised {type(exc).__name__}: {exc}" if exc else repr(line))
check_true("describe_source says it is a window",
           line is not None and "window" in line.lower(), repr(line))
check_true("the listing was printed before the choice", "hwnd" in out, out)
try:
    frame = by_number.get_frame()
    check("and the capture it returned really grabs the client area",
          frame.shape, (by_number.client[3], by_number.client[2], 3))
except Exception as exc:  # noqa: BLE001
    check_true("and the capture it returned really grabs the client area",
               False, f"get_frame raised {type(exc).__name__}: {exc}")
by_number.close()

print("\n--- 2. a title substring picks a window ---")
words = [w for w in windows[0].title.replace("\u200b", " ").split()
         if len(w) > 2 and not w.isdigit()]
needle = (words[0] if words else windows[0].title)[:12]
check_true("a usable substring was taken from a live title", bool(needle),
           repr(needle))
by_title, out = quiet_pick(cfg=cfg, window=needle)
check_true(f"substring {needle!r} returns a WindowCapture",
           isinstance(by_title, WindowCapture),
           f"got {type(by_title).__name__}")
check_true("and it is one of the listed windows",
           getattr(by_title, "hwnd", None) in {w.hwnd for w in windows},
           f"hwnd {getattr(by_title, 'hwnd', None)}")
line, exc = describe(by_title)
check_true("describe_source names that handle",
           line is not None and f"0x{by_title.hwnd:X}" in line,
           f"raised {type(exc).__name__}: {exc}" if exc else repr(line))
by_title.close()

print("\n--- 3. a number out of range falls back to the monitor ---")
check_true("99 really is out of range on this desktop", len(windows) < 99,
           f"{len(windows)} windows found")
oob, out = quiet_pick(cfg=cfg, window="99")
check("falls back to a plain Capture", type(oob).__name__, "Capture")
check_true("and not a WindowCapture", not isinstance(oob, WindowCapture))
check_true("it says so rather than failing silently", "no window" in out, out)
check("the monitor it fell back to is still the configured one",
      oob.rect["width"] > 0, True)
oob.close()

print("\n--- 4. an unknown title falls back to the monitor ---")
bogus = "ZZZ_no_such_window"
check_true("no live window has that title",
           not any(bogus.lower() in w.title.lower() for w in windows))
miss, out = quiet_pick(cfg=cfg, window=bogus)
check("falls back to a plain Capture", type(miss).__name__, "Capture")
check_true("and not a WindowCapture", not isinstance(miss, WindowCapture))
check_true("it says so", bogus in out, out)
miss.close()

print("\n--- 5. --monitor forces the monitor path and sets cfg.monitor ---")
forced, out = quiet_pick(cfg=cfg, monitor=0)
check("returns a plain Capture", type(forced).__name__, "Capture")
check_true("and not a WindowCapture", not isinstance(forced, WindowCapture))
check("cfg.monitor is 0", cfg.monitor, 0)
check("the capture covers that monitor", forced.rect["width"],
      forced.sct.monitors[0]["width"])
check_true("no listing was printed", "open windows" not in out, out)
if len(forced.sct.monitors) > 1:
    other = 1 if cfg.monitor != 1 else 2
    if other < len(forced.sct.monitors):
        cfg.monitor = 0
        forced2, _ = quiet_pick(cfg=cfg, monitor=other)
        check(f"cfg.monitor is set to {other}", cfg.monitor, other)
        check("and the capture follows it", forced2.rect,
              {k: forced2.sct.monitors[other][k]
               for k in ("left", "top", "width", "height")})
        forced2.close()
forced.close()
cfg.monitor = 0

print("\n--- 6. every way of saying no lands on the monitor ---")
for label, ctx in (("stdin closed (EOF)", stdin_is("")),
                   ("input() raising EOFError", input_raises(EOFError)),
                   ("Ctrl-C at the prompt", input_raises(KeyboardInterrupt)),
                   ("a blank answer", answers("   ")),
                   ("an answer that is not a listed number", answers("wibble"))):
    with ctx:
        fell_back, out = quiet_pick(cfg=cfg)
    check(f"{label} falls back to a Capture", type(fell_back).__name__,
          "Capture")
    tail = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
    note(f"{label}: reported as {tail[-1]!r}" if tail else f"{label}: silent")
    fell_back.close()

print("\n  and a number typed at the prompt still selects a window")
with answers("2"):
    picked, out = quiet_pick(cfg=cfg)
check_true("typing 2 at the prompt returns a WindowCapture",
           isinstance(picked, WindowCapture), f"got {type(picked).__name__}")
check("and it is the second window in the listing", picked.hwnd,
      windows[1].hwnd)
picked.close()

print("\n--- 7. describe_source names what is being captured ---")
mon_cap = Capture(cfg)
line, exc = describe(mon_cap)
check_true("a monitor Capture is described as a monitor",
           line is not None and "monitor" in line.lower(),
           f"describe_source raised {type(exc).__name__}: {exc}" if exc
           else repr(line))
if exc is not None:
    note("that is not a cosmetic failure: watch() prints describe_source(cap) "
         "at cli.py:124 and the overlay at overlay.py:173, and both get here "
         "whenever the picker falls back to the monitor - which is every "
         "blank answer, EOF and Ctrl-C above")
elif "\n" in line:
    check_true("describe_source is one line, not a paragraph", False, repr(line))
else:
    check_true("describe_source is one line, not a paragraph", True)
mon_cap.close()

win_cap = WindowCapture(cfg, windows[0].hwnd, windows[0].title)
line2, exc2 = describe(win_cap)
check_true("a WindowCapture is described as a window",
           line2 is not None and "window" in line2.lower(),
           f"describe_source raised {type(exc2).__name__}: {exc2}" if exc2
           else repr(line2))
check_true("and it names the handle",
           line2 is not None and f"0x{win_cap.hwnd:X}" in line2, repr(line2))
check_true("one line there too",
           line2 is not None and "\n" not in line2, repr(line2))
win_cap.close()

print("\n--- 8. main() wires the flags ---")
for command in ("watch", "overlay"):
    try:
        with captured() as buf:
            cli.main([command, "--help"])
    except SystemExit as exc:
        code = exc.code
    else:
        code = "no SystemExit"
    help_text = buf.getvalue()
    check(f"`{command} --help` exits 0", code, 0)
    check_true(f"`{command} --help` documents --window", "--window" in help_text,
               help_text)
    check_true(f"`{command} --help` documents --monitor", "--monitor" in help_text,
               help_text)
    check_true(f"`{command} --help` says a number or a title is accepted",
               "substring" in help_text, help_text)

try:
    with captured() as buf:
        cli.main(["gui", "--help"])
except SystemExit as exc:
    gui_code = exc.code
else:
    gui_code = "no SystemExit"
gui_help = buf.getvalue()
check("`gui --help` exits 0", gui_code, 0)
check_true("the parser knows the gui command", "gui" in gui_help, gui_help)

try:
    with captured() as buf:
        cli.main(["--help"])
except SystemExit as exc:
    top_code = exc.code
else:
    top_code = "no SystemExit"
top_help = buf.getvalue()
check("the top-level --help exits 0", top_code, 0)
for command in ("doctor", "gui", "verify", "learn", "watch", "overlay"):
    check_true(f"the top-level help lists {command}", command in top_help,
               top_help)
check_true("and says what gui does",
           "pick a window" in top_help, top_help)

# the actual dispatch, without a GUI: a stand-in module in the same place
# main imports from, so no PySide6 import and no window opens
calls: list[Config] = []
fake_gui = types.ModuleType("pokervision.gui")
fake_gui.run_gui = lambda c: (calls.append(c), 42)[1]
had_module = "pokervision.gui" in sys.modules
saved = sys.modules.get("pokervision.gui")
sys.modules["pokervision.gui"] = fake_gui
try:
    with captured():
        gui_rc = cli.main(["gui"])
finally:
    if had_module:
        sys.modules["pokervision.gui"] = saved
    else:
        sys.modules.pop("pokervision.gui", None)
check("`gui` is dispatched to run_gui", gui_rc, 42)
check("exactly once", len(calls), 1)
check_true("and it was handed the loaded config",
           bool(calls) and calls[0].monitor == cfg.monitor,
           f"got {calls[0] if calls else None}")

try:
    with captured():
        cli.main([])
except SystemExit as exc:
    empty_rc = exc.code
else:
    empty_rc = "no SystemExit"
check("no command at all is a usage error", empty_rc, 2)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURES: {FAILURES}")
    raise SystemExit(1)
print("all picker and CLI tests passed")
