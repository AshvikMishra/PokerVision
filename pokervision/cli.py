from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import cv2

from . import regions as rc
from . import windows as win
from .config import Config
from .capture import Capture, WindowCapture
from .perception import CardReader, GlyphBank
from .state import Reading, ReadingTracker, read_frame


def _use_window(cfg: Config, chosen) -> WindowCapture:
    print(f"capturing {win.console_safe(chosen.title)}")
    print(f"  hwnd 0x{chosen.hwnd:X}, pid {chosen.pid}")
    print(f"  client rect {chosen.client[2]}x{chosen.client[3]} "
          f"at {chosen.client[:2]}  (chrome excluded)")
    return WindowCapture(cfg, chosen.hwnd, chosen.title)


def describe_source(cap) -> str:
    """One line saying exactly what is being captured, so there is never any
    doubt about whether the window or the whole monitor is in use."""
    if isinstance(cap, WindowCapture):
        l, t, w, h = cap.client
        return f"window 0x{cap.hwnd:X}  client {w}x{h} at ({l},{t})"
    return f"monitor {cap.monitor}  {cap.rect}"


def pick_capture(cfg: Config, window: str | None = None,
                 monitor: int | None = None) -> Capture:
    """Window capture by default, whole monitor as the fallback.

    `window` is a number from the listing or a substring of a title. `monitor`
    forces the old path. With neither, the picker runs; a blank answer, a bad
    number, EOF or Ctrl-C all fall back to the monitor rather than failing.
    """
    if monitor is not None:
        cfg.monitor = monitor
        return Capture(cfg)

    if window is not None and window.isdigit():
        found = win.list_windows()
        index = int(window)
        if not 1 <= index <= len(found):
            print(f"no window [{index}] - {len(found)} found. "
                  f"Using monitor {cfg.monitor}.")
            return Capture(cfg)
        return _use_window(cfg, found[index - 1])

    if window:
        matches = [w for w in win.list_windows()
                   if window.lower() in w.title.lower()]
        if not matches:
            print(f"no window title contains {window!r}. "
                  f"Using monitor {cfg.monitor}.")
            return Capture(cfg)
        return _use_window(cfg, max(matches, key=lambda w: w.area))

    found = win.list_windows()
    if not found:
        print(f"no candidate windows found - using the whole monitor "
              f"({cfg.monitor}).")
        return Capture(cfg)

    print("\nopen windows, largest first:")
    print(win.format_picker(found))
    print(f"\npick the number of the game window, or press enter to use the "
          f"whole monitor ({cfg.monitor}):")
    try:
        answer = input("  > ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        answer = ""
    if answer.isdigit() and 1 <= int(answer) <= len(found):
        return _use_window(cfg, found[int(answer) - 1])
    if answer:
        print(f"  {answer!r} is not one of the listed numbers - using the "
              f"whole monitor ({cfg.monitor}).")
    else:
        print(f"using the whole monitor ({cfg.monitor}).")
    return Capture(cfg)


def report(r: Reading, source: str, event: str | None = None,
           frame_no: int | None = None) -> None:
    head = f"--- {source}"
    if frame_no:
        head += f"  read {frame_no}"
    if event and event != "same":
        head += f"  [{event}]"
    print(head + " ---")
    print(f"  {r.describe()}   [{r.street}]")
    money = r.money()
    if money:
        print("  " + "   ".join(money))
    for p in r.problems:
        print(f"  ! {p}")


def watch(cfg: Config, window: str | None = None,
          monitor: int | None = None) -> None:
    if not cfg.templates.exists():
        print("No learned glyphs yet. Run:  "
              "python -m pokervision learn --images unmarked-images")
        return
    cap = pick_capture(cfg, window, monitor)
    print(f"source  : {describe_source(cap)}")
    reader = CardReader(cfg, GlyphBank(cfg.templates, cfg.match_score).load())
    tracker = ReadingTracker()
    period = 1.0 / max(1, cfg.fps)
    print(f"watching at {cfg.fps} fps - Ctrl-C to stop.")
    print("prints when the read changes, not once per frame.\n")
    try:
        while True:
            t0 = time.perf_counter()
            r = read_frame(reader, cap.get_frame(), cap, cfg)
            event = tracker.update(r)
            if event in ("new-hand", "deal", "unstable", "waiting"):
                report(r, time.strftime("%H:%M:%S"), event, tracker.hand_no)
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))
    except win.WindowLost as exc:
        print(f"\nSTOPPED: {exc}")
        return
    except KeyboardInterrupt:
        print("\nstopped")
        print(f"distinct reads seen: {tracker.hand_no}")
    finally:
        cap.close()


def cmd_doctor(cfg: Config) -> int:
    import importlib
    for mod in ("cv2", "numpy", "mss", "yaml"):
        try:
            m = importlib.import_module(mod)
            print(f"  {mod:7} ok  {getattr(m, '__version__', '')}")
        except ImportError:
            print(f"  {mod:7} MISSING")
    try:
        cap = Capture(cfg)
        print(f"  capture monitor {cfg.monitor} -> {cap.rect}")
        cap.close()
    except Exception as e:
        print(f"  capture failed: {e}")
    missing = cfg.missing()
    print("\nregions still to calibrate:" if missing else "\nall card regions set")
    for m in missing:
        print(f"  - {m}")
    print(f"glyphs learned: {'yes' if cfg.templates.exists() else 'no - run `python -m pokervision learn`'}")
    return 0


def cmd_verify(cfg: Config, pattern: str) -> int:
    bank = GlyphBank(cfg.templates, cfg.match_score).load()
    if not bank.ready:
        print("No learned glyphs. Run:  "
              "python -m pokervision learn --images unmarked-images")
        return 1
    reader = CardReader(cfg, bank)
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"no images matched {pattern}")
        return 1
    print(f"knowing {sorted(set(bank.labels))}\n")
    for path in paths:
        frame = cv2.imread(path)
        r = read_frame(reader, frame, Capture.offline(), cfg)
        print(f"  {Path(path).name:<56} {r.describe()}")
        for p in r.problems:
            print(f"      ! {p}")
    return 0


def _source_args(p: argparse.ArgumentParser) -> None:
    """Capture source for the live commands. With neither, the window
    picker runs and the whole monitor is the fallback."""
    p.add_argument("--window", default=None, metavar="N|TEXT",
                   help="capture this window: a number from the picker, "
                        "or a substring of its title")
    p.add_argument("--monitor", type=int, default=None, metavar="N",
                   help="skip the picker and capture mss monitor N")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m pokervision",
        description="PokerVision: read cards off the Monopoly Poker table "
                    "(screen capture and OpenCV only).")
    ap.add_argument("--config", default=str(rc.DEFAULT_CONFIG))
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="check deps and calibration state")

    sub.add_parser("gui", help="pick a window, drag the game area, watch it read")

    v = sub.add_parser("verify", help="read the marked screenshots")
    v.add_argument("--images", default="unmarked-images/*.png")

    l = sub.add_parser("learn", help="cluster the glyph shapes and label them")
    l.add_argument("--images", default="unmarked-images/*.png")
    l.add_argument("--ranks", default=None,
                   help="skip the prompt, one label per cluster found")
    l.add_argument("--suits", default=None, help="skip the prompt, e.g. --suits 's h d c'")
    l.add_argument("--digits", default=None,
                   help="skip the prompt, e.g. --digits '0 1 2 3 4 5 6 7 8 9 K'")

    w = sub.add_parser("watch", help="live loop: read the table")
    _source_args(w)

    o = sub.add_parser("overlay", help="click-through overlay: boxes and reads")
    _source_args(o)

    a = ap.parse_args(argv)
    cfg = Config.load(a.config)
    if a.cmd == "doctor":
        return cmd_doctor(cfg)
    if a.cmd == "gui":
        from .gui import run_gui
        return run_gui(cfg)
    if a.cmd == "verify":
        return cmd_verify(cfg, a.images)
    if a.cmd == "learn":
        from .learning import cmd_learn
        return cmd_learn(cfg, a.images,
                         a.ranks.split() if a.ranks else None,
                         a.suits.split() if a.suits else None,
                         a.digits.split() if a.digits else None)
    if a.cmd == "watch":
        watch(cfg, a.window, a.monitor)
        return 0
    if a.cmd == "overlay":
        from .overlay import cmd_overlay_gui
        return cmd_overlay_gui(cfg, a.window, a.monitor)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
