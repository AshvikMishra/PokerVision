from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import cv2

import regions_config as rc
from .config import Config
from .capture import Capture
from .perception import CardReader, GlyphBank
from .state import Hand, HandTracker, build_state
from .decision import Advice, advise


def report(hand: Hand, adv: Advice | None, source: str, event: str | None = None,
           hand_no: int | None = None) -> None:
    head = f"--- {source}"
    if hand_no:
        head += f"  hand {hand_no}"
    if event and event != "same":
        head += f"  [{event}]"
    print(head + " ---")
    print(f"  {hand.describe()}   [{hand.street}]")
    money = []
    if hand.pot is not None:
        money.append(f"pot {hand.pot:,}")
    if hand.hero_stack is not None:
        money.append(f"stack {hand.hero_stack:,}")
    if hand.to_call is not None:
        money.append(f"to call {hand.to_call:,}")
    if money:
        print("  " + "   ".join(money))
    for p in hand.problems:
        print(f"  ! {p}")
    if adv:
        print(f"  >> {adv.action}: {adv.reason}")
        if adv.equity is not None:
            extra = f"   pot odds {adv.pot_odds:.1%}" if adv.pot_odds else ""
            print(f"     equity {adv.equity:.1%}{extra}")


def watch(cfg: Config, sims: int) -> None:
    if not cfg.templates.exists():
        print("No learned glyphs yet. Run:  python poker.py learn --images unmarked-images")
        return
    cap = Capture(cfg)
    reader = CardReader(cfg, GlyphBank(cfg.templates, cfg.match_score).load())
    tracker = HandTracker()
    period = 1.0 / max(1, cfg.fps)
    print(f"watching at {cfg.fps} fps - Ctrl-C to stop. This build never clicks.")
    print("advice is printed once per decision point, not once per frame.\n")
    try:
        while True:
            t0 = time.perf_counter()
            hand = build_state(reader, cap.get_frame(), cap, cfg)
            event = tracker.update(hand)
            # postflop needs street state this build does not keep, and a
            # wrong suggestion there is worse than none
            interesting = event in ("new-hand", "deal", "unstable", "waiting")
            preflop = not hand.board_codes
            adv = advise(hand, sims) if (interesting and preflop) else None
            if interesting or adv:
                report(hand, adv, time.strftime("%H:%M:%S"), event, tracker.hand_no)
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        print("\nstopped")
        print(f"hands seen: {tracker.hand_no}")
    finally:
        cap.close()


def cmd_doctor(cfg: Config) -> int:
    import importlib
    for mod in ("cv2", "numpy", "mss", "yaml", "treys"):
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
    print(f"glyphs learned: {'yes' if cfg.templates.exists() else 'no - run `poker.py learn`'}")
    return 0


def cmd_verify(cfg: Config, pattern: str) -> int:
    bank = GlyphBank(cfg.templates, cfg.match_score).load()
    if not bank.ready:
        print("No learned glyphs. Run:  python poker.py learn --images unmarked-images")
        return 1
    reader = CardReader(cfg, bank)
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"no images matched {pattern}")
        return 1
    print(f"knowing {sorted(set(bank.labels))}\n")
    for path in paths:
        frame = cv2.imread(path)
        hand = build_state(reader, frame, Capture.offline(), cfg)
        print(f"  {Path(path).name:<56} {hand.describe()}")
        for p in hand.problems:
            print(f"      ! {p}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Monopoly Poker table assistant (observe only).")
    ap.add_argument("--config", default=str(rc.DEFAULT_CONFIG))
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="check deps and calibration state")

    v = sub.add_parser("verify", help="read the marked screenshots")
    v.add_argument("--images", default="unmarked-images/*.png")

    l = sub.add_parser("learn", help="cluster the glyph shapes and label them")
    l.add_argument("--images", default="unmarked-images/*.png")
    l.add_argument("--ranks", default=None,
                   help="skip the prompt, one label per cluster found")
    l.add_argument("--suits", default=None, help="skip the prompt, e.g. --suits 's h d c'")
    l.add_argument("--digits", default=None,
                   help="skip the prompt, e.g. --digits '0 1 2 3 4 5 6 7 8 9 K'")

    w = sub.add_parser("watch", help="live loop: observe and advise")
    w.add_argument("--sims", type=int, default=20000)

    o = sub.add_parser("overlay", help="click-through overlay: boxes, reads, advice")
    o.add_argument("--sims", type=int, default=20000)

    a = ap.parse_args(argv)
    cfg = Config.load(a.config)
    if a.cmd == "doctor":
        return cmd_doctor(cfg)
    if a.cmd == "verify":
        return cmd_verify(cfg, a.images)
    if a.cmd == "learn":
        from .learning import cmd_learn
        return cmd_learn(cfg, a.images,
                         a.ranks.split() if a.ranks else None,
                         a.suits.split() if a.suits else None,
                         a.digits.split() if a.digits else None)
    if a.cmd == "watch":
        watch(cfg, a.sims)
        return 0
    if a.cmd == "overlay":
        from .overlay import cmd_overlay_gui
        return cmd_overlay_gui(cfg, a.sims)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
