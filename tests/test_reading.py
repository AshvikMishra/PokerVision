"""Reading and ReadingTracker: what one frame said, and when to believe it.

Replaces the old `test_decision.py`, whose other half tested advice that no
longer exists. The tracker half is kept because `watch` still depends on it -
it is the only thing standing between a one-frame misread and a printed card -
so deleting the file outright would have dropped coverage of live code.

Nothing here is a poker decision. A `Reading` is a description of pixels, and
`ReadingTracker` is a stability filter over those descriptions.
"""

from __future__ import annotations

import _boot  # noqa: F401,E402  - repo root on sys.path, cwd at root

import sys

import pokervision as pv
import pokervision.regions as rc
from pokervision.perception import Card

FAILURES: list[str] = []


def check(name: str, got, want) -> None:
    if got == want:
        print(f"  ok    {name}: got {got}")
    else:
        print(f"  FAIL  {name}: got {got!r}, want {want!r}")
        FAILURES.append(name)


def card(code: str, slot: int = 0) -> Card:
    return Card(rank=code[0], suit=code[1], rect=rc.Rect(slot * 90, 0, 80, 120),
                slot=slot, rank_score=1.0, rank_margin=1.0,
                suit_score=1.0, suit_margin=1.0)


def reading(hole, board=(), pot=None, stack=None) -> pv.Reading:
    return pv.Reading(hole=[card(c, i) for i, c in enumerate(hole)],
                      board=[card(c, i) for i, c in enumerate(board)],
                      pot=pot, hero_stack=stack)


def main() -> int:
    print("--- Reading: describing a frame ---")
    r = reading(["As", "Ah"], ["Ks", "Kd", "Qs"], pot=2000, stack=100000)
    check("hole codes", r.hole_codes, ["As", "Ah"])
    check("board codes", r.board_codes, ["Ks", "Kd", "Qs"])
    check("three board cards is a flop", r.street, "flop")
    check("describe names both rows",
          r.describe(), "hole [As Ah]  board [Ks Kd Qs]")
    check("money lists what was read", r.money(), ["pot 2,000", "stack 100,000"])
    check("a complete read says so", r.complete, True)
    check("no to_call field any more", hasattr(r, "to_call"), False)

    check("two hole cards and no board is preflop",
          reading(["As", "Ah"]).street, "preflop")
    check("and is complete", reading(["As", "Ah"]).complete, True)
    check("five board cards is a river",
          reading(["As", "Ah"], ["Ks", "Kd", "Qs", "Js", "Ts"]).street, "river")
    check("one hole card is not a complete read",
          reading(["As"]).complete, False)
    check("and it says how many are missing",
          reading(["As"]).problems,
          ["1/2 hole cards visible"])
    check("no cards at all reports nothing missing",
          reading([]).problems, ["0/2 hole cards visible"])
    check("an unnamed rank is reported as such",
          [p for p in reading([]).problems], ["0/2 hole cards visible"])

    broken = reading(["As", "Ah"])
    broken.hole[0].rank = None
    check("a read with an unnamed glyph is not complete", broken.complete, False)
    check("and names the glyph", len(broken.problems), 1)

    print("\n--- ReadingTracker: nothing is believed until it repeats ---")
    t = pv.ReadingTracker(stable_frames=3)
    seq = [
        (reading([]), "waiting"),
        (reading(["As", "Ah"]), "settling"),
        (reading(["As", "Ah"]), "settling"),
        (reading(["As", "Ah"]), "new-hand"),
        (reading(["As", "Ah"]), "same"),
        (reading(["As", "Ah"]), "same"),
    ]
    for r_in, want in seq:
        check(f"stable_frames=3: {want}", t.update(r_in), want)

    t = pv.ReadingTracker(stable_frames=2)
    t.update(reading(["As", "Ah"]))
    t.update(reading(["As", "Ah"]))
    check("committed", t.update(reading(["As", "Ah"])), "same")

    events = []
    for board in (["Ks"], ["Ks", "Qd"], ["Ks", "Qd", "Jh"]):
        events += [t.update(reading(["As", "Ah"], board))] * 2
    check("still one read after the flop", t.hand_no, 1)
    check("board growth is a deal, not a new hand",
          set(events) <= {"deal", "same", "settling"}, True)

    t = pv.ReadingTracker(stable_frames=2)
    for _ in range(2):
        t.update(reading(["As", "Ah"]))
    check("preflop committed", t.update(reading(["As", "Ah"])), "same")
    river = reading(["As", "Ah"], ["Ks", "Qd", "Jh", "2c", "3d"])
    check("a new board needs stable_frames reads", t.update(river), "settling")
    check("board growth to the river is a deal", t.update(river), "deal")
    check("and the river is then stable", t.update(river), "same")
    t.update(reading(["Ts", "9h"]))
    check("fresh hole cards is a new hand",
          t.update(reading(["Ts", "9h"])), "new-hand")
    check("the counter advanced", t.hand_no, 2)

    print("  a jump straight from a full river to a new flop, no empty frame")
    t = pv.ReadingTracker(stable_frames=2)
    for _ in range(3):
        t.update(reading(["As", "Ah"], ["Ks", "Qd", "Jh", "2c", "3d"]))
    t.update(reading(["2d", "3c"], ["9h", "8s", "7d"]))
    check("board shrank, so it is a new hand",
          t.update(reading(["2d", "3c"], ["9h", "8s", "7d"])), "new-hand")

    print("  an unmatched glyph suspends judgement")
    t = pv.ReadingTracker(stable_frames=2)
    t.update(reading(["As", "Ah"]))
    broken = reading(["As", "Ah"])
    broken.hole[0].rank = None
    check("unstable read ignored", t.update(broken), "unstable")
    check("and it forgets the candidate", t.candidate, None)

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURES: {FAILURES}")
        return 1
    print("\nall reading and stability tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
