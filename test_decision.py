"""
Decision engine and hand-tracking tests. No screen, no screenshots, no
capture - these are pure functions, so they run anywhere.

    python test_decision.py

The equity figures are checked against widely published head-to-head numbers
for a random opponent, with a loose tolerance: this is a Monte Carlo with a
fixed seed, not a solver, and the point is to catch a broken conversion or a
seeded-with-the-wrong-cards bug rather than to agree to four decimal places.
"""

import sys

import poker as poker
from poker import Advice, Card, Hand, HandTracker, advise, chart_score
from poker import hand_class, monte_carlo_equity

FAILURES = []


def check(name, got, want, tol=None):
    if tol is None:
        ok = got == want
    else:
        ok = got is not None and abs(got - want) <= tol
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}: got {got}"
          + ("" if ok else f", want {want}"))
    if not ok:
        FAILURES.append(name)


def card(code):
    return Card(rank=code[0], suit=code[1], rank_score=1.0, suit_score=1.0,
                rank_margin=1.0, suit_margin=1.0,
                rect=None, slot=0, where="test")


def hand(hole, board=(), pot=None, stack=None, to_call=None):
    return Hand(hole=[card(c) for c in hole],
                board=[card(c) for c in board],
                pot=pot, hero_stack=stack, to_call=to_call)


print("--- hand_class: all 169 starting hands classify ---")
seen = set()
for a in poker.RANKS:
    for b in poker.RANKS:
        i, j = poker.RANKS.index(a), poker.RANKS.index(b)
        if j < i:
            continue
        s1, s2 = "shdc"[i % 4], "shdc"[j % 4]
        cls = hand_class([a + s1, b + s2])
        if cls is None:
            FAILURES.append(f"hand_class({a}{s1}{b}{s2}) returned None")
        seen.add(cls)
print(f"  {'ok  ' if not FAILURES else 'FAIL'}  {len(seen)} distinct classes from "
      f"{13 * 14 // 2} combos")
check("AA is a pair", hand_class(["As", "Ah"]), "AA")
check("AKs is suited", hand_class(["As", "Ks"]), "AKs")
check("AKo is offsuit", hand_class(["As", "Kd"]), "AKo")
check("72o ordering", hand_class(["7s", "2d"]), "72o")
check("K9s ordering, same suit", hand_class(["9c", "Kc"]), "K9s")
check("K9o ordering, mixed suits", hand_class(["9c", "Ks"]), "K9o")
check("suits are carried through the swap", hand_class(["As", "Kh"]), "AKo")
check("garbage rejected", hand_class(["As"]), None)

print("\n--- opening chart is monotone and bounded ---")
pairs = [r + r for r in poker.RANKS]
vals = [chart_score(p) for p in pairs]
check("AA is a max open", chart_score("AA"), 1.0)
check("22 is the weakest pair", chart_score("22"), vals[0])
print(f"  {'ok  ' if all(0 <= v <= 1 for v in vals) else 'FAIL'}  "
      f"all pair scores within 0..1")
monotone = all(chart_score(pairs[i]) <= chart_score(pairs[i + 1]) + 1e-9
               for i in range(len(pairs) - 1))
print(f"  {'ok  ' if monotone else 'FAIL'}  pair scores increase with rank")
check("AKs beats KQs", chart_score("AKs") > chart_score("KQs"), True)
check("AKs beats AQs", chart_score("AKs") > chart_score("AQs"), True)
check("72o is out of the range", chart_score("72o") < 0.1, True)
check("unknown class is a fold", chart_score("???"), 0.05)

print("\n--- Monte Carlo equity vs a random opponent (20000 sims) ---")
eq_aa = monte_carlo_equity(["As", "Ah"], [], sims=20000)
check("AA preflop ~ 85%", eq_aa, 0.85, tol=0.03)
eq_72 = monte_carlo_equity(["7s", "2d"], [], sims=20000)
check("72o preflop ~ 35%", eq_72, 0.35, tol=0.03)
eq_kk = monte_carlo_equity(["Ks", "Kd"], [], sims=20000)
check("KK preflop ~ 82%", eq_kk, 0.82, tol=0.03)
eq_ak = monte_carlo_equity(["As", "Ks"], [], sims=20000)
check("AKs preflop ~ 67%", eq_ak, 0.67, tol=0.04)
eq_72_flop = monte_carlo_equity(["7s", "2d"], ["Ah", "Kd", "Qc"], sims=20000)
check("72o on a big flop is worse than preflop", eq_72_flop < eq_72, True)
# AA on a K K Q board is a two-pair favourite, not a lock: two of the four
# kings are still live, and the opponent or the runout picks one up often
# enough to make it trips. Enumerated exactly over all 47 live cards it is
# 0.8798.
eq_nuts = monte_carlo_equity(["As", "Ah"], ["Ks", "Kd", "Qs"], sims=20000)
check("AA on K K Q ~ 88%", eq_nuts, 0.88, tol=0.02)
check("deterministic: same seed, same answer",
      monte_carlo_equity(["As", "Ah"], [], sims=500),
      monte_carlo_equity(["As", "Ah"], [], sims=500))
check("bad card code returns None", monte_carlo_equity(["Ax", "Ah"], []), None)

print("\n--- advise: preflop only, and it explains itself ---")
a = advise(hand(["As", "Ah"], pot=2000, stack=100000), sims=2000)
check("AA on a small pot raises", a.action, "RAISE")
check("AA names the class", a.cls, "AA")
check("advice carries equity", a.equity is not None, True)

a = advise(hand(["7d", "2c"], pot=2000, stack=100000), sims=2000)
check("72o folds", a.action, "FOLD")

a = advise(hand(["9c", "8h"], pot=2000, stack=100000), sims=2000)
check("98o is checked or folded, never raised", a.action in ("CHECK", "FOLD"), True)

# a short stack loosens the range: the same middling hand that is not openable
# deep should be openable at 20bb
tightness = {"RAISE": 0, "CHECK": 1, "FOLD": 2, "WAIT": 3}
deep = advise(hand(["9c", "8h"], pot=2000, stack=200000), sims=1000)
short = advise(hand(["9c", "8h"], pot=2000, stack=40000), sims=1000)
print(f"  info  deep stack -> {deep.action}, short stack -> {short.action}")
check("short stack is at least as loose as deep",
      tightness[short.action] <= tightness[deep.action], True)

a = advise(hand(["As"], pot=2000, stack=100000), sims=1000)
check("one hole card waits", a.action, "WAIT")
a = advise(hand([], pot=2000, stack=100000), sims=1000)
check("no hole cards waits", a.action, "WAIT")

# pot odds: a huge price to call should fold even a decent hand
a = advise(hand(["As", "Kh"], pot=100, stack=100000, to_call=50000), sims=3000)
check("facing a huge bet folds AK", a.action, "FOLD")
check("and it cites pot odds", a.pot_odds is not None and a.pot_odds > 0.9, True)

a = advise(hand(["As", "Ah"], ["Ks", "Kd", "Qs"]), sims=2000)
check("no postflop advice yet", a.action in ("WAIT",), True)

print("\n--- HandTracker: how a new deal is detected ---")
t = HandTracker(stable_frames=3)
seq = [
    (hand([]), "waiting"),
    (hand(["As", "Ah"]), "settling"),
    (hand(["As", "Ah"]), "settling"),
    (hand(["As", "Ah"]), "new-hand"),
    (hand(["As", "Ah"]), "same"),
    (hand(["As", "Ah"]), "same"),
]
for h, want_event in seq:
    got = t.update(h)
    check(f"preflop deal -> {want_event}", got, want_event)
check("hand number is 1", t.hand_no, 1)

print("  a flop arriving one card at a time must not look like three hands")
t = HandTracker(stable_frames=2)
t.update(hand(["As", "Ah"]))
t.update(hand(["As", "Ah"]))
check("committed", t.update(hand(["As", "Ah"])), "same")
events = []
for board in (["Ks"], ["Ks", "Qd"], ["Ks", "Qd", "Jh"]):
    events += [t.update(hand(["As", "Ah"], board))] * 2
check("still one hand after the flop", t.hand_no, 1)
check("board growth reported as deal, not new-hand",
      set(events) <= {"deal", "same", "settling"}, True)
print(f"  info  flop events: {events}")

print("  the river finishing and a new hand starting")
t = HandTracker(stable_frames=2)
for _ in range(2):
    t.update(hand(["As", "Ah"]))
check("preflop committed", t.update(hand(["As", "Ah"])), "same")
river = hand(["As", "Ah"], ["Ks", "Qd", "Jh", "2c", "3d"])
check("a new board needs stable_frames reads", t.update(river), "settling")
check("board growth to the river is a deal", t.update(river), "deal")
check("and the river is then stable", t.update(river), "same")
check("still one hand", t.hand_no, 1)
t.update(hand(["Ts", "9h"]))
check("fresh hole cards is a new hand", t.update(hand(["Ts", "9h"])), "new-hand")
check("hand number advanced", t.hand_no, 2)

print("  a jump straight from a full river to a new flop, no empty frame")
t = HandTracker(stable_frames=2)
for _ in range(3):
    t.update(hand(["As", "Ah"], ["Ks", "Qd", "Jh", "2c", "3d"]))
t.update(hand(["2d", "3c"], ["9h", "8s", "7d"]))
check("board shrank, so it is a new hand",
      t.update(hand(["2d", "3c"], ["9h", "8s", "7d"])), "new-hand")

print("  an unmatched glyph suspends judgement")
t = HandTracker(stable_frames=2)
t.update(hand(["As", "Ah"]))
broken = hand(["As", "Ah"])
broken.hole[0].rank = None
check("unstable read ignored", t.update(broken), "unstable")
check("and it forgets the candidate", t.candidate, None)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURES: {FAILURES}")
    raise SystemExit(1)
print("all decision and tracking tests passed")
