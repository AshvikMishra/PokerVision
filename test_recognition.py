"""Recognition regression test.

The 15 screenshots in unmarked-images/ have hand-written ground truth (see
GROUND_TRUTH). This checks the learned glyph bank reads every card correctly
and reports the score margin the matching threshold has to work with.

    python test_recognition.py
"""

import glob
from pathlib import Path

import cv2
import numpy as np

import poker as poker

# file stem -> (board cards in slot order, hero cards)
GROUND_TRUTH = {
    "Screenshot (34)":            (["5s", "Ac", "7s", "Ad", "8h"], ["9h", "Ah"]),
    "Screenshot (35)":            (["6d", "4s", "9h", None, None], ["8d", "Th"]),
    "Screenshot (36)":            (["6d", "4s", "9h", "8s", None], ["8d", "Th"]),
    "Screenshot (37)":            (["5s", "Ts", "Jd", None, None], ["Td", "8d"]),
    "Screenshot (38)":            (["3c", "Kd", "Qc", None, None], ["Ah", "9c"]),
    "Screenshot (39)":            (["6s", "9d", "Jc", "Qd", "3d"], ["Js", "8c"]),
    "Screenshot (40)":            (["3h", "5h", "Jd", None, None], ["Qd", "Ah"]),
    "Screenshot (41)":            ([None] * 5,                 ["6h", "2c"]),
    "my-turn-hear-9-heart-A":     ([None] * 5,                 ["9h", "Ah"]),
    "my-turn-heart-9-heart-A-board-spade-5-clubs-A-spade-7-diamond-A-bet-options":
                                  (["5s", "Ac", "7s", "Ad", None], ["9h", "Ah"]),
    "my-turn-heart-9-heart-A-board-spade-5-clubs-A-spade-7-diamond-A":
                                  (["5s", "Ac", "7s", "Ad", None], ["9h", "Ah"]),
    "my-turn-heart-9-heart-A-board-spade-5-clubs-A-spade-7":
                                  (["5s", "Ac", "7s", None, None], ["9h", "Ah"]),
    "no-turn-heart-9-heart-A-board-spade-5-clubs-A-spade-7":
                                  (["5s", "Ac", "7s", None, None], ["9h", "Ah"]),
    "no-turn-heart-9-heart-A":    ([None] * 5,                 ["9h", "Ah"]),
    "waiting-for-round-to-finish": ([None] * 5,                [None, None]),
}


def collect_tagged(cfg):
    """(rank_samples, pip_samples) each with the true card code for that slot."""
    rank, pip = [], []
    for path in sorted(glob.glob("unmarked-images/*.png")):
        stem = Path(path).stem
        if stem not in GROUND_TRUTH:
            continue
        board, hero = GROUND_TRUTH[stem]
        frame = cv2.imread(path)
        cap = poker.Capture.offline()
        for i, r in enumerate(list(cfg.board_slots) + list(cfg.hero_slots)):
            truth = (board + hero)[i]
            crop = cap.crop(frame, r)
            if not poker.is_dealt(crop):
                if truth is not None:
                    print(f"  MISSED a dealt card: {stem} slot{i} = {truth}")
                continue
            if truth is None:
                print(f"  FALSE POSITIVE: {stem} slot{i} read as a card but is empty")
                continue
            rm = poker.pick_rank_glyph(crop, cfg)
            pm = poker.pick_corner_pip(crop, cfg)
            if rm is not None:
                rank.append((poker.rasterise(crop, rm), truth))
            else:
                print(f"  no rank glyph: {stem} slot{i} = {truth}")
            if pm is not None:
                pip.append((poker.rasterise(crop, pm), truth))
            else:
                print(f"  no corner pip: {stem} slot{i} = {truth}")
    return rank, pip


def separation(pairs, key):
    same, diff = [], []
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            s = poker.corr(pairs[i][0], pairs[j][0])
            (same if key(pairs[i][1]) == key(pairs[j][1]) else diff).append(s)
    return same, diff


def leave_one_out(pairs, key):
    """Nearest-neighbour accuracy: is each glyph's closest *other* glyph the
    same character? This is what the matcher actually does at runtime."""
    right, wrong, orphan = 0, [], []
    for i, (m, truth) in enumerate(pairs):
        k = key(truth)
        best, bj = 0.0, None
        for j, (mj, tj) in enumerate(pairs):
            if i == j:
                continue
            s = poker.corr(m, mj)
            if s > best:
                best, bj = s, j
        if bj is None:
            orphan.append((truth, "no other sample"))
            continue
        if key(pairs[bj][1]) == k:
            right += 1
        else:
            wrong.append((best, truth, pairs[bj][1]))
    wrong.sort()
    return right, wrong, orphan


def scored_against_truth(cfg):
    """Full-pipeline accuracy: every dealt card, read and compared to truth."""
    r_ok = r_bad = s_ok = s_bad = 0
    misses = []
    bank = poker.GlyphBank(cfg.templates, cfg.match_score).load()
    cap = poker.Capture.offline()
    slots = list(cfg.board_slots) + list(cfg.hero_slots)
    for path in sorted(glob.glob("unmarked-images/*.png")):
        stem = Path(path).stem
        if stem not in GROUND_TRUTH:
            continue
        board, hero = GROUND_TRUTH[stem]
        frame = cv2.imread(path)
        for i, r in enumerate(slots):
            truth = (board + hero)[i]
            if truth is None:
                continue
            window = cap.crop(frame, r)
            found = poker.locate_card(window)
            if found is None:
                r_bad += 1
                misses.append(f"RANK {stem[:40]:42s} slot{i} no card found")
                continue
            fx, fy, fw, fh = found
            crop = window[fy:fy + fh, fx:fx + fw]
            rank, _, _ = bank.match(poker.rasterise(crop, poker.pick_rank_glyph(crop, cfg)))
            suit, _, _ = bank.match(poker.rasterise(crop, poker.pick_corner_pip(crop, cfg)))
            if rank == truth[0]:
                r_ok += 1
            else:
                r_bad += 1
                misses.append(f"RANK {stem[:40]:42s} slot{i} got {rank} want {truth}")
            if suit == truth[1]:
                s_ok += 1
            else:
                s_bad += 1
                misses.append(f"SUIT {stem[:40]:42s} slot{i} got {suit} want {truth}")
    return r_ok, r_bad, s_ok, s_bad, misses


def main() -> int:
    cfg = poker.Config.load()
    r_ok, r_bad, s_ok, s_bad, misses = scored_against_truth(cfg)
    print(f"\nend-to-end vs ground truth:  rank {r_ok}/{r_ok + r_bad}"
          f"   suit {s_ok}/{s_ok + s_bad}")
    for m in misses:
        print("  MISS", m)

    rank, pip = collect_tagged(cfg)
    print(f"\n{len(rank)} rank glyphs, {len(pip)} pips, "
          f"ranks seen: {sorted({t[1][0] for t in rank})}")

    right, wrong, orphan = leave_one_out(rank, lambda t: t[0])
    print(f"\n--- rank: leave-one-out nearest neighbour ---")
    print(f"  correct {right}/{len(rank)}"
          f"  ({100.0 * right / max(1, len(rank)):.1f}%)")
    for s, a, b in wrong:
        print(f"  MISS {a} -> nearest was {b} (score {s:.3f})")
    for t, why in orphan:
        print(f"  MISS {t} ({why})")

    rp, rw, ro = leave_one_out(pip, lambda t: t[1])
    print(f"\n--- suit pip: leave-one-out nearest neighbour ---")
    print(f"  correct {rp}/{len(pip)} ({100.0 * rp / max(1, len(pip)):.1f}%)")
    for s, a, b in rw[:12]:
        print(f"  MISS {a} -> nearest was {b} (score {s:.3f})")
    for t, why in ro:
        print(f"  MISS {t} ({why})")

    print("\n--- score distribution (context for the threshold) ---")
    same, diff = separation(rank, lambda t: t[0])
    print(f"  rank  same: min={min(same):.3f} p05={np.percentile(same,5):.3f} "
          f"mean={np.mean(same):.3f}")
    print(f"  rank  diff: max={max(diff):.3f} p99={np.percentile(diff,99):.3f} "
          f"mean={np.mean(diff):.3f}  (distributions overlap; that is why the")
    print(f"  matcher keeps several exemplars per shape instead of one)")
    return 1 if (r_bad or s_bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
