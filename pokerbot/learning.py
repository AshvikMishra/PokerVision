from __future__ import annotations

import glob
from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .capture import Capture
from .perception import (GlyphBank, corr, is_dealt, pick_corner_pip,
                         pick_rank_glyph, rasterise, segment_money)


def collect_samples(cfg: Config, pattern: str):
    rank_samples, rank_where, pip_samples, pip_where = [], [], [], []
    digit_samples, digit_where = [], []
    for path in sorted(glob.glob(pattern)):
        frame = cv2.imread(path)
        if frame is None:
            continue
        cap = Capture.offline()
        name = Path(path).name
        for i, r in enumerate(list(cfg.board_slots) + list(cfg.hero_slots)):
            crop = cap.crop(frame, r)
            if not is_dealt(crop):
                continue
            rm = pick_rank_glyph(crop, cfg)
            if rm is not None:
                rank_samples.append(rasterise(crop, rm))
                rank_where.append(f"{name}:{i}")
            pm = pick_corner_pip(crop, cfg)
            if pm is not None:
                pip_samples.append(rasterise(crop, pm))
                pip_where.append(f"{name}:{i}")
        for label, r in (("pot", cfg.pot), ("stack", cfg.hero_stack),
                         ("to_call", cfg.to_call)):
            if r is None:
                continue
            crop = cap.crop(frame, r)
            if crop.size == 0:
                continue
            for raster in segment_money(crop):
                digit_samples.append(raster)
                digit_where.append(f"{name}:{label}")
    return ((rank_samples, rank_where), (pip_samples, pip_where),
            (digit_samples, digit_where))


def _ask(kind: str, count: int, example: str) -> list[str] | None:
    print(f"\n  {count} distinct {kind} shapes found. Open the montage, read them off"
          f" left to right, and type one label each ({example}).")
    print(f"  There is no need for exactly 13 rank groups: if two shapes are the"
          f" same character, give them the same label twice.")
    try:
        raw = input("  > ").strip()
    except EOFError:
        return None
    got = raw.split()
    if len(got) != count:
        print(f"  got {len(got)} labels for {count} shapes - aborting")
        return None
    return got


def dump_montage(masks: list[np.ndarray], path: Path) -> None:
    if not masks:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = 7
    rows = []
    for i in range(0, len(masks), cols):
        chunk = list(masks[i:i + cols])
        while len(chunk) < cols:
            chunk.append(np.zeros_like(masks[0]))
        row = []
        for k, m in enumerate(chunk):
            img = cv2.cvtColor((np.clip(m, 0, 1) * 255).astype(np.uint8),
                               cv2.COLOR_GRAY2BGR)
            cv2.putText(img, str(i + k), (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 200, 255), 1, cv2.LINE_AA)
            row.append(img)
        rows.append(np.hstack(row))
    w = max(r.shape[1] for r in rows)
    cv2.imwrite(str(path), np.vstack(
        [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]))


def _learn_pass(bank: GlyphBank, cfg: Config, path: Path, samples: list[np.ndarray],
                wheres: list[str], labels: list[str] | None, kind: str,
                example: str) -> list[str] | None:
    groups = GlyphBank.cluster(samples, cfg.cluster_score)
    groups.sort(key=len, reverse=True)
    montage = path.parent / f"{kind}_clusters.png"
    dump_montage([samples[GlyphBank.medoid_index(samples, m)] for m in groups], montage)
    print(f"\n{kind}: {len(groups)} shapes from {len(samples)} samples, "
          f"montage at {montage}")
    for gi, m in enumerate(groups):
        print(f"  [{gi:2d}] n={len(m):3d}  e.g. {wheres[m[0]]}")
    if labels is None:
        labels = _ask(kind, len(groups), example)
        if labels is None:
            return None
    if len(labels) != len(groups):
        print(f"  {kind}: {len(groups)} shapes but {len(labels)} labels - aborting")
        print(f"  (it is fine for two shapes to be given the same label)")
        return None
    decided = ["?"] * len(samples)
    for gi, (members, label) in enumerate(zip(groups, labels)):
        for m in members:
            decided[m] = label
        for idx in GlyphBank.exemplars(samples, members):
            bank.add(label, samples[idx], gi)
    return decided


def _self_check(kind: str, samples: list[np.ndarray], labels: list[str]) -> None:
    """Leave-one-out over the freshly learned shapes: a wrong label shows up
    here as a miss instead of quietly poisoning every future read."""
    if len(samples) < 3:
        return
    from collections import Counter
    counts = Counter(labels)
    right, misses, unfair = 0, [], 0
    for i, m in enumerate(samples):
        best, bj = -2.0, None
        for j, mj in enumerate(samples):
            if i == j:
                continue
            s = corr(m, mj)
            if s > best:
                best, bj = s, j
        if bj is None:
            continue
        if labels[bj] == labels[i]:
            right += 1
        elif counts[labels[i]] == 1:
            unfair += 1
        else:
            misses.append((best, labels[i], labels[bj]))
    misses.sort()
    total = right + len(misses) + unfair
    print(f"\n{kind} self-check: {right}/{total} correct"
          + (f", {unfair} with no same-shape peer" if unfair else ""))
    for s, a, b in misses[:6]:
        print(f"  {kind} MISS {a!r} read as {b!r} (score {s:.2f}) - check that label")


def cmd_learn(cfg: Config, pattern: str,
              rank_labels: list[str] | None, suit_labels: list[str] | None,
              digit_labels: list[str] | None) -> int:
    (rank_samples, rank_where), (pip_samples, pip_where), \
        (digit_samples, digit_where) = collect_samples(cfg, pattern)
    if not rank_samples:
        print(f"no cards found in {pattern} - check board.slots / hero.hole_cards")
        return 1
    print(f"{len(rank_samples)} rank glyphs, {len(pip_samples)} pips, "
          f"{len(digit_samples)} money glyphs from {len(glob.glob(pattern))} screenshots")

    jobs = [
        (cfg.templates, rank_samples, rank_where, rank_labels, "rank",
         "one label per shape, e.g. 9 A A 5 8 7 6 T J"),
        (cfg.templates, pip_samples, pip_where, suit_labels, "suit", "e.g. s h d c"),
        (cfg.digits, digit_samples, digit_where, digit_labels, "digit",
         "e.g. 0 1 2 3 4 5 6 7 8 9 K"),
    ]
    banks: dict[Path, GlyphBank] = {}
    checks: list[tuple[str, list[np.ndarray], list[str]]] = []
    for path, samples, wheres, labels, kind, example in jobs:
        if not samples:
            print(f"\n{kind}: nothing found, skipping")
            continue
        bank = banks.setdefault(path, GlyphBank(path, cfg.match_score))
        decided = _learn_pass(bank, cfg, path, samples, wheres, labels, kind, example)
        if decided is None:
            return 1
        checks.append((kind, samples, decided))

    for kind, samples, labels in checks:
        _self_check(kind, samples, labels)

    for path, bank in banks.items():
        bank.save()
        print(f"saved {len(bank.masks)} prototypes covering {len(set(bank.labels))} "
              f"shapes to {path}")
    return 0
