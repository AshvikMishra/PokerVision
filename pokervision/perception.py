from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import regions as rc
from .config import Config
from .capture import Capture

GLYPH_N = 64
GLYPH_PAD = 6
RANKS = "23456789TJQKA"
SUITS = "shdc"
PAPER_S_MAX = 160
MIN_RANK_W_OVER_H = 0.25
MIN_RANK_H_OVER_CARD = 0.28
INK_LEVEL = 1.0
DIGIT_MIN_W_OVER_H = 0.32
DIGIT_MAX_W_OVER_H = 1.05
MONEY_S_MAX = 70
MONEY_V_MIN = 170
MIN_MARGIN = 0.04


def card_outline(card: np.ndarray) -> np.ndarray:
    s = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)[:, :, 1]
    paper = (s < PAPER_S_MAX).astype(np.uint8) * 255
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return paper
    out = np.zeros_like(paper)
    cv2.drawContours(out, [cv2.convexHull(max(cnts, key=cv2.contourArea))], -1, 255, -1)
    return out


def locate_card(window: np.ndarray) -> tuple[int, int, int, int] | None:
    mask = card_outline(window)
    if mask.size == 0 or not mask.any():
        return None
    x, y, w, h = cv2.boundingRect(mask)
    if w < window.shape[1] * 0.45 or h < window.shape[0] * 0.45:
        return None
    return x, y, w, h


def card_ink_soft(card: np.ndarray) -> np.ndarray:
    if card is None or card.size == 0:
        return np.zeros((0, 0), np.float32)
    paper = np.median(card.reshape(-1, 3), axis=0)
    delta = np.abs(card.astype(np.int16) - paper).max(axis=2).astype(np.float32)
    thr, _ = cv2.threshold(delta.astype(np.uint8), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    d = np.clip(delta / max(1.0, float(thr)), 0.0, 1.0).astype(np.float32)

    # the green "your turn" ring is a UI overlay, not card artwork
    hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)
    hue, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    d[(hue >= 35) & (hue <= 90) & (s > 70) & (v > 90)] = 0.0

    d[card_outline(card) == 0] = 0.0
    return d


def card_ink(card: np.ndarray) -> np.ndarray:
    soft = card_ink_soft(card)
    if soft.size == 0:
        return np.zeros((0, 0), np.uint8)
    return (soft >= INK_LEVEL).astype(np.uint8) * 255


def is_dealt(card: np.ndarray) -> bool:
    if card is None or card.size == 0:
        return False
    s = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)[:, :, 1]
    return float(np.median(s)) < PAPER_S_MAX


def components(mask: np.ndarray, min_area: int = 1) -> list[tuple[int, int, int, int, int]]:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    return [tuple(int(v) for v in stats[i])
            for i in range(1, n) if stats[i][4] >= min_area]


def pick_rank_glyph(card: np.ndarray, cfg: Config) -> np.ndarray | None:
    if card is None or card.size == 0:
        return None
    h, w = card.shape[:2]
    if h < 10 or w < 10:
        return None
    rx, ry, rw, rh = cfg.rank_region
    y0, x0 = int(ry * h), int(rx * w)
    y1, x1 = min(h, int((ry + rh) * h)), min(w, int((rx + rw) * w))
    sub = card_ink(card[y0:y1, x0:x1])
    if sub.size == 0:
        return None
    sh, sw = sub.shape[:2]

    tall = [c for c in components(sub, min_area=int(sh * 0.10))
            if c[3] >= h * MIN_RANK_H_OVER_CARD and c[2] <= sw * 0.92
            and c[2] >= c[3] * MIN_RANK_W_OVER_H]
    if not tall:
        return None
    tall.sort(key=lambda c: (c[1], c[0]))

    # '10' prints as two blobs on one baseline; pull them back together
    bx, by, bw, bh = tall[0][0], tall[0][1], tall[0][2], tall[0][3]
    for cx, cy, cw, ch, _ in tall[1:]:
        if abs(cy - by) < bh * 0.55 and cx < bx + bw + sw * 0.30:
            nx0, nx1 = min(bx, cx), max(bx + bw, cx + cw)
            ny0, ny1 = min(by, cy), max(by + bh, cy + ch)
            bx, by, bw, bh = nx0, ny0, nx1 - nx0, ny1 - ny0

    out = np.zeros(card.shape[:2], np.uint8)
    out[y0 + by:y0 + by + bh, x0 + bx:x0 + bx + bw] = 255
    out = cv2.bitwise_and(out, card_ink(card))
    return out if cv2.countNonZero(out) >= 20 else None


def pick_corner_pip(card: np.ndarray, cfg: Config) -> np.ndarray | None:
    if card is None or card.size == 0:
        return None
    h, w = card.shape[:2]
    rank = pick_rank_glyph(card, cfg)
    if rank is None:
        return None
    _, ry, _, rh = cv2.boundingRect(rank)
    rbot = ry + rh
    ink = card_ink(card)

    best = None
    for x, y, cw, ch, a in components(ink, min_area=12):
        if y < rbot - 2 or y > rbot + rh * 1.15:
            continue
        if cw > rh * 0.85:
            continue
        # felt slivers are tall and narrow; the corner pip is roughly square
        if cw < ch * MIN_RANK_W_OVER_H:
            continue
        if best is None or y - rbot < best[0]:
            best = (y - rbot, x, y, cw, ch)
    if best is None:
        return None
    _, x, y, cw, ch = best
    out = np.zeros((h, w), np.uint8)
    out[y:y + ch, x:x + cw] = 255
    out = cv2.bitwise_and(out, ink)
    return out if cv2.countNonZero(out) >= 12 else None


def _fit(tight: np.ndarray) -> np.ndarray:
    th, tw = tight.shape[:2]
    inner = GLYPH_N - 2 * GLYPH_PAD
    s = inner / max(th, tw)
    nh, nw = max(1, int(round(th * s))), max(1, int(round(tw * s)))
    scaled = cv2.resize(tight.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((GLYPH_N, GLYPH_N), np.float32)
    oy, ox = (GLYPH_N - nh) // 2, (GLYPH_N - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = scaled
    return out


def rasterise(card: np.ndarray, mask: np.ndarray) -> np.ndarray:
    empty = np.zeros((GLYPH_N, GLYPH_N), np.float32)
    if mask is None or card is None or card.size == 0 or not mask.any():
        return empty
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return empty
    return _fit(card_ink_soft(card)[ys.min():ys.max() + 1, xs.min():xs.max() + 1])


def corr(a: np.ndarray, b: np.ndarray) -> float:
    x = a.ravel().astype(np.float64) - a.mean()
    y = b.ravel().astype(np.float64) - b.mean()
    d = np.sqrt(float((x * x).sum()) * float((y * y).sum()))
    return float((x * y).sum() / d) if d else 0.0


class GlyphBank:
    def __init__(self, path: Path, match_score: float = 0.80):
        self.path = Path(path)
        self.match_score = match_score
        self.masks: list[np.ndarray] = []
        self.labels: list[str] = []
        self.groups: list[int] = []

    def load(self) -> "GlyphBank":
        if self.path.exists():
            with np.load(self.path, allow_pickle=False) as z:
                self.masks = list(z["masks"])
                self.labels = [str(x) for x in z["labels"]]
                self.groups = [int(x) for x in z["groups"]]
        return self

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.path,
            masks=np.array(self.masks, np.float32),
            labels=np.array(self.labels, dtype="U8"),
            groups=np.array(self.groups, np.int32),
        )

    @property
    def ready(self) -> bool:
        return bool(self.masks)

    def add(self, label: str, mask: np.ndarray, group: int) -> None:
        self.masks.append(mask)
        self.labels.append(label)
        self.groups.append(group)

    @staticmethod
    def cluster(samples: list[np.ndarray], floor: float) -> list[list[int]]:
        groups: list[list[int]] = []
        for i, s in enumerate(samples):
            best, bi = floor, -1
            for j, g in enumerate(groups):
                score = max(corr(s, samples[k]) for k in g)
                if score > best:
                    best, bi = score, j
            if bi >= 0:
                groups[bi].append(i)
            else:
                groups.append([i])
        return groups

    @staticmethod
    def medoid_index(samples: list[np.ndarray], members: list[int]) -> int:
        best, bi = -2.0, members[0]
        for i in members:
            score = sum(corr(samples[i], samples[j]) for j in members) / len(members)
            if score > best:
                best, bi = score, i
        return bi

    @classmethod
    def exemplars(cls, samples: list[np.ndarray], members: list[int],
                  k: int = 6) -> list[int]:
        chosen = [cls.medoid_index(samples, members)]
        while len(chosen) < min(k, len(members)):
            best, bi = 2.0, None
            for i in members:
                if i in chosen:
                    continue
                d = min(corr(samples[i], samples[c]) for c in chosen)
                if d < best:
                    best, bi = d, i
            if bi is None:
                break
            chosen.append(bi)
        return chosen

    def match(self, mask: np.ndarray) -> tuple[str | None, float, float]:
        if not self.masks or mask is None or not mask.any():
            return None, 0.0, 0.0
        per_label: dict[str, float] = {}
        for m, lab in zip(self.masks, self.labels):
            s = corr(mask, m)
            if s > per_label.get(lab, 0.0):
                per_label[lab] = s
        if not per_label:
            return None, 0.0, 0.0
        ranked = sorted(per_label.items(), key=lambda kv: -kv[1])
        label, best = ranked[0]
        runner = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = best - runner
        if best < self.match_score or margin < MIN_MARGIN:
            return None, best, margin
        return label, best, margin


@dataclass
class Card:
    rank: str | None
    suit: str | None
    rank_score: float
    suit_score: float
    rank_margin: float
    suit_margin: float
    rect: rc.Rect
    slot: int
    where: str = "?"

    @property
    def code(self) -> str | None:
        return f"{self.rank}{self.suit}" if self.rank and self.suit else None


class CardReader:
    def __init__(self, cfg: Config, bank: GlyphBank):
        self.cfg = cfg
        self.bank = bank

    def read_slots(self, frame, cap, slots, where):
        out = []
        for i, r in enumerate(slots):
            window = cap.crop(frame, r)
            found = locate_card(window)
            if found is None:
                continue
            fx, fy, fw, fh = found
            crop = window[fy:fy + fh, fx:fx + fw]
            ox, oy = cap.origin
            abs_rect = rc.Rect(x=r.x + fx, y=r.y + fy, w=fw, h=fh)

            rank_mask = pick_rank_glyph(crop, self.cfg)
            rank, rscore, rmargin = self.bank.match(rasterise(crop, rank_mask))
            pip = pick_corner_pip(crop, self.cfg)
            suit, sscore, smargin = self.bank.match(rasterise(crop, pip))
            out.append(Card(rank, suit, rscore, sscore, rmargin, smargin,
                            abs_rect, i, f"{where}{i}"))
        return out

    def read_board(self, frame: np.ndarray, cap: Capture) -> list[Card]:
        return self.read_slots(frame, cap, self.cfg.board_slots, "board")

    def read_hero(self, frame: np.ndarray, cap: Capture) -> list[Card]:
        return self.read_slots(frame, cap, self.cfg.hero_slots, "hole")

    def read_money(self, frame: np.ndarray, cap: Capture, r: rc.Rect | None) -> int | None:
        if r is None:
            return None
        crop = cap.crop(frame, r)
        if crop.size == 0:
            return None
        bank = GlyphBank(self.cfg.digits, self.cfg.match_score).load()
        return read_money_value(crop, bank) if bank.ready else None


def segment_money(crop: np.ndarray) -> list[np.ndarray]:
    h, w = crop.shape[:2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] < MONEY_S_MAX) & (hsv[:, :, 2] > MONEY_V_MIN)).astype(np.uint8) * 255

    # the text band, relative to the busiest row, so a fainter line clipping
    # the top of the box cannot drag it upwards
    rows = mask.sum(axis=1) / 255.0
    live = np.nonzero(rows >= max(2.0, 0.30 * float(rows.max())))[0]
    if len(live) < 4:
        return []
    runs, start = [], live[0]
    for a, b in zip(live, live[1:]):
        if b != a + 1:
            runs.append((start, a))
            start = b
    runs.append((start, live[-1]))
    y0, y1 = min(runs, key=lambda r: r[0] - r[1])
    y0, y1 = int(y0), int(y1) + 1
    band_h = y1 - y0
    if band_h < 4:
        return []

    # cut on empty columns, not connected components: this is one line of
    # evenly spaced glyphs and component labelling bridges or splits them
    cols = mask[y0:y1].sum(axis=0) / 255.0
    groups, run = [], None
    for x in range(w):
        if cols[x] > 0:
            run = x if run is None else run
        elif run is not None:
            groups.append((run, x))
            run = None
    if run is not None:
        groups.append((run, w))

    out = []
    for gx0, gx1 in groups:
        gw = gx1 - gx0
        if not (band_h * DIGIT_MIN_W_OVER_H <= gw <= band_h * DIGIT_MAX_W_OVER_H):
            continue
        patch = mask[y0:y1, gx0:gx1]
        if patch.sum() == 0:
            continue
        out.append(_fit((patch > 0).astype(np.float32)))
    return out


def read_money_value(crop: np.ndarray, bank: GlyphBank) -> int | None:
    digits: list[str] = []
    scale = 1
    for raster in segment_money(crop):
        label, _, _ = bank.match(raster)
        if label == "K":
            scale = 1_000
        elif label == "M":
            scale = 1_000_000
        elif label is not None and label.isdigit():
            digits.append(label)
    if not digits:
        return None
    return int("".join(digits)) * scale
