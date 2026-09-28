"""Dynamic card detection: find cards anywhere in a region, no fixed positions.

Anchor-based. Each card prints exactly one rank glyph at its top-left, so the
search hunts for rank glyphs and each one anchors a card. That is what makes
this work where a paper-region search cannot: the two hole cards overlap and
their paper merges with the "PAIR" label and the turn-ring into one 430x230
blob, so there is no card-shaped region to find. Two overlapping cards still
have two separate rank glyphs.

Once a rank blob is located, everything inside the card box it implies is put
through the same `pick_rank_glyph` / `pick_corner_pip` / `rasterise` steps, so
the rasters reaching the matcher are the shape the bank was learned from. Only
the *position* comes from the search. Skipping that split and matching the raw
search blobs directly read a 6, a 9 and a 3 all as an 8.

Those three are re-derived below rather than called, because a whole-frame search
asks the same question of several hundred blobs and each of them re-segments the
crop from scratch: `card_ink_soft` seven times per angle, and seven angles, to
name one card. The ink maps are built once per view and handed down instead,
which leaves the numbers reaching `_fit` - and so the matcher - unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import regions as rc
from .config import Config
from .perception import (GLYPH_N, INK_LEVEL, MIN_MARGIN, MIN_RANK_H_OVER_CARD,
                         MIN_RANK_W_OVER_H, PAPER_S_MAX, GlyphBank, _fit,
                         card_ink_soft, components)

# a card is ~0.70 as wide as it is tall, and a rank glyph is ~0.30 of the card
# height. The bracket is the range seen in practice, 0.28 being the narrowest
# real case (the '1' of a '10') and 0.40 a generous top for a heavy glyph.
CARD_ASPECT = 0.70
RANK_OVER_CARD_H = 0.30
RANK_MIN_OVER_CARD_H = 0.22
RANK_MAX_OVER_CARD_H = 0.40
# paper must be bright: a dark desaturated background is not paper
PAPER_V_MIN = 140
# a pixel counts as ink once it is this much darker than the paper, or this
# much more saturated red
DARK_OVER_PAPER = 0.45
RED_HUE_MAX = 15
RED_HUE_MIN = 165
RED_S_FLOOR = 70
RED_S_SPAN = 90
# the green "your turn" ring: hue, saturation and value it has to clear to be
# treated as a UI overlay rather than card artwork
RING_HUE_LO = 35
RING_HUE_HI = 90
RING_V_MIN = 90
# lookup tables for `ink_map`: each is a function of one 8-bit channel, so it
# only ever has 256 possible answers
_CHANNEL = np.arange(256, dtype=np.float32)
_RED_RAMP = np.clip((_CHANNEL - RED_S_FLOOR) / RED_S_SPAN, 0.0, 1.0).astype(np.float32)
_RED_HUES = np.array([1.0 if (h < RED_HUE_MAX or h > RED_HUE_MIN) else 0.0
                      for h in range(256)], np.float32)
_RED_HUE_BYTES = np.array([255 if v else 0 for v in _RED_HUES], np.uint8)
# rank-glyph-sized blobs, in pixels and as a fraction of frame height
MIN_RANK_H = 16
MAX_RANK_H_FRAC = 0.25
MIN_GLYPH_W_OVER_H = 0.20
MAX_GLYPH_W_OVER_H = 1.90
MIN_BLOB_AREA = 40
# how hard to erode before labelling, to break touching glyphs apart
ERODE_ITERS = 0
# a card prints its rank in its own top-left corner; anything found further in
# than this is a photograph or a label, not a card
RANK_MAX_X = 0.45
RANK_MAX_Y = 0.45
# paper must continue at least this far below a rank, as a fraction of the
# tallest card the rank could belong to
MIN_FACE_BELOW = 0.55
# How much of the implied card box has to be paper, and how much paper means
# the card cannot be turned. The box is drawn around the card, so a card
# printed theta off axis overhangs its own axis-aligned box at the corners and
# the overhang grows with the angle: measured over these screenshots the two
# hole cards, which print about five degrees out, cover 0.71 to 0.75 of their
# box, and every card that reads upright covers at least 0.78. So 0.65 is the
# floor under which the box is not a card at all, and 0.80 is the point above
# which a deskew cannot be what is missing.
MIN_FACE_COVER = 0.65
UPRIGHT_COVER = 0.80
# the corner pip sits below the rank, no further away than this many rank
# heights, and a pip-sized blob is at least this big
PIP_GAP = 1.15
MIN_PIP_AREA = 12
# real cards on a table are all much the same size, so drop lone outliers
SIZE_TOLERANCE_LO = 0.60
SIZE_TOLERANCE_HI = 1.70
# Deskew candidates, upright first. The hole cards are printed a few degrees
# off axis; 5 matches what the table actually does. A box that is nearly all
# paper tries `UPRIGHT_SWEEP` instead, which is most candidates.
ANGLE_SWEEP = (0.0, 5.0, -5.0, 3.0, -3.0, 7.0, -7.0)
UPRIGHT_SWEEP = (0.0,)
# two anchors within this fraction of card height are the same row
ROW_TOLERANCE = 0.55
# what `perception.rasterise` hands back for a mask with nothing in it
_EMPTY_RASTER = np.zeros((GLYPH_N, GLYPH_N), np.float32)


class _VectorBank:
    """Every prototype correlated at once, instead of one call per shape.

    `GlyphBank.match` is a plain Python loop of Pearson correlations, which is
    fine when there are seven candidate cards and hopeless when a whole-frame
    search offers four hundred and each may be tried at seven angles - that
    came to 1.9 s a frame, or 0.5 fps. Stacking the masks into one matrix makes
    the whole bank a single matrix product.

    Deliberately a separate class rather than a change to `GlyphBank`: the
    arithmetic is the same, but the learned bank and every existing caller stay
    exactly as they were.

    The original subtracts a per-glyph mean before correlating, so the raster is
    centred against its own mean and each mask against its own, once, at load.
    """

    def __init__(self, bank: GlyphBank):
        self.match_score = bank.match_score
        if not bank.masks:
            self.matrix = np.zeros((0, 1), np.float64)
            self.norms = np.zeros(0, np.float64)
            self.columns: dict[str, np.ndarray] = {}
            return
        m = np.stack(bank.masks).reshape(len(bank.masks), -1).astype(np.float64)
        m -= m.mean(axis=1, keepdims=True)
        self.matrix = m
        self.norms = np.sqrt((m * m).sum(axis=1))
        self.columns = {}
        for i, lab in enumerate(bank.labels):
            self.columns.setdefault(lab, []).append(i)
        self.columns = {k: np.array(v) for k, v in self.columns.items()}

    def match(self, raster: np.ndarray) -> tuple[str | None, float, float]:
        if self.matrix.shape[0] == 0:
            return None, 0.0, 0.0
        x = raster.reshape(-1).astype(np.float64)
        x = x - x.mean()
        x_norm = float(np.sqrt((x * x).sum()))
        denom = x_norm * self.norms
        # Correlate every prototype, then divide in place. Selecting the usable
        # rows first with a boolean mask copies the whole 83x4096 bank on every
        # call, which measured 0.47 ms against 0.055 ms for the plain product -
        # and the two scores are bit-identical, because a matrix row's dot
        # product does not depend on the other rows.
        scores = self.matrix @ x
        ok = denom > 0
        scores[ok] /= denom[ok]
        best: list[tuple[float, str]] = []
        for lab, idx in self.columns.items():
            best.append((float(scores[idx].max()), lab))
        if not best:
            return None, 0.0, 0.0
        best.sort(reverse=True)
        top, label = best[0]
        runner = best[1][0] if len(best) > 1 else 0.0
        margin = top - runner
        if top < self.match_score or margin < MIN_MARGIN:
            return None, top, margin
        return label, top, margin


@dataclass(frozen=True)
class Detected:
    """One card found by search, already named.

    `angle` is kept because it is what made the card readable: a hole card
    printed a few degrees off axis only names its glyphs once deskewed, and
    without this the caller cannot reproduce the read from the rect.
    """

    rank: str
    suit: str
    rect: rc.Rect
    angle: float = 0.0

    @property
    def code(self) -> str:
        return f"{self.rank}{self.suit}"


def _dark_table(white: float) -> np.ndarray:
    """The dark-ink ramp for a paper this bright: 256 entries, falling."""
    return np.clip((np.float32(white) - _CHANNEL)
                   / max(np.float32(1.0), np.float32(white * DARK_OVER_PAPER)),
                   0.0, 1.0).astype(np.float32)


def _ring_mask(hsv: np.ndarray) -> np.ndarray:
    """The green "your turn" ring: a UI overlay, not card artwork."""
    return cv2.inRange(hsv, (RING_HUE_LO, RED_S_FLOOR + 1, RING_V_MIN + 1),
                       (RING_HUE_HI, 255, 255))


def paper_mask(frame: np.ndarray, hsv: np.ndarray | None = None) -> np.ndarray:
    """Solid card faces: light unsaturated pixels, holes filled.

    Brightness is required as well as low saturation. Over a whole screenshot
    every desaturated dark pixel - page background, sidebar, chrome - also
    passes a saturation-only test and would then be read as ink.

    The fill is load-bearing, not cosmetic. Ink is darker than paper, so a
    glyph is itself excluded by the brightness test; without filling, every
    glyph is zeroed as "not on paper" and shatters into specks.

    `hsv` is accepted so a caller doing both of this frame's passes can convert
    once instead of twice; it must be `cvtColor(frame, COLOR_BGR2HSV)`.
    """
    if hsv is None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # One `inRange` rather than a compare, an and, a cast and a scale. The
    # bounds are nudged because both tests were strict.
    m = cv2.inRange(hsv, (0, 0, PAPER_V_MIN + 1), (255, PAPER_S_MAX - 1, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    if cnts:
        cv2.drawContours(filled, cnts, -1, 255, -1)
    return filled


def ink_map(frame: np.ndarray, paper: np.ndarray,
            hsv: np.ndarray | None = None) -> np.ndarray:
    """Ink strength over the paper, float in [0, 1].

    Not an Otsu split. That is fine inside a single card crop where paper and
    ink are all there is, but across a whole screenshot the threshold lands far
    too low and the white space between glyphs reads as ink. The two real kinds
    of ink are named outright instead: dark print, and saturated red print.

    Known gap: the paper reference is one global median, so a dimmed
    winner-reveal card - dull pink paper, compressed ink contrast - falls below
    the threshold and its row goes undetected. Referencing each card against
    its own local face brightness fixes that in principle, and measured worse
    (45/68 against 55/68), because the local estimate blurs into the felt
    around the card. Left as the simpler known-good version on purpose.

    `hsv` is accepted so a caller that wants this map and `paper_mask` from one
    frame converts the pixels once. `find` does not pass it, because it
    thresholds without building the float map at all - see `_ink_binary`, whose
    cut-offs are read out of the tables below so the two forms agree by
    construction.
    """
    if paper.size == 0 or not paper.any():
        return np.zeros(frame.shape[:2], np.float32)
    if hsv is None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    white = float(np.median(v[paper > 0]))
    # Every one of these is a function of a single channel, so each is a
    # 256-entry table and a lookup rather than a chain of float operations over
    # two million pixels. Same values, a third of the passes: this map is the
    # single most expensive thing a whole-frame search does.
    dark = cv2.LUT(v, _dark_table(white))
    red = cv2.LUT(s, _RED_RAMP) * cv2.LUT(hue, _RED_HUES)
    d = np.maximum(dark, red, out=dark)
    d[_ring_mask(hsv) != 0] = 0.0
    d[paper == 0] = 0.0
    return d


def _level_bound(table: np.ndarray, level: float) -> tuple[int, float]:
    """The channel comparison that keeps every value whose table entry reaches
    `level`.

    Read back out of the table rather than restated as a number, so
    `_ink_binary` cannot drift away from `ink_map` if the constants are ever
    retuned. Each ramp is monotone - dark ink falls as the pixel brightens,
    red ink rises with saturation - so the values that reach the level sit
    against one end of the table or the other, and which end says which way
    round the comparison goes. Nothing reaches the level, so nothing is ink:
    `CMP_LE` against a value no channel can take.
    """
    hit = np.flatnonzero(table >= level)
    if not len(hit):
        return cv2.CMP_LE, -1.0
    if hit[0] == 0:
        return cv2.CMP_LE, float(hit[-1])
    return cv2.CMP_GE, float(hit[0])


def _ink_binary(hsv: np.ndarray, paper: np.ndarray) -> np.ndarray:
    """`ink_map(...) >= INK_LEVEL`, without building the float map.

    `ink_map` is the public, float-valued form; a search wants only where the
    ink saturates, and on a whole screenshot that is the difference between
    three float32 planes over two million pixels and one integer comparison per
    channel. Each cut-off comes from the same lookup table `ink_map` uses, so
    this is the same set of pixels by construction rather than by agreement -
    verified equal on every screenshot in `unmarked-images`.
    """
    v = hsv[:, :, 2]
    if paper.size == 0 or not paper.any():
        return np.zeros(v.shape[:2], np.uint8)
    white = float(np.median(v[paper > 0]))
    op, bound = _level_bound(_dark_table(white), INK_LEVEL)
    ink = cv2.compare(v, bound, op)
    red_op, red_bound = _level_bound(_RED_RAMP, INK_LEVEL)
    red = cv2.compare(hsv[:, :, 1], red_bound, red_op)
    cv2.bitwise_and(red, cv2.LUT(hsv[:, :, 0], _RED_HUE_BYTES), dst=red)
    cv2.bitwise_or(ink, red, dst=ink)
    cv2.bitwise_and(ink, cv2.bitwise_not(_ring_mask(hsv)), dst=ink)
    return cv2.bitwise_and(ink, paper)


def _components(mask: np.ndarray, min_area: int) -> list[tuple[int, ...]]:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    return [tuple(int(v) for v in stats[i])
            for i in range(1, n) if stats[i][4] >= min_area]


def _clip(frame: np.ndarray, r: rc.Rect) -> np.ndarray:
    h, w = frame.shape[:2]
    x0, y0 = max(0, r.x), max(0, r.y)
    x1, y1 = min(w, r.right), min(h, r.bottom)
    if x1 <= x0 or y1 <= y0:
        return np.empty((0, 0, 3), np.uint8)
    return frame[y0:y1, x0:x1]


def _card_height(paper: np.ndarray, x: int, y: int, rank_h: int) -> int:
    """Card height below a rank anchor, measured rather than guessed.

    `pick_rank_glyph` accepts a rank only if it is at least MIN_RANK_H_OVER_CARD
    of the crop it is given, so the crop has to be the card to within a few
    percent. Dividing the rank height by a nominal ratio is not accurate enough
    - a blob one part in fifty too tall drops a 0.30 ratio to 0.294, and one
    merged with the pip above it drops it under the floor, which is exactly
    why the tilted hole card never read.

    So walk down the anchor's column: the card face is paper, the felt below
    it is not. Clamped either side so a blob sitting on a large white panel
    cannot return an absurd height.
    """
    col_h, col_w = paper.shape[:2]
    if not 0 <= x < col_w:
        return max(1, int(round(rank_h / RANK_OVER_CARD_H)))
    col = paper[:, x]
    bottom = y
    while bottom < col_h and col[bottom]:
        bottom += 1
    measured = bottom - y
    lo = max(1, int(round(rank_h / RANK_MAX_OVER_CARD_H)))
    hi = max(lo, int(round(rank_h / RANK_MIN_OVER_CARD_H)))
    return max(lo, min(measured, hi))


def _face_cover(paper: np.ndarray, x: int, y: int, card_w: int,
                card_h: int) -> float:
    """How much of the box a rank implies is actually paper.

    One mean over a sub-array, so it is free next to segmenting the crop, and
    it is the difference between a card and a white chat bubble: the box is
    the card's own face, which is paper, and almost nothing else on the table
    is a rectangle of that shape which is.
    """
    h, w = paper.shape[:2]
    box = paper[max(0, y):min(h, y + card_h), max(0, x):min(w, x + card_w)]
    return 0.0 if box.size == 0 else float(box.mean()) / 255.0


def _pip_below(ink: np.ndarray, x: int, y: int, w: int, h: int,
               card_w: int) -> bool:
    """Whether a pip-sized blob sits under the rank, in the frame's own ink map.

    A card prints its rank over its suit, so `pick_corner_pip` cannot name one
    without a pip under the rank - which makes this a necessary condition of a
    read rather than a guess at one. It runs on the ink map the frame already
    segmented rather than on a fresh card crop, because it has to cost almost
    nothing: one connected-components pass over a window the size of the rank
    and the pip together. Together with the coverage test it drops three
    quarters of the blobs a whole-frame search offers, and over these
    screenshots it kept every card the ungated search could read.
    """
    ih, iw = ink.shape[:2]
    top = y + h
    y0, y1 = max(0, top - 2), min(ih, int(top + h * PIP_GAP) + 1)
    x0, x1 = max(0, x), min(iw, x + card_w)
    if y1 <= y0 or x1 <= x0:
        return False
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink[y0:y1, x0:x1], 8)
    first, span = top - y0, h * PIP_GAP
    for i in range(1, n):
        by, bw, bh = int(stats[i][1]), int(stats[i][2]), int(stats[i][3])
        if int(stats[i][4]) < MIN_PIP_AREA or by < first - 2 or by > first + span:
            continue
        # same shape bracket `pick_corner_pip` uses, and the same way round: a
        # blob too wide is a label and one too narrow is a sliver of felt, so
        # either way it is not the pip
        if bw > h * 0.85 or bw < bh * MIN_RANK_W_OVER_H:
            continue
        return True
    return False


def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
    if not angle or not img.size:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def _rank_area(h: int, w: int, cfg: Config) -> tuple[int, int, int, int]:
    """`cfg.rank_region` over an h x w card, as (y0, y1, x0, x1)."""
    rx, ry, rw, rh = cfg.rank_region
    return (int(ry * h), min(h, int((ry + rh) * h)),
            int(rx * w), min(w, int((rx + rw) * w)))


def _binarise(soft: np.ndarray) -> np.ndarray:
    return (soft >= INK_LEVEL).astype(np.uint8) * 255


def _raster(soft: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """`perception.rasterise` against an ink map that already exists.

    `cv2.boundingRect` is the same box `np.nonzero` would find, at a fraction
    of the cost, and the soft map is shared by every angle.
    """
    if mask is None or not mask.size or not mask.any():
        return _EMPTY_RASTER
    x, y, w, h = cv2.boundingRect(mask)
    if w <= 0 or h <= 0:
        return _EMPTY_RASTER
    return _fit(soft[y:y + h, x:x + w])


def _rank_glyph(soft: np.ndarray, soft_sub: np.ndarray, ink: np.ndarray,
                cfg: Config) -> np.ndarray | None:
    """`perception.pick_rank_glyph` with its segmentation already done.

    `soft_sub` is deliberately a map of the rank area *on its own*, not
    `soft` sliced: the stock version re-derives the paper reference, the Otsu
    split and the convex hull per crop, so a slice of the whole-card map is
    not the same threshold.
    """
    h, w = soft.shape[:2]
    if h < 10 or w < 10:
        return None
    sub = _binarise(soft_sub)
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

    y0, _, x0, _ = _rank_area(h, w, cfg)
    out = np.zeros((h, w), np.uint8)
    out[y0 + by:y0 + by + bh, x0 + bx:x0 + bx + bw] = 255
    out = cv2.bitwise_and(out, ink)
    return out if cv2.countNonZero(out) >= 20 else None


def _corner_pip(ink: np.ndarray, rank: np.ndarray) -> np.ndarray | None:
    """`perception.pick_corner_pip` given the rank it would re-derive.

    The stock version calls `pick_rank_glyph` again purely for its bounding
    box. The caller already has that exact mask, so the second segmentation
    of the same pixels was pure waste.
    """
    h, w = ink.shape[:2]
    _, ry, _, rh = cv2.boundingRect(rank)
    rbot = ry + rh
    best = None
    for x, y, cw, ch, _a in components(ink, min_area=12):
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


class CardFinder:
    def __init__(self, cfg: Config, bank: GlyphBank):
        self.cfg = cfg
        self.bank = _VectorBank(bank)

    def find(self, frame: np.ndarray) -> list["Detected"]:
        """Every fully readable card in `frame`, already named.

        The cards come back read, not merely located. Returning bare rects
        loses the deskew angle: a card that only reads at 5 degrees is found
        correctly here and then reads as nothing when the caller re-reads the
        same rect upright, which is exactly the tilted hole card.
        """
        if frame is None or frame.size == 0:
            return []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        paper = paper_mask(frame, hsv)
        ink = _ink_binary(hsv, paper)
        if not ink.any():
            return []
        # A light erode separates touching glyphs, so that a 'Q' tail does not
        # weld itself to its own corner pip. Kept at zero: eroding hard enough
        # to be sure of that also eats the thin strokes of a 5 or a 7, and
        # measured across these screenshots it cost more than it saved. Zero is
        # also OpenCV's identity, so the call is skipped rather than paid for.
        thin = (ink if not ERODE_ITERS
                else cv2.erode(ink, np.ones((3, 3), np.uint8), ERODE_ITERS))

        max_rank_h = frame.shape[0] * MAX_RANK_H_FRAC
        found: list[Detected] = []
        for x, y, w, h, area in _components(thin, MIN_BLOB_AREA):
            if h < MIN_RANK_H or h > max_rank_h:
                continue
            if w > h * MAX_GLYPH_W_OVER_H or w < h * MIN_GLYPH_W_OVER_H:
                continue
            card = self._read(frame, paper, ink, x, y, w, h)
            if card is not None:
                found.append(card)
        return _consistent_sizes(found)

    def _read(self, frame: np.ndarray, paper: np.ndarray, ink: np.ndarray,
              x: int, y: int, w: int, h: int) -> "Detected | None":
        """Read the card anchored by a rank blob at (x, y), or None.

        The box is the card itself, unpadded. `pick_rank_glyph` judges the rank
        by its height as a fraction of the box, so padding shrinks that
        fraction: 8px of padding on a 130px card put a 39px rank at 0.267,
        under the 0.28 floor, and every board card was rejected.

        Angles are swept rather than estimated. `minAreaRect` on a compact
        glyph rotated five degrees hands back the axis-aligned rectangle,
        because the area difference is negligible - it measured 0.0 for the
        tilted hole card. No clamp fixes that. Trying a small set and keeping
        the first read that names both glyphs does, and 0 degrees is tried
        first - a box that is nearly all paper takes `UPRIGHT_SWEEP` alone,
        which is every card but the two hole cards.

        The box is segmented once per angle here rather than inside each of the
        three helpers, which between them re-derive it seven times. The upright
        attempt - what every card but the two hole cards takes - is
        bit-identical to the stock path, because a zero-degree warp is the
        identity. Warping the finished ink maps instead of the crop they came
        from was measured and rejected: it lost the tilted hole card and
        invented an 's7' and a 'Qd' out of two UI blobs, because a fixed
        threshold read off the upright card no longer matches the resampling
        that a sheared copy would have applied.
        """
        card_h = _card_height(paper, x, y, h)
        # Cheap gates before any glyph work. A real card's face continues well
        # below its rank, is paper right across the box, and has its pip under
        # the rank; most of the several hundred blobs a whole-frame search
        # offers are UI fragments that fail all three. Rejecting them here is
        # what keeps a full-screen search affordable.
        if card_h < h / RANK_MAX_OVER_CARD_H * MIN_FACE_BELOW:
            return None
        card_w = max(1, int(round(card_h * CARD_ASPECT)))
        cover = _face_cover(paper, x, y, card_w, card_h)
        if cover < MIN_FACE_COVER or not _pip_below(ink, x, y, w, h, card_w):
            return None
        box = rc.Rect(x=x, y=y, w=card_w, h=card_h)
        crop = _clip(frame, box)
        if crop.size == 0:
            return None
        sweep = UPRIGHT_SWEEP if cover >= UPRIGHT_COVER else ANGLE_SWEEP
        for angle in sweep:
            view = _rotate(crop, angle)
            if view.size == 0:
                continue
            soft = card_ink_soft(view)
            if soft.size == 0:
                continue
            y0, y1, x0, x1 = _rank_area(soft.shape[0], soft.shape[1], self.cfg)
            sub = card_ink_soft(view[y0:y1, x0:x1])
            if sub.size == 0:
                continue
            got = self._names_both(soft, sub)
            if got is not None:
                rank, suit = got
                return Detected(rank, suit, box, angle)
        return None

    def _names_both(self, soft: np.ndarray, soft_sub: np.ndarray
                    ) -> tuple[str, str] | None:
        """Name the rank and suit in a card view, or None.

        `soft` is the view's whole-card ink map and `soft_sub` the rank area's,
        both built by `_read`. This is the stock `pick_rank_glyph` /
        `rasterise` / `pick_corner_pip` chain with the segmentation hoisted out
        of it, which is what makes a seven-angle sweep affordable.

        The rank position test is what keeps a seat portrait from reading as a
        card. A face photograph inside the table region has skin tones that
        pass the paper test and eyes and lips that pass the ink test, so it
        anchors a box; but the blob that wins the glyph match sits mid-box,
        where a real card never prints anything. It runs before the match, not
        after - the match is the expensive half of the test and the position is
        free.
        """
        h, w = soft.shape[:2]
        ink = _binarise(soft)
        rm = _rank_glyph(soft, soft_sub, ink, self.cfg)
        if rm is None:
            return None
        rx, ry, _, _ = cv2.boundingRect(rm)
        if rx > w * RANK_MAX_X and ry > h * RANK_MAX_Y:
            return None
        rank = self.bank.match(_raster(soft, rm))[0]
        if rank is None:
            return None
        pm = _corner_pip(ink, rm)
        if pm is None:
            return None
        suit = self.bank.match(_raster(soft, pm))[0]
        return (rank, suit) if suit is not None else None


def _consistent_sizes(cards: list["Detected"]) -> list["Detected"]:
    """Drop detections whose size is wildly unlike the rest.

    Cards on a table are all one size for the board and one for the hole
    cards, so a genuine detection is never a lone outlier. This is the backstop
    for anything that slipped through the shape and glyph tests.
    """
    if len(cards) < 3:
        return cards
    heights = sorted(c.rect.h for c in cards)
    median = heights[len(heights) // 2]
    return [c for c in cards
            if SIZE_TOLERANCE_LO * median <= c.rect.h <= SIZE_TOLERANCE_HI * median]


def group_rows(cards: list["Detected"]) -> tuple[list["Detected"], list["Detected"]]:
    """Split detections into (board, hero) by row.

    Rows are clusters of similar y-centre, top to bottom. The lowest row of at
    most two cards is the hero's; everything above is the board, sorted left to
    right, which is flop -> turn -> river order.
    """
    if not cards:
        return [], []
    heights = sorted(c.rect.h for c in cards)
    typical = heights[len(heights) // 2]
    ordered = sorted(cards, key=lambda c: c.rect.y + c.rect.h / 2)
    rows: list[list[Detected]] = [[ordered[0]]]
    for c in ordered[1:]:
        prev = rows[-1][-1].rect
        here = c.rect
        if abs((here.y + here.h / 2) - (prev.y + prev.h / 2)) <= typical * ROW_TOLERANCE:
            rows[-1].append(c)
        else:
            rows.append([c])
    hero_rows = [i for i, r in enumerate(rows) if len(r) <= 2]
    hero_i = max(hero_rows) if hero_rows else -1
    hero = rows[hero_i] if hero_i >= 0 else []
    board = [c for i, r in enumerate(rows) if i != hero_i for c in r]
    board.sort(key=lambda c: c.rect.x)
    hero.sort(key=lambda c: c.rect.x)
    return board, hero
