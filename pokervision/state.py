from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .capture import Capture
from .perception import Card, CardReader


@dataclass
class Reading:
    """What one frame of the calibrated path saw. A description of pixels, not
    an opinion about them."""

    hole: list[Card] = field(default_factory=list)
    board: list[Card] = field(default_factory=list)
    pot: int | None = None
    hero_stack: int | None = None

    @property
    def hole_codes(self) -> list[str]:
        return [c.code for c in self.hole if c.code]

    @property
    def board_codes(self) -> list[str]:
        return [c.code for c in self.board if c.code]

    @property
    def street(self) -> str:
        return {0: "preflop", 3: "flop", 4: "turn", 5: "river"}.get(
            len(self.board_codes), f"{len(self.board_codes)} board cards")

    @property
    def complete(self) -> bool:
        """Both hole cards and at most five board cards, all of them named."""
        return (len(self.hole) == 2
                and len(self.board) <= 5
                and not self.problems)

    @property
    def problems(self) -> list[str]:
        out = []
        if len(self.hole) != 2:
            out.append(f"{len(self.hole)}/2 hole cards visible")
        for c in self.hole + self.board:
            if c.rank is None:
                out.append(f"rank glyph at {c.where} unnamed: best {c.rank_score:.2f}, "
                           f"margin {c.rank_margin:.2f}")
            if c.suit is None:
                out.append(f"suit pip at {c.where} unnamed: best {c.suit_score:.2f}, "
                           f"margin {c.suit_margin:.2f}")
        return out

    def describe(self) -> str:
        h = " ".join(self.hole_codes) or "? ?"
        b = " ".join(self.board_codes) or "(none)"
        return f"hole [{h}]  board [{b}]"

    def money(self) -> list[str]:
        out = []
        if self.pot is not None:
            out.append(f"pot {self.pot:,}")
        if self.hero_stack is not None:
            out.append(f"stack {self.hero_stack:,}")
        return out


def read_frame(reader: CardReader, frame: np.ndarray, cap: Capture,
              cfg: Config) -> Reading:
    return Reading(
        hole=reader.read_hero(frame, cap),
        board=reader.read_board(frame, cap),
        pot=reader.read_money(frame, cap, cfg.pot),
        hero_stack=reader.read_money(frame, cap, cfg.hero_stack),
    )


class ReadingTracker:
    """Nothing is believed until it repeats for a few frames, and a read only
    counts as a new table state if the board shrank or the hole cards changed.

    This is a *stability* filter, not a decision. It exists because a single
    frame of template matching is noisy: a glyph halfway matched reads as a
    different card. Requiring the same read N frames running removes almost all
    of that flicker without inventing anything.
    """

    def __init__(self, stable_frames: int = 3):
        self.stable_frames = stable_frames
        self.hand_no = 0
        self.committed: tuple | None = None
        self.candidate: tuple | None = None
        self.hits = 0

    @staticmethod
    def _signature(r: Reading) -> tuple:
        return (tuple(r.hole_codes), tuple(r.board_codes))

    def update(self, r: Reading) -> str:
        if not r.hole and not r.board:
            sig = self._signature(r)
            if self.committed is None:
                self.committed = sig
            return "waiting"
        if r.problems:
            self.candidate, self.hits = None, 0
            return "unstable"

        sig = self._signature(r)
        if sig == self.candidate:
            self.hits += 1
        else:
            self.candidate, self.hits = sig, 1
        if self.hits < self.stable_frames:
            return "settling"

        if self.committed is None:
            self.committed = sig
            if any(sig):
                self.hand_no = 1
                return "new-hand"
            return "waiting"

        if sig == self.committed:
            return "same"

        prev_hole, prev_board = self.committed
        hole, board = sig
        if len(board) < len(prev_board) or hole != prev_hole:
            self.committed = sig
            self.hand_no += 1
            return "new-hand"

        self.committed = sig
        return "deal"
