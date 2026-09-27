from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import regions_config as rc

REPO = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    board_slots: list[rc.Rect]
    hero_slots: list[rc.Rect]
    pot: rc.Rect | None
    hero_stack: rc.Rect | None
    to_call: rc.Rect | None
    rank_region: tuple
    match_score: float
    cluster_score: float
    templates: Path
    digits: Path
    fps: int
    monitor: int

    @classmethod
    def load(cls, path: Path | str = rc.DEFAULT_CONFIG) -> "Config":
        raw = rc.load(path)
        per = raw.get("perception") or {}
        cap = raw.get("capture") or {}

        def rects(key: str) -> list[rc.Rect]:
            node = rc.resolve(raw, key) or []
            return [r for i in range(len(node))
                    if (r := rc.get_rect(raw, f"{key}[{i}]")) and r.is_set]

        def one(key: str) -> rc.Rect | None:
            r = rc.get_rect(raw, key)
            return r if r and r.is_set else None

        return cls(
            board_slots=rects("board.slots"),
            hero_slots=rects("hero.hole_cards"),
            pot=one("texts.pot"),
            hero_stack=one("texts.hero_stack"),
            to_call=one("texts.to_call"),
            rank_region=tuple(per.get("rank_region", (0.0, 0.0, 0.95, 0.55))),
            match_score=float(per.get("match_score", 0.80)),
            cluster_score=float(per.get("cluster_score", 0.88)),
            templates=REPO / str(per.get("templates", "templates/glyphs.npz")),
            digits=REPO / str(per.get("digits", "templates/digits.npz")),
            fps=int(cap.get("fps", 10)),
            monitor=int(cap.get("monitor", 0)),
        )

    def missing(self) -> list[str]:
        out = []
        if len(self.board_slots) != 5:
            out.append(f"board.slots ({len(self.board_slots)}/5)")
        if len(self.hero_slots) != 2:
            out.append(f"hero.hole_cards ({len(self.hero_slots)}/2)")
        for name, r in (("texts.pot", self.pot), ("texts.hero_stack", self.hero_stack)):
            if r is None:
                out.append(name)
        return out
