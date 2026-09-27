from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state import Hand

RANKS = "23456789TJQKA"
SUITS = "shdc"


def _opening_chart() -> dict[str, float]:
    chart: dict[str, float] = {}
    for r, v in zip(RANKS, [0.52, 0.55, 0.58, 0.62, 0.68, 0.74,
                            0.86, 0.92, 1.00, 1.00, 1.00, 1.00, 1.00]):
        assert r + r not in chart
        chart[r + r] = v
    if len(chart) != len(RANKS):
        raise AssertionError("pair scores and RANKS are out of step")
    for hi in "AKQJ":
        base = {"A": 0.86, "K": 0.74, "Q": 0.60, "J": 0.46}[hi]
        for gap, lo in enumerate(reversed(RANKS[:RANKS.index(hi)])):
            chart[hi + lo + "s"] = max(0.05, base - 0.075 * gap)
    for hi in "98765432":
        for gap, lo in enumerate(reversed(RANKS[:RANKS.index(hi)])):
            chart[hi + lo + "s"] = max(0.04, 0.32 - 0.06 * gap)
    for h, v in {"54s": 0.24, "63s": 0.22, "53s": 0.20, "52s": 0.20, "43s": 0.18}.items():
        chart[h] = v
    for hi in "AQJ":
        for gap, lo in enumerate(reversed(RANKS[:RANKS.index(hi)])):
            chart[hi + lo + "o"] = max(0.03, 0.30 - 0.06 * gap)
    chart["A2o"] = 0.18
    return chart


OPENING_CHART = _opening_chart()


def hand_class(hole: list[str]) -> str | None:
    if len(hole) != 2 or any(len(c) != 2 for c in hole):
        return None
    order = {r: i for i, r in enumerate(RANKS)}
    (r1, s1), (r2, s2) = hole
    if r1 not in order or r2 not in order:
        return None
    if r1 == r2:
        return r1 + r2
    if order[r1] < order[r2]:
        r1, r2, s1, s2 = r2, r1, s2, s1
    return f"{r1}{r2}{'s' if s1 == s2 else 'o'}"


def chart_score(cls: str | None) -> float:
    if cls is None:
        return 0.0
    if cls in OPENING_CHART:
        return OPENING_CHART[cls]
    flipped = cls[:2] + ("o" if cls.endswith("s") else "s")
    return OPENING_CHART.get(flipped, 0.05)


def monte_carlo_equity(hole: list[str], board: list[str], sims: int = 20000,
                       seed: int = 7) -> float | None:
    """Head-to-head equity vs one random opponent. Fixed seed, so two runs of
    the same hand always agree."""
    try:
        from treys import Card as TCard
        from treys import Deck, Evaluator
    except ImportError:
        return None

    def to_t(code: str) -> int:
        if len(code) != 2 or code[0] not in RANKS or code[1] not in SUITS:
            raise ValueError(f"not a card: {code!r}")
        return TCard.new(code)

    try:
        hero = [to_t(c) for c in hole]
        board_cards = [to_t(c) for c in board]
    except (ValueError, KeyError):
        return None
    if len(hero) != 2 or len(board_cards) > 5:
        return None

    ev = Evaluator()
    dead = set(hero) | set(board_cards)
    live = [c for c in Deck.GetFullDeck() if c not in dead]
    rs = np.random.RandomState(seed)
    wins = ties = played = 0
    for _ in range(sims):
        pick = rs.permutation(len(live))[:2 + (5 - len(board_cards))]
        b = list(board_cards) + [live[i] for i in pick[2:2 + (5 - len(board_cards))]]
        hs = ev.evaluate(hero, b)
        os_ = ev.evaluate([live[pick[0]], live[pick[1]]], b)
        played += 1
        if hs < os_:
            wins += 1
        elif hs == os_:
            ties += 1
    return (wins + 0.5 * ties) / played if played else None


@dataclass
class Advice:
    action: str
    reason: str
    equity: float | None = None
    pot_odds: float | None = None
    cls: str | None = None


def advise(hand: Hand, sims: int = 20000) -> Advice:
    """Preflop only: chart score, Monte Carlo equity, pot odds and stack depth."""
    if len(hand.hole_codes) != 2:
        return Advice("WAIT", "both hole cards must read first")
    if hand.board_codes:
        return Advice("WAIT", "postflop advice is not implemented yet")

    cls = hand_class(hand.hole_codes)
    score = chart_score(cls)
    equity = monte_carlo_equity(hand.hole_codes, hand.board_codes, sims=sims)

    pot = hand.pot or 0
    to_call = hand.to_call or 0
    pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.0
    spr = (hand.hero_stack / pot) if (hand.hero_stack and pot) else None

    adjusted = score
    if spr is not None:
        if spr < 3:
            adjusted = min(1.0, score * 1.25 + 0.05)
        elif spr > 20:
            adjusted = score * 0.80

    spr_txt = f", spr {spr:.1f}" if spr is not None else ""
    if to_call > 0 and equity is not None:
        if equity < pot_odds:
            return Advice("FOLD", f"equity {equity:.1%} is under the {pot_odds:.1%} "
                                  f"pot odds", equity, pot_odds, cls)
        if equity < pot_odds * 1.15:
            return Advice("CHECK", f"marginal: equity {equity:.1%} against "
                                   f"{pot_odds:.1%} pot odds", equity, pot_odds, cls)
    if adjusted >= 0.62:
        return Advice("RAISE", f"{cls} scores {score:.2f} in the opening chart{spr_txt}",
                      equity, pot_odds, cls)
    if adjusted >= 0.30:
        return Advice("CHECK", f"{cls} is a marginal open - check or raise small{spr_txt}",
                      equity, pot_odds, cls)
    return Advice("FOLD", f"{cls} is outside the opening range", equity, pot_odds, cls)
