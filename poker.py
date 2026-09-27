#!/usr/bin/env python3
"""Compatibility facade and CLI entrypoint. The implementation lives in the
``pokerbot`` package."""

from pokerbot.capture import Capture
from pokerbot.cli import cmd_doctor, cmd_verify, main, report, watch
from pokerbot.config import Config
from pokerbot.decision import RANKS, SUITS
from pokerbot.decision import (
    Advice,
    OPENING_CHART,
    advise,
    chart_score,
    hand_class,
    monte_carlo_equity,
)
from pokerbot.learning import collect_samples, cmd_learn
from pokerbot.perception import (
    Card,
    CardReader,
    GlyphBank,
    card_ink,
    card_ink_soft,
    card_outline,
    components,
    corr,
    is_dealt,
    locate_card,
    pick_corner_pip,
    pick_rank_glyph,
    rasterise,
    read_money_value,
    segment_money,
)
from pokerbot.state import Hand, HandTracker, build_state


if __name__ == "__main__":
    raise SystemExit(main())
