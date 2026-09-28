"""PokerVision: read cards off a Monopoly Poker table.

A computer-vision pipeline and nothing else. It captures a screen, finds the
cards, and names them by correlating their glyphs against a bank learned from
your own screenshots. It does not decide what to do with a hand, does not
model poker, and never sends a click to the table.

The public surface is deliberately the perception half only:

    Config, Capture, WindowCapture      where frames come from
    CardFinder, Detected, group_rows    how cards are found
    GlyphBank, CardReader               how they are named
    Reading, read_frame                 what one frame said
"""

from .regions import Rect
from .config import Config
from .capture import Capture, WindowCapture
from .detect import CardFinder, Detected, group_rows
from .perception import (
    RANKS,
    SUITS,
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
from .state import Reading, ReadingTracker, read_frame
from .learning import collect_samples, cmd_learn
from .cli import cmd_doctor, cmd_verify, describe_source, main, watch

__all__ = [
    "RANKS", "SUITS", "Rect",
    "Capture", "Card", "CardFinder", "CardReader", "Config", "Detected",
    "GlyphBank", "Reading", "ReadingTracker", "WindowCapture",
    "card_ink", "card_ink_soft", "card_outline", "cmd_doctor", "cmd_verify",
    "collect_samples", "components", "corr", "describe_source",
    "group_rows", "is_dealt", "locate_card", "main", "pick_corner_pip",
    "pick_rank_glyph", "rasterise", "read_frame", "read_money_value",
    "segment_money", "watch",
]
