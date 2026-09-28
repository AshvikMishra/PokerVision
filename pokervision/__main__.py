#!/usr/bin/env python3
"""CLI entrypoint: ``python -m pokervision <command>``.

The implementation lives in the ``pokervision`` package. This module re-exports
the pipeline's public surface, so a script can do either

    from pokervision import CardFinder
    from pokervision.__main__ import CardFinder
"""

from .capture import Capture, WindowCapture
from .cli import (cmd_doctor, cmd_verify, describe_source, main,
                  pick_capture, report, watch)
from .config import Config
from .detect import CardFinder, Detected, group_rows
from .learning import collect_samples, cmd_learn
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
from .regions import Rect
from .state import Reading, ReadingTracker, read_frame


if __name__ == "__main__":
    raise SystemExit(main())
