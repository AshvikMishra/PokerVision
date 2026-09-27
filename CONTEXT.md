# CONTEXT.md

Findings from the cleanup and bug-fix pass over the Monopoly Poker table
assistant. Written to be the thing you read before touching the code.

---

## 1. What this is

A local, offline, **observe-only** screen reader for Monopoly Poker. It grabs
the table with `mss`, finds the cards inside calibrated regions, names each
card by template-matching a self-learned glyph bank, reads the pot and stack
from money boxes, and prints preflop advice. It never clicks anything.

There is no OCR engine and no ML model. Recognition is OpenCV segmentation
plus Pearson correlation against shapes clustered out of the project's own
screenshots.

## 2. The pipeline, end to end

```
poker.py                      CLI entrypoint + compatibility facade
  |
  +-- pokerbot/cli.py         argparse: doctor | verify | learn | watch | overlay
        |
        +-- pokerbot/config.py       reads config/regions.yaml -> Config dataclass
        |     |
        |     +-- regions_config.py  schema, loader, comment-preserving editor
        |
        +-- pokerbot/capture.py      mss grab; crop(rect) applies the tilt deskew
        |     |
        +-- pokerbot/perception.py   locate_card -> pick_rank_glyph / pick_corner_pip
        |     |                       -> rasterise -> GlyphBank.match
        |     +-- pokerbot/state.py        build_state() assembles a Hand
        |     +-- pokerbot/decision.py     advise(): chart + Monte Carlo + pot odds
        +-- pokerbot/overlay.py     click-through Qt window drawing the above
```

Data flow for one frame:

1. `Capture.get_frame()` -> BGR numpy array.
2. `Capture.crop(frame, rect)` slices the ROI and, **if `rect.angle` is set**,
   deskews it with `cv2.getRotationMatrix2D` before anything else looks at it.
3. `locate_card(window)` thresholds saturation to find the paper, takes a
   convex hull, and returns the card's real bounding rect inside the search box.
4. `pick_rank_glyph` / `pick_corner_pip` find the two glyph blobs.
5. `rasterise` tight-crops, rescales to 64x64 keeping aspect, staying soft.
6. `GlyphBank.match` correlates against every learned prototype, takes the best
   score per label, and names it only if it clears `match_score` **and** beats
   the runner-up by `MIN_MARGIN`.
7. `HandTracker` decides whether the read is a new hand or the same one.
8. `advise` returns RAISE / CHECK / FOLD / WAIT.

Bootstrap: `poker.py learn` clusters the glyphs out of `unmarked-images/`,
shows a montage, asks you to label each cluster, self-checks, and writes
`templates/glyphs.npz` and `templates/digits.npz`.

## 3. The tilt bug (fixed)

`config/regions.yaml` had:

```yaml
- {x: 905, y: 740, w: 104, h: 124, angle: 90}   # left hole card
```

`angle: 90` is a quarter turn. The left hero card was rotated onto its side
before glyph matching ever ran, so it matched nothing.

Measured on the 15 screenshots, **left hero card (`hole0`)**:

| `angle` | hole0 rank correct |
|---------|--------------------|
| 90 (before) | **0 / 14** |
| 5 (after)   | **14 / 14** |

`angle: 90` -> `angle: 5`, i.e. 5 degrees counter-clockwise, matching the
`positive = counter-clockwise` convention already documented in the config
header. The right hero card and all five board cards needed no deskew.

**Why it was never caught:** every screenshot still produced *a* `hole0` card,
so `verify` printed a line and exited 0. Only the `! rank glyph ... unnamed`
warnings below it showed the failure, and those were easy to skim past. The
left card being permanently blank looked like a weak glyph bank rather than a
config typo.

## 4. Everything removed, and why

### Files and directories
| Removed | Reason |
|---|---|
| `unmarked-samples/` | empty, referenced only by a stale README line |
| `__pycache__/` | build artifacts |
| `templates/*_clusters.png` | stale outputs of an old `learn` run; `learn` rewrites them |

`templates/glyphs.npz` and `templates/digits.npz` were **kept** - they are
learned data, and re-learning means re-labelling every cluster by hand.

### Dead code
| Removed | Location | Reason |
|---|---|---|
| duplicated import block | `perception.py` top | `cv2`, `np`, `rc`, `Path`, `dataclass` imported twice; `REPO` defined **twice with different values**, the second (wrong) one winning. `glob`, `time`, `field` were imported and never used. |
| `ink_colour()` | `perception.py` | ran a full HSV conversion per card to return red/black, and `Card.colour` was **never read** by state, decision or overlay. The suit comes from the pip's *shape*, so colour was pure cost. |
| `Card.colour`, `Card.colour_score` | `perception.py` | fields fed only by the above |
| `Config.raw` | `config.py` | stored the whole yaml dict, read by nobody |
| `Config.paper_v_min` | `config.py` | read from yaml, then never used - perception used the module constant `PAPER_S_MAX` |
| `Config.pip_region` | `config.py` | read from yaml, then never used - `pick_corner_pip` finds the pip as a connected component, not by cropping a window |
| `set_scalar()`, `_scalar_text()` | `regions_config.py` | no caller anywhere; `calibrate.py` only ever writes rects |
| `Rect.intersects`, `Rect.area`, `Rect.center` | `regions_config.py` | no caller |
| `format_money()` | `regions_config.py` | no caller |
| dead `if not samples` guard | `learning.py` | caller already skips empty sample lists |
| duplicate `_self_check` loop | `learning.py` | the self-check was **printed twice** per glyph family on every `learn` run |
| unused `sims` parameter | `cli.py` | `cmd_verify` took a sims count and ignored it |
| `overlay-gui` subcommand | `cli.py` | an exact duplicate of `overlay`; both called `cmd_overlay_gui` |

### Config removed from `regions.yaml` and the schema
These were never read by any code path, and all of them describe automation
that does not exist:

`buttons.*` (fold, check_call, raise, raise_slider, raise_confirm,
raise_presets), `signals.my_turn`, `seats`, `texts.hero_bet`,
`perception.filled_ratio`, `perception.min_margin`, `perception.paper_v_min`,
`perception.pip_region`.

`calibrate.py set`/`nudge`/`mark` still recreate any of them on demand -
`_ensure_container` builds the parent chain if it is missing.

### Comments
Docstrings and comments were stripped from the pipeline modules. Kept: the
`regions_config.py` module docstring (it explains the non-obvious reason the
write path is hand-rolled), short one-line notes where the code is genuinely
surprising, and the coordinate-space / tilt block at the top of both YAML
files, which the calibration tool is built around.

## 5. Things deliberately NOT removed

- **`test_recognition.py` / `test_decision.py`** - not runtime, but they are
  the only safety net, and `test_recognition` is the only thing that catches a
  regression like the `angle: 90` bug.
- **`calibrate.py` and `config/regions.schema.yaml`** - the schema is what
  `calibrate.py reset` regenerates from, and `calibrate.py` is the only way to
  edit `regions.yaml` without hand-writing pixel coordinates.
- **`templates/*_clusters.png`** - regenerable, but they record the labelling
  pass you already did; deleting them throws that away for no gain.
- **`is_dealt()`** - no longer used by the runtime reader (which uses
  `locate_card`), but `learn` and the test harness still use it.
- **Future-stage config keys** - removed as above, but `calibrate.py` will
  recreate any of them. If you want to start marking the FOLD button, add
  `buttons: {fold: {x: 0, y: 0, w: 0, h: 0}}` back and the tool handles it.

## 6. Two real bugs found while cleaning up

**`calibrate.py nudge` silently dropped the tilt.** It rebuilt the `Rect`
without carrying `angle`, so nudging the left hero card reset it to 0 and
re-broke recognition. Fixed by passing `rect.angle` through.

**`test_decision.py` had three failing tests, none of them code bugs:**

- `check("short stack is at least as loose as deep", chart_score("98o") * 0, True)`
  compares `0.0` to `True`, which is **always** false. The assertion was
  vacuous. Replaced with a real looseness comparison.
- The `HandTracker` river/new-hand block reused a tracker left over from an
  earlier block with `stable_frames=2` while asserting `stable_frames=3`
  timing, so the new-hand event fired one call early. Given its own tracker.
- `check("AA on K K Q is near 100%", ..., 1.0, tol=0.02)` was simply wrong.
  AA on a K K Q board is **not** a lock: two of the four kings are still live,
  and the opponent drawing one - or the runout - makes it trips against hero's
  two pair. Brute-force enumeration over all 47 live cards gives **0.8798**;
  the Monte Carlo returns 0.8792. Expectation corrected to ~88%.

## 7. A real improvement, not just deletion

`pick_corner_pip` picked the connected component *nearest below the rank
glyph*. On a fanned hero card the strip of felt leaking into the box
produces a tall thin sliver that is nearer the rank than the real pip, so the
sliver won and the pip went unnamed. Added a shape guard rejecting blobs
narrower than `MIN_RANK_W_OVER_H` of their height - the same rule the rank
picker already used.

Suit pips extracted from the 15 screenshots: **55 -> 68**, and leave-one-out
pip accuracy **54/55 -> 68/68**. This is why `Screenshot (35)`, `(36)` and
`(41)` now read their left card fully.

## 8. Current measured state

`python test_recognition.py`:

```text
end-to-end vs ground truth:  rank 68/68   suit 68/68
suit pip leave-one-out:      68/68 (100.0%)
```

`python poker.py verify` - all 15 screenshots read, zero problems, every card
matching the hand-written ground truth. `waiting-for-round-to-finish.png`
correctly reads no cards.

`python test_decision.py` - all decision and tracking tests pass.

`python poker.py doctor` - cv2, numpy, mss, yaml, treys all ok; all card
regions set; glyphs learned.

## 9. Known limitations (unchanged by this pass)

- **Preflop advice only.** `advise` returns `WAIT` on any board card.
- **`texts.to_call` is unset**, so `pot_odds` is always 0 and the equity
  comparison never fires in practice. Mark it to enable that branch.
- **No opponent reads.** No stacks, no villain ranges, no position.
- **Nothing is clicked.** No automation code exists at all now - `pytesseract`
  and `PyAutoGUI` were listed in `requirements.txt` and in `calibrate.py
  doctor` but were never imported by any pipeline code. `requirements.txt` now
  lists `PySide6` and `keyboard`, which the overlay *does* need and which were
  previously missing entirely.
- **Fixed skin.** The thresholds in `perception.py` (`PAPER_S_MAX`,
  `MIN_RANK_H_OVER_CARD`, the money HSV bounds) are tuned for this skin's
  colours. They are module constants, not config, on purpose.
- **Leave-one-out rank accuracy is 65/68, not 68/68.** That metric is a
  nearest-neighbour proxy on raw correlations; the real matcher keeps six
  exemplars per shape and applies a margin test, which is why end-to-end is
  perfect. Do not tune `match_score` against the proxy.
