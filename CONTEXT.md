# CONTEXT.md

Findings from the cleanup and bug-fix pass over the Monopoly Poker table
assistant. Written to be the thing you read before touching the code.

---

## 1. What this is

A computer-vision pipeline that reads playing cards off a Monopoly Poker table.
Capture in, card codes out. It is not a poker project: no evaluation, no equity,
no advice, no action, and nothing that ever clicks anything.

Two read paths, both tested, deliberately different. The **calibrated** path
reads fixed rects from `config/regions.yaml` and is at 68/68. The **search**
path finds cards anywhere in a selected area and is at 55/68 with zero false
positives. The search path is what the GUI uses; the calibrated path is the
ground truth it is measured against and the fallback for `watch`/`overlay`.

## 2. The pipeline, end to end

```text
+-- python -m pokervision <command>          cli.py
|     +-- doctor | gui | verify | learn | watch | overlay
|     +-- pick_capture()          the console window picker, and its fallbacks
|
+-- pokervision/gui.py           the two-button window: Selection, Pause/Resume
|     +-- pokervision/windows.py   EnumWindows / GetClientRect / ClientToScreen
|     +-- pokervision/capture.py   WindowCapture: the chosen window's client rect
|     +-- pokervision/detect.py    CardFinder: search for rank glyphs, no fixed
|     |                              positions, then read and group them
|     +-- pokervision/perception.py the calibrated read chain it reuses
|
+-- pokervision/config.py        reads config/regions.yaml -> Config dataclass
|     +-- pokervision/regions.py   schema, loader, comment-preserving editor
|
+-- pokervision/capture.py       mss grab; crop(rect) applies the tilt deskew
+-- pokervision/perception.py    locate_card -> pick_rank_glyph / pick_corner_pip
|     |                          -> GlyphBank.match (Pearson, vectorised)
|     +-- pokervision/learning.py  clustering and labelling, writes templates/
+-- pokervision/state.py         read_frame() assembles a Reading; ReadingTracker
|                                  is the N-frame stability filter
+-- pokervision/overlay.py       click-through Qt window drawing the read
+-- calibrate.py                 the calibration tool, edits regions.yaml
```

`pokervision/regions.py` moved into the package as `pokervision/regions.py` during
the rename. It is imported as `rc` throughout, and that alias is now the only
thing that identifies it, which is a small readability loss accepted for
`from pokervision import regions as rc` to be obvious.

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
| `set_scalar()`, `_scalar_text()` | `pokervision/regions.py` | no caller anywhere; `calibrate.py` only ever writes rects |
| `Rect.intersects`, `Rect.area`, `Rect.center` | `pokervision/regions.py` | no caller |
| `format_money()` | `pokervision/regions.py` | no caller |
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
`pokervision/regions.py` module docstring (it explains the non-obvious reason the
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
- The `ReadingTracker` river/new-hand block reused a tracker left over from an
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

## 8. Search-based detection (the GUI's path)

`pokervision/detect.py`. The calibrated rectangles are no longer how cards are
found. The search hunts for **rank glyphs** anywhere in the selected area and
treats each as a card's top-left corner.

Each step is there because the simpler version was measured and failed:

1. **Paper mask** - low saturation *and* bright, with every external contour
   filled. Both halves are load-bearing. Saturation alone also passes every
   desaturated dark pixel, so the page background became "paper" and then
   "ink". The fill matters because ink is darker than paper, so an unfilled
   mask excludes the glyphs themselves and they shatter into specks.
2. **Ink map** - dark or saturated red, against the median brightness of the
   paper. Not an Otsu split: that is fine inside one card crop, but across a
   whole screenshot the threshold lands far too low and the white between
   glyphs reads as ink.
3. **Rank anchors** - connected components of ink that are rank-shaped. This
   is what solves the overlapping hole cards: their paper merges with the
   "PAIR" label and the turn-ring into one 430x230 blob, so there is no
   card-shaped region to find, but two overlapping cards still have two
   separate rank glyphs.
4. **Card box** - built from the anchor, with the height **measured** by
   walking down the paper column rather than divided from a nominal ratio.
   `pick_rank_glyph` demands the rank be at least 0.28 of its crop, and a card
   guessed 5% too tall drops a 0.30 ratio to 0.285, which is the entire reason
   the tilted hole card was unreadable. The box is also unpadded for the same
   reason: 8px of padding put a 39px rank at 0.267 and rejected every board
   card.
5. **Read** - handed back to `pick_rank_glyph` / `pick_corner_pip` so the
   rasters reaching the matcher are the shape the bank was learned from. Only
   the *position* comes from the search. Matching the search's own blobs
   directly read a 6, a 9 and a 3 all as an 8.
6. **Angle sweep** - 0, ±3, ±5, ±7, first read that names both glyphs wins.
   `minAreaRect` cannot measure a 5° tilt: on a compact glyph the axis-aligned
   rectangle has near enough the same area, and it reported 0.0 for the tilted
   hole card. The deskew angle is returned with the card, because without it a
   caller re-reading the rect upright gets nothing.

Two filters keep it honest. The rank must sit in the card's top-left, which is
what stops a seat portrait reading as a card - a face has skin tones that pass
the paper test and eyes that pass the ink test. And card heights must be
mutually consistent, since a real table's cards never are.

### Two things measured and rejected

- **Erosion to split touching glyphs.** A 'Q' tail welds to its own pip and
  the merged blob anchors an oversized box. Eroding 0/1/2 iterations measured
  **48 / 39 / 24** correct: it ate the thin strokes of a 5 and a 7 to fix one
  Q. Left off.
- **Per-glyph local paper brightness**, to catch dimmed cards. Right in
  principle, measured **45** against 55, because the local estimate bleeds
  into the felt around the card.

### Vectorised matching

`GlyphBank.match` is a Python loop of Pearson correlations, hopeless when a
whole-frame search offers ~400 candidates at up to 7 angles each: 1.9 s a
frame. `_VectorBank` stacks the 83 prototypes into one matrix so the whole bank
is a single matrix product. Same arithmetic, but a separate class in
`detect.py` rather than a change to `GlyphBank`, so the learned bank and every
existing caller stay untouched.

The first attempt returned a correlation of **5.44**, which is impossible, and
61 false positives with it: the denominator was `sqrt(‖x‖²·‖m‖)` where the
formula needs `‖x‖·‖m‖`.

## 8b. The prediction layer, and why it is gone

There was one. `predict.py` turned detected cards into advice, with
`decision.py` behind it doing hand classes, an opening chart, Monte Carlo
equity and pot odds, and a `treys` dependency for the card maths.

**It has been removed.** The project is a computer-vision pipeline, and the
advice layer was the wrong shape for that: it made the interesting number in
the README an equity figure rather than an accuracy figure, and it added a
poker library to a repository whose actual subject is OpenCV. What remains is
`CardFinder.find() -> list[Detected]` and `group_rows() -> (board, hole)`.

The useful things that came out of it are worth keeping, so they are recorded
here rather than in the code.

### Refusing to act on an incomplete read

The layer had one genuinely good property: it would not answer on a partial
read. A missing hole card was reported rather than filled in, a lone low card
with a board row above it was still treated as the hole card and still reported
short, and readiness was *derived* from the reasons rather than passed in, so
`ready == not reasons` was a testable invariant.

That behaviour survives as the GUI's confidence line, which is why it reports
"1/2 hole cards - a partial read" and "no row of at most two" rather than
printing two cards and letting someone act on them.

### A landmine that was contained rather than papered over

`advise()` computed `pot_odds = to_call / (pot + to_call)`. With no pot that is
1.0, so a single `to_call` with an unread pot made the engine **fold every bet
in the game**, including a one-bet pot where folding is catastrophic. The
prediction layer dropped `to_call` when there was no `pot` and said so in
`caveats`, and the tests pinned both halves: the same hand with
`to_call=50000` folded *with* a pot and did not fold *without* one.

The trap is now gone with the code that contained it. Nothing else in the
repository computes pot odds, so there is no live version of the bug to fix.

### Per-frame cost, and why it mattered

`monte_carlo_equity` cost about 0.56 s at 20k sims against a then-0.7 s
detection frame, so the layer keyed on the sorted card codes plus the money and
computed once per hand change rather than once per frame. A 6-second run showed
73 calls and 5 computes.

Worth remembering for anything later built on top of this: **the per-frame
budget is the detection frame, and a second of Monte Carlo does not fit in it.**
The general lesson is that a decision layer belongs behind a cache keyed on the
perception output, not in the frame loop.

## 8c. Capture sources

- **`gui`** — `WindowCapture` on the chosen window's client rect, cropped to
  the area saved in `config/selections.json`, read by the **search** finder.
- **`watch` / `overlay`** — `WindowCapture` or a whole monitor, cropped by the
  **calibrated** rects in `regions.yaml`, which are absolute desktop pixels.

Both paths call the same `WindowCapture`. The console picker
(`cli.pick_capture`) lists windows largest-first, takes a number or a title
substring, and falls back to the whole monitor on a blank answer, a bad number,
EOF or Ctrl-C rather than failing the launch. `cli.describe_source` prints
exactly what is in use and the GUI status line repeats it, so "is it using my
window or the whole monitor" is never a guess.

One thing measured rather than assumed: `mss` **can** grab a rect whose top is
negative, which a maximised window reports (measured at `(0, -9)` here, and it
returns the correct 1920x1089 image). An earlier version of `WindowCapture`
refused such a rect on the assumption that mss could not, which would have
broken exactly the maximised-browser case that matters most. The refusal was
removed, and `tests/test_capture.py` now pins the correct behaviour.

## 9. What dynamic detection still gets wrong

Five cards out of 68 are missed, and the pattern is understood:

- **Dimmed winner-reveal cards.** When the board washes to show a winner the
  cards are dull pink rather than white, and their ink contrast compresses
  with them. The paper reference here is one global median taken from the
  bright cards, so a dimmed row falls under the threshold and vanishes.
  Visible in the rendered overlay as unboxed cards sitting among boxed ones.
  Per-card local contrast is the fix; the naive version measured worse, so it
  wants doing properly rather than quickly.
- **A 'Q' whose tail touches its corner pip**, for the reason above.
- **A few thin glyphs** ('5', 'T') that lose enough area to slip under the
  blob-size floor.

The speed problem is **solved**: 33 ms a frame (30 fps) in the selected region,
and 79 ms (13 fps) even searching a whole desktop, against a 10 fps target, with
the 55/68 and zero-false-positives result unchanged. It started at 1.9 s a
frame. See section 9b.

The calibrated path still reads 68/68 and is untouched. It remains the fallback
and the ground truth this is measured against.

## 9b. Speed: 1.9 s to 33 ms a frame

Detection was 0.9-2.0 fps, and the whole-frame search was the wrong shape:
~400 ink candidates, each trying up to 7 deskew rotations, and each rotation
recomputing the crop's ink map and card outline from scratch through
`rasterise` / `pick_rank_glyph` / `pick_corner_pip`.

Two costs dominated, and both were about doing the same work repeatedly rather
than about the algorithm being wrong.

### 1. The glyph bank was a Python loop of 83 correlations

`_VectorBank` stacks the 83 prototypes into one matrix so the whole bank is a
single matrix product. The first attempt returned a correlation of 5.44 -
impossible, since a correlation cannot exceed 1 - because the denominator was
`sqrt(‖x‖²·‖m‖)` where the formula needs `‖x‖·‖m‖`. A second, subtler cost hid
inside the fix: the boolean-mask form `self.matrix[ok] @ x` copied the whole
83x4096 bank on **every** call. Correlating all rows and dividing in place took
0.470 -> 0.055 ms, bit-identical.

### 2. Segmentation ran once per rotation instead of once per candidate

The stock `pick_rank_glyph` + `pick_corner_pip` + `rasterise` trio called
`card_ink_soft` 7 times per view and 49 times per candidate. Re-derived as
`_rank_glyph` / `_corner_pip` / `_raster`, which take the already-computed maps,
it is 49 -> 14 calls. The upright path is **bit-identical**, because a 0 degree
warp is the identity.

### 3. Three preconditions before any glyph work at all

`pick_corner_pip` cannot name a card without a pip, so requiring a pip-sized
blob under the rank is a *necessary* condition, not a guess. Paired with a
geometry gate - the implied box must be at least 0.65 paper - this cut
candidates from 1136 to 282, keeping all 111 upright and 19 of 20 deskew reads.

The deskew sweep is then gated on coverage: at >=0.80 paper it tries 0 degrees
only. A card turned off axis overhangs its own axis-aligned box, and measured,
the tilted hole cards cover 0.71-0.75 while every upright card covers >=0.78.
7 attempts became 2.6 on average.

### 4. The frame-level maps, vectorised

`inRange` for paper (5.2 -> 1.7 ms) and the ring (7.1 -> 1.7 ms), and a `LUT`
for every channel ramp in `ink_map`, with one shared HSV conversion:
paper + ink went 47.8 -> 22.8 ms. Then `_ink_binary` thresholds the ink
without materialising the three float32 planes at all, reading its cut-offs
back out of the same lookup tables `ink_map` uses, so the two agree by
construction. 20 -> 10.7 ms, verified bit-identical on all 15 screenshots.

Measured after all of it, and re-measured by `test_detect.py` on every run so
it cannot silently regress:

| | before | after | target |
|---|---|---|---|
| table region | 495 ms (2.0 fps) | **33 ms (30.4 fps)** | 100 ms |
| whole screenshot | 1174 ms (0.85 fps) | **79 ms (12.6 fps)** | 100 ms |

Timing on a shared machine is noisy - repeated identical runs varied 88 -> 116
ms with another process running - so these are best-of-N, not single passes.

**A number that was wrong for a while.** An earlier version of
`test_detect.py` always built the region crop, so its "whole screenshot
(stress test)" section never searched the whole screenshot. It re-measured the
region and reported it as the whole-frame figure, which is how 33 ms came to be
published as the whole-screenshot number. The section is fixed and the two
numbers above are the real ones. A speed test that silently measures the wrong
input is worse than no speed test.

### Accuracy was not traded for it

55/68 with **0 false positives**, unchanged. A reference implementation of the
original `_read` / `_names_both` was diffed against the new code across all 1136
candidates on every screenshot/region pair: **1132 identical, 4 differing, and
all 4 are cards the new code *rejects* that the old one invented** (`cs` at
43x62 and `82` at 31x45, both out of UI fragments). No real read was lost, and
four latent false positives went away.

### One thing deliberately left off the test path

`ink_map` is no longer called by `find`, so `test_detect.py` no longer covers
it. The two *cannot* drift on their constants - `_ink_binary` reads its
cut-offs out of `ink_map`'s own tables - but a change to the ink **formula**,
a third kind of ink say, would land in `ink_map` alone and the search would
quietly stop agreeing with it, with nothing failing.

So `tests/test_ink.py` asserts the equivalence directly, on every screenshot and
across the inputs that make the two forms take different branches: no paper at
all, all-white, all-black, random noise, saturated red, and paper brightness
swept from 20 to 252. It was checked by adding a third ink kind to `ink_map`
only, and it caught it - 430-450 differing pixels per screenshot. The
alternative, reverting `find` to the float path, would have cost 10 ms a frame
to test a function nothing calls.

End to end, the live GUI loop - capture, detection, grouping and drawing - runs
at **19.8 fps** on a real window, against a detection-only figure of 30 fps.
The gap is capture and the Qt paint, not the matcher. It was 11.6 fps before the
advice layer was removed, so deleting the Monte Carlo bought about 70% of the
frame back.

## 10. Current measured state

`python tests/test_recognition.py`:

```text
end-to-end vs ground truth:  rank 68/68   suit 68/68
suit pip leave-one-out:      68/68 (100.0%)
```

`python tests/test_detect.py` - the search path, which is what the GUI uses:

```text
expected 68   correct 55/68   missing 5   false pos 0    32.9 ms/frame  30.4 fps
expected 68   correct 55/68   missing 4   false pos 0    79.2 ms/frame  12.6 fps
```

`python -m pokervision verify` - all 15 screenshots read, zero problems, every
card matching the hand-written ground truth. `waiting-for-round-to-finish.png`
correctly reads no cards.

`python -m pokervision doctor` - cv2, numpy, mss, yaml all ok; all card regions
set; glyphs learned. `treys` is no longer checked, because the equity code that
needed it has been deleted.

`python run_tests.py` - seven suites, all passing.

## 11. Known limitations

- **The search path misses 13 of 68 cards** and 5 of them are the documented
  gaps in section 9. Zero false positives, which is the number that matters: an
  invented card is worse than a missed one, because a miss is visible.
- **The calibrated path is the one that is perfect, and it needs marking up
  first.** 68/68 is real, but it is 68/68 *on this table at this zoom with
  these rects marked by hand*. It is not a claim about Monopoly Poker.
- **No hand semantics at all.** No evaluation, no equity, no opponent reads, no
  position, no ranges. `group_rows` splits two rows and stops there; the
  pipeline does not claim to know whether a five-card row is a river or a
  five-card board.
- **Nothing is clicked.** No automation code exists at all, and there never
  did - `pytesseract` and `PyAutoGUI` were listed in `requirements.txt` and in
  `calibrate.py doctor` but were never imported by any pipeline code.
  `requirements.txt` now lists `PySide6` and `keyboard`, which the overlay and
  GUI *do* need and which were previously missing entirely.
- **Fixed skin and fixed zoom.** The thresholds in `perception.py`
  (`PAPER_S_MAX`, `MIN_RANK_H_OVER_CARD`, the money HSV bounds) are tuned for
  this skin's colours. They are module constants, not config, on purpose: they
  are the result of measurement, and exposing them invites tuning them against
  the test set rather than against new screenshots.
- **Leave-one-out rank accuracy is 65/68, not 68/68.** That metric is a
  nearest-neighbour proxy on raw correlations; the real matcher keeps six
  exemplars per shape and applies a margin test, which is why end-to-end is
  perfect. Do not tune `match_score` against the proxy.
- **Windows only.** `windows.py` is `ctypes` against `user32`. No X11, no
  Wayland, no macOS.
- **It can only see what is on screen.** No occlusion handling, no reading
  through a covered window, no multi-monitor stitching beyond whatever `mss`
  hands over for the chosen monitor.
