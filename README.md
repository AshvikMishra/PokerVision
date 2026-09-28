# PokerVision

**A computer-vision pipeline that reads playing cards off a Monopoly Poker
table.**

Point it at the game window, drag a box around the table, and it finds every
card in that box and tells you what it is. Screen capture in, card codes out.

It is a vision project, not a poker project. There is no hand evaluator, no
equity model, no opening chart, no advice and no action. Nothing in this
repository decides what to do with a hand, and nothing ever sends a click to
the table. If you want a "what should I do" layer, build it on top of
`CardFinder` — that is what the clean boundary is for.

## Contents

- [How it works](#how-it-works) — the five stages, and the two read paths
- [Quick start](#quick-start)
- [Commands](#commands)
- [The GUI](#the-gui)
- [Layout](#layout)
- [Implementation notes](#implementation-notes)
- [Accuracy](#accuracy)
- [Limitations](#limitations)

## How it works

Five stages, each of which can be run and checked on its own.

```
   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌───────────┐   ┌──────────┐
   │ 1. capture   │──▶│ 2. find      │──▶│ 3. segment  │──▶│ 4. name   │──▶│ 5. group │
   │ a window's   │   │ candidate    │   │ the card     │   │ the rank  │   │ into     │
   │ client rect  │   │ card boxes   │   │ face         │   │ and the   │   │ board    │
   │              │   │ anywhere     │   │              │   │ suit pip  │   │ and hole │
   └──────────────┘   └──────────────┘   └──────────────┘   └───────────┘   └──────────┘
    mss + ctypes      ink blobs +        paper vs felt,     Pearson           row
    (windows.py,      geometry gates     deskew by the     correlation        clustering
     capture.py)                        match's own       against a bank
                                        angle             of your own glyphs
                                                           (perception.py,
                                                            detect.py)
```

### 1. Capture a window, not a screen

`windows.py` enumerates top-level windows through `EnumWindows` in pure
ctypes — no dependency, no injection, no accessibility API. For the window you
pick it reads the **client rect** (`GetClientRect` + `ClientToScreen`), so the
browser's tabs, URL bar and scrollbars are never in the frame.

The rect is re-read **every frame**, so moving or resizing the window needs no
reconfiguration. A window that closes raises `WindowLost` rather than quietly
grabbing whatever has since appeared at those coordinates, and a client rect
that has collapsed to zero (minimised) is refused rather than returned as an
empty frame.

### 2. Find candidate cards by search, not by position

The original approach hardcoded where the board lives. That breaks the moment
the window moves, the browser zoom changes, or the layout shifts. `detect.py`
instead hunts: every ink blob in the selected area is a *candidate* anchor for a
card's top-left corner, and two cheap gates throw out almost all of them.

- the implied card box must be at least `MIN_FACE_COVER` paper — this is what
  separates a card from a white chat bubble
- a pip-sized blob must sit under the rank, checked on the frame's own ink map
  rather than on a fresh crop, so it costs one small connected-components pass

The second gate is a *necessary* condition rather than a heuristic: a card
prints its rank over its suit, so `pick_corner_pip` cannot name a card without
a pip under the rank. Together the gates cut candidates by about three
quarters while keeping every card the ungated search could read.

### 3. Segment the card face

Paper is separated from felt in HSV — light **and** unsaturated. Brightness is
required as well as low saturation because over a whole screenshot every dark
desaturated pixel (page background, sidebar) also passes a saturation-only test
and would then be read as ink. Holes are filled, which is load-bearing rather
than cosmetic: ink is darker than paper, so an unpunched interior reads as ink
too.

The card height is **measured**, not assumed — the pipeline walks down the
anchor's column until paper ends, clamped to a plausible range. This is what
makes the tilted hole card work: `pick_rank_glyph` will only accept a rank that
occupies a minimum fraction of the crop, so a crop that is even slightly wrong
loses the card, and dividing by a nominal ratio is not accurate enough (a blob
one part in fifty too tall drops a 0.30 ratio to 0.294, under the floor).

### 4. Name the glyphs

The rank glyph and the corner suit pip are isolated, normalised, and
**correlated** against a bank of prototypes. The bank is learned from your own
screenshots by `learn` — no template pack ships with this repository and nothing
is uploaded anywhere.

Two-stage on purpose. A naive "search the whole image for something that looks
like a 6" is ambiguous in a way that is not fixable by tuning: 6, 9 and 3 were
all being read as 8. Locating the card first and then reading the glyphs inside
it removes the ambiguity entirely, because by then there is only one candidate
answer.

Matching is vectorised — the prototypes are stacked into one matrix so a read is
a single matrix product rather than a Python loop of 83 correlations.

### 5. Group into board and hole

`group_rows` clusters detections into rows by y-centre and takes the **lowest
row of at most two cards** as the hero's, everything above as the board, sorted
left to right. This is the one geometric guess in the pipeline, and the GUI
says so when it cannot be made.

## Two read paths

Both exist, both are tested, and they are deliberately different.

| | calibrated | search-based |
|---|---|---|
| where cards are | fixed rects in `config/regions.yaml` | found anywhere in a selected area |
| set up by | `calibrate.py` | drag a box in the GUI |
| accuracy | **68/68** ranks and suits | **55/68**, 0 false positives |
| speed | per-region, milliseconds | 33 ms a frame over the whole area |
| used by | `watch`, `overlay`, `verify` | the GUI |

The calibrated path is the ground truth and the fallback: you mark where the
board is once, and it reads perfectly. The search path is what makes the GUI
usable without calibration, and it is the one with known gaps.

## Quick start

```powershell
pip install -r requirements.txt
```

Python 3.12, Windows. Everything runs offline — there is no API call, no model
download and no telemetry in this project.

```powershell
python -m pokervision doctor      # check deps and calibration state
python -m pokervision gui         # pick a window, drag the table area, read live
```

## Commands

```powershell
python -m pokervision doctor                                  # deps + calibration state
python -m pokervision gui                                     # select an area, read it live
python -m pokervision verify --images "unmarked-images/*.png" # read the labelled screenshots
python -m pokervision learn  --images unmarked-images         # one-time glyph bootstrap
python -m pokervision watch                                   # live loop, prints each new read
python -m pokervision overlay                                 # live click-through boxes
```

`watch` and `overlay` capture a window by default. With no flag they print a
numbered list of open windows and wait for a choice; a blank answer, a bad
number, Ctrl-C or EOF all fall back to the whole monitor rather than failing.

```powershell
python -m pokervision watch --window 1            # by number, no prompt
python -m pokervision watch --window "Monopoly"   # by title substring
python -m pokervision watch --monitor 0           # whole desktop, no prompt
```

## The GUI

```powershell
python -m pokervision gui
```

- **Selection** lists the open windows largest-first — each showing how much
  *chrome* it has, so a browser window is obvious — then opens a snapshot of
  the one you pick, scaled to fit your screen, and lets you drag a box around
  the table. The rect is shown back to you and the band stays on screen before
  you commit to it. It is stored as a **fraction** of that window's client rect
  in `config/selections.json`, keyed by window title, so it survives moving and
  resizing the window and each window keeps its own.
- **Pause / Resume** freezes and restarts the live read so a selection can be
  checked without closing anything.

A **Saved game areas** panel lists what is stored as a percentage of its window,
with **Use**, **Forget** and **Forget all**. *Use* finds the window again by
title, so a saved area comes back without re-dragging.

The status line always names exactly what is being captured —
`capturing window 0x1234 client 1920x1080 at (0,0) | game area 1300x560 at
(310,490)` — so it is never a guess whether the window or the whole monitor is
in use.

Cards are boxed in the preview: green for the board row, amber for the hero's,
each labelled with the code it read. Underneath is a **confidence line** that
reports what is trustworthy about the read rather than what to do about it. It
turns amber when the hero's row could not be identified or only one hole card
was found, because a partial read is not a read.

## Layout

```text
pokervision/
  __main__.py      CLI entrypoint: python -m pokervision <command>
  cli.py           the commands, and the console window picker
  windows.py       win32 window list + client rect (pure ctypes)
  capture.py       mss grab; WindowCapture re-queries the client rect each frame
  detect.py        search-based CardFinder: find cards, read them, group them
  perception.py    the calibrated path: locate_card -> rank glyph / suit pip
  learning.py      glyph clustering and labelling (`learn`)
  state.py         Reading: what one frame said, and a stability filter
  regions.py       config/regions.yaml schema, loader, comment-preserving editor
  config.py        regions.yaml -> Config
  gui.py           the two-button window
  overlay.py       the live click-through overlay
calibrate.py       calibration tool and regions.yaml editor
config/
  regions.yaml          the calibrated regions
  regions.schema.yaml   the blank template `calibrate.py reset` restores
  selections.json       per-window game areas, written by the GUI (gitignored)
templates/              learned glyph and digit banks (+ labelling montages)
unmarked-images/        15 labelled screenshots: the ground truth
tests/                  every test script
run_tests.py            run them all
CONTEXT.md              how it works, and what was tried and rejected
```

## Implementation notes

**Coordinate space.** Every rect in `regions.yaml` is `{x, y, w, h}` in
**absolute virtual-desktop pixels**, top-left origin — the same grid as the PNGs
in `unmarked-images/`. Never normalized fractions. `{0,0,0,0}` means "not
calibrated yet" and is skipped rather than treated as an error. An optional
`angle` deskews the crop about its own centre, **positive = counter-clockwise**.

**The tilt.** The left hero card is printed about 5° counter-clockwise on this
table. It is read by deskewing the crop, not by rotating the search, and the
angle lives in the region config because it is a property of the table, not of
the algorithm. A card read at a non-zero angle is reported with that angle
attached, so the read can be reproduced from the rect alone.

**Speed.** Detection runs at **33 ms a frame** (30 fps) over the whole selected
area, and 79 ms even searching an entire 1920×1080 desktop. It started at
1.9 s. Two changes did it, and both were about doing the same work repeatedly:

1. the glyph bank became one matrix product instead of a Python loop of 83
   correlations;
2. segmentation runs **once per candidate** instead of once per rotation — the
   stock chain called `card_ink_soft` 49 times per candidate, and hoisting it
   out is worth the rest, with the upright path staying bit-identical because a
   0° warp is the identity.

The frame-level maps are vectorised too, and the ink threshold reads its
cut-offs back out of the same lookup tables the readable `ink_map` uses, so the
fast path and the slow path cannot drift apart. `tests/test_ink.py` asserts that
equivalence directly, on every screenshot and across the degenerate inputs,
because the two must agree and nothing else would notice if they stopped.

## Accuracy

Measured by `tests/test_detect.py` and `tests/test_recognition.py` against
hand-labelled ground truth in `unmarked-images/`, and re-measured on every test
run so they cannot silently regress.

| path | correct | missed | false positives | speed |
|---|---|---|---|---|
| calibrated | **68/68** | 0 | 0 | per region |
| search, table region | 55/68 | 5 | **0** | 33 ms · 30 fps |
| search, whole screenshot | 55/68 | 4 | **0** | 79 ms · 13 fps |

**Zero false positives is the number that matters here.** A reader that invents
a card is worse than one that misses one, because a miss is visible and an
invention is not.

The whole-screenshot row is a stress test rather than a target — it searches the
entire desktop including browser chrome and the taskbar.

## Limitations

**The search path misses 13 of 68 cards.** Known causes, all documented in
[CONTEXT.md](CONTEXT.md#9-what-dynamic-detection-still-gets-wrong):

- **Dimmed winner-reveal cards.** A dimmed row has dull pink paper and
  compressed ink contrast, and the paper reference is one global median, so the
  whole row falls under the threshold and disappears. Referencing each card
  against its own local brightness fixes it in principle and **measured worse**
  (45/68 against 55/68) because the local estimate bleeds into the felt.
- **A 'Q' whose tail touches its corner pip.** Same mechanism: the two blobs
  merge and the rank no longer matches.
- **A few thin glyphs** ('5', 'T') that lose enough area to slip under the
  blob-size floor.

**Tuned for one skin and one zoom.** The thresholds in `perception.py` are
module constants on purpose rather than config: they are the result of
measurement, and exposing them invites tuning them against the test set rather
than against new screenshots. A different table theme or browser zoom is a
re-learn, not a config change.

**Money reading is calibrated-path only.** Pot and stack are read from
calibrated boxes using a learned digit bank. The search path does not attempt
them.

**No hand semantics.** No evaluation, no equity, no opponent reads, no
position, no ranges. `group_rows` splits two rows and nothing more; a
board-only read is reported as board, and the pipeline does not claim to know
whether a five-card row is a river or a five-card board.

**Windows only.** `windows.py` is `ctypes` against `user32`/`dwmapi`. There is
no X11 or Wayland path.

**Screen capture only.** It reads what the compositor hands over. It cannot see
a card that is not on screen, and it cannot read anything through an occluding
window.

**Reading the game may be against its terms.** That is a judgement for the
person running it, not a technical limitation — but worth stating plainly
rather than burying.

## Tests

```powershell
python run_tests.py                    # all of them, ~25s
python run_tests.py detect recognition # a subset
python tests/test_detect.py            # one directly
```

| script | covers |
|---|---|
| `tests/test_capture.py` | window enumeration, client-rect capture, move/resize, dead handles, console-safe output for real titles |
| `tests/test_cli.py` | the console window picker, every fallback, flag wiring |
| `tests/test_gui.py` | the drag in every shape, coordinate mapping, persistence, the confidence readout |
| `tests/test_ink.py` | the fast ink path still agrees with the readable one |
| `tests/test_detect.py` | search accuracy vs. ground truth, plus a timing report |
| `tests/test_recognition.py` | the calibrated path, every card vs. ground truth |
| `tests/test_reading.py` | the `Reading` shape, and the N-frame stability filter `watch` depends on |

They resolve the repository root themselves, so they run from any working
directory.
