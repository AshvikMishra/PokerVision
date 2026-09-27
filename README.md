# Monopoly Poker table assistant

A local, offline poker table reader and decision aid for Monopoly Poker. It is
built around screen capture, OpenCV glyph matching and deterministic poker
math, not a general ML model or an external service.

**Observe only.** It reads the table and prints advice. It does not click
buttons or act on the table.

See [CONTEXT.md](CONTEXT.md) for how the pipeline fits together and what was
changed in the last cleanup pass.

## What it does

- capture a desktop frame with `mss`
- read calibrated board and hero-card regions from `config/regions.yaml`
- deskew any region that carries an `angle`, so a tilted card is straightened
  before it is read
- detect card paper vs. felt with OpenCV and HSV filtering
- isolate the rank glyph and suit pip on each card
- name cards by correlating against a glyph bank learned from your own
  screenshots - no template pack ships with the repo and nothing is uploaded
- read pot and stack from money boxes with learned digit shapes
- assemble a hand state and emit preflop advice from an opening chart, Monte
  Carlo equity and pot odds
- run headless against saved images, or live as a click-through overlay

## Setup

```powershell
pip install -r requirements.txt
```

Python 3.12. Everything runs offline with no API dependency.

## Commands

```powershell
python poker.py doctor                              # deps + calibration state
python poker.py verify --images "unmarked-images/*.png"
python poker.py learn  --images unmarked-images     # one-time glyph bootstrap
python poker.py watch                               # live loop, prints advice
python poker.py overlay                             # live click-through window
```

Calibration, whenever the table layout shifts:

```powershell
python calibrate.py list
python calibrate.py grab --name table
python calibrate.py mark --image samples/table.png --set hero.hole_cards
python calibrate.py set  texts.pot 912 480 158 48
python calibrate.py nudge texts.pot --dx 3 --dy -1
python calibrate.py verify --image samples/table.png
```

`mark` opens the screenshot and you drag a box over each thing in the order
listed. Those commands rewrite only the line they touch, so the comments in
`regions.yaml` survive.

## Tests

```powershell
python test_recognition.py   # every card vs. hand-written ground truth
python test_decision.py      # equity, advice, hand tracking - no screen needed
```

Current state: **68/68 ranks and 68/68 suits correct** end to end against the
15 labelled screenshots.

## Layout

```text
CONTEXT.md                   how it works + what changed
calibrate.py                 calibration GUI and regions.yaml editing
regions_config.py            config schema, loader, comment-preserving editor
config/regions.yaml          the calibrated regions for this table
config/regions.schema.yaml   the blank template `calibrate.py reset` restores
poker.py                     CLI entrypoint and compatibility facade
pokerbot/                    the implementation
  config.py                    regions.yaml -> Config
  capture.py                   mss grab, ROI crop, tilt deskew
  perception.py                card/glyph/money recognition
  learning.py                  glyph clustering and labelling
  state.py                     hand state and new-hand tracking
  decision.py                  hand class, equity, advice
  overlay.py                   live Qt overlay
  cli.py                       command implementations
templates/                   learned glyph and digit banks (+ labelling montages)
unmarked-images/            the 15 labelled screenshots
test_recognition.py
test_decision.py
requirements.txt
```

## Coordinate space

Every rect in `regions.yaml` is `{x, y, w, h}` in **absolute virtual-desktop
pixels**, top-left origin - the same grid as the PNGs in `unmarked-images/`.
Never normalized fractions.

`{x: 0, y: 0, w: 0, h: 0}` means "not calibrated yet"; perception skips
zero-sized regions rather than failing.

An optional `angle` deskews the crop, in degrees, about the crop's own
centre. **Positive is counter-clockwise.** Use it for a card that sits rotated
on the table:

```yaml
- {x: 905, y: 740, w: 104, h: 124, angle: 5}
```

## Limits

- Preflop advice only; postflop returns `WAIT`.
- `texts.to_call` is unset, so the pot-odds branch never fires.
- No opponent reads: no villain stacks, ranges or position.
- Tuned for one fixed skin. The thresholds in `perception.py` are module
  constants on purpose, not config.
