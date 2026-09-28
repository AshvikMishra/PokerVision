#!/usr/bin/env python
"""Calibration tool for config/regions.yaml.

  python calibrate.py doctor                       check deps, show monitors
  python calibrate.py grab --name table            save a screenshot to samples/
  python calibrate.py list                         coverage report
  python calibrate.py mark --image samples/table.png --set hero.hole_cards
  python calibrate.py set texts.pot 912 480 158 48
  python calibrate.py nudge texts.pot --dx 3 --dy -1
  python calibrate.py clear texts.pot
  python calibrate.py verify --image samples/table.png
  python calibrate.py reset                        regenerate from the schema
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from pokervision import regions as rc

REPO_ROOT = Path(__file__).resolve().parent
SAMPLES = REPO_ROOT / "samples"
CONFIG = rc.DEFAULT_CONFIG

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (120, 220, 120)
CYAN = (255, 220, 80)
YELLOW = (60, 230, 255)
GREY = (150, 150, 150)
BAR_BG = (38, 34, 30)


def _label(img, text, org, colour, scale=0.45, thickness=1):
    h, w = img.shape[:2]
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thickness)
    x = max(2, min(org[0], w - tw - 4))
    y = max(th + base + 2, min(org[1], h - 3))
    cv2.rectangle(img, (x - 2, y - th - base - 2), (x + tw + 2, y + 2), (20, 18, 16), -1)
    cv2.putText(img, text, (x, y), FONT, scale, colour, thickness, cv2.LINE_AA)


def _grid_step(scale: float, min_px: int = 46) -> int:
    target = min_px / max(scale, 1e-6)
    for exponent in range(7):
        for mult in (1, 2, 5):
            if mult * 10**exponent >= target:
                return mult * 10**exponent
    return 10**6


def _expand_targets(config: dict, specs: list[str]) -> list[str]:
    out: list[str] = []
    for spec in specs:
        node = rc.resolve(config, spec)
        segments = rc.parse_path(spec)
        if isinstance(node, list):
            out.extend(rc.format_path(segments + [i]) for i in range(len(node)))
        else:
            out.append(spec)
    return out


def _load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"could not read image: {path}")
    return img


def _screen_size() -> tuple[int, int]:
    try:
        import mss

        with mss.mss() as sct:
            mon = sct.monitors[1]
            return int(mon["width"]), int(mon["height"])
    except Exception:
        return 1920, 1080


def cmd_doctor(args) -> int:
    print(f"python      {sys.version.split()[0]}  ({REPO_ROOT})\n")
    print("dependencies")
    for module, why in [
        ("yaml", "config/regions.yaml"),
        ("cv2", "calibration GUI + template matching"),
        ("mss", "screen capture"),
        ("numpy", "image arrays"),
    ]:
        try:
            mod = __import__(module)
            print(f"  [ok]   {module:12s} {getattr(mod, '__version__', '?'):12s} {why}")
        except Exception as exc:
            print(f"  [MISS] {module:12s} {'':12s} {why}  ({exc.__class__.__name__})")

    print("\nmonitors (mss pixel rectangles - regions.yaml uses these coordinates)")
    for index, rect in rc.mon_rects():
        tag = "virtual desktop" if index == 0 else f"display {index}"
        print(f"  {index}: {rect.as_flow()}  {tag}")
    active = rc.load(CONFIG).get("capture", {}).get("monitor", 0)
    print(f"\nconfig      {CONFIG}")
    print(f"  capture.monitor = {active}")
    return 0


def cmd_grab(args) -> int:
    import mss

    config = rc.load(CONFIG)
    index = args.monitor if args.monitor is not None else int(config.get("capture", {}).get("monitor", 0))
    with mss.mss() as sct:
        monitors = {i: m for i, m in enumerate(sct.monitors)}
        if index not in monitors:
            raise SystemExit(f"no monitor {index}; run `python calibrate.py doctor`")
        monitor = monitors[index]
        shot = np.asanyarray(sct.grab(monitor))
    frame = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)

    SAMPLES.mkdir(exist_ok=True)
    stem = args.name or datetime.now().strftime("table-%Y%m%d-%H%M%S")
    out = SAMPLES / f"{stem}.png"
    cv2.imwrite(str(out), frame)
    print(f"saved {out}  ({frame.shape[1]}x{frame.shape[0]})  from monitor {index} {monitor}")
    if index != 0:
        print(f"  note: monitor {index} starts at ({monitor['left']}, {monitor['top']}) on the")
        print(f"        virtual desktop. regions.yaml stores absolute desktop pixels, so if you")
        print(f"        calibrated from a full-desktop grab, offsets differ by that much.")
    return 0


def cmd_list(args) -> int:
    config = rc.load(CONFIG)
    screen = rc.capture_rect(config)
    if screen is None:
        print("  (could not query monitors)")
    print(f"{'region':34s} {'rect':34s} state")
    print("-" * 88)
    all_rects = rc.rects(config)
    unset = rc.unset_paths(config)
    for path, rect in all_rects.items():
        state = "set" if rect.is_set else "UNSET"
        if rect.is_set and screen is not None and not screen.contains(rect):
            state = "OUTSIDE capture area"
        print(f"{path:34s} {rect.as_flow():34s} {state}")
    print("-" * 88)
    print(f"{len(all_rects) - len(unset)}/{len(all_rects)} regions set")
    for problem in rc.validate(config, screen):
        print(f"  ! {problem}")
    return 0


def cmd_set(args) -> int:
    rect = rc.Rect(args.x, args.y, args.w, args.h)
    if rect.w < 0 or rect.h < 0:
        raise SystemExit(f"negative size: {rect.as_flow()}")
    rc.set_rect(CONFIG, args.path, rect)
    print(f"{args.path} = {rect.as_flow()}")
    return 0


def cmd_nudge(args) -> int:
    config = rc.load(CONFIG)
    rect = rc.get_rect(config, args.path)
    if rect is None:
        raise SystemExit(f"no such region: {args.path}")
    moved = rc.Rect(
        rect.x + args.dx, rect.y + args.dy,
        rect.w + args.dw, rect.h + args.dh,
        rect.angle,
    )
    if moved.w < 0 or moved.h < 0:
        raise SystemExit(f"nudge would make the rect negative: {moved.as_flow()}")
    rc.set_rect(CONFIG, args.path, moved)
    print(f"{args.path}  {rect.as_flow()}  ->  {moved.as_flow()}")
    return 0


def cmd_clear(args) -> int:
    rc.set_rect(CONFIG, args.path, rc.ZERO)
    print(f"{args.path} = {rc.ZERO.as_flow()}  (unmarked)")
    return 0


def cmd_verify(args) -> int:
    config = rc.load(CONFIG)
    frame = _load_image(args.image)
    drawn = 0
    for path, rect in rc.rects(config).items():
        if not rect.is_set:
            continue
        if not (0 <= rect.x < frame.shape[1] and 0 <= rect.y < frame.shape[0]):
            continue
        cv2.rectangle(frame, (rect.x, rect.y), (rect.right - 1, rect.bottom - 1), GREEN, 2)
        _label(frame, f"{path} {rect.w}x{rect.h}", (rect.x, rect.y - 6), CYAN)
        drawn += 1
    out = args.out or args.image.with_suffix(".rois.png")
    cv2.imwrite(str(out), frame)
    print(f"{drawn} region(s) drawn -> {out}")
    missing = rc.unset_paths(config)
    if missing:
        print(f"{len(missing)} still unset:")
        for path in missing:
            print(f"  {path}")
    return 0


def cmd_reset(args) -> int:
    schema = REPO_ROOT / "config" / "regions.schema.yaml"
    if not schema.exists():
        raise SystemExit(f"schema template missing: {schema}")
    if CONFIG.exists() and not args.force:
        backup = CONFIG.with_suffix(f".yaml.bak-{datetime.now():%Y%m%d-%H%M%S}")
        shutil.copy2(CONFIG, backup)
        print(f"backed up existing config to {backup}")
    shutil.copy2(schema, CONFIG)
    print(f"wrote a fresh {CONFIG}")
    return 0


class Marker:
    """Drag boxes over a screenshot to produce rects, in the order requested."""

    def __init__(self, image: np.ndarray, targets: list[str], config: dict):
        self.image = image
        self.ih, self.iw = image.shape[:2]
        self.targets = targets
        self.existing = {p: r for p, r in rc.rects(config).items() if r.is_set}
        self.marked: list[rc.Rect] = []
        self.drag_start: tuple[int, int] | None = None
        self.drag_now: tuple[int, int] | None = None
        self.panning = False
        self.show_grid = True
        self.message = ""
        self.saved = False
        self.cancelled = False
        self.scale, self.pan_x, self.pan_y = 1.0, 0.0, 0.0

        sw, sh = _screen_size()
        self.cw = max(1100, min(2000, int(sw * 0.92)))
        self.ch = max(700, min(1300, int(sh * 0.92)))
        self.bar = 52
        self.ih_canvas = self.ch - self.bar
        self._fit()

    def _fit(self):
        self.scale = min(self.cw / self.iw, self.ih_canvas / self.ih) * 0.98
        self.pan_x = (self.cw - self.iw * self.scale) / 2
        self.pan_y = (self.ih_canvas - self.ih * self.scale) / 2

    def _to_full(self, dx, dy):
        return ((dx - self.pan_x) / self.scale, (dy - self.pan_y) / self.scale)

    def _to_disp(self, fx, fy):
        return (fx * self.scale + self.pan_x, fy * self.scale + self.pan_y)

    def _in_image(self, mx, my):
        return 0 <= mx < self.cw and 0 <= my < self.ih_canvas

    def _clamped(self, rect: rc.Rect) -> rc.Rect:
        x0, x1 = sorted((rect.x, rect.right))
        y0, y1 = sorted((rect.y, rect.bottom))
        x0 = max(0, min(x0, self.iw))
        x1 = max(0, min(x1, self.iw))
        y0 = max(0, min(y0, self.ih))
        y1 = max(0, min(y1, self.ih))
        return rc.Rect(int(x0), int(y0), max(1, int(x1 - x0)), max(1, int(y1 - y0)))

    def on_mouse(self, event, mx, my, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and self._in_image(mx, my):
            self.drag_start, self.drag_now = (mx, my), (mx, my)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drag_start is not None:
                self.drag_now = (mx, my)
            elif self.panning:
                self.pan_x += mx - self._last[0]
                self.pan_y += my - self._last[1]
                self._last = (mx, my)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            fx0, fy0 = self._to_full(*self.drag_start)
            fx1, fy1 = self._to_full(mx, my)
            self.drag_start = self.drag_now = None
            rect = self._clamped(rc.Rect(int(fx0), int(fy0), int(fx1 - fx0), int(fy1 - fy0)))
            if rect.w >= 2 and rect.h >= 2:
                self.marked.append(rect)
                remaining = len(self.targets) - len(self.marked)
                self.message = f"marked {self.targets[len(self.marked) - 1]}"
                if remaining == 0:
                    self.saved = True
            else:
                self.message = "box too small, ignored"
        elif event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN):
            self.panning, self._last = True, (mx, my)
        elif event in (cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP):
            self.panning = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            factor = 1.15 if flags > 0 else (1 / 1.15)
            fx, fy = self._to_full(mx, my)
            self.scale = max(0.05, min(40.0, self.scale * factor))
            self.pan_x = mx - fx * self.scale
            self.pan_y = my - fy * self.scale

    def _draw_grid(self, view):
        step = _grid_step(self.scale)
        x0, y0 = self._to_full(0, 0)
        x1, y1 = self._to_full(self.cw, self.ih_canvas)
        gx = int(x0 // step) * step
        gy = int(y0 // step) * step
        for fx in range(gx, int(x1) + step, step):
            dx = int(self._to_disp(fx, 0)[0])
            if 0 <= dx < self.cw:
                colour = (95, 95, 95) if fx % (step * 5) == 0 else (62, 62, 62)
                cv2.line(view, (dx, 0), (dx, self.ih_canvas), colour, 1)
        for fy in range(gy, int(y1) + step, step):
            dy = int(self._to_disp(0, fy)[1])
            if 0 <= dy < self.ih_canvas:
                colour = (95, 95, 95) if fy % (step * 5) == 0 else (62, 62, 62)
                cv2.line(view, (0, dy), (self.cw, dy), colour, 1)
        return step

    def render(self) -> np.ndarray:
        canvas = np.full((self.ch, self.cw, 3), BAR_BG, np.uint8)
        matrix = np.array(
            [[self.scale, 0.0, self.pan_x], [0.0, self.scale, self.pan_y]], np.float32
        )
        canvas[: self.ih_canvas] = cv2.warpAffine(
            self.image, matrix, (self.cw, self.ih_canvas),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(25, 22, 20),
        )
        view = canvas[: self.ih_canvas]

        step = self._draw_grid(view) if self.show_grid else 0

        for path, rect in self.existing.items():
            if path in self.targets:
                continue
            (x0, y0), (x1, y1) = self._to_disp(rect.x, rect.y), self._to_disp(rect.right, rect.bottom)
            cv2.rectangle(view, (int(x0), int(y0)), (int(x1), int(y1)), GREEN, 1)
            _label(view, path, (int(x0), int(y0) - 4), GREY, 0.38)

        for index, rect in enumerate(self.marked):
            if index >= len(self.targets):
                break
            (x0, y0), (x1, y1) = self._to_disp(rect.x, rect.y), self._to_disp(rect.right, rect.bottom)
            cv2.rectangle(view, (int(x0), int(y0)), (int(x1), int(y1)), CYAN, 2)
            _label(view, f"{self.targets[index]}  {rect.w}x{rect.h}", (int(x0), int(y0) - 5), CYAN, 0.45)

        if self.drag_start and self.drag_now:
            fx0, fy0 = self._to_full(*self.drag_start)
            fx1, fy1 = self._to_full(*self.drag_now)
            (dx0, dy0), (dx1, dy1) = self._to_disp(fx0, fy0), self._to_disp(fx1, fy1)
            cv2.rectangle(view, (int(dx0), int(dy0)), (int(dx1), int(dy1)), YELLOW, 1)
            _label(
                view,
                f"{int(fx1 - fx0)} x {int(fy1 - fy0)}  @ ({int(min(fx0, fx1))}, {int(min(fy0, fy1))})",
                (int(dx0), int(dy0) - 5), YELLOW, 0.45,
            )

        done = len(self.marked)
        total = len(self.targets)
        nxt = self.targets[done] if done < total else "(none left)"
        cv2.putText(canvas, f"  {done}/{total} marked", (10, 22), FONT, 0.55, YELLOW, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"next: {nxt}", (150, 22), FONT, 0.55, CYAN, 1, cv2.LINE_AA)
        info = f"image {self.iw}x{self.ih}   zoom {self.scale * 100:5.1f}%   grid {step}px"
        cv2.putText(canvas, info, (self.cw - cv2.getTextSize(info, FONT, 0.45, 1)[0][0] - 12, 22),
                    FONT, 0.45, GREY, 1, cv2.LINE_AA)
        help_text = "drag = mark   wheel = zoom   right-drag = pan   0 = fit   u = undo   s = save+quit   esc = cancel"
        cv2.putText(canvas, help_text, (10, 43), FONT, 0.44, (185, 180, 175), 1, cv2.LINE_AA)
        if self.message:
            cv2.putText(canvas, self.message[:110],
                        (max(10, self.cw - cv2.getTextSize(self.message, FONT, 0.44, 1)[0][0] - 12), 43),
                        FONT, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    def run(self, title: str) -> None:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, self.cw, self.ch)
        cv2.setMouseCallback(title, self.on_mouse)
        while True:
            cv2.imshow(title, self.render())
            key = cv2.waitKey(20)
            if key == -1:
                if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
                continue
            key &= 0xFF
            if key in (27, ord("q")):
                self.cancelled = True
                break
            if key in (13, 10, ord("s")):
                self.saved = True
                break
            if key in (ord("u"), 8) and self.marked:
                self.marked.pop()
                self.message = "undone"
            if key == ord("0"):
                self._fit()
            if key == ord("g"):
                self.show_grid = not self.show_grid
        cv2.destroyAllWindows()


def cmd_mark(args) -> int:
    config = rc.load(CONFIG)
    targets = _expand_targets(config, args.set)
    if not targets:
        raise SystemExit("--set needs at least one region path")
    for path in targets:
        rc.parse_path(path)

    print(f"marking {len(targets)} region(s) on {args.image}:")
    for index, path in enumerate(targets, 1):
        print(f"  {index}. {path}")
    print("\nOpen the Monopoly Poker table in the same state as the screenshot, then")
    print("drag a box over each item in the order above. Saves when the last one is")
    print("marked, or press 's'. Press 'esc' to discard.\n")

    frame = _load_image(Path(args.image))
    marker = Marker(frame, targets, config)
    marker.run("calibrate - mark regions")

    if marker.cancelled or not marker.marked:
        print("discarded, config unchanged")
        return 1
    for path, rect in zip(targets, marker.marked):
        rc.set_rect(CONFIG, path, rect)
        print(f"  {path:34s} = {rect.as_flow()}")
    remaining = rc.unset_paths(rc.load(CONFIG))
    print(f"\nsaved to {CONFIG}")
    if remaining:
        print(f"{len(remaining)} region(s) still unset: {', '.join(remaining)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calibrate.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=CONFIG, help="path to regions.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check dependencies and show monitor rectangles").set_defaults(fn=cmd_doctor)

    grab = sub.add_parser("grab", help="save a screenshot of the table to samples/")
    grab.add_argument("--name", help="filename stem (default: table-<timestamp>)")
    grab.add_argument("--monitor", type=int, help="mss monitor index (default: from config)")
    grab.set_defaults(fn=cmd_grab)

    sub.add_parser("list", help="show every region and whether it is set").set_defaults(fn=cmd_list)

    marker = sub.add_parser("mark", help="interactively mark regions on a screenshot")
    marker.add_argument("--image", required=True, type=Path, help="screenshot to mark on")
    marker.add_argument("--set", required=True, nargs="+", metavar="PATH",
                        help="region paths, in the order you will drag them "
                             "(a list path such as board.slots expands to all its items)")
    marker.set_defaults(fn=cmd_mark)

    setter = sub.add_parser("set", help="set one region to exact values")
    setter.add_argument("path", help="region path, e.g. texts.pot")
    for name in ("x", "y", "w", "h"):
        setter.add_argument(name, type=int)
    setter.set_defaults(fn=cmd_set)

    nudge = sub.add_parser("nudge", help="shift or resize one region by a delta")
    nudge.add_argument("path")
    nudge.add_argument("--dx", type=int, default=0)
    nudge.add_argument("--dy", type=int, default=0)
    nudge.add_argument("--dw", type=int, default=0)
    nudge.add_argument("--dh", type=int, default=0)
    nudge.set_defaults(fn=cmd_nudge)

    clear = sub.add_parser("clear", help="reset one region to unmarked")
    clear.add_argument("path")
    clear.set_defaults(fn=cmd_clear)

    verify = sub.add_parser("verify", help="draw every set region on a screenshot")
    verify.add_argument("--image", required=True, type=Path)
    verify.add_argument("--out", type=Path, help="defaults to <image>.rois.png")
    verify.set_defaults(fn=cmd_verify)

    reset = sub.add_parser("reset", help="regenerate regions.yaml from the schema")
    reset.add_argument("--force", action="store_true", help="overwrite without backing up")
    reset.set_defaults(fn=cmd_reset)
    return parser


def main(argv=None) -> int:
    global CONFIG
    args = build_parser().parse_args(argv)
    CONFIG = args.config.resolve()
    try:
        return args.fn(args)
    except rc.ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"bad region path: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
