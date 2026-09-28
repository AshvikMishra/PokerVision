"""config/regions.yaml: schema, loader, and a comment-preserving editor.

The write path is hand-rolled because the yaml is hand-tuned and carries
comments: `yaml.safe_dump` throws those and the key order away, so
`set_rect` locates the single line owning a config path and rewrites just
that one, leaving the rest of the file byte-for-byte alone.

Coordinate space: absolute virtual-desktop pixels, top-left origin.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, NamedTuple

import yaml

# this module lives in the package, so the repository
# root is one level up - not the package directory
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "regions.yaml"

RECT_KEYS = frozenset({"x", "y", "w", "h"})


class Rect(NamedTuple):
    x: int
    y: int
    w: int
    h: int
    angle: float = 0.0

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    @property
    def is_set(self) -> bool:
        return self.w > 0 and self.h > 0

    def contains(self, other: "Rect") -> bool:
        return (
            other.x >= self.x
            and other.y >= self.y
            and other.right <= self.right
            and other.bottom <= self.bottom
        )

    def as_flow(self) -> str:
        suffix = f", angle: {self.angle:g}" if self.angle else ""
        return f"{{x: {self.x}, y: {self.y}, w: {self.w}, h: {self.h}{suffix}}}"


ZERO = Rect(0, 0, 0, 0)


def as_rect(value: Any) -> Rect | None:
    if not isinstance(value, dict) or RECT_KEYS - set(value):
        return None
    try:
        return Rect(
            *(int(value[k]) for k in ("x", "y", "w", "h")),
            float(value.get("angle", 0.0)),
        )
    except (TypeError, ValueError):
        return None


_PATH_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)((?:\[\d+\]|\.[A-Za-z_][A-Za-z0-9_]*)*)$")
_SEG_RE = re.compile(r"\[(\d+)\]|([A-Za-z_][A-Za-z0-9_]*)")


def parse_path(path: str) -> list[str | int]:
    match = _PATH_RE.match(path.strip())
    if not match:
        raise ValueError(
            f"bad region path {path!r}: expected something like "
            "'texts.pot', 'hero.hole_cards[0]' or 'board.slots[2]'"
        )
    head, tail = match.group(1), match.group(2)
    segments: list[str | int] = [head]
    for index, name in _SEG_RE.findall(tail):
        segments.append(int(index) if index else name)
    return segments


def format_path(segments: list[str | int]) -> str:
    out = ""
    for seg in segments:
        out += f"[{seg}]" if isinstance(seg, int) else (f".{seg}" if out else seg)
    return out


def resolve(config: dict, path: str) -> Any:
    node: Any = config
    for seg in parse_path(path):
        if isinstance(seg, int):
            if not isinstance(node, list) or seg >= len(node):
                return None
            node = node[seg]
        else:
            if not isinstance(node, dict) or seg not in node:
                return None
            node = node[seg]
    return node


def get_rect(config: dict, path: str) -> Rect | None:
    return as_rect(resolve(config, path))


def iter_rect_paths(node: Any, prefix: str = "") -> Iterator[str]:
    if isinstance(node, dict):
        if as_rect(node) is not None:
            yield prefix
            return
        for key, value in node.items():
            yield from iter_rect_paths(value, format_path([key]) if not prefix else f"{prefix}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from iter_rect_paths(value, f"{prefix}[{index}]")


@dataclass
class _Entry:
    lineno: int
    path: str
    indent: int
    inline: str | None
    comment: str
    is_item: bool
    is_leaf: bool
    is_container: bool

    @property
    def key(self) -> str:
        return self.path.rsplit(".", 1)[-1].split("[", 1)[0]


_KEY_LINE = re.compile(r"^(?P<indent>[ ]*)(?P<dash>- )?(?P<key>[A-Za-z_][A-Za-z0-9_]*):(?:[ ]*(?P<rest>.*))?$")


def _split_comment(text: str) -> tuple[str, str]:
    idx = 0
    while True:
        idx = text.find("#", idx)
        if idx == -1:
            return text.rstrip(), ""
        if idx == 0 or text[idx - 1].isspace():
            start = idx
            while start > 0 and text[start - 1].isspace():
                start -= 1
            return text[:idx].rstrip(), text[start:]
        idx += 1


def _scan(lines: list[str]) -> list[_Entry]:
    entries: list[_Entry] = []
    stack: list[tuple[int, str, int]] = []

    for lineno, raw in enumerate(lines):
        line = raw.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue

        match = _KEY_LINE.match(line)
        is_item = bool(match and match.group("dash"))
        if not match:
            # a bare "- {..}" flow item carries no key of its own
            stripped = line.strip()
            if stripped.startswith("- ") and stack:
                indent = len(line) - len(line.lstrip())
                while stack and stack[-1][0] >= indent:
                    stack.pop()
                if not stack:
                    continue
                _, parent, index = stack[-1]
                path = f"{parent}[{index}]"
                value, comment = _split_comment(stripped[2:])
                entries.append(_Entry(lineno, path, indent, value, comment, True, True, True))
                stack[-1] = (stack[-1][0], parent, index + 1)
                stack.append((indent, path, 0))
            continue

        indent = len(match.group("indent"))
        key = match.group("key")
        rest = (match.group("rest") or "").strip()
        dash = is_item

        while stack and stack[-1][0] >= indent:
            stack.pop()

        if dash:
            if not stack:
                continue
            _, parent, index = stack[-1]
            path = f"{parent}[{index}]"
            stack[-1] = (stack[-1][0], parent, index + 1)
        else:
            path = key if not stack else f"{stack[-1][1]}.{key}"

        value, comment = _split_comment(rest)
        if dash:
            entries.append(_Entry(lineno, path, indent, value or None, comment, True, True, True))
            stack.append((indent, path, 0))
        else:
            entries.append(_Entry(lineno, path, indent, value or None, comment, False, True, not value))
            if not value:
                stack.append((indent, path, 0))
    return entries


class ConfigError(RuntimeError):
    pass


def _render_key_line(entry: _Entry, rect: Rect) -> str:
    if entry.is_item:
        body = "- " + rect.as_flow() + entry.comment
    else:
        body = entry.key + ": " + rect.as_flow() + entry.comment
    return " " * entry.indent + body


def _descendants(entries: list[_Entry], path: str) -> list[int]:
    return [
        e.lineno
        for e in entries
        if e.path.startswith(path + ".") or e.path.startswith(path + "[")
    ]


def _block_end(entries: list[_Entry], path: str, fallback: int) -> int:
    return max(_descendants(entries, path), default=fallback)


def _children(entries: list[_Entry], path: str) -> list[_Entry]:
    return sorted(
        (e for e in entries if _parent_of(e.path) == path),
        key=lambda e: e.lineno,
    )


def _set_rect_lines(lines: list[str], path: str, rect: Rect) -> list[str]:
    entries = _scan(lines)
    entry = next((e for e in entries if e.path == path), None)

    # 1. a container we now know is a rect: collapse the block to one line
    if entry is not None and entry.is_container and not entry.is_item:
        end = max(_descendants(entries, path), default=entry.lineno)
        lines[entry.lineno:end + 1] = [
            " " * entry.indent + entry.key + ": " + rect.as_flow() + entry.comment
        ]
        return lines

    # 2. exact hit: rewrite that one line
    if entry is not None and entry.is_leaf:
        lines[entry.lineno] = _render_key_line(entry, rect)
        return lines

    # 3. new value: make sure the parent chain exists, then splice in a line
    segments = parse_path(path)
    if len(segments) < 2:
        raise ConfigError(f"cannot create top-level region {path!r}")
    parent_path = format_path(segments[:-1])
    leaf = segments[-1]

    _ensure_container(lines, parent_path)
    entries = _scan(lines)
    parent = next(e for e in entries if e.path == parent_path)
    pad = " " * (parent.indent + 2)

    if isinstance(leaf, int):
        siblings = [e for e in _children(entries, parent_path) if e.is_item]
        if leaf < len(siblings):
            block = [pad + "- " + rect.as_flow()]
            at = siblings[leaf].lineno
        else:
            last = siblings[-1] if siblings else None
            at = _block_end(entries, last.path, last.lineno) + 1 if last else parent.lineno + 1
            block = [pad + "- " + ZERO.as_flow()] * (leaf - len(siblings))
            block.append(pad + "- " + rect.as_flow())

    elif parent.is_item and (parent.inline or "").strip() == "{}":
        lines[parent.lineno] = " " * parent.indent + "- " + leaf + ": " + rect.as_flow()
        return lines

    else:
        at = _block_end(entries, parent_path, parent.lineno) + 1
        block = [pad + leaf + ": " + rect.as_flow()]

    for offset, text in enumerate(block):
        lines.insert(at + offset, text)
    return lines


def _ensure_container(lines: list[str], path: str) -> None:
    """Create any missing containers along `path` so a child can be written."""
    if not path:
        return
    entry = next((e for e in _scan(lines) if e.path == path), None)
    if entry is not None and entry.is_container:
        return
    if entry is not None and (entry.inline or "").strip() in ("[]", "{}"):
        lines[entry.lineno] = " " * entry.indent + entry.key + ":" + entry.comment
        return

    parent_path = _parent_of(path)
    segment = parse_path(path)[-1]
    if not parent_path:
        raise ConfigError(
            f"no top-level config key {path!r} - add it to {DEFAULT_CONFIG.name} first"
        )
    _ensure_container(lines, parent_path)

    entries = _scan(lines)
    parent = next(e for e in entries if e.path == parent_path)
    if not parent.is_container:
        raise ConfigError(
            f"{parent_path!r} already holds a value, so {path!r} cannot live "
            f"inside it - check the region path"
        )

    if isinstance(segment, int):
        if parent.is_item:
            raise ConfigError(f"{parent_path!r} is not a list, cannot add item {segment} to it")
        siblings = [e for e in _children(entries, parent_path) if e.is_item]
        if segment < len(siblings):
            return
        last = siblings[-1] if siblings else None
        at = _block_end(entries, last.path, last.lineno) + 1 if last else parent.lineno + 1
        for _ in range(segment - len(siblings) + 1):
            lines.insert(at, " " * (parent.indent + 2) + "- {}")
            at += 1
    else:
        at = _block_end(entries, parent_path, parent.lineno) + 1
        lines.insert(at, " " * (parent.indent + 2) + segment + ":")


def _parent_of(path: str) -> str:
    segments = parse_path(path)
    return format_path(segments[:-1])


def load(path: Path | str = DEFAULT_CONFIG) -> dict:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config not found: {path} (run `python calibrate.py reset`)")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"{path} did not parse to a mapping")
    return data


def set_rect(path: Path | str, dotted: str, rect: Rect) -> None:
    path = Path(path)
    lines = path.read_text(encoding="utf-8").split("\n")
    _set_rect_lines(lines, dotted, rect)
    path.write_text("\n".join(lines), encoding="utf-8")


def rects(config: dict) -> dict[str, Rect]:
    return {p: r for p in iter_rect_paths(config) if (r := get_rect(config, p)) is not None}


def unset_paths(config: dict) -> list[str]:
    return [p for p, r in rects(config).items() if not r.is_set]


def validate(config: dict, screen: Rect | None = None) -> list[str]:
    problems: list[str] = []
    for dotted, rect in rects(config).items():
        if rect.x < 0 or rect.y < 0:
            problems.append(f"{dotted}: negative origin ({rect.x}, {rect.y})")
        if rect.w < 0 or rect.h < 0:
            problems.append(f"{dotted}: negative size ({rect.w}x{rect.h})")
        if rect.is_set and screen is not None and not screen.contains(rect):
            problems.append(
                f"{dotted}: {rect.as_flow()} falls outside the capture area "
                f"{screen.as_flow()}"
            )
    return problems


def mon_rects() -> list[tuple[int, Rect]]:
    import mss

    with mss.mss() as sct:
        return [(i, Rect(m["left"], m["top"], m["width"], m["height"])) for i, m in enumerate(sct.monitors)]


def capture_rect(config: dict) -> Rect | None:
    index = int(config.get("capture", {}).get("monitor", 0) or 0)
    for i, rect in mon_rects():
        if i == index:
            return rect
    return None
