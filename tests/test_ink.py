"""Pin `ink_map` to `_ink_binary`.

`find` thresholds the frame ink with `_ink_binary`, which is on the hot path
and so no longer calls `ink_map`. The two agree because `_ink_binary` reads its
cut-offs back out of the same lookup tables `ink_map` uses - but that only
covers the *constants*. A change to the ink formula itself, a third kind of
ink say, would land in `ink_map` alone and the search would quietly stop
agreeing with it, with nothing failing.

So assert the equivalence directly, on every screenshot and across the
degenerate inputs that make the two forms take different branches: an empty
paper mask, a mask covering nothing bright, an all-dark frame, and the paper
brightness extremes.
"""

import _boot  # noqa: F401,E402  - repo root on sys.path, cwd at root

import glob
import sys

import cv2
import numpy as np

import pokervision as pv
from pokervision.detect import _ink_binary, ink_map, paper_mask
from pokervision.perception import INK_LEVEL

FAILURES: list[str] = []


def check(name: str, got, want) -> None:
    if got == want:
        print(f"  ok    {name}: got {got}")
    else:
        print(f"  FAIL  {name}: got {got}, want {want}")
        FAILURES.append(name)


def equivalent(frame: np.ndarray) -> tuple[bool, int]:
    """Do both forms mark exactly the same pixels?"""
    paper = paper_mask(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    soft = ink_map(frame, paper, hsv)
    hard = _ink_binary(hsv, paper)
    got = hard > 0
    want = soft >= INK_LEVEL
    return bool(np.array_equal(got, want)), int(np.count_nonzero(got != want))


def main() -> int:
    shots = sorted(glob.glob("unmarked-images/*.png"))
    if not shots:
        print("no screenshots in unmarked-images/ - nothing to compare")
        return 1

    print("=== every screenshot ===")
    total_bad = 0
    for path in shots:
        frame = cv2.imread(path)
        same, bad = equivalent(frame)
        total_bad += bad
        if not same:
            FAILURES.append(path)
            print(f"  FAIL  {path}: {bad} pixels disagree")
    check("all screenshots agree with ink_map", total_bad, 0)
    print(f"  info  {len(shots)} screenshots compared, "
          f"{total_bad} differing pixels in total")

    print("\n=== degenerate inputs, where the two forms take different branches ===")
    h, w = 120, 160
    rng = np.random.default_rng(0)
    cases = {
        "an all-white frame": np.full((h, w, 3), 245, np.uint8),
        "an all-black frame": np.zeros((h, w, 3), np.uint8),
        "random noise": rng.integers(0, 256, (h, w, 3), dtype=np.uint8),
        "saturated red": np.zeros((h, w, 3), np.uint8),
    }
    cases["saturated red"][:, :, 2] = 230
    cases["saturated red"][:, :, 1] = 40
    for name, frame in cases.items():
        same, bad = equivalent(frame)
        check(f"{name} agrees", same, True)

    # a frame with no paper at all: paper.size == 0 must return zeros on both
    empty = np.full((h, w, 3), 240, np.uint8)
    hsv = cv2.cvtColor(empty, cv2.COLOR_BGR2HSV)
    none_paper = np.zeros((h, w), np.uint8)
    check("no paper gives no ink from ink_map",
          int(ink_map(empty, none_paper, hsv).max()), 0)
    check("no paper gives no ink from _ink_binary",
          int(_ink_binary(hsv, none_paper).max()), 0)

    # paper brightness swept across the range, including values the
    # global median lands on at the extremes
    print("\n=== paper brightness swept ===")
    for level in (20, 60, 120, 180, 230, 252):
        frame = np.full((h, w, 3), level, np.uint8)
        frame[40:80, 40:80] = 0            # dark ink
        frame[90:120, 40:80] = (30, 30, 210)  # red ink
        same, bad = equivalent(frame)
        check(f"paper {level} agrees", same, True)

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURES: {FAILURES}")
        return 1
    print("\nink_map and _ink_binary agree everywhere")
    return 0


if __name__ == "__main__":
    sys.exit(main())
