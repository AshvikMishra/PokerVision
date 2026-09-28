"""Shared bootstrap for the test scripts.

Two things every test needs and neither should repeat:

  - the repository root on `sys.path`, so `import pokervision` resolves.
    Running `python tests/test_x.py` puts `tests/` on the path, not the root,
    so without this the package import fails outright.
  - the working directory at the repository root, so the relative
    `unmarked-images/*.png` paths that two of the tests glob resolve the same
    way no matter where the script was invoked from.
"""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.chdir(REPO)
