"""Give this package access to the ``lems`` reference clone without copying it.

``lems/`` (github.com/lems-svd/lems, the official code for Thoma et al., ICML
2026) carries no LICENSE file. We depend on it the way any research codebase
without a license is safe to depend on -- by running it locally, unmodified,
never by copying its source into this repo. Concretely: it sits as a sibling
directory to this package, on `sys.path`, and we `import` its modules like any
other library. Nothing in ``lems/`` is ever edited or vendored.

Every entry point that touches ``compression.*`` from ``lems/`` must call
``add_lems_to_syspath()`` first.
"""

from __future__ import annotations

import sys
from pathlib import Path

_LEMS_DIR = Path(__file__).resolve().parent.parent / "lems"

_added = False


def add_lems_to_syspath() -> Path:
    """Insert the ``lems/`` reference clone onto ``sys.path``. Idempotent."""
    global _added
    if not _LEMS_DIR.is_dir():
        raise FileNotFoundError(
            f"expected the lems-svd/lems reference clone at {_LEMS_DIR}, "
            "but it doesn't exist. Clone it there first:\n"
            f"  git clone https://github.com/lems-svd/lems.git {_LEMS_DIR}"
        )
    if not (_LEMS_DIR / "compression").is_dir():
        raise FileNotFoundError(
            f"{_LEMS_DIR} exists but has no compression/ subpackage -- "
            "is this actually the lems-svd/lems clone?"
        )
    if not _added:
        sys.path.insert(0, str(_LEMS_DIR))
        _added = True
    return _LEMS_DIR
