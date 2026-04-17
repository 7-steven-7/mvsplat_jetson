from __future__ import annotations

import sys
from contextlib import nullcontext

from jaxtyping import install_import_hook


def install_runtime_import_hook(packages=("src",)):
    # Python < 3.10 cannot evaluate this project's modern type hints under beartype.
    if sys.version_info < (3, 10):
        return nullcontext()
    return install_import_hook(packages, ("beartype", "beartype"))
