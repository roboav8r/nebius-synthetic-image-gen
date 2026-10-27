"""Where Isaac Sim lives in the container, and how its caches are kept warm."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Isaac's on-disk caches (shaders, assets, compiled kernels). A cold start
# takes about 6 min; with these on the shared filesystem it takes about 80 s.
# `kit/cache` is fixed by the image layout; the rest follow HOME, which is
# /root because the job image runs as root.
_HOME = Path.home()
CACHE_DIRS = {
    "kit": Path("/isaac-sim/kit/cache"),
    "ov": _HOME / ".cache/ov",
    "pip": _HOME / ".cache/pip",
    "glcache": _HOME / ".cache/nvidia/GLCache",
    "computecache": _HOME / ".nv/ComputeCache",
    "logs": _HOME / ".nvidia-omniverse/logs",
    "data": _HOME / ".local/share/ov/data",
    "documents": _HOME / "Documents",
}


def python() -> list[str]:
    """The interpreter that can import `isaacsim`: $SDG_ISAAC_PYTHON, else the
    container's python.sh, else this one (a pip install of Isaac Sim)."""
    if os.environ.get("SDG_ISAAC_PYTHON"):
        return [os.environ["SDG_ISAAC_PYTHON"]]
    if Path("/isaac-sim/python.sh").exists():
        return ["/isaac-sim/python.sh"]
    return [sys.executable]


def link_caches(root: Path) -> None:
    """Replace each cache directory with a symlink to `root/<name>`."""
    for name, target in CACHE_DIRS.items():
        src = root / name
        src.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            target.unlink()
        elif target.exists():
            target.rename(target.with_name(target.name + ".orig"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(src, target_is_directory=True)
    print(f"==> caches linked under {root}")
