"""Run configs: one JSON file per experiment.

A config may name a parent with `"extends": "common.json"` (relative to the
file); the two are deep-merged, the child winning. Strings may reference the
environment as `${VAR}` or `${VAR:-default}`, so account details stay in
`.env` and out of git.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def load_config(path: str | Path) -> dict:
    """Read a config, resolving `extends` and then `${VAR}` references.

    Raises KeyError if a referenced variable is unset and has no default.
    """
    return _expand(_load(Path(path).resolve()))


def _load(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    parent = cfg.pop("extends", None)
    return _merge(_load(path.parent / parent), cfg) if parent else cfg


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _expand(value):
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, str):
        return _VAR.sub(_lookup, value)
    return value


def _lookup(match: re.Match) -> str:
    name, default = match[1], match[2]
    value = os.environ.get(name)
    if value:
        return value
    if default is not None:
        return default
    raise KeyError(f"{name} is not set; see .env.example")
