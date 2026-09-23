"""Method registry: the preprocessing "menu".

A method entry is a FACTORY: given a dict of parameters (possibly empty
or None), it returns a frame transform (BGR uint8 frame -> BGR uint8
frame) plus a dict of the parameter values actually used (recorded in
preprocessing.json). Factories VALIDATE their parameters and raise on
unknown keys -- config typos must fail loudly, not silently fall back
to defaults.

Adding a method = one module with a factory + one entry here. Methods
are per-frame transforms; an ordered list of them (config "steps")
composes into a single per-frame pipeline.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

from .clahe import DEFAULT_CLIP_LIMIT, DEFAULT_TILE_GRID, make_clahe

FrameFn = Callable[[np.ndarray], np.ndarray]


def _check_params(method: str, params: dict, allowed: set) -> None:
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(
            f"method '{method}': unknown parameter(s) {sorted(unknown)}; "
            f"allowed: {sorted(allowed)}")


def _make_none(params: Optional[dict] = None) -> Tuple[FrameFn, dict]:
    _check_params("none", dict(params or {}), set())
    return (lambda frame: frame), {}


def _make_clahe(params: Optional[dict] = None) -> Tuple[FrameFn, dict]:
    p = dict(params or {})
    _check_params("clahe", p, {"clip_limit", "tile_grid"})
    clip = float(p.get("clip_limit", DEFAULT_CLIP_LIMIT))
    tile = int(p.get("tile_grid", DEFAULT_TILE_GRID))
    fn = make_clahe(clip_limit=clip, tile_grid=tile)
    used = {"clip_limit": clip,
            "tile_grid_size": [tile, tile],
            "channels": "gray->3ch"}
    return fn, used


#: method name -> factory(params_dict) -> (frame_fn, params_used)
METHODS: Dict[str, Callable] = {
    "none": _make_none,
    "clahe": _make_clahe,
}
