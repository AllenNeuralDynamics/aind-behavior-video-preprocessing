"""Method registry: the preprocessing "menu".

A method entry is a FACTORY: given the parsed CLI args, it returns a
frame transform (BGR uint8 frame -> BGR uint8 frame) plus a dict of the
parameter values actually used (recorded in preprocessing.json).

Adding a method = one module with a factory + one entry here. Its
sub-parameters get their own CLI flags in run_capsule.py.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

from .clahe import make_clahe

FrameFn = Callable[[np.ndarray], np.ndarray]


def _make_none(args) -> Tuple[FrameFn, dict]:
    return (lambda frame: frame), {}


def _make_clahe(args) -> Tuple[FrameFn, dict]:
    fn = make_clahe(clip_limit=args.clahe_clip_limit,
                    tile_grid=args.clahe_tile_grid)
    params = {"clip_limit": float(args.clahe_clip_limit),
              "tile_grid_size": [int(args.clahe_tile_grid)] * 2,
              "channels": "gray->3ch"}
    return fn, params


#: method name -> factory(args) -> (frame_fn, params_used)
METHODS: Dict[str, Callable] = {
    "none": _make_none,
    "clahe": _make_clahe,
}
