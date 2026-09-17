"""clahe: CLAHE contrast enhancement.

DEFAULTS match the lp_clahe5 eye-model training recipe exactly
(lightningPose-eye-tracking capsule): clipLimit=5.0, tileGridSize=8x8,
grayscale replicated to 3 identical channels.

The parameters are exposed so other teams/models can use their own
settings -- but a model must be inferred with the SAME preprocessing it
was trained with. For the lp_clahe5 eye models, keep the defaults.
Whatever values are used are recorded in results/preprocessing.json.
"""
from __future__ import annotations

import cv2
import numpy as np

# lp_clahe5 training recipe -- the defaults.
DEFAULT_CLIP_LIMIT = 5.0
DEFAULT_TILE_GRID = 8


def make_clahe(clip_limit: float = DEFAULT_CLIP_LIMIT,
               tile_grid: int = DEFAULT_TILE_GRID):
    """Build a frame transform applying CLAHE with the given parameters.

    Returns a function (H, W, 3) BGR uint8 -> (H, W, 3) BGR uint8:
    grayscale -> CLAHE -> replicate to 3 identical channels.
    """
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit),
                            tileGridSize=(int(tile_grid), int(tile_grid)))

    def apply(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eq = clahe.apply(gray)
        return cv2.merge([eq, eq, eq])

    return apply
