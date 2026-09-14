"""Parity test: this capsule's clahe5 must be pixel-identical to the
lightningPose-eye-tracking capsule's preprocessing (which itself matches
the lp_clahe5 training recipe).

Reference implementation copied verbatim from the eye capsule so any
divergence in ours is caught here, without needing that capsule installed.
"""
import cv2
import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocess.clahe import make_clahe  # noqa: E402
from preprocess.registry import METHODS  # noqa: E402


def reference_clahe5(frame_bgr: np.ndarray) -> np.ndarray:
    """Verbatim reference: eye capsule / training recipe."""
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    eq = clahe.apply(gray)
    return cv2.merge([eq, eq, eq])


def _random_frames(n=5, h=492, w=658, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        # synthetic eye-like frame: dark disc on bright textured background
        img = rng.integers(120, 200, size=(h, w, 3), dtype=np.uint8)
        cv2.circle(img, (w // 2, h // 2), 90, (25, 25, 25), -1)
        cv2.circle(img, (w // 2 + 10, h // 2 - 5), 12, (230, 230, 230), -1)
        yield img


def test_clahe5_matches_reference():
    fn = make_clahe()  # DEFAULTS must equal the training recipe
    for frame in _random_frames():
        ours = fn(frame)
        ref = reference_clahe5(frame)
        assert ours.shape == ref.shape == frame.shape
        assert np.array_equal(ours, ref), "clahe5 diverges from training recipe"


def test_none_is_identity():
    fn, _ = METHODS["none"](None)
    for frame in _random_frames(n=2):
        assert np.array_equal(fn(frame), frame)


def test_clahe5_channels_identical():
    fn = make_clahe()
    frame = next(_random_frames(n=1))
    out = fn(frame)
    assert np.array_equal(out[..., 0], out[..., 1])
    assert np.array_equal(out[..., 1], out[..., 2])
