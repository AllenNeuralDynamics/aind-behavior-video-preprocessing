"""Config-file interface (issue #4): the config file drives the capsule.

Covers: missing config file is a loud, example-bearing error; defaults
for omitted glob/workers; loud validation of malformed config files;
empty steps = bypass spec; multi-step composition equal to manual
nesting.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import run_capsule  # noqa: E402
from preprocess.registry import METHODS  # noqa: E402


def _frames(n=3, h=64, w=96, seed=1):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
            for _ in range(n)]


def _write(tmp_path, text):
    p = tmp_path / "preprocessing.yaml"
    p.write_text(text)
    return p


def test_missing_config_is_loud(tmp_path):
    with pytest.raises(FileNotFoundError, match="config file not found"):
        run_capsule.load_spec(tmp_path / "absent.yaml")


def test_minimal_config_gets_defaults(tmp_path):
    spec = run_capsule.load_spec(_write(tmp_path, "steps: []\n"))
    assert spec["steps"] == []
    assert spec["video_glob"] == "**/*.mp4"
    assert 1 <= spec["workers"] <= 8
    assert spec["source"].endswith("preprocessing.yaml")


def test_full_config(tmp_path):
    spec = run_capsule.load_spec(_write(tmp_path, """
video_glob: "**/*special*.mp4"
workers: 3
steps:
  - method: clahe
    clip_limit: 2.5
"""))
    assert spec["video_glob"] == "**/*special*.mp4"
    assert spec["workers"] == 3
    assert spec["steps"] == [{"method": "clahe",
                              "params": {"clip_limit": 2.5}}]


@pytest.mark.parametrize("text,exc,match", [
    ("video_glob: x\n", ValueError, "'steps' is required"),
    ("steps:\n  - method: sharpen\n", ValueError, "unknown method"),
    ("steps:\n  - method: clahe\n    cliplimit: 3\n", ValueError,
     "unknown parameter"),
    ("steps:\n  - method: none\n", ValueError, "not a step"),
    ("stepz: []\n", ValueError, "unknown key"),
    ("steps: {}\n", ValueError, "must be a list"),
    ("- a\n- b\n", ValueError, "must be a mapping"),
])
def test_config_validation_is_loud(tmp_path, text, exc, match):
    with pytest.raises(exc, match=match):
        run_capsule.load_spec(_write(tmp_path, text))


def test_two_step_composition_equals_manual_nesting():
    steps = [{"method": "clahe", "params": {"clip_limit": 2.0}},
             {"method": "clahe", "params": {"clip_limit": 5.0}}]
    pipeline, used = run_capsule.build_pipeline(steps)
    f1, _ = METHODS["clahe"]({"clip_limit": 2.0})
    f2, _ = METHODS["clahe"]({"clip_limit": 5.0})
    for frame in _frames():
        assert np.array_equal(pipeline(frame), f2(f1(frame)))
    assert [u["method"] for u in used] == ["clahe", "clahe"]


def test_defaults_recorded_when_params_omitted():
    _, used = run_capsule.build_pipeline([{"method": "clahe", "params": {}}])
    assert used[0]["params"]["clip_limit"] == 5.0
    assert used[0]["params"]["tile_grid_size"] == [8, 8]
