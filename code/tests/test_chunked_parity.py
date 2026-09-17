"""Chunked preprocessing must be pixel-identical to sequential.

Builds a small synthetic video, runs the capsule's processing path with
--workers 1 (sequential) and --workers 3 (chunked+concat), and asserts:
  - identical frame counts
  - every decoded frame identical (max abs diff == 0)
Also checks the seek guard fires on obviously-bad ranges.

Run from code/:  python -m pytest ../tests/test_chunked_parity.py -q
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import run_capsule  # noqa: E402


N_FRAMES, W, H, FPS = 137, 96, 64, 30.0   # deliberately not divisible by 3


@pytest.fixture(scope="module")
def source_video(tmp_path_factory) -> Path:
    """Synthesize a small lossless test video with per-frame structure."""
    d = tmp_path_factory.mktemp("src")
    path = d / "toy.mp4"
    proc = run_capsule._ffmpeg_writer(path, W, H, FPS)
    rng = np.random.default_rng(0)
    for i in range(N_FRAMES):
        frame = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
        frame[:, : (i % W)] = i % 256          # frame-index watermark
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    assert proc.wait() == 0
    return path


def _run(video: Path, out: Path, workers: int) -> dict:
    args = run_capsule.parse_args([
        "--method", "clahe", "--workers", str(workers)])
    return run_capsule.process_video(video, out, "clahe", args, workers, None)


def _decode_all(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return np.stack(frames)


def test_chunked_equals_sequential(source_video, tmp_path, monkeypatch):
    monkeypatch.setattr(run_capsule, "SCRATCH_DIR", tmp_path / "scratch")
    rec_seq = _run(source_video, tmp_path / "seq.mp4", workers=1)
    rec_chk = _run(source_video, tmp_path / "chk.mp4", workers=3)

    assert rec_seq["frames_processed"] == N_FRAMES
    assert rec_chk["frames_processed"] == N_FRAMES
    assert sum(rec_chk["chunk_frames"]) == N_FRAMES
    assert len(rec_chk["chunk_frames"]) == 3

    a = _decode_all(tmp_path / "seq.mp4")
    b = _decode_all(tmp_path / "chk.mp4")
    assert a.shape == b.shape == (N_FRAMES, H, W, 3)
    assert int(np.abs(a.astype(int) - b.astype(int)).max()) == 0, \
        "chunked output differs from sequential"


def test_worker_count_capped_by_frames(source_video, tmp_path, monkeypatch):
    monkeypatch.setattr(run_capsule, "SCRATCH_DIR", tmp_path / "scratch")
    rec = _run(source_video, tmp_path / "many.mp4", workers=1)  # baseline
    a = _decode_all(tmp_path / "many.mp4")
    rec2 = _run(source_video, tmp_path / "cap.mp4",
                workers=N_FRAMES + 50)          # absurd worker count
    assert rec2["workers"] <= N_FRAMES
    b = _decode_all(tmp_path / "cap.mp4")
    assert int(np.abs(a.astype(int) - b.astype(int)).max()) == 0


def test_short_read_is_loud(source_video, tmp_path):
    args = run_capsule.parse_args(["--method", "clahe"])
    with pytest.raises(RuntimeError, match="expected"):
        run_capsule._transform_range(
            str(source_video), 0, N_FRAMES + 10,   # ask for more than exists
            str(tmp_path / "short.mp4"), "clahe", args, W, H, FPS, "t")
