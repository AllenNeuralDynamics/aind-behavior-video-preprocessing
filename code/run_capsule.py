"""Behavior Video Preprocessing capsule.

Finds behavior video(s) under /data, applies the selected preprocessing
method to every frame, and writes a LOSSLESS re-encoded video to /results,
mirroring the input's relative path so downstream capsules see the same
layout (e.g. behavior-videos/Eye/video.mp4 stays at that relative path).

Also writes results/preprocessing.json recording the method, its
parameters, and per-video facts, for provenance.

Lossless choice: libx264 with -qp 0 (mathematically lossless), yuv444p or
rgb as appropriate, keeping the .mp4 extension so downstream path
expectations are unchanged. Frames are piped raw to ffmpeg.

Usage:
    python code/run_capsule.py --method clahe5 --video-glob "**/*[eE]ye*.mp4"
    python code/run_capsule.py --method none                 # passthrough copy
Env:
    PREPROC_MAX_FRAMES  optional cap for smoke tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import cv2

from preprocess.clahe import DEFAULT_CLIP_LIMIT, DEFAULT_TILE_GRID
from preprocess.registry import METHODS

DATA_DIR = Path("/data")
RESULTS_DIR = Path("/results")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--method",
        default="none",
        choices=sorted(METHODS),
        help="preprocessing method to apply (the menu)",
    )
    p.add_argument(
        "--clahe-clip-limit",
        type=float,
        default=DEFAULT_CLIP_LIMIT,
        help="clahe: contrast clip limit (default = lp_clahe5 training recipe)",
    )
    p.add_argument(
        "--clahe-tile-grid",
        type=int,
        default=DEFAULT_TILE_GRID,
        help="clahe: tile grid size NxN (default = lp_clahe5 training recipe)",
    )
    p.add_argument(
        "--video-glob",
        default="**/*.mp4",
        help="glob (relative to /data) selecting input videos",
    )
    return p.parse_args()


def find_videos(pattern: str) -> list[Path]:
    vids = sorted(DATA_DIR.glob(pattern))
    # exclude anything already inside /results or hidden dirs
    vids = [v for v in vids if v.is_file() and not any(
        part.startswith(".") for part in v.parts)]
    if not vids:
        raise FileNotFoundError(f"no videos match /data/{pattern}")
    return vids


def sha256_head(path: Path, nbytes: int = 64 * 1024 * 1024) -> str:
    """Hash the first nbytes of a file (fast identity check for provenance)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()


def process_video(video: Path, transform, out_path: Path,
                  max_frames: int | None) -> dict:
    """Transform one video frame-by-frame and encode losslessly."""

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ffmpeg: raw BGR in -> mathematically lossless H.264 (-qp 0) out.
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", f"{fps}",
        "-i", "-",
        "-c:v", "libx264", "-qp", "0", "-preset", "veryfast",
        "-pix_fmt", "yuv444p",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    n = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        proc.stdin.write(transform(frame).tobytes())
        n += 1
        if n % 25000 == 0:
            rate = n / (time.time() - t0)
            print(f"  {n}/{total} frames ({rate:.0f} fps)", flush=True)
        if max_frames and n >= max_frames:
            print(f"  PREPROC_MAX_FRAMES={max_frames} reached (smoke test)",
                  flush=True)
            break
    cap.release()
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {video}")

    return {
        "input": str(video),
        "input_sha256_head": sha256_head(video),
        "output": str(out_path),
        "frames_processed": n,
        "fps": fps,
        "width": width,
        "height": height,
        "elapsed_s": round(time.time() - t0, 1),
    }


def main() -> None:
    args = parse_args()
    max_frames = int(os.environ.get("PREPROC_MAX_FRAMES", 0)) or None
    videos = find_videos(args.video_glob)
    transform, method_params = METHODS[args.method](args)
    print(f"method={args.method}  params={method_params}  "
          f"videos={len(videos)}", flush=True)

    records = []
    for video in videos:
        rel = video.relative_to(DATA_DIR)
        # drop the leading data-asset folder so the layout below the asset
        # is preserved (behavior-videos/<cam>/video.mp4)
        rel_below_asset = Path(*rel.parts[1:]) if len(rel.parts) > 1 else rel
        out_path = RESULTS_DIR / rel_below_asset
        print(f"processing {rel} -> {out_path.relative_to(RESULTS_DIR)}",
              flush=True)

        if args.method == "none":
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(video, out_path)  # bit-identical passthrough
            records.append({"input": str(video), "output": str(out_path),
                            "frames_processed": None, "copied": True})
        else:
            records.append(process_video(video, transform, out_path,
                                         max_frames))

    # provenance sidecar -- written LAST as the success marker.
    manifest = {
        "method": args.method,
        "method_params": method_params,
        "encoder": "libx264 -qp 0 yuv444p (lossless)" if args.method != "none"
                   else "file copy",
        "videos": records,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "preprocessing.json").write_text(
        json.dumps(manifest, indent=2))
    print("wrote /results/preprocessing.json", flush=True)


if __name__ == "__main__":
    main()
