"""Behavior Video Preprocessing capsule.

Finds behavior video(s) under /data, applies the selected preprocessing
method to every frame, and writes a LOSSLESS re-encoded video to /results,
mirroring the input's relative path so downstream capsules see the same
layout (e.g. behavior-videos/Eye/video.mp4 stays at that relative path).

Also writes results/preprocessing.json recording the method, its
parameters, and per-video facts, for provenance.

Lossless choice: libx264 with -qp 0 (mathematically lossless), yuv444p,
keeping the .mp4 extension so downstream path expectations are unchanged.
Frames are piped raw to ffmpeg.

PARALLEL CHUNKING: with --workers N (default: all CPUs, max 8),
the video is split into N contiguous frame ranges; each worker decodes its
range, applies the transform, and encodes its own lossless part; the parts
are then concatenated with ffmpeg stream copy (no re-encode). Because the
transform is per-frame and every step is lossless, the chunked output is
pixel-identical to the sequential one -- enforced by tests and by built-in
frame-count verification. --workers 1 runs the original sequential path.

Usage:
    python code/run_capsule.py --method clahe --video-glob "**/*[eE]ye*.mp4"
    python code/run_capsule.py --method none                # passthrough copy
Env:
    PREPROC_MAX_FRAMES  optional cap for smoke tests (forces sequential).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2

from preprocess.clahe import DEFAULT_CLIP_LIMIT, DEFAULT_TILE_GRID
from preprocess.registry import METHODS

DATA_DIR = Path("/data")
RESULTS_DIR = Path("/results")
SCRATCH_DIR = Path(os.environ.get("PREPROC_SCRATCH", "/scratch"))


def parse_args(argv=None) -> argparse.Namespace:
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
    p.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="parallel chunk workers per video (default: all CPUs, max 8; "
             "1 = sequential, original behavior)",
    )
    return p.parse_args(argv)


def find_videos(pattern: str) -> list[Path]:
    vids = sorted(DATA_DIR.glob(pattern))
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


def video_facts(video: Path) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    facts = {"fps": fps, "width": width, "height": height,
             "frame_count": total}
    bad = {k: v for k, v in facts.items() if not v or v <= 0}
    if bad:
        raise RuntimeError(
            f"{video}: container reports invalid metadata: {bad}. "
            "Refusing to guess -- the output must preserve the input's real "
            "dimensions and timing. Fix or re-mux the input video first."
        )
    return fps, width, height, total


def _ffmpeg_writer(out_path: Path, width: int, height: int, fps: float):
    """Raw BGR in -> mathematically lossless H.264 (-qp 0) out."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", f"{fps}",
        "-i", "-",
        "-c:v", "libx264", "-qp", "0", "-preset", "veryfast",
        "-pix_fmt", "yuv444p",
        str(out_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def _transform_range(video: str, start: int, n_frames: int, out_path: str,
                     method: str, args_ns: argparse.Namespace,
                     width: int, height: int, fps: float,
                     label: str) -> int:
    """Decode [start, start+n_frames) of video, transform, encode losslessly.

    Runs in a worker process (spawn): rebuilds the frame transform locally.
    Returns the number of frames written; raises if the count is short
    (e.g. inaccurate seek or truncated read), so failures are loud.
    """
    transform, _ = METHODS[method](args_ns)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        got = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if got != start:
            raise RuntimeError(
                f"{label}: seek to frame {start} landed at {got} -- "
                "container not frame-accurately seekable; rerun with --workers 1")
    proc = _ffmpeg_writer(Path(out_path), width, height, fps)
    n = 0
    t0 = time.time()
    while n < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        proc.stdin.write(transform(frame).tobytes())
        n += 1
        if n % 25000 == 0:
            rate = n / (time.time() - t0)
            print(f"  {label}: {n}/{n_frames} frames ({rate:.0f} fps)",
                  flush=True)
    cap.release()
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"{label}: ffmpeg failed")
    if n != n_frames:
        raise RuntimeError(
            f"{label}: wrote {n} frames, expected {n_frames} (truncated read)")
    return n


def _concat_parts(parts: list[Path], out_path: Path) -> None:
    """Losslessly join encoded parts with ffmpeg stream copy (no re-encode)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    list_file = parts[0].parent / "parts.txt"
    list_file.write_text("".join(f"file '{p}'\n" for p in parts))
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "concat", "-safe", "0", "-i", str(list_file),
         "-c", "copy", str(out_path)],
        check=True,
    )


def process_video(video: Path, out_path: Path, method: str,
                  args: argparse.Namespace, workers: int,
                  max_frames: int | None) -> dict:
    """Transform one video and encode losslessly, chunked across workers."""
    fps, width, height, total = video_facts(video)
    print(f"  video facts: {width}x{height}  {fps:g} fps  {total} frames "
          f"(~{total / fps / 60:.1f} min)", flush=True)
    t0 = time.time()

    if max_frames:
        total = min(total, max_frames)
        workers = 1
        print(f"  PREPROC_MAX_FRAMES={max_frames} (smoke test, sequential)",
              flush=True)
    workers = max(1, min(workers, total))

    if workers == 1:
        n = _transform_range(str(video), 0, total, str(out_path), method,
                             args, width, height, fps, label="seq")
        chunk_frames = [n]
    else:
        bounds = [round(i * total / workers) for i in range(workers + 1)]
        part_dir = SCRATCH_DIR / "preproc_parts" / video.stem
        part_dir.mkdir(parents=True, exist_ok=True)
        parts = [part_dir / f"part_{i:04d}.mp4" for i in range(workers)]
        jobs = [(bounds[i], bounds[i + 1] - bounds[i], parts[i], f"chunk {i}")
                for i in range(workers)]
        print(f"  chunking into {workers} ranges of ~{total // workers} frames",
              flush=True)
        chunk_frames = [0] * workers
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_transform_range, str(video), start, count,
                            str(part), method, args, width, height, fps,
                            label): i
                for i, (start, count, part, label) in enumerate(jobs)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                chunk_frames[i] = fut.result()  # re-raises worker errors
        _concat_parts(parts, out_path)
        n = sum(chunk_frames)
        shutil.rmtree(part_dir, ignore_errors=True)

    # verify the joined output advertises the same frame count
    out_total = video_facts(out_path)[3]
    if out_total != n:
        raise RuntimeError(
            f"output frame count {out_total} != processed {n} for {out_path}")

    return {
        "input": str(video),
        "input_sha256_head": sha256_head(video),
        "output": str(out_path),
        "frames_processed": n,
        "fps": fps,
        "width": width,
        "height": height,
        "workers": workers,
        "chunk_frames": chunk_frames,
        "elapsed_s": round(time.time() - t0, 1),
    }


def main() -> None:
    args = parse_args()
    max_frames = int(os.environ.get("PREPROC_MAX_FRAMES", 0)) or None
    workers = max(1, args.workers)
    videos = find_videos(args.video_glob)
    _, method_params = METHODS[args.method](args)
    print(f"method={args.method}  params={method_params}  "
          f"videos={len(videos)}  workers={workers}", flush=True)

    records = []
    for video in videos:
        rel = video.relative_to(DATA_DIR)
        rel_below_asset = Path(*rel.parts[1:]) if len(rel.parts) > 1 else rel
        out_path = RESULTS_DIR / rel_below_asset
        print(f"processing {rel} -> {out_path.relative_to(RESULTS_DIR)}",
              flush=True)

        if args.method == "none":
            # bypass: no transform requested -- do not spend time/space
            # copying inputs; downstream should use the original asset.
            print("  method=none: bypass (no output video written)",
                  flush=True)
            records.append({"input": str(video), "output": None,
                            "frames_processed": None, "bypassed": True})
        else:
            records.append(process_video(video, out_path, args.method, args,
                                         workers, max_frames))

    manifest = {
        "method": args.method,
        "method_params": method_params,
        "encoder": "libx264 -qp 0 yuv444p (lossless); chunked parts joined "
                   "by ffmpeg stream copy" if args.method != "none"
                   else "none (bypassed)",
        "workers": workers,
        "videos": records,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "preprocessing.json").write_text(
        json.dumps(manifest, indent=2))
    print("wrote /results/preprocessing.json", flush=True)


if __name__ == "__main__":
    main()
