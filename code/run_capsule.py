"""Behavior Video Preprocessing capsule.

The capsule is driven by ONE thing: a config file. It reads
/data/preprocessing.yaml (or the path given via --config), applies the
ordered preprocessing steps to every matching video, and writes LOSSLESS
re-encoded videos to /results, mirroring the input's relative layout.

    # /data/preprocessing.yaml -- the config file (REQUIRED)
    video_glob: "**/*[eE]ye*.mp4"   # optional, default **/*.mp4
    workers: 8                       # optional, default min(8, CPUs)
    steps:                           # required; [] = bypass (no outputs)
      - method: clahe
        clip_limit: 5.0
        tile_grid: 8

Steps compose per-frame (frame -> step1 -> step2 -> ONE lossless encode),
so chunked parallelism is unaffected. Validation is loud: a missing
config file, unknown keys, unknown methods, or unknown per-method parameters
all fail immediately with the allowed options named.

Lossless: libx264 -qp 0 (mathematically lossless), yuv444p, .mp4.
Also writes results/preprocessing.json recording the ordered steps with
the parameter values actually applied (provenance; written last as the
success marker).

Env: PREPROC_MAX_FRAMES  optional frame cap for smoke tests (sequential).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
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
#: auto-detected config location (Manager-writable in the future pipeline)
DEFAULT_CONFIG = Path("/data/preprocessing.yaml")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="path to the preprocessing config YAML "
             f"(default: {DEFAULT_CONFIG})",
    )
    return p.parse_args(argv)


def load_spec(path: Path) -> dict:
    """Load and validate the config file. The config file IS the interface."""
    if not path.is_file():
        raise FileNotFoundError(
            f"config file not found: {path}\n"
            "This capsule is driven by a config file. Provide one, e.g.:\n"
            "  # /data/preprocessing.yaml\n"
            "  video_glob: \"**/*[eE]ye*.mp4\"\n"
            "  steps:\n"
            "    - method: clahe\n"
            "      clip_limit: 5.0\n"
            "      tile_grid: 8\n"
        )
    import yaml  # PyYAML (pinned in the Dockerfile)

    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    allowed = {"video_glob", "workers", "steps"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"{path}: unknown key(s) {sorted(unknown)}; "
                         f"allowed: {sorted(allowed)}")
    if "steps" not in raw:
        raise ValueError(f"{path}: 'steps' is required (may be [] to bypass)")
    steps_raw = raw["steps"]
    if not isinstance(steps_raw, list):
        raise ValueError(f"{path}: 'steps' must be a list (may be empty)")
    steps = []
    for i, step in enumerate(steps_raw):
        if not isinstance(step, dict) or "method" not in step:
            raise ValueError(f"{path}: steps[{i}] must be a mapping with "
                             "a 'method' key")
        method = step["method"]
        if method not in METHODS:
            raise ValueError(f"{path}: steps[{i}]: unknown method "
                             f"'{method}'; available: {sorted(METHODS)}")
        if method == "none":
            raise ValueError(f"{path}: steps[{i}]: 'none' is not a step; "
                             "use an empty steps list to bypass")
        params = {k: v for k, v in step.items() if k != "method"}
        METHODS[method](params)  # validate params now, fail loudly
        steps.append({"method": method, "params": params})
    return {
        "video_glob": raw.get("video_glob", "**/*.mp4"),
        "workers": int(raw.get("workers", min(8, os.cpu_count() or 1))),
        "steps": steps,
        "source": str(path),
    }


def build_pipeline(steps: list) -> tuple:
    """Compose the steps' transforms into one per-frame function.

    Returns (frame_fn, steps_used) where steps_used records each step's
    method and the parameter values actually applied.
    """
    fns, used = [], []
    for s in steps:
        fn, params_used = METHODS[s["method"]](s.get("params"))
        fns.append(fn)
        used.append({"method": s["method"], "params": params_used})

    def apply(frame):
        for f in fns:
            frame = f(frame)
        return frame

    return apply, used


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
                     steps: list,
                     width: int, height: int, fps: float,
                     label: str) -> int:
    """Decode [start, start+n_frames) of video, transform, encode losslessly.

    Runs in a worker process (spawn): rebuilds the step transforms
    locally from the picklable steps spec and composes them in order.
    Returns the number of frames written; raises if the count is short
    (e.g. inaccurate seek or truncated read), so failures are loud.
    """
    transform, _ = build_pipeline(steps)
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


def process_video(video: Path, out_path: Path, steps: list,
                  workers: int, max_frames: int | None) -> dict:
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
        n = _transform_range(str(video), 0, total, str(out_path), steps,
                             width, height, fps, label="seq")
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
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {
                pool.submit(_transform_range, str(video), start, count,
                            str(part), steps, width, height, fps,
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
    spec = load_spec(Path(args.config))
    workers = max(1, int(spec["workers"]))
    videos = find_videos(spec["video_glob"])
    _, steps_used = build_pipeline(spec["steps"])
    method_label = "+".join(s["method"] for s in steps_used) or "none"
    print(f"steps={method_label}  "
          f"params={[s['params'] for s in steps_used]}  "
          f"videos={len(videos)}  workers={workers}  "
          f"(config: {spec['source']})", flush=True)

    records = []
    for video in videos:
        rel = video.relative_to(DATA_DIR)
        rel_below_asset = Path(*rel.parts[1:]) if len(rel.parts) > 1 else rel
        out_path = RESULTS_DIR / rel_below_asset
        print(f"processing {rel} -> {out_path.relative_to(RESULTS_DIR)}",
              flush=True)

        if not spec["steps"]:
            # bypass: no transform requested -- do not spend time/space
            # copying inputs; downstream should use the original asset.
            print("  no steps: bypass (no output video written)",
                  flush=True)
            records.append({"input": str(video), "output": None,
                            "frames_processed": None, "bypassed": True})
        else:
            records.append(process_video(video, out_path, spec["steps"],
                                         workers, max_frames))

    manifest = {
        "method": method_label,
        "steps": steps_used,
        "spec_source": spec["source"],
        "encoder": "libx264 -qp 0 yuv444p (lossless); chunked parts joined "
                   "by ffmpeg stream copy" if spec["steps"]
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
