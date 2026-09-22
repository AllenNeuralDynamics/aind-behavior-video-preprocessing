# Behavior Video Preprocessing

Preprocess behavior videos before pose inference. Applies a selected
frame-level method to every frame and writes a **lossless** re-encoded
video, preserving the input's relative layout so downstream capsules
(e.g. Lightning Pose Inference and Evaluation) see the same paths.

## Why lossless

Models must see at inference exactly the pixel statistics they saw at
training. A lossy intermediate encode would add compression artifacts on
top of the preprocessing and silently degrade predictions. Output is
libx264 `-qp 0` (mathematically lossless), yuv444p, `.mp4`.

## Methods (the menu)

| method  | description |
|---------|-------------|
| `none`  | bypass — writes **no output videos**, only the manifest; downstream should keep using the original data asset |
| `clahe` | CLAHE contrast enhancement: grayscale → CLAHE → replicated to 3 identical channels. Parameter defaults (clip limit 5.0, 8×8 tiles) match the `lp_clahe5` eye-model training config file exactly — keep them for those models. Values actually used are recorded in `preprocessing.json`. |

Adding a method: implement a factory in `code/preprocess/<name>.py`,
register it in `code/preprocess/registry.py` (`METHODS`).

## What the CLAHE parameters mean

CLAHE (Contrast-Limited Adaptive Histogram Equalization) boosts local
contrast: the frame is divided into a grid of tiles, each tile's
histogram is equalized independently (so a dark pupil region and a
bright IR-reflection region each get contrast appropriate to *their own*
brightness range), and results are blended smoothly across tile borders.

- **clip limit (default 5.0)** — caps how much any tile's contrast may
  be amplified before the excess is redistributed. Higher = stronger
  enhancement but more amplified sensor noise; lower = gentler. For
  scale: OpenCV's own default is 2.0 (mild). 5.0 is a moderately
  aggressive setting chosen when training the eye models to sharpen
  pupil and corneal-reflection edges under IR illumination.
- **tile grid (default 8)** — the grid is N×N tiles, so 8 means 64
  local regions per frame. Larger N = smaller tiles = more local
  adaptation (finer, but can amplify local noise); smaller N approaches
  ordinary global histogram equalization.

Illustrative settings:

| clip limit | tile grid | character |
|-----------:|----------:|-----------|
| 2.0 | 8 | mild, OpenCV default — general-purpose cleanup |
| **5.0** | **8** | **the `lp_clahe5` eye-model training config file (our defaults)** |
| 10.0 | 16 | very aggressive + very local — strong edges, visible noise |

**The rule that matters:** a model must be inferred with the *same*
preprocessing it was trained with. For `lp_clahe5`-family eye models,
keep the defaults; changing them produces frames the model never saw in
training and silently degrades tracking. The parameters exist for
*other* teams/models trained with their own config files.

## The config file (the interface)

The capsule is driven by ONE thing: a config file. It reads
`/data/preprocessing.yaml` (or the path given as App Panel field 1 /
`--config`), applies the ordered steps, and writes the outputs. **No
config file = a loud error** that includes an example.

    # /data/preprocessing.yaml
    video_glob: "**/*[eE]ye*.mp4"   # optional, default **/*.mp4
    workers: 8                       # optional, default min(8, CPUs)
    steps:                           # required; [] = bypass (no outputs)
      - method: clahe
        clip_limit: 5.0
        tile_grid: 8

Steps compose per-frame (frame → step1 → step2 → **one** lossless
encode), so chunked parallelism is unaffected. Validation is loud:
unknown top-level keys, unknown methods, unknown per-method parameters,
and `none` used as a step all fail immediately with the allowed options
named. The manifest records the ordered steps with the parameter values
actually applied.

## Parallel chunked processing

With `workers: N` the video is split into N contiguous frame ranges;
each worker (its own process, explicit `spawn` context) decodes its
range, applies the transform, and encodes its own lossless part; parts
are joined by ffmpeg **stream copy** (no re-encode). Because the
transform is per-frame and every step is lossless, chunked output is
pixel-identical to sequential — enforced by tests and by built-in
verification. `--workers 1` runs the original sequential path.
`PREPROC_MAX_FRAMES` (smoke tests) forces sequential.

Failure philosophy — every ambiguity is a loud stop, never a guess:

- container reporting invalid metadata (fps / width / height / frame
  count missing or ≤ 0) → error before any work starts
- a chunk seek that does not land exactly on its start frame → error
  (with "rerun with `--workers 1`" guidance)
- a chunk reading fewer frames than assigned → error
- joined output whose frame count differs from frames processed → error

Each video's log starts with its facts
(`video facts: 658x492  60 fps  249916 frames (~69.4 min)`); chunk
progress lines report throughput as `frames/s processed` (machine
speed — not the video's fps).

## Outputs

- `/results/<relative path below the data asset>/<video>.mp4` —
  transformed, lossless (`clahe`); nothing for `none`.
- `/results/preprocessing.json` — ordered steps (with the parameter
  values actually applied), workers, per-video facts, per-chunk frame
  counts, and input content hashes. Written last
  (success marker).

## Tests

`python -m pytest tests -q` from `code/` — 18 tests:

- `test_clahe_parity.py` — the `clahe` transform matches the
  eye-tracking training config file exactly (pixel parity vs reference,
  `none` is identity, output channels identical).
- `test_chunked_parity.py` — chunked output is **pixel-identical** to
  sequential (max abs diff = 0) on a video whose frame count does not
  divide evenly; absurd worker counts are capped; short reads raise.
- `test_config_spec.py` — config precedence and full-override semantics,
  auto-detection, loud validation of every malformed-config case, and
  multi-step composition equal to manual nesting.
