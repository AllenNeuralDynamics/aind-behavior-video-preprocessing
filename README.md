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
| `none`  | passthrough (bit-identical file copy) |
| `clahe` | CLAHE contrast enhancement: grayscale -> CLAHE -> replicated to 3 identical channels. Parameter defaults (clip limit 5.0, 8x8 tiles) match the `lp_clahe5` eye-model training recipe exactly — keep them for those models. Values actually used are recorded in `preprocessing.json`. |

Adding a method: implement a factory in `code/preprocess/<name>.py`,
register it in `code/preprocess/registry.py` (METHODS).

## What the CLAHE parameters mean

CLAHE (Contrast-Limited Adaptive Histogram Equalization) boosts local
contrast: the frame is divided into a grid of tiles, each tile's
histogram is equalized independently (so a dark pupil region and a
bright IR-reflection region each get contrast appropriate to *their own*
brightness range), and results are blended smoothly across tile borders.

- **`--clahe-tile-grid` (default 8)** — the grid is N x N tiles, so 8
  means 64 local regions per frame. Larger N = smaller tiles = more
  local adaptation (finer, but can amplify local noise); smaller N
  approaches ordinary global histogram equalization.
- **`--clahe-clip-limit` (default 5.0)** — caps how much any tile's
  contrast may be amplified before the excess is redistributed. Higher =
  stronger enhancement but more amplified sensor noise; lower = gentler.
  For scale: OpenCV's own default is 2.0 (mild). 5.0 is a moderately
  aggressive setting chosen when training the eye models to sharpen
  pupil and corneal-reflection edges under IR illumination.

Illustrative settings:

| clip limit | tile grid | character |
|-----------:|----------:|-----------|
| 2.0 | 8 | mild, OpenCV default — general-purpose cleanup |
| **5.0** | **8** | **the `lp_clahe5` eye-model training recipe (our defaults)** |
| 10.0 | 16 | very aggressive + very local — strong edges, visible noise |

**The rule that matters:** a model must be inferred with the *same*
preprocessing it was trained with. For `lp_clahe5`-family eye models,
keep the defaults; changing them produces frames the model never saw in
training and silently degrades tracking. The parameters exist for
*other* teams/models trained with their own recipes.

## Parameters

- `--method` (`none` | `clahe`): which transform to apply.
- `--clahe-clip-limit` (float, default 5.0) / `--clahe-tile-grid`
  (int, default 8): see above; only used when `--method clahe`.
- `--video-glob` (default `**/*.mp4`): selects input videos under `/data`.
- env `PREPROC_MAX_FRAMES`: cap frames for smoke tests.

## Outputs

- `/results/<relative path below the data asset>/<video>.mp4` — transformed,
  lossless.
- `/results/preprocessing.json` — method, parameters, per-video frame counts
  and input hashes. Written last (success marker).

## Tests

`pytest tests/` — includes a pixel-parity test asserting `clahe` matches
the eye-tracking capsule's training recipe exactly.
