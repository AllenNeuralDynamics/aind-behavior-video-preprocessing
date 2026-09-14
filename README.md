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

| method   | description                                                       |
|----------|-------------------------------------------------------------------|
| `none`   | passthrough (bit-identical file copy)                             |
| `clahe5` | CLAHE clipLimit 5.0, 8x8 tiles, grayscale replicated to 3 channels. Matches the `lp_clahe5` eye-model training recipe exactly. Defaults match the lp_clahe5 training recipe exactly -- keep them for eye models; values used are recorded in preprocessing.json. |

Adding a method: implement `frame -> frame` in `code/preprocess/<name>.py`,
register it in `code/preprocess/registry.py` (METHODS + METHOD_PARAMS).

## Parameters

- `--method` (`none` | `clahe5`): which transform to apply.
- `--video-glob` (default `**/*.mp4`): selects input videos under `/data`.
- env `PREPROC_MAX_FRAMES`: cap frames for smoke tests.

## Outputs

- `/results/<relative path below the data asset>/<video>.mp4` — transformed,
  lossless.
- `/results/preprocessing.json` — method, parameters, per-video frame counts
  and input hashes. Written last (success marker).

## Tests

`pytest tests/` — includes a pixel-parity test asserting `clahe5` is
identical to the eye-tracking capsule's training recipe.
