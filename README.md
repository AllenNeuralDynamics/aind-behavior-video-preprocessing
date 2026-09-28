# Behavior Video Preprocessing

This capsule preprocesses behavior videos (e.g. CLAHE contrast enhancement of the Eye camera video) and writes lossless copies of the processed videos to `/results`. Use it to prepare videos for pose-estimation models that were trained on preprocessed frames — for example the `lp_clahe5` eye-tracking models, whose training recipe is this capsule's default setting.

> **Rule of thumb:** a model must see the same preprocessing at inference that it saw at training. If you are preparing videos for the lp_clahe5 eye models, use the default CLAHE settings shown below and don't change them.

## How to Install

Nothing to install — the capsule's Dockerfile builds its environment (ffmpeg, OpenCV, PyYAML) automatically on Code Ocean.

## How to Run

1. **Attach your data asset(s)** containing the videos. Videos are found under `/data`.
2. **Provide a config file** named `preprocessing.yaml` (attach it as a small data asset, so it lands at `/data/preprocessing.yaml`). See Inputs below.
3. **Run the capsule.** No other arguments are needed. 
4. **Collect the results.** Processed videos appear under `/results`, in the same folder layout as the input, plus a `preprocessing.json` file describing exactly what was done.

## Inputs (Config)

The config file is the capsule's only interface — everything is controlled from it. A ready-to-use example for the eye videos:

```yaml
video_glob: "**/*[eE]ye*.mp4"   # process the Eye camera video only
workers: 8                       # parallel workers (optional)
steps:                           # what to do to each frame (required)
  - method: clahe
    clip_limit: 5.0
    tile_grid: 8
```

| key | required? | meaning | default |
|---|---|---|---|
| `video_glob` | no | which videos to process, as a glob pattern relative to `/data` | `**/*.mp4` (all mp4s) |
| `workers` | no | how many videos chunks to process in parallel; `1` = sequential | `min(8, CPUs)` |
| `steps` | **yes** | ordered list of preprocessing steps applied to every frame | — |

If you list several steps, they are applied in order to each frame, and the video is still encoded only once at the end.

### Turning preprocessing off (bypass)

To run without any preprocessing, write an **explicit empty list**:

```yaml
steps: []
```

The capsule then writes no output videos — only the `preprocessing.json` record — and downstream analyses should keep using the original data asset. Note that simply *omitting* the config file, or leaving out `steps`, is an **error**, not a bypass: skipping preprocessing has to be stated on purpose.

### If your config has a mistake

The capsule fails immediately with a message naming what's allowed — it never silently falls back to defaults. This happens for: a missing config file, an unknown key, an unknown method name, or a misspelled parameter (e.g. `cliplimit` instead of `clip_limit`). Read the error message; it tells you the valid options.

## Outputs

Everything lands in `/results`:

- **Processed videos**, mirroring the input folder layout (e.g. `/data/<asset>/a/b/Eye_video.mp4` → `/results/a/b/Eye_video.mp4`). The encoding is *mathematically lossless* (`libx264 -qp 0`, `yuv444p`, `.mp4`), so no quality is lost beyond the preprocessing you asked for.
- **`preprocessing.json`** — a record of what was actually done: the steps and parameter values applied, each input video (with a checksum), frame counts, dimensions, fps, and timing. This file is written last, so **if it's present, the run succeeded**; if it's missing, treat the run as failed.

## Methods

Currently one method is available.

### `clahe`

Contrast Limited Adaptive Histogram Equalization. This boosts local contrast: instead of stretching the brightness of the whole image at once, it divides each frame into a grid of small tiles and equalizes the contrast *within each tile separately*. That makes dim, low-contrast regions (like a dark pupil against a dark eyelid) much easier to see and track, without blowing out already-bright areas. The capsule converts each frame to grayscale first, applies CLAHE, then replicates the enhanced grayscale image to 3 identical channels (so the output is still a standard color-format video that models expect).

| parameter | meaning | default |
|---|---|---|
| `clip_limit` | How strongly contrast may be amplified in each tile. Higher values mean more aggressive enhancement — dim details become more visible, but noise gets amplified too. Lower values are gentler. (`1.0` ≈ almost no enhancement; OpenCV's usual default is `2.0`; this capsule defaults to `5.0` because that's what the lp_clahe5 models were trained with.) | `5.0` |
| `tile_grid` | How finely the frame is divided for local equalization: `8` means an 8×8 grid, i.e. 64 tiles, each enhanced based on its own local brightness. A larger number means smaller tiles → more local adaptation (good for small features, but can look patchy); a smaller number means bigger tiles → smoother, more global enhancement. | `8` |

The defaults match the **lp_clahe5** eye-model training recipe exactly — for those models, keep them.

## Parallel Chunked Processing

Each video is split into `workers` frame ranges that are processed in parallel, each encoded losslessly, then joined with an ffmpeg stream copy (no re-encode). The result is identical to processing the video sequentially. Set `workers: 1` in the config to process sequentially.

## Tests

Run `python -m pytest ../tests -q` from `code/`. The tests check that the CLAHE defaults match the lp_clahe5 training recipe, that chunked output is pixel-identical to sequential output, and that config validation fails loudly.
