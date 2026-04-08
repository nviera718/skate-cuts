# skate-cuts

CLI tool to split skateboarding compilation videos into individual clips.

## How it works

1. **Scene detection** — PySceneDetect's AdaptiveDetector finds hard cuts in compilation videos
2. **Duration filtering** — Drops clips too short (<1.5s transitions) or too long (>30s filler)
3. **Motion filtering** (optional) — OpenCV optical flow scores motion per clip
4. **Output** — Timestamps to terminal (table/json/csv) or split video into files (lossless via ffmpeg)

## Project structure

- `skate_cuts.py` — Single-file CLI application (~240 lines)
- `setup.py` — Package config, installs `skate-cuts` command
- `requirements.txt` — Dependencies: scenedetect, click, tqdm

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

```bash
# Basic: detect scenes and print timestamps
skate-cuts video.mp4

# Split into individual files (lossless, fast)
skate-cuts video.mp4 --split -o ./clips/

# Tune scene detection sensitivity (lower = more cuts)
skate-cuts video.mp4 -t 2.0

# Filter short/long clips
skate-cuts video.mp4 --min-duration 2.0 --max-duration 15.0

# Score and filter by motion
skate-cuts video.mp4 --filter-motion --motion-threshold 5.0

# JSON output (pipeable)
skate-cuts video.mp4 --format json
```

## Dependencies

- `scenedetect[opencv]>=0.6` — Scene detection + OpenCV
- `click>=8.0` — CLI framework
- `tqdm>=4.64` — Progress bars
- `ffmpeg` — Required on system PATH for `--split`

## Repo

- GitHub: nviera718/skate-cuts
- Secrets: `GEMINI_API_KEY` (legacy, not currently used)
