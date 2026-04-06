# skate-cuts

CLI tool that analyzes raw skateboarding footage and generates timestamped edit logs with rough in/out cut points for good moments.

Uses PySceneDetect for shot boundary detection and Ollama vision models for frame classification.

## Requirements

- Python 3.9+
- [Ollama](https://ollama.ai) running locally with a vision model

## Install

```bash
pip install -r requirements.txt
```

Or install as a CLI tool:

```bash
pip install -e .
```

## Usage

```bash
# Basic usage — scene detection + classification
skate-cuts video.mp4

# Fixed interval sampling (every 3 seconds)
skate-cuts video.mp4 --interval 3

# Dry run — only scene detection, no model calls
skate-cuts video.mp4 --dry-run

# Use a different model
skate-cuts video.mp4 --model llava:13b

# Output only JSON
skate-cuts video.mp4 --format json

# Custom output name
skate-cuts video.mp4 -o my_edit
```

## Output

- **Terminal table** — quick overview of suggested clips
- **JSON** — full data with all frame classifications
- **CSV** — importable into spreadsheets or Premiere Pro
- **EDL** — Edit Decision List for NLE import

## Configuration

Categories, prompts, and thresholds are defined at the top of `skate_cuts.py` for easy modification.
