#!/usr/bin/env python3
"""
skate-cuts: Automated skateboarding video edit log generator.

Takes raw skateboarding footage and outputs timestamped cut points
for good moments using scene detection + vision model classification.
"""

import base64
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import click
import requests
from PIL import Image
from tqdm import tqdm

# =============================================================================
# CONFIGURATION — Edit these to tune classification behavior
# =============================================================================

OLLAMA_URL = "http://localhost:11434/api/chat"

DEFAULT_MODEL = "gemma3:27b-it-q8_0"

# How many sequential keyframes to send per API request (temporal context)
BATCH_SIZE = 3

# Max video duration (minutes) before showing a warning
MAX_DURATION_WARN_MINUTES = 30

# Max pixel dimension for keyframes sent to the model
MAX_KEYFRAME_SIZE = 1024

# Scene detection threshold (lower = more sensitive, default ~27)
SCENE_THRESHOLD = 27.0

# Default fixed-interval sampling in seconds
DEFAULT_INTERVAL = 2.0

# Frame categories — add/remove/rename as needed
CATEGORIES = {
    "trick_attempt": "skater actively attempting a trick",
    "trick_landed": "skater has landed/completed a trick",
    "bail": "failed attempt / fall",
    "approach": "skating toward an obstacle, setup for a trick",
    "celebration": "reaction after landing",
    "broll": "interesting environmental/artistic shots",
    "filler": "walking between spots, talking, setting up camera, etc.",
}

# System prompt for the vision model
SYSTEM_PROMPT = """You are a skateboarding footage analyst. You will be shown keyframes from a skateboarding video in chronological order.

For each frame, classify it into exactly ONE of these categories:
{categories}

Respond with a JSON array (one object per frame) with these fields:
- "category": one of the category keys listed above
- "confidence": "high", "medium", or "low"
- "description": a short (1 sentence) description of what's happening in the frame

IMPORTANT:
- Return ONLY valid JSON, no markdown fences, no extra text.
- The array length must match the number of frames shown.
- Pay attention to body position, board orientation, obstacles, and motion blur to determine trick vs approach vs filler.
- If a skater is in the air or the board is flipping, that's likely trick_attempt or trick_landed.
- If the skater is on the ground rolling toward something, that's approach.
- Artistic angles, fisheye closeups of wheels/trucks, or scenic shots are broll.
"""

# IN/OUT point padding in seconds
IN_PADDING = 1.5
OUT_PADDING = 1.5

# Categories that are "interesting" for the edit
KEEP_CATEGORIES = {"trick_attempt", "trick_landed", "bail", "approach", "celebration", "broll"}


# =============================================================================
# SCENE DETECTION
# =============================================================================

def detect_scenes(video_path, threshold=SCENE_THRESHOLD):
    """Use PySceneDetect ContentDetector to find shot boundaries."""
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    click.echo(f"Detecting scenes (threshold={threshold})...")
    scene_manager.detect_scenes(video, show_progress=True)
    scene_list = scene_manager.get_scene_list()

    if not scene_list:
        click.echo("No scene cuts detected.")
        return []

    click.echo(f"Found {len(scene_list)} scenes.")
    return scene_list


def get_video_duration(video_path):
    """Get video duration in seconds using OpenCV."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0:
        return frame_count / fps
    return 0


def extract_keyframes_from_scenes(video_path, scene_list, tmp_dir):
    """Extract a representative keyframe from each scene (midpoint)."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    keyframes = []

    for i, (start, end) in enumerate(tqdm(scene_list, desc="Extracting keyframes")):
        mid_frame = (start.get_frames() + end.get_frames()) // 2
        mid_time = mid_frame / fps if fps > 0 else 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        if not ret:
            continue

        img_path = os.path.join(tmp_dir, f"scene_{i:04d}_{mid_time:.2f}s.jpg")
        cv2.imwrite(img_path, frame)

        keyframes.append({
            "index": i,
            "frame_path": img_path,
            "timestamp": mid_time,
            "scene_start": start.get_seconds(),
            "scene_end": end.get_seconds(),
        })

    cap.release()
    return keyframes


def extract_keyframes_at_interval(video_path, interval, tmp_dir):
    """Sample frames at a fixed interval instead of using scene detection."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    frame_step = int(fps * interval)

    keyframes = []
    idx = 0
    current_frame = 0

    pbar = tqdm(total=int(duration / interval), desc="Sampling frames")
    while current_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = current_frame / fps if fps > 0 else 0
        img_path = os.path.join(tmp_dir, f"frame_{idx:04d}_{timestamp:.2f}s.jpg")
        cv2.imwrite(img_path, frame)

        keyframes.append({
            "index": idx,
            "frame_path": img_path,
            "timestamp": timestamp,
            "scene_start": max(0, timestamp - interval / 2),
            "scene_end": min(duration, timestamp + interval / 2),
        })

        idx += 1
        current_frame += frame_step
        pbar.update(1)

    pbar.close()
    cap.release()
    click.echo(f"Sampled {len(keyframes)} frames at {interval}s intervals.")
    return keyframes


# =============================================================================
# FRAME CLASSIFICATION (Ollama Vision API)
# =============================================================================

def resize_keyframe(img_path, max_size=MAX_KEYFRAME_SIZE):
    """Resize image so longest edge is max_size. Returns base64 string."""
    img = Image.open(img_path)
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Save to buffer and encode
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_category_text():
    """Format categories for the system prompt."""
    lines = []
    for key, desc in CATEGORIES.items():
        lines.append(f'- "{key}": {desc}')
    return "\n".join(lines)


def classify_batch(batch, model, ollama_url=OLLAMA_URL):
    """Send a batch of keyframes to Ollama for classification."""
    images = []
    for kf in batch:
        images.append(resize_keyframe(kf["frame_path"]))

    n = len(batch)
    timestamps = ", ".join(f'{kf["timestamp"]:.1f}s' for kf in batch)

    user_content = (
        f"Here are {n} sequential keyframes from a skateboarding video "
        f"(timestamps: {timestamps}). "
        f"Classify each frame. Return a JSON array of {n} objects."
    )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(categories=build_category_text()),
            },
            {
                "role": "user",
                "content": user_content,
                "images": images,
            },
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
        },
    }

    try:
        resp = requests.post(ollama_url, json=payload, timeout=120)
        resp.raise_for_status()
        result = resp.json()
        content = result["message"]["content"]

        parsed = json.loads(content)

        # Normalize: if the model returns a dict with an array inside, extract it
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    parsed = v
                    break

        if isinstance(parsed, list):
            return parsed
        else:
            return [parsed]

    except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
        click.echo(f"\n  Warning: API call failed ({e}), marking batch as filler.", err=True)
        return [
            {"category": "filler", "confidence": "low", "description": "classification failed"}
            for _ in batch
        ]


def classify_keyframes(keyframes, model):
    """Classify all keyframes in batches."""
    results = []
    batches = [keyframes[i:i + BATCH_SIZE] for i in range(0, len(keyframes), BATCH_SIZE)]

    for batch in tqdm(batches, desc="Classifying frames"):
        classifications = classify_batch(batch, model)

        # Pad or truncate to match batch size
        while len(classifications) < len(batch):
            classifications.append({"category": "filler", "confidence": "low", "description": "no classification"})
        classifications = classifications[:len(batch)]

        for kf, cls in zip(batch, classifications):
            # Validate category
            cat = cls.get("category", "filler")
            if cat not in CATEGORIES:
                cat = "filler"

            results.append({
                **kf,
                "category": cat,
                "confidence": cls.get("confidence", "low"),
                "description": cls.get("description", ""),
            })

    return results


# =============================================================================
# EDIT LOG GENERATION
# =============================================================================

def generate_edit_log(classified_frames):
    """Generate edit decision log with IN/OUT points from classified frames."""
    if not classified_frames:
        return []

    # Group consecutive frames by category
    groups = []
    current_group = None

    for frame in classified_frames:
        cat = frame["category"]
        if current_group is None or current_group["category"] != cat:
            if current_group is not None:
                groups.append(current_group)
            current_group = {
                "category": cat,
                "frames": [frame],
                "start": frame["scene_start"],
                "end": frame["scene_end"],
            }
        else:
            current_group["frames"].append(frame)
            current_group["end"] = frame["scene_end"]

    if current_group:
        groups.append(current_group)

    # Build edit points
    clips = []
    for i, group in enumerate(groups):
        cat = group["category"]

        if cat not in KEEP_CATEGORIES:
            continue

        in_point = max(0, group["start"] - IN_PADDING)
        out_point = group["end"] + OUT_PADDING

        # Look ahead: if this is approach/trick_attempt followed by trick_landed,
        # extend the clip to include it
        if cat in ("approach", "trick_attempt") and i + 1 < len(groups):
            next_g = groups[i + 1]
            if next_g["category"] in ("trick_landed", "celebration"):
                out_point = next_g["end"] + OUT_PADDING
                # Skip the next group so we don't double-count
                groups[i + 1] = {**next_g, "category": "_consumed"}

        # Look further: if trick_landed is followed by celebration, extend
        if cat == "trick_landed" and i + 1 < len(groups):
            next_g = groups[i + 1]
            if next_g["category"] == "celebration":
                out_point = next_g["end"] + OUT_PADDING
                groups[i + 1] = {**next_g, "category": "_consumed"}

        descriptions = [f["description"] for f in group["frames"] if f.get("description")]
        best_confidence = "low"
        for f in group["frames"]:
            c = f.get("confidence", "low")
            if c == "high":
                best_confidence = "high"
            elif c == "medium" and best_confidence != "high":
                best_confidence = "medium"

        clips.append({
            "clip_number": len(clips) + 1,
            "in_point": round(in_point, 2),
            "out_point": round(out_point, 2),
            "duration": round(out_point - in_point, 2),
            "category": cat,
            "confidence": best_confidence,
            "description": descriptions[0] if descriptions else "",
            "keep": cat in ("trick_landed", "bail", "broll", "celebration"),
            "note": "bail - keep for montage?" if cat == "bail" else "",
        })

    return clips


def format_timestamp(seconds):
    """Format seconds as HH:MM:SS.mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# =============================================================================
# OUTPUT FORMATTERS
# =============================================================================

def print_table(clips):
    """Pretty-print the edit log as a terminal table."""
    if not clips:
        click.echo("No interesting clips found.")
        return

    click.echo()
    header = f"{'#':>3}  {'IN':>12}  {'OUT':>12}  {'DUR':>6}  {'CATEGORY':<15}  {'CONF':<6}  {'KEEP':<4}  DESCRIPTION"
    click.echo(header)
    click.echo("-" * len(header))

    for c in clips:
        keep_str = "YES" if c["keep"] else ""
        note = f" ({c['note']})" if c["note"] else ""
        click.echo(
            f"{c['clip_number']:>3}  "
            f"{format_timestamp(c['in_point']):>12}  "
            f"{format_timestamp(c['out_point']):>12}  "
            f"{c['duration']:>5.1f}s  "
            f"{c['category']:<15}  "
            f"{c['confidence']:<6}  "
            f"{keep_str:<4}  "
            f"{c['description']}{note}"
        )

    click.echo()
    total = sum(c["duration"] for c in clips)
    keepers = [c for c in clips if c["keep"]]
    click.echo(f"Total clips: {len(clips)} | Suggested keeps: {len(keepers)} | Total duration: {total:.1f}s")


def write_json(clips, classified_frames, output_path):
    """Write full results as JSON."""
    data = {
        "clips": clips,
        "all_frames": [
            {k: v for k, v in f.items() if k != "frame_path"}
            for f in classified_frames
        ],
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    click.echo(f"JSON written to: {output_path}")


def write_csv(clips, output_path):
    """Write clips as CSV for Premiere Pro import."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Clip", "In", "Out", "Duration", "Category", "Confidence", "Keep", "Description"])
        for c in clips:
            writer.writerow([
                c["clip_number"],
                format_timestamp(c["in_point"]),
                format_timestamp(c["out_point"]),
                c["duration"],
                c["category"],
                c["confidence"],
                "YES" if c["keep"] else "NO",
                c["description"],
            ])
    click.echo(f"CSV written to: {output_path}")


def write_edl(clips, output_path, fps=29.97):
    """Write a simple CMX 3600 EDL file."""
    with open(output_path, "w") as f:
        f.write("TITLE: skate-cuts\n")
        f.write(f"FCM: NON-DROP FRAME\n\n")

        for c in clips:
            event = c["clip_number"]
            src_in = format_timestamp(c["in_point"])
            src_out = format_timestamp(c["out_point"])
            # For EDL, record times are cumulative but we'll keep it simple
            f.write(f"{event:03d}  001      V     C        {src_in} {src_out} {src_in} {src_out}\n")
            f.write(f"* CATEGORY: {c['category']}\n")
            if c["description"]:
                f.write(f"* {c['description']}\n")
            f.write("\n")

    click.echo(f"EDL written to: {output_path}")


# =============================================================================
# CLI
# =============================================================================

@click.command()
@click.argument("video", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Output base name (default: video name)")
@click.option("--model", "-m", default=DEFAULT_MODEL, help=f"Ollama model name (default: {DEFAULT_MODEL})")
@click.option("--interval", "-i", type=float, default=None, help="Fixed sampling interval in seconds (bypasses scene detection)")
@click.option("--threshold", "-t", type=float, default=SCENE_THRESHOLD, help=f"Scene detection threshold (default: {SCENE_THRESHOLD})")
@click.option("--dry-run", is_flag=True, help="Only run scene detection, skip classification")
@click.option("--format", "fmt", type=click.Choice(["all", "json", "csv", "edl", "table"]), default="all", help="Output format")
@click.option("--ollama-url", default=OLLAMA_URL, help="Ollama API URL")
def main(video, output, model, interval, threshold, dry_run, fmt, ollama_url):
    """Analyze skateboarding footage and generate an edit decision log.

    VIDEO is the path to a video file (mp4, mov, avi, etc.)
    """
    global OLLAMA_URL
    OLLAMA_URL = ollama_url

    video_path = str(Path(video).resolve())
    if output is None:
        output = Path(video).stem

    click.echo(f"skate-cuts: {Path(video).name}")
    click.echo(f"Model: {model}")

    # Check duration
    duration = get_video_duration(video_path)
    if duration > 0:
        click.echo(f"Duration: {format_timestamp(duration)}")
        if duration > MAX_DURATION_WARN_MINUTES * 60:
            click.echo(
                f"Warning: Video is over {MAX_DURATION_WARN_MINUTES} minutes. "
                "This will generate many API calls.",
                err=True,
            )

    # Pass 1: Scene detection / frame sampling
    with tempfile.TemporaryDirectory(prefix="skate_cuts_") as tmp_dir:
        if interval is not None:
            keyframes = extract_keyframes_at_interval(video_path, interval, tmp_dir)
        else:
            scene_list = detect_scenes(video_path, threshold=threshold)
            if not scene_list:
                click.echo("Falling back to fixed-interval sampling (2s)...")
                keyframes = extract_keyframes_at_interval(video_path, DEFAULT_INTERVAL, tmp_dir)
            else:
                keyframes = extract_keyframes_from_scenes(video_path, scene_list, tmp_dir)

        if not keyframes:
            click.echo("No keyframes extracted. Is the video valid?", err=True)
            sys.exit(1)

        click.echo(f"Extracted {len(keyframes)} keyframes.")

        if dry_run:
            click.echo("\n--dry-run: Skipping classification. Scene timestamps:")
            for kf in keyframes:
                click.echo(f"  [{format_timestamp(kf['timestamp'])}] scene {kf['scene_start']:.2f}s - {kf['scene_end']:.2f}s")
            return

        # Pass 2: Classification
        click.echo(f"\nClassifying with {model} ({len(keyframes)} frames in batches of {BATCH_SIZE})...")
        classified = classify_keyframes(keyframes, model)

        # Pass 3: Edit log
        clips = generate_edit_log(classified)

        # Output
        if fmt in ("all", "table"):
            print_table(clips)

        if fmt in ("all", "json"):
            write_json(clips, classified, f"{output}.json")

        if fmt in ("all", "csv"):
            write_csv(clips, f"{output}.csv")

        if fmt in ("all", "edl"):
            write_edl(clips, f"{output}.edl")


if __name__ == "__main__":
    main()
