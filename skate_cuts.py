#!/usr/bin/env python3
"""
skate-cuts: Split skateboarding compilation videos into individual clips.

Uses scene detection to find hard cuts, filters by duration and motion,
and optionally splits the video into separate files (lossless).
"""

import csv
import io
import json
import sys
from pathlib import Path

import click
import cv2
from tqdm import tqdm
from scenedetect import open_video, SceneManager
from scenedetect.detectors import AdaptiveDetector
from scenedetect.video_splitter import split_video_ffmpeg


# =============================================================================
# SCENE DETECTION
# =============================================================================

def detect_scenes(video_path, threshold=3.0, min_scene_len=15):
    """Detect scene boundaries using AdaptiveDetector."""
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        AdaptiveDetector(adaptive_threshold=threshold, min_scene_len=min_scene_len)
    )

    click.echo(f"Detecting scenes (adaptive threshold={threshold})...")
    scene_manager.detect_scenes(video, show_progress=True)
    scene_list = scene_manager.get_scene_list()

    if not scene_list:
        return [], []

    scenes = []
    for i, (start, end) in enumerate(scene_list):
        scenes.append({
            "clip": i + 1,
            "start": round(start.get_seconds(), 3),
            "end": round(end.get_seconds(), 3),
            "duration": round(end.get_seconds() - start.get_seconds(), 3),
        })

    return scenes, scene_list


# =============================================================================
# FILTERING
# =============================================================================

def filter_by_duration(scenes, min_dur=1.5, max_dur=30.0):
    """Drop scenes outside the duration range."""
    filtered = [s for s in scenes if min_dur <= s["duration"] <= max_dur]
    for i, s in enumerate(filtered):
        s["clip"] = i + 1
    return filtered


def score_motion(video_path, scenes):
    """Score each scene by optical flow magnitude (higher = more motion)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    for scene in tqdm(scenes, desc="Scoring motion"):
        start_frame = int(scene["start"] * fps)
        end_frame = int(scene["end"] * fps)
        total_frames = max(end_frame - start_frame, 2)

        # Sample 6 evenly-spaced frames, compute flow between consecutive pairs
        sample_points = [start_frame + int(total_frames * p) for p in [0.1, 0.3, 0.5, 0.7, 0.9]]
        sample_points = [min(p, end_frame - 1) for p in sample_points]

        magnitudes = []
        prev_gray = None

        for frame_num in sample_points:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue

            # Downscale to 320px wide for speed
            h, w = frame.shape[:2]
            if w > 320:
                scale = 320 / w
                frame = cv2.resize(frame, (320, int(h * scale)))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                )
                mag = cv2.magnitude(flow[..., 0], flow[..., 1])
                magnitudes.append(mag.mean())

            prev_gray = gray

        scene["motion"] = round(sum(magnitudes) / len(magnitudes), 2) if magnitudes else 0.0

    cap.release()
    return scenes


def filter_by_motion(scenes, motion_threshold=2.0):
    """Drop scenes below the motion threshold."""
    filtered = [s for s in scenes if s.get("motion", 0) >= motion_threshold]
    for i, s in enumerate(filtered):
        s["clip"] = i + 1
    return filtered


# =============================================================================
# VIDEO INFO
# =============================================================================

def get_video_duration(video_path):
    """Get video duration in seconds."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0:
        return frame_count / fps
    return 0


def format_timestamp(seconds):
    """Format seconds as HH:MM:SS.mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# =============================================================================
# OUTPUT
# =============================================================================

def print_table(scenes, show_motion=False):
    """Pretty-print the clip list."""
    if not scenes:
        click.echo("No clips found.")
        return

    click.echo()
    if show_motion:
        header = f"{'CLIP':>4}  {'IN':>12}  {'OUT':>12}  {'DUR':>6}  {'MOTION':>6}"
    else:
        header = f"{'CLIP':>4}  {'IN':>12}  {'OUT':>12}  {'DUR':>6}"
    click.echo(header)
    click.echo("-" * len(header))

    for s in scenes:
        line = (
            f"{s['clip']:>4}  "
            f"{format_timestamp(s['start']):>12}  "
            f"{format_timestamp(s['end']):>12}  "
            f"{s['duration']:>5.1f}s"
        )
        if show_motion:
            line += f"  {s.get('motion', 0):>6.1f}"
        click.echo(line)

    click.echo()
    total = sum(s["duration"] for s in scenes)
    click.echo(f"{len(scenes)} clip(s) | Total duration: {total:.1f}s")


def write_json(scenes):
    """Write scenes as JSON to stdout."""
    # Strip non-serializable fields
    output = [{k: v for k, v in s.items()} for s in scenes]
    click.echo(json.dumps(output, indent=2))


def write_csv_output(scenes):
    """Write scenes as CSV to stdout."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    has_motion = any("motion" in s for s in scenes)
    header = ["Clip", "In", "Out", "Duration"]
    if has_motion:
        header.append("Motion")
    writer.writerow(header)
    for s in scenes:
        row = [s["clip"], format_timestamp(s["start"]), format_timestamp(s["end"]), s["duration"]]
        if has_motion:
            row.append(s.get("motion", ""))
        writer.writerow(row)
    click.echo(buf.getvalue().rstrip())


# =============================================================================
# VIDEO SPLITTING
# =============================================================================

def split_clips(video_path, scenes, raw_scene_list, output_dir):
    """Split video into individual clip files using ffmpeg (lossless copy)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Map filtered scenes back to raw scene_list indices by matching timestamps
    keep_pairs = []
    for s in scenes:
        for raw_start, raw_end in raw_scene_list:
            if abs(raw_start.get_seconds() - s["start"]) < 0.01:
                keep_pairs.append((raw_start, raw_end))
                break

    if not keep_pairs:
        click.echo("No scenes to split.")
        return

    video_name = Path(video_path).stem
    click.echo(f"\nSplitting {len(keep_pairs)} clip(s) to {output_dir}...")

    split_video_ffmpeg(
        input_video_path=video_path,
        scene_list=keep_pairs,
        output_dir=str(out),
        output_file_template=f"{video_name}-clip-$SCENE_NUMBER.mp4",
        arg_override="-map 0 -c copy",
        show_progress=True,
    )

    click.echo(f"Done. Files in {output_dir}")


# =============================================================================
# CLI
# =============================================================================

@click.command()
@click.argument("video", type=click.Path(exists=True))
@click.option("--output-dir", "-o", default="./clips/", help="Directory for split clips")
@click.option("--threshold", "-t", default=3.0, type=float, help="Scene detection sensitivity (default: 3.0)")
@click.option("--min-duration", default=1.5, type=float, help="Min clip duration in seconds (default: 1.5)")
@click.option("--max-duration", default=30.0, type=float, help="Max clip duration in seconds (default: 30)")
@click.option("--split", is_flag=True, help="Split video into individual clip files (lossless)")
@click.option("--filter-motion", is_flag=True, help="Score and filter clips by motion")
@click.option("--motion-threshold", default=2.0, type=float, help="Min motion score to keep (default: 2.0)")
@click.option("--format", "fmt", type=click.Choice(["table", "json", "csv"]), default="table", help="Output format")
@click.option("--min-scene-len", default=15, type=int, help="Min scene length in frames (default: 15)")
def main(video, output_dir, threshold, min_duration, max_duration, split, filter_motion, motion_threshold, fmt, min_scene_len):
    """Split skateboarding compilation videos into individual clips.

    Detects hard cuts using scene detection, filters by duration and
    optionally by motion, then outputs timestamps or splits the video.
    """
    video_path = str(Path(video).resolve())
    click.echo(f"skate-cuts: {Path(video).name}")

    duration = get_video_duration(video_path)
    if duration > 0:
        click.echo(f"Duration: {format_timestamp(duration)}")

    # Detect scenes
    scenes, raw_scene_list = detect_scenes(video_path, threshold, min_scene_len)
    if not scenes:
        click.echo("No scene cuts detected.")
        return

    click.echo(f"Found {len(scenes)} scene(s).")

    # Filter by duration
    before = len(scenes)
    scenes = filter_by_duration(scenes, min_duration, max_duration)
    dropped = before - len(scenes)
    if dropped:
        click.echo(f"Filtered {dropped} scene(s) by duration ({min_duration}-{max_duration}s). Keeping {len(scenes)}.")

    if not scenes:
        click.echo("No clips remaining after duration filter.")
        return

    # Optional: motion scoring + filtering
    if filter_motion:
        scenes = score_motion(video_path, scenes)
        before = len(scenes)
        scenes = filter_by_motion(scenes, motion_threshold)
        dropped = before - len(scenes)
        if dropped:
            click.echo(f"Filtered {dropped} scene(s) by motion (<{motion_threshold}). Keeping {len(scenes)}.")

    if not scenes:
        click.echo("No clips remaining after motion filter.")
        return

    # Output
    if fmt == "table":
        print_table(scenes, show_motion=filter_motion)
    elif fmt == "json":
        write_json(scenes)
    elif fmt == "csv":
        write_csv_output(scenes)

    # Split if requested
    if split:
        split_clips(video_path, scenes, raw_scene_list, output_dir)


if __name__ == "__main__":
    main()
