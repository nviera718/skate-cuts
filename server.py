"""
skate-cuts HTTP API — thin wrapper around scene detection functions.

Runs as a sidecar container for the content-farm-api.
"""

import traceback
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from skate_cuts import detect_scenes, filter_by_duration, score_motion, filter_by_motion

app = FastAPI(title="skate-cuts", version="0.2.0")


class AnalyzeRequest(BaseModel):
    video_path: str
    threshold: float = Field(default=3.0, description="Scene detection sensitivity")
    min_duration: float = Field(default=1.5, description="Min clip duration in seconds")
    max_duration: float = Field(default=30.0, description="Max clip duration in seconds")
    filter_motion_enabled: bool = Field(default=False, description="Score and filter by motion")
    motion_threshold: float = Field(default=2.0, description="Min motion score to keep")
    min_scene_len: int = Field(default=15, description="Min scene length in frames")


class Clip(BaseModel):
    clip: int
    start: float
    end: float
    duration: float
    motion: float | None = None


class AnalyzeResponse(BaseModel):
    clips: list[Clip]
    total: int
    video_path: str


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    video = Path(req.video_path)
    if not video.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {req.video_path}")

    try:
        scenes, raw_scene_list = detect_scenes(
            str(video), req.threshold, req.min_scene_len
        )

        if not scenes:
            return AnalyzeResponse(clips=[], total=0, video_path=req.video_path)

        scenes = filter_by_duration(scenes, req.min_duration, req.max_duration)

        if req.filter_motion_enabled and scenes:
            scenes = score_motion(str(video), scenes)
            scenes = filter_by_motion(scenes, req.motion_threshold)

        clips = [Clip(**s) for s in scenes]
        return AnalyzeResponse(clips=clips, total=len(clips), video_path=req.video_path)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
