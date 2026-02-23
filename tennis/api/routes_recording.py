"""
Recording routes — Guided match setup and recording control.
"""

from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, HTTPException

from tennis.engine.recording import (
    RecordingSession, MatchSetupConfig, MatchType, Environment, Handedness,
)

router = APIRouter()

# Active recording sessions
_recordings: dict[str, RecordingSession] = {}


@router.post("/setup")
async def setup_recording(
    match_type: str = "singles",
    environment: str = "outdoor",
    player_names: Optional[list[str]] = None,
    court_surface: str = "hard",
):
    """
    Step 1: Configure match before recording.
    All fields are optional — defaults produce a ready-to-record session.
    """
    config = MatchSetupConfig(
        match_type=MatchType(match_type),
        environment=Environment(environment),
        player_names=player_names or ["Player 1", "Player 2"],
        court_surface=court_surface,
    )
    if match_type == "doubles":
        config.player_count = 4
        if not player_names:
            config.player_names = ["Player 1", "Player 2", "Player 3", "Player 4"]
        config.player_handedness = [Handedness.AUTO] * 4

    session = RecordingSession()
    result = session.setup(config)

    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("errors", []))

    _recordings[session.id] = session
    return result


@router.post("/{session_id}/start")
async def start_recording(session_id: str):
    """Step 2: Begin recording. Processing starts automatically."""
    session = _recordings.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Recording session not found")

    result = session.start_recording()
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result


@router.post("/{session_id}/stop")
async def stop_recording(session_id: str):
    """Step 3: Stop recording. Segmentation finalized automatically."""
    session = _recordings.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Recording session not found")

    result = session.stop_recording()
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result


@router.get("/{session_id}/status")
async def recording_status(session_id: str):
    """Check recording status and progress."""
    session = _recordings.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Recording session not found")

    return {
        "session_id": session.id,
        "state": session.state.value,
        "frames_processed": session.frame_count,
        "points_detected": len(session.points),
        "line_calls": len(session.line_calls),
        "duration_seconds": session._get_duration_seconds(),
    }


@router.get("/{session_id}/summary")
async def recording_summary(session_id: str):
    """Get complete session summary after recording stops."""
    session = _recordings.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Recording session not found")
    return session.get_summary()
