"""
Inference Pipeline — Orchestrated per-frame ML inference.
Coordinates BallNet, PlayerNet, CourtNet, and ShotNet in production.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from tennis.ml.ball_tracker import BallTracker
from tennis.ml.player_detector import PlayerDetector
from tennis.ml.court_detector import CourtDetector
from tennis.models.events import BallEvent, BoundingBox, PlayerEvent


@dataclass
class FrameResult:
    """Result of processing a single frame."""
    frame_number: int
    ball_event: Optional[BallEvent] = None
    player_events: list[PlayerEvent] = field(default_factory=list)
    court_calibrated: bool = False
    latency_ms: float = 0.0


class InferencePipeline:
    """
    Orchestrated per-frame ML inference pipeline.
    
    In production (iOS), each model runs on CoreML/ANE.
    This Python version provides the same interface for cloud processing.
    """

    def __init__(self, session_id: str = "", fps: float = 30.0):
        self.session_id = session_id
        self.fps = fps
        self.ball_tracker = BallTracker(fps=fps)
        self.player_detector = PlayerDetector(max_players=4)
        self.court_detector = CourtDetector()
        self.frame_count = 0
        self.is_initialized = False

    def initialize(self):
        """Load models and prepare for inference."""
        self.is_initialized = True

    def process_frame(
        self,
        ball_detection: Optional[BoundingBox] = None,
        player_detections: Optional[list[BoundingBox]] = None,
        player_keypoints: Optional[list[list[tuple[float, float, float]]]] = None,
        court_keypoints: Optional[list[tuple[float, float, float]]] = None,
    ) -> FrameResult:
        """
        Process a single frame through all models.
        
        In production, raw frame pixels are passed and models run inference.
        Here we accept pre-computed detections for the pipeline orchestration.
        """
        self.frame_count += 1
        result = FrameResult(frame_number=self.frame_count)

        # 1. Court detection (runs every N frames for recalibration)
        if court_keypoints:
            self.court_detector.process_frame(court_keypoints, self.frame_count)
        result.court_calibrated = self.court_detector.is_calibrated

        # 2. Ball tracking
        if ball_detection is not None:
            ball_event = self.ball_tracker.process_frame(
                ball_detection, self.frame_count, self.session_id
            )
            if ball_event and self.court_detector.is_calibrated and ball_event.position_image:
                cx = (ball_event.position_image.x1 + ball_event.position_image.x2) / 2
                cy = (ball_event.position_image.y1 + ball_event.position_image.y2) / 2
                ball_event.position_court = self.court_detector.image_to_court(cx, cy)
            result.ball_event = ball_event

        # 3. Player detection + pose
        if player_detections:
            player_events = self.player_detector.process_frame(
                player_detections,
                keypoints_list=player_keypoints,
                frame_number=self.frame_count,
                session_id=self.session_id,
                fps=self.fps,
            )
            for pe in player_events:
                if self.court_detector.is_calibrated and pe.position_image:
                    cx = (pe.position_image.x1 + pe.position_image.x2) / 2
                    cy = pe.position_image.y2  # feet position
                    pe.position_court = self.court_detector.image_to_court(cx, cy)
            result.player_events = player_events

        return result

    def reset(self):
        self.ball_tracker.reset()
        self.player_detector.reset()
        self.court_detector.reset()
        self.frame_count = 0

    def get_stats(self) -> dict:
        return {
            "frames_processed": self.frame_count,
            "active_players": self.player_detector.get_active_players(),
            "court_calibrated": self.court_detector.is_calibrated,
            "ball_tracking": self.ball_tracker.kalman.is_tracking,
        }
