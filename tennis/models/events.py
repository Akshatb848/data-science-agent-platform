"""
Event data models — Ball events, player events, rallies, and line calls.
These represent raw CV detections converted into tennis-meaningful events.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────────

class EventType(str, Enum):
    BALL_BOUNCE = "ball_bounce"
    BALL_HIT = "ball_hit"
    BALL_OUT = "ball_out"
    BALL_NET = "ball_net"
    BALL_LET = "ball_let"
    RALLY_START = "rally_start"
    RALLY_END = "rally_end"
    POINT_START = "point_start"
    POINT_END = "point_end"
    SERVE = "serve"
    FAULT = "fault"
    PLAYER_POSITION = "player_position"
    POSE_SNAPSHOT = "pose_snapshot"
    CHALLENGE_REQUESTED = "challenge_requested"
    CHALLENGE_RESOLVED = "challenge_resolved"


class LineCallVerdict(str, Enum):
    IN = "in"
    OUT = "out"
    LET = "let"
    NET = "net"
    UNKNOWN = "unknown"


class ChallengeStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    OVERTURNED = "overturned"
    EXPIRED = "expired"


class BounceConfidence(str, Enum):
    HIGH = "high"         # >90% confidence
    MEDIUM = "medium"     # 70-90%
    LOW = "low"           # 50-70%
    UNCERTAIN = "uncertain"  # <50%


# ── Position & Geometry ──────────────────────────────────────────────────────

class Point2D(BaseModel):
    """2D point in court coordinates (meters from court center)."""
    x: float
    y: float


class Point3D(BaseModel):
    """3D point for ball position (meters)."""
    x: float
    y: float
    z: float = 0.0


class BoundingBox(BaseModel):
    """Bounding box in image coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 0.0


class PoseKeypoint(BaseModel):
    """Single pose keypoint."""
    name: str
    x: float
    y: float
    confidence: float = 0.0


class PlayerPose(BaseModel):
    """Full body pose with 17 keypoints (COCO format)."""
    keypoints: list[PoseKeypoint] = Field(default_factory=list)
    overall_confidence: float = 0.0

    # Named accessors for key joints
    @property
    def left_shoulder(self) -> Optional[PoseKeypoint]:
        return next((k for k in self.keypoints if k.name == "left_shoulder"), None)

    @property
    def right_shoulder(self) -> Optional[PoseKeypoint]:
        return next((k for k in self.keypoints if k.name == "right_shoulder"), None)

    @property
    def left_elbow(self) -> Optional[PoseKeypoint]:
        return next((k for k in self.keypoints if k.name == "left_elbow"), None)

    @property
    def right_elbow(self) -> Optional[PoseKeypoint]:
        return next((k for k in self.keypoints if k.name == "right_elbow"), None)

    @property
    def left_wrist(self) -> Optional[PoseKeypoint]:
        return next((k for k in self.keypoints if k.name == "left_wrist"), None)

    @property
    def right_wrist(self) -> Optional[PoseKeypoint]:
        return next((k for k in self.keypoints if k.name == "right_wrist"), None)

    @property
    def left_hip(self) -> Optional[PoseKeypoint]:
        return next((k for k in self.keypoints if k.name == "left_hip"), None)

    @property
    def right_hip(self) -> Optional[PoseKeypoint]:
        return next((k for k in self.keypoints if k.name == "right_hip"), None)


# ── Event Models ─────────────────────────────────────────────────────────────

class BallEvent(BaseModel):
    """A ball detection/tracking event from CV pipeline."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType
    timestamp_ms: int
    frame_number: int
    session_id: str

    # ── Ball state ───────────────────────────────────────
    position_image: Optional[BoundingBox] = None
    position_court: Optional[Point2D] = None
    position_3d: Optional[Point3D] = None
    velocity_mph: Optional[float] = None
    velocity_kph: Optional[float] = None
    spin_proxy_rpm: Optional[float] = None
    trajectory_angle_deg: Optional[float] = None

    # ── Detection quality ────────────────────────────────
    detection_confidence: float = 0.0
    is_occluded: bool = False
    is_interpolated: bool = False
    tracker_id: Optional[int] = None

    # ── Line call (for bounces) ──────────────────────────
    line_call: Optional[LineCallVerdict] = None
    line_call_confidence: float = 0.0
    distance_from_line_cm: Optional[float] = None
    uncertainty_radius_cm: Optional[float] = None


class PlayerEvent(BaseModel):
    """A player detection/pose event from CV pipeline."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.PLAYER_POSITION
    timestamp_ms: int
    frame_number: int
    session_id: str
    player_id: str

    # ── Player state ─────────────────────────────────────
    position_image: Optional[BoundingBox] = None
    position_court: Optional[Point2D] = None
    pose: Optional[PlayerPose] = None
    court_zone: Optional[str] = None
    velocity_mps: Optional[float] = None
    facing_direction_deg: Optional[float] = None

    # ── Detection quality ────────────────────────────────
    detection_confidence: float = 0.0
    tracker_id: Optional[int] = None
    is_serving: bool = False


class RallyEvent(BaseModel):
    """A complete rally from serve to point conclusion."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    match_id: Optional[str] = None
    point_number: int

    # ── Timeline ─────────────────────────────────────────
    start_frame: int
    end_frame: int
    start_timestamp_ms: int
    end_timestamp_ms: int
    duration_seconds: float = 0.0

    # ── Rally content ────────────────────────────────────
    server_id: str
    winner_id: Optional[str] = None
    rally_length: int = 0
    shots: list[dict] = Field(default_factory=list, description="Ordered shot events")
    ball_events: list[str] = Field(
        default_factory=list, description="Ball event IDs in this rally"
    )

    # ── Outcome ──────────────────────────────────────────
    outcome_type: Optional[str] = None
    last_shot_type: Optional[str] = None
    last_shot_player_id: Optional[str] = None

    # ── Analytics ────────────────────────────────────────
    max_shot_speed_mph: float = 0.0
    avg_shot_speed_mph: float = 0.0
    excitement_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="ML-scored excitement level for highlight ranking"
    )


class LineCallEvent(BaseModel):
    """A line call decision, potentially subject to challenge."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    match_id: Optional[str] = None
    rally_id: str
    ball_event_id: str

    # ── Decision ─────────────────────────────────────────
    verdict: LineCallVerdict
    confidence: float = 0.0
    bounce_position_court: Point2D
    closest_line: str = ""
    distance_from_line_cm: float = 0.0
    uncertainty_radius_cm: float = 0.0

    # ── Challenge ────────────────────────────────────────
    is_challenged: bool = False
    challenge_status: ChallengeStatus = ChallengeStatus.PENDING
    challenged_by_player_id: Optional[str] = None
    challenge_timestamp_ms: Optional[int] = None
    replay_frame_start: Optional[int] = None
    replay_frame_end: Optional[int] = None
    original_verdict: Optional[LineCallVerdict] = None
    final_verdict: Optional[LineCallVerdict] = None

    timestamp_ms: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class EventBatch(BaseModel):
    """Batch of events for bulk ingestion from device."""
    session_id: str
    device_id: str
    batch_number: int = 0
    ball_events: list[BallEvent] = Field(default_factory=list)
    player_events: list[PlayerEvent] = Field(default_factory=list)
    timestamp_start_ms: int = 0
    timestamp_end_ms: int = 0
    frame_count: int = 0
