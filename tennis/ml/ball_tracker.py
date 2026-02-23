"""
Ball Tracker — Ball detection and Kalman-filtered tracking.
Wraps BallNet model with tracking state management.
"""

from __future__ import annotations
import math
from typing import Optional
from tennis.models.events import BallEvent, BoundingBox, EventType, Point2D, Point3D


class KalmanState:
    """Simple 2D Kalman filter state for ball tracking."""
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.confidence = 0.0
        self.frames_since_detection = 0

    def predict(self, dt: float = 1/30) -> tuple[float, float]:
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += 9.81 * dt  # gravity
        self.frames_since_detection += 1
        self.confidence *= 0.95
        return self.x, self.y

    def update(self, x: float, y: float, alpha: float = 0.7):
        old_x, old_y = self.x, self.y
        self.x = alpha * x + (1 - alpha) * self.x
        self.y = alpha * y + (1 - alpha) * self.y
        self.vx = (self.x - old_x) * 30  # fps
        self.vy = (self.y - old_y) * 30
        self.frames_since_detection = 0
        self.confidence = 1.0

    @property
    def is_tracking(self) -> bool:
        return self.frames_since_detection < 15 and self.confidence > 0.2

    @property
    def speed_pixels_per_frame(self) -> float:
        return math.sqrt(self.vx**2 + self.vy**2)


class BallTracker:
    """Ball detection + tracking with Kalman filtering and bounce detection."""

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.kalman = KalmanState()
        self.trajectory: list[tuple[float, float, int]] = []
        self._prev_vy: float = 0.0
        self._bounce_cooldown: int = 0
        self.frame_count: int = 0

    def process_frame(
        self, detection: Optional[BoundingBox], frame_number: int, session_id: str = ""
    ) -> Optional[BallEvent]:
        """Process a single frame detection. Returns a BallEvent if significant."""
        self.frame_count = frame_number
        if self._bounce_cooldown > 0:
            self._bounce_cooldown -= 1

        if detection and detection.confidence > 0.25:
            cx = (detection.x1 + detection.x2) / 2
            cy = (detection.y1 + detection.y2) / 2
            self.kalman.update(cx, cy)
            self.trajectory.append((cx, cy, frame_number))
            if len(self.trajectory) > 300:
                self.trajectory = self.trajectory[-200:]

            # Check for bounce (vertical velocity sign change)
            is_bounce = self._detect_bounce()
            event_type = EventType.BALL_BOUNCE if is_bounce else EventType.BALL_HIT

            speed = self.kalman.speed_pixels_per_frame * self.fps
            speed_mph = speed * 0.03  # rough pixel-to-mph conversion

            return BallEvent(
                event_type=event_type,
                timestamp_ms=int(frame_number / self.fps * 1000),
                frame_number=frame_number,
                session_id=session_id,
                position_image=detection,
                velocity_mph=speed_mph,
                detection_confidence=detection.confidence,
                tracker_id=0,
            )
        else:
            # No detection — predict
            self.kalman.predict(1 / self.fps)
            if self.kalman.is_tracking:
                return BallEvent(
                    event_type=EventType.BALL_HIT,
                    timestamp_ms=int(frame_number / self.fps * 1000),
                    frame_number=frame_number,
                    session_id=session_id,
                    detection_confidence=self.kalman.confidence,
                    is_interpolated=True,
                    is_occluded=True,
                    tracker_id=0,
                )
            return None

    def _detect_bounce(self) -> bool:
        """Detect ball bounce via vertical velocity sign change."""
        if self._bounce_cooldown > 0:
            return False
        cur_vy = self.kalman.vy
        if self._prev_vy > 0 and cur_vy < 0:
            self._bounce_cooldown = 10
            self._prev_vy = cur_vy
            return True
        self._prev_vy = cur_vy
        return False

    def reset(self):
        self.kalman = KalmanState()
        self.trajectory.clear()
        self._prev_vy = 0.0
        self._bounce_cooldown = 0
        self.frame_count = 0
