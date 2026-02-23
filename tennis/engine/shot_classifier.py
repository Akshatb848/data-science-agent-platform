"""
Shot Classifier — Classifies tennis shot types from ball + pose data.
"""

from __future__ import annotations
from typing import Optional
from tennis.models.events import PlayerPose, Point2D
from tennis.models.match import ShotType


class ShotClassifier:
    """Classifies shot types from trajectory and pose analysis."""

    VOLLEY_MAX_DIST = 4.0
    LOB_MIN_CLEARANCE = 200.0
    DROP_MAX_SPEED = 30.0

    def classify(
        self,
        ball_position: Optional[Point2D],
        ball_velocity_mph: float,
        ball_trajectory_angle: float,
        player_pose: Optional[PlayerPose],
        player_position: Optional[Point2D],
        shot_number_in_rally: int,
        is_serve: bool = False,
        net_clearance_cm: Optional[float] = None,
    ) -> tuple[ShotType, float]:
        if is_serve:
            return ShotType.SERVE, 0.95
        if shot_number_in_rally <= 2:
            side = self._side(player_pose, player_position, ball_position)
            return (ShotType.RETURN_FH if side == "fh" else ShotType.RETURN_BH), 0.80
        if net_clearance_cm and net_clearance_cm > self.LOB_MIN_CLEARANCE:
            return ShotType.LOB, 0.70
        if ball_velocity_mph < self.DROP_MAX_SPEED and ball_trajectory_angle > 30:
            return ShotType.DROP_SHOT, 0.65
        if player_position and abs(player_position.y) < self.VOLLEY_MAX_DIST:
            side = self._side(player_pose, player_position, ball_position)
            return (ShotType.VOLLEY_FH if side == "fh" else ShotType.VOLLEY_BH), 0.75
        side = self._side(player_pose, player_position, ball_position)
        if ball_trajectory_angle < -10 and ball_velocity_mph < 60:
            return (ShotType.SLICE_FH if side == "fh" else ShotType.SLICE_BH), 0.70
        return (ShotType.FOREHAND if side == "fh" else ShotType.BACKHAND), 0.80

    def _side(self, pose, player_pos, ball_pos) -> str:
        if pose and pose.right_wrist and pose.right_shoulder:
            return "fh" if pose.right_wrist.x > pose.right_shoulder.x else "bh"
        if player_pos and ball_pos:
            return "fh" if ball_pos.x > player_pos.x else "bh"
        return "fh"
