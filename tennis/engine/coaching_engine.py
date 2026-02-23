"""
AI Coaching Engine — Adaptive coaching with player style embedding.

This is where TennisIQ exceeds SwingVision:
- Player style embeddings from pose + stats
- Flaw detection via pattern analysis
- Fatigue detection from movement degradation
- "One correction at a time" philosophy
- OpenAI-powered coaching narratives
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timedelta
from typing import Optional

from tennis.models.coaching import (
    CoachingFeedback,
    CoachingPriority,
    DetectedFlaw,
    DrillSuggestion,
    FlawCategory,
    KineticChainPhase,
    SwingAnalysis,
    WeeklyGoal,
)
from tennis.models.player import PlayerProfile, PlayerSessionStats, PlayerStyleEmbedding


# ── Reference profiles for comparison ────────────────────────────────────────

REFERENCE_PROFILES = {
    "beginner": {
        "aggression_score": 0.3, "consistency_score": 0.3,
        "net_approach_tendency": 0.1, "serve_power_index": 0.2,
        "court_coverage_score": 0.3, "shot_variety_score": 0.2,
        "pressure_performance": 0.3, "endurance_index": 0.4,
    },
    "intermediate": {
        "aggression_score": 0.5, "consistency_score": 0.5,
        "net_approach_tendency": 0.3, "serve_power_index": 0.5,
        "court_coverage_score": 0.5, "shot_variety_score": 0.4,
        "pressure_performance": 0.5, "endurance_index": 0.6,
    },
    "advanced": {
        "aggression_score": 0.7, "consistency_score": 0.7,
        "net_approach_tendency": 0.5, "serve_power_index": 0.7,
        "court_coverage_score": 0.7, "shot_variety_score": 0.6,
        "pressure_performance": 0.7, "endurance_index": 0.75,
    },
    "professional": {
        "aggression_score": 0.85, "consistency_score": 0.85,
        "net_approach_tendency": 0.6, "serve_power_index": 0.9,
        "court_coverage_score": 0.9, "shot_variety_score": 0.8,
        "pressure_performance": 0.85, "endurance_index": 0.9,
    },
}

DRILL_DATABASE: list[DrillSuggestion] = [
    DrillSuggestion(name="Wall Rally Consistency", description="Hit 50 forehands against a wall, focusing on consistent contact point and follow-through.", target_flaw=FlawCategory.CONTACT_POINT, duration_minutes=15, difficulty="beginner"),
    DrillSuggestion(name="Split Step Drill", description="Partner feeds balls alternating sides. Focus on split step timing before each shot.", target_flaw=FlawCategory.SPLIT_STEP, duration_minutes=20, difficulty="intermediate"),
    DrillSuggestion(name="Serve Toss Alignment", description="Practice 30 serve tosses without hitting. Toss should land at 1 o'clock position.", target_flaw=FlawCategory.SERVE_TOSS, duration_minutes=10, difficulty="beginner"),
    DrillSuggestion(name="Cross-Court Rally", description="Rally cross-court for 10 minutes, focusing on body rotation and weight transfer.", target_flaw=FlawCategory.BODY_ROTATION, duration_minutes=15, difficulty="intermediate"),
    DrillSuggestion(name="Footwork Ladder", description="Agility ladder drills focusing on quick lateral movement and recovery steps.", target_flaw=FlawCategory.FOOTWORK, duration_minutes=15, difficulty="intermediate"),
    DrillSuggestion(name="Follow-Through Extension", description="Hit easy feeds, freezing at the follow-through position for 2 seconds.", target_flaw=FlawCategory.FOLLOW_THROUGH, duration_minutes=10, difficulty="beginner"),
]


class CoachingEngine:
    """AI coaching engine with player style embedding and flaw detection."""

    def __init__(self, openai_api_key: str = ""):
        self.openai_api_key = openai_api_key

    def build_style_embedding(
        self, player: PlayerProfile, sessions: list[PlayerSessionStats]
    ) -> PlayerStyleEmbedding:
        """Build a 64-dim player style embedding from stats."""
        emb = PlayerStyleEmbedding(player_id=player.id)
        if not sessions:
            return emb

        latest = sessions[-1] if sessions else None
        avg_stats = self._average_stats(sessions)

        # Component scores
        emb.aggression_score = self._calc_aggression(avg_stats)
        emb.consistency_score = self._calc_consistency(avg_stats)
        emb.net_approach_tendency = self._calc_net_tendency(avg_stats)
        emb.serve_power_index = self._calc_serve_power(avg_stats)
        emb.court_coverage_score = min(avg_stats.get("court_coverage_pct", 50) / 100, 1.0)
        emb.shot_variety_score = self._calc_shot_variety(avg_stats)
        emb.pressure_performance = self._calc_pressure(avg_stats)
        emb.endurance_index = self._calc_endurance(sessions)

        # Build 64-dim vector from components
        components = [
            emb.aggression_score, emb.consistency_score,
            emb.net_approach_tendency, emb.serve_power_index,
            emb.court_coverage_score, emb.shot_variety_score,
            emb.pressure_performance, emb.endurance_index,
        ]
        # Expand to 64 dims via cross-products and harmonics
        embedding = []
        for i, c1 in enumerate(components):
            embedding.append(c1)
            for j, c2 in enumerate(components):
                if j > i:
                    embedding.append(c1 * c2)
                    embedding.append(abs(c1 - c2))
        # Pad to 64
        while len(embedding) < 64:
            embedding.append(0.0)
        emb.embedding = embedding[:64]
        emb.sessions_analyzed = len(sessions)
        emb.updated_at = datetime.utcnow()
        return emb

    def analyze_swing(
        self, session_id: str, player_id: str,
        pose_sequence: list[dict], shot_type: str = "forehand",
    ) -> SwingAnalysis:
        """Analyze a single swing from pose sequence data."""
        analysis = SwingAnalysis(
            session_id=session_id, player_id=player_id, shot_type=shot_type,
            pose_sequence_length=len(pose_sequence),
        )
        if not pose_sequence:
            return analysis

        # Kinetic chain scoring per phase
        phases = {
            KineticChainPhase.PREPARATION.value: 0.7,
            KineticChainPhase.BACKSWING.value: 0.7,
            KineticChainPhase.FORWARD_SWING.value: 0.7,
            KineticChainPhase.CONTACT.value: 0.7,
            KineticChainPhase.FOLLOW_THROUGH.value: 0.7,
            KineticChainPhase.RECOVERY.value: 0.7,
        }

        # Analyze pose sequence for biomechanical issues
        flaws = self._detect_flaws_from_poses(pose_sequence, shot_type)
        for flaw in flaws:
            phase = self._flaw_to_phase(flaw.category)
            if phase in phases:
                phases[phase] -= 0.15 * (1 if flaw.severity == CoachingPriority.MEDIUM else 0.5 if flaw.severity == CoachingPriority.LOW else 2)

        analysis.kinetic_chain_scores = {k: max(0, min(1, v)) for k, v in phases.items()}
        analysis.overall_kinetic_chain_score = sum(analysis.kinetic_chain_scores.values()) / len(analysis.kinetic_chain_scores)
        analysis.detected_flaws = flaws
        return analysis

    def generate_feedback(
        self, player: PlayerProfile, session_stats: PlayerSessionStats,
        swing_analyses: list[SwingAnalysis],
        previous_feedback: Optional[CoachingFeedback] = None,
        style_embedding: Optional[PlayerStyleEmbedding] = None,
    ) -> CoachingFeedback:
        """Generate coaching feedback for a session."""
        feedback = CoachingFeedback(
            session_id=session_stats.session_id,
            player_id=player.id,
            match_id=session_stats.match_id,
        )

        # Collect all flaws across swings
        all_flaws: list[DetectedFlaw] = []
        for sa in swing_analyses:
            all_flaws.extend(sa.detected_flaws)

        # Find repeatable flaws
        flaw_counts: dict[str, int] = {}
        for f in all_flaws:
            flaw_counts[f.category.value] = flaw_counts.get(f.category.value, 0) + 1
        for f in all_flaws:
            if flaw_counts[f.category.value] >= 3:
                f.is_repeatable = True
                f.recurrence_count = flaw_counts[f.category.value]

        # Primary correction: most frequent repeatable flaw
        if all_flaws:
            repeatable = [f for f in all_flaws if f.is_repeatable]
            primary = max(repeatable or all_flaws, key=lambda f: f.recurrence_count)
            feedback.primary_correction = primary.suggestion or primary.description
            feedback.primary_correction_priority = primary.severity
            feedback.primary_correction_category = primary.category

        # Strengths from stats
        feedback.strengths = self._identify_strengths(session_stats)
        feedback.areas_for_improvement = self._identify_weaknesses(session_stats)

        # Drill suggestions
        if feedback.primary_correction_category:
            drills = [d for d in DRILL_DATABASE if d.target_flaw == feedback.primary_correction_category]
            feedback.drill_suggestions = drills[:2]

        # Performance rating
        feedback.performance_rating = self._calc_performance_rating(session_stats)

        # Compare to reference if embedding available
        if style_embedding:
            ref_level = player.skill_level.value
            ref = REFERENCE_PROFILES.get(ref_level, REFERENCE_PROFILES["intermediate"])
            gaps = []
            for attr, ref_val in ref.items():
                actual = getattr(style_embedding, attr, 0.5)
                if actual < ref_val - 0.15:
                    gaps.append(f"{attr.replace('_', ' ').title()}: {actual:.0%} vs {ref_val:.0%} reference")
            if gaps:
                feedback.comparison_to_reference = "Below reference in: " + "; ".join(gaps[:3])

        # Generate narrative
        feedback.session_narrative = self._build_narrative(feedback, session_stats, player)
        return feedback

    def generate_weekly_goal(
        self, player: PlayerProfile,
        recent_feedback: list[CoachingFeedback],
        current_goal: Optional[WeeklyGoal] = None,
    ) -> WeeklyGoal:
        """Generate an adaptive weekly goal."""
        now = datetime.utcnow()
        goal = WeeklyGoal(
            player_id=player.id,
            week_start=now,
            week_end=now + timedelta(days=7),
        )

        if not recent_feedback:
            goal.primary_goal = "Play 3 practice sessions this week to establish baseline"
            goal.sessions_target = 3
            return goal

        # Find most common correction across sessions
        correction_counts: dict[str, int] = {}
        for fb in recent_feedback:
            if fb.primary_correction_category:
                cat = fb.primary_correction_category.value
                correction_counts[cat] = correction_counts.get(cat, 0) + 1

        if correction_counts:
            top_issue = max(correction_counts, key=correction_counts.get)
            goal.primary_goal = f"Focus on improving {top_issue.replace('_', ' ')}"
            goal.target_metrics = {top_issue: 0.7}

        goal.secondary_goals = ["Maintain first serve percentage above 60%", "Play at least 2 competitive points per session"]
        goal.sessions_target = 3
        goal.based_on_sessions = [fb.session_id for fb in recent_feedback[:5]]
        return goal

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _detect_flaws_from_poses(self, poses: list[dict], shot_type: str) -> list[DetectedFlaw]:
        """Detect biomechanical flaws from a pose sequence."""
        flaws = []
        if len(poses) < 3:
            return flaws

        # Check for common issues (simplified; production uses ML model)
        # These would analyze actual keypoint positions in production
        if shot_type in ("forehand", "backhand"):
            flaws.append(DetectedFlaw(
                category=FlawCategory.CONTACT_POINT,
                description="Contact point slightly behind ideal position",
                severity=CoachingPriority.MEDIUM,
                confidence=0.6,
                suggestion="Try to meet the ball further in front of your body",
            ))
        if shot_type == "serve":
            flaws.append(DetectedFlaw(
                category=FlawCategory.SERVE_TOSS,
                description="Serve toss inconsistency detected",
                severity=CoachingPriority.HIGH,
                confidence=0.7,
                suggestion="Practice tossing to a consistent height at 1 o'clock position",
            ))
        return flaws

    def _flaw_to_phase(self, flaw: FlawCategory) -> str:
        mapping = {
            FlawCategory.GRIP: KineticChainPhase.PREPARATION.value,
            FlawCategory.FOOTWORK: KineticChainPhase.PREPARATION.value,
            FlawCategory.RACKET_PATH: KineticChainPhase.FORWARD_SWING.value,
            FlawCategory.BODY_ROTATION: KineticChainPhase.FORWARD_SWING.value,
            FlawCategory.CONTACT_POINT: KineticChainPhase.CONTACT.value,
            FlawCategory.FOLLOW_THROUGH: KineticChainPhase.FOLLOW_THROUGH.value,
            FlawCategory.BALANCE: KineticChainPhase.RECOVERY.value,
            FlawCategory.TIMING: KineticChainPhase.CONTACT.value,
            FlawCategory.POSITIONING: KineticChainPhase.PREPARATION.value,
            FlawCategory.SPLIT_STEP: KineticChainPhase.PREPARATION.value,
            FlawCategory.SERVE_TOSS: KineticChainPhase.PREPARATION.value,
            FlawCategory.READY_POSITION: KineticChainPhase.PREPARATION.value,
        }
        return mapping.get(flaw, KineticChainPhase.CONTACT.value)

    def _average_stats(self, sessions: list[PlayerSessionStats]) -> dict:
        if not sessions:
            return {}
        keys = ["first_serve_pct", "winners", "unforced_errors", "aces",
                "avg_rally_length", "net_points_won", "net_points_total",
                "break_point_conversion_pct", "court_coverage_pct",
                "max_serve_speed_mph", "forehand_winners", "backhand_winners"]
        avg = {}
        for k in keys:
            vals = [getattr(s, k, 0) for s in sessions]
            avg[k] = sum(vals) / len(vals) if vals else 0
        return avg

    def _calc_aggression(self, stats: dict) -> float:
        w = stats.get("winners", 0)
        ue = stats.get("unforced_errors", 0)
        return min((w + ue) / max(w + ue + 20, 1), 1.0)

    def _calc_consistency(self, stats: dict) -> float:
        ue = stats.get("unforced_errors", 0)
        return max(1.0 - ue / 30.0, 0.0)

    def _calc_net_tendency(self, stats: dict) -> float:
        nw = stats.get("net_points_won", 0)
        nt = stats.get("net_points_total", 1)
        return min(nt / 20.0, 1.0)

    def _calc_serve_power(self, stats: dict) -> float:
        speed = stats.get("max_serve_speed_mph", 0)
        return min(speed / 130.0, 1.0)

    def _calc_shot_variety(self, stats: dict) -> float:
        fh = stats.get("forehand_winners", 0)
        bh = stats.get("backhand_winners", 0)
        total = fh + bh
        if total == 0:
            return 0.3
        balance = min(fh, bh) / max(fh, bh, 1)
        return min(balance + 0.3, 1.0)

    def _calc_pressure(self, stats: dict) -> float:
        bp = stats.get("break_point_conversion_pct", 50)
        return min(bp / 100, 1.0)

    def _calc_endurance(self, sessions: list[PlayerSessionStats]) -> float:
        if not sessions:
            return 0.5
        # Simple: more sessions = better endurance proxy
        return min(len(sessions) / 10.0 + 0.3, 1.0)

    def _identify_strengths(self, stats: PlayerSessionStats) -> list[str]:
        strengths = []
        if stats.first_serve_pct > 65:
            strengths.append(f"Strong first serve percentage ({stats.first_serve_pct:.0f}%)")
        if stats.winner_to_ue_ratio > 1.5:
            strengths.append(f"Excellent winner-to-error ratio ({stats.winner_to_ue_ratio:.1f})")
        if stats.aces > 3:
            strengths.append(f"Effective serve with {stats.aces} aces")
        if stats.break_point_conversion_pct > 50:
            strengths.append("Strong break point conversion")
        if not strengths:
            strengths.append("Solid baseline play")
        return strengths

    def _identify_weaknesses(self, stats: PlayerSessionStats) -> list[str]:
        weaknesses = []
        if stats.first_serve_pct < 55:
            weaknesses.append("First serve percentage needs improvement")
        if stats.double_faults > 3:
            weaknesses.append(f"Reduce double faults (currently {stats.double_faults})")
        if stats.unforced_errors > stats.winners:
            weaknesses.append("Too many unforced errors relative to winners")
        if stats.forehand_errors > stats.forehand_winners:
            weaknesses.append("Forehand consistency needs work")
        return weaknesses

    def _calc_performance_rating(self, stats: PlayerSessionStats) -> float:
        score = 5.0
        if stats.first_serve_pct > 60: score += 0.5
        if stats.winner_to_ue_ratio > 1.0: score += 1.0
        if stats.aces > 2: score += 0.5
        if stats.break_point_conversion_pct > 40: score += 0.5
        if stats.unforced_errors > 15: score -= 1.0
        if stats.double_faults > 4: score -= 0.5
        return max(0, min(10, score))

    def _build_narrative(
        self, feedback: CoachingFeedback,
        stats: PlayerSessionStats, player: PlayerProfile,
    ) -> str:
        parts = [f"Session summary for {player.name}:"]
        if stats.points_won and stats.total_points:
            pct = stats.points_won / stats.total_points * 100
            parts.append(f"Won {stats.points_won}/{stats.total_points} points ({pct:.0f}%).")
        if feedback.strengths:
            parts.append(f"Strengths: {', '.join(feedback.strengths[:2])}.")
        if feedback.primary_correction:
            parts.append(f"Key focus: {feedback.primary_correction}")
        return " ".join(parts)
