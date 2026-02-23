"""
Stats Calculator — Real-time statistical computation from match events.

Computes per-player and per-match statistics including:
- Serve percentages and speeds
- Winner/error ratios
- Rally length distributions
- Shot placement heatmaps
- Break point conversion rates
"""

from __future__ import annotations
from typing import Optional
from tennis.models.match import MatchState, PointOutcome, PointOutcomeType, ShotType
from tennis.models.player import PlayerSessionStats, PlayerComparison
from tennis.models.events import RallyEvent


class StatsCalculator:
    """Computes match and player statistics from point-level data."""

    def compute_player_stats(
        self,
        player_id: str,
        match: MatchState,
        rallies: Optional[list[RallyEvent]] = None,
    ) -> PlayerSessionStats:
        stats = PlayerSessionStats(player_id=player_id, session_id=match.id, match_id=match.id)
        for pt in match.points_timeline:
            self._process_point(stats, pt, player_id)
        self._finalize(stats)
        if rallies:
            self._process_rallies(stats, rallies, player_id)
        return stats

    def compute_match_comparison(self, match: MatchState) -> PlayerComparison:
        s1 = self.compute_player_stats(match.player1_id, match)
        s2 = self.compute_player_stats(match.player2_id, match)
        highlights = self._find_highlight_stats(s1, s2)
        return PlayerComparison(
            player1=s1, player2=s2,
            player1_name=match.player1_name, player2_name=match.player2_name,
            highlight_stats=highlights,
        )

    def _process_point(self, stats: PlayerSessionStats, pt: PointOutcome, pid: str) -> None:
        stats.total_points += 1
        is_winner = pt.winner_id == pid
        is_server = pt.server_id == pid

        if is_winner:
            stats.points_won += 1

        # Serve stats
        if is_server:
            stats.total_serve_points += 1
            if pt.outcome_type == PointOutcomeType.ACE:
                stats.aces += 1
                stats.first_serves_in += 1
                if is_winner:
                    stats.points_won += 0  # already counted
            elif pt.outcome_type == PointOutcomeType.DOUBLE_FAULT:
                stats.double_faults += 1
            else:
                stats.first_serves_in += 1
        else:
            stats.total_return_points += 1
            if is_winner:
                stats.return_points_won += 1

        # Winner/error classification
        if is_winner and pt.outcome_type == PointOutcomeType.WINNER:
            stats.winners += 1
            if pt.last_shot and pt.last_shot.shot_type in (ShotType.FOREHAND, ShotType.SLICE_FH):
                stats.forehand_winners += 1
            elif pt.last_shot and pt.last_shot.shot_type in (ShotType.BACKHAND, ShotType.SLICE_BH):
                stats.backhand_winners += 1
        elif not is_winner and pt.outcome_type == PointOutcomeType.UNFORCED_ERROR:
            stats.unforced_errors += 1
            if pt.last_shot and "forehand" in pt.last_shot.shot_type.value:
                stats.forehand_errors += 1
            elif pt.last_shot and "backhand" in pt.last_shot.shot_type.value:
                stats.backhand_errors += 1
        elif not is_winner and pt.outcome_type == PointOutcomeType.FORCED_ERROR:
            stats.forced_errors += 1

        # Rally length
        stats.avg_rally_length = (
            (stats.avg_rally_length * (stats.total_points - 1) + pt.rally_length)
            / stats.total_points
        )
        stats.longest_rally = max(stats.longest_rally, pt.rally_length)

        # Break points
        if pt.is_break_point:
            if is_server:
                stats.break_points_faced += 1
                if is_winner:
                    stats.break_points_saved += 1
            else:
                stats.break_points_total += 1
                if is_winner:
                    stats.break_points_won += 1

    def _process_rallies(self, stats: PlayerSessionStats, rallies: list[RallyEvent], pid: str) -> None:
        speeds = []
        for r in rallies:
            for shot_d in r.shots:
                if shot_d.get("player_id") == pid and shot_d.get("speed_mph"):
                    speeds.append(shot_d["speed_mph"])
        if speeds:
            stats.max_serve_speed_mph = max(speeds)
            stats.avg_first_serve_speed_mph = sum(speeds) / len(speeds)

    def _finalize(self, stats: PlayerSessionStats) -> None:
        if stats.total_serve_points > 0:
            stats.first_serve_pct = stats.first_serves_in / stats.total_serve_points * 100
        if stats.total_return_points > 0:
            stats.return_win_pct = stats.return_points_won / stats.total_return_points * 100
        if stats.unforced_errors > 0:
            stats.winner_to_ue_ratio = stats.winners / stats.unforced_errors
        if stats.break_points_total > 0:
            stats.break_point_conversion_pct = stats.break_points_won / stats.break_points_total * 100
        if stats.break_points_faced > 0:
            stats.break_point_save_pct = stats.break_points_saved / stats.break_points_faced * 100

    def _find_highlight_stats(self, s1: PlayerSessionStats, s2: PlayerSessionStats) -> list[str]:
        highlights = []
        comparisons = [
            ("aces", s1.aces, s2.aces),
            ("winners", s1.winners, s2.winners),
            ("first_serve_pct", s1.first_serve_pct, s2.first_serve_pct),
            ("unforced_errors", s1.unforced_errors, s2.unforced_errors),
            ("break_point_conversion", s1.break_point_conversion_pct, s2.break_point_conversion_pct),
        ]
        for name, v1, v2 in comparisons:
            if v1 != v2:
                diff = abs(v1 - v2) / max(v1, v2, 1) * 100
                if diff > 20:
                    highlights.append(name)
        return highlights

    def generate_placement_heatmap(
        self, points: list[PointOutcome], player_id: str
    ) -> dict[str, int]:
        """Generate shot placement heatmap data by court zone."""
        heatmap: dict[str, int] = {}
        for pt in points:
            for shot in pt.shot_sequence:
                if shot.player_id == player_id and shot.placement_zone:
                    zone = shot.placement_zone.value
                    heatmap[zone] = heatmap.get(zone, 0) + 1
        return heatmap

    def generate_speed_distribution(
        self, points: list[PointOutcome], player_id: str
    ) -> list[float]:
        """Get distribution of shot speeds for a player."""
        speeds = []
        for pt in points:
            for shot in pt.shot_sequence:
                if shot.player_id == player_id and shot.speed_mph:
                    speeds.append(shot.speed_mph)
        return sorted(speeds)
