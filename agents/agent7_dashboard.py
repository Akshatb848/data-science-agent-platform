import pandas as pd
from utils.db_manager import get_engine


class DashboardAgent:
    """
    Fetches all project data needed for visualization.
    """

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.engine = get_engine()

    def get_data(self):
        """Used by Streamlit UI"""
        return {
            "project": pd.read_sql(
                "SELECT * FROM project_metadata WHERE project_id=%s",
                self.engine,
                params=(self.project_id,),
            ),
            "agents": pd.read_sql(
                "SELECT * FROM agent_status WHERE project_id=%s ORDER BY started_at",
                self.engine,
                params=(self.project_id,),
            ),
            "features": pd.read_sql(
                "SELECT * FROM feature_store WHERE project_id=%s",
                self.engine,
                params=(self.project_id,),
            ),
            "models": pd.read_sql(
                "SELECT * FROM model_results WHERE project_id=%s",
                self.engine,
                params=(self.project_id,),
            ),
        }

    def run(self):
        """Used only for pipeline logging"""
        return "dashboard/rendered"
