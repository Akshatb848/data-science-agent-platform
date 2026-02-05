import pandas as pd
from utils.db_manager import get_engine


class DashboardAgent:
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.engine = get_engine()

    def run(self):
        data = {}

        data["project"] = pd.read_sql(
            "SELECT * FROM project_metadata WHERE project_id = %(pid)s",
            self.engine,
            params={"pid": self.project_id},
        )

        data["agents"] = pd.read_sql(
            """
            SELECT *
            FROM agent_status
            WHERE project_id = %(pid)s
            ORDER BY started_at
            """,
            self.engine,
            params={"pid": self.project_id},
        )

        data["features"] = pd.read_sql(
            "SELECT * FROM feature_store WHERE project_id = %(pid)s",
            self.engine,
            params={"pid": self.project_id},
        )

        data["models"] = pd.read_sql(
            "SELECT * FROM model_results WHERE project_id = %(pid)s",
            self.engine,
            params={"pid": self.project_id},
        )

        return data
