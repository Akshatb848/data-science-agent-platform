import pandas as pd
from sqlalchemy import text
from utils.db_manager import get_engine


class ModelTrainingAgent:
    def __init__(self, project_id: str):
        self.project_id = project_id

    def run(self):
        """
        Stub model training agent.
        This keeps the pipeline stable without AutoML.
        """

        engine = get_engine()

        # Ensure table exists
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS model_results (
                    id SERIAL PRIMARY KEY,
                    project_id TEXT,
                    model_name TEXT,
                    metric_name TEXT,
                    metric_value FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))

            # Insert dummy baseline result
            conn.execute(
                text("""
                    INSERT INTO model_results (
                        project_id, model_name, metric_name, metric_value
                    )
                    VALUES (
                        :pid, 'baseline_dummy_model', 'accuracy', 0.0
                    )
                """),
                {"pid": self.project_id},
            )

        print("[ModelTrainingAgent] Dummy model result stored")
