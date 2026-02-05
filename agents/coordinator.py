from utils.db_manager import (
    log_agent_start,
    log_agent_success,
    log_agent_failure,
)

import pandas as pd


class CoordinatorAgent:
    def __init__(self, project_id: str):
        self.project_id = project_id

    # -------------------------------------------------
    # REQUIRED: run_agent (THIS WAS MISSING)
    # -------------------------------------------------
    def run_agent(self, agent_name: str, agent_callable):
        log_agent_start(self.project_id, agent_name)

        try:
            output_location = agent_callable()
            log_agent_success(
                self.project_id,
                agent_name,
                output_location=output_location,
            )
            return output_location

        except Exception as e:
            log_agent_failure(
                self.project_id,
                agent_name,
                error_message=str(e),
            )
            raise

    # -------------------------------------------------
    # PIPELINE
    # -------------------------------------------------
    def run_pipeline(self):
        print(f"[Coordinator] Running pipeline for project: {self.project_id}")

        # Agent 3 – Warehouse
        from agents.agent3_warehouse import WarehouseAgent
        self.run_agent(
            "warehouse",
            WarehouseAgent(self.project_id).run
        )

        # Agent 1 – Data Cleaner (REQUIRES DF)
        from agents.agent1_cleaner import DataCleanerAgent
        sample_df = self._get_sample_df()
        self.run_agent(
            "data_cleaner",
            lambda: DataCleanerAgent(self.project_id).clean(sample_df)
        )

        # Agent 4 – Feature Engineering
        from agents.agent4_features import FeatureEngineeringAgent
        self.run_agent(
            "feature_engineering",
            FeatureEngineeringAgent(self.project_id).run
        )

        # Agent 2 – Visualizer
        from agents.agent2_visualizer import VisualizerAgent
        self.run_agent(
            "visualizer",
            VisualizerAgent(self.project_id).run
        )

        # Agent 5 – AutoML
        from agents.agent5_automl import AutoMLAgent
        self.run_agent(
            "automl_recommendation",
            AutoMLAgent(self.project_id).run
        )

        # Agent 6 – Model Training
        from agents.agent6_model_training import ModelTrainingAgent
        self.run_agent(
            "model_training",
            ModelTrainingAgent(self.project_id).run
        )

        print("[Coordinator] Pipeline completed")

    # -------------------------------------------------
    # INTERNAL SAMPLE DATA
    # -------------------------------------------------
    def _get_sample_df(self):
        return pd.DataFrame({
            "age": [25, None, 40, 40],
            "income": [50000, 60000, None, 70000],
            "city": ["Delhi", "Mumbai", None, "Delhi"]
        })
