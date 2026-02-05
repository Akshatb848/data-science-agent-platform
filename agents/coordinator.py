from utils.db_manager import log_agent_start, log_agent_success, log_agent_failure
#from utils.llm_client import query_llm
#from utils.rag_retriever import query_rag  # Assuming rag_retriever.py exists
import pandas as pd
import json

class CoordinatorAgent:
    def __init__(self, project_id: str):
        self.project_id = project_id

    def run_agent(self, agent_name: str, agent_callable):
        log_agent_start(self.project_id, agent_name)
        try:
            output = agent_callable()
            log_agent_success(self.project_id, agent_name, output_location=output)
            return output
        except Exception as e:
            log_agent_failure(self.project_id, agent_name, error_message=str(e))
            raise

    def run_pipeline(self):
        print(f"[Coordinator] Running pipeline for project: {self.project_id}")

        from agents.agent3_warehouse import WarehouseAgent
        self.run_agent("warehouse", WarehouseAgent(self.project_id).run)

        from agents.agent1_cleaner import DataCleanerAgent
        self.run_agent(
            "data_cleaner",
            DataCleanerAgent(self.project_id).clean,
            self.dataset,  # IMPORTANT: dataset must be passed
        )

        from agents.agent4_features import FeatureEngineeringAgent
        self.run_agent("feature_engineering", FeatureEngineeringAgent(self.project_id).run)

        from agents.agent2_visualizer import VisualizerAgent
        self.run_agent("visualizer", VisualizerAgent(self.project_id).run)

        from agents.agent5_automl import AutoMLAgent
        self.run_agent("automl_recommendation", AutoMLAgent(self.project_id).run)

        from agents.agent6_model_training import ModelTrainingAgent
        self.run_agent("model_training", ModelTrainingAgent(self.project_id).run)

        print("[Coordinator] Pipeline completed")
