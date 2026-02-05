from utils.db_manager import log_agent_start, log_agent_success


class CoordinatorAgent:
    def __init__(self, project_id: str):
        self.project_id = project_id

    def run_agent(self, agent_name: str, agent_callable, *args, **kwargs):
        log_agent_start(self.project_id, agent_name)
        result = agent_callable(*args, **kwargs)
        log_agent_success(self.project_id, agent_name)
        return result

    def run_pipeline(self, df=None):
        print(f"[Coordinator] Running pipeline for project: {self.project_id}")

        # Agent 3 – Warehouse
        from agents.agent3_warehouse import WarehouseAgent
        self.run_agent(
            "warehouse",
            WarehouseAgent(self.project_id).run,
            df
        )

        # Agent 1 – Cleaner
        from agents.agent1_cleaner import DataCleanerAgent
        cleaned_path = self.run_agent(
            "data_cleaner",
            DataCleanerAgent(self.project_id).clean,
            df
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

        print("[Coordinator] Pipeline completed successfully")
