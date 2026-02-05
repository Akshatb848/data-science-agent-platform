from utils.db_manager import log_agent_start, log_agent_success, log_agent_failure
from utils.llm_client import query_llm
from utils.rag_retriever import query_rag  # Assuming rag_retriever.py exists
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

    def run_pipeline(self, user_task: str = ""):
        print(f"[Coordinator] Running dynamic pipeline for project: {self.project_id}")

        # Load data to infer nature
        df = self._load_data()
        data_summary = df.describe().to_string() + f"\nColumns: {df.dtypes.to_string()}"

        # Use RAG + LLM for dynamic plan (e.g., propose ML/DL/RL)
        rag_docs = query_rag(f"Best models for data like: {data_summary}")
        prompt = f"""
        Act as expert data scientist. Data summary: {data_summary}
        User task: {user_task}
        RAG knowledge: {rag_docs}
        Available agents: Warehouse, DataCleaner, FeatureEngineering, Visualizer (for EDA), AutoML (propose ML/DL/RL), ModelTraining (implement best), Dashboard.
        Output JSON: {{"plan": ["agent1", "agent2", ...], "model_proposal": "Rationale for ML/DL/RL choice"}}
        For proposals: If tabular/supervised → ML (XGBoost); Time-series → DL (LSTM); Sequential decisions → RL (Gym); etc.
        """
        response = query_llm(prompt)
        plan_data = json.loads(response)
        plan = plan_data["plan"]
        print(f"[LLM Plan]: {plan}\n[Proposal]: {plan_data['model_proposal']}")

        agent_map = {
            "Warehouse": lambda: self._run_warehouse(),
            "DataCleaner": lambda: self._run_cleaner(df),
            "FeatureEngineering": lambda: self._run_features(),
            "Visualizer": lambda: self._run_visualizer(),  # EDA
            "AutoML": lambda: self._run_automl(plan_data['model_proposal']),
            "ModelTraining": lambda: self._run_model_training(),
            "Dashboard": lambda: self._run_dashboard()
        }

        for agent in plan:
            if agent in agent_map:
                self.run_agent(agent.lower(), agent_map[agent])

        print("[Coordinator] Pipeline completed")
        return plan_data['model_proposal']  # For dashboard display

    # Helper runners (map to actual agents)
    def _run_warehouse(self):
        from agents.agent3_warehouse import WarehouseAgent
        return WarehouseAgent(self.project_id).run()

    def _run_cleaner(self, df):
        from agents.agent1_cleaner import DataCleanerAgent
        return DataCleanerAgent(self.project_id).clean(df)

    def _run_features(self):
        from agents.agent4_features import FeatureEngineeringAgent
        return FeatureEngineeringAgent(self.project_id).run()

    def _run_visualizer(self):
        from agents.agent2_visualizer import VisualizerAgent
        return VisualizerAgent(self.project_id).run()  # Generates EDA plots/insights

    def _run_automl(self, proposal):
        from agents.agent5_automl import AutoMLAgent
        return AutoMLAgent(self.project_id, proposal).run()  # Updated to take proposal

    def _run_model_training(self):
        from agents.agent6_model_training import ModelTrainingAgent
        return ModelTrainingAgent(self.project_id).run()  # Trains, selects best, saves metrics/SHAP

    def _run_dashboard(self):
        from agents.agent7_dashboard import DashboardAgent
        return DashboardAgent(self.project_id).run()

    def _load_data(self):
        # Load from DB or sample
        try:
            from utils.db_manager import query_table
            return query_table('cleaned_data', self.project_id)  # Assume exists
        except:
            return pd.DataFrame({"age": [25, 30], "income": [50000, 60000]})  # Fallback
