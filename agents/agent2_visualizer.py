import pandas as pd
import plotly.express as px
import os
from utils.db_manager import get_engine

class VisualizerAgent:
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.artifacts_path = f"artifacts/{self.project_id}/eda/"
        os.makedirs(self.artifacts_path, exist_ok=True)

    def run(self) -> str:
        df = self._load_df()  # From cleaned_data
        # EDA: Stats, plots, insights
        stats = df.describe()
        corr = df.corr()
        insights = f"Key insights: {len(df)} rows, correlations: {corr.to_string()}"

        # Plots (interactive Plotly)
        fig_hist = px.histogram(df, title="Distributions")
        fig_hist.write_html(self.artifacts_path + "hist.html")
        fig_corr = px.imshow(corr, title="Correlation Heatmap")
        fig_corr.write_html(self.artifacts_path + "corr.html")

        # Save insights to DB
        engine = get_engine()
        pd.DataFrame({"insights": [insights]}).to_sql("eda_insights", engine, if_exists="append")

        return self.artifacts_path
