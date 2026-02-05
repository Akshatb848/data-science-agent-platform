import os
import pandas as pd
import matplotlib.pyplot as plt
import sqlalchemy as sa
from utils.db_manager import get_engine


class VisualizerAgent:
    """
    Generates EDA plots from feature store.
    Handles numeric and dict-based features safely.
    """

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.engine = get_engine()
        self.output_dir = f"artifacts/{project_id}/eda"
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        df = pd.read_sql(
            "SELECT features FROM feature_store WHERE project_id=%s",
            self.engine,
            params=(self.project_id,)
        )

        if df.empty:
            raise ValueError("No features found for visualization")

        features = df.iloc[0]["features"]

        saved_plots = []

        for name, value in features.items():

            # -----------------------
            # Numeric feature → Histogram
            # -----------------------
            if isinstance(value, (int, float)):
                plt.figure()
                plt.hist([value])
                plt.title(name)

                path = f"{self.output_dir}/{name}_hist.png"
                plt.savefig(path)
                plt.close()

                saved_plots.append((name, "histogram", path))

            # -----------------------
            # Dict feature → Bar chart
            # -----------------------
            elif isinstance(value, dict):
                plt.figure()
                plt.bar(value.keys(), value.values())
                plt.title(name)
                plt.xticks(rotation=45)

                path = f"{self.output_dir}/{name}_bar.png"
                plt.tight_layout()
                plt.savefig(path)
                plt.close()

                saved_plots.append((name, "bar", path))

            # -----------------------
            # Unsupported → Skip
            # -----------------------
            else:
                continue

        # Persist metadata
        with self.engine.begin() as conn:
            for feature, plot_type, path in saved_plots:
                conn.execute(
                    sa.text("""
                        INSERT INTO visualizations
                        (project_id, plot_type, plot_path)
                        VALUES (:pid, :ptype, :path)
                    """),
                    {
                        "pid": self.project_id,
                        "ptype": f"{plot_type}_{feature}",
                        "path": path
                    }
                )

        return self.output_dir
