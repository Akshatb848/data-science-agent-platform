import sqlalchemy as sa
import pandas as pd
import json
from utils.db_manager import get_engine


class WarehouseAgent:
    def __init__(self, project_id: str):
        self.project_id = project_id

    def run(self):
        engine = get_engine()

        # 1️⃣ Load cleaned data
        df = pd.read_sql(
            "SELECT data FROM cleaned_data WHERE project_id=%s ORDER BY id DESC LIMIT 1",
            engine,
            params=(self.project_id,)
        )

        data = pd.DataFrame(df.iloc[0]["data"])

        # 2️⃣ Store warehouse metadata (not duplicating full data yet)
        with engine.begin() as conn:
            conn.execute(
                sa.text("""
                    INSERT INTO data_warehouse
                    (project_id, table_name, row_count, columns)
                    VALUES (:pid, :table, :rows, :cols)
                """),
                {
                    "pid": self.project_id,
                    "table": "cleaned_dataset_v1",
                    "rows": len(data),
                    "cols": json.dumps(list(data.columns))
                }
            )

        return "warehouse/cleaned_dataset_v1"
