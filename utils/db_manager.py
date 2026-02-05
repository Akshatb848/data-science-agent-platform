import os
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.engine import Engine

_ENGINE: Engine | None = None


def get_engine() -> Engine:
    """
    Creates a singleton SQLAlchemy engine for Neon (Postgres).
    Requires DATABASE_URL to be set with sslmode=require.
    """
    global _ENGINE

    if _ENGINE is not None:
        return _ENGINE

    db_url = os.getenv("DATABASE_URL")

    if not db_url:
        raise RuntimeError(
            "DATABASE_URL is not set. "
            "Add it in Streamlit Secrets or environment variables."
        )

    # Enforce psycopg3 + SSL (required by Neon)
    if not db_url.startswith("postgresql+psycopg://"):
        db_url = db_url.replace(
            "postgresql://",
            "postgresql+psycopg://",
        )

    if "sslmode=" not in db_url:
        sep = "&" if "?" in db_url else "?"
        db_url = f"{db_url}{sep}sslmode=require"

    _ENGINE = create_engine(
        db_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        future=True,
    )

    return _ENGINE


def ensure_project_exists(project_id: str, description: str = ""):
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS project_metadata (
                project_id TEXT PRIMARY KEY,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(
            text("""
                INSERT INTO project_metadata (project_id, description)
                VALUES (:pid, :desc)
                ON CONFLICT (project_id) DO NOTHING
            """),
            {"pid": project_id, "desc": description},
        )


def log_agent_start(project_id: str, agent_name: str):
    engine = get_engine()
    ensure_project_exists(project_id)

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS agent_status (
                id SERIAL PRIMARY KEY,
                project_id TEXT,
                agent_name TEXT,
                status TEXT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                error_message TEXT
            )
        """))

        conn.execute(
            text("""
                INSERT INTO agent_status (project_id, agent_name, status, started_at)
                VALUES (:pid, :agent, 'running', CURRENT_TIMESTAMP)
            """),
            {"pid": project_id, "agent": agent_name},
        )


def log_agent_success(project_id: str, agent_name: str):
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text("""
                UPDATE agent_status
                SET status='completed', completed_at=CURRENT_TIMESTAMP
                WHERE project_id=:pid AND agent_name=:agent AND status='running'
            """),
            {"pid": project_id, "agent": agent_name},
        )


def log_agent_failure(project_id: str, agent_name: str, error: str):
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text("""
                UPDATE agent_status
                SET status='failed', completed_at=CURRENT_TIMESTAMP, error_message=:err
                WHERE project_id=:pid AND agent_name=:agent AND status='running'
            """),
            {"pid": project_id, "agent": agent_name, "err": error},
        )


def load_table(table: str, project_id: str) -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql(
        text(f"SELECT * FROM {table} WHERE project_id=:pid"),
        engine,
        params={"pid": project_id},
    )
