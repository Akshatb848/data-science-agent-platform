import streamlit as st
from agents.coordinator import CoordinatorAgent
from agents.agent7_dashboard import DashboardAgent

st.set_page_config(page_title="Multi-Agent Data Science Platform", layout="wide")
st.title("Multi-Agent Data Science Platform")

project_id = st.text_input("Project ID", value="test_project")

if st.button("Run Pipeline"):
    CoordinatorAgent(project_id).run_pipeline()
    st.success("Pipeline executed successfully")
    st.session_state.pop("dashboard", None)

if st.button("Load Dashboard"):
    dashboard = DashboardAgent(project_id).run()
    st.session_state["dashboard"] = dashboard

if "dashboard" in st.session_state:
    data = st.session_state["dashboard"]

    st.header("Project Metadata")
    st.dataframe(data["project"], use_container_width=True)

    st.header("Agent Execution Timeline")
    st.dataframe(data["agents"], use_container_width=True)

    st.header("Feature Store")
    st.dataframe(data["features"], use_container_width=True)

    st.header("Model Results")
    st.dataframe(data["models"], use_container_width=True)
