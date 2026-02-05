import streamlit as st
import pandas as pd

from agents.coordinator import CoordinatorAgent
from agents.agent7_dashboard import DashboardAgent

st.set_page_config(page_title="Data Science Agent Platform", layout="wide")
st.title("Multi-Agent Data Science Platform")

project_id = st.text_input("Project ID", value="test_project")

uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])

if uploaded_file:
    st.session_state.dataset = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded")

if st.button("Run Pipeline"):
    if "dataset" not in st.session_state:
        st.error("Please upload a CSV first")
    else:
        coord = CoordinatorAgent(project_id)
        coord.dataset = st.session_state.dataset
        coord.run_pipeline()
        st.success("Pipeline executed successfully")

if st.button("Load Dashboard"):
    dashboard = DashboardAgent(project_id)
    data = dashboard.run()

    if not isinstance(data, dict):
        st.error("Dashboard data invalid. Please re-run pipeline.")
    else:
        st.header("Project Metadata")
        st.dataframe(data["project"])

        st.header("Agent Execution Timeline")
        st.dataframe(data["agents"])

        st.header("Feature Store")
        st.dataframe(data["features"])

        st.header("Model Results")
        st.dataframe(data["models"])
