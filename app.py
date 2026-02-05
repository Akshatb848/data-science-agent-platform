import streamlit as st
import pandas as pd

from agents.coordinator import CoordinatorAgent
from agents.agent7_dashboard import DashboardAgent

st.set_page_config(page_title="Multi-Agent Data Science Platform", layout="wide")

st.title("Multi-Agent Data Science Platform")

project_id = st.text_input("Project ID", value="test_project")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("CSV uploaded successfully")
    st.dataframe(df.head())

    if st.button("Run Pipeline"):
        coord = CoordinatorAgent(project_id)
        coord.run_pipeline(df)

        st.success("Pipeline executed successfully")

        dashboard = DashboardAgent(project_id)
        dashboard.render()
