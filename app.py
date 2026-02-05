import streamlit as st
from agents.coordinator import CoordinatorAgent
from agents.agent7_dashboard import DashboardAgent

st.set_page_config(page_title="Data Science Agent Platform", layout="wide")
st.title("Multi-Agent Data Science Platform")

project_id = st.text_input("Project ID", value="test_project")

if st.button("Run Pipeline"):
    CoordinatorAgent(project_id).run_pipeline()
    st.success("Pipeline completed")

if st.button("Load Dashboard"):
    data = DashboardAgent(project_id).run()

    st.subheader("Project")
    st.dataframe(data["project"])

    st.subheader("Agents")
    st.dataframe(data["agents"])

    st.subheader("Features")
    st.dataframe(data["features"])

    st.subheader("Models")
    st.dataframe(data["models"])
