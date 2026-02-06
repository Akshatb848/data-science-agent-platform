"""
Data Science Agent Platform - Main Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import json
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.coordinator_agent import CoordinatorAgent
from agents.data_cleaner_agent import DataCleanerAgent
from agents.eda_agent import EDAAgent
from agents.feature_engineer_agent import FeatureEngineerAgent
from agents.model_trainer_agent import ModelTrainerAgent
from agents.automl_agent import AutoMLAgent
from agents.dashboard_builder_agent import DashboardBuilderAgent
from agents.data_visualizer_agent import DataVisualizerAgent
from utils.helpers import generate_sample_data

# Page config
st.set_page_config(
    page_title="Data Science Agent Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state."""
    if 'coordinator' not in st.session_state:
        st.session_state.coordinator = CoordinatorAgent()
        st.session_state.coordinator.register_agent(DataCleanerAgent())
        st.session_state.coordinator.register_agent(EDAAgent())
        st.session_state.coordinator.register_agent(FeatureEngineerAgent())
        st.session_state.coordinator.register_agent(ModelTrainerAgent())
        st.session_state.coordinator.register_agent(AutoMLAgent())
        st.session_state.coordinator.register_agent(DashboardBuilderAgent())
        st.session_state.coordinator.register_agent(DataVisualizerAgent())
    
    defaults = {
        'current_project': None,
        'current_dataset': None,
        'df': None,
        'eda_report': None,
        'model_results': None,
        'step': 1,
        'target_column': None
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def render_header():
    st.markdown('<h1 class="main-header">🔬 Data Science Agent Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Your AI-Powered Expert Data Scientist</p>', unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown("### 🎯 Workflow Progress")
        progress = (st.session_state.step - 1) / 3
        st.progress(progress)
        
        steps = ["1️⃣ Create Project", "2️⃣ Upload Dataset", "3️⃣ Run Analysis"]
        for i, step in enumerate(steps, 1):
            st.markdown(f"{'✅' if st.session_state.step >= i else '⬜'} {step}")
        
        st.markdown("---")
        st.markdown("### 📊 Status")
        
        if st.session_state.current_project:
            st.success(f"**Project:** {st.session_state.current_project['name']}")
        else:
            st.info("No project selected")
        
        if st.session_state.df is not None:
            st.success(f"**Dataset:** {st.session_state.df.shape[0]:,} × {st.session_state.df.shape[1]}")
        else:
            st.info("No dataset loaded")
        
        st.markdown("---")
        if st.button("🔄 Reset All"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


def render_step1():
    st.markdown("## Step 1: Create or Select Project")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🆕 Create New Project")
        name = st.text_input("Project Name", placeholder="My Data Science Project")
        desc = st.text_area("Description", placeholder="Describe your project...")
        
        if st.button("Create Project", key="create"):
            if name:
                project = st.session_state.coordinator.create_project(name=name, description=desc)
                st.session_state.current_project = project
                st.session_state.step = 2
                st.success(f"✅ Project '{name}' created!")
                st.rerun()
            else:
                st.error("Please enter a project name")
    
    with col2:
        st.markdown("### 📂 Existing Projects")
        projects = st.session_state.coordinator.projects
        if projects:
            options = {f"{p['name']} ({pid})": pid for pid, p in projects.items()}
            selected = st.selectbox("Select project", list(options.keys()))
            if st.button("Load Project"):
                st.session_state.current_project = projects[options[selected]]
                st.session_state.step = 2
                st.rerun()
        else:
            st.info("No existing projects")


def render_step2():
    st.markdown("## Step 2: Upload Your Dataset")
    
    tab1, tab2, tab3 = st.tabs(["📤 Upload File", "🔗 From URL", "📝 Sample Data"])
    
    with tab1:
        uploaded = st.file_uploader("Upload dataset", type=["csv", "xlsx", "xls", "json", "parquet"])
        if uploaded:
            try:
                if uploaded.name.endswith('.csv'):
                    df = pd.read_csv(uploaded)
                elif uploaded.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded)
                elif uploaded.name.endswith('.json'):
                    df = pd.read_json(uploaded)
                elif uploaded.name.endswith('.parquet'):
                    df = pd.read_parquet(uploaded)
                
                st.session_state.df = df
                st.session_state.current_dataset = {"name": uploaded.name, "rows": len(df), "columns": len(df.columns)}
                st.success(f"✅ Loaded: {len(df):,} rows × {len(df.columns)} columns")
                st.dataframe(df.head(10), use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")
    
    with tab2:
        url = st.text_input("Dataset URL", placeholder="https://example.com/data.csv")
        if st.button("Load from URL") and url:
            try:
                df = pd.read_csv(url) if url.endswith('.csv') else pd.read_json(url)
                st.session_state.df = df
                st.success(f"✅ Loaded: {len(df):,} rows × {len(df.columns)} columns")
                st.dataframe(df.head(10), use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")
    
    with tab3:
        samples = {"Iris (Classification)": "iris", "Housing (Regression)": "housing", 
                   "Titanic (Classification)": "titanic", "Random Data": "random"}
        selected = st.selectbox("Choose sample", list(samples.keys()))
        if st.button("Load Sample"):
            df = generate_sample_data(samples[selected])
            st.session_state.df = df
            st.session_state.current_dataset = {"name": selected, "rows": len(df), "columns": len(df.columns)}
            st.success(f"✅ Loaded: {len(df):,} rows × {len(df.columns)} columns")
            st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown("---")
    if st.session_state.df is not None:
        st.markdown("### 🎯 Select Target Column")
        cols = ["None (Unsupervised)"] + list(st.session_state.df.columns)
        target = st.selectbox("Target Column", cols)
        st.session_state.target_column = None if target == "None (Unsupervised)" else target
        
        if st.button("▶️ Proceed to Analysis", type="primary"):
            st.session_state.step = 3
            st.rerun()


def render_step3():
    st.markdown("## Step 3: Data Science Analysis")
    
    df = st.session_state.df
    target = st.session_state.target_column
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 🔧 Configuration")
        run_cleaning = st.checkbox("Data Cleaning", value=True)
        run_eda = st.checkbox("EDA", value=True)
        run_fe = st.checkbox("Feature Engineering", value=True)
        run_modeling = st.checkbox("Model Training", value=target is not None, disabled=target is None)
        run_automl = st.checkbox("AutoML", value=target is not None, disabled=target is None)
    
    with col2:
        st.markdown("### ⚙️ Settings")
        cv_folds = st.slider("CV Folds", 2, 10, 5)
    
    if st.button("🚀 Start Analysis", type="primary"):
        run_analysis(df, target, run_cleaning, run_eda, run_fe, run_modeling, run_automl, cv_folds)


def run_analysis(df, target, cleaning, eda, fe, modeling, automl, cv_folds):
    progress = st.progress(0)
    status = st.empty()
    results = {}
    current_df = df.copy()
    
    # Cleaning
    if cleaning:
        status.text("🧹 Cleaning data...")
        progress.progress(10)
        cleaner = DataCleanerAgent()
        result = asyncio.run(cleaner.run({"action": "clean_data", "dataframe": current_df}))
        if result.success:
            current_df = result.data["dataframe"]
            results["cleaning"] = result.data.get("cleaning_report", {})
            st.success("✅ Data cleaning complete")
    
    progress.progress(25)
    
    # EDA
    if eda:
        status.text("📊 Running EDA...")
        eda_agent = EDAAgent()
        result = asyncio.run(eda_agent.run({"action": "full_eda", "dataframe": current_df, "target_column": target}))
        if result.success:
            st.session_state.eda_report = result.data
            results["eda"] = result.data
            st.success("✅ EDA complete")
    
    progress.progress(40)
    
    # Feature Engineering
    if fe:
        status.text("⚙️ Engineering features...")
        fe_agent = FeatureEngineerAgent()
        result = asyncio.run(fe_agent.run({"action": "engineer_features", "dataframe": current_df, "target_column": target}))
        if result.success:
            current_df = result.data["dataframe"]
            results["feature_engineering"] = result.data.get("feature_report", {})
            st.success("✅ Feature engineering complete")
    
    progress.progress(60)
    
    # Modeling
    if modeling and target:
        status.text("🤖 Training models...")
        trainer = ModelTrainerAgent()
        result = asyncio.run(trainer.run({"action": "train_models", "dataframe": current_df, "target_column": target, "cv_folds": cv_folds}))
        if result.success:
            st.session_state.model_results = result.data
            results["modeling"] = result.data
            st.success(f"✅ Best model: {result.data.get('best_model', 'N/A')}")
    
    progress.progress(80)
    
    # AutoML
    if automl and target:
        status.text("🔮 Running AutoML...")
        automl_agent = AutoMLAgent()
        result = asyncio.run(automl_agent.run({"action": "auto_select_models", "dataframe": current_df, "target_column": target}))
        if result.success:
            results["automl"] = result.data
            st.success("✅ AutoML complete")
    
    progress.progress(100)
    status.text("✅ Analysis complete!")
    
    st.session_state.analysis_results = results
    display_results(results)


def display_results(results):
    st.markdown("---")
    st.markdown("## 📈 Results")
    
    tabs = st.tabs(["📊 EDA", "🤖 Models", "📉 Visualizations"])
    
    with tabs[0]:
        if "eda" in results:
            eda = results["eda"]
            info = eda.get("dataset_info", {})
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{info.get('shape', {}).get('rows', 0):,}")
            c2.metric("Columns", info.get("shape", {}).get("columns", 0))
            c3.metric("Missing", f"{sum(info.get('missing_values', {}).values()):,}")
            c4.metric("Duplicates", info.get("duplicates", 0))
            
            st.markdown("### 💡 Insights")
            for insight in eda.get("insights", [])[:5]:
                st.info(insight)
    
    with tabs[1]:
        if "modeling" in results:
            data = results["modeling"]
            st.success(f"🏆 Best Model: **{data.get('best_model', 'N/A')}**")
            
            metrics = data.get("best_metrics", {})
            if metrics:
                cols = st.columns(len(metrics))
                for i, (k, v) in enumerate(metrics.items()):
                    cols[i].metric(k.upper(), f"{v:.4f}" if isinstance(v, float) else str(v))
            
            st.markdown("### All Models")
            all_results = data.get("results", {})
            if all_results:
                rows = [{"Model": name, **d.get("metrics", {})} for name, d in all_results.items()]
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
    
    with tabs[2]:
        df = st.session_state.df
        if df is not None:
            import plotly.express as px
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            viz = st.selectbox("Visualization", ["Distribution", "Correlation", "Scatter"])
            
            if viz == "Distribution" and numeric_cols:
                col = st.selectbox("Column", numeric_cols)
                fig = px.histogram(df, x=col, marginal="box", title=f"Distribution: {col}")
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz == "Correlation" and len(numeric_cols) > 1:
                corr = df[numeric_cols].corr()
                fig = px.imshow(corr, text_auto=True, title="Correlation Matrix", color_continuous_scale="RdBu_r")
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz == "Scatter" and len(numeric_cols) >= 2:
                c1, c2 = st.columns(2)
                x = c1.selectbox("X-axis", numeric_cols)
                y = c2.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1))
                fig = px.scatter(df, x=x, y=y, title=f"{x} vs {y}")
                st.plotly_chart(fig, use_container_width=True)


def main():
    init_session_state()
    render_header()
    render_sidebar()
    
    if st.session_state.step == 1:
        render_step1()
    elif st.session_state.step == 2:
        render_step2()
    elif st.session_state.step == 3:
        render_step3()


if __name__ == "__main__":
    main()
