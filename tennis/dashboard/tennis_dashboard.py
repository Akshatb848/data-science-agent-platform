"""
TennisIQ Analytics Dashboard — Streamlit-powered match intelligence.

Features:
- Interactive scoreboard with match timeline
- Shot placement heatmaps (Plotly)
- Rally & speed distributions
- Player comparison radar charts
- AI coaching insights panel
- Session history
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from datetime import datetime

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TennisIQ — Match Intelligence",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0a0a0a; }
    .stat-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 16px; padding: 20px; margin: 8px 0;
        border: 1px solid #2a2a4a;
    }
    .stat-value { font-size: 2.2rem; font-weight: 700; color: #00ff88; }
    .stat-label { font-size: 0.85rem; color: #8888aa; text-transform: uppercase; }
    .score-display {
        background: linear-gradient(135deg, #0d1117, #161b22);
        border-radius: 20px; padding: 24px; text-align: center;
        border: 2px solid #30363d;
    }
    .player-name { font-size: 1.1rem; color: #c9d1d9; font-weight: 600; }
    .score-num { font-size: 3rem; font-weight: 800; color: #00ff88; }
    .section-header {
        font-size: 1.3rem; font-weight: 700; color: #e6edf3;
        border-bottom: 2px solid #00ff88; padding-bottom: 8px; margin: 20px 0 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Sample Data Generator ────────────────────────────────────────────────────

def generate_sample_data():
    """Generate realistic sample match data."""
    np.random.seed(42)
    return {
        "match": {
            "player1": "Alex Chen", "player2": "Jordan Smith",
            "score": "6-4, 3-6, 7-5", "surface": "Hard",
            "duration": "2h 15m", "winner": "Alex Chen",
            "sets": [(6, 4), (3, 6), (7, 5)],
        },
        "p1_stats": {
            "aces": 8, "double_faults": 3, "first_serve_pct": 67,
            "second_serve_pct": 88, "winners": 32, "unforced_errors": 22,
            "forced_errors": 12, "net_points": "12/18",
            "break_points_won": "4/7", "total_points_won": 96,
            "avg_first_serve_speed": 118, "avg_second_serve_speed": 89,
            "max_serve_speed": 131, "forehand_winners": 18,
            "backhand_winners": 10, "avg_rally_length": 4.8,
        },
        "p2_stats": {
            "aces": 5, "double_faults": 5, "first_serve_pct": 61,
            "second_serve_pct": 82, "winners": 28, "unforced_errors": 30,
            "forced_errors": 15, "net_points": "8/14",
            "break_points_won": "3/6", "total_points_won": 89,
            "avg_first_serve_speed": 112, "avg_second_serve_speed": 84,
            "max_serve_speed": 125, "forehand_winners": 15,
            "backhand_winners": 9, "avg_rally_length": 5.1,
        },
        "shot_placement": {
            "zones": ["DW", "DB", "DT", "AW", "AB", "AT", "BL", "BC", "BR"],
            "p1_counts": [12, 8, 15, 10, 6, 14, 22, 18, 20],
            "p2_counts": [10, 9, 12, 11, 7, 13, 19, 16, 18],
        },
        "rally_lengths": np.random.exponential(4, 200).astype(int).clip(1, 30).tolist(),
        "serve_speeds_p1": np.random.normal(115, 12, 80).clip(85, 140).tolist(),
        "serve_speeds_p2": np.random.normal(110, 10, 75).clip(80, 135).tolist(),
        "coaching": {
            "primary_correction": "Move contact point 6 inches further in front of your body on forehand groundstrokes",
            "strengths": [
                "Strong first serve percentage (67%)",
                "Excellent winner-to-error ratio (1.45)",
                "Effective net play (67% success)",
            ],
            "improvements": [
                "Reduce double faults — consider a safer second serve toss",
                "Improve backhand consistency under pressure",
                "Work on split step timing on returns",
            ],
            "drills": [
                "Cross-court forehand rally (15 min, focus on contact point)",
                "Serve toss consistency drill (10 min)",
                "Split step ladder drill (15 min)",
            ],
            "performance_rating": 7.2,
            "weekly_goal": "Focus on meeting the ball further in front on groundstrokes",
        },
    }

data = generate_sample_data()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🎾 TennisIQ")
    st.markdown("### AI Match Intelligence")
    st.divider()
    
    view = st.radio("Dashboard View", [
        "📊 Match Overview",
        "🎯 Shot Analysis",
        "🧠 AI Coaching",
        "📈 Trends",
    ])
    
    st.divider()
    st.markdown("**Current Match**")
    st.markdown(f"🏆 {data['match']['player1']} vs {data['match']['player2']}")
    st.markdown(f"📍 {data['match']['surface']} Court")
    st.markdown(f"⏱️ {data['match']['duration']}")

# ── Match Overview ───────────────────────────────────────────────────────────
if "Match Overview" in view:
    st.markdown("## 📊 Match Overview")
    
    # Scoreboard
    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        st.markdown(f"""
        <div class="score-display">
            <div class="player-name">{data['match']['player1']} 🏆</div>
            <div class="score-num">{data['match']['sets'][0][0]} {data['match']['sets'][1][0]} {data['match']['sets'][2][0]}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="text-align:center; padding-top:30px;">
            <div style="color:#555; font-size:0.9rem;">FINAL</div>
            <div style="color:#00ff88; font-size:1.5rem; font-weight:800;">VS</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="score-display">
            <div class="player-name">{data['match']['player2']}</div>
            <div class="score-num">{data['match']['sets'][0][1]} {data['match']['sets'][1][1]} {data['match']['sets'][2][1]}</div>
        </div>
        """, unsafe_allow_html=True)

    # Key Stats Row
    st.markdown('<div class="section-header">Key Statistics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    stats_display = [
        ("Aces", data['p1_stats']['aces'], data['p2_stats']['aces']),
        ("Winners", data['p1_stats']['winners'], data['p2_stats']['winners']),
        ("UE", data['p1_stats']['unforced_errors'], data['p2_stats']['unforced_errors']),
        ("1st Serve %", f"{data['p1_stats']['first_serve_pct']}%", f"{data['p2_stats']['first_serve_pct']}%"),
        ("Max Speed", f"{data['p1_stats']['max_serve_speed']}", f"{data['p2_stats']['max_serve_speed']}"),
        ("BP Won", data['p1_stats']['break_points_won'], data['p2_stats']['break_points_won']),
    ]
    for col, (label, v1, v2) in zip([c1, c2, c3, c4, c5, c6], stats_display):
        with col:
            st.metric(label, v1, f"vs {v2}")

    # Radar Chart — Player Comparison
    st.markdown('<div class="section-header">Player Comparison</div>', unsafe_allow_html=True)
    categories = ['Serve Power', 'Accuracy', 'Net Play', 'Consistency', 'Aggression', 'Defense']
    p1_vals = [0.8, 0.75, 0.7, 0.65, 0.8, 0.6]
    p2_vals = [0.7, 0.65, 0.55, 0.7, 0.6, 0.75]
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=p1_vals + [p1_vals[0]], theta=categories + [categories[0]], fill='toself', name=data['match']['player1'], line_color='#00ff88', fillcolor='rgba(0,255,136,0.15)'))
    fig_radar.add_trace(go.Scatterpolar(r=p2_vals + [p2_vals[0]], theta=categories + [categories[0]], fill='toself', name=data['match']['player2'], line_color='#ff6b6b', fillcolor='rgba(255,107,107,0.15)'))
    fig_radar.update_layout(polar=dict(bgcolor='#0d1117', radialaxis=dict(visible=True, range=[0, 1], gridcolor='#21262d'), angularaxis=dict(gridcolor='#21262d')), paper_bgcolor='#0a0a0a', font_color='#c9d1d9', showlegend=True, height=400, margin=dict(t=30, b=30))
    st.plotly_chart(fig_radar, use_container_width=True)

    # Rally Length Distribution
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-header">Rally Length Distribution</div>', unsafe_allow_html=True)
        fig_rally = px.histogram(x=data['rally_lengths'], nbins=20, color_discrete_sequence=['#00ff88'])
        fig_rally.update_layout(paper_bgcolor='#0a0a0a', plot_bgcolor='#0d1117', font_color='#c9d1d9', xaxis_title='Rally Length (shots)', yaxis_title='Frequency', height=300, margin=dict(t=10, b=40))
        st.plotly_chart(fig_rally, use_container_width=True)
    
    with col_b:
        st.markdown('<div class="section-header">Serve Speed Distribution</div>', unsafe_allow_html=True)
        fig_speed = go.Figure()
        fig_speed.add_trace(go.Histogram(x=data['serve_speeds_p1'], name=data['match']['player1'], marker_color='#00ff88', opacity=0.7, nbinsx=20))
        fig_speed.add_trace(go.Histogram(x=data['serve_speeds_p2'], name=data['match']['player2'], marker_color='#ff6b6b', opacity=0.7, nbinsx=20))
        fig_speed.update_layout(barmode='overlay', paper_bgcolor='#0a0a0a', plot_bgcolor='#0d1117', font_color='#c9d1d9', xaxis_title='Speed (mph)', yaxis_title='Count', height=300, margin=dict(t=10, b=40))
        st.plotly_chart(fig_speed, use_container_width=True)

# ── Shot Analysis ────────────────────────────────────────────────────────────
elif "Shot Analysis" in view:
    st.markdown("## 🎯 Shot Placement Analysis")
    
    # Court heatmap
    player = st.radio("Select Player", [data['match']['player1'], data['match']['player2']], horizontal=True)
    counts = data['shot_placement']['p1_counts'] if player == data['match']['player1'] else data['shot_placement']['p2_counts']
    
    # Create court visualization
    court_data = np.array(counts).reshape(3, 3)
    fig_heat = px.imshow(court_data, color_continuous_scale='Greens', aspect='equal',
                         labels=dict(x="Court Width", y="Court Depth", color="Shots"),
                         x=["Wide", "Body/Center", "T/Down the line"],
                         y=["Service Box", "Mid Court", "Baseline"])
    fig_heat.update_layout(paper_bgcolor='#0a0a0a', plot_bgcolor='#0d1117', font_color='#c9d1d9', height=400, margin=dict(t=30))
    st.plotly_chart(fig_heat, use_container_width=True)
    
    # Shot type breakdown
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Shot Type Distribution")
        shot_types = ['Forehand', 'Backhand', 'Serve', 'Volley', 'Slice', 'Drop']
        shot_counts = [45, 38, 22, 12, 8, 3]
        fig_pie = px.pie(values=shot_counts, names=shot_types, color_discrete_sequence=px.colors.sequential.Greens_r)
        fig_pie.update_layout(paper_bgcolor='#0a0a0a', font_color='#c9d1d9', height=350)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.markdown("### Winner/Error by Shot")
        shot_df = pd.DataFrame({
            'Shot': ['FH', 'BH', 'Serve', 'Volley'] * 2,
            'Type': ['Winners'] * 4 + ['Errors'] * 4,
            'Count': [18, 10, 8, 4, 8, 12, 3, 2],
        })
        fig_bar = px.bar(shot_df, x='Shot', y='Count', color='Type', barmode='group', color_discrete_map={'Winners': '#00ff88', 'Errors': '#ff6b6b'})
        fig_bar.update_layout(paper_bgcolor='#0a0a0a', plot_bgcolor='#0d1117', font_color='#c9d1d9', height=350)
        st.plotly_chart(fig_bar, use_container_width=True)

# ── AI Coaching ──────────────────────────────────────────────────────────────
elif "AI Coaching" in view:
    st.markdown("## 🧠 AI Coaching Insights")
    
    coaching = data['coaching']
    
    # Performance Rating
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(f"""
        <div class="stat-card" style="text-align:center;">
            <div class="stat-value">{coaching['performance_rating']}</div>
            <div class="stat-label">Performance Rating</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.info(f"🎯 **Focus This Week:** {coaching['weekly_goal']}")
    
    # Primary Correction
    st.warning(f"⚡ **Your One Correction:** {coaching['primary_correction']}")
    
    # Strengths & Improvements
    col_s, col_i = st.columns(2)
    with col_s:
        st.markdown("### 💪 Strengths")
        for s in coaching['strengths']:
            st.success(f"✅ {s}")
    with col_i:
        st.markdown("### 📋 Areas for Improvement")
        for i in coaching['improvements']:
            st.warning(f"📌 {i}")
    
    # Suggested Drills
    st.markdown("### 🏋️ Recommended Drills")
    for i, drill in enumerate(coaching['drills'], 1):
        with st.expander(f"Drill {i}: {drill.split('(')[0].strip()}"):
            st.write(drill)

# ── Trends ───────────────────────────────────────────────────────────────────
elif "Trends" in view:
    st.markdown("## 📈 Performance Trends")
    
    dates = pd.date_range(end=datetime.now(), periods=12, freq='W')
    trend_df = pd.DataFrame({
        'Date': dates,
        'First Serve %': np.random.normal(63, 5, 12).clip(45, 80),
        'Winners': np.random.normal(25, 8, 12).clip(8, 45).astype(int),
        'Unforced Errors': np.random.normal(20, 6, 12).clip(5, 40).astype(int),
        'Performance Rating': np.random.normal(6.5, 1, 12).clip(3, 9.5),
    })
    
    metric = st.selectbox("Select Metric", ['First Serve %', 'Winners', 'Unforced Errors', 'Performance Rating'])
    fig_trend = px.line(trend_df, x='Date', y=metric, markers=True, line_shape='spline')
    fig_trend.update_traces(line_color='#00ff88', line_width=3, marker_size=8)
    fig_trend.update_layout(paper_bgcolor='#0a0a0a', plot_bgcolor='#0d1117', font_color='#c9d1d9', height=400, xaxis_gridcolor='#21262d', yaxis_gridcolor='#21262d')
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Summary stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matches Played", "12")
    c2.metric("Win Rate", "67%", "+8%")
    c3.metric("Avg Rating", "6.8", "+0.3")
    c4.metric("Goals Achieved", "8/10")

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption("TennisIQ v1.0 — AI-First Tennis Match Intelligence Platform")
