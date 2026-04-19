"""
eval_dashboard.py  —  Phase 3 Streamlit leaderboard dashboard
Run:  streamlit run eval_dashboard.py
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json, glob
 
st.set_page_config(
    page_title="RAG Eval Leaderboard",
    page_icon="🏆",
    layout="wide"
)
 
# ── custom CSS ──────────────────────────────────────────────────
st.markdown("""<style>
.metric-card {
    background: #1A56A0; color: white;
    padding: 1rem; border-radius: 8px; text-align: center;
}
.metric-value { font-size: 2rem; font-weight: bold; }
.metric-label { font-size: 0.85rem; opacity: 0.8; }
.best-config { background: #DFF2EC; border-left: 4px solid #0F6E56;
               padding: 0.75rem 1rem; border-radius: 4px; }
</style>""", unsafe_allow_html=True)
 
# ── load data ───────────────────────────────────────────────────
LEADERBOARD_PATH = Path("results") / "leaderboard.csv"
SUMMARY_PATH     = Path("results") / "summary.csv"
 
st.title("RAG Evaluation Leaderboard")
st.caption("Healthcare Document RAG Assistant — Benchmark Results")
 
if not LEADERBOARD_PATH.exists():
    st.warning("No leaderboard.csv found. Run python run_benchmark.py first.")
    st.stop()
 
df = pd.read_csv(LEADERBOARD_PATH, index_col=0)
best = df.iloc[0]
# ── top metric cards ────────────────────────────────────────────
st.markdown("### Best Configuration")
st.markdown(
    f'<div class="best-config">Best config: <strong>{best["config"]}</strong>'
    f' — Composite score: <strong>{best["composite_score"]:.3f}</strong></div>',
    unsafe_allow_html=True)
st.markdown("")
 
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Faithfulness",     f"{best['faithfulness']:.3f}",     help="Is answer grounded in context?")
with c2: st.metric("Answer Relevancy",  f"{best['answer_relevancy']:.3f}",  help="Does answer address the question?")
with c3: st.metric("Context Recall",    f"{best['context_recall']:.3f}",    help="Did retrieval surface right info?")
with c4: st.metric("Avg Latency",       f"{best['avg_latency_s']:.2f}s",    help="Average response time")
 
st.divider()
 
# ── bar chart comparison ─────────────────────────────────────────
st.markdown("### Score Comparison Across Configs")
fig = go.Figure()
colors = {"faithfulness": "#1A56A0", "answer_relevancy": "#0F6E56",
          "context_recall": "#C05621"}
for metric, color in colors.items():
    fig.add_trace(go.Bar(
        name=metric.replace("_", " ").title(),
        x=df["config"], y=df[metric],
        marker_color=color, opacity=0.85
    ))
fig.update_layout(
    barmode="group", yaxis_range=[0, 1.1],
    height=420, plot_bgcolor="white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    yaxis=dict(gridcolor="#EEEEEE"),
    xaxis=dict(tickangle=-20)
)
st.plotly_chart(fig, use_container_width=True)
 
# ── composite score ranking ──────────────────────────────────────
st.markdown("### Composite Score Ranking")
fig2 = px.bar(
    df.reset_index(), x="config", y="composite_score",
    color="composite_score", color_continuous_scale="Blues",
    height=300
)
fig2.update_layout(plot_bgcolor="white", yaxis_range=[0, 1.1],
    yaxis=dict(gridcolor="#EEEEEE"), xaxis=dict(tickangle=-20))
st.plotly_chart(fig2, use_container_width=True)
 
# ── full table ───────────────────────────────────────────────────
st.markdown("### Full Results Table")
st.dataframe(
    df[["config", "chunk_size", "chunk_overlap", "top_k",
        "faithfulness", "answer_relevancy", "context_recall",
        "avg_latency_s", "composite_score"]].style
    .highlight_max(subset=["faithfulness","answer_relevancy",
                           "context_recall","composite_score"],
                   color="#DFF2EC")
    .format({"faithfulness": "{:.3f}", "answer_relevancy": "{:.3f}",
             "context_recall": "{:.3f}", "composite_score": "{:.3f}",
             "avg_latency_s": "{:.2f}s"}),
    use_container_width=True, height=280
)
 
# ── run history ─────────────────────────────────────────────────
if SUMMARY_PATH.exists():
    st.divider()
    st.markdown("### Run History")
    hist = pd.read_csv(SUMMARY_PATH)
    hist["timestamp"] = pd.to_datetime(hist["timestamp"], format="%Y%m%d_%H%M%S")
    fig3 = px.line(hist, x="timestamp", y="context_recall",
                   color="config", markers=True, height=300)
    fig3.update_layout(plot_bgcolor="white", yaxis_range=[0, 1.1],
        yaxis=dict(gridcolor="#EEEEEE"))
    st.plotly_chart(fig3, use_container_width=True)
