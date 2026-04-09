# ============================================================
# app.py
# Entry point — page config, CSS injection, session init, router
# Run with: streamlit run app.py
# ============================================================

import streamlit as st

from config.settings import SESSION_DEFAULTS
from config.theme import inject_css
from components.ui import hero, pipeline_stepper
from pipeline.steps import (
    step_load,
    step_eda,
    step_clean,
    step_features,
    step_cluster,
    step_results,
    step_learn,
)

# ── Page config (must be first Streamlit call) ──────────────
st.set_page_config(
    page_title="Clustering",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject dark theme CSS ───────────────────────────────────
inject_css()

# ── Session state init ──────────────────────────────────────
for k, v in SESSION_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Guide")

    step_labels = [
        "📂 Load Data",
        "📊 EDA",
        "🧹 Cleaning",
        "⚙️ Feature Engineering",
        "🤖 Clustering",
        "📈 Results",
        "🎓 Learn ML",
    ]

    for i, name in enumerate(step_labels):
        is_active = i == st.session_state.step
        prefix = "👉" if is_active else "•"
        if st.button(
            f"{prefix} {name}",
            key=f"nav_{i}",
            use_container_width=True,
        ):
            st.session_state.step = i
            st.rerun()

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "📂 Upload CSV",
        type=["csv"],
        help="Max 50 MB. Supports numeric and categorical columns.",
    )

# ── Header ───────────────────────────────────────────────────
hero()
pipeline_stepper()

# ── Router ───────────────────────────────────────────────────
_STEP_FUNCTIONS = {
    0: lambda: step_load(uploaded_file),
    1: step_eda,
    2: step_clean,
    3: step_features,
    4: step_cluster,
    5: step_results,
    6: step_learn,
}

step = st.session_state.step
handler = _STEP_FUNCTIONS.get(step)

try:
    if handler:
        handler()
    else:
        st.error(f"Unknown step: {step}")
        st.session_state.step = 0
        st.rerun()
except Exception as e:
    st.error("🚨 Application Error")
    st.exception(e)

# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.caption("Work in Progress")
