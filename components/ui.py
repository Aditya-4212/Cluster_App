# ============================================================
# components/ui.py
# Reusable UI helper components used across pipeline steps
# ============================================================

import html as _html
import streamlit as st
from config.settings import PIPELINE_STEPS


# ── Section divider ─────────────────────────────────────────

def section(title: str) -> None:
    """Render a monospaced uppercase section divider."""
    safe = _html.escape(str(title))
    st.markdown(f'<div class="sec">{safe}</div>', unsafe_allow_html=True)


# ── Learn / info / warn boxes ────────────────────────────────

def explain(title: str, body: str, kind: str = "learn") -> None:
    """
    Render a contextual info box.

    kind: 'learn' (violet) | 'warn' (amber) | 'success' (emerald) | 'insight' (cyan)
    """
    cls_map = {
        "learn":   "learn-box",
        "warn":    "warn-box",
        "success": "success-box",
        "insight": "insight",
    }
    cls = cls_map.get(kind, "insight")
    safe_title = _html.escape(str(title))

    if cls == "learn-box":
        st.markdown(
            f'<div class="{cls}">'
            f'<div class="learn-title">{safe_title}</div>'
            f'<div class="learn-body">{body}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="{cls}"><strong>{safe_title}</strong> {body}</div>',
            unsafe_allow_html=True,
        )


# ── Metric strip ─────────────────────────────────────────────

def metric_strip(metrics: dict, model_name: str) -> None:
    """Render a row of st.metric tiles for clustering scores."""
    display = {k: v for k, v in metrics.items() if k not in ("Clusters",)}
    n_clusters = metrics.get("Clusters", "?")

    st.markdown(
        f'<div class="sec">Model: {_html.escape(str(model_name))} '
        f'· {n_clusters} cluster(s)</div>',
        unsafe_allow_html=True,
    )

    keys = list(display.keys())
    if not keys:
        return

    cols = st.columns(len(keys))
    for col, key in zip(cols, keys):
        col.metric(label=key, value=display[key])


# ── Progress tracker ──────────────────────────────────────────

def progress_tracker() -> None:
    """Render a simple pipeline completion indicator."""
    flags = [
        ("preprocessing_done", "Data Cleaned"),
        ("eda_done",           "EDA Complete"),
        ("engineering_done",   "Features Locked"),
        ("clustering_done",    "Model Trained"),
    ]
    done = [(label) for key, label in flags if st.session_state.get(key)]
    total = len(flags)
    count = len(done)

    bar_pct = int(count / total * 100)
    labels_str = " · ".join(done) if done else "Not started"

    st.markdown(
        f'<div class="insight">'
        f'<strong>Pipeline Progress: {count}/{total}</strong><br>'
        f'<div style="background:#1e2035;border-radius:4px;height:4px;margin:0.5rem 0;">'
        f'<div style="background:#22d3ee;height:4px;border-radius:4px;width:{bar_pct}%;"></div>'
        f'</div>'
        f'<span style="font-size:0.75rem;">{_html.escape(labels_str)}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Pipeline stepper (visual header nav) ─────────────────────

def pipeline_stepper() -> None:
    """Render the horizontal pipeline step indicator."""
    current = st.session_state.get("step", 0)
    steps_html = ""
    for i, (icon, label) in enumerate(PIPELINE_STEPS):
        active_cls = " active" if i == current else ""
        steps_html += (
            f'<div class="step-btn{active_cls}">'
            f'<span class="step-num">{i + 1:02d}</span>'
            f'<span class="step-icon">{icon}</span>'
            f'<span class="step-label">{_html.escape(label)}</span>'
            f'</div>'
        )
    st.markdown(
        f'<div class="pipeline-nav">{steps_html}</div>',
        unsafe_allow_html=True,
    )


# ── Hero header ───────────────────────────────────────────────

def hero(title: str = "ML Clustering Studio", subtitle: str = "End-to-End Production Pipeline") -> None:
    """Render the gradient hero title block."""
    st.markdown(
        f'<div class="hero">'
        f'<div class="hero-title">{_html.escape(title)}</div>'
        f'<div class="hero-sub">{_html.escape(subtitle)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
