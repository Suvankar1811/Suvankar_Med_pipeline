"""Human-facing Streamlit entry point for the IDAMP pipeline."""

import html
import json
import sys
from pathlib import Path
from textwrap import dedent

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.orchestrator import (
    run_bronze_to_silver_sttm,
    run_gold_and_report,
    run_silver_to_gold_sttm,
    run_until_bronze_sttm,
)
from core.audit import AuditLogger
from core.config import LANDING_DIR, STTM_DIR


st.set_page_config(
    page_title="IDAMP - Intent-Driven Agentic Medallion Pipeline",
    page_icon="🏗️",
    layout="wide",
)


PROGRESS_STEPS = [
    ("upload", "Upload & Intent"),
    ("bronze_sttm", "Bronze STTM Review"),
    ("bronze_load", "Bronze Layer Load"),
    ("silver_sttm", "Silver STTM Review"),
    ("silver_load", "Silver Layer Load"),
    ("gold_sttm", "Gold STTM Review"),
    ("gold_load", "Gold Layer Load"),
    ("report", "Executive Report"),
]

SELECTION_COL = "_selected_for_approval"


def render_progress_banner(current_phase: str, state: dict, report_complete: bool) -> None:
    """Render the sticky pipeline progress stepper."""
    phase_keys = [key for key, _ in PROGRESS_STEPS]
    current_index = phase_keys.index(current_phase) if current_phase in phase_keys else 0
    fully_complete = report_complete or state.get("status") == "completed"

    step_markup = []
    for index, (key, label) in enumerate(PROGRESS_STEPS):
        is_complete = fully_complete or index < current_index
        is_current = not fully_complete and index == current_index
        if is_complete:
            step_class = "idamp-step is-complete"
            indicator = "✓"
        elif is_current:
            step_class = "idamp-step is-current"
            indicator = str(index + 1)
        else:
            step_class = "idamp-step"
            indicator = str(index + 1)

        connector = "" if index == len(PROGRESS_STEPS) - 1 else '<span class="idamp-connector"></span>'
        step_markup.append(
            f'<div class="{step_class}">'
            f'<span class="idamp-indicator">{indicator}</span>'
            f'<span class="idamp-label">{html.escape(label)}</span>'
            f"</div>{connector}"
        )

    css = dedent(
        """
        <style>
        .idamp-progress-shell {
            position: sticky;
            top: 0;
            z-index: 999;
            margin: -1rem -1rem 1.4rem;
            padding: 1rem clamp(1rem, 3vw, 2.4rem) 1.15rem;
            background: linear-gradient(110deg, #12343b 0%, #176b70 58%, #e28e4d 100%);
            box-shadow: 0 10px 28px rgba(18, 52, 59, .2);
        }
        .idamp-progress-inner {
            display: flex;
            align-items: flex-start;
            max-width: 1280px;
            margin: 0 auto;
            overflow-x: auto;
            scrollbar-width: thin;
        }
        .idamp-step {
            display: flex;
            min-width: 96px;
            flex-direction: column;
            align-items: center;
            gap: .42rem;
            color: rgba(255, 255, 255, .68);
            text-align: center;
        }
        .idamp-indicator {
            display: grid;
            width: 2rem;
            height: 2rem;
            place-items: center;
            border: 2px solid rgba(255, 255, 255, .42);
            border-radius: 50%;
            font: 700 .78rem/1 Arial, sans-serif;
            transition: .2s ease;
        }
        .idamp-label {
            max-width: 108px;
            font: 600 .68rem/1.2 Arial, sans-serif;
        }
        .idamp-connector {
            flex: 1 0 18px;
            height: 2px;
            margin: 1rem .3rem 0;
            background: rgba(255, 255, 255, .3);
        }
        .idamp-step.is-current { color: #fff; }
        .idamp-step.is-current .idamp-indicator {
            border-color: #ffd29a;
            background: #e28e4d;
            box-shadow: 0 0 0 5px rgba(255, 210, 154, .2);
        }
        .idamp-step.is-complete { color: #fff; }
        .idamp-step.is-complete .idamp-indicator {
            border-color: #b9f0d1;
            color: #12343b;
            background: #b9f0d1;
        }
        .idamp-step.is-complete + .idamp-connector { background: #b9f0d1; }
        @media (max-width: 720px) {
            .idamp-progress-shell { margin-left: -.5rem; margin-right: -.5rem; }
            .idamp-step { min-width: 86px; }
            .idamp-label { font-size: .62rem; }
        }
        </style>
        """
    )
    markup = f'<div class="idamp-progress-shell"><div class="idamp-progress-inner">{"".join(step_markup)}</div></div>'
    st.markdown(css + markup, unsafe_allow_html=True)


def _reset_analysis_session() -> None:
    """Reset the active pipeline run and return the UI to the upload phase."""
    st.session_state.phase = "upload"
    st.session_state.pipeline_state = {}
    st.session_state.current_run_id = ""


def _prepare_sttm_editor_df(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare STTM rows for checkbox-based human approval."""
    edited_df = df.copy()
    edited_df[SELECTION_COL] = True
    for column in edited_df.columns:
        if column != SELECTION_COL:
            edited_df[column] = edited_df[column].astype(str)
    return edited_df


def _extract_selected_rows(edited_df: pd.DataFrame) -> pd.DataFrame:
    """Return only STTM rows selected for approval."""
    selected = edited_df[edited_df[SELECTION_COL].fillna(False).astype(bool)].copy()
    return selected.drop(columns=[SELECTION_COL])


def _current_audit_logs() -> list[dict]:
    """Load audit events for the run currently shown in the UI."""
    run_id = st.session_state.get("current_run_id", "")
    if not run_id:
        return []
    return AuditLogger(run_id).get_logs()


def _format_audit_entry(entry: dict) -> str:
    """Format the main fields of an audit event for display."""
    timestamp = entry.get("timestamp", "")
    agent = entry.get("agent", "unknown")
    action = entry.get("action", "unknown")
    rationale = entry.get("rationale", "")
    phase = entry.get("phase", "")
    status = entry.get("status", "")
    details = [f"{timestamp} | {agent} | {action}"]
    if phase:
        details.append(f"phase={phase}")
    if status:
        details.append(f"status={status}")
    if rationale:
        details.append(f"rationale={rationale}")
    return " | ".join(details)


def _status_class(entry: dict) -> str:
    """Return the CSS status class for an audit event."""
    status = str(entry.get("status", "")).lower().replace("-", "_")
    if status in {"failed", "failure", "error"}:
        return "is-failed"
    if status in {"success", "succeeded", "completed", "complete"}:
        return "is-success"
    if status in {"in_progress", "progress", "running"}:
        return "is-progress"
    return "is-neutral"


def _render_audit_card(entry: dict, latest: bool = False) -> str:
    """Build an escaped HTML card for one audit event."""
    latest_class = " is-latest" if latest else ""
    status = str(entry.get("status", "neutral"))
    metadata = []
    if entry.get("phase"):
        metadata.append(f"Phase: {entry['phase']}")
    if entry.get("rationale"):
        metadata.append(f"Rationale: {entry['rationale']}")
    metadata_text = " | ".join(metadata)
    return dedent(
        f"""
        <article class="idamp-audit-card {_status_class(entry)}{latest_class}">
            <div class="idamp-audit-topline">
                <span class="idamp-audit-agent">{html.escape(str(entry.get('agent', 'unknown')))}</span>
                <span class="idamp-audit-time">{html.escape(str(entry.get('timestamp', '')))}</span>
            </div>
            <div class="idamp-audit-action">{html.escape(str(entry.get('action', 'unknown action')))}</div>
            <div class="idamp-audit-meta">{html.escape(metadata_text)}</div>
            <span class="idamp-audit-status">{html.escape(status)}</span>
        </article>
        """
    ).strip()


def render_audit_trail_panel() -> None:
    """Render the current run's latest and historical audit events."""
    logs = _current_audit_logs()
    st.markdown(
        dedent(
            """
            <style>
            .idamp-audit-card {
                position: relative;
                margin: .55rem 0;
                padding: .9rem 1rem;
                border: 1px solid #dbe5e2;
                border-left: 4px solid #9aa9a7;
                border-radius: 9px;
                background: #fffdf8;
                box-shadow: 0 4px 14px rgba(18, 52, 59, .06);
            }
            .idamp-audit-card.is-latest { padding: 1.15rem 1.2rem; border-width: 2px 2px 2px 6px; }
            .idamp-audit-card.is-failed { border-left-color: #c94b52; }
            .idamp-audit-card.is-success { border-left-color: #258a68; }
            .idamp-audit-card.is-progress { border-left-color: #d68c42; }
            .idamp-audit-card.is-neutral { border-left-color: #83959a; }
            .idamp-audit-topline { display: flex; justify-content: space-between; gap: 1rem; color: #62707c; font: .72rem Arial, sans-serif; }
            .idamp-audit-agent { color: #15545c; font-weight: 700; }
            .idamp-audit-action { margin-top: .35rem; color: #17212b; font: 700 1rem Georgia, serif; }
            .idamp-audit-meta { margin-top: .4rem; color: #62707c; font: .8rem/1.4 Arial, sans-serif; }
            .idamp-audit-status { display: inline-block; margin-top: .55rem; color: #52616a; font: 700 .68rem Arial, sans-serif; text-transform: uppercase; }
            .idamp-audit-scroll { max-height: 340px; overflow-y: auto; padding-right: .35rem; }
            </style>
            """
        ),
        unsafe_allow_html=True,
    )
    st.markdown("## Audit Trail")
    if not logs:
        st.caption("No audit events for this run yet.")
        return

    st.markdown(_render_audit_card(logs[-1], latest=True), unsafe_allow_html=True)
    if len(logs) > 1:
        previous_cards = "".join(_render_audit_card(entry) for entry in reversed(logs[:-1]))
        st.markdown(f'<div class="idamp-audit-scroll">{previous_cards}</div>', unsafe_allow_html=True)
    st.download_button(
        "Download audit JSON",
        data=json.dumps(logs, indent=2, default=str),
        file_name="audit_trail.json",
        mime="application/json",
        key="download_audit_trail",
    )


def _load_sttm_for_review(sttm_path: str, layer_name: str) -> pd.DataFrame | None:
    """Load an STTM file and show a restart action when it is unusable."""
    try:
        sttm_df = pd.read_csv(sttm_path)
    except pd.errors.EmptyDataError:
        st.error(f"The {layer_name} STTM file is empty or corrupted.")
        if st.button("↺ Restart Workflow", key=f"restart_{layer_name.lower()}_empty"):
            _reset_analysis_session()
            st.rerun()
        return None
    except (FileNotFoundError, pd.errors.ParserError, OSError) as error:
        st.error(f"Could not load the {layer_name} STTM file: {error}")
        if st.button("↺ Restart Workflow", key=f"restart_{layer_name.lower()}_error"):
            _reset_analysis_session()
            st.rerun()
        return None

    if sttm_df.empty or len(sttm_df.columns) == 0:
        st.error(f"The {layer_name} STTM file is empty or corrupted.")
        if st.button("↺ Restart Workflow", key=f"restart_{layer_name.lower()}_blank"):
            _reset_analysis_session()
            st.rerun()
        return None
    return sttm_df


def _render_sttm_editor(sttm_df: pd.DataFrame, layer_name: str) -> pd.DataFrame:
    prepared_df = _prepare_sttm_editor_df(sttm_df)
    st.caption(f"{len(prepared_df):,} total transformations")
    return st.data_editor(
        prepared_df,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        column_config={
            SELECTION_COL: st.column_config.CheckboxColumn("Select", default=True),
        },
        key=f"{layer_name.lower()}_sttm_editor",
    )


def _restart_from_sttm_error(layer_name: str) -> None:
    if st.button("↺ Restart Workflow", key=f"restart_{layer_name.lower()}_unexpected"):
        _reset_analysis_session()
        st.rerun()


st.title("Intent-Driven Agentic Medallion Workflow")
st.markdown(
    "Move raw retail data through governed Bronze, Silver, and Gold layers with human approval at each mapping stage."
)

st.session_state.setdefault("pipeline_state", None)
st.session_state.setdefault("phase", "upload")
st.session_state.setdefault("current_run_id", "")
st.session_state.setdefault("report_complete", False)

pipeline_state = st.session_state.pipeline_state
if pipeline_state:
    st.session_state.current_run_id = pipeline_state.get("run_id", "")

current_phase = st.session_state.phase
render_progress_banner(
    current_phase,
    pipeline_state or {},
    st.session_state.report_complete,
)

main_col, audit_col = st.columns([3.4, 1.4])

with audit_col:
    render_audit_trail_panel()

with main_col:
    if current_phase == "upload":
        st.header("📤 Phase 1: Upload Data & Define Intent")
        uploaded_files = st.file_uploader(
            "Upload CSV files",
            type=["csv"],
            accept_multiple_files=True,
        )
        business_intent = st.text_area(
            "Business question",
            placeholder="Example: Which regions and product categories drive the most revenue?",
            height=120,
        )
        start_workflow = st.button(
            "🚀 Start Workflow",
            type="primary",
            disabled=not uploaded_files or not business_intent.strip(),
            use_container_width=True,
        )

        if start_workflow:
            saved_paths = []
            for uploaded_file in uploaded_files:
                destination = LANDING_DIR / Path(uploaded_file.name).name
                destination.write_bytes(uploaded_file.getbuffer())
                saved_paths.append(str(destination))

            with st.spinner("🔍 Profiling data..."):
                result = run_until_bronze_sttm(saved_paths, business_intent.strip())
            st.session_state.pipeline_state = result
            st.session_state.current_run_id = result.get("run_id", "")
            st.session_state.phase = "bronze_sttm"
            st.rerun()

    elif current_phase == "bronze_sttm":
        st.header("🥉 Phase 2: Review Bronze Layer STTM")
        st.info("Bronze is the raw ingestion layer. It preserves source data while adding load metadata.")
        state = st.session_state.pipeline_state or {}
        sttm_path = state.get("sttm_bronze_path", "")
        sttm_df = _load_sttm_for_review(sttm_path, "Bronze")
        if sttm_df is not None:
            edited_df = _render_sttm_editor(sttm_df, "bronze")
            if st.button("✅ Approve & Continue", type="primary", key="approve_bronze_sttm"):
                try:
                    _extract_selected_rows(edited_df).to_csv(sttm_path, index=False)
                    updated_state = run_bronze_to_silver_sttm(state)
                    st.session_state.pipeline_state = updated_state
                    if updated_state.get("status") == "awaiting_silver_sttm_approval":
                        st.session_state.phase = "silver_sttm"
                    st.rerun()
                except Exception as error:
                    st.error(f"Bronze approval failed: {error}")
                    _restart_from_sttm_error("Bronze")

    elif current_phase == "silver_sttm":
        st.header("🥈 Phase 3: Review Silver Layer STTM")
        st.info("Silver is the cleansing layer. It standardizes types, handles nulls, and removes duplicates.")
        state = st.session_state.pipeline_state or {}
        sttm_path = state.get("sttm_silver_path", "")
        sttm_df = _load_sttm_for_review(sttm_path, "Silver")
        if sttm_df is not None:
            edited_df = _render_sttm_editor(sttm_df, "silver")
            if st.button("✅ Approve & Continue", type="primary", key="approve_silver_sttm"):
                try:
                    _extract_selected_rows(edited_df).to_csv(sttm_path, index=False)
                    updated_state = run_silver_to_gold_sttm(state)
                    st.session_state.pipeline_state = updated_state
                    if updated_state.get("status") == "awaiting_gold_sttm_approval":
                        st.session_state.phase = "gold_sttm"
                    st.rerun()
                except Exception as error:
                    st.error(f"Silver approval failed: {error}")
                    _restart_from_sttm_error("Silver")

    elif current_phase == "gold_sttm":
        st.header("🥇 Phase 4: Review Gold Layer STTM")
        st.info("Gold is the analytics layer. It joins and aggregates cleansed data for business reporting.")
        state = st.session_state.pipeline_state or {}
        sttm_path = state.get("sttm_gold_path", "")
        sttm_df = _load_sttm_for_review(sttm_path, "Gold")
        if sttm_df is not None:
            edited_df = _render_sttm_editor(sttm_df, "gold")
            if st.button("✅ Approve & Execute", type="primary", key="approve_gold_sttm"):
                try:
                    _extract_selected_rows(edited_df).to_csv(sttm_path, index=False)
                    updated_state = run_gold_and_report(state)
                    st.session_state.pipeline_state = updated_state
                    if updated_state.get("status") == "completed":
                        st.session_state.phase = "report"
                        st.session_state.report_complete = True
                    st.rerun()
                except Exception as error:
                    st.error(f"Gold execution failed: {error}")
                    _restart_from_sttm_error("Gold")

    elif current_phase == "report":
        st.header("📊 Phase 5: Executive Report")
        st.session_state.report_complete = True
        state = st.session_state.pipeline_state or {}
        report_path = state.get("report_path", "")

        if report_path and Path(report_path).is_file():
            report_html = Path(report_path).read_text(encoding="utf-8")
            st.components.v1.html(report_html, height=2000, scrolling=True)
            download_col, restart_col = st.columns(2)
            with download_col:
                st.download_button(
                    "📥 Download Report",
                    data=report_html,
                    file_name=Path(report_path).name,
                    mime="text/html",
                    key="download_report",
                    use_container_width=True,
                )
            with restart_col:
                if st.button("🔄 Start New Analysis", key="start_new_analysis", use_container_width=True):
                    _reset_analysis_session()
                    st.rerun()
        else:
            st.error("Report file not found. Please check the pipeline execution.")
