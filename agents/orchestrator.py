"""Supervisor coordinator for the retail analytics pipeline.

The pipeline runs in four phases:

1. Phase 1: Profile data and generate Bronze STTM.
2. Phase 2: Execute Bronze and generate Silver STTM.
3. Phase 3: Execute Silver and generate Gold STTM.
4. Phase 4: Execute Gold and generate the business report.

The supervisor coordinates specialist agents and pauses between phases when
human approval of generated STTM rules is required.
"""

import json
import uuid
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from agents.bronze_agent import execute_bronze
from agents.gold_agent import execute_gold
from agents.profiler import profile_multiple_datasets
from agents.reporter import generate_report
from agents.silver_agent import execute_silver
from agents.sttm_generator import (
    generate_bronze_sttm,
    generate_gold_sttm,
    generate_silver_sttm,
)
from core.audit import AuditLogger
from core.config import (
    GITHUB_BASE_URL,
    GITHUB_MODEL,
    GITHUB_TOKEN,
    GEMINI_MODEL,
    GOOGLE_API_KEY,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)
from core.observability import AgentTrace
from core.state import PipelineState as PydanticPipelineState


class PipelineState(TypedDict):
    """State exposed to the UI while the supervisor advances the pipeline."""

    run_id: str
    status: str
    uploaded_files: list[str]
    business_intent: str
    profile_path: str
    sttm_bronze_path: str
    sttm_silver_path: str
    sttm_gold_path: str
    bronze_sttm_approved: bool
    silver_sttm_approved: bool
    gold_sttm_approved: bool
    bronze_output_paths: list[str]
    silver_output_paths: list[str]
    gold_output_paths: list[str]
    report_path: str
    error: str | None


SUPERVISOR_PROMPT = """
You are the Supervisor agent coordinating a retail analytics pipeline.
Coordinate specialist AI agents by calling them as tools.
You will receive TWO STEPS to execute for each supervisor turn.
Call each tool EXACTLY ONCE, in the order provided.
Verify each tool's output before proceeding to the next step.
Each tool is an autonomous agent that handles its own execution; do not duplicate its work.
Only call the tools exposed in the current turn; never invent or call tools from another agent.
If a tool fails or its output cannot be verified, stop and report the failure.
""".strip()


def _make_llm():
    """Build the deterministic chat model used for supervisor tool-calling."""
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=0,
            reasoning_format="hidden",
            reasoning_effort="low",
        )

    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(api_key=OPENAI_API_KEY, model=OPENAI_MODEL, temperature=0)

    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            google_api_key=GOOGLE_API_KEY,
            model=GEMINI_MODEL,
            temperature=0,
        )

    if LLM_PROVIDER == "github":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=GITHUB_TOKEN,
            model=GITHUB_MODEL,
            base_url=GITHUB_BASE_URL,
            temperature=0,
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


def _make_phase1_tools(uploaded_files: list[str], run_id: str):
    """Create profiling and Bronze-STTM dispatch tools for Phase 1."""
    scratchpad: dict[str, str] = {}

    @tool("profiler_agent_tool")
    def profiler_agent_tool(goal: str) -> str:
        """Profile all uploaded CSV files."""
        print(f"[Supervisor] Dispatching profiler agent: {goal}")
        profile_path = profile_multiple_datasets(
            uploaded_files,
            run_id,
            f"Profile the uploaded retail CSV files and identify structure, quality issues, and join keys. Goal: {goal}",
        )
        scratchpad["profile_path"] = profile_path
        return json.dumps({"profile_path": profile_path})

    @tool("sttm_agent_tool")
    def sttm_agent_tool(goal: str) -> str:
        """Generate Bronze STTM rules from the completed profile."""
        print(f"[Supervisor] Dispatching Bronze STTM agent: {goal}")
        profile_path = scratchpad.get("profile_path")
        if not profile_path:
            raise ValueError("Profiler agent must run before the Bronze STTM agent")
        sttm_path = generate_bronze_sttm(
            profile_path,
            run_id,
            f"Generate Bronze source-to-target mappings for every profiled column, including ingestion metadata. Goal: {goal}",
        )
        scratchpad["sttm_bronze_path"] = sttm_path
        return json.dumps({"sttm_bronze_path": sttm_path})

    return [profiler_agent_tool, sttm_agent_tool], scratchpad


def _make_phase2_tools(uploaded_files: list[str], sttm_bronze_path: str, run_id: str):
    """Create Bronze-ingestion and Silver-STTM dispatch tools for Phase 2."""
    scratchpad: dict[str, object] = {}

    @tool("bronze_agent_tool")
    def bronze_agent_tool(goal: str) -> str:
        """Execute Bronze ingestion for the uploaded files."""
        print(f"[Supervisor] Dispatching Bronze agent: {goal}")
        bronze_output_paths = execute_bronze(
            uploaded_files,
            sttm_bronze_path,
            run_id,
            f"Ingest the uploaded CSV files into Bronze Parquet using the approved Bronze STTM rules. Goal: {goal}",
        )
        scratchpad["bronze_output_paths"] = bronze_output_paths
        return json.dumps({"bronze_output_paths": bronze_output_paths})

    @tool("sttm_agent_tool")
    def sttm_agent_tool(goal: str) -> str:
        """Generate Silver STTM rules from completed Bronze outputs."""
        print(f"[Supervisor] Dispatching Silver STTM agent: {goal}")
        bronze_output_paths = scratchpad.get("bronze_output_paths")
        if not bronze_output_paths:
            raise ValueError("Bronze agent must run before the Silver STTM agent")
        sttm_path = generate_silver_sttm(
            bronze_output_paths,
            sttm_bronze_path,
            run_id,
            f"Generate Silver cleansing and standardization mappings from the Bronze outputs. Goal: {goal}",
        )
        scratchpad["sttm_silver_path"] = sttm_path
        return json.dumps({"sttm_silver_path": sttm_path})

    return [bronze_agent_tool, sttm_agent_tool], scratchpad


def _make_phase3_tools(
    bronze_output_paths: list[str],
    sttm_silver_path: str,
    business_intent: str,
    run_id: str,
):
    """Create Silver-ingestion and Gold-STTM dispatch tools for Phase 3."""
    scratchpad: dict[str, object] = {}

    @tool("silver_agent_tool")
    def silver_agent_tool(goal: str) -> str:
        """Execute Silver cleansing for Bronze outputs."""
        print(f"[Supervisor] Dispatching Silver agent: {goal}")
        silver_output_paths = execute_silver(
            bronze_output_paths,
            sttm_silver_path,
            run_id,
            f"Cleanse Bronze Parquet data into Silver, handling nulls, duplicates, types, and dates. Goal: {goal}",
        )
        scratchpad["silver_output_paths"] = silver_output_paths
        return json.dumps({"silver_output_paths": silver_output_paths})

    @tool("sttm_agent_tool")
    def sttm_agent_tool(goal: str) -> str:
        """Generate Gold STTM rules from completed Silver outputs."""
        print(f"[Supervisor] Dispatching Gold STTM agent: {goal}")
        silver_output_paths = scratchpad.get("silver_output_paths")
        if not silver_output_paths:
            raise ValueError("Silver agent must run before the Gold STTM agent")
        sttm_path = generate_gold_sttm(
            silver_output_paths,
            sttm_silver_path,
            business_intent,
            run_id,
            f"Generate Gold joins, aggregations, and analytics mappings for the business intent. Goal: {goal}",
        )
        scratchpad["sttm_gold_path"] = sttm_path
        return json.dumps({"sttm_gold_path": sttm_path})

    return [silver_agent_tool, sttm_agent_tool], scratchpad


def _make_phase4_tools(
    silver_output_paths: list[str],
    sttm_gold_path: str,
    business_intent: str,
    run_id: str,
):
    """Create Gold-ingestion and reporting dispatch tools for Phase 4."""
    scratchpad: dict[str, object] = {}

    @tool("gold_agent_tool")
    def gold_agent_tool(goal: str) -> str:
        """Execute Gold joins and aggregations for Silver outputs."""
        print(f"[Supervisor] Dispatching Gold agent: {goal}")
        gold_output_paths = execute_gold(
            silver_output_paths,
            sttm_gold_path,
            run_id,
            f"Build analytics-ready Gold tables with joins and aggregations. Goal: {goal}",
        )
        scratchpad["gold_output_paths"] = gold_output_paths
        return json.dumps({"gold_output_paths": gold_output_paths})

    @tool("reporter_agent_tool")
    def reporter_agent_tool(goal: str) -> str:
        """Generate the final report from completed Gold outputs."""
        print(f"[Supervisor] Dispatching reporter agent: {goal}")
        gold_output_paths = scratchpad.get("gold_output_paths")
        if not gold_output_paths:
            raise ValueError("Gold agent must run before the reporter agent")
        report_path = generate_report(
            gold_output_paths,
            business_intent,
            run_id,
            f"Answer the business question with SQL, analysis, and charts from the Gold tables. Goal: {goal}",
        )
        scratchpad["report_path"] = report_path
        return json.dumps({"report_path": report_path})

    return [gold_agent_tool, reporter_agent_tool], scratchpad


def _run_supervisor(tools, phase_goal: str, phase_name: str, run_id: str):
    """Run one supervisor turn and return the complete ReAct result."""
    trace = AgentTrace(f"supervisor_{phase_name.lower()}", run_id).set_input(
        phase=phase_name,
        phase_goal=phase_goal,
    )
    try:
        agent = create_react_agent(
            model=_make_llm(),
            tools=tools,
            prompt=SUPERVISOR_PROMPT,
        )
        result = agent.invoke({
            "messages": [HumanMessage(content=phase_goal)],
        })
        trace.extract_from_messages(result.get("messages", []))
        trace.set_output(phase=phase_name).complete()
        return result
    except Exception as error:
        trace.fail(error)
        raise


def _initial_state(run_id: str, uploaded_files: list[str], business_intent: str) -> PipelineState:
    return {
        "run_id": run_id,
        "status": "initialized",
        "uploaded_files": uploaded_files,
        "business_intent": business_intent,
        "profile_path": "",
        "sttm_bronze_path": "",
        "sttm_silver_path": "",
        "sttm_gold_path": "",
        "bronze_sttm_approved": False,
        "silver_sttm_approved": False,
        "gold_sttm_approved": False,
        "bronze_output_paths": [],
        "silver_output_paths": [],
        "gold_output_paths": [],
        "report_path": "",
        "error": None,
    }


def run_until_bronze_sttm(uploaded_files: list[str], business_intent: str) -> PipelineState:
    """Run Phase 1 and pause for Bronze STTM approval."""
    run_id = str(uuid.uuid4())
    state = _initial_state(run_id, uploaded_files, business_intent)
    audit = AuditLogger(run_id)
    audit.log(agent="orchestrator", action="phase_start", phase="phase1")
    try:
        tools, scratchpad = _make_phase1_tools(uploaded_files, run_id)
        _run_supervisor(
            tools,
            "Call profiler_agent_tool exactly once, verify its profile_path, then call sttm_agent_tool exactly once to generate Bronze STTM rules.",
            "phase1",
            run_id,
        )
        state["profile_path"] = scratchpad.get("profile_path", "")
        state["sttm_bronze_path"] = scratchpad.get("sttm_bronze_path", "")
        state["status"] = "awaiting_bronze_sttm_approval"
        audit.log(agent="orchestrator", action="phase_complete", phase="phase1", state=state)
    except Exception as error:
        state["error"] = str(error)
        state["status"] = "failed"
        audit.log(agent="orchestrator", action="phase_fail", phase="phase1", error=str(error))
    return state


def run_bronze_to_silver_sttm(state: PipelineState) -> PipelineState:
    """Run Phase 2 and pause for Silver STTM approval."""
    state = dict(state)
    state["bronze_sttm_approved"] = True
    audit = AuditLogger(state["run_id"])
    audit.log(agent="orchestrator", action="phase_start", phase="phase2")
    try:
        tools, scratchpad = _make_phase2_tools(
            state["uploaded_files"],
            state["sttm_bronze_path"],
            state["run_id"],
        )
        _run_supervisor(
            tools,
            "Call bronze_agent_tool exactly once, verify its bronze_output_paths, then call sttm_agent_tool exactly once to generate Silver STTM rules.",
            "phase2",
            state["run_id"],
        )
        state["bronze_output_paths"] = scratchpad.get("bronze_output_paths", [])
        state["sttm_silver_path"] = scratchpad.get("sttm_silver_path", "")
        state["status"] = "awaiting_silver_sttm_approval"
        audit.log(agent="orchestrator", action="phase_complete", phase="phase2", state=state)
    except Exception as error:
        state["error"] = str(error)
        state["status"] = "failed"
        audit.log(agent="orchestrator", action="phase_fail", phase="phase2", error=str(error))
    return state


def run_silver_to_gold_sttm(state: PipelineState) -> PipelineState:
    """Run Phase 3 and pause for Gold STTM approval."""
    state = dict(state)
    state["silver_sttm_approved"] = True
    audit = AuditLogger(state["run_id"])
    audit.log(agent="orchestrator", action="phase_start", phase="phase3")
    try:
        tools, scratchpad = _make_phase3_tools(
            state["bronze_output_paths"],
            state["sttm_silver_path"],
            state["business_intent"],
            state["run_id"],
        )
        _run_supervisor(
            tools,
            "Call silver_agent_tool exactly once, verify its silver_output_paths, then call sttm_agent_tool exactly once to generate Gold STTM rules.",
            "phase3",
            state["run_id"],
        )
        state["silver_output_paths"] = scratchpad.get("silver_output_paths", [])
        state["sttm_gold_path"] = scratchpad.get("sttm_gold_path", "")
        state["status"] = "awaiting_gold_sttm_approval"
        audit.log(agent="orchestrator", action="phase_complete", phase="phase3", state=state)
    except Exception as error:
        state["error"] = str(error)
        state["status"] = "failed"
        audit.log(agent="orchestrator", action="phase_fail", phase="phase3", error=str(error))
    return state


def run_gold_and_report(state: PipelineState) -> PipelineState:
    """Run Phase 4 and complete the pipeline with a business report."""
    state = dict(state)
    state["gold_sttm_approved"] = True
    audit = AuditLogger(state["run_id"])
    audit.log(agent="orchestrator", action="phase_start", phase="phase4")
    try:
        tools, scratchpad = _make_phase4_tools(
            state["silver_output_paths"],
            state["sttm_gold_path"],
            state["business_intent"],
            state["run_id"],
        )
        _run_supervisor(
            tools,
            "Call gold_agent_tool exactly once, verify its gold_output_paths, then call reporter_agent_tool exactly once to generate the final report.",
            "phase4",
            state["run_id"],
        )
        state["gold_output_paths"] = scratchpad.get("gold_output_paths", [])
        state["report_path"] = scratchpad.get("report_path", "")
        state["status"] = "completed"
        audit.log(agent="orchestrator", action="phase_complete", phase="phase4", state=state)
    except Exception as error:
        state["error"] = str(error)
        state["status"] = "failed"
        audit.log(agent="orchestrator", action="phase_fail", phase="phase4", error=str(error))
    return state
