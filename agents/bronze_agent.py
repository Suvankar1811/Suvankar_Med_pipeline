import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from core.audit import AuditLogger
from core.config import (
    BRONZE_DIR,
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


BRONZE_AGENT_PROMPT = """
You are a Bronze-layer data ingestion agent.
You MUST call both tools in this exact order: inspect_task_tool, then bronze_ingestion_tool.
Do not just plan; actually execute the ingestion by calling bronze_ingestion_tool.
Planning without executing is FAILURE. After execution, verify that the requested Parquet files exist.
""".strip()


def _inspect_task(input_files: list[str], sttm_path: str) -> dict[str, Any]:
    """Preview input CSV shapes and the Bronze STTM rules."""
    previews: dict[str, Any] = {}
    for input_file in input_files:
        frame = pd.read_csv(input_file)
        previews[str(input_file)] = {
            "shape": [int(frame.shape[0]), int(frame.shape[1])],
            "columns": [str(column) for column in frame.columns],
            "sample": frame.head(3).to_dict(orient="records"),
        }

    sttm = pd.read_csv(sttm_path)
    return {
        "input_files": previews,
        "sttm_path": sttm_path,
        "sttm_shape": [int(sttm.shape[0]), int(sttm.shape[1])],
        "sttm_rules": sttm.to_dict(orient="records"),
    }


def _apply_bronze_rules(
    input_files: list[str],
    sttm_path: str,
    run_id: str,
) -> list[str]:
    """Apply Bronze mappings, add ingestion metadata, and write Parquet files."""
    audit = AuditLogger(run_id)
    sttm = pd.read_csv(sttm_path)
    output_paths: list[str] = []
    load_timestamp = datetime.now(timezone.utc).isoformat()

    for input_file in input_files:
        frame = pd.read_csv(input_file)
        source_name = Path(input_file).stem
        rules = sttm[sttm["source_table"].astype(str).isin({source_name, Path(input_file).name})]
        if rules.empty:
            rules = sttm

        for _, rule in rules.iterrows():
            source_column = str(rule.get("source_column", ""))
            target_column = str(rule.get("target_column", ""))
            if not source_column or not target_column or source_column not in frame.columns:
                continue

            if source_column != target_column:
                frame = frame.rename(columns={source_column: target_column})

            transformation = " ".join(
                str(rule.get(field, ""))
                for field in ("transformation_type", "transformation_logic")
            ).lower()
            if any(token in transformation for token in ("numeric", "decimal", "integer", "int", "float")):
                frame[target_column] = pd.to_numeric(frame[target_column], errors="coerce")
            elif any(token in transformation for token in ("datetime", "date", "timestamp")):
                frame[target_column] = pd.to_datetime(frame[target_column], errors="coerce")

        frame["_load_timestamp"] = load_timestamp
        frame["_source_file"] = str(input_file)
        output_path = BRONZE_DIR / f"{source_name}_bronze.parquet"
        frame.to_parquet(output_path, index=False)
        output_paths.append(str(output_path))
        audit.log(
            agent="bronze_agent",
            action="bronze_ingestion",
            input_file=str(input_file),
            output_path=str(output_path),
            rows=int(len(frame)),
        )

    return output_paths


def _make_bronze_tools(
    input_files: list[str],
    sttm_path: str,
    run_id: str,
):
    """Create tools scoped to one Bronze ingestion task."""
    @tool("inspect_task_tool")
    def inspect_task_tool() -> str:
        """Preview source CSV files and the Bronze STTM rules."""
        return json.dumps(_inspect_task(input_files, sttm_path), default=str)

    @tool("bronze_ingestion_tool")
    def bronze_ingestion_tool() -> str:
        """Execute Bronze mappings and write Parquet outputs."""
        return json.dumps(_apply_bronze_rules(input_files, sttm_path, run_id))

    return [inspect_task_tool, bronze_ingestion_tool]


def _make_llm():
    """Build the configured chat model for Bronze ingestion."""
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            reasoning_format="hidden",
            reasoning_effort="low",
        )

    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(api_key=OPENAI_API_KEY, model=OPENAI_MODEL)

    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            google_api_key=GOOGLE_API_KEY,
            model=GEMINI_MODEL,
        )

    if LLM_PROVIDER == "github":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=GITHUB_TOKEN,
            model=GITHUB_MODEL,
            base_url=GITHUB_BASE_URL,
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


def _run_bronze_agent(
    input_files: list[str],
    sttm_path: str,
    run_id: str,
    task_description: str,
) -> list[str]:
    trace = AgentTrace("bronze_agent", run_id).set_input(
        input_files=input_files,
        sttm_path=sttm_path,
        task_description=task_description,
    )
    audit = AuditLogger(run_id)

    try:
        tools = _make_bronze_tools(input_files, sttm_path, run_id)
        agent = create_react_agent(model=_make_llm(), tools=tools, prompt=BRONZE_AGENT_PROMPT)
        result = agent.invoke({
            "messages": [
                HumanMessage(
                    content=(
                        f"Ingest these files into Bronze: {', '.join(input_files)}. "
                        f"Use STTM rules at {sttm_path}. Task: {task_description}"
                    )
                )
            ]
        })
        messages = result.get("messages", [])
        trace.extract_from_messages(messages)
        expected_paths = [str(BRONZE_DIR / f"{Path(path).stem}_bronze.parquet") for path in input_files]
        missing_paths = [path for path in expected_paths if not Path(path).is_file()]
        if missing_paths:
            raise FileNotFoundError(f"Bronze ingestion did not create: {missing_paths}")

        audit.log(agent="bronze_agent", action="completed", output_paths=expected_paths)
        trace.set_output(output_paths=expected_paths).complete()
        return expected_paths
    except Exception as error:
        audit.log(agent="bronze_agent", action="failed", error=str(error))
        trace.fail(error)
        raise


def execute_bronze(
    input_files: list[str],
    sttm_path: str,
    run_id: str,
    task_description: str,
) -> list[str]:
    """Run the Bronze ingestion agent and return its Parquet output paths."""
    return _run_bronze_agent(input_files, sttm_path, run_id, task_description)
