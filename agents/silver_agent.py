import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

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
    SILVER_DIR,
)
from core.observability import AgentTrace


SILVER_AGENT_PROMPT = """
You are a Silver-layer data cleansing agent.
You MUST call both tools in this exact order: inspect_task_tool, then silver_ingestion_tool.
Do not just plan; actually execute the cleansing by calling silver_ingestion_tool.
Planning without executing is FAILURE. After execution, verify that the requested Parquet files exist.
""".strip()


def _normalize_sttm(sttm_path: str) -> pd.DataFrame:
    sttm = pd.read_csv(sttm_path)
    sttm.columns = [str(column).strip().lower() for column in sttm.columns]
    return sttm


def _inspect_task(input_files: list[str], sttm_path: str) -> dict[str, Any]:
    """Preview Bronze Parquet schemas and normalized Silver STTM rules."""
    previews: dict[str, Any] = {}
    for input_file in input_files:
        frame = pd.read_parquet(input_file)
        previews[str(input_file)] = {
            "shape": [int(frame.shape[0]), int(frame.shape[1])],
            "columns": [str(column) for column in frame.columns],
            "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
            "sample": frame.head(3).to_dict(orient="records"),
        }

    sttm = _normalize_sttm(sttm_path)
    return {
        "input_files": previews,
        "sttm_path": sttm_path,
        "sttm_shape": [int(sttm.shape[0]), int(sttm.shape[1])],
        "sttm_rules": sttm.to_dict(orient="records"),
    }


def _rule_text(rule: pd.Series) -> str:
    return " ".join(
        str(rule.get(field, ""))
        for field in ("transformation_type", "transformation_logic")
    ).lower()


def _apply_null_rule(frame: pd.DataFrame, column: str, text: str) -> pd.DataFrame:
    if "dropna" in text or "drop null" in text:
        return frame.dropna(subset=[column])
    if "fillna" not in text and "fill null" not in text and "impute" not in text:
        return frame

    if "median" in text:
        value = frame[column].median() if pd.api.types.is_numeric_dtype(frame[column]) else None
    elif "mean" in text:
        value = frame[column].mean() if pd.api.types.is_numeric_dtype(frame[column]) else None
    elif "mode" in text:
        modes = frame[column].mode(dropna=True)
        value = modes.iloc[0] if not modes.empty else None
    else:
        marker = text.split("constant", 1)[1].strip(" :='") if "constant" in text else "unknown"
        value = marker.split()[0] if marker else "unknown"

    if value is not None:
        frame[column] = frame[column].fillna(value)
    return frame


def _cast_column(frame: pd.DataFrame, column: str, text: str) -> pd.DataFrame:
    if any(token in text for token in ("integer", "int", "whole number")):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").round().astype("Int64")
    elif any(token in text for token in ("float", "decimal", "numeric")):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    elif any(token in text for token in ("date", "datetime", "timestamp")):
        parsed = pd.to_datetime(frame[column], errors="coerce", format="mixed")
        frame[column] = parsed.dt.strftime("%Y-%m-%d")
    return frame


def _normalize_text(frame: pd.DataFrame, column: str, text: str) -> pd.DataFrame:
    if not pd.api.types.is_object_dtype(frame[column]) and not pd.api.types.is_string_dtype(frame[column]):
        return frame
    values = frame[column].astype("string")
    if "lowercase" in text or "lower case" in text:
        frame[column] = values.str.lower()
    elif "uppercase" in text or "upper case" in text:
        frame[column] = values.str.upper()
    elif "title case" in text or "titlecase" in text:
        frame[column] = values.str.title()
    if "strip" in text or "trim" in text:
        frame[column] = frame[column].astype("string").str.strip()
    return frame


def _rules_for_file(sttm: pd.DataFrame, input_file: str) -> pd.DataFrame:
    input_path = Path(input_file)
    candidates = {input_path.name.lower(), input_path.stem.lower()}
    if input_path.stem.lower().endswith("_bronze"):
        candidates.add(input_path.stem[:-7].lower())
    if "source_table" not in sttm.columns:
        return sttm
    matches = sttm[sttm["source_table"].astype(str).str.strip().str.lower().isin(candidates)]
    return matches if not matches.empty else sttm


def _apply_silver_rules(
    input_files: list[str],
    sttm_path: str,
    run_id: str,
) -> list[str]:
    """Clean Bronze Parquet files and write approved Silver Parquet outputs."""
    del run_id
    sttm = _normalize_sttm(sttm_path)
    output_paths: list[str] = []

    for input_file in input_files:
        frame = pd.read_parquet(input_file)
        rules = _rules_for_file(sttm, input_file)
        approved_columns: list[str] = []

        for _, rule in rules.iterrows():
            source_column = str(rule.get("source_column", "")).strip()
            target_column = str(rule.get("target_column", "")).strip()
            if not source_column or not target_column or source_column not in frame.columns:
                continue

            if source_column != target_column:
                frame = frame.rename(columns={source_column: target_column})
            text = _rule_text(rule)
            frame = _apply_null_rule(frame, target_column, text)
            frame = _cast_column(frame, target_column, text)
            frame = _normalize_text(frame, target_column, text)
            if target_column not in approved_columns:
                approved_columns.append(target_column)

        frame = frame.drop_duplicates().reset_index(drop=True)
        if approved_columns:
            join_keys = [
                column
                for column in frame.columns
                if column.lower().endswith("_id") and column not in approved_columns
            ]
            frame = frame[
                [column for column in approved_columns + join_keys if column in frame.columns]
            ]

        table_name = Path(input_file).stem
        if table_name.endswith("_bronze"):
            table_name = table_name[:-7]
        surrogate_key = f"pk_{table_name}_silver_id"
        frame.insert(0, surrogate_key, range(1, len(frame) + 1))

        output_path = SILVER_DIR / f"{table_name}_silver.parquet"
        frame.to_parquet(output_path, index=False)
        output_paths.append(str(output_path))

    return output_paths


def _make_silver_tools(input_files: list[str], sttm_path: str, run_id: str):
    """Create tools scoped to one Silver cleansing task."""
    @tool("inspect_task_tool")
    def inspect_task_tool() -> str:
        """Preview Bronze Parquet schemas and Silver STTM rules."""
        return json.dumps(_inspect_task(input_files, sttm_path), default=str)

    @tool("silver_ingestion_tool")
    def silver_ingestion_tool() -> str:
        """Execute Silver cleansing and write Parquet outputs."""
        return json.dumps(_apply_silver_rules(input_files, sttm_path, run_id))

    return [inspect_task_tool, silver_ingestion_tool]


def _make_llm():
    """Build the configured chat model for Silver cleansing."""
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


def _run_silver_agent(
    input_files: list[str],
    sttm_path: str,
    run_id: str,
    task_description: str,
) -> list[str]:
    trace = AgentTrace("silver_agent", run_id).set_input(
        input_files=input_files,
        sttm_path=sttm_path,
        task_description=task_description,
    )
    try:
        agent = create_react_agent(
            model=_make_llm(),
            tools=_make_silver_tools(input_files, sttm_path, run_id),
            prompt=SILVER_AGENT_PROMPT,
        )
        result = agent.invoke({
            "messages": [
                HumanMessage(
                    content=(
                        f"Cleanse these Bronze files: {', '.join(input_files)}. "
                        f"Use STTM rules at {sttm_path}. Task: {task_description}"
                    )
                )
            ]
        })
        messages = result.get("messages", [])
        trace.extract_from_messages(messages)
        expected_paths = []
        for input_file in input_files:
            table_name = Path(input_file).stem.removesuffix("_bronze")
            expected_paths.append(str(SILVER_DIR / f"{table_name}_silver.parquet"))
        missing = [path for path in expected_paths if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"Silver ingestion did not create: {missing}")
        trace.set_output(output_paths=expected_paths).complete()
        return expected_paths
    except Exception as error:
        trace.fail(error)
        raise


def execute_silver(
    input_files: list[str],
    sttm_path: str,
    run_id: str,
    task_description: str,
) -> list[str]:
    """Run the Silver cleansing agent and return its Parquet output paths."""
    return _run_silver_agent(input_files, sttm_path, run_id, task_description)
