import json
import re
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
    GOLD_DIR,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)
from core.observability import AgentTrace


GOLD_AGENT_PROMPT = """
You are a Gold-layer analytics data agent.
You MUST call both tools in this exact order: inspect_task_tool, then gold_ingestion_tool.
Do not just plan; actually execute the joins and aggregations by calling gold_ingestion_tool.
Planning without executing is FAILURE. After execution, verify that the requested Gold Parquet files exist.
""".strip()


def _normalize_sttm(sttm_path: str) -> pd.DataFrame:
    sttm = pd.read_csv(sttm_path)
    sttm.columns = [str(column).strip().lower() for column in sttm.columns]
    return sttm


def _source_name(input_file: str) -> str:
    name = Path(input_file).stem
    return name[:-7] if name.endswith("_silver") else name


def _load_silver_files(input_files: list[str]) -> dict[str, pd.DataFrame]:
    return {_source_name(path): pd.read_parquet(path) for path in input_files}


def _inspect_task(input_files: list[str], sttm_path: str) -> dict[str, Any]:
    """Preview Silver schemas and Gold STTM rules grouped by target table."""
    schemas: dict[str, Any] = {}
    for input_file in input_files:
        frame = pd.read_parquet(input_file)
        schemas[_source_name(input_file)] = {
            "path": input_file,
            "shape": [int(frame.shape[0]), int(frame.shape[1])],
            "columns": [str(column) for column in frame.columns],
            "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        }

    sttm = _normalize_sttm(sttm_path)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for target_table, rules in sttm.groupby("target_table", dropna=False):
        table_name = str(target_table) if pd.notna(target_table) else "default"
        grouped[table_name] = rules.to_dict(orient="records")
    return {"silver_schemas": schemas, "sttm_path": sttm_path, "rules_by_target_table": grouped}


def _rules_for_source(rules: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if "source_table" not in rules.columns:
        return rules
    source_values = rules["source_table"].astype(str).str.strip().str.lower()
    return rules[source_values.isin({source_name.lower(), f"{source_name.lower()}_silver"})]


def _join_sources(frames: list[pd.DataFrame]) -> pd.DataFrame:
    result = frames[0].copy()
    for frame in frames[1:]:
        keys = [column for column in result.columns if column in frame.columns and column.lower().endswith("_id")]
        if not keys:
            raise ValueError("Cannot join Gold sources: no matching _id column was found")
        result = result.merge(frame, how="outer", on=keys[0], suffixes=("", "_right"))
    return result


def _apply_renames(frame: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    for _, rule in rules.iterrows():
        source_column = str(rule.get("source_column", "")).strip()
        target_column = str(rule.get("target_column", "")).strip()
        if source_column in frame.columns and target_column and source_column != target_column:
            frame = frame.rename(columns={source_column: target_column})
    return frame


def _aggregation_operation(text: str) -> str | None:
    for operation in ("sum", "avg", "average", "mean", "count", "max", "min"):
        if re.search(rf"\b{operation}\b", text):
            return "avg" if operation in {"avg", "average", "mean"} else operation
    return None


def _apply_aggregations(frame: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    group_columns: list[str] = []
    aggregations: dict[str, tuple[str, str]] = {}
    for _, rule in rules.iterrows():
        source_column = str(rule.get("source_column", "")).strip()
        target_column = str(rule.get("target_column", "")).strip() or source_column
        text = " ".join(str(rule.get(field, "")) for field in ("transformation_type", "transformation_logic")).lower()
        if "group" in text:
            candidate = target_column if target_column in frame.columns else source_column
            if candidate in frame.columns and candidate not in group_columns:
                group_columns.append(candidate)
        operation = _aggregation_operation(text)
        candidate = source_column if source_column in frame.columns else target_column
        if operation and candidate in frame.columns:
            aggregations[target_column] = (candidate, operation)

    if not aggregations:
        return frame
    if not group_columns:
        group_columns = [column for column in frame.columns if column not in {source for source, _ in aggregations.values()}]
    measure_columns = {source for source, _ in aggregations.values()} | set(aggregations)
    group_columns = [column for column in group_columns if column not in measure_columns]
    if not group_columns:
        return pd.DataFrame([{target: _aggregate(frame[source], operation) for target, (source, operation) in aggregations.items()}])

    grouped = frame.groupby(group_columns, dropna=False)
    result = grouped.size().reset_index(name="_group_count")
    for target, (source, operation) in aggregations.items():
        values = grouped[source].agg(operation if operation != "avg" else "mean").rename(target).reset_index()
        result = result.drop(columns=["_group_count"], errors="ignore").merge(
            values,
            on=group_columns,
            how="left",
        )
    return result


def _aggregate(series: pd.Series, operation: str) -> Any:
    if operation == "count":
        return int(series.count())
    if operation == "avg":
        return series.mean()
    return getattr(series, operation)()


def _apply_gold_rules(
    input_files: list[str],
    sttm_path: str,
    run_id: str,
) -> list[str]:
    """Join and aggregate Silver data into approved Gold Parquet tables."""
    del run_id
    silver_frames = _load_silver_files(input_files)
    sttm = _normalize_sttm(sttm_path)
    if "target_table" not in sttm.columns:
        sttm["target_table"] = next(iter(silver_frames), "gold_table")

    output_paths: list[str] = []
    for target_table_value, target_rules in sttm.groupby("target_table", dropna=False):
        target_table = str(target_table_value) if pd.notna(target_table_value) else "gold_table"
        source_frames: list[pd.DataFrame] = []
        for source_name, frame in silver_frames.items():
            if not _rules_for_source(target_rules, source_name).empty:
                source_frames.append(frame)
        if not source_frames:
            source_frames = list(silver_frames.values())
        if not source_frames:
            continue

        result = _join_sources(source_frames) if len(source_frames) > 1 else source_frames[0].copy()
        result = _apply_renames(result, target_rules)
        result = _apply_aggregations(result, target_rules)

        approved_columns = [
            str(column).strip()
            for column in target_rules.get("target_column", pd.Series(dtype=str)).dropna()
            if str(column).strip() in result.columns
        ]
        if approved_columns:
            result = result.loc[:, list(dict.fromkeys(approved_columns))]
        result.insert(0, "pk_gold_id", range(1, len(result) + 1))

        output_path = GOLD_DIR / f"{target_table}.parquet"
        result.to_parquet(output_path, index=False)
        output_paths.append(str(output_path))

    return output_paths


def _make_gold_tools(input_files: list[str], sttm_path: str, run_id: str):
    """Create tools scoped to one Gold analytics task."""
    @tool("inspect_task_tool")
    def inspect_task_tool() -> str:
        """Preview Silver schemas and Gold rules grouped by target table."""
        return json.dumps(_inspect_task(input_files, sttm_path), default=str)

    @tool("gold_ingestion_tool")
    def gold_ingestion_tool() -> str:
        """Execute Gold joins and aggregations and write Parquet tables."""
        return json.dumps(_apply_gold_rules(input_files, sttm_path, run_id))

    return [inspect_task_tool, gold_ingestion_tool]


def _make_llm():
    """Build the configured chat model for Gold analytics."""
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

        return ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model=GEMINI_MODEL)

    if LLM_PROVIDER == "github":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(api_key=GITHUB_TOKEN, model=GITHUB_MODEL, base_url=GITHUB_BASE_URL)

    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


def _run_gold_agent(
    input_files: list[str],
    sttm_path: str,
    run_id: str,
    task_description: str,
) -> list[str]:
    trace = AgentTrace("gold_agent", run_id).set_input(
        input_files=input_files,
        sttm_path=sttm_path,
        task_description=task_description,
    )
    try:
        agent = create_react_agent(
            model=_make_llm(),
            tools=_make_gold_tools(input_files, sttm_path, run_id),
            prompt=GOLD_AGENT_PROMPT,
        )
        result = agent.invoke({
            "messages": [
                HumanMessage(
                    content=(
                        f"Create Gold tables from {', '.join(input_files)} using {sttm_path}. "
                        f"Task: {task_description}"
                    )
                )
            ]
        })
        messages = result.get("messages", [])
        trace.extract_from_messages(messages)
        sttm = _normalize_sttm(sttm_path)
        expected_tables = [str(value) for value in sttm["target_table"].dropna().unique()]
        expected_paths = [str(GOLD_DIR / f"{table}.parquet") for table in expected_tables]
        missing = [path for path in expected_paths if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"Gold ingestion did not create: {missing}")
        trace.set_output(output_paths=expected_paths).complete()
        return expected_paths
    except Exception as error:
        trace.fail(error)
        raise


def execute_gold(
    input_files: list[str],
    sttm_path: str,
    run_id: str,
    task_description: str,
) -> list[str]:
    """Run the Gold analytics agent and return its Parquet output paths."""
    return _run_gold_agent(input_files, sttm_path, run_id, task_description)
