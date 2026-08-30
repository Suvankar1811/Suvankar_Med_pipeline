import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from core.config import (
    GITHUB_BASE_URL,
    GITHUB_MODEL,
    GITHUB_TOKEN,
    GOOGLE_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    PROFILES_DIR,
)
from core.observability import AgentTrace


PROFILER_AGENT_PROMPT = """
You are a Data Analyst specializing in data profiling.

Follow this sequence for every request: THINK -> INSPECT -> PLAN -> ACT -> VERIFY.
First call inspect_files_tool to preview the files. Then call profiler_tool to get full statistics.
Use the tool results to identify semantic meanings, possible join keys, and data quality issues.
Return only valid JSON with exactly these keys:
- semantic_meanings: an object mapping each important column to its business meaning
- join_keys: an array of likely keys that can join the datasets
- quality_notes: an array of data quality observations
""".strip()


def _read_csv(file_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(file_path)


def _inspect_files(file_paths: list[str]) -> dict[str, Any]:
    """Preview the structure and representative values of each CSV file."""
    inspected: dict[str, Any] = {}
    for file_path in file_paths:
        frame = _read_csv(file_path)
        inspected[str(file_path)] = {
            "shape": [int(frame.shape[0]), int(frame.shape[1])],
            "columns": [str(column) for column in frame.columns],
            "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
            "sample_values": {
                str(column): frame[column].dropna().head(3).tolist()
                for column in frame.columns
            },
        }
    return inspected


def _compute_stats(file_paths: list[str]) -> dict[str, Any]:
    """Compute column-level statistics for each CSV file."""
    profiles: dict[str, Any] = {}
    for file_path in file_paths:
        frame = _read_csv(file_path)
        columns: dict[str, Any] = {}
        row_count = len(frame)

        for column in frame.columns:
            series = frame[column]
            non_null = series.dropna()
            stats: dict[str, Any] = {
                "dtype": str(series.dtype),
                "null_count": int(series.isna().sum()),
                "null_pct": round(float(series.isna().mean() * 100), 2) if row_count else 0.0,
                "unique_count": int(series.nunique(dropna=True)),
            }

            if pd.api.types.is_numeric_dtype(series):
                stats.update({
                    "min": _json_value(series.min()) if not non_null.empty else None,
                    "max": _json_value(series.max()) if not non_null.empty else None,
                    "mean": _json_value(series.mean()) if not non_null.empty else None,
                })
            else:
                stats["sample_values"] = [str(value) for value in non_null.head(3).tolist()]

            columns[str(column)] = stats

        profiles[str(file_path)] = {
            "shape": [int(frame.shape[0]), int(frame.shape[1])],
            "columns": columns,
        }
    return profiles


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def _make_profiler_tools(file_paths: list[str], run_id: str):
    """Create tools scoped to the datasets being profiled."""
    del run_id

    @tool("inspect_files_tool")
    def inspect_files_tool() -> str:
        """Preview CSV shapes, columns, dtypes, and sample values."""
        return json.dumps(_inspect_files(file_paths), default=str)

    @tool("profiler_tool")
    def profiler_tool() -> str:
        """Compute full column-level statistics for the CSV files."""
        return json.dumps(_compute_stats(file_paths), default=str)

    return [inspect_files_tool, profiler_tool]


def _make_llm():
    """Build the configured chat model for the profiler agent."""
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


def _run_profile(file_paths: list[str], run_id: str, task_description: str) -> str:
    trace = AgentTrace("profiler", run_id).set_input(
        file_paths=file_paths,
        task_description=task_description,
    )

    try:
        tools = _make_profiler_tools(file_paths, run_id)
        agent = create_react_agent(
            model=_make_llm(),
            tools=tools,
            prompt=PROFILER_AGENT_PROMPT,
        )
        request = (
            f"Profile these CSV files: {', '.join(file_paths)}\n"
            f"Business task: {task_description}"
        )
        result = agent.invoke({"messages": [HumanMessage(content=request)]})
        messages = result.get("messages", [])
        trace.extract_from_messages(messages)

        agent_output = messages[-1].content if messages else ""
        combined_profile = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "files": _compute_stats(file_paths),
            "agent_analysis": _parse_json(agent_output),
        }
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        profile_path = PROFILES_DIR / f"profile_combined_{timestamp}.json"
        with profile_path.open("w", encoding="utf-8") as file:
            json.dump(combined_profile, file, indent=2, default=str)

        trace.set_output(profile_path=str(profile_path)).complete()
        return str(profile_path)
    except Exception as error:
        trace.fail(error)
        raise


def profile_dataset(file_path: str, run_id: str, task_description: str) -> str:
    """Profile one CSV file and return the combined profile path."""
    return _run_profile([file_path], run_id, task_description)


def profile_multiple_datasets(
    file_paths: list[str],
    run_id: str,
    task_description: str,
) -> str:
    """Profile multiple CSV files and return the combined profile path."""
    return _run_profile(file_paths, run_id, task_description)


def _parse_json(content: Any) -> Any:
    if not isinstance(content, str):
        return content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"raw_response": content}
