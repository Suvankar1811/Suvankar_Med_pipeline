import json
import re
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
    GOOGLE_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    STTM_DIR,
)
from core.observability import AgentTrace


STTM_COLUMNS = [
    "source_schema",
    "source_table",
    "source_column",
    "target_schema",
    "target_table",
    "target_column",
    "transformation_type",
    "transformation_logic",
]

STTM_AGENT_PROMPT = """
You generate Source-to-Target Mapping (STTM) rules for Bronze, Silver, or Gold data layers.
Follow this sequence: THINK -> INSPECT -> PLAN -> ACT -> VERIFY.
First call inspect_context_tool. Then call the appropriate generation tool exactly once.
For Bronze, map every source column and add _load_timestamp and _source_file metadata columns.
For Silver, add a surrogate key first, then apply cleansing rules.
For Gold, add a surrogate key first, then define joins and aggregations.
The generation tool writes the STTM CSV. Verify that the result is non-empty and return its path.
""".strip()


def _prepare_bronze_context(profile_path: str) -> dict[str, Any]:
    """Load the profiler output used to generate Bronze mappings."""
    with Path(profile_path).open("r", encoding="utf-8") as file:
        return json.load(file)


def _parquet_metadata(file_paths: list[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for file_path in file_paths:
        frame = pd.read_parquet(file_path)
        metadata[Path(file_path).name] = {
            "path": str(file_path),
            "columns": [str(column) for column in frame.columns],
            "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
            "row_count": int(len(frame)),
        }
    return metadata


def _prepare_silver_context(
    bronze_paths: list[str],
    bronze_sttm_path: str,
) -> dict[str, Any]:
    """Load Bronze schemas and retain columns approved by the Bronze STTM."""
    bronze_sttm = pd.read_csv(bronze_sttm_path)
    approved_columns = {
        str(column)
        for column in bronze_sttm.get("target_column", pd.Series(dtype=str)).dropna()
    }
    metadata = _parquet_metadata(bronze_paths)
    for file_metadata in metadata.values():
        file_metadata["approved_columns"] = [
            column for column in file_metadata["columns"] if column in approved_columns
        ]
    return {
        "bronze_sttm_path": bronze_sttm_path,
        "approved_columns": sorted(approved_columns),
        "files": metadata,
    }


def _prepare_gold_context(
    silver_paths: list[str],
    silver_sttm_path: str,
) -> dict[str, Any]:
    """Load Silver schemas and the Silver mappings used for Gold design."""
    silver_sttm = pd.read_csv(silver_sttm_path)
    approved_columns = {
        str(column)
        for column in silver_sttm.get("target_column", pd.Series(dtype=str)).dropna()
    }
    metadata = _parquet_metadata(silver_paths)
    for file_metadata in metadata.values():
        file_metadata["approved_columns"] = [
            column for column in file_metadata["columns"] if column in approved_columns
        ]
    return {
        "silver_sttm_path": silver_sttm_path,
        "approved_columns": sorted(approved_columns),
        "files": metadata,
    }


def _parse_json_array(raw: str) -> list[dict[str, Any]]:
    """Extract a JSON array of STTM rows from raw LLM text output."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.IGNORECASE | re.DOTALL)
    if "```json" in raw:
        raw = raw.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in raw:
        raw = raw.split("```", 1)[1].split("```", 1)[0]
    raw = raw.strip()
    start = raw.find("[")
    if start == -1:
        return []
    try:
        rows, _ = json.JSONDecoder().raw_decode(raw[start:])
    except (json.JSONDecodeError, ValueError):
        return []
    return rows if isinstance(rows, list) and all(isinstance(row, dict) for row in rows) else []


def _extract_sttm_rows(messages: list[Any]) -> list[dict[str, Any]]:
    """Extract the last valid non-empty STTM JSON array from agent messages."""
    for message in reversed(messages):
        content = getattr(message, "content", message)
        rows = _parse_json_array(content if isinstance(content, str) else str(content))
        if rows:
            return rows
    return []


def _invoke_llm_for_sttm_rows(prompt: str, layer: str) -> list[dict[str, Any]]:
    """Invoke the LLM twice at most, failing loudly when no rows are returned."""
    llm = _make_llm()
    last_raw = ""
    for attempt in range(2):
        current_prompt = (
            prompt
            if attempt == 0
            else prompt + "\n\nReturn ONLY the JSON array. No other text, no explanation."
        )
        response = llm.invoke(current_prompt)
        last_raw = response.content if hasattr(response, "content") else str(response)
        rows = _parse_json_array(last_raw)
        if rows:
            return rows
    raise ValueError(
        f"{layer} STTM generation failed: LLM did not return a valid, non-empty JSON array "
        f"after 2 attempts. Raw response (truncated): {last_raw[:500]!r}"
    )


def _fallback_silver_rows(context: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file_name, metadata in context.get("files", {}).items():
        source_table = Path(file_name).stem.removesuffix("_bronze")
        rows.append({
            "source_schema": "bronze",
            "source_table": source_table,
            "source_column": "ROW_NUMBER()",
            "target_schema": "silver",
            "target_table": source_table,
            "target_column": f"sttm_{source_table}_sk",
            "transformation_type": "surrogate_key",
            "transformation_logic": "ROW_NUMBER()",
        })
        for column in metadata.get("approved_columns", []):
            rows.append({
                "source_schema": "bronze",
                "source_table": source_table,
                "source_column": column,
                "target_schema": "silver",
                "target_table": source_table,
                "target_column": column,
                "transformation_type": "identity",
                "transformation_logic": "copy as-is",
            })
    return rows


def _make_llm():
    """Build the configured chat model for STTM generation."""
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


def _generation_prompt(layer: str, context: dict[str, Any], business_intent: str = "") -> str:
    return f"""
Generate a compact, non-empty JSON array of STTM rows for the {layer} layer.
Each row must contain exactly these keys: {', '.join(STTM_COLUMNS)}.
Return one minified JSON array only: no Markdown fences, reasoning, table, or explanation.
Use one row per required mapping; keep transformation_logic short.
Requirements: Bronze maps every column and adds _load_timestamp/_source_file; Silver adds a surrogate key first, then cleansing; Gold adds a surrogate key first, then joins/aggregations.
Business intent: {business_intent}
Context (use only the column names and mappings needed for the rows):
{json.dumps(context, indent=2, default=str)}
""".strip()


def _write_sttm(rows: list[dict[str, Any]], output_path: Path) -> str:
    frame = pd.DataFrame(rows)
    for column in STTM_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    frame = frame[STTM_COLUMNS]
    frame.to_csv(output_path, index=False)
    return str(output_path)


def _make_sttm_tools(
    layer: str,
    context: dict[str, Any],
    output_path: str,
    run_id: str,
    business_intent: str = "",
):
    """Create context and one-shot generation tools for the current STTM run."""
    del run_id
    destination = Path(output_path)

    @tool("inspect_context_tool")
    def inspect_context_tool() -> str:
        """Inspect the prepared context for the current STTM layer."""
        return json.dumps(context, default=str)

    @tool("generate_bronze_sttm_tool")
    def generate_bronze_sttm_tool() -> str:
        """Generate and save Bronze STTM rules."""
        rows = _invoke_llm_for_sttm_rows(
            _generation_prompt("Bronze", context, business_intent),
            "Bronze",
        )
        return _write_sttm(rows, destination)

    @tool("generate_silver_sttm_tool")
    def generate_silver_sttm_tool() -> str:
        """Generate and save Silver STTM rules."""
        try:
            rows = _invoke_llm_for_sttm_rows(
                _generation_prompt("Silver", context, business_intent),
                "Silver",
            )
        except ValueError:
            rows = _fallback_silver_rows(context)
        if not rows:
            raise ValueError("Silver STTM fallback could not create any mappings")
        return _write_sttm(rows, destination)

    @tool("generate_gold_sttm_tool")
    def generate_gold_sttm_tool() -> str:
        """Generate and save Gold STTM rules."""
        rows = _invoke_llm_for_sttm_rows(
            _generation_prompt("Gold", context, business_intent),
            "Gold",
        )
        return _write_sttm(rows, destination)

    return [
        inspect_context_tool,
        generate_bronze_sttm_tool,
        generate_silver_sttm_tool,
        generate_gold_sttm_tool,
    ]


def _run_sttm_agent(
    layer: str,
    context: dict[str, Any],
    output_path: str,
    run_id: str,
    task_description: str,
    business_intent: str = "",
) -> str:
    trace = AgentTrace("sttm_generator", run_id).set_input(
        layer=layer,
        task_description=task_description,
    )
    try:
        tools = _make_sttm_tools(
            layer,
            context,
            output_path,
            run_id,
            business_intent,
        )
        agent = create_react_agent(model=_make_llm(), tools=tools, prompt=STTM_AGENT_PROMPT)
        result = agent.invoke({
            "messages": [
                HumanMessage(
                    content=f"Generate {layer} STTM rules. Task: {task_description}. "
                    f"The output path is {output_path}."
                )
            ]
        })
        messages = result.get("messages", [])
        trace.extract_from_messages(messages)
        if not Path(output_path).is_file():
            raise FileNotFoundError(f"{layer} STTM tool did not create {output_path}")
        trace.set_output(sttm_path=output_path).complete()
        return output_path
    except Exception as error:
        trace.fail(error)
        raise


def _output_path(layer: str, run_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return str(STTM_DIR / f"sttm_{layer.lower()}_{timestamp}_{run_id[:8]}.csv")


def generate_bronze_sttm(profile_path: str, run_id: str, task_description: str) -> str:
    """Generate Bronze mappings from a profile JSON file."""
    return _run_sttm_agent(
        "Bronze",
        _prepare_bronze_context(profile_path),
        _output_path("Bronze", run_id),
        run_id,
        task_description,
    )


def generate_silver_sttm(
    bronze_output_paths: list[str],
    bronze_sttm_path: str,
    run_id: str,
    task_description: str,
) -> str:
    """Generate Silver mappings from Bronze Parquet files and rules."""
    return _run_sttm_agent(
        "Silver",
        _prepare_silver_context(bronze_output_paths, bronze_sttm_path),
        _output_path("Silver", run_id),
        run_id,
        task_description,
    )


def generate_gold_sttm(
    silver_output_paths: list[str],
    silver_sttm_path: str,
    business_intent: str,
    run_id: str,
    task_description: str,
) -> str:
    """Generate Gold mappings from Silver Parquet files and business intent."""
    return _run_sttm_agent(
        "Gold",
        _prepare_gold_context(silver_output_paths, silver_sttm_path),
        _output_path("Gold", run_id),
        run_id,
        task_description,
        business_intent,
    )
