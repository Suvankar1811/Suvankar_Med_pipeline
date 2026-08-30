import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import plotly.express as px
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
    REPORTS_DIR,
)
from core.observability import AgentTrace


REPORTER_AGENT_PROMPT = """
You are a business analytics reporter specializing in Gold data.
Call tools in this exact order: inspect_gold_tables_tool -> load_gold_data_tool -> execute_query_tool -> submit_analysis_tool.
After reviewing query results, call submit_analysis_tool with valid JSON containing:
- direct_answer: a concise answer to the business question
- charts: an array of chart specs, each with type, title, x_column, and y_column
- detailed_analysis: a clear explanation of findings and limitations
Use execute_query_tool to answer the question with SQL before submitting analysis.
""".strip()


def _inspect_gold_tables(gold_files: list[str]) -> dict[str, Any]:
    """Preview Gold Parquet schemas and sample rows without DuckDB."""
    tables: dict[str, Any] = {}
    for gold_file in gold_files:
        frame = pd.read_parquet(gold_file)
        tables[Path(gold_file).stem] = {
            "path": gold_file,
            "shape": [int(frame.shape[0]), int(frame.shape[1])],
            "columns": [str(column) for column in frame.columns],
            "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
            "sample_rows": frame.head(3).to_dict(orient="records"),
        }
    return tables


def generate_chart_from_spec(
    df: pd.DataFrame,
    chart_spec: dict[str, Any],
    chart_id: str,
) -> str:
    """Render a supported Plotly chart specification as embeddable HTML."""
    chart_type = str(chart_spec.get("type", "bar")).lower()
    title = str(chart_spec.get("title", "Analytics chart"))
    x_column = chart_spec.get("x_column")
    y_column = chart_spec.get("y_column")
    if x_column not in df.columns:
        return ""
    if chart_type != "pie" and y_column not in df.columns:
        return ""

    chart_builders = {
        "bar": px.bar,
        "line": px.line,
        "scatter": px.scatter,
    }
    if chart_type == "pie":
        if y_column not in df.columns:
            return ""
        figure = px.pie(df, names=x_column, values=y_column, title=title)
    elif chart_type in chart_builders:
        figure = chart_builders[chart_type](df, x=x_column, y=y_column, title=title)
    else:
        return ""

    figure.update_layout(template="plotly_white", title_x=0.02)
    return figure.to_html(full_html=False, include_plotlyjs=False, div_id=chart_id)


def _make_reporter_tools(gold_files: list[str], run_id: str):
    """Create reporting tools and return them with the shared scratchpad and connection."""
    connection = duckdb.connect()
    scratchpad: dict[str, Any] = {"run_id": run_id, "query_results": [], "analysis": {}}

    @tool("inspect_gold_tables_tool")
    def inspect_gold_tables_tool() -> str:
        """Inspect Gold table schemas and sample rows."""
        result = _inspect_gold_tables(gold_files)
        scratchpad["inspection"] = result
        return json.dumps(result, default=str)

    @tool("load_gold_data_tool")
    def load_gold_data_tool() -> str:
        """Load each Gold Parquet file into DuckDB tables and return the catalog."""
        catalog: dict[str, Any] = {}
        for gold_file in gold_files:
            table_name = _safe_table_name(Path(gold_file).stem)
            escaped_file = str(gold_file).replace("'", "''")
            connection.execute(
                f'''CREATE OR REPLACE VIEW "{table_name}" AS
                    SELECT * FROM read_parquet('{escaped_file}')''',
            )
            columns = connection.execute(f'DESCRIBE "{table_name}"').fetchall()
            catalog[table_name] = {
                "source_file": gold_file,
                "columns": [{"name": row[0], "type": row[1]} for row in columns],
            }
        scratchpad["catalog"] = catalog
        return json.dumps(catalog, default=str)

    @tool("execute_query_tool")
    def execute_query_tool(sql_query: str) -> str:
        """Execute a read-only SQL query against loaded Gold views."""
        try:
            if not sql_query.strip().lower().startswith(("select", "with", "from")):
                raise ValueError("Only read-only SELECT queries are allowed")
            result = connection.execute(sql_query).fetchdf()
            records = result.to_dict(orient="records")
            scratchpad["sql_query"] = sql_query
            scratchpad["query_results"] = records
            scratchpad["query_columns"] = [str(column) for column in result.columns]
            return json.dumps(records, default=str)
        except Exception as error:
            return json.dumps({"error": str(error)})

    @tool("submit_analysis_tool")
    def submit_analysis_tool(analysis_json: str) -> str:
        """Store the final analysis JSON submitted by the reporting agent."""
        try:
            analysis = json.loads(analysis_json)
        except json.JSONDecodeError as error:
            raise ValueError("analysis_json must be valid JSON") from error
        required = {"direct_answer", "charts", "detailed_analysis"}
        missing = required.difference(analysis)
        if missing:
            raise ValueError(f"Analysis is missing keys: {sorted(missing)}")
        scratchpad["analysis"] = analysis
        return "Analysis submitted successfully."

    return (
        [
            inspect_gold_tables_tool,
            load_gold_data_tool,
            execute_query_tool,
            submit_analysis_tool,
        ],
        scratchpad,
        connection,
    )


def _safe_table_name(name: str) -> str:
    return "".join(character if character.isalnum() or character == "_" else "_" for character in name)


def _make_llm():
    """Build the configured chat model for narrative reporting."""
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=0.5,
            reasoning_format="hidden",
            reasoning_effort="low",
        )

    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(api_key=OPENAI_API_KEY, model=OPENAI_MODEL, temperature=0.5)

    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model=GEMINI_MODEL, temperature=0.5)

    if LLM_PROVIDER == "github":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=GITHUB_TOKEN,
            model=GITHUB_MODEL,
            base_url=GITHUB_BASE_URL,
            temperature=0.5,
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


def _run_reporter(
    gold_files: list[str],
    business_intent: str,
    run_id: str,
    task_description: str,
) -> str:
    trace = AgentTrace("reporter", run_id).set_input(
        gold_files=gold_files,
        business_intent=business_intent,
        task_description=task_description,
    )
    tools, scratchpad, connection = _make_reporter_tools(gold_files, run_id)
    try:
        agent = create_react_agent(model=_make_llm(), tools=tools, prompt=REPORTER_AGENT_PROMPT)
        result = agent.invoke({
            "messages": [
                HumanMessage(
                    content=(
                        f"Answer this business question: {business_intent}\n"
                        f"Task: {task_description}\nGold files: {', '.join(gold_files)}"
                    )
                )
            ]
        })
        trace.extract_from_messages(result.get("messages", []))
        analysis = scratchpad.get("analysis", {})
        if not analysis:
            raise ValueError("Reporter agent did not submit analysis")

        query_frame = pd.DataFrame(scratchpad.get("query_results", []))
        charts_html = []
        for index, chart_spec in enumerate(analysis.get("charts", [])):
            charts_html.append(generate_chart_from_spec(query_frame, chart_spec, f"chart_{index}"))

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_stem = f"report_{timestamp}_{run_id[:8]}"
        json_path = REPORTS_DIR / f"{report_stem}.json"
        html_path = REPORTS_DIR / f"{report_stem}.html"
        report_data = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "business_question": business_intent,
            "task_description": task_description,
            "sql_query": scratchpad.get("sql_query", ""),
            "analysis": analysis,
            "query_results": scratchpad.get("query_results", []),
        }
        with json_path.open("w", encoding="utf-8") as file:
            json.dump(report_data, file, indent=2, default=str)
        html_path.write_text(_render_report(report_data, charts_html), encoding="utf-8")
        trace.set_output(html_path=str(html_path), json_path=str(json_path)).complete()
        return str(html_path)
    except Exception as error:
        trace.fail(error)
        raise
    finally:
        connection.close()


def _render_report(report_data: dict[str, Any], charts_html: list[str]) -> str:
    analysis = report_data["analysis"]
    chart_section = "\n".join(charts_html) or "<p>No charts were submitted.</p>"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Retail Analytics Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root {{ --ink: #17212b; --muted: #62707c; --accent: #0c7c86; --warm: #f0a35b; --paper: #fffdf8; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; color: var(--ink); font-family: Georgia, 'Times New Roman', serif; background: linear-gradient(135deg, #e8f1ef, #fff8ed); }}
main {{ max-width: 1120px; margin: 0 auto; padding: 48px 24px; }}
header {{ padding: 42px; color: white; background: linear-gradient(120deg, #123c45, #0c7c86 65%, #e39151); border-radius: 18px; box-shadow: 0 18px 40px rgba(18,60,69,.18); }}
h1 {{ margin: 0 0 12px; font-size: clamp(2rem, 5vw, 4rem); }}
.eyebrow {{ margin: 0 0 14px; font: 700 .78rem/1.2 Arial, sans-serif; letter-spacing: .16em; text-transform: uppercase; opacity: .8; }}
section {{ margin-top: 26px; padding: 28px; background: rgba(255,253,248,.9); border: 1px solid rgba(12,124,134,.14); border-radius: 14px; }}
h2 {{ margin-top: 0; color: #15545c; }}
.answer {{ font-size: 1.35rem; line-height: 1.5; }}
pre {{ overflow-x: auto; padding: 18px; color: #e8f1ef; background: #17212b; border-radius: 10px; }}
footer {{ padding: 28px 0 0; color: var(--muted); text-align: center; font: .85rem Arial, sans-serif; }}
.plot {{ margin-top: 20px; overflow-x: auto; }}
</style>
</head>
<body><main>
<header><p class="eyebrow">Retail intelligence brief</p><h1>Analytics Report</h1><p>{html.escape(str(report_data['business_question']))}</p></header>
<section><h2>Answer</h2><p class="answer">{html.escape(str(analysis.get('direct_answer', '')))}</p></section>
<section><h2>Approach</h2><pre>{html.escape(str(report_data.get('sql_query', '')))}</pre><p>{html.escape(str(analysis.get('detailed_analysis', '')))}</p></section>
<section><h2>Charts</h2><div class="plot">{chart_section}</div></section>
<footer>Generated by IDAMP</footer>
</main></body></html>"""


def generate_report(
    gold_files: list[str],
    business_intent: str,
    run_id: str,
    task_description: str,
) -> str:
    """Generate HTML and JSON reports from Gold Parquet files."""
    return _run_reporter(gold_files, business_intent, run_id, task_description)
