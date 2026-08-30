import json
from pathlib import Path

import pandas as pd
import pytest

import agents.reporter as reporter_module
from agents.reporter import _make_reporter_tools, generate_chart_from_spec


ANALYSIS_JSON = {
    "direct_answer": "West generated the most sales.",
    "charts": [
        {
            "type": "bar",
            "title": "Sales by Region",
            "x_column": "region",
            "y_column": "total_sales",
        }
    ],
    "detailed_analysis": "Sales were compared by region using the Gold table.",
}


def _mock_reporter_agent(tools, analysis_json=ANALYSIS_JSON):
    """Return a mock agent that executes the reporter tools in order."""
    class MockAgent:
        def invoke(self, payload):
            tools[0].invoke({})
            tools[1].invoke({})
            tools[2].invoke({
                "sql_query": "SELECT region, total_sales FROM regional_sales ORDER BY total_sales DESC"
            })
            tools[3].invoke({"analysis_json": json.dumps(analysis_json)})
            return {"messages": payload.get("messages", [])}

    return MockAgent()


class TestGenerateChartFromSpec:
    @pytest.fixture
    def chart_data(self):
        return pd.DataFrame({
            "region": ["East", "West"],
            "total_sales": [20, 35],
        })

    def test_bar_chart_returns_html(self, chart_data):
        chart_html = generate_chart_from_spec(
            chart_data,
            {
                "type": "bar",
                "title": "Sales by Region",
                "x_column": "region",
                "y_column": "total_sales",
            },
            "bar_chart",
        )

        assert "<div" in chart_html
        assert "bar_chart" in chart_html

    def test_pie_chart_returns_html(self, chart_data):
        chart_html = generate_chart_from_spec(
            chart_data,
            {
                "type": "pie",
                "title": "Regional Share",
                "x_column": "region",
                "y_column": "total_sales",
            },
            "pie_chart",
        )

        assert "<div" in chart_html
        assert "pie_chart" in chart_html

    def test_unknown_chart_type_returns_empty(self, chart_data):
        chart_html = generate_chart_from_spec(
            chart_data,
            {"type": "radar", "x_column": "region", "y_column": "total_sales"},
            "unknown_chart",
        )

        assert chart_html == ""

    def test_bad_column_does_not_raise(self, chart_data):
        chart_html = generate_chart_from_spec(
            chart_data,
            {"type": "bar", "x_column": "missing", "y_column": "total_sales"},
            "bad_column_chart",
        )

        assert chart_html == ""


class TestMakeReporterTools:
    def test_load_tool_returns_catalog(self, tmp_path):
        gold_file = tmp_path / "regional_sales.parquet"
        pd.DataFrame({"region": ["East", "West"], "total_sales": [20, 35]}).to_parquet(
            gold_file,
            index=False,
        )
        tools, scratchpad, connection = _make_reporter_tools([str(gold_file)], "catalog-run")
        try:
            catalog = json.loads(tools[1].invoke({}))
            assert "regional_sales" in catalog
            assert scratchpad["catalog"]["regional_sales"]["source_file"] == str(gold_file)
            assert connection.execute("SELECT COUNT(*) FROM regional_sales").fetchone()[0] == 2
        finally:
            connection.close()

    def test_execute_tool_stores_result_in_scratchpad(self, tmp_path):
        gold_file = tmp_path / "regional_sales.parquet"
        pd.DataFrame({"region": ["East", "West"], "total_sales": [20, 35]}).to_parquet(
            gold_file,
            index=False,
        )
        tools, scratchpad, connection = _make_reporter_tools([str(gold_file)], "query-run")
        try:
            tools[1].invoke({})
            result = json.loads(tools[2].invoke({
                "sql_query": "SELECT region, total_sales FROM regional_sales ORDER BY region"
            }))
            assert len(result) == 2
            assert scratchpad["sql_query"].startswith("SELECT")
            assert scratchpad["query_results"] == result
        finally:
            connection.close()

    def test_execute_tool_returns_error_on_bad_sql(self, tmp_path):
        gold_file = tmp_path / "regional_sales.parquet"
        pd.DataFrame({"region": ["East"], "total_sales": [20]}).to_parquet(gold_file, index=False)
        tools, _, connection = _make_reporter_tools([str(gold_file)], "bad-query-run")
        try:
            result = tools[2].invoke({"sql_query": "DROP TABLE regional_sales"})
            error = json.loads(result)
            assert "error" in error
        finally:
            connection.close()
