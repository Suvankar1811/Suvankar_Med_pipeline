"""Bronze test strategy: test _apply_bronze_rules directly because it is pure Python,
while mock the agent used by execute_bronze because that is the AI entry point.
"""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import agents.bronze_agent as bronze_module
from agents.bronze_agent import _apply_bronze_rules, execute_bronze


def _make_sttm_csv(tmp_path, extra_rows=None):
    rows = [
        {
            "source_schema": "raw",
            "source_table": "input",
            "source_column": "id",
            "target_schema": "bronze",
            "target_table": "input",
            "target_column": "id",
            "transformation_type": "identity",
            "transformation_logic": "copy as-is",
        },
        {
            "source_schema": "raw",
            "source_table": "input",
            "source_column": "amount",
            "target_schema": "bronze",
            "target_table": "input",
            "target_column": "amount",
            "transformation_type": "numeric_cast",
            "transformation_logic": "to_numeric",
        },
    ]
    if extra_rows:
        rows.extend(extra_rows)
    sttm_path = tmp_path / "bronze_sttm.csv"
    pd.DataFrame(rows).to_csv(sttm_path, index=False)
    return sttm_path


def _mock_agent_that_calls_tool(tool):
    class MockAgent:
        def invoke(self, payload):
            tool.invoke({})
            return {"messages": payload.get("messages", [])}

    return MockAgent()


class TestApplyBronzeRules:
    @pytest.fixture
    def input_file(self, tmp_path):
        path = tmp_path / "input.csv"
        pd.DataFrame({"id": ["A", "B"], "amount": ["10.5", "20.0"]}).to_csv(path, index=False)
        return path

    def test_metadata_columns_added(self, tmp_path, input_file, monkeypatch):
        monkeypatch.setattr(bronze_module, "BRONZE_DIR", tmp_path)
        sttm_path = _make_sttm_csv(tmp_path)

        output_path = _apply_bronze_rules([str(input_file)], str(sttm_path), "metadata-run")[0]
        output = pd.read_parquet(output_path)

        assert "_load_timestamp" in output.columns
        assert "_source_file" in output.columns
        assert output["_source_file"].eq(str(input_file)).all()

    def test_row_count_preserved(self, tmp_path, input_file, monkeypatch):
        monkeypatch.setattr(bronze_module, "BRONZE_DIR", tmp_path)
        sttm_path = _make_sttm_csv(tmp_path)

        output_path = _apply_bronze_rules([str(input_file)], str(sttm_path), "rows-run")[0]

        assert len(pd.read_parquet(output_path)) == 2

    def test_column_rename_applied(self, tmp_path, input_file, monkeypatch):
        monkeypatch.setattr(bronze_module, "BRONZE_DIR", tmp_path)
        sttm_path = _make_sttm_csv(tmp_path, [
            {
                "source_schema": "raw",
                "source_table": "input",
                "source_column": "id",
                "target_schema": "bronze",
                "target_table": "input",
                "target_column": "transaction_id",
                "transformation_type": "rename",
                "transformation_logic": "rename column",
            }
        ])

        output_path = _apply_bronze_rules([str(input_file)], str(sttm_path), "rename-run")[0]
        output = pd.read_parquet(output_path)

        assert "transaction_id" in output.columns
        assert "id" not in output.columns

    def test_multiple_files_produce_multiple_outputs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bronze_module, "BRONZE_DIR", tmp_path)
        first = tmp_path / "first.csv"
        second = tmp_path / "second.csv"
        pd.DataFrame({"id": ["A"], "amount": ["10"]}).to_csv(first, index=False)
        pd.DataFrame({"id": ["B"], "amount": ["20"]}).to_csv(second, index=False)
        sttm_path = _make_sttm_csv(tmp_path)

        outputs = _apply_bronze_rules([str(first), str(second)], str(sttm_path), "multi-run")

        assert len(outputs) == 2
        assert all(Path(path).exists() for path in outputs)

    def test_output_file_is_valid_parquet(self, tmp_path, input_file, monkeypatch):
        monkeypatch.setattr(bronze_module, "BRONZE_DIR", tmp_path)
        sttm_path = _make_sttm_csv(tmp_path)

        output_path = _apply_bronze_rules([str(input_file)], str(sttm_path), "parquet-run")[0]
        output = pd.read_parquet(output_path)

        assert output.shape == (2, 4)
        assert output["amount"].tolist() == [10.5, 20.0]


class TestExecuteBronze:
    def test_execute_bronze_returns_paths(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bronze_module, "BRONZE_DIR", tmp_path)
        input_file = tmp_path / "input.csv"
        pd.DataFrame({"id": ["A"], "amount": ["10"]}).to_csv(input_file, index=False)
        sttm_path = _make_sttm_csv(tmp_path)
        expected_path = str(tmp_path / "input_bronze.parquet")

        def make_agent(model, tools, prompt):
            return _mock_agent_that_calls_tool(tools[1])

        with patch.object(bronze_module, "AuditLogger") as audit_logger, patch.object(
            bronze_module, "create_react_agent", side_effect=make_agent
        ):
            paths = execute_bronze([str(input_file)], str(sttm_path), "execute-run", "Load the input file")

        assert paths == [expected_path]
        audit_logger.assert_called()
