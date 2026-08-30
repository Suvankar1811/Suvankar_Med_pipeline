"""Silver agent tests use Parquet fixtures because Silver reads Bronze Parquet files.
Pure cleansing is tested directly, while execute_silver uses a mocked agent.
"""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import agents.silver_agent as silver_module
from agents.silver_agent import _apply_silver_rules, execute_silver


def _make_silver_sttm(tmp_path, rows):
    sttm_path = tmp_path / "silver_sttm.csv"
    pd.DataFrame(rows).to_csv(sttm_path, index=False)
    return sttm_path


def _passthrough_sttm(tmp_path, source_table, columns):
    rows = [
        {
            "source_table": source_table,
            "source_column": column,
            "target_column": column,
            "transformation_type": "identity",
            "transformation_logic": "copy as-is",
        }
        for column in columns
    ]
    return _make_silver_sttm(tmp_path, rows)


def _mock_agent_that_calls_tool(tool):
    class MockAgent:
        def invoke(self, payload):
            tool.invoke({})
            return {"messages": payload.get("messages", [])}

    return MockAgent()


class TestApplySilverRules:
    def test_output_written_for_each_input(self, tmp_path, monkeypatch):
        monkeypatch.setattr(silver_module, "SILVER_DIR", tmp_path)
        first = tmp_path / "first_bronze.parquet"
        second = tmp_path / "second_bronze.parquet"
        pd.DataFrame({"id": [1]}).to_parquet(first, index=False)
        pd.DataFrame({"id": [2]}).to_parquet(second, index=False)
        sttm_path = _passthrough_sttm(tmp_path, "first", ["id"])

        outputs = _apply_silver_rules([str(first), str(second)], str(sttm_path), "output-run")

        assert len(outputs) == 2
        assert Path(outputs[0]).exists()
        assert Path(outputs[1]).exists()

    def test_null_handling_fill_mean(self, tmp_path, monkeypatch):
        monkeypatch.setattr(silver_module, "SILVER_DIR", tmp_path)
        source = tmp_path / "metrics_bronze.parquet"
        pd.DataFrame({"metric": [10.0, None, 30.0]}).to_parquet(source, index=False)
        sttm_path = _make_silver_sttm(tmp_path, [
            {
                "source_table": "metrics",
                "source_column": "metric",
                "target_column": "metric",
                "transformation_type": "numeric",
                "transformation_logic": "fillna mean",
            }
        ])

        output = pd.read_parquet(_apply_silver_rules([str(source)], str(sttm_path), "mean-run")[0])

        assert output["metric"].tolist() == [10.0, 20.0, 30.0]
        assert output["metric"].isna().sum() == 0

    def test_dedup_removes_duplicate_rows(self, tmp_path, monkeypatch):
        monkeypatch.setattr(silver_module, "SILVER_DIR", tmp_path)
        source = tmp_path / "duplicates_bronze.parquet"
        pd.DataFrame({"id": [1, 1, 2], "label": ["a", "a", "b"]}).to_parquet(source, index=False)
        sttm_path = _passthrough_sttm(tmp_path, "duplicates", ["id", "label"])

        output = pd.read_parquet(_apply_silver_rules([str(source)], str(sttm_path), "dedup-run")[0])

        assert len(output) == 2
        assert output[["id", "label"]].drop_duplicates().shape[0] == 2

    def test_column_rename_applied(self, tmp_path, monkeypatch):
        monkeypatch.setattr(silver_module, "SILVER_DIR", tmp_path)
        source = tmp_path / "rename_bronze.parquet"
        pd.DataFrame({"old_name": ["Alice"]}).to_parquet(source, index=False)
        sttm_path = _make_silver_sttm(tmp_path, [
            {
                "source_table": "rename",
                "source_column": "old_name",
                "target_column": "customer_name",
                "transformation_type": "rename",
                "transformation_logic": "strip",
            }
        ])

        output = pd.read_parquet(_apply_silver_rules([str(source)], str(sttm_path), "rename-run")[0])

        assert "customer_name" in output.columns
        assert "old_name" not in output.columns

    def test_join_keys_are_preserved_when_sttm_omits_them(self, tmp_path, monkeypatch):
        monkeypatch.setattr(silver_module, "SILVER_DIR", tmp_path)
        source = tmp_path / "products_bronze.parquet"
        pd.DataFrame({"product_id": ["P01"], "category": ["Electronics"]}).to_parquet(source, index=False)
        sttm_path = _passthrough_sttm(tmp_path, "products", ["category"])

        output = pd.read_parquet(_apply_silver_rules([str(source)], str(sttm_path), "join-key-run")[0])

        assert "product_id" in output.columns
        assert "category" in output.columns

    def test_output_is_valid_parquet(self, tmp_path, monkeypatch):
        monkeypatch.setattr(silver_module, "SILVER_DIR", tmp_path)
        source = tmp_path / "valid_bronze.parquet"
        pd.DataFrame({"id": [1, 2]}).to_parquet(source, index=False)
        sttm_path = _passthrough_sttm(tmp_path, "valid", ["id"])

        output_path = _apply_silver_rules([str(source)], str(sttm_path), "valid-run")[0]
        output = pd.read_parquet(output_path)

        assert output.columns.tolist() == ["pk_valid_silver_id", "id"]
        assert output["pk_valid_silver_id"].tolist() == [1, 2]


class TestExecuteSilver:
    def test_execute_silver_returns_paths(self, tmp_path, monkeypatch):
        monkeypatch.setattr(silver_module, "SILVER_DIR", tmp_path)
        source = tmp_path / "input_bronze.parquet"
        pd.DataFrame({"id": [1], "amount": [10]}).to_parquet(source, index=False)
        sttm_path = _passthrough_sttm(tmp_path, "input", ["id", "amount"])

        def make_agent(model, tools, prompt):
            return _mock_agent_that_calls_tool(tools[1])

        with patch.object(silver_module, "create_react_agent", side_effect=make_agent):
            paths = execute_silver([str(source)], str(sttm_path), "execute-silver-run", "Cleanse input")

        assert paths == [str(tmp_path / "input_silver.parquet")]
        assert Path(paths[0]).exists()
