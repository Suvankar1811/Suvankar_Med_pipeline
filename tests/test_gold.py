"""Gold agent tests use Silver Parquet fixtures and deterministic STTM rules.
Pure Gold shaping is tested directly, while execute_gold uses a mocked agent.
"""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import agents.gold_agent as gold_module
from agents.gold_agent import _apply_gold_rules, execute_gold


def _make_gold_sttm(tmp_path, rows):
    sttm_path = tmp_path / "gold_sttm.csv"
    pd.DataFrame(rows).to_csv(sttm_path, index=False)
    return sttm_path


def _passthrough_sttm(tmp_path, source_table, columns):
    rows = [
        {
            "source_table": source_table,
            "source_column": column,
            "target_table": source_table,
            "target_column": column,
            "transformation_type": "identity",
            "transformation_logic": "copy as-is",
        }
        for column in columns
    ]
    return _make_gold_sttm(tmp_path, rows)


class TestApplyGoldRules:
    def test_output_written_for_each_target_table(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gold_module, "GOLD_DIR", tmp_path)
        source = tmp_path / "sales_silver.parquet"
        pd.DataFrame({"store_id": ["S01"], "sales": [100]}).to_parquet(source, index=False)
        sttm_path = _passthrough_sttm(tmp_path, "sales", ["store_id", "sales"])

        outputs = _apply_gold_rules([str(source)], str(sttm_path), "output-run")

        assert outputs == [str(tmp_path / "sales.parquet")]
        assert Path(outputs[0]).exists()

    def test_output_is_valid_parquet(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gold_module, "GOLD_DIR", tmp_path)
        source = tmp_path / "sales_silver.parquet"
        pd.DataFrame({"store_id": ["S01", "S02"], "sales": [100, 200]}).to_parquet(source, index=False)
        sttm_path = _passthrough_sttm(tmp_path, "sales", ["store_id", "sales"])

        output_path = _apply_gold_rules([str(source)], str(sttm_path), "valid-run")[0]
        output = pd.read_parquet(output_path)

        assert output.shape == (2, 3)
        assert output.columns[0] == "pk_gold_id"
        assert output["sales"].tolist() == [100, 200]

    def test_multiple_target_tables(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gold_module, "GOLD_DIR", tmp_path)
        source = tmp_path / "sales_silver.parquet"
        pd.DataFrame({"store_id": ["S01"], "sales": [100], "units": [2]}).to_parquet(source, index=False)
        sttm_path = _make_gold_sttm(tmp_path, [
            {
                "source_table": "sales",
                "source_column": "store_id",
                "target_table": "sales_summary",
                "target_column": "store_id",
                "transformation_type": "identity",
                "transformation_logic": "copy as-is",
            },
            {
                "source_table": "sales",
                "source_column": "sales",
                "target_table": "sales_summary",
                "target_column": "sales",
                "transformation_type": "identity",
                "transformation_logic": "copy as-is",
            },
            {
                "source_table": "sales",
                "source_column": "store_id",
                "target_table": "unit_summary",
                "target_column": "store_id",
                "transformation_type": "identity",
                "transformation_logic": "copy as-is",
            },
            {
                "source_table": "sales",
                "source_column": "units",
                "target_table": "unit_summary",
                "target_column": "units",
                "transformation_type": "identity",
                "transformation_logic": "copy as-is",
            },
        ])

        outputs = _apply_gold_rules([str(source)], str(sttm_path), "multi-target-run")

        assert {Path(path).name for path in outputs} == {"sales_summary.parquet", "unit_summary.parquet"}

    def test_column_subset_applied(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gold_module, "GOLD_DIR", tmp_path)
        source = tmp_path / "sales_silver.parquet"
        pd.DataFrame({"store_id": ["S01"], "sales": [100], "internal_note": ["omit"]}).to_parquet(source, index=False)
        sttm_path = _passthrough_sttm(tmp_path, "sales", ["store_id", "sales"])

        output = pd.read_parquet(_apply_gold_rules([str(source)], str(sttm_path), "subset-run")[0])

        assert output.columns.tolist() == ["pk_gold_id", "store_id", "sales"]
        assert "internal_note" not in output.columns

    def test_multiple_aggregations_are_applied(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gold_module, "GOLD_DIR", tmp_path)
        source = tmp_path / "sales_silver.parquet"
        pd.DataFrame({"store_id": ["S01", "S01"], "sales": [100, 200], "units": [2, 3]}).to_parquet(source, index=False)
        sttm_path = _make_gold_sttm(tmp_path, [
            {
                "source_table": "sales",
                "source_column": "store_id",
                "target_table": "summary",
                "target_column": "store_id",
                "transformation_type": "group",
                "transformation_logic": "group by store_id",
            },
            {
                "source_table": "sales",
                "source_column": "sales",
                "target_table": "summary",
                "target_column": "total_sales",
                "transformation_type": "aggregation",
                "transformation_logic": "SUM(sales)",
            },
            {
                "source_table": "sales",
                "source_column": "units",
                "target_table": "summary",
                "target_column": "total_units",
                "transformation_type": "aggregation",
                "transformation_logic": "SUM(units)",
            },
        ])

        output = pd.read_parquet(_apply_gold_rules([str(source)], str(sttm_path), "multi-aggregation-run")[0])

        assert output[["store_id", "total_sales", "total_units"]].to_dict("records") == [
            {"store_id": "S01", "total_sales": 300, "total_units": 5}
        ]

    def test_renamed_aggregate_target_does_not_collide(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gold_module, "GOLD_DIR", tmp_path)
        source = tmp_path / "sales_silver.parquet"
        pd.DataFrame({"store_id": ["S01", "S01"], "total_amount": [100, 200]}).to_parquet(source, index=False)
        sttm_path = _make_gold_sttm(tmp_path, [
            {
                "source_table": "sales",
                "source_column": "store_id",
                "target_table": "summary",
                "target_column": "store_id",
                "transformation_type": "group",
                "transformation_logic": "group by store_id",
            },
            {
                "source_table": "sales",
                "source_column": "total_amount",
                "target_table": "summary",
                "target_column": "total_revenue",
                "transformation_type": "aggregation",
                "transformation_logic": "SUM(total_amount)",
            },
        ])

        output = pd.read_parquet(_apply_gold_rules([str(source)], str(sttm_path), "renamed-aggregate-run")[0])

        assert output[["store_id", "total_revenue"]].to_dict("records") == [
            {"store_id": "S01", "total_revenue": 300}
        ]


class TestExecuteGold:
    def test_execute_gold_returns_paths(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gold_module, "GOLD_DIR", tmp_path)
        source = tmp_path / "sales_silver.parquet"
        pd.DataFrame({"store_id": ["S01"], "sales": [100]}).to_parquet(source, index=False)
        sttm_path = _passthrough_sttm(tmp_path, "sales", ["store_id", "sales"])

        def make_agent(model, tools, prompt):
            class MockAgent:
                def invoke(self, payload):
                    tools[1].invoke({})
                    return {"messages": payload.get("messages", [])}

            return MockAgent()

        with patch.object(gold_module, "create_react_agent", side_effect=make_agent):
            paths = execute_gold([str(source)], str(sttm_path), "execute-run", "Build sales report")

        assert paths == [str(tmp_path / "sales.parquet")]
        assert Path(paths[0]).exists()
