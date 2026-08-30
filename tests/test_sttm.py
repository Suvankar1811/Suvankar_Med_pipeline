import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from langchain_core.messages import AIMessage

import agents.sttm_generator as sttm_module
from agents.sttm_generator import (
    _extract_sttm_rows,
    _invoke_llm_for_sttm_rows,
    _make_sttm_tools,
    _parse_json_array,
    _prepare_bronze_context,
    _prepare_gold_context,
    _prepare_silver_context,
)


BRONZE_ROW = {
    "source_schema": "raw",
    "source_table": "sales",
    "source_column": "amount",
    "target_schema": "bronze",
    "target_table": "sales",
    "target_column": "amount",
    "transformation_type": "identity",
    "transformation_logic": "copy as-is",
}
SILVER_ROW = {**BRONZE_ROW, "target_schema": "silver", "transformation_type": "numeric", "transformation_logic": "fillna mean"}
GOLD_ROW = {**BRONZE_ROW, "target_schema": "gold", "target_table": "sales_summary", "transformation_type": "sum", "transformation_logic": "sum by store_id"}


def _mock_agent(tool):
    class MockAgent:
        def invoke(self, payload):
            tool.invoke({})
            return {"messages": payload.get("messages", [])}

    return MockAgent()


class TestPrepareBronzeContext:
    def test_returns_profile_dict(self, tmp_path):
        profile_path = tmp_path / "profile.json"
        expected = {"files": {"sales.csv": {"shape": [2, 1]}}}
        profile_path.write_text(json.dumps(expected), encoding="utf-8")

        assert _prepare_bronze_context(str(profile_path)) == expected

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _prepare_bronze_context(str(tmp_path / "missing.json"))


class TestPrepareSilverContext:
    def test_filters_to_approved_columns(self, tmp_path):
        parquet_path = tmp_path / "sales_bronze.parquet"
        pd.DataFrame({"amount": [10], "unapproved": ["x"]}).to_parquet(parquet_path, index=False)
        sttm_path = tmp_path / "bronze_sttm.csv"
        pd.DataFrame([BRONZE_ROW]).to_csv(sttm_path, index=False)

        context = _prepare_silver_context([str(parquet_path)], str(sttm_path))

        assert context["files"][parquet_path.name]["approved_columns"] == ["amount"]

    def test_returns_filename_not_full_path(self, tmp_path):
        parquet_path = tmp_path / "sales_bronze.parquet"
        pd.DataFrame({"amount": [10]}).to_parquet(parquet_path, index=False)
        sttm_path = tmp_path / "bronze_sttm.csv"
        pd.DataFrame([BRONZE_ROW]).to_csv(sttm_path, index=False)

        context = _prepare_silver_context([str(parquet_path)], str(sttm_path))

        assert list(context["files"]) == [parquet_path.name]


class TestPrepareGoldContext:
    def test_filters_to_approved_columns(self, tmp_path):
        parquet_path = tmp_path / "sales_silver.parquet"
        pd.DataFrame({"amount": [10], "unapproved": ["x"]}).to_parquet(parquet_path, index=False)
        sttm_path = tmp_path / "silver_sttm.csv"
        pd.DataFrame([SILVER_ROW]).to_csv(sttm_path, index=False)

        context = _prepare_gold_context([str(parquet_path)], str(sttm_path))

        assert context["files"][parquet_path.name]["approved_columns"] == ["amount"]


class TestExtractSttmRows:
    def test_extracts_plain_json_array(self):
        assert _extract_sttm_rows([AIMessage(content=json.dumps([BRONZE_ROW]))]) == [BRONZE_ROW]

    def test_extracts_json_with_fences(self):
        raw = "```json\n" + json.dumps([SILVER_ROW]) + "\n```"
        assert _extract_sttm_rows([AIMessage(content=raw)]) == [SILVER_ROW]

    def test_returns_empty_list_when_no_array(self):
        assert _extract_sttm_rows([AIMessage(content="No STTM rows available")]) == []

    def test_prefers_last_message(self):
        first = AIMessage(content=json.dumps([BRONZE_ROW]))
        last = AIMessage(content=json.dumps([GOLD_ROW]))
        assert _extract_sttm_rows([first, last]) == [GOLD_ROW]


class TestParseJsonArrayAndInvokeLlmForSttmRows:
    def test_parse_json_array_handles_think_tags(self):
        raw = "<think>Consider this decoy [not the answer]</think>" + json.dumps([{"a": 1}])

        assert _parse_json_array(raw) == [{"a": 1}]

    def test_invoke_llm_raises_when_rows_empty_after_retries(self):
        class EmptyLLM:
            def __init__(self):
                self.calls = 0

            def invoke(self, prompt):
                self.calls += 1
                return AIMessage(content="No valid JSON here")

        llm = EmptyLLM()
        with patch.object(sttm_module, "_make_llm", return_value=llm):
            with pytest.raises(ValueError, match="after 2 attempts"):
                _invoke_llm_for_sttm_rows("Generate rules", "Bronze")
        assert llm.calls == 2

    def test_invoke_llm_returns_rows_on_first_success(self):
        class SuccessfulLLM:
            def __init__(self):
                self.calls = 0

            def invoke(self, prompt):
                self.calls += 1
                return AIMessage(content=json.dumps([GOLD_ROW]))

        llm = SuccessfulLLM()
        with patch.object(sttm_module, "_make_llm", return_value=llm):
            rows = _invoke_llm_for_sttm_rows("Generate rules", "Gold")

        assert rows == [GOLD_ROW]
        assert llm.calls == 1


class TestSilverFallback:
    def test_invalid_model_response_uses_schema_fallback(self, tmp_path):
        context = {
            "files": {
                "products_bronze.parquet": {
                    "approved_columns": ["product_id", "category"],
                }
            }
        }
        destination = tmp_path / "silver_sttm.csv"
        tools = _make_sttm_tools("Silver", context, str(destination), "fallback-run")

        class EmptyLLM:
            def invoke(self, prompt):
                return AIMessage(content="No valid JSON here")

        with patch.object(sttm_module, "_make_llm", return_value=EmptyLLM()):
            tools[2].invoke({})

        output = pd.read_csv(destination)
        assert output["target_column"].tolist() == [
            "sttm_products_sk",
            "product_id",
            "category",
        ]
