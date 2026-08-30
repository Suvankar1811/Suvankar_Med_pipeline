import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage

import agents.profiler as profiler_module
from agents.profiler import profile_dataset, profile_multiple_datasets


def _mock_agent_factory(model, tools, prompt):
    class MockAgent:
        def invoke(self, payload):
            tools[0].invoke({})
            stats = tools[1].invoke({})
            response = model.invoke(stats)
            return {
                "messages": [
                    *payload.get("messages", []),
                    AIMessage(content=response.content),
                ]
            }

    return MockAgent()


def _mock_llm():
    class MockLLM:
        def invoke(self, prompt):
            return AIMessage(content=json.dumps({
                "semantic_meanings": {"id": "Record identifier"},
                "join_keys": ["id"],
                "quality_notes": [],
            }))

    return MockLLM()


class TestProfiler:
    def test_profile_dataset_creates_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr(profiler_module, "PROFILES_DIR", tmp_path)
        csv_path = tmp_path / "sales.csv"
        pd.DataFrame({
            "id": [1, 2, 3],
            "amount": [10.5, 20.0, 30.25],
            "category": ["electronics", "home", "sports"],
        }).to_csv(csv_path, index=False)

        with patch.object(profiler_module, "_make_llm", return_value=_mock_llm()), patch.object(
            profiler_module, "create_react_agent", side_effect=_mock_agent_factory
        ):
            result_path = profile_dataset(str(csv_path), "profile-test-1", "Profile sales data")

        profile_path = Path(result_path)
        assert profile_path.exists()
        with profile_path.open(encoding="utf-8") as file:
            profile = json.load(file)

        dataset = profile["files"][str(csv_path)]
        assert dataset["shape"] == [3, 3]
        assert set(dataset["columns"]) == {"id", "amount", "category"}
        assert dataset["columns"]["amount"]["mean"] == pytest.approx(20.25)
        assert "agent_analysis" in profile

    def test_profile_multiple_datasets(self, tmp_path, monkeypatch):
        monkeypatch.setattr(profiler_module, "PROFILES_DIR", tmp_path)
        first_path = tmp_path / "sales.csv"
        second_path = tmp_path / "stores.csv"
        pd.DataFrame({"id": [1, 2], "amount": [10, 15]}).to_csv(first_path, index=False)
        pd.DataFrame({"id": ["S01", "S02"], "region": ["East", "West"]}).to_csv(second_path, index=False)

        with patch.object(profiler_module, "_make_llm", return_value=_mock_llm()), patch.object(
            profiler_module, "create_react_agent", side_effect=_mock_agent_factory
        ):
            result_path = profile_multiple_datasets(
                [str(first_path), str(second_path)],
                "profile-test-2",
                "Profile sales and store data",
            )

        with Path(result_path).open(encoding="utf-8") as file:
            profile = json.load(file)

        assert set(profile["files"]) == {str(first_path), str(second_path)}
        assert profile["files"][str(first_path)]["shape"] == [2, 2]
        assert profile["files"][str(second_path)]["shape"] == [2, 2]
        assert set(profile["files"][str(second_path)]["columns"]) == {"id", "region"}
