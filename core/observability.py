import json
import time
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

TRACES_DIR = Path("data/traces")
TRACES_DIR.mkdir(parents=True, exist_ok=True)


class AgentTrace:
    def __init__(self, agent_name: str, run_id: str):
        self.agent_name = agent_name
        self.run_id = run_id
        self.started_at = time.time()

        self.trace = {
            "agent": agent_name,
            "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "input": {},
            "plan": "",
            "tool_calls": [],
            "reasoning_steps": [],
            "output": {},
            "duration_seconds": 0.0,
            "status": "in_progress",
        }

    def set_input(self, **kwargs):
        self.trace["input"] = kwargs
        return self

    def set_plan(self, plan_str: str):
        self.trace["plan"] = plan_str
        return self

    def set_output(self, **kwargs):
        self.trace["output"] = kwargs
        return self

    def extract_from_messages(self, messages):
        for message in messages:
            if isinstance(message, HumanMessage):
                self.trace["reasoning_steps"].append({
                    "type": "task_input",
                    "content": message.content,
                })

            elif isinstance(message, AIMessage):
                if getattr(message, "tool_calls", None):
                    self.trace["tool_calls"].extend(message.tool_calls)

                content = message.content
                if content:
                    if not self.trace["plan"]:
                        self.trace["plan"] = str(content)
                    self.trace["reasoning_steps"].append({
                        "type": "ai_reasoning",
                        "content": content,
                    })

            elif isinstance(message, ToolMessage):
                self.trace["reasoning_steps"].append({
                    "type": "tool_result",
                    "tool": getattr(message, "name", "unknown_tool"),
                    "content": message.content,
                })

        return self

    def complete(self, status="success"):
        self.trace["status"] = status
        self.trace["duration_seconds"] = round(time.time() - self.started_at, 4)

        trace_file = TRACES_DIR / f"trace_{self.agent_name}_{self.run_id[:8]}.json"
        with trace_file.open("w", encoding="utf-8") as file:
            json.dump(self.trace, file, indent=2, default=str)

        print(f"[{self.agent_name}] {status} in {self.trace['duration_seconds']}s | run_id={self.run_id}")
        return self

    def fail(self, error):
        self.trace["error"] = str(error)
        return self.complete(status="failed")
