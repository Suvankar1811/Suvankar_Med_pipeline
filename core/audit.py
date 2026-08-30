import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from core.config import AUDIT_DIR


class AuditLogger:
    def __init__(self, run_id: str | None = None):
        self.run_id = run_id or str(uuid.uuid4())
        self.log_path = AUDIT_DIR / f"{self.run_id}.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, agent: str, action: str, **kwargs):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "agent": agent,
            "action": action,
            **kwargs,
        }

        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry) + "\n")

    def get_logs(self):
        if not self.log_path.exists():
            return []

        logs = []
        with self.log_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    logs.append(json.loads(line))

        return logs
