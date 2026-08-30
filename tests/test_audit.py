import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.audit import AuditLogger
import core.audit as audit_module


class TestAuditLogger:
    def test_log_creates_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(audit_module, "AUDIT_DIR", tmp_path)
        logger = AuditLogger(run_id="test-run-123")

        logger.log("test_agent", "test_action", rationale="unit test", phase="phase1")

        assert (tmp_path / "test-run-123.jsonl").exists()
        logs = logger.get_logs()
        assert len(logs) == 1
        assert logs[0]["run_id"] == "test-run-123"
        assert logs[0]["agent"] == "test_agent"
        assert logs[0]["action"] == "test_action"
        assert logs[0]["rationale"] == "unit test"
        assert logs[0]["phase"] == "phase1"

    def test_multiple_logs_appended(self, tmp_path, monkeypatch):
        monkeypatch.setattr(audit_module, "AUDIT_DIR", tmp_path)
        logger = AuditLogger(run_id="test-run-123")

        logger.log("test_agent", "first_action")
        logger.log("test_agent", "second_action")

        logs = logger.get_logs()
        assert len(logs) == 2
        assert logs[0]["action"] == "first_action"
        assert logs[1]["action"] == "second_action"

    def test_get_logs_empty_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(audit_module, "AUDIT_DIR", tmp_path)
        logger = AuditLogger(run_id="nonexistent-run")

        assert logger.get_logs() == []
