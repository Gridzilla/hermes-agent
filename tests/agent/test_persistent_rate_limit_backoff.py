import json
import os

from agent.retry_utils import (
    _should_send_rate_limit_alert,
    persistent_rate_limit_backoff,
    rate_limit_fallback_stage,
)


def test_persistent_rate_limit_backoff_schedule():
    quick = [persistent_rate_limit_backoff(i) for i in range(1, 6)]
    assert all(0 < delay < 120 for delay in quick)

    fifteen = [persistent_rate_limit_backoff(i) for i in range(6, 10)]
    assert all(900 <= delay < 960 for delay in fifteen)

    hourly = persistent_rate_limit_backoff(10)
    assert 3600 <= hourly < 3670


def test_persistent_rate_limit_backoff_honors_long_retry_after():
    assert persistent_rate_limit_backoff(1, retry_after=300) == 300


def test_rate_limit_fallback_stage_labels():
    assert rate_limit_fallback_stage(1) == "quick_retry"
    assert rate_limit_fallback_stage(5) == "quick_retry"
    assert rate_limit_fallback_stage(6) == "15m_fallback"
    assert rate_limit_fallback_stage(9) == "15m_fallback"
    assert rate_limit_fallback_stage(10) == "hourly_fallback"


def test_rate_limit_alert_throttle(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    assert _should_send_rate_limit_alert("15m_fallback", now=1000) is True
    assert _should_send_rate_limit_alert("15m_fallback", now=1100) is False
    assert _should_send_rate_limit_alert("hourly_fallback", now=1200) is True
    assert _should_send_rate_limit_alert("hourly_fallback", now=4700) is False
    assert _should_send_rate_limit_alert("hourly_fallback", now=4901) is True

    state_path = tmp_path / "rate_limits" / "fallback_alerts.json"
    state = json.loads(state_path.read_text())
    assert state["stage"] == "hourly_fallback"
    assert state["sent_at"] == 4901
