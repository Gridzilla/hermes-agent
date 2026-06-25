"""Retry utilities — jittered backoff for decorrelated retries.

Replaces fixed exponential backoff with jittered delays to prevent
thundering-herd retry spikes when multiple sessions hit the same
rate-limited provider concurrently.
"""

import json
import os
import random
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

# Monotonic counter for jitter seed uniqueness within the same process.
# Protected by a lock to avoid race conditions in concurrent retry paths
# (e.g. multiple gateway sessions retrying simultaneously).
_jitter_counter = 0
_jitter_lock = threading.Lock()


def jittered_backoff(
    attempt: int,
    *,
    base_delay: float = 5.0,
    max_delay: float = 120.0,
    jitter_ratio: float = 0.5,
) -> float:
    """Compute a jittered exponential backoff delay.

    Args:
        attempt: 1-based retry attempt number.
        base_delay: Base delay in seconds for attempt 1.
        max_delay: Maximum delay cap in seconds.
        jitter_ratio: Fraction of computed delay to use as random jitter
            range.  0.5 means jitter is uniform in [0, 0.5 * delay].

    Returns:
        Delay in seconds: min(base * 2^(attempt-1), max_delay) + jitter.

    The jitter decorrelates concurrent retries so multiple sessions
    hitting the same provider don't all retry at the same instant.
    """
    global _jitter_counter
    with _jitter_lock:
        _jitter_counter += 1
        tick = _jitter_counter

    exponent = max(0, attempt - 1)
    if exponent >= 63 or base_delay <= 0:
        delay = max_delay
    else:
        delay = min(base_delay * (2 ** exponent), max_delay)

    # Seed from time + counter for decorrelation even with coarse clocks.
    seed = (time.time_ns() ^ (tick * 0x9E3779B9)) & 0xFFFFFFFF
    rng = random.Random(seed)
    jitter = rng.uniform(0, jitter_ratio * delay)

    return delay + jitter


# Persistent rate-limit fallback policy. Kept here (not in the conversation
# loop) so tests and other model-call paths can share one schedule.
def persistent_rate_limit_backoff(
    attempt: int,
    *,
    retry_after: Optional[float] = None,
) -> float:
    """Return the wait time for a provider rate-limit retry.

    User-facing policy:
      * attempts 1-5: quick jittered retries
      * attempts 6-9: every ~15 minutes
      * attempts 10+: every ~1 hour

    ``attempt`` is 1-based: the first retry after the first failed request is
    attempt=1. Retry-After is honored only when it is longer than the computed
    quick retry, so a provider asking for a long wait naturally moves us into
    fallback timing without exceeding the requested floor.
    """
    attempt = max(1, int(attempt))
    if attempt <= 5:
        quick = jittered_backoff(attempt, base_delay=2.0, max_delay=60.0)
        if retry_after and retry_after > quick:
            return float(retry_after)
        return quick
    if attempt <= 9:
        # 15 minutes + small jitter so multiple sessions don't wake together.
        return 900.0 + jittered_backoff(attempt - 5, base_delay=5.0, max_delay=30.0)
    # Hourly + small jitter.
    return 3600.0 + jittered_backoff(attempt - 9, base_delay=5.0, max_delay=60.0)


def rate_limit_fallback_stage(attempt: int) -> str:
    if attempt <= 5:
        return "quick_retry"
    if attempt <= 9:
        return "15m_fallback"
    return "hourly_fallback"


def _hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except Exception:
        return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))


def _read_env_value(name: str) -> Optional[str]:
    val = os.environ.get(name)
    if val:
        return val
    env_path = _hermes_home() / ".env"
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if not line or line.lstrip().startswith("#") or "=" not in line:
                continue
            key, raw = line.split("=", 1)
            if key.strip() == name:
                return raw.strip().strip('"').strip("'") or None
    except Exception:
        return None
    return None


def _read_rate_limit_notify_target() -> Optional[str]:
    """Read ``agent.rate_limit_notify_target`` from config.yaml.

    Format is the same as cron/send_message delivery targets, currently only
    ``telegram:<chat_id>`` is used here.  Missing/unsupported targets are a
    silent no-op so retry resilience never depends on notification health.
    """
    cfg = _hermes_home() / "config.yaml"
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        target = (data.get("agent") or {}).get("rate_limit_notify_target")
        if isinstance(target, str) and target.strip():
            return target.strip()
    except Exception:
        pass
    return None


def _rate_limit_alert_state_path() -> Path:
    return _hermes_home() / "rate_limits" / "fallback_alerts.json"


def _should_send_rate_limit_alert(stage: str, *, now: Optional[float] = None) -> bool:
    """Throttle fallback alerts: always on stage change, otherwise hourly."""
    now = time.time() if now is None else now
    path = _rate_limit_alert_state_path()
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        state = {}
    last_stage = state.get("stage")
    last_sent = float(state.get("sent_at") or 0)
    if last_stage != stage or now - last_sent >= 3600:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"stage": stage, "sent_at": now}), encoding="utf-8")
        except Exception:
            pass
        return True
    return False


def maybe_notify_rate_limit_fallback(
    *,
    provider: str,
    model: str,
    attempt: int,
    wait_time: float,
    stage: Optional[str] = None,
    target: Optional[str] = None,
) -> None:
    """Best-effort Telegram alert for persistent rate-limit fallback.

    Never raises.  Intended for the central API retry loop: if Telegram is not
    configured or the network is down, the model retry path still continues.
    """
    stage = stage or rate_limit_fallback_stage(attempt)
    if stage == "quick_retry":
        return
    if not _should_send_rate_limit_alert(stage):
        return
    target = target or _read_rate_limit_notify_target()
    if not target or not target.startswith("telegram:"):
        return
    token = _read_env_value("TELEGRAM_BOT_TOKEN")
    if not token:
        return
    chat_id = target.split(":", 1)[1].strip()
    if not chat_id:
        return
    mins = int(round(wait_time / 60))
    if wait_time >= 3600:
        wait_text = f"~{wait_time / 3600:.1f}h"
    else:
        wait_text = f"~{mins}m"
    text = (
        "⚠️ Hermes rate-limit fallback active\n"
        f"Provider/model: {provider}/{model}\n"
        f"Stage: {stage} (attempt {attempt})\n"
        f"Next retry in {wait_text}. Hermes will keep retrying instead of stopping."
    )
    try:
        body = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=body,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10):
            pass
    except Exception:
        return


def _read_task_completion_target() -> Optional[str]:
    """Read ``agent.task_completion_target`` from config.yaml.

    Format matches cron/send_message targets: ``telegram:<chat_id>`` or
    ``telegram:<chat_id>:<thread_id>`` (forum topic).  Missing/unsupported
    targets are a silent no-op so task delivery never depends on this.
    """
    cfg = _hermes_home() / "config.yaml"
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        target = (data.get("agent") or {}).get("task_completion_target")
        if isinstance(target, str) and target.strip():
            return target.strip()
    except Exception:
        pass
    return None


def send_task_completion_notification(
    *,
    session_id: str,
    command: str,
    exit_code,
    output: str,
    origin_platform: str = "",
    origin_chat_id: str = "",
    origin_thread_id: str = "",
) -> None:
    """Best-effort copy of a background-task completion to a central topic.

    Reads ``agent.task_completion_target`` (see ``_read_task_completion_target``).
    Skips the send when the task's origin chat already matches the target, so a
    task started *from* the target topic is not pinged twice.  Sends directly via
    the Telegram Bot API (no adapter dependency).  Never raises — notification
    health must not affect task delivery.

    LOCAL PATCH (task-completion-notify). Anchor: ``send_task_completion_notification``.
    """
    target = _read_task_completion_target()
    if not target or not target.startswith("telegram:"):
        return
    # Dedup against origin: if the task already ran in the target topic, the
    # gateway's normal agent_notify injection already surfaced it there.
    origin_key = f"{origin_platform}:{origin_chat_id}:{origin_thread_id}".strip(":").lower()
    target_key = ":".join(p.strip() for p in target.split(":")[:3] if p.strip()).lower()
    if origin_key and origin_key == target_key:
        return
    token = _read_env_value("TELEGRAM_BOT_TOKEN")
    if not token:
        return
    parts = target.split(":")
    chat_id = parts[1].strip() if len(parts) > 1 else ""
    thread_id = parts[2].strip() if len(parts) > 2 else ""
    if not chat_id:
        return
    status_icon = "✅" if str(exit_code) == "0" else "❌"
    cmd_disp = command[:80] + ("…" if len(command) > 80 else "")
    _LIMIT = 1500
    if len(output) > _LIMIT:
        tail = output[-_LIMIT:]
        nl = tail.find("\n")
        tail = tail[nl + 1:] if nl != -1 else tail
        out_disp = f"…\n{tail}"
    else:
        out_disp = output or "(no output)"
    text = (
        f"{status_icon} Background task completed\n"
        f"Command: {cmd_disp}\n"
        f"Exit code: {exit_code}\n"
        f"Output:\n{out_disp}"
    )
    payload = {"chat_id": chat_id, "text": text}
    if thread_id:
        payload["message_thread_id"] = thread_id
    try:
        body = urllib.parse.urlencode(payload).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=body,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10):
            pass
    except Exception:
        return
