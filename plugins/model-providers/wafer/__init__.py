"""Wafer AI provider profile.

Wafer (https://pass.wafer.ai/v1) is an OpenAI-compatible inference provider
hosting models from several families (GLM 5.x, Kimi K2.6/K2.7, Qwen 3.5/3.6/3.7,
DeepSeek v4, MiniMax M3).  Two provider-specific behaviours are wired here:

1. **ZDR (Zero Data Retention) header.**  Wafer enforces per-request data
   retention policy via the ``Wafer-ZDR`` HTTP header.  ``required`` tells the
   upstream not to store or train on the prompt/response.  We declare it on
   ``default_headers`` so it rides on every chat-completion request; the
   generic ``profile.default_headers`` fallback in ``run_agent.py`` and
   ``agent/auxiliary_client.py`` picks it up at client construction.

2. **Reasoning via ``extra_body.thinking``.**  Wafer accepts a ``thinking``
   object in the request body (``{"type": "enabled"|"disabled", "effort":
   ...}``) — the same wire shape as Kimi/Moonshot.  Unlike Kimi, Wafer has no
   top-level ``reasoning_effort`` toggle: effort lives *inside* the thinking
   object.  We translate Hermes' ``reasoning_config`` (``{"enabled": bool,
   "effort": str}``) into that shape.  When reasoning is on and an effort is
   requested, effort is embedded in the thinking object; when off, ``disabled``
   is sent.  Verified live on 2026-06-22: GLM-5.2 returns ``reasoning_content``
   (DeepSeek/Moonshot-style) on enabled, 0 reasoning tokens on disabled.

The response-side ``reasoning_content`` field is already captured by
``ChatCompletionsTransport.normalize_response`` into ``provider_data``, so no
response handling changes are needed here.

Model catalog verified live against GET /v1/models on 2026-06-22 (10 models).
Wafer has split endpoints (pass / serverless) sharing one base_url — see
upstream issue #31739.  This profile targets the ``pass`` tier
(https://pass.wafer.ai/v1).
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class WaferProfile(ProviderProfile):
    """Wafer AI — ZDR header + thinking-body reasoning controls."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **context: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Translate Hermes reasoning_config into Wafer's ``thinking`` body.

        Wafer's wire shape puts both the on/off toggle and the effort level
        inside a single ``extra_body.thinking`` object — there is no separate
        top-level ``reasoning_effort`` (unlike Kimi).  So we always emit the
        thinking object and never a top-level effort field.

        - reasoning_config is None or enabled (default) -> ``{"type": "enabled"}``
          (+ ``"effort"`` when a recognised effort is requested)
        - reasoning_config.enabled is False or effort == "none" -> ``{"type":
          "disabled"}``
        """
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}

        if not reasoning_config or not isinstance(reasoning_config, dict):
            # No config -> thinking enabled, server picks the depth.
            extra_body["thinking"] = {"type": "enabled"}
            return extra_body, top_level

        enabled = reasoning_config.get("enabled", True)
        effort = (reasoning_config.get("effort") or "").strip().lower()

        if enabled is False or effort == "none":
            extra_body["thinking"] = {"type": "disabled"}
            return extra_body, top_level

        thinking: dict[str, Any] = {"type": "enabled"}
        if effort in {"minimal", "low", "medium", "high", "xhigh"}:
            thinking["effort"] = effort
        extra_body["thinking"] = thinking
        return extra_body, top_level


wafer = WaferProfile(
    name="wafer",
    aliases=("wafer-ai", "waferai"),
    display_name="Wafer AI",
    description="Wafer AI — multi-model OpenAI-compatible direct API (ZDR-capable)",
    signup_url="https://wafer.ai/",
    env_vars=("WAFER_API_KEY", "WAFER_BASE_URL"),
    base_url="https://pass.wafer.ai/v1",
    auth_type="api_key",
    # ZDR (Zero Data Retention): tell Wafer not to store/train on
    # prompts/responses.  Rides on every request via the profile.default_headers
    # fallback in run_agent.py / agent/auxiliary_client.py.
    default_headers={"Wafer-ZDR": "required"},
    # Cheap fast model for auxiliary tasks (compression, vision, etc.).
    default_aux_model="GLM-5.1",
    fallback_models=(
        "GLM-5.2",
        "GLM-5.1",
        "Kimi-K2.6",
        "Kimi-K2.7-Code",
        "Qwen3.5-397B-A17B",
        "Qwen3.6-35B-A3B",
        "qwen3.7-max",
        "deepseek-v4-pro",
        "deepseek-v4-flash",
        "MiniMax-M3",
    ),
)

register_provider(wafer)
