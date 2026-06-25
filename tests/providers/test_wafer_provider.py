"""Tests for the Wafer AI provider profile.

Asserts the two provider-specific behaviours that distinguish Wafer from a
generic OpenAI-compatible endpoint:

1. **ZDR header** — ``Wafer-ZDR: required`` is declared on the profile's
   ``default_headers``, which is the field the generic
   ``profile.default_headers`` fallback in ``run_agent.py`` and
   ``agent/auxiliary_client.py`` reads at client construction.  So the header
   rides on every chat-completion request without any core code change.

2. **Reasoning via ``extra_body.thinking``** — Wafer accepts a ``thinking``
   object (``{"type": "enabled"|"disabled", "effort": ...}``) in the request
   body.  Unlike Kimi there is no top-level ``reasoning_effort``: effort lives
   inside the thinking object.  The profile's ``build_api_kwargs_extras``
   translates Hermes' ``reasoning_config`` into that shape, and the transport
   merges it into ``extra_body``.

These tests exercise the real transport path
(``ChatCompletionsTransport._build_kwargs_from_profile``) so they catch any
wiring drift between the profile hooks and the transport assembly.
"""

from types import SimpleNamespace

from providers import get_provider_profile
from agent.transports.chat_completions import ChatCompletionsTransport


def _wafer_profile():
    """Return the Wafer profile, asserting it registered."""
    p = get_provider_profile("wafer")
    assert p is not None, "wafer provider profile not registered"
    return p


def test_wafer_profile_declares_zdr_header():
    """The ZDR header is on profile.default_headers — the field the generic
    fallback in run_agent.py / auxiliary_client.py reads at client build time.
    """
    p = _wafer_profile()
    assert p.default_headers.get("Wafer-ZDR") == "required", (
        f"Wafer-ZDR header missing or wrong: {p.default_headers!r}"
    )


def test_wafer_profile_identity():
    p = _wafer_profile()
    assert p.name == "wafer"
    assert p.base_url == "https://pass.wafer.ai/v1"
    assert p.auth_type == "api_key"
    assert "WAFER_API_KEY" in p.env_vars
    assert p.default_aux_model == "GLM-5.1"
    # Live-verified catalog (2026-06-22): 10 models on /v1/models.
    assert "GLM-5.2" in p.fallback_models
    assert len(p.fallback_models) == 10


def test_wafer_transport_emits_zdr_header_and_thinking_default():
    """Default call (no reasoning_config) → ZDR header on client kwargs,
    thinking:{type:enabled} in extra_body.  Exercises the real transport path.
    """
    p = _wafer_profile()
    transport = ChatCompletionsTransport()
    # Simulate the kwargs the transport builds when a provider_profile is set.
    # The header comes from profile.default_headers (read by run_agent at
    # client construction), not from build_kwargs — so we assert it directly
    # on the profile and assert the thinking body via build_kwargs.
    assert p.default_headers["Wafer-ZDR"] == "required"

    kwargs = transport.build_kwargs(
        model="GLM-5.2",
        messages=[{"role": "user", "content": "hi"}],
        provider_profile=p,
        reasoning_config=None,
    )
    assert kwargs["model"] == "GLM-5.2"
    assert kwargs["extra_body"]["thinking"] == {"type": "enabled"}


def test_wafer_transport_thinking_with_effort():
    """reasoning_config enabled + effort=high → effort embedded in thinking."""
    p = _wafer_profile()
    transport = ChatCompletionsTransport()
    kwargs = transport.build_kwargs(
        model="GLM-5.2",
        messages=[{"role": "user", "content": "hi"}],
        provider_profile=p,
        reasoning_config={"enabled": True, "effort": "high"},
    )
    assert kwargs["extra_body"]["thinking"] == {"type": "enabled", "effort": "high"}
    # No top-level reasoning_effort (Wafer has no such field — effort lives
    # inside the thinking object, unlike Kimi).
    assert "reasoning_effort" not in kwargs


def test_wafer_transport_thinking_disabled():
    """reasoning_config.enabled=False → thinking:{type:disabled}."""
    p = _wafer_profile()
    transport = ChatCompletionsTransport()
    kwargs = transport.build_kwargs(
        model="GLM-5.2",
        messages=[{"role": "user", "content": "hi"}],
        provider_profile=p,
        reasoning_config={"enabled": False},
    )
    assert kwargs["extra_body"]["thinking"] == {"type": "disabled"}


def test_wafer_transport_thinking_effort_none():
    """effort='none' is treated as disabled (mirrors Kimi/custom profiles)."""
    p = _wafer_profile()
    transport = ChatCompletionsTransport()
    kwargs = transport.build_kwargs(
        model="GLM-5.2",
        messages=[{"role": "user", "content": "hi"}],
        provider_profile=p,
        reasoning_config={"enabled": True, "effort": "none"},
    )
    assert kwargs["extra_body"]["thinking"] == {"type": "disabled"}


def test_wafer_alias_resolution():
    """wafer-ai and waferai resolve to the wafer canonical profile."""
    for alias in ("wafer-ai", "waferai"):
        p = get_provider_profile(alias)
        assert p is not None, f"alias {alias!r} did not resolve"
        assert p.name == "wafer"
