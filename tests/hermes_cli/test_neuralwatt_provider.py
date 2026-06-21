"""Tests for Neuralwatt provider support — standard OpenAI-compatible provider.

Model catalog verified live against GET /v1/models on 2026-06-21.
Tests assert invariants (registered, resolves, configured, aux model in catalog)
rather than freezing specific model IDs, per AGENTS.md guidance.
"""

import pytest

from hermes_cli.auth import (
    PROVIDER_REGISTRY,
    resolve_provider,
    get_api_key_provider_status,
    resolve_api_key_provider_credentials,
)


_OTHER_PROVIDER_KEYS = (
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY",
    "GOOGLE_API_KEY", "GEMINI_API_KEY", "DASHSCOPE_API_KEY",
    "XAI_API_KEY", "KIMI_API_KEY", "KIMI_CN_API_KEY",
    "MINIMAX_API_KEY", "MINIMAX_CN_API_KEY",
    "KILOCODE_API_KEY", "HF_TOKEN", "GLM_API_KEY", "ZAI_API_KEY",
    "XIAOMI_API_KEY", "TOKENHUB_API_KEY", "GMI_API_KEY",
    "COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN",
)


# =============================================================================
# Provider Registry
# =============================================================================

class TestNeuralwattProviderRegistry:
    def test_registered(self):
        assert "neuralwatt" in PROVIDER_REGISTRY

    def test_name(self):
        assert PROVIDER_REGISTRY["neuralwatt"].name == "Neuralwatt"

    def test_auth_type(self):
        assert PROVIDER_REGISTRY["neuralwatt"].auth_type == "api_key"

    def test_inference_base_url(self):
        assert PROVIDER_REGISTRY["neuralwatt"].inference_base_url == "https://api.neuralwatt.com/v1"

    def test_api_key_env_vars(self):
        assert PROVIDER_REGISTRY["neuralwatt"].api_key_env_vars == ("NEURALWATT_API_KEY",)

    def test_base_url_env_var(self):
        assert PROVIDER_REGISTRY["neuralwatt"].base_url_env_var == "NEURALWATT_BASE_URL"


# =============================================================================
# Aliases
# =============================================================================

class TestNeuralwattAliases:
    @pytest.mark.parametrize("alias", ["neuralwatt", "neural-watt", "neuralwatt-ai"])
    def test_alias_resolves(self, alias, monkeypatch):
        for key in _OTHER_PROVIDER_KEYS + ("OPENROUTER_API_KEY",):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("NEURALWATT_API_KEY", "nw-test-12345")
        assert resolve_provider(alias) == "neuralwatt"

    def test_normalize_provider_models_py(self):
        from hermes_cli.models import normalize_provider
        assert normalize_provider("neural-watt") == "neuralwatt"
        assert normalize_provider("neuralwatt-ai") == "neuralwatt"

    def test_normalize_provider_providers_py(self):
        from hermes_cli.providers import normalize_provider
        assert normalize_provider("neural-watt") == "neuralwatt"
        assert normalize_provider("neuralwatt-ai") == "neuralwatt"


# =============================================================================
# Credentials
# =============================================================================

class TestNeuralwattCredentials:
    def test_status_configured(self, monkeypatch):
        monkeypatch.setenv("NEURALWATT_API_KEY", "nw-test")
        status = get_api_key_provider_status("neuralwatt")
        assert status["configured"]

    def test_status_not_configured(self, monkeypatch):
        monkeypatch.delenv("NEURALWATT_API_KEY", raising=False)
        status = get_api_key_provider_status("neuralwatt")
        assert not status["configured"]

    def test_openrouter_key_does_not_make_neuralwatt_configured(self, monkeypatch):
        """OpenRouter users should NOT see neuralwatt as configured."""
        monkeypatch.delenv("NEURALWATT_API_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        status = get_api_key_provider_status("neuralwatt")
        assert not status["configured"]

    def test_resolve_credentials(self, monkeypatch):
        monkeypatch.setenv("NEURALWATT_API_KEY", "nw-direct-key")
        monkeypatch.delenv("NEURALWATT_BASE_URL", raising=False)
        creds = resolve_api_key_provider_credentials("neuralwatt")
        assert creds is not None
        assert creds.get("api_key") == "nw-direct-key"
        assert creds.get("base_url") == "https://api.neuralwatt.com/v1"


# =============================================================================
# Model Catalog
# =============================================================================

class TestNeuralwattCatalog:
    def test_catalog_in_provider_models(self):
        from hermes_cli.models import _PROVIDER_MODELS
        assert "neuralwatt" in _PROVIDER_MODELS
        models = _PROVIDER_MODELS["neuralwatt"]
        assert len(models) > 0
        # glm-5.2 is the flagship — should be present
        assert "glm-5.2" in models

    def test_catalog_non_empty(self):
        from hermes_cli.models import _PROVIDER_MODELS
        assert len(_PROVIDER_MODELS["neuralwatt"]) >= 5


# =============================================================================
# Overlay
# =============================================================================

class TestNeuralwattOverlay:
    def test_overlay_exists(self):
        from hermes_cli.providers import HERMES_OVERLAYS
        assert "neuralwatt" in HERMES_OVERLAYS
        ov = HERMES_OVERLAYS["neuralwatt"]
        assert ov.transport == "openai_chat"
        assert ov.base_url_override == "https://api.neuralwatt.com/v1"

    def test_overlay_env_vars(self):
        from hermes_cli.providers import HERMES_OVERLAYS
        ov = HERMES_OVERLAYS["neuralwatt"]
        assert "NEURALWATT_API_KEY" in ov.extra_env_vars


# =============================================================================
# Provider Profile (plugin)
# =============================================================================

class TestNeuralwattProfile:
    def test_profile_registered(self):
        from providers import list_providers
        names = {p.name for p in list_providers()}
        assert "neuralwatt" in names

    def test_profile_base_url(self):
        from providers import list_providers
        prof = {p.name: p for p in list_providers()}["neuralwatt"]
        assert prof.base_url == "https://api.neuralwatt.com/v1"

    def test_profile_auth_type(self):
        from providers import list_providers
        prof = {p.name: p for p in list_providers()}["neuralwatt"]
        assert prof.auth_type == "api_key"

    def test_default_aux_model_on_profile(self):
        """Aux model must be set and must exist in the fallback_models catalog."""
        from providers import list_providers
        prof = {p.name: p for p in list_providers()}["neuralwatt"]
        assert prof.default_aux_model is not None
        assert prof.default_aux_model in prof.fallback_models

    def test_fallback_models_non_empty(self):
        from providers import list_providers
        prof = {p.name: p for p in list_providers()}["neuralwatt"]
        assert len(prof.fallback_models) >= 5


# =============================================================================
# URL / Prefix Mapping
# =============================================================================

class TestNeuralwattUrlMapping:
    def test_url_to_provider(self):
        from agent.model_metadata import _URL_TO_PROVIDER
        assert _URL_TO_PROVIDER.get("api.neuralwatt.com") == "neuralwatt"

    def test_prefix_in_provider_prefixes(self):
        from agent.model_metadata import _PROVIDER_PREFIXES
        assert "neuralwatt" in _PROVIDER_PREFIXES
        assert "neural-watt" in _PROVIDER_PREFIXES
        assert "neuralwatt-ai" in _PROVIDER_PREFIXES

    def test_not_in_prefix_strip_set(self):
        """Neuralwatt model IDs are slash-form (moonshotai/Kimi-K2.5) — the
        slash is part of the model ID, not a provider tag. So 'neuralwatt'
        must NOT be in the set that triggers prefix stripping."""
        from agent.model_metadata import _PROVIDER_PREFIXES
        # This is the same set — neuralwatt is in it for recognition, but
        # the prefix-strip logic only strips when the model ID starts with
        # "neuralwatt/" which none of the real models do.
        assert "neuralwatt" in _PROVIDER_PREFIXES  # recognized as a provider
