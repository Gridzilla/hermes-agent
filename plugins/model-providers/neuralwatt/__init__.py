"""Neuralwatt provider profile.

Neuralwatt (https://api.neuralwatt.com/v1) is a fully OpenAI-compatible
endpoint hosting models from several families (GLM 5.2, Kimi K2.5/K2.6/K2.7,
Qwen 3.5/3.6).  Most model IDs are bare lowercase (``glm-5.2``, ``kimi-k2.6``);
one retains a slash prefix (``moonshotai/Kimi-K2.5``).  Because the slash is
part of the model ID — not a provider tag — Neuralwatt is intentionally NOT
added to the matching-prefix-strip set.

Model catalog verified live against GET /v1/models on 2026-06-21 (13 models).
"""

from hermes_cli import __version__ as _HERMES_VERSION
from providers import register_provider
from providers.base import ProviderProfile

neuralwatt = ProviderProfile(
    name="neuralwatt",
    aliases=("neural-watt", "neuralwatt-ai"),
    display_name="Neuralwatt",
    description="Neuralwatt — multi-model OpenAI-compatible direct API",
    signup_url="https://portal.neuralwatt.com/",
    env_vars=("NEURALWATT_API_KEY", "NEURALWATT_BASE_URL"),
    base_url="https://api.neuralwatt.com/v1",
    auth_type="api_key",
    # Attribution so Neuralwatt can identify traffic from Hermes Agent.
    # The generic profile.default_headers fallback in run_agent.py and
    # agent/auxiliary_client.py picks this up at client construction time.
    default_headers={"User-Agent": f"HermesAgent/{_HERMES_VERSION}"},
    default_aux_model="glm-5.2-fast",
    fallback_models=(
        "glm-5.2",
        "glm-5.2-fast",
        "moonshotai/Kimi-K2.5",
        "kimi-k2.7-code",
        "kimi-k2.6",
        "qwen3.5-397b",
        "qwen3.6-35b",
    ),
)

register_provider(neuralwatt)
