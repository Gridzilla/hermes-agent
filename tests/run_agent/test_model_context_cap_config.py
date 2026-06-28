"""Tests for model.max_context_length as a context-window cap alias."""
from unittest.mock import patch

from run_agent import AIAgent


def _make_agent(model_cfg):
    cfg = {"agent": {}, "model": model_cfg}
    with patch("run_agent.OpenAI"), \
         patch("hermes_cli.config.load_config", return_value=cfg):
        return AIAgent(
            api_key="test-key",
            base_url="https://api.z.ai/api/paas/v4",
            provider="zai",
            model=model_cfg.get("default", "glm-5.2"),
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )


def test_model_max_context_length_caps_zai_native_default():
    """GLM-5.2 defaults to 1M, but config can cap Hermes lower."""
    agent = _make_agent({
        "provider": "zai",
        "default": "glm-5.2",
        "max_context_length": 262144,
    })

    assert getattr(agent, "_config_context_length") == 262144
    assert getattr(agent, "context_compressor").context_length == 262144


def test_model_context_length_precedence_over_max_context_length_alias():
    """The existing context_length key remains authoritative if both are set."""
    agent = _make_agent({
        "provider": "zai",
        "default": "glm-5.2",
        "context_length": 131072,
        "max_context_length": 262144,
    })

    assert getattr(agent, "_config_context_length") == 131072
    assert getattr(agent, "context_compressor").context_length == 131072
