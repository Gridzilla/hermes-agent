"""``hermes usage`` subcommand parser.

Shows current token usage against hourly/weekly allowance across configured
providers (Z.AI, OpenAI Codex, Nous). Data comes from the standalone collector
at ``~/.hermes/scripts/usage_collect.py``.

Handler injected to avoid importing ``main`` (mirrors the insights pattern).
"""

from __future__ import annotations

from typing import Callable


def build_usage_parser(subparsers, *, cmd_usage: Callable) -> None:
    """Attach the ``usage`` subcommand to ``subparsers``."""
    usage_parser = subparsers.add_parser(
        "usage",
        help="Show token usage against hourly/weekly allowance",
        description=(
            "Query current token usage and allowance consumption across Z.AI, "
            "OpenAI Codex, and Nous. Z.AI and Codex data comes from their native "
            "usage APIs; Nous uses local accounting from session history."
        ),
    )
    usage_parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of the human-readable report",
    )
    usage_parser.add_argument(
        "--zai-only", action="store_true", help="Show Z.AI / GLM usage only"
    )
    usage_parser.add_argument(
        "--codex-only", action="store_true", help="Show OpenAI Codex usage only"
    )
    usage_parser.add_argument(
        "--record",
        action="store_true",
        help="Also persist this snapshot to the usage history database",
    )
    usage_parser.set_defaults(func=cmd_usage)
