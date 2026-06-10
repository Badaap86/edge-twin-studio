"""EdgeTwin V107 AI Copilot Adapter helper.

This file intentionally contains no live provider call and no API key.
It defines a safe boundary for a future AI provider integration.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List


@dataclass
class CopilotRequestV107:
    customer_message: str
    desired_outcome: str
    allowed_tasks: List[str]
    provider_mode: str = "offline_rules"
    include_raw_customer_data: bool = False


def build_provider_payload(request: CopilotRequestV107, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Create a provider-agnostic payload for later OpenAI/Paddle/manual adapters.

    API keys and secrets must be supplied by the runtime environment, not stored here.
    """
    return {
        "schema_version": "v107",
        "request": asdict(request),
        "context": context or {},
        "rules": {
            "ai_may": ["summarize", "suggest_pack", "draft_safe_copy", "flag_risky_claims"],
            "ai_must_not": ["approve_payment", "guarantee_accuracy", "make_legal_claims", "delete_data", "serve_downloads"],
            "source_of_truth": ["policy_engine", "payment_webhook", "fulfillment_state_machine"],
        },
    }
