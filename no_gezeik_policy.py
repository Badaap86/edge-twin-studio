"""EdgeTwin simple risk policy.

Purpose:
- Keep the customer buying flow simple and automatic.
- Block only the things that can create real legal, safety, payment or trust problems.
- Avoid exposing internal 'review' language to customers unless a human decision is truly needed.

This module does not provide legal advice. It provides deterministic product guardrails.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

POLICY_NAME = "Simple Risk Policy"

SAFE_VALUE_STATEMENT = (
    "EdgeTwin sells pilot/evidence and decision-support packs. If the data is not ready, "
    "the deliverable explains why, what is missing and which lower-risk next step is recommended."
)

SAFE_BOUNDARY_TEXT = (
    "EdgeTwin does not sell production accuracy guarantees, legal/compliance certification, "
    "safety certification, liability transfer or replacement of responsible human oversight."
)

# Blocks should be rare and clear. They protect both customer and seller.
BLOCK_RULES = [
    {
        "rule_id": "accuracy_guarantee",
        "label": "Accuracy guarantee",
        "patterns": [r"\b100\s*%\b", r"\bguarantee(?:d)?\b", r"\b99\s*%\b", r"always predict", r"never miss", r"zero false"],
        "customer_message": "We cannot sell guaranteed accuracy from this pack. Accuracy can only be measured after validation with representative labelled real-world data.",
        "safe_rewrite": "This pack helps evaluate whether the data is ready for a controlled pilot and defines what is needed to measure accuracy later.",
    },
    {
        "rule_id": "production_ready_claim",
        "label": "Production-ready claim",
        "patterns": [r"production[-\s]?ready", r"direct(?:ly)?\s+live", r"go live", r"production use", r"deploy immediately"],
        "customer_message": "We cannot automatically sell this as production-ready. Production use requires extra validation, customer approval and environment-specific checks.",
        "safe_rewrite": "This pack prepares a pilot/evidence bundle and a production-readiness checklist for the next controlled step.",
    },
    {
        "rule_id": "legal_compliance_certification",
        "label": "Legal/compliance/safety certification",
        "patterns": [r"legal approved", r"compliance approved", r"certified", r"ce certified", r"ai act compliant", r"safety certified", r"certification"],
        "customer_message": "We cannot automatically promise legal, compliance or safety certification. Formal certification requires a separate qualified review.",
        "safe_rewrite": "This pack is compliance-aware and documents assumptions, risks and evidence, but it is not a formal certification.",
    },
    {
        "rule_id": "liability_transfer",
        "label": "Liability/risk transfer",
        "patterns": [r"liability", r"liable", r"risk[-\s]?free", r"no risk", r"take responsibility", r"money back if"],
        "customer_message": "Liability, refund and commercial risk terms cannot be generated automatically inside the product flow.",
        "safe_rewrite": "Commercial terms, refunds and liability conditions must be handled separately from the automated pack output.",
    },
    {
        "rule_id": "safety_critical_autonomy",
        "label": "Safety-critical autonomy",
        "patterns": [r"safety critical", r"life critical", r"human safety", r"without operator", r"replace human", r"autonomous shutdown", r"automatically stop machines"],
        "customer_message": "Safety-critical autonomous decisions cannot be approved automatically. Human oversight and formal safety validation are required.",
        "safe_rewrite": "This pack supports decision-making and risk discovery with human oversight gates for critical actions.",
    },
]

PAYMENT_BLOCK_STATUSES = {"failed", "refunded", "disputed", "chargeback", "cancelled", "canceled"}

@dataclass
class PolicyDecision:
    decision: str
    customer_status: str
    blocked: bool
    matched_rules: List[Dict[str, Any]]
    safe_rewrites: List[str]
    customer_message: str
    value_statement: str
    boundary: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _matches(pattern: str, text: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def evaluate_simple_risk_policy(
    text: str = "",
    payment_status: str = "not_started",
    delivery_status: str = "not_started",
) -> Dict[str, Any]:
    """Return a deterministic, customer-friendly risk decision."""
    text = str(text or "")
    payment_status_l = str(payment_status or "not_started").lower().strip()

    matched: List[Dict[str, Any]] = []
    rewrites: List[str] = []

    for rule in BLOCK_RULES:
        hits = [p for p in rule["patterns"] if _matches(p, text)]
        if hits:
            matched.append({
                "rule_id": rule["rule_id"],
                "label": rule["label"],
                "customer_message": rule["customer_message"],
                "matched_patterns": hits,
            })
            rewrites.append(rule["safe_rewrite"])

    if payment_status_l in PAYMENT_BLOCK_STATUSES:
        matched.append({
            "rule_id": "payment_not_safe",
            "label": "Payment not safe",
            "customer_message": "Delivery/download is locked because the payment status is failed, refunded, disputed or cancelled.",
            "matched_patterns": [payment_status_l],
        })
        rewrites.append("Access can be restored only after payment status is confirmed as paid or manually approved.")

    blocked = bool(matched)
    if blocked:
        message = (
            "This request cannot continue automatically yet. EdgeTwin will not make unsafe promises. "
            "Use the safe rewrite or reduce the risky scope, then continue."
        )
        decision = "BLOCK_UNSAFE_PROMISE"
        customer_status = "Needs safe rewrite"
    else:
        message = "This request is inside the standard automatic custom-pack policy. Checkout/delivery can continue once payment and data steps are complete."
        decision = "AUTO_OK"
        customer_status = "Automatic"

    return PolicyDecision(
        decision=decision,
        customer_status=customer_status,
        blocked=blocked,
        matched_rules=matched,
        safe_rewrites=rewrites,
        customer_message=message,
        value_statement=SAFE_VALUE_STATEMENT,
        boundary=SAFE_BOUNDARY_TEXT,
    ).to_dict()


def get_customer_plain_rules() -> List[str]:
    return [
        "Normal custom packs are automatic.",
        "The customer gets a clear evidence pack, even if the answer is 'not ready yet'.",
        "Bad or incomplete data is not hidden; EdgeTwin explains what must improve.",
        "No production accuracy guarantees are sold automatically.",
        "No legal, compliance or safety certification is promised automatically.",
        "Refunds, disputes and failed payments lock delivery/download until resolved.",
        "Human involvement is only for true exceptions, not normal custom packs.",
        "Until company registration, invoice/VAT and payment provider are ready, use Request payment link / invoice mode instead of Pay now.",
    ]
