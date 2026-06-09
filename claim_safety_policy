"""EdgeTwin Studio V108 claim safety and prompt policy layer.

This module is intentionally rule-first. It is not legal advice and it does not
certify compliance. It creates a conservative policy gate around AI/copilot text,
sales claims, quote copy and customer-facing report language.
"""
from __future__ import annotations

import datetime
import re
from typing import Any, Dict, List, Tuple

V108_VERSION = "V108"

FORBIDDEN_CLAIM_PATTERNS: List[Dict[str, Any]] = [
    {
        "id": "accuracy_guarantee",
        "severity": "block",
        "patterns": [
            r"\b100\s*%\s*(accurate|accuracy|accuraat|zeker)\b",
            r"\b99\s*%\s*(accurate|accuracy|accuraat)\b",
            r"\bguarantee(?:d|s)?\b.*\b(accuracy|accurate|detect|predict|voorspel|detectie)\b",
            r"\bgarandeer(?:d|t)?\b.*\b(accuraat|detectie|voorspel|storing)\b",
            r"\baltijd\b.*\b(voorspel|detecteer|vinden|detectie)\b",
        ],
        "reason": "Accuracy guarantees require representative labelled field validation and acceptance criteria.",
        "safe_rewrite": "We can measure pilot accuracy once representative labelled field data is available; this pack does not guarantee production accuracy.",
    },
    {
        "id": "production_ready_guarantee",
        "severity": "block",
        "patterns": [
            r"\bproduction[- ]?ready\b",
            r"\bproductie[- ]?klaar\b",
            r"\bready for production\b",
            r"\bdirect in productie\b",
            r"\bgo[- ]?live guaranteed\b",
        ],
        "reason": "Production deployment needs real-data validation, security/privacy review and customer approval.",
        "safe_rewrite": "This pack prepares a controlled pilot and production-handoff checklist; production use requires additional validation and approval.",
    },
    {
        "id": "legal_or_compliance_certification",
        "severity": "block",
        "patterns": [
            r"\blegal(?:ly)? approved\b",
            r"\bjuridisch goedgekeurd\b",
            r"\bcompliance certified\b",
            r"\bcompliance[- ]?gecertificeerd\b",
            r"\bgdpr compliant guaranteed\b",
            r"\bai act compliant guaranteed\b",
            r"\bce certified\b",
            r"\bveiligheids[- ]?gecertificeerd\b",
            r"\bsafety certified\b",
        ],
        "reason": "Legal/compliance/certification claims need qualified external review and formal evidence.",
        "safe_rewrite": "EdgeTwin is designed to be compliance-aware, but this pack is not a legal/compliance certification.",
    },
    {
        "id": "human_replacement_or_full_autonomy",
        "severity": "review",
        "patterns": [
            r"\breplaces? human\b",
            r"\bno human review\b",
            r"\bfully autonomous\b",
            r"\bvolledig autonoom\b",
            r"\bvervangt menselijke controle\b",
        ],
        "reason": "Critical operational decisions should retain human oversight unless formally validated.",
        "safe_rewrite": "EdgeTwin supports decision-making and automation with clear review gates for high-risk actions.",
    },
    {
        "id": "liability_or_risk_transfer",
        "severity": "block",
        "patterns": [
            r"\bwe accept liability\b",
            r"\bwij nemen aansprakelijkheid\b",
            r"\bno risk for customer\b",
            r"\bgeen risico voor klant\b",
        ],
        "reason": "Liability terms belong in contract/legal review, not automatic pack copy.",
        "safe_rewrite": "Commercial and liability terms must be agreed separately in the order/SOW.",
    },
]

SAFE_CLAIM_LIBRARY: List[Dict[str, str]] = [
    {
        "name": "pilot_evidence_pack",
        "claim": "EdgeTwin prepares a controlled pilot/evidence pack with data-quality checks, trust notes, assumptions, limitations and concrete next steps.",
        "use_when": "General pack, quote, marketplace and customer-summary copy.",
    },
    {
        "name": "decision_support",
        "claim": "EdgeTwin supports anomaly screening and decision support when the provided data is suitable for the selected pack.",
        "use_when": "Analysis result and operational-readiness language.",
    },
    {
        "name": "accuracy_boundary",
        "claim": "Accuracy can be evaluated only after representative labelled field data and acceptance criteria are available.",
        "use_when": "Customer asks about accuracy, detection rate or predictive claims.",
    },
    {
        "name": "production_boundary",
        "claim": "Production deployment requires additional validation, security/privacy review and customer approval.",
        "use_when": "Customer asks whether this can go directly live.",
    },
    {
        "name": "compliance_boundary",
        "claim": "EdgeTwin is compliance-aware, but this deliverable is not a legal, safety or compliance certification.",
        "use_when": "Compliance, GDPR, AI Act, CE or certification questions.",
    },
]

PROMPT_POLICY_RULES: List[Dict[str, Any]] = [
    {
        "rule": "AI drafts only",
        "required": True,
        "description": "AI may summarize, draft and recommend; EdgeTwin state machines and policy gates remain source of truth.",
    },
    {
        "rule": "No hard accuracy claims",
        "required": True,
        "description": "Do not claim guaranteed accuracy, guaranteed detection or guaranteed prediction.",
    },
    {
        "rule": "No production-ready claims",
        "required": True,
        "description": "Use pilot-ready / production-handoff candidate language unless validated otherwise.",
    },
    {
        "rule": "No legal/compliance/certification claims",
        "required": True,
        "description": "Compliance-aware language is allowed; certified/legal-approved language is blocked.",
    },
    {
        "rule": "No card/payment data",
        "required": True,
        "description": "Payment details stay with Stripe/Paddle/manual provider. EdgeTwin stores only payment status metadata.",
    },
    {
        "rule": "Redact customer data by default",
        "required": True,
        "description": "Prompts should use summaries/metadata unless explicit data policy allows more.",
    },
]

ALLOWED_AI_TASKS = [
    "summarize_customer_request",
    "suggest_pack",
    "suggest_addons",
    "draft_safe_quote_text",
    "draft_missing_input_questions",
    "draft_delivery_message",
    "flag_risky_claims",
    "rewrite_to_safe_claims",
]

BLOCKED_AI_TASKS = [
    "approve_payment",
    "unlock_private_download",
    "sign_contract",
    "delete_customer_data",
    "guarantee_accuracy",
    "claim_legal_or_compliance_certification",
    "declare_production_ready",
    "accept_liability",
]


def _now() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def scan_claim_text(text: Any) -> Dict[str, Any]:
    """Scan customer-facing language for risky claims."""
    raw = str(text or "")
    lowered = raw.lower()
    flags: List[Dict[str, Any]] = []
    seen = set()
    for rule in FORBIDDEN_CLAIM_PATTERNS:
        for pat in rule["patterns"]:
            if re.search(pat, lowered, flags=re.IGNORECASE):
                key = (rule["id"], pat)
                if key in seen:
                    continue
                seen.add(key)
                flags.append({
                    "claim_id": rule["id"],
                    "severity": rule["severity"],
                    "pattern": pat,
                    "reason": rule["reason"],
                    "safe_rewrite": rule["safe_rewrite"],
                })
    blocked = any(f["severity"] == "block" for f in flags)
    review = bool(flags) and not blocked
    if blocked:
        decision = "BLOCKED CLAIM - REWRITE REQUIRED"
        risk_level = "high"
    elif review:
        decision = "CLAIM NEEDS REVIEW"
        risk_level = "medium"
    else:
        decision = "CLAIM SAFE FOR STANDARD PACK COPY"
        risk_level = "low"
    score = 100 - sum(30 if f["severity"] == "block" else 15 for f in flags)
    score = max(0, min(100, score))
    return {
        "text_scanned_preview": raw[:500],
        "flags": flags,
        "flag_count": len(flags),
        "blocked": blocked,
        "review_required": review,
        "risk_level": risk_level,
        "claim_score": int(score),
        "decision": decision,
    }


def build_safe_rewrite(original_text: Any, scan: Dict[str, Any] | None = None) -> str:
    scan = scan or scan_claim_text(original_text)
    if not scan.get("flags"):
        return _clean_text(original_text) or SAFE_CLAIM_LIBRARY[0]["claim"]
    lines = [
        "EdgeTwin prepares a controlled pilot/evidence pack with data-quality checks, trust notes, assumptions, limitations and concrete next steps.",
        "It supports anomaly screening and decision support when the provided data is suitable for the selected pack.",
        "It does not guarantee production accuracy, legal/compliance certification or direct production readiness without additional validation and approval.",
    ]
    return " ".join(lines)


def build_claim_safety_prompt_policy_snapshot(
    project_name: str = "EdgeTwin Project",
    proposed_customer_claim: str = "",
    customer_context: str = "",
    pack_type: str = "Professional Pilot Pack",
    channel: str = "quote_or_report",
    copilot_snapshot: Dict[str, Any] | None = None,
    allow_external_ai: bool = False,
    include_raw_customer_data: bool = False,
) -> Dict[str, Any]:
    copilot_snapshot = copilot_snapshot or {}
    base_claim = proposed_customer_claim or copilot_snapshot.get("safe_customer_claim") or copilot_snapshot.get("structured_output", {}).get("safe_customer_claim") or SAFE_CLAIM_LIBRARY[0]["claim"]
    combined_text = "\n".join([str(base_claim), str(customer_context or "")])
    scan = scan_claim_text(combined_text)
    safe_rewrite = build_safe_rewrite(base_claim, scan)

    extra_flags: List[Dict[str, Any]] = []
    if allow_external_ai and include_raw_customer_data:
        extra_flags.append({
            "claim_id": "raw_data_external_ai",
            "severity": "review",
            "reason": "External AI with raw customer data needs explicit data-policy approval.",
            "safe_rewrite": "Use redacted summaries/metadata unless data policy allows raw customer data.",
        })
    if channel in {"public_homepage", "ad", "sales_page"} and scan.get("flags"):
        extra_flags.append({
            "claim_id": "public_claim_risk",
            "severity": "review",
            "reason": "Public marketing copy should use the strictest safe-claim library.",
            "safe_rewrite": "Use pilot/evidence-pack language instead of hard outcome claims.",
        })

    all_flags = list(scan.get("flags", [])) + extra_flags
    blocked = any(f.get("severity") == "block" for f in all_flags)
    review_required = bool(all_flags) and not blocked
    if blocked:
        final_decision = "BLOCK CUSTOMER CLAIM - SAFE REWRITE REQUIRED"
        auto_publish_allowed = False
        risk_level = "high"
    elif review_required:
        final_decision = "REVIEW CUSTOMER CLAIM BEFORE USE"
        auto_publish_allowed = False
        risk_level = "medium"
    else:
        final_decision = "SAFE CLAIM POLICY PASS"
        auto_publish_allowed = True
        risk_level = "low"

    prompt_guard = {
        "system_boundary": "You are EdgeTwin Copilot. Draft only. Never approve payment, legal/compliance claims, production readiness, accuracy guarantees, contracts, refunds or deletion.",
        "allowed_tasks": ALLOWED_AI_TASKS,
        "blocked_tasks": BLOCKED_AI_TASKS,
        "must_include_in_customer_copy": [
            "assumptions and limitations",
            "pilot/evidence-pack framing",
            "data-quality dependency",
            "production validation boundary",
        ],
        "must_not_include": [
            "100% accuracy",
            "guaranteed detection",
            "production-ready without validation",
            "legal/compliance certified",
            "replaces human review",
            "we accept liability",
        ],
        "output_contract": {
            "recommended_claim": "string",
            "risk_flags": "list[dict]",
            "safe_rewrite": "string",
            "founder_review_required": "bool",
            "auto_publish_allowed": "bool",
        },
    }

    score = int(scan.get("claim_score", 100))
    if extra_flags:
        score = max(0, score - 10 * len(extra_flags))
    return {
        "version": V108_VERSION,
        "module": "Claim Safety & Prompt Policy Pack",
        "created_at": _now(),
        "project_name": str(project_name),
        "pack_type": str(pack_type),
        "channel": str(channel),
        "proposed_customer_claim": str(base_claim),
        "customer_context_preview": str(customer_context or "")[:800],
        "scan": scan,
        "policy_flags": all_flags,
        "safe_rewrite": safe_rewrite,
        "safe_claim_library": SAFE_CLAIM_LIBRARY,
        "prompt_policy_rules": PROMPT_POLICY_RULES,
        "prompt_guard": prompt_guard,
        "auto_publish_allowed": bool(auto_publish_allowed),
        "founder_review_required": bool(blocked or review_required),
        "claim_score": int(max(0, min(100, score))),
        "risk_level": risk_level,
        "decision": final_decision,
        "source_of_truth": [
            "V95 Policy Approval Engine",
            "V96 Pricing & Assurance OS",
            "V102 Payment Provider Adapter",
            "V106 Order Fulfillment State Machine",
            "V108 Claim Safety & Prompt Policy Pack",
        ],
        "important_boundary": "V108 is a claim-safety and prompt-policy gate. It is not legal advice, compliance certification or production validation.",
    }
