"""EdgeTwin Studio commercial readiness pack.

Purpose:
- Keep EdgeTwin positioned as an industrial AI readiness/evidence product.
- Make the offer simple: Professional Pilot Pack first, Starter as qualifier,
  Real-Data Evidence Pack as the deeper proof offer.
- Add practical privacy/data-minimisation and Buyer Data Room checklists.

This module is deterministic product copy/support logic. It is not legal advice.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from typing import Any, Dict, List

MODULE = "Commercial Readiness Pack"

PRIMARY_POSITIONING = (
    "EdgeTwin Studio helps industrial teams quickly and defensibly assess whether "
    "machine, sensor, maintenance or production data is ready for a controlled "
    "AI / Edge AI / predictive-maintenance pilot."
)

SHORT_PITCH = (
    "Proof before implementation. EdgeTwin delivers a pilot-readiness evidence pack "
    "with data-quality checks, assumptions, limitations, risks and a clear next step."
)

SAFE_EXTERNAL_CLAIM = (
    "EdgeTwin assesses AI/predictive-maintenance pilot readiness. It does not sell "
    "production accuracy guarantees, legal approval, compliance certification or "
    "safety certification through the automatic pack flow."
)

NOT_POSITIONED_AS = [
    "a full predictive-maintenance production platform",
    "a certified safety/compliance product",
    "a legal approval tool",
    "a replacement for human engineering or management oversight",
    "a guarantee that failures will be predicted in production",
]

PACK_LADDER = [
    {
        "pack": "Starter Diagnostic Pack",
        "first_customer_price_eur": 750,
        "role": "Low-friction qualification step",
        "best_for": "Customer has an idea or small sample and needs quick direction.",
        "deliverables": [
            "first readiness screen",
            "obvious data gaps",
            "basic risk notes",
            "recommended next step",
        ],
    },
    {
        "pack": "Professional Pilot Pack",
        "first_customer_price_eur": 1500,
        "future_price_eur": 2500,
        "role": "Hero offer / main first sale",
        "best_for": "Customer wants a management-ready decision before spending on a bigger AI project.",
        "deliverables": [
            "data quality scorecard",
            "pilot-readiness decision",
            "assumptions and limitations",
            "risk register",
            "go / conditional go / no-go recommendation",
            "next-step roadmap",
        ],
    },
    {
        "pack": "Real-Data Evidence Pack",
        "first_customer_price_eur": 3500,
        "future_price_eur": 5500,
        "role": "Deeper proof offer / upsell",
        "best_for": "Customer can share real exports from sensors, machines, historian, SCADA, CMMS/EAM or maintenance records.",
        "deliverables": [
            "deeper real-data quality assessment",
            "signal/use-case fit notes",
            "maintenance/failure history alignment where available",
            "stronger evidence room",
            "pilot-to-production readiness checklist",
        ],
    },
    {
        "pack": "Premium Custom Pack",
        "first_customer_price_eur": 7500,
        "future_price_eur": 15000,
        "role": "Bounded premium custom work",
        "best_for": "Customer needs multiple data sources, multiple use-case variants, reusable template, or management-ready custom package.",
        "deliverables": [
            "bounded custom scope",
            "fixed modules",
            "safe claim boundaries",
            "premium Buyer Data Room",
            "clear exclusions and change rules",
        ],
    },
]

BEST_FIRST_ICP = [
    "asset-heavy mid-market manufacturers",
    "machine builders with customer/field data",
    "maintenance or reliability teams with downtime pain",
    "production teams with sensor/historian/SCADA/PLC/CSV exports",
    "operations teams that want AI but do not yet know if the data is good enough",
]

BUYER_DATA_ROOM_SECTIONS = [
    {"section": "Executive decision summary", "purpose": "One-page go / conditional go / no-go decision for management."},
    {"section": "Data scope and source map", "purpose": "Shows which files/sources were assessed and what was excluded."},
    {"section": "Data quality scorecard", "purpose": "Completeness, continuity, timestamp health, label/failure-history quality and fitness-for-use."},
    {"section": "Use-case fit", "purpose": "Assesses whether the desired AI/predictive-maintenance pilot is realistic with the current data."},
    {"section": "Assumptions", "purpose": "Documents what EdgeTwin had to assume because the available data was incomplete."},
    {"section": "Limitations", "purpose": "Makes clear what cannot be claimed yet."},
    {"section": "Risk register", "purpose": "Lists data, technical, privacy, operational and commercial risks."},
    {"section": "Safe claims", "purpose": "Gives customer-safe wording for internal sharing."},
    {"section": "Missing inputs", "purpose": "Explains what data or context must be improved next."},
    {"section": "Next-step roadmap", "purpose": "Gives the lowest-risk next move: improve data, run controlled pilot, or pause."},
]

DATA_MINIMISATION_RULES = [
    "Upload only the data needed for the readiness/evidence assessment.",
    "Remove or pseudonymise names, email addresses, phone numbers, employee IDs and customer identifiers where possible.",
    "Prefer machine IDs, asset IDs, timestamps, sensor values, event labels and maintenance/failure dates over personal data.",
    "Do not upload payroll, HR, medical, private customer or unrelated commercial data.",
    "Use small representative samples first when the full dataset is large or sensitive.",
    "Define a retention period before sharing customer data.",
    "Do not reuse customer data for templates or benchmarks without explicit written permission.",
    "Keep downloads and delivery links limited, logged and time-bounded.",
]

FIRST_CUSTOMER_ACTIONS = [
    "Finish company/sole-proprietor registration and business bank/payment route.",
    "Keep EdgeTwin in Request quote / Request payment link mode until payments are legally ready.",
    "Use the LinkedIn warm-network launch to collect introductions, not hard-sell immediately.",
    "Lead with the Professional Pilot Pack as the main offer.",
    "Use Starter only for low-budget or uncertain prospects.",
    "Use Real-Data Evidence Pack when the customer already has useful machine/sensor/maintenance exports.",
    "Collect the first 1-3 customer cases before raising prices.",
    "Do not add more random features until first customer feedback is collected.",
]


def _now() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8", errors="replace")).hexdigest()


def get_commercial_readiness_snapshot() -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "module": MODULE,
        "created_at": _now(),
        "product_category": "Industrial AI readiness & evidence pack",
        "primary_positioning": PRIMARY_POSITIONING,
        "short_pitch": SHORT_PITCH,
        "safe_external_claim": SAFE_EXTERNAL_CLAIM,
        "not_positioned_as": NOT_POSITIONED_AS,
        "best_first_icp": BEST_FIRST_ICP,
        "hero_offer": "Professional Pilot Pack",
        "pack_ladder": PACK_LADDER,
        "buyer_data_room_sections": BUYER_DATA_ROOM_SECTIONS,
        "data_minimisation_rules": DATA_MINIMISATION_RULES,
        "first_customer_actions": FIRST_CUSTOMER_ACTIONS,
        "pricing_note": "First-customer prices are intentionally accessible. Raise after 1-3 real cases and proof of delivery quality.",
    }
    snapshot["snapshot_hash"] = _sha(snapshot)
    return snapshot


def render_markdown(snapshot: Dict[str, Any] | None = None) -> str:
    snapshot = snapshot or get_commercial_readiness_snapshot()
    lines: List[str] = [
        "# EdgeTwin Studio Commercial Readiness Pack",
        "",
        "## Positioning",
        snapshot["primary_positioning"],
        "",
        "## Short pitch",
        snapshot["short_pitch"],
        "",
        "## Safe external claim",
        snapshot["safe_external_claim"],
        "",
        "## Not positioned as",
    ]
    lines.extend([f"- {x}" for x in snapshot.get("not_positioned_as", [])])
    lines.extend(["", "## Pack ladder"])
    for p in snapshot.get("pack_ladder", []):
        price = p.get("first_customer_price_eur")
        future = p.get("future_price_eur")
        price_line = f"EUR {price}" + (f" now, later EUR {future}+" if future else "")
        lines.extend([
            f"### {p.get('pack')}",
            f"- Price: {price_line}",
            f"- Role: {p.get('role')}",
            f"- Best for: {p.get('best_for')}",
            "- Deliverables:",
        ])
        lines.extend([f"  - {d}" for d in p.get("deliverables", [])])
        lines.append("")
    lines.extend(["## Buyer Data Room sections"])
    for s in snapshot.get("buyer_data_room_sections", []):
        lines.append(f"- **{s.get('section')}** — {s.get('purpose')}")
    lines.extend(["", "## Data minimisation rules"])
    lines.extend([f"- {x}" for x in snapshot.get("data_minimisation_rules", [])])
    lines.extend(["", "## First customer actions"])
    lines.extend([f"- {x}" for x in snapshot.get("first_customer_actions", [])])
    return "\n".join(lines) + "\n"
