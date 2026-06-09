"""EdgeTwin Studio V102 payment adapter helpers.

This module intentionally avoids storing or processing card data. It normalizes
provider/manual payment events into the small fields EdgeTwin needs for unlock.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class PaymentEventV102:
    provider: str
    event_type: str
    provider_reference: str
    order_id: str
    payment_status: str
    amount_eur: float
    customer_email: str = ""
    webhook_verified: bool = False

    def to_safe_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_payment_event(payload: Dict[str, Any]) -> PaymentEventV102:
    """Normalize a provider/manual payment payload into a safe EdgeTwin event."""
    payload = payload or {}
    return PaymentEventV102(
        provider=str(payload.get("provider", "manual")).strip().lower(),
        event_type=str(payload.get("event_type", payload.get("type", "manual.confirmed"))).strip(),
        provider_reference=str(payload.get("provider_reference", payload.get("session_id", payload.get("transaction_id", "")))).strip(),
        order_id=str(payload.get("order_id", payload.get("client_reference_id", ""))).strip(),
        payment_status=str(payload.get("payment_status", payload.get("status", "unpaid"))).strip().lower(),
        amount_eur=float(payload.get("amount_eur", payload.get("amount", 0)) or 0),
        customer_email=str(payload.get("customer_email", payload.get("email", ""))).strip(),
        webhook_verified=bool(payload.get("webhook_verified", False)),
    )
