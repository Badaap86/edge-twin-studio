"""EdgeTwin V105 private delivery endpoint helper.

Framework-light delivery decision helpers for a future backend endpoint.
No card data. No raw storage paths.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

@dataclass
class DeliveryDecision:
    allowed: bool
    http_status: int
    reason: str
    file_id: Optional[str] = None
    order_id: Optional[str] = None

def decide_private_delivery(token_validation: Dict[str, Any], order_lookup: Callable[[str], Optional[Dict[str, Any]]], require_ready_delivery: bool = True) -> DeliveryDecision:
    if not token_validation.get("valid"):
        reason = str(token_validation.get("reason", "invalid_token"))
        return DeliveryDecision(False, 410 if reason == "expired" else 401, reason)
    payload = token_validation.get("payload") or {}
    order_id = str(payload.get("order_id", ""))
    file_id = str(payload.get("file_id", ""))
    order = order_lookup(order_id) or {}
    payment_status = str(order.get("payment_status", "")).lower()
    delivery_status = str(order.get("delivery_status", "")).lower()
    allowed_files = set(order.get("allowed_file_ids", []))
    if payment_status in {"failed", "refunded", "disputed", "cancelled"}:
        return DeliveryDecision(False, 403, "payment_status_lock", file_id, order_id)
    if payment_status not in {"paid", "confirmed"}:
        return DeliveryDecision(False, 402, "payment_required", file_id, order_id)
    if require_ready_delivery and delivery_status not in {"ready", "delivered", "unlocked"}:
        return DeliveryDecision(False, 423, "delivery_not_ready", file_id, order_id)
    if allowed_files and file_id not in allowed_files:
        return DeliveryDecision(False, 404, "file_not_allowed_for_order", file_id, order_id)
    return DeliveryDecision(True, 200, "private_stream_allowed", file_id, order_id)

def safe_audit_row(decision: DeliveryDecision, customer_hash: str = "unknown") -> Dict[str, Any]:
    return {"allowed": decision.allowed, "http_status": decision.http_status, "reason": decision.reason, "order_id": decision.order_id, "file_id": decision.file_id, "customer_hash": customer_hash, "raw_path_exposed": False, "card_data_stored": False}
