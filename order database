'''EdgeTwin V106 order database + fulfillment state machine.

Small payment-provider neutral order backbone.
It stores order/fulfillment status, not card details.
'''
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path('storage') / 'edgetwin_orders_v106.sqlite3'

ORDER_STATES = [
    'created', 'quoted', 'awaiting_payment', 'deposit_paid', 'paid',
    'intake_unlocked', 'data_received', 'generating', 'ready', 'delivered',
    'archived', 'blocked_review', 'refunded', 'disputed', 'cancelled'
]

TRANSITIONS = {
    'created': ['quoted', 'blocked_review', 'cancelled'],
    'quoted': ['awaiting_payment', 'blocked_review', 'cancelled'],
    'awaiting_payment': ['deposit_paid', 'paid', 'cancelled', 'blocked_review'],
    'deposit_paid': ['intake_unlocked', 'paid', 'refunded', 'disputed', 'blocked_review'],
    'paid': ['intake_unlocked', 'data_received', 'generating', 'ready', 'refunded', 'disputed', 'blocked_review'],
    'intake_unlocked': ['data_received', 'paid', 'blocked_review', 'cancelled'],
    'data_received': ['generating', 'blocked_review', 'cancelled'],
    'generating': ['ready', 'blocked_review'],
    'ready': ['delivered', 'blocked_review', 'refunded', 'disputed'],
    'delivered': ['archived', 'blocked_review', 'refunded', 'disputed'],
    'archived': [],
    'blocked_review': ['quoted', 'awaiting_payment', 'paid', 'generating', 'ready', 'cancelled'],
    'refunded': ['archived'],
    'disputed': ['blocked_review', 'archived'],
    'cancelled': ['archived'],
}

NO_CARD_DATA_FIELDS = {'card_number', 'cvc', 'cvv', 'iban_full', 'stripe_secret_key', 'paddle_secret_key'}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_status(value: Optional[str], default: str = 'created') -> str:
    value = str(value or default).strip().lower().replace(' ', '_')
    aliases = {
        'confirmed': 'paid', 'payment_confirmed': 'paid', 'unpaid': 'awaiting_payment',
        'failed': 'awaiting_payment', 'locked': 'awaiting_payment',
        'unlocked': 'intake_unlocked', 'delivery_ready': 'ready', 'done': 'delivered',
    }
    value = aliases.get(value, value)
    return value if value in ORDER_STATES else default


def can_transition(current: str, target: str) -> bool:
    current = normalize_status(current)
    target = normalize_status(target)
    return target in TRANSITIONS.get(current, []) or current == target


def safe_order_record(record: Dict[str, Any]) -> Dict[str, Any]:
    clean = {}
    for key, value in (record or {}).items():
        if str(key).lower() in NO_CARD_DATA_FIELDS:
            continue
        clean[str(key)] = value
    return clean


def init_db(db_path: Path = DB_PATH) -> Path:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.execute('''
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            customer_email TEXT,
            pack_name TEXT,
            amount_eur REAL,
            currency TEXT DEFAULT 'EUR',
            state TEXT NOT NULL,
            payment_status TEXT,
            delivery_status TEXT,
            provider TEXT,
            created_at TEXT,
            updated_at TEXT,
            metadata_json TEXT
        )
        ''')
        con.execute('''
        CREATE TABLE IF NOT EXISTS order_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id TEXT,
            event_type TEXT,
            from_state TEXT,
            to_state TEXT,
            allowed INTEGER,
            reason TEXT,
            created_at TEXT,
            actor TEXT,
            metadata_json TEXT
        )
        ''')
    return db_path


def upsert_order(record: Dict[str, Any], db_path: Path = DB_PATH) -> Dict[str, Any]:
    init_db(db_path)
    r = safe_order_record(dict(record or {}))
    now = utc_now()
    order_id = str(r.get('order_id') or 'ORDER-DEMO-V106')
    state = normalize_status(r.get('state') or r.get('order_state') or 'created')
    metadata = r.get('metadata') if isinstance(r.get('metadata'), dict) else {}
    with sqlite3.connect(db_path) as con:
        existing = con.execute('SELECT state, created_at FROM orders WHERE order_id=?', (order_id,)).fetchone()
        created_at = existing[1] if existing else now
        con.execute('''
            INSERT INTO orders(order_id, customer_email, pack_name, amount_eur, currency, state, payment_status, delivery_status, provider, created_at, updated_at, metadata_json)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(order_id) DO UPDATE SET
                customer_email=excluded.customer_email,
                pack_name=excluded.pack_name,
                amount_eur=excluded.amount_eur,
                currency=excluded.currency,
                state=excluded.state,
                payment_status=excluded.payment_status,
                delivery_status=excluded.delivery_status,
                provider=excluded.provider,
                updated_at=excluded.updated_at,
                metadata_json=excluded.metadata_json
        ''', (
            order_id, str(r.get('customer_email') or ''), str(r.get('pack_name') or ''),
            float(r.get('amount_eur') or 0), str(r.get('currency') or 'EUR'), state,
            str(r.get('payment_status') or ''), str(r.get('delivery_status') or ''),
            str(r.get('provider') or ''), created_at, now, json.dumps(metadata, ensure_ascii=False),
        ))
    return {**r, 'order_id': order_id, 'state': state, 'created_at': created_at, 'updated_at': now}


def transition_order(order_id: str, target_state: str, actor: str = 'system', reason: str = '', db_path: Path = DB_PATH) -> Dict[str, Any]:
    init_db(db_path)
    target = normalize_status(target_state)
    now = utc_now()
    with sqlite3.connect(db_path) as con:
        row = con.execute('SELECT state FROM orders WHERE order_id=?', (order_id,)).fetchone()
        current = normalize_status(row[0] if row else 'created')
        allowed = can_transition(current, target)
        if allowed and row:
            con.execute('UPDATE orders SET state=?, updated_at=? WHERE order_id=?', (target, now, order_id))
        con.execute('''
            INSERT INTO order_events(order_id, event_type, from_state, to_state, allowed, reason, created_at, actor, metadata_json)
            VALUES(?,?,?,?,?,?,?,?,?)
        ''', (order_id, 'transition_request', current, target, int(allowed), reason, now, actor, '{}'))
    return {'order_id': order_id, 'from_state': current, 'to_state': target, 'allowed': allowed, 'reason': reason, 'created_at': now}


def list_orders(db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    init_db(db_path)
    with sqlite3.connect(db_path) as con:
        rows = con.execute('SELECT order_id, customer_email, pack_name, amount_eur, currency, state, payment_status, delivery_status, provider, created_at, updated_at FROM orders ORDER BY updated_at DESC').fetchall()
    keys = ['order_id','customer_email','pack_name','amount_eur','currency','state','payment_status','delivery_status','provider','created_at','updated_at']
    return [dict(zip(keys, row)) for row in rows]
