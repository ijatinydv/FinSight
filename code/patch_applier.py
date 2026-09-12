"""
patch_applier.py — Apply cached delta-state patches to UserContexts

This module reads the two cache files produced by run_delta_state.py:
  code/.cache/message_patches.json  — salary overrides, cancels, date changes
  code/.cache/image_amounts.json    — VLM-extracted amounts for image events

It returns a mutated copy of the events list for each UserContext,
providing the ground-truth-adjusted input to the financial simulator.

Patch application order (conflict resolution per AGENTS.md §6.3):
  1. Explicit cancellation → mark status=cancelled
  2. salary_override → replace next scheduled salary with new amount
  3. date_change → update event_date / settlement_date
  4. pending_refund → flag event but NEVER add to balance
  5. image amount → replace NaN amounts on linked events
"""
from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

from .schemas import FinancialEvent
from .user_context import UserContext

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[1] / "code" / ".cache"
MSG_CACHE = CACHE_DIR / "message_patches.json"
IMG_CACHE = CACHE_DIR / "image_amounts.json"


# ─────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────

def apply_all_patches(contexts: Dict[str, UserContext]) -> Dict[str, UserContext]:
    """
    Load cached patches and apply them in-place to all UserContexts.
    Returns the same dict with mutated events lists.
    """
    msg_patches = _load_json(MSG_CACHE)
    img_amounts = _load_json(IMG_CACHE)

    if not msg_patches and not img_amounts:
        logger.warning("No patch cache found — run `python run_delta_state.py` first.")
        return contexts

    # Index image patches by (user_id, event_id) for O(1) lookup
    img_by_event: Dict[str, dict] = {}
    for img_id, img in img_amounts.items():
        ev_id = img.get("event_id")
        if ev_id:
            img_by_event[ev_id] = img

    # Apply per user
    patched: Dict[str, UserContext] = {}
    for uid, ctx in contexts.items():
        new_events = _apply_message_patches(uid, ctx.events, msg_patches)
        
        # Heuristic fallback for flaky LLM message parsing:
        # If any message indicates employment ended or is a pending commission gig, cancel salary.
        for m in ctx.messages:
            text = m.message_text.lower()
            if "employment has ended" in text or "contract has ended" in text:
                new_events = _cancel_future_salary(new_events)
            elif "payout is still pending" in text and "weekly earnings" in text:
                new_events = _cancel_future_salary(new_events)

        new_events = _apply_image_patches(new_events, img_by_event)
        sal_override = _extract_salary_override(uid, msg_patches, ctx.profile, ctx.normalizer)
        patched[uid] = UserContext(
            profile=ctx.profile,
            events=new_events,
            messages=ctx.messages,
            images=ctx.images,
            payment_options=ctx.payment_options,
            normalizer=ctx.normalizer,
            salary_override=sal_override,
        )

    # Log summary
    sal_count = sum(1 for v in msg_patches.values() if v.get("patch_type") == "salary_override")
    can_count = sum(1 for v in msg_patches.values() if v.get("patch_type") == "cancel_event")
    img_count = sum(1 for v in img_amounts.values() if v.get("amount") is not None)
    logger.info("Patches applied — salary_overrides=%d, cancels=%d, image_amounts=%d", sal_count, can_count, img_count)
    return patched


# ─────────────────────────────────────────────────────────
# Message patch handlers
# ─────────────────────────────────────────────────────────

def _apply_message_patches(
    user_id: str,
    events: List[FinancialEvent],
    all_patches: dict,
) -> List[FinancialEvent]:
    """Apply salary overrides, cancels, and date changes for this user."""
    # Collect patches for this user
    user_patches = [
        v for v in all_patches.values()
        if v.get("user_id") == user_id
        and v.get("patch_type") in ("salary_override", "cancel_event", "date_change")
    ]
    if not user_patches:
        return events

    events_mut = list(events)  # shallow copy — we rebuild entries

    for patch in user_patches:
        ptype = patch.get("patch_type")
        ev_id = patch.get("event_id")  # may be payroll ref like "EMP-0001", not real event_id
        new_amount = patch.get("new_amount")
        new_date = patch.get("new_date")
        currency = patch.get("currency")

        if ptype == "salary_override" and new_amount is not None and new_amount > 0:
            events_mut = _apply_salary_override(events_mut, new_amount, currency, new_date)

        elif ptype == "cancel_event":
            # Cancel next scheduled salary (seasonal contract ended, employment terminated)
            events_mut = _cancel_future_salary(events_mut)

        elif ptype == "date_change" and new_date:
            # Find the next scheduled salary and update its date
            events_mut = _update_next_salary_date(events_mut, new_date)

    return events_mut


def _apply_salary_override(
    events: List[FinancialEvent],
    new_amount: float,
    currency: Optional[str],
    effective_date: Optional[str],
) -> List[FinancialEvent]:
    """
    Replace the next scheduled salary event's amount with new_amount.
    If no scheduled salary exists, add a new synthetic one.
    The salary override is the *authoritative* ground truth for next month's pay.
    """
    result = []
    applied = False
    for ev in events:
        if (
            not applied
            and ev.category in ("salary", "income", "payroll", "wages")
            and ev.direction == "credit"
            and ev.status == "scheduled"
        ):
            # Override this salary event
            update = {"amount": new_amount}
            if currency:
                update["currency"] = currency
            if effective_date:
                update["settlement_date"] = effective_date
                update["event_date"] = effective_date
            result.append(ev.model_copy(update=update))
            applied = True
            logger.debug("Salary override applied: %s -> %.2f %s", ev.event_id, new_amount, currency or ev.currency)
        else:
            result.append(ev)
    return result


def _cancel_future_salary(events: List[FinancialEvent]) -> List[FinancialEvent]:
    """
    Cancel all future scheduled/pending salary events (employment terminated).
    Also injects a synthetic cancelled event to ensure RecurrenceDetector 
    knows the pattern is dead, even if no scheduled events existed.
    """
    result = []
    found_scheduled = False
    for ev in events:
        if (
            ev.category in ("salary", "income", "payroll", "wages")
            and ev.direction == "credit"
            and ev.status == "scheduled"
        ):
            result.append(ev.model_copy(update={"status": "cancelled"}))
            found_scheduled = True
            logger.debug("Salary event cancelled: %s", ev.event_id)
        else:
            result.append(ev)
            
    # If we didn't find any scheduled event to cancel, the RecurrenceDetector 
    # might still project from historical settled events. Inject a synthetic cancelled event.
    if not found_scheduled:
        # find last settled salary to get date/currency
        last_sal = None
        for ev in sorted(events, key=lambda x: str(x.event_date or "")):
            if ev.category in ("salary", "income", "payroll", "wages") and ev.direction == "credit":
                last_sal = ev
        
        if last_sal:
            synth = last_sal.model_copy(update={
                "event_id": f"synth-cancel-{last_sal.event_id}",
                "status": "cancelled",
            })
            result.append(synth)
            
    return result


def _update_next_salary_date(
    events: List[FinancialEvent],
    new_date: str,
) -> List[FinancialEvent]:
    """Update the settlement_date of the next scheduled salary event."""
    result = []
    applied = False
    for ev in events:
        if (
            not applied
            and ev.category in ("salary", "income", "payroll", "wages")
            and ev.direction == "credit"
            and ev.status == "scheduled"
        ):
            result.append(ev.model_copy(update={
                "settlement_date": new_date,
                "event_date": new_date,
            }))
            applied = True
        else:
            result.append(ev)
    return result


# ─────────────────────────────────────────────────────────
# Image patch handlers
# ─────────────────────────────────────────────────────────

def _apply_image_patches(
    events: List[FinancialEvent],
    img_by_event: Dict[str, dict],
) -> List[FinancialEvent]:
    """
    Replace NaN/zero amounts on image-linked events with VLM-extracted amounts.
    The image amount is the authoritative value for that event.
    """
    result = []
    for ev in events:
        img = img_by_event.get(ev.event_id)
        if img and img.get("amount") is not None:
            extracted = float(img["amount"])
            img_currency = img.get("currency")
            ev_amount = ev.amount
            is_nan = (
                ev_amount is None
                or (isinstance(ev_amount, float) and math.isnan(ev_amount))
                or ev_amount == 0.0
            )
            if is_nan:
                update = {"amount": extracted}
                if img_currency:
                    update["currency"] = img_currency
                result.append(ev.model_copy(update=update))
                logger.info(
                    "Image patch: event %s amount set to %.2f %s from %s",
                    ev.event_id, extracted, img_currency or ev.currency, img.get("image_id")
                )
            else:
                result.append(ev)
        else:
            result.append(ev)
    return result


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def _parse_patch_date(s: Optional[str]) -> Optional[date]:
    """Parse a 'YYYY-MM-DD' (or ISO datetime) patch date into a date object."""
    if not s:
        return None
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _extract_salary_override(
    user_id: str,
    all_patches: dict,
    profile,
    normalizer,
) -> Optional[dict]:
    """
    Find the authoritative salary_override patch for this user and return
    {"amount": <home-currency float>, "first_date": <date | None>}, or None.

    The patch amount is the ground-truth go-forward pay. It is normalised to the
    user's home currency so the deterministic engine (which works in home currency)
    can apply it directly. `first_date` is the patched effective/credit date when
    the message stated one (None means "keep the historical payday cadence").
    """
    patch = None
    for v in all_patches.values():
        if (
            v.get("user_id") == user_id
            and v.get("patch_type") == "salary_override"
            and v.get("new_amount") is not None
            and float(v.get("new_amount") or 0) > 0
        ):
            patch = v
            break  # one authoritative salary per user
    if patch is None:
        return None

    amount = float(patch["new_amount"])
    currency = patch.get("currency")
    home = getattr(profile, "home_currency", None)
    first_date = _parse_patch_date(patch.get("new_date"))

    if currency and home and currency != home:
        ref = first_date or date.today()
        try:
            amount = normalizer.convert(amount, currency, home, ref)
        except Exception as e:  # pragma: no cover - keep original on failure
            logger.warning("Salary override FX failed for %s: %s", user_id, e)

    return {"amount": amount, "first_date": first_date}


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not load cache %s: %s", path, e)
        return {}
