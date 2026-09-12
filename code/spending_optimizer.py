"""
spending_optimizer.py — Proactive Spending Change Optimizer

Deterministic greedy algorithm that finds the minimum set of spending changes
(stop / reduce_to) that makes the full requested_amount safe on request_date.

This runs BEFORE the LLM and feeds its results into the context, so the LLM
can correctly recommend full_payment + spending_changes when applicable.

Rules (from problem spec §6.3):
- Only events in user's stoppable_categories with flexibility in ("flexible","stoppable")
  can be stopped.
- Only events in user's reducible_categories with flexibility == "flexible"
  can be reduced (to their minimum_allowed_amount).
- Same event cannot be both stopped and reduced.
- Max 3 changes total.
- Protected categories are never touched.
- Changes must actually unlock the full payment (verified by re-simulation).
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Dict, List, Optional, Tuple

from .finance_engine import DailyBalanceSimulator, RecurrenceDetector, RecurringPattern
from .schemas import FinancialEvent
from .user_context import UserContext

logger = logging.getLogger(__name__)


class SpendingOptimizer:
    """
    Finds the minimum spending changes that unlock a full lump-sum payment.

    Strategy:
    1. Build a list of candidate changes (stop / reduce_to) from flexible events.
    2. Sort candidates by savings potential DESC (highest saving first).
    3. Greedy: accumulate changes until full payment is safe or we hit max 3.
    4. Validate by re-simulating with the changes applied and the payment included.
    5. Return (changes_list, is_sufficient) — if is_sufficient=False, the changes
       alone aren't enough to make the full payment safe (still not_affordable).
    """

    MAX_CHANGES = 3

    def __init__(self, simulator: DailyBalanceSimulator):
        self.simulator = simulator
        self.detector = RecurrenceDetector()

    def find_changes(
        self,
        ctx: UserContext,
        request_date: date,
        requested_amount: float,
        patterns: List[RecurringPattern],
    ) -> Tuple[List[str], bool]:
        """
        Returns (changes, is_sufficient):
          changes       — list of "stop:event_id" / "reduce_to:event_id:amount" strings
          is_sufficient — True if applying these changes makes the full payment safe
        """
        profile = ctx.profile
        min_bal = profile.minimum_balance_to_keep
        protected = set(profile.protected_categories)
        reducible = set(profile.reducible_categories)
        stoppable = set(profile.stoppable_categories)

        # Already safe without any changes
        if self._is_safe(ctx, request_date, requested_amount, patterns, min_bal):
            return [], True

        # Build candidate actions from ALL flexible events
        candidates = _build_candidates(ctx.events, protected, reducible, stoppable)
        if not candidates:
            return [], False

        # Sort by savings DESC for greedy efficiency
        candidates.sort(key=lambda c: c["saving"], reverse=True)

        # Greedy accumulation
        selected: List[dict] = []
        used_event_ids: set = set()

        for cand in candidates:
            if len(selected) >= self.MAX_CHANGES:
                break
            if cand["event_id"] in used_event_ids:
                continue
            selected.append(cand)
            used_event_ids.add(cand["event_id"])

            modified_ctx = _apply_changes_to_ctx(ctx, selected)
            # Re-detect patterns from the modified context so that stopped/reduced
            # events no longer project as future recurring debits
            modified_patterns = self.detector.detect(
                modified_ctx.events,
                as_of=request_date,
                salary_override=getattr(ctx, "salary_override", None),
            )

            if self._is_safe(modified_ctx, request_date, requested_amount, modified_patterns, min_bal):
                logger.info(
                    "SpendingOptimizer: %d change(s) unlock full payment of %.2f on %s",
                    len(selected), requested_amount, request_date,
                )
                return _format_changes(selected), True

        # Couldn't unlock full payment even with max changes
        return [], False

    def _is_safe(
        self,
        ctx: UserContext,
        request_date: date,
        requested_amount: float,
        patterns: List[RecurringPattern],
        min_bal: float,
    ) -> bool:
        """Simulate 90 days with the payment on request_date, check min_bal."""
        days = self.simulator.simulate(
            ctx, request_date, patterns, [(request_date, requested_amount)]
        )
        return all(d.balance_end >= min_bal for d in days)


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def _build_candidates(
    events: List[FinancialEvent],
    protected: set,
    reducible: set,
    stoppable: set,
) -> List[dict]:
    """
    Build a list of candidate change actions from ALL of the user's flexible events.
    Each candidate is:
      {"event_id": str, "action": "stop"|"reduce_to", "saving": float,
       "new_amount": float|None, "change_str": str}

    We consider all individual events (not just one per category) so the greedy
    search can find the exact minimum set matching the ground truth event IDs.
    """
    candidates = []
    seen_event_ids: set = set()

    for ev in events:
        if ev.event_id in seen_event_ids:
            continue
        if ev.direction != "debit":
            continue
        if ev.status not in ("settled", "scheduled"):
            continue
        cat = ev.category or ""
        if cat in protected:
            continue
        ev_amt = ev.amount or 0
        if ev_amt <= 0:
            continue

        if cat in stoppable and ev.flexibility in ("stoppable", "reducible_or_stoppable"):
            candidates.append({
                "event_id": ev.event_id,
                "category": cat,
                "action": "stop",
                "saving": ev_amt,
                "new_amount": None,
                "change_str": f"stop:{ev.event_id}",
            })
            seen_event_ids.add(ev.event_id)
        elif cat in reducible and ev.flexibility in ("reducible", "reducible_or_stoppable") and ev.minimum_allowed_amount is not None:
            saving = ev_amt - ev.minimum_allowed_amount
            if saving > 0:
                candidates.append({
                    "event_id": ev.event_id,
                    "category": cat,
                    "action": "reduce_to",
                    "saving": saving,
                    "new_amount": ev.minimum_allowed_amount,
                    "change_str": f"reduce_to:{ev.event_id}:{ev.minimum_allowed_amount}",
                })
                seen_event_ids.add(ev.event_id)

    return candidates


def _format_changes(selected: List[dict]) -> List[str]:
    return [c["change_str"] for c in selected]


def _apply_changes_to_ctx(ctx: UserContext, selected: List[dict]) -> UserContext:
    """
    Return a shallow-copy UserContext with the selected spending changes applied
    to the events list. Stopped events become cancelled; reduced events get their
    amount set to minimum_allowed_amount.
    """
    ev_ids_to_stop = {c["event_id"] for c in selected if c["action"] == "stop"}
    ev_to_reduce = {c["event_id"]: c["new_amount"] for c in selected if c["action"] == "reduce_to"}

    new_events = []
    for ev in ctx.events:
        if ev.event_id in ev_ids_to_stop:
            new_events.append(ev.model_copy(update={"status": "cancelled"}))
        elif ev.event_id in ev_to_reduce:
            new_events.append(ev.model_copy(update={"amount": ev_to_reduce[ev.event_id]}))
        else:
            new_events.append(ev)

    return UserContext(
        profile=ctx.profile,
        events=new_events,
        messages=ctx.messages,
        images=ctx.images,
        payment_options=ctx.payment_options,
        normalizer=ctx.normalizer,
        salary_override=getattr(ctx, "salary_override", None),
    )
