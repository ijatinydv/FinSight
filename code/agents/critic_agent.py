"""
agents/critic_agent.py — Risk Critic Agent

Takes the Strategy Agent's proposed decision and validates it against
the deterministic cashflow simulator. If invalid, generates domain-specific
feedback for the Strategy Agent to revise.

Max 3 iterations — then fallback to deterministic conservative output.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import List, Optional, Tuple

from ..finance_engine import AffordabilityResult, DailyBalanceSimulator
from ..user_context import UserContext
from .strategy_agent import AgentDecisionSchema, PaymentInstallmentSchema

logger = logging.getLogger(__name__)


class RiskCritic:
    """
    Validates a proposed AgentDecisionSchema against hard financial constraints.

    Validates:
    1. Balance never falls below min_balance after any planned payment
    2. payment_plan amounts sum to requested_amount (for partial/installments)
    3. last payment date <= desired_completion_date
    4. amount_safe_to_pay between 0 and requested_amount
    5. spending_changes only target flexible, non-protected categories (max 3)
    """

    def __init__(self, simulator: DailyBalanceSimulator):
        self.simulator = simulator

    def validate(
        self,
        decision: AgentDecisionSchema,
        request_amount: float,
        request_date: date,
        desired_completion_date: Optional[str],
        ctx: UserContext,
        affordability: AffordabilityResult,
    ) -> Tuple[bool, Optional[str]]:
        """
        Returns (is_valid, feedback_message).
        feedback_message is None when valid, or a domain-specific critique.
        """
        min_bal = ctx.profile.minimum_balance_to_keep

        # ── 1. amount_safe_to_pay bounds ──────────────────────────────
        if decision.amount_safe_to_pay < 0:
            return False, f"amount_safe_to_pay cannot be negative. Got {decision.amount_safe_to_pay:.2f}."
        if decision.amount_safe_to_pay > request_amount + 0.01:
            return False, (
                f"amount_safe_to_pay ({decision.amount_safe_to_pay:.2f}) exceeds "
                f"requested_amount ({request_amount:.2f}). Cap at {request_amount:.2f}."
            )

        # ── 2. payment_plan sum ───────────────────────────────────────
        if decision.payment_plan:
            plan_total = sum(p.amount for p in decision.payment_plan)
            if decision.recommended_payment_method in ("partial_payment", "installments"):
                if abs(plan_total - request_amount) > 0.02:
                    return False, (
                        f"payment_plan sums to {plan_total:.2f} but requested_amount is {request_amount:.2f}. "
                        f"Adjust amounts so they sum exactly to {request_amount:.2f}."
                    )

        # ── 3. Deadline constraint ────────────────────────────────────
        if desired_completion_date and decision.payment_plan:
            last_date = max(p.date for p in decision.payment_plan)
            if last_date > desired_completion_date:
                return False, (
                    f"Last payment date {last_date} exceeds desired_completion_date {desired_completion_date}. "
                    f"Restructure the plan so all payments complete by {desired_completion_date}."
                )

        # ── 4. Balance simulation ─────────────────────────────────────
        if decision.payment_plan:
            installments = [
                (_parse_date(p.date), p.amount)
                for p in decision.payment_plan
            ]
            valid, failure = affordability.validate_plan(
                installments, min_bal, self.simulator, ctx, request_date
            )
            if not valid:
                # Find the deterministically safe amount for guidance
                safe_guidance = affordability.amount_safe_to_pay
                return False, (
                    f"Balance constraint violated: {failure}. "
                    f"The deterministic engine shows amount_safe_to_pay={safe_guidance:.2f}. "
                    f"Reduce the initial payment to at most {safe_guidance:.2f} or restructure installments."
                )
        else:
            # No payment plan — check the immediate payment on request_date
            method = decision.recommended_payment_method
            if method == "full_payment":
                installments = [(_parse_date(str(request_date)), request_amount)]
                valid, failure = affordability.validate_plan(
                    installments, min_bal, self.simulator, ctx, request_date
                )
                if not valid:
                    safe = affordability.amount_safe_to_pay
                    return False, (
                        f"full_payment of {request_amount:.2f} violates minimum balance: {failure}. "
                        f"Safe amount today is {safe:.2f}. Consider partial_payment or wait."
                    )

        # ── 5. spending_changes validation ────────────────────────────
        if decision.spending_changes_needed:
            if len(decision.spending_changes_needed) > 3:
                return False, "spending_changes_needed must have at most 3 entries."

            protected = set(ctx.profile.protected_categories)
            reducible = set(ctx.profile.reducible_categories)
            stoppable = set(ctx.profile.stoppable_categories)
            event_map = {ev.event_id: ev for ev in ctx.events}

            stop_events = set()
            reduce_events = set()

            for change in decision.spending_changes_needed:
                parts = change.split(":")
                if parts[0] == "stop":
                    ev_id = parts[1] if len(parts) > 1 else ""
                    ev = event_map.get(ev_id)
                    if ev:
                        if ev.category in protected:
                            return False, f"Cannot stop protected category '{ev.category}' (event {ev_id})."
                        if ev.flexibility not in ("flexible", "stoppable"):
                            return False, f"Event {ev_id} has flexibility='{ev.flexibility}', cannot stop it."
                    stop_events.add(ev_id)
                elif parts[0] == "reduce_to":
                    ev_id = parts[1] if len(parts) > 1 else ""
                    ev = event_map.get(ev_id)
                    if ev:
                        if ev.category in protected:
                            return False, f"Cannot reduce protected category '{ev.category}' (event {ev_id})."
                        if ev.flexibility not in ("flexible",):
                            return False, f"Event {ev_id} has flexibility='{ev.flexibility}', cannot reduce it."
                    reduce_events.add(ev_id)

            # stop and reduce cannot target the same event
            overlap = stop_events & reduce_events
            if overlap:
                return False, f"spending_changes_needed: cannot both stop and reduce the same event(s): {overlap}"

        return True, None


def _parse_date(date_str: str) -> date:
    try:
        return date.fromisoformat(date_str[:10])
    except Exception:
        return date.today()
