"""
agents/reflexion_loop.py — Actor-Critic-Simulator Reflexion Loop

Orchestrates the full decision pipeline for a single request:
  1. Compute affordability baseline (deterministic)
  2. Strategy Agent (Actor) proposes a plan
  3. Risk Critic validates against cashflow simulator
  4. If invalid: Critic feeds back → Strategy Agent revises (max 3 iterations)
  5. Explainer Agent generates final decision_explanation
  6. Return validated OutputRow + trace log
"""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from ..config import cfg
from ..finance_engine import (
    AffordabilityCalculator,
    AffordabilityResult,
    DailyBalanceSimulator,
    RecurrenceDetector,
)
from ..schemas import OutputRow, PaymentOption, PurchaseRequest
from ..user_context import UserContext
from .critic_agent import RiskCritic
from .explainer_agent import ExplainerAgent
from .strategy_agent import AgentDecisionSchema, StrategyAgent

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 3


class ReflexionLoop:
    """
    Per-request Actor-Critic-Simulator loop.
    Call process_request() for each PurchaseRequest.
    """

    def __init__(self):
        self.detector  = RecurrenceDetector()
        self.simulator = DailyBalanceSimulator()
        self.calc      = AffordabilityCalculator(self.simulator, self.detector)
        self.actor     = StrategyAgent()
        self.critic    = RiskCritic(self.simulator)
        self.explainer = ExplainerAgent()

    def process_request(
        self,
        request: PurchaseRequest,
        ctx: UserContext,
        msg_patches: dict,
    ) -> Tuple[OutputRow, str]:
        """
        Returns (OutputRow, trace_log_str).
        trace_log_str is the structured audit trail for log.txt.
        """
        t_start = time.time()
        req_date = date.fromisoformat(str(request.request_date)[:10])
        payment_options = ctx.payment_options.get(request.request_id, [])

        # Filter options by user payment preferences and max_installment_months
        eligible_options = _filter_payment_options(payment_options, ctx.profile)

        # 1. Deterministic affordability baseline
        affordability = self.calc.compute(ctx, req_date, float(request.requested_amount))

        # 2. Build message context string
        msg_context = _build_message_context(request, ctx, msg_patches)

        # 3. Actor-Critic reflexion loop
        trace_iterations = []
        decision: Optional[AgentDecisionSchema] = None
        critic_feedback: Optional[str] = None

        for iteration in range(1, MAX_ITERATIONS + 1):
            # Actor proposes
            decision = self.actor.propose(
                request, ctx, affordability, eligible_options,
                msg_context, critic_feedback
            )
            trace_iterations.append({
                "iter": iteration,
                "proposal": decision.affordability_status + "/" + decision.recommended_payment_method,
                "amount_safe": decision.amount_safe_to_pay,
                "reasoning_excerpt": decision.internal_reasoning[:120],
            })

            # Critic validates
            valid, feedback = self.critic.validate(
                decision,
                float(request.requested_amount),
                req_date,
                str(request.desired_completion_date) if request.desired_completion_date else None,
                ctx,
                affordability,
            )

            if valid:
                trace_iterations[-1]["result"] = "APPROVED"
                break
            else:
                trace_iterations[-1]["result"] = f"REJECTED: {feedback[:80]}"
                critic_feedback = feedback
                if iteration == MAX_ITERATIONS:
                    # Max iterations reached — use deterministic fallback
                    logger.warning("Max iterations reached for %s — using deterministic fallback", request.request_id)
                    decision = self.actor._deterministic_fallback(request, ctx, affordability, eligible_options)
                    trace_iterations.append({"iter": "FALLBACK", "reason": feedback[:120]})

        # 4. Explainer
        explanation = self.explainer.explain(request, ctx, decision)
        decision.decision_explanation = explanation

        # 5. Build OutputRow
        row = _build_output_row(request, ctx, affordability, decision)

        # 6. Build trace log
        elapsed = time.time() - t_start
        trace = _build_trace(request, ctx, affordability, decision, trace_iterations, elapsed)

        return row, trace


def _filter_payment_options(
    options: List[PaymentOption],
    profile,
) -> List[PaymentOption]:
    """Filter options to those compatible with user preferences."""
    user_methods = set(m.strip().lower() for m in profile.payment_methods)
    max_months = profile.max_installment_months

    filtered = []
    for opt in options:
        method = opt.payment_method.lower()
        if method not in user_methods:
            continue
        if method == "installments" and max_months is not None:
            if opt.number_of_payments and opt.number_of_payments > max_months:
                continue
        filtered.append(opt)
    return filtered


def _build_message_context(
    request: PurchaseRequest,
    ctx: UserContext,
    msg_patches: dict,
) -> str:
    """Build a concise summary of relevant messages for this request."""
    lines = []
    for msg in ctx.messages:
        if msg.request_id == request.request_id:
            patch = msg_patches.get(msg.message_id, {})
            ptype = patch.get("patch_type", "informational")
            if ptype != "informational":
                lines.append(f"[{msg.message_id} / {ptype}]: {msg.message_text[:200]}")
            else:
                lines.append(f"[{msg.message_id}]: {msg.message_text[:150]}")
    # Also include salary overrides for this user (not tied to specific request)
    for msg_id, patch in msg_patches.items():
        if patch.get("user_id") == request.user_id and patch.get("request_id") is None:
            ptype = patch.get("patch_type", "")
            if ptype == "salary_override":
                lines.append(f"[{msg_id} / salary_override]: new salary = {patch.get('new_amount')} {patch.get('currency')}")
    return "\n".join(lines[:5]) if lines else ""


def _build_output_row(
    request: PurchaseRequest,
    ctx: UserContext,
    affordability: AffordabilityResult,
    decision: AgentDecisionSchema,
) -> OutputRow:
    """Build the final OutputRow from the validated decision."""
    plan_str = _format_payment_plan(decision.payment_plan)
    changes_str = _format_spending_changes(decision.spending_changes_needed)

    return OutputRow(
        request_id=request.request_id,
        amount_safe_to_pay=round(decision.amount_safe_to_pay, 2),
        affordability_status=decision.affordability_status,
        recommended_payment_method=decision.recommended_payment_method,
        payment_plan=plan_str,
        earliest_date_for_full_payment=decision.earliest_date_for_full_payment or "",
        spending_changes_needed=changes_str,
        decision_explanation=decision.decision_explanation,
    )


def _format_payment_plan(installments) -> str:
    if not installments:
        return "none"
    return "|".join(f"{p.date}:{p.amount:.2f}" for p in installments)


def _format_spending_changes(changes: List[str]) -> str:
    if not changes:
        return "none"
    return "|".join(changes)


def _build_trace(
    request: PurchaseRequest,
    ctx: UserContext,
    affordability: AffordabilityResult,
    decision: AgentDecisionSchema,
    iterations: list,
    elapsed: float,
) -> str:
    profile = ctx.profile
    iters_text = "\n".join([
        f"    Iter {it['iter']}: {it.get('proposal','?')} | safe={it.get('amount_safe',0):.2f} | {it.get('result','?')}"
        for it in iterations
    ])

    return (
        f"\n[REQUEST: {request.request_id} | USER: {request.user_id} | {request.request_date}]\n"
        f"{'='*60}\n"
        f"1. CONTEXT:\n"
        f"   Request: {request.request_type} {profile.home_currency} {request.requested_amount} "
        f"(deadline {request.desired_completion_date})\n"
        f"   Balance: {profile.current_available_balance:.2f} | Min: {profile.minimum_balance_to_keep:.2f}\n"
        f"   Recurring: {len(affordability.recurring_patterns)} patterns detected\n"
        f"2. DETERMINISTIC BASELINE:\n"
        f"   90-day min balance: {affordability.min_balance_in_baseline:.2f}\n"
        f"   Safe today: {affordability.amount_safe_to_pay:.2f}\n"
        f"   Earliest full: {affordability.earliest_date_for_full_payment}\n"
        f"3. AGENT DELIBERATION:\n"
        f"{iters_text}\n"
        f"4. FINAL VERDICT:\n"
        f"   Status: {decision.affordability_status} | Method: {decision.recommended_payment_method}\n"
        f"   Amount safe: {decision.amount_safe_to_pay:.2f}\n"
        f"   Spending changes: {decision.spending_changes_needed}\n"
        f"   Latency: {elapsed:.2f}s\n"
        f"{'='*60}\n"
    )
