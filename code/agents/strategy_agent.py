"""
agents/strategy_agent.py — Strategy Agent (Actor)

Given a UserContext, affordability baseline, payment options, and message context,
the Strategy Agent proposes a complete financial decision using GLM-5.3-Flash.

Output: Pydantic AgentDecisionSchema (structured JSON)
"""
from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from ..config import cfg
from ..finance_engine import AffordabilityResult, RecurringPattern
from ..schemas import Message, OutputRow, PaymentOption, PurchaseRequest
from ..user_context import UserContext

logger = logging.getLogger(__name__)

client = OpenAI(api_key=cfg.novita_api_key, base_url=cfg.novita_base_url)
MODEL = cfg.model_name


# ─────────────────────────────────────────────────────────
# Decision schema (structured LLM output)
# ─────────────────────────────────────────────────────────

class PaymentInstallmentSchema(BaseModel):
    date: str          # YYYY-MM-DD
    amount: float


class AgentDecisionSchema(BaseModel):
    amount_safe_to_pay: float
    affordability_status: str   # affordable_now | affordable_with_plan | affordable_later | not_affordable
    recommended_payment_method: str  # full_payment | partial_payment | installments | wait | not_recommended
    payment_plan: List[PaymentInstallmentSchema]  # empty = none
    earliest_date_for_full_payment: Optional[str] = None
    spending_changes_needed: List[str] = []   # ["stop:event_id", "reduce_to:event_id:amount"]
    decision_explanation: str = ""
    internal_reasoning: str = ""              # Not written to output.csv — logged for AI Judge interview


# ─────────────────────────────────────────────────────────
# Strategy Agent
# ─────────────────────────────────────────────────────────

class StrategyAgent:
    """
    Actor: proposes a financial decision using GLM-5.3-Flash.
    Uses structured JSON output (response_format=json_object) and validates
    with Pydantic before returning.
    """

    SYSTEM_PROMPT = """\
You are a financial decision agent. Given a user's financial profile, a 90-day balance projection, and payment options for a purchase request, you must output a single JSON object representing the optimal decision.

Rules (MANDATORY):
1. balance must NEVER fall below minimum_balance_to_keep on any day, including after any payment
2. amount_safe_to_pay is the max safe to pay TODAY (on request_date), between 0 and requested_amount inclusive
3. affordability_status: "affordable_now" if full amount safe today; "affordable_with_plan" if completable via installments/partial/spending-changes by deadline; "affordable_later" if full amount safe after request_date but no plan needed; "not_affordable" if no safe path exists
4. recommended_payment_method: "full_payment", "partial_payment", "installments", "wait", or "not_recommended"
5. payment_plan: [{date, amount}] chronologically. For partial_payment: exactly 2 payments summing to requested_amount. For installments: follow the chosen option exactly.
6. spending_changes_needed: only non-protected, flexible categories the user permits; max 3 items; format "stop:<event_id>" or "reduce_to:<event_id>:<new_amount>"
7. Prefer: deadline completion > no spending changes > min total cost > earlier start > fewer payments > lower payment_option_id
8. If installments are the best option, choose the one with minimum total_payable_amount whose number_of_payments <= max_installment_months
9. NEVER count pending credits, refunds, bonuses, or investment gains as usable income
10. If the user's payment_methods_user_will_consider does not include a method, do NOT recommend it

Output ONLY a valid JSON object matching this schema exactly. No markdown, no explanation outside the JSON.
"""

    def propose(
        self,
        request: PurchaseRequest,
        ctx: UserContext,
        affordability: AffordabilityResult,
        payment_options: List[PaymentOption],
        message_context: str,
        critic_feedback: Optional[str] = None,
    ) -> AgentDecisionSchema:
        """Generate a decision proposal. If critic_feedback is provided, revise accordingly."""
        prompt = self._build_prompt(request, ctx, affordability, payment_options, message_context, critic_feedback)

        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
                max_tokens=1024,
            )
            raw = resp.choices[0].message.content or "{}"
            data = json.loads(raw)
            return AgentDecisionSchema(**data)
        except (json.JSONDecodeError, ValidationError, Exception) as e:
            logger.warning("StrategyAgent parse error: %s", e)
            # Deterministic fallback
            return self._deterministic_fallback(request, ctx, affordability, payment_options)

    def _build_prompt(
        self,
        request: PurchaseRequest,
        ctx: UserContext,
        affordability: AffordabilityResult,
        payment_options: List[PaymentOption],
        message_context: str,
        critic_feedback: Optional[str],
    ) -> str:
        profile = ctx.profile
        bal_summary = _balance_summary(affordability)

        # Format payment options
        opts_text = "\n".join([
            f"  - Option {o.payment_option_id}: {o.payment_method}, "
            f"amount={o.payment_amount} {profile.home_currency}, "
            f"n_payments={o.number_of_payments}, "
            f"first={o.first_payment_date}, "
            f"freq={o.payment_frequency_days}d, "
            f"fee={o.financing_fee or 0:.2f}, "
            f"total={o.total_payable_amount}"
            for o in payment_options
        ])

        # Recurring patterns
        patterns_text = "\n".join([
            f"  - {p.category} ({p.direction}): ~{p.avg_amount:.0f} {profile.home_currency} every {p.avg_period_days:.0f}d"
            for p in affordability.recurring_patterns
        ])

        revision = f"\n\nCRITIC FEEDBACK (revise your answer accordingly):\n{critic_feedback}" if critic_feedback else ""

        return f"""
REQUEST:
  request_id: {request.request_id}
  request_date: {request.request_date}
  request_type: {request.request_type}
  requested_amount: {request.requested_amount} {profile.home_currency}
  desired_completion_date: {request.desired_completion_date}
  allows_partial_payment: {request.allows_partial_payment}
  request_text: {request.request_text or 'N/A'}

USER PROFILE:
  user_id: {profile.user_id}
  home_currency: {profile.home_currency}
  current_balance: {profile.current_available_balance:.2f}
  minimum_balance_to_keep: {profile.minimum_balance_to_keep:.2f}
  max_installment_months: {profile.max_installment_months or 'none (installments not considered)'}
  financial_priorities: {', '.join(profile.priorities)}
  protected_categories: {', '.join(profile.protected_categories)}
  reducible_categories: {', '.join(profile.reducible_categories)}
  stoppable_categories: {', '.join(profile.stoppable_categories)}
  payment_methods_user_will_consider: {', '.join(profile.payment_methods)}

90-DAY BALANCE SIMULATION (no payment):
  day_0 balance: {_safe_balance(affordability, request.request_date, profile.current_available_balance):.2f} {profile.home_currency}
  minimum projected balance: {affordability.min_balance_in_baseline:.2f} {profile.home_currency}
  amount_safe_to_pay TODAY: {affordability.amount_safe_to_pay:.2f} {profile.home_currency}
  earliest date for FULL payment: {affordability.earliest_date_for_full_payment or 'not within 90 days'}
  balance milestones: {bal_summary}

RECURRING PATTERNS DETECTED:
{patterns_text or '  (none detected)'}

PAYMENT OPTIONS AVAILABLE:
{opts_text or '  (none available)'}

RELEVANT MESSAGES/CONTEXT:
{message_context or '  (none)'}
{revision}

Now output a JSON decision object.
"""

    def _deterministic_fallback(
        self,
        request: PurchaseRequest,
        ctx: UserContext,
        affordability: AffordabilityResult,
        payment_options: List[PaymentOption],
    ) -> AgentDecisionSchema:
        """Conservative fallback when LLM fails — uses pure deterministic values."""
        safe = round(affordability.amount_safe_to_pay, 2)
        full = affordability.earliest_date_for_full_payment
        req_amt = request.requested_amount

        if safe >= req_amt:
            status = "affordable_now"
            method = "full_payment"
            plan = [PaymentInstallmentSchema(date=str(request.request_date), amount=req_amt)]
            earliest = str(request.request_date)
        elif full and str(full) <= str(request.desired_completion_date or "9999-12-31"):
            status = "affordable_later"
            method = "wait"
            plan = [PaymentInstallmentSchema(date=str(full), amount=req_amt)]
            earliest = str(full)
        elif safe > 0:
            status = "not_affordable"
            method = "not_recommended"
            plan = []
            earliest = str(full) if full else ""
        else:
            status = "not_affordable"
            method = "not_recommended"
            plan = []
            earliest = ""

        return AgentDecisionSchema(
            amount_safe_to_pay=safe,
            affordability_status=status,
            recommended_payment_method=method,
            payment_plan=plan,
            earliest_date_for_full_payment=earliest or None,
            spending_changes_needed=[],
            decision_explanation=f"Deterministic fallback: safe={safe}, earliest_full={earliest}",
            internal_reasoning="LLM failed; using deterministic engine output.",
        )


def _balance_summary(affordability: AffordabilityResult) -> str:
    """Show balance on key dates (day 0, 15, 30, 60, 90)."""
    days = affordability.baseline_days
    key_offsets = [0, 15, 30, 60, 90]
    parts = []
    for offset in key_offsets:
        if offset < len(days):
            d = days[offset]
            parts.append(f"day{offset}={d.balance_end:.0f}")
    return ", ".join(parts)


def _safe_balance(affordability, request_date, fallback: float) -> float:
    """Safely get day_0 balance regardless of whether request_date is str or date."""
    try:
        from datetime import date as _date
        d = request_date if isinstance(request_date, _date) else _date.fromisoformat(str(request_date)[:10])
        val = affordability.balance_on_day(d)
        return val if val is not None else fallback
    except Exception:
        return fallback
