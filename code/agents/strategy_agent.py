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
You are a financial decision agent. Output ONLY a JSON object — no markdown, no text outside JSON.

REQUIRED OUTPUT FORMAT (all fields mandatory):
{
  "amount_safe_to_pay": <number 0 to requested_amount>,
  "affordability_status": "<affordable_now|affordable_with_plan|affordable_later|not_affordable>",
  "recommended_payment_method": "<full_payment|partial_payment|installments|wait|not_recommended>",
  "payment_plan": [{"date": "YYYY-MM-DD", "amount": <number>}],
  "earliest_date_for_full_payment": "<YYYY-MM-DD or null>",
  "spending_changes_needed": [],
  "decision_explanation": "<max 30 words, include key numbers>",
  "internal_reasoning": "<max 20 words>"
}

STATUS RULES:
- affordable_now: amount_safe_to_pay >= requested_amount → full_payment
- affordable_with_plan: full amount achievable via installments/partial_payment by deadline → installments or partial_payment
- affordable_later: full amount safe on future date but deadline prevents plan → wait
- not_affordable: no safe path in 90-day horizon → not_recommended

KEY RULES:
- amount_safe_to_pay = MAX payable TODAY keeping balance >= minimum_balance every future day
- NEVER count pending credits, refunds, or bonuses as income
- Only recommend installments if "installments" is in payment_methods_user_will_consider
- partial_payment requires allows_partial_payment=True; plan = exactly 2 entries summing to requested_amount
- payment_plan for wait/full_payment = 1 entry on payment date
"""

    MINI_SYSTEM = """\
Output ONLY valid JSON. Use EXACTLY this structure (no extra keys, no markdown):
{
  "amount_safe_to_pay": 1234.56,
  "affordability_status": "affordable_now",
  "recommended_payment_method": "full_payment",
  "payment_plan": [{"date": "2024-01-15", "amount": 1234.56}],
  "earliest_date_for_full_payment": "2024-01-15",
  "spending_changes_needed": [],
  "decision_explanation": "one sentence",
  "internal_reasoning": "brief"
}
payment_plan MUST be a JSON array of objects with "date" and "amount" keys.
spending_changes_needed MUST be a JSON array (empty [] or list of strings).
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
        import time
        prompt = self._build_prompt(request, ctx, affordability, payment_options, message_context, critic_feedback)
        time.sleep(1.0)  # Rate limit throttle

        def _parse(raw: str) -> AgentDecisionSchema:
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            # Attempt to repair truncated JSON
            if raw and not raw.endswith("}"):
                # Find last complete comma-separated entry and close
                for pos in range(len(raw) - 1, 0, -1):
                    if raw[pos] == ',':
                        try:
                            json.loads(raw[:pos] + "}")
                            raw = raw[:pos] + "}"
                            break
                        except Exception:
                            continue
            data = json.loads(raw)
            if not isinstance(data, dict) or not data:
                raise ValueError("Empty or non-dict JSON")
            return AgentDecisionSchema(**data)

        def _call(system: str, user_prompt: str) -> AgentDecisionSchema:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0,
                max_tokens=4096,
            )
            raw = resp.choices[0].message.content or "{}"
            logger.info("Raw LLM output for %s: %s", request.request_id, repr(raw[:400]))
            return _parse(raw)

        try:
            return _call(self.SYSTEM_PROMPT, prompt)
        except Exception as e:
            logger.warning("StrategyAgent first attempt failed (%s) — retrying with mini-prompt", e)
            time.sleep(2.0)
            try:
                mini_prompt = self._build_mini_prompt(request, ctx, affordability, payment_options)
                return _call(self.MINI_SYSTEM, mini_prompt)
            except Exception as e2:
                logger.warning("StrategyAgent mini-prompt failed (%s) — deterministic fallback", e2)
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

        # Format payment options (compact)
        opts_text = "\n".join([
            f"  opt{o.payment_option_id}: {o.payment_method} n={o.number_of_payments} amt={o.payment_amount} first={o.first_payment_date} fee={o.financing_fee or 0:.0f} total={o.total_payable_amount}"
            for o in payment_options
        ])

        # Recurring patterns — top 5 by amount (most impactful)
        top_patterns = sorted(affordability.recurring_patterns, key=lambda p: p.avg_amount, reverse=True)[:5]
        patterns_text = "\n".join([
            f"  {p.category}({p.direction}): ~{p.avg_amount:.0f} every {p.avg_period_days:.0f}d"
            for p in top_patterns
        ])

        revision = f"\nCRITIC FEEDBACK:\n{critic_feedback}" if critic_feedback else ""

        return f"""REQUEST: {request.request_id} | {request.request_date} | {request.request_type}
amount={request.requested_amount} {profile.home_currency} | deadline={request.desired_completion_date} | partial_ok={request.allows_partial_payment}

PROFILE: balance={profile.current_available_balance:.2f} min_keep={profile.minimum_balance_to_keep:.2f} {profile.home_currency}
methods={', '.join(profile.payment_methods)} | max_installment_months={profile.max_installment_months or 'N/A'}
protected={', '.join(profile.protected_categories)} | reducible={', '.join(profile.reducible_categories)} | stoppable={', '.join(profile.stoppable_categories)}

SIMULATION (90d, no payment):
  day0={_safe_balance(affordability, request.request_date, profile.current_available_balance):.2f} min_projected={affordability.min_balance_in_baseline:.2f} safe_today={affordability.amount_safe_to_pay:.2f} {profile.home_currency}
  earliest_full_payment={affordability.earliest_date_for_full_payment or 'beyond 90d'}

RECURRING: {patterns_text or 'none'}

OPTIONS: {opts_text or 'none'}

MESSAGES: {(message_context or 'none')[:300]}
{revision}
Output JSON now."""

    def _build_mini_prompt(
        self,
        request: PurchaseRequest,
        ctx: UserContext,
        affordability: AffordabilityResult,
        payment_options: List[PaymentOption],
    ) -> str:
        """Ultra-compact prompt for when the main prompt fails."""
        profile = ctx.profile
        opts = " | ".join([
            f"opt{o.payment_option_id}:{o.payment_method}:n={o.number_of_payments}:total={o.total_payable_amount}"
            for o in payment_options
        ])
        return (
            f"requested={request.requested_amount} {profile.home_currency} on {request.request_date} deadline={request.desired_completion_date} partial_ok={request.allows_partial_payment}\n"
            f"balance={profile.current_available_balance:.2f} min={profile.minimum_balance_to_keep:.2f} safe_today={affordability.amount_safe_to_pay:.2f} earliest_full={affordability.earliest_date_for_full_payment or 'N/A'}\n"
            f"methods={','.join(profile.payment_methods)} max_installments={profile.max_installment_months or 'N/A'}\n"
            f"options: {opts or 'none'}\n"
            "Output JSON with all required fields."
        )

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
