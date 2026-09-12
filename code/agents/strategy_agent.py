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

SPENDING CHANGES FORMAT:
- stop:<event_id>
- reduce_to:<event_id>:<new_amount>
Only use event_ids from the FLEXIBLE SPENDING block. Max 3 changes.

STATUS-METHOD COUPLING RULES (follow exactly):
- method=full_payment AND spending_changes=[] → status=affordable_now
- method=full_payment AND spending_changes non-empty → status=affordable_with_plan
- method=installments → status=affordable_with_plan (ALWAYS, even if safe >= requested)
- method=partial_payment → status=affordable_with_plan (ALWAYS)
- method=wait → status=affordable_later
- method=not_recommended → status=not_affordable

METHOD SELECTION RULES (in priority order):
1. If full_payment NOT in user methods AND no installments option → not_recommended / not_affordable
2. If safe_today >= requested_amount AND full_payment in user methods → full_payment / affordable_now
3. If safe_today >= requested AND only installments in methods → installments / affordable_with_plan
4. If BOTH installments (fitting deadline) AND partial_payment eligible exist:
   - Compare COST: partial_payment always costs requested_amount (no fee)
   - If installment has financing_fee > 0 → partial_payment wins (cheaper)
   - If installment fee = 0 → installments wins (more structured, no lump-sum risk)
5. If eligible installment option exists AND all payments fit within deadline → installments / affordable_with_plan
6. If partial_payment eligible (allows_partial=True, safe > 1% of requested, remainder fits deadline) → partial_payment / affordable_with_plan
7. If earliest_full_payment <= deadline AND full_payment in methods → wait / affordable_later
8. If spending changes (stop/reduce flexible events) would unlock full payment → full_payment / affordable_with_plan (with changes)
9. Otherwise → not_recommended / not_affordable

CRITICAL RULES:
- amount_safe_to_pay = engine value (do NOT change unless you have strong evidence from messages)
- If safe_today is within 0.5% of requested (e.g. 166.60 vs 166.61) → treat as safe_today >= requested → affordable_now
- NEVER use full_payment if "full_payment" is NOT in the methods list
- NEVER use installments if "installments" is NOT in methods list OR if n > max_installment_months
- partial_payment: only when "partial_payment" in methods list AND allows_partial_payment=True AND 0 < safe < requested AND remainder payable by deadline
- DO NOT recommend partial_payment when the remainder is < 1% of requested_amount
- payment_plan for installments: use EXACT dates from the option (first_payment_date + frequency_days * i)
- payment_plan for wait: single entry [{"date": earliest_full, "amount": requested}]
- payment_plan for full_payment: single entry [{"date": request_date, "amount": requested}]
- payment_plan for partial_payment: exactly 2 entries summing to requested_amount
- earliest_date_for_full_payment: ALWAYS populate when any safe full-payment date exists, even if after deadline
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
STATUS RULE: installments/partial_payment → affordable_with_plan; wait → affordable_later; not_recommended → not_affordable; full_payment+no_changes → affordable_now.
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
            decision = AgentDecisionSchema(**data)
            # Enforce status-method coupling post-parse
            decision = _enforce_status_method(decision)
            return decision

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

        # Determine eligibility hint
        user_methods = set(m.strip().lower() for m in profile.payment_methods)
        full_pay_eligible = "full_payment" in user_methods
        inst_eligible = [o for o in payment_options if o.payment_method.lower() == "installments"]
        eligible_hint = f"full_payment_allowed={full_pay_eligible} | eligible_installment_options={len(inst_eligible)}"

        # Flexible spending options — show ALL individual events so LLM can use specific IDs
        flex_events = []
        reducible_set = set(profile.reducible_categories)
        stoppable_set = set(profile.stoppable_categories)
        protected_set = set(profile.protected_categories)
        for ev in ctx.events:
            if ev.direction != "debit":
                continue
            if ev.status not in ("settled", "scheduled"):
                continue
            cat = ev.category or ""
            if cat in protected_set or cat not in (reducible_set | stoppable_set):
                continue
            if (ev.amount or 0) <= 0:
                continue
            flex = f"  {cat}/{ev.event_id}: amount={ev.amount:.2f}"
            if cat in reducible_set and ev.minimum_allowed_amount is not None:
                flex += f" min_allowed={ev.minimum_allowed_amount}"
                flex += " [reduce_to eligible]"
            elif cat in stoppable_set and ev.flexibility in ("flexible", "stoppable"):
                flex += " [stop eligible]"
            flex_events.append(flex)
        flex_text = "\n".join(flex_events[:15])  # cap at 15 to avoid prompt overflow

        # Spending changes hint from deterministic optimizer
        spending_hint_section = ""
        if message_context and message_context.startswith("[SPENDING_CHANGES_HINT]"):
            hint_line = message_context.split("\n")[0]
            spending_hint_section = f"\n⚠️  OPTIMIZER HINT — These changes unlock full payment:\n  {hint_line}\n"

        return f"""REQUEST: {request.request_id} | {request.request_date} | {request.request_type}
amount={request.requested_amount} {profile.home_currency} | deadline={request.desired_completion_date} | partial_ok={request.allows_partial_payment}

PROFILE: balance={profile.current_available_balance:.2f} min_keep={profile.minimum_balance_to_keep:.2f} {profile.home_currency}
methods={', '.join(profile.payment_methods)} | max_installment_months={profile.max_installment_months or 'N/A'}
protected={', '.join(profile.protected_categories)} | reducible={', '.join(profile.reducible_categories)} | stoppable={', '.join(profile.stoppable_categories)}
{eligible_hint}

SIMULATION (90d, no payment):
  day0={_safe_balance(affordability, request.request_date, profile.current_available_balance):.2f} min_projected={affordability.min_balance_in_baseline:.2f} safe_today={affordability.amount_safe_to_pay:.2f} {profile.home_currency}
  earliest_full_payment={affordability.earliest_date_for_full_payment or 'beyond 90d'}

RECURRING: {patterns_text or 'none'}
FLEXIBLE SPENDING (use exact event_ids below in spending_changes_needed, max 3 changes):
{flex_text or '  none'}
{spending_hint_section}
OPTIONS (already filtered to user preferences): {opts_text or 'none (no eligible options exist)'}

MESSAGES: {(message_context or 'none')[:400]}
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
        deadline_str = str(request.desired_completion_date or "9999-12-31")
        req_date_str = str(request.request_date)

        # Determine user-accepted methods
        user_methods = set(m.strip().lower() for m in ctx.profile.payment_methods)
        can_full = "full_payment" in user_methods
        can_installments = "installments" in user_methods
        can_partial = "partial_payment" in user_methods and request.allows_partial_payment

        # Tolerance: if safe is within 0.1% of requested, treat as safe >= requested
        tolerance_safe = safe >= req_amt * 0.999

        # Partial payment eligibility: safe > 1% and < 99.9% of requested, remainder payable by deadline
        partial_second_date = str(full) if full and str(full) <= deadline_str else deadline_str
        partial_eligible = (
            can_partial
            and safe > req_amt * 0.01
            and safe < req_amt * 0.999
            and partial_second_date <= deadline_str
        )

        # Eligible installment options (fit within deadline and max months)
        max_months = ctx.profile.max_installment_months
        eligible_inst = [
            o for o in payment_options
            if o.payment_method.lower() == "installments"
            and (max_months is None or (o.number_of_payments or 0) <= max_months)
            and str(o.first_payment_date or "") >= req_date_str
        ]
        eligible_inst_in_deadline = [
            o for o in eligible_inst
            if _installment_fits_deadline(o, deadline_str)
        ]
        # Sort installment options by total cost ascending, then by option id (tiebreaker)
        eligible_inst_in_deadline.sort(
            key=lambda o: (float(o.total_payable_amount or 0), str(o.payment_option_id))
        )

        # ── Decision tree (implements spec §6.3 ranking) ─────────────────────
        # Priority order:
        # 1. full_payment (affordable_now, no changes needed)
        # 2. full_payment + installments when user can't do full_payment alone
        # 3. partial_payment vs installments — prefer whichever costs less;
        #    partial_payment has no financing fee, so it beats any installment with fee
        # 4. wait (full payment achievable before deadline)
        # 5. partial_payment (last resort before not_recommended)
        # 6. not_recommended

        status = "not_affordable"
        method = "not_recommended"
        plan: list = []
        earliest = str(full) if full else ""

        if tolerance_safe and can_full:
            # ── 1. Full payment now ──────────────────────────────────────────
            status = "affordable_now"
            method = "full_payment"
            plan = [PaymentInstallmentSchema(date=req_date_str, amount=req_amt)]
            earliest = req_date_str

        elif tolerance_safe and can_installments and eligible_inst_in_deadline:
            # ── 2. Full amount safe but user only considers installments ──────
            opt = eligible_inst_in_deadline[0]
            status = "affordable_with_plan"
            method = "installments"
            plan = _build_installment_plan(opt)
            earliest = req_date_str

        elif can_installments and eligible_inst_in_deadline and partial_eligible:
            # ── 3. Both installments and partial available — pick cheaper ─────
            # Partial payment costs exactly req_amt (no fee).
            # Installment total may include a financing fee.
            best_inst = eligible_inst_in_deadline[0]
            inst_total = float(best_inst.total_payable_amount or req_amt)
            inst_fee = float(best_inst.financing_fee or 0)
            # partial_payment always costs req_amt; use it when installments have a fee
            # or when they start on the same day (no time advantage for installments)
            if inst_fee <= 0:
                # Same cost — prefer installments (more structured; avoids needing a
                # lump-sum on a future date the user may not have)
                opt = best_inst
                status = "affordable_with_plan"
                method = "installments"
                plan = _build_installment_plan(opt)
                earliest_val = str(full) if full else ""
                earliest = earliest_val
            else:
                # Installments cost more due to financing fee → partial_payment wins
                remainder = round(req_amt - safe, 2)
                status = "affordable_with_plan"
                method = "partial_payment"
                plan = [
                    PaymentInstallmentSchema(date=req_date_str, amount=safe),
                    PaymentInstallmentSchema(date=partial_second_date, amount=remainder),
                ]
                earliest = partial_second_date

        elif can_installments and eligible_inst_in_deadline:
            # ── 4a. Installments available, partial not eligible ──────────────
            opt = eligible_inst_in_deadline[0]
            status = "affordable_with_plan"
            method = "installments"
            plan = _build_installment_plan(opt)
            earliest = str(full) if full else ""

        elif partial_eligible:
            # ── 4b. Partial payment (installments not available) ──────────────
            remainder = round(req_amt - safe, 2)
            status = "affordable_with_plan"
            method = "partial_payment"
            plan = [
                PaymentInstallmentSchema(date=req_date_str, amount=safe),
                PaymentInstallmentSchema(date=partial_second_date, amount=remainder),
            ]
            earliest = partial_second_date

        elif full and str(full) <= deadline_str and can_full:
            # ── 5. Wait until full payment is safe ───────────────────────────
            status = "affordable_later"
            method = "wait"
            plan = [PaymentInstallmentSchema(date=str(full), amount=req_amt)]
            earliest = str(full)

        # ── Spending changes recovery (before final not_recommended) ─────────
        if status == "not_affordable" and can_full:
            shortfall = req_amt - safe
            reducible_cats = set(ctx.profile.reducible_categories)
            stoppable_cats = set(ctx.profile.stoppable_categories)
            protected = set(ctx.profile.protected_categories)
            flex_evs = [
                e for e in ctx.events
                if e.direction == "debit"
                and e.status in ("settled", "scheduled")
                and (e.category in reducible_cats or e.category in stoppable_cats)
                and e.category not in protected
            ]
            flex_evs.sort(key=lambda x: x.amount or 0, reverse=True)

            saved = 0.0
            changes: list = []
            for ev in flex_evs:
                if saved >= shortfall:
                    break
                if len(changes) >= 3:
                    break
                if ev.category in stoppable_cats and ev.flexibility in ("flexible", "stoppable"):
                    changes.append(f"stop:{ev.event_id}")
                    saved += ev.amount or 0
                elif ev.category in reducible_cats and ev.flexibility == "flexible" and ev.minimum_allowed_amount is not None:
                    saving = (ev.amount or 0) - ev.minimum_allowed_amount
                    if saving > 0:
                        changes.append(f"reduce_to:{ev.event_id}:{ev.minimum_allowed_amount}")
                        saved += saving

            if saved >= shortfall and len(changes) <= 3:
                return AgentDecisionSchema(
                    amount_safe_to_pay=safe,
                    affordability_status="affordable_with_plan",
                    recommended_payment_method="full_payment",
                    payment_plan=[PaymentInstallmentSchema(date=req_date_str, amount=req_amt)],
                    earliest_date_for_full_payment=req_date_str,
                    spending_changes_needed=changes,
                    decision_explanation="Fallback recovered via spending changes.",
                    internal_reasoning="Deterministic fallback found valid savings."
                )

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


def _enforce_status_method(decision: AgentDecisionSchema) -> AgentDecisionSchema:
    """Post-parse: enforce that status is consistent with method.

    STATUS-METHOD coupling:
      installments / partial_payment → affordable_with_plan
      wait                          → affordable_later
      not_recommended               → not_affordable
      full_payment + no spending    → affordable_now
      full_payment + spending       → affordable_with_plan
    """
    method = decision.recommended_payment_method
    changes = decision.spending_changes_needed or []
    current_status = decision.affordability_status

    correct_status = current_status  # default: keep as-is

    if method == "installments":
        correct_status = "affordable_with_plan"
    elif method == "partial_payment":
        correct_status = "affordable_with_plan"
    elif method == "wait":
        correct_status = "affordable_later"
    elif method == "not_recommended":
        correct_status = "not_affordable"
    elif method == "full_payment":
        if changes:
            if current_status not in ("affordable_with_plan", "affordable_now"):
                correct_status = "affordable_with_plan"
        else:
            if current_status not in ("affordable_now", "affordable_with_plan"):
                correct_status = "affordable_now"

    if correct_status != current_status:
        try:
            # Pydantic v2
            return decision.model_copy(update={"affordability_status": correct_status})
        except AttributeError:
            # Pydantic v1
            d = decision.dict()
            d["affordability_status"] = correct_status
            return AgentDecisionSchema(**d)
    return decision


def _installment_fits_deadline(option, deadline_str: str) -> bool:
    """Check whether the last installment of this option falls on or before deadline_str."""
    try:
        from datetime import date as _date, timedelta
        first = _date.fromisoformat(str(option.first_payment_date)[:10])
        n = option.number_of_payments or 1
        freq = int(option.payment_frequency_days or 30)
        # Last payment date uses actual frequency (not hardcoded 30d)
        last = first + timedelta(days=freq * (n - 1))
        return str(last)[:10] <= deadline_str
    except Exception:
        return True  # If we can't determine, allow it


def _build_installment_plan(option) -> list:
    """Build a PaymentInstallmentSchema list from a payment option.

    Uses payment_frequency_days from the option (not a hardcoded 30-day period).
    The last payment absorbs any cent-level rounding so the plan sums exactly
    to total_payable_amount.
    """
    try:
        from datetime import date as _date, timedelta
        first = _date.fromisoformat(str(option.first_payment_date)[:10])
        n = option.number_of_payments or 1
        freq = int(option.payment_frequency_days or 30)
        amt = round(float(option.payment_amount or 0), 2)
        total = round(float(option.total_payable_amount or amt * n), 2)
        plans = []
        running = 0.0
        for i in range(n):
            d = first + timedelta(days=freq * i)
            if i < n - 1:
                pay = amt
            else:
                # Last payment: absorb rounding remainder
                pay = round(total - running, 2)
            running += pay
            plans.append(PaymentInstallmentSchema(date=str(d), amount=pay))
        return plans
    except Exception:
        return []
