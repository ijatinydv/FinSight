"""
finance_engine.py — Deterministic Financial State Reconstruction (Phase 2)

Components:
  RecurrenceDetector  — finds recurring events from history (≥3 occurrences, stable period)
  DailyBalanceSimulator — projects 90-day rolling balance from request_date
  AffordabilityCalc   — computes amount_safe_to_pay and earliest_date_for_full_payment

Design rules (from AGENTS.md §6.3):
  - Reserve pending debits. NEVER count pending credits / bonuses / refunds / investment gains.
  - Count confirmed salary on its settlement date only.
  - Balance must never fall below minimum_balance_to_keep on any day in the plan.
  - Detect recurrence only when ≥3 settled occurrences with period std_dev < 5 days.
  - Forecast essential variable spending conservatively (use trimmed mean of last 3 months).
"""
from __future__ import annotations

import calendar
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from .schemas import FinancialEvent, FinancialProfile
from .user_context import UserContext


# ─────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────

@dataclass
class RecurringPattern:
    key: str                    # "{category}:{direction}"
    avg_amount: float
    avg_period_days: float
    last_date: date
    direction: str              # "debit" | "credit"
    category: str
    flexibility: Optional[str] = None
    event_ids: List[str] = field(default_factory=list)  # source event ids


@dataclass
class DayEntry:
    date: date
    balance_start: float
    balance_end: float
    credits: float
    debits: float
    notes: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────
# 1. Recurrence Detector
# ─────────────────────────────────────────────────────────

class RecurrenceDetector:
    """
    Identify recurring patterns from settled (and confirmed-scheduled) events.

    Primary rule: ≥3 settled occurrences with period std_dev < 5 days.
    Fallback rule: ≥2 settled occurrences + 1 matching scheduled future event
    confirms the pattern (e.g. salary already confirmed for next month).
    """

    MIN_OCCURRENCES = 3
    MIN_OCCURRENCES_WITH_SCHEDULED = 2   # lower bar when a scheduled event confirms
    MIN_OCCURRENCES_SALARY = 1           # salary/income: 1 settled + 1 scheduled suffices
    MAX_PERIOD_STDDEV = 5.0
    MAX_LOOKBACK_DAYS = 365

    SALARY_CATEGORIES = {"salary", "income", "payroll", "wages"}

    def detect(
        self,
        events: List[FinancialEvent],
        as_of: date,
        salary_override: Optional[dict] = None,
    ) -> List[RecurringPattern]:
        """Return list of detected recurring patterns as of `as_of` date.

        `salary_override` (when present) is the authoritative go-forward pay from a
        payroll message patch: {"amount": float, "first_date": date | None}. It wins
        over history-derived salary projection (a raise/reduction/new job).
        """
        cutoff = as_of - timedelta(days=self.MAX_LOOKBACK_DAYS)

        settled = [
            e for e in events
            if e.status == "settled"
            and e.direction in ("debit", "credit")
            and _parse_date(e.settlement_date or e.event_date) <= as_of
            and _parse_date(e.settlement_date or e.event_date) >= cutoff
        ]

        # Index of scheduled future events by (category, direction) for confirmation
        scheduled_future: Dict[str, List[FinancialEvent]] = defaultdict(list)
        for e in events:
            if e.status == "scheduled" and e.direction in ("debit", "credit"):
                ev_date = _parse_date(e.settlement_date or e.event_date)
                if ev_date > as_of:
                    key = f"{e.category or 'uncategorized'}:{e.direction}"
                    scheduled_future[key].append(e)

        # Group settled by (category, direction)
        groups: Dict[str, List[FinancialEvent]] = defaultdict(list)
        
        # Keywords that indicate variable income that should NOT be projected
        volatile_keywords = {"bonus", "commission", "refund", "lottery", "gain"}
        
        for ev in settled:
            cat = ev.category or "uncategorized"
            desc = (ev.description or "").lower()
            
            # Do not project historical bonuses/commissions per AGENTS.md rules
            if ev.direction == "credit" and any(k in desc for k in volatile_keywords):
                continue
                
            key = f"{cat}:{ev.direction}"
            groups[key].append(ev)

        patterns: List[RecurringPattern] = []
        seen_keys = set()
        
        # Build set of cancelled categories to skip pattern projection
        cancelled_keys = set()
        for ev in events:
            if ev.status == "cancelled":
                cat = ev.category or "uncategorized"
                cancelled_keys.add(f"{cat}:{ev.direction}")

        for key, evs in groups.items():
            if key in cancelled_keys:
                continue
                
            has_scheduled_confirm = key in scheduled_future
            category_name = key.rsplit(":", 1)[0]
            is_salary = category_name in self.SALARY_CATEGORIES

            if has_scheduled_confirm and is_salary:
                min_occ = self.MIN_OCCURRENCES_SALARY
            elif has_scheduled_confirm:
                min_occ = self.MIN_OCCURRENCES_WITH_SCHEDULED
            else:
                min_occ = self.MIN_OCCURRENCES

            if len(evs) < min_occ:
                continue

            # Sort by date
            evs_sorted = sorted(evs, key=lambda e: _parse_date(e.settlement_date or e.event_date))
            dates = [_parse_date(e.settlement_date or e.event_date) for e in evs_sorted]

            # Compute inter-arrival periods; if scheduled confirm exists, include its date too
            if has_scheduled_confirm and len(evs) >= 1:
                # Use the nearest scheduled occurrence to compute period
                sched_dates = sorted(
                    _parse_date(e.settlement_date or e.event_date)
                    for e in scheduled_future[key]
                )
                all_dates = sorted(dates + sched_dates)
                periods = [(all_dates[i+1] - all_dates[i]).days for i in range(len(all_dates)-1)]
            else:
                periods = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]

            if not periods:
                continue
            # Filter outlier periods (e.g. one-time bonus between regular salary dates)
            periods_filtered = _filter_period_outliers(periods)
            if not periods_filtered:
                continue
            avg_period = sum(periods_filtered) / len(periods_filtered)
            std_period = _stddev(periods_filtered)

            if std_period > self.MAX_PERIOD_STDDEV and not has_scheduled_confirm:
                continue
            # If scheduled confirms the pattern, be a bit more lenient on std_dev
            if std_period > 8.0:
                continue


            # Compute amount
            amounts = [e.amount for e in evs_sorted[-6:]]
            avg_amount = _trimmed_mean(amounts)
            
            # CRITICAL FIX for salary_override patches:
            # If the most recent amount is significantly different from the trimmed mean
            # (e.g., >10% diff), assume a structural change (like a salary increase) 
            # and use the most recent amount for future projections instead.
            latest_amount = evs_sorted[-1].amount
            if (
                is_salary
                and abs(latest_amount - avg_amount) > (0.10 * avg_amount)
                and len(amounts) >= 2
            ):
                # Structural pay change (raise/reduction) — salary/income only.
                # Variable expenses keep the trimmed mean so a one-off spike
                # (e.g. a bulk grocery run) is not baked in as the recurring amount.
                avg_amount = latest_amount
            # last_date = most recent known occurrence (settled or scheduled)
            if has_scheduled_confirm:
                sched_evs = scheduled_future[key]
                last_known = max(
                    _parse_date(e.settlement_date or e.event_date) for e in sched_evs
                )
                last_date = max(dates[-1], last_known)
            else:
                last_date = dates[-1]

            category, direction = key.rsplit(":", 1)
            patterns.append(RecurringPattern(
                key=key,
                avg_amount=avg_amount,
                avg_period_days=avg_period,
                last_date=last_date,
                direction=direction,
                category=category,
                flexibility=evs_sorted[-1].flexibility,
                event_ids=[e.event_id for e in evs_sorted[-3:]],
            ))
            seen_keys.add(key)

        # A payroll-message salary override is authoritative: it replaces the
        # history-derived salary projection (fixes silently-dropped overrides).
        if salary_override and salary_override.get("amount"):
            patterns = self._apply_salary_override(patterns, salary_override, as_of)

        return patterns

    def _apply_salary_override(
        self,
        patterns: List[RecurringPattern],
        override: dict,
        as_of: date,
    ) -> List[RecurringPattern]:
        """
        Fold an authoritative salary override into the detected patterns.

        - Set the primary salary/income credit pattern's amount to the override.
        - Force a monthly cadence when the detected period is implausible for pay
          (e.g. gig income merged into 10.7d, or a sparse 91d gap).
        - Anchor the first occurrence to the patched effective date when given.
        - Drop secondary salary/income patterns (the override is the total pay).
        - Synthesize a monthly pattern when history had none (e.g. a first salary).
        """
        amount = float(override["amount"])
        first_date = override.get("first_date")

        sal_pats = [
            p for p in patterns
            if p.direction == "credit" and p.category in self.SALARY_CATEGORIES
        ]

        if sal_pats:
            primary = max(sal_pats, key=lambda p: p.avg_amount)
            primary.avg_amount = amount
            if primary.avg_period_days < 20 or primary.avg_period_days > 45:
                primary.avg_period_days = 30.0
            if first_date:
                primary.last_date = first_date
            for p in sal_pats:
                if p is not primary:
                    patterns.remove(p)
        else:
            anchor = first_date if first_date else (as_of + timedelta(days=30))
            patterns.append(RecurringPattern(
                key="salary:credit",
                avg_amount=amount,
                avg_period_days=30.0,
                last_date=anchor,
                direction="credit",
                category="salary",
                flexibility="essential",
                event_ids=[],
            ))
        return patterns


# ─────────────────────────────────────────────────────────
# 2. Daily Balance Simulator
# ─────────────────────────────────────────────────────────

class DailyBalanceSimulator:
    """
    Project the user's balance day-by-day over [request_date, request_date + 90 days].

    Cash-flow rules:
      ADD:      settled credits on settlement_date
                projected recurring credits (salary confirmed from settled history)
      SUBTRACT: settled debits on settlement_date (if future of request_date)
                pending debits (on event_date)
                scheduled debits (on event_date)
                projected recurring debits
      EXCLUDE:  pending credits, failed/cancelled events, unrealized investment values
    """

    HORIZON_DAYS = 90

    def simulate(
        self,
        ctx: UserContext,
        request_date: date,
        recurring_patterns: List[RecurringPattern],
        payment_installments: Optional[List[Tuple[date, float]]] = None,
    ) -> List[DayEntry]:
        """
        Returns a list of 91 DayEntry objects (day 0 through day 90).
        payment_installments: optional list of (date, amount) debits to include in projection.
        """
        balance = ctx.profile.current_available_balance
        day_entries: List[DayEntry] = []

        # Pre-index concrete future events by date
        future_debits: Dict[date, List[FinancialEvent]] = defaultdict(list)
        future_credits: Dict[date, List[FinancialEvent]] = defaultdict(list)

        for ev in ctx.events:
            ev_date = _parse_date(ev.settlement_date or ev.event_date)
            if ev_date < request_date:
                continue  # historical, already reflected in current_balance
            if ev.status in ("failed", "cancelled", "unrealized"):
                continue
            # Skip events with missing/NaN amounts (image events awaiting VLM extraction)
            safe_amt = _safe_amount(ev.amount)
            if safe_amt == 0.0:
                continue
            if ev.direction == "debit" and ev.status in ("pending", "scheduled", "settled"):
                future_debits[ev_date].append(ev)
            elif ev.direction == "credit" and ev.status in ("settled", "scheduled"):
                # settled = confirmed past income; scheduled = confirmed future income (e.g. salary)
                # pending credits are excluded per rules (bonuses, refunds, commissions, investment gains)
                future_credits[ev_date].append(ev)

        # Project recurring patterns into the horizon
        proj_debits: Dict[date, float] = defaultdict(float)
        proj_credits: Dict[date, float] = defaultdict(float)
        end_date = request_date + timedelta(days=self.HORIZON_DAYS)

        for pat in recurring_patterns:
            # Next occurrence after request_date
            next_occ = pat.last_date
            while next_occ <= request_date:
                next_occ = _step_recurring(next_occ, pat.avg_period_days)

            while next_occ <= end_date:
                # Don't double-count if a concrete event exists on same day for same category
                if pat.direction == "debit":
                    # check if concrete event covers this category on this day
                    concrete = future_debits.get(next_occ, [])
                    if not any(e.category == pat.category for e in concrete):
                        proj_debits[next_occ] += pat.avg_amount
                else:
                    concrete_c = future_credits.get(next_occ, [])
                    if not any(e.category == pat.category for e in concrete_c):
                        proj_credits[next_occ] += pat.avg_amount
                next_occ = _step_recurring(next_occ, pat.avg_period_days)

        # Simulate day by day
        for day_offset in range(self.HORIZON_DAYS + 1):
            current_date = request_date + timedelta(days=day_offset)
            notes: List[str] = []

            day_credits = 0.0
            day_debits = 0.0

            # Concrete settled credits
            for ev in future_credits.get(current_date, []):
                day_credits += ev.amount
                notes.append(f"+{ev.amount:.0f} {ev.category} ({ev.event_id})")

            # Projected recurring credits
            if current_date in proj_credits:
                amt = proj_credits[current_date]
                day_credits += amt
                notes.append(f"+{amt:.0f} (projected credit)")

            # Concrete debits (pending + scheduled + settled future)
            for ev in future_debits.get(current_date, []):
                day_debits += ev.amount
                notes.append(f"-{ev.amount:.0f} {ev.category} ({ev.event_id})")

            # Projected recurring debits
            if current_date in proj_debits:
                amt = proj_debits[current_date]
                day_debits += amt
                notes.append(f"-{amt:.0f} (projected debit)")

            # Optional payment installments (for plan simulation)
            if payment_installments:
                for pay_date, pay_amt in payment_installments:
                    if pay_date == current_date:
                        day_debits += pay_amt
                        notes.append(f"-{pay_amt:.0f} (payment installment)")

            balance_end = balance + day_credits - day_debits
            day_entries.append(DayEntry(
                date=current_date,
                balance_start=balance,
                balance_end=balance_end,
                credits=day_credits,
                debits=day_debits,
                notes=notes,
            ))
            balance = balance_end

        return day_entries


# ─────────────────────────────────────────────────────────
# 3. Affordability Calculator
# ─────────────────────────────────────────────────────────

class AffordabilityCalculator:
    """
    Compute:
      amount_safe_to_pay  — max amount payable on request_date without ever breaching min_balance
      earliest_date_for_full_payment — first day in [0..90] when the full requested_amount is safe
    """

    def __init__(self, simulator: DailyBalanceSimulator, detector: RecurrenceDetector):
        self.simulator = simulator
        self.detector = detector

    def compute(
        self,
        ctx: UserContext,
        request_date: date,
        requested_amount: float,
    ) -> "AffordabilityResult":
        min_bal = ctx.profile.minimum_balance_to_keep
        patterns = self.detector.detect(
            ctx.events,
            as_of=request_date,
            salary_override=getattr(ctx, "salary_override", None),
        )

        # Baseline 90-day projection (no payment)
        baseline = self.simulator.simulate(ctx, request_date, patterns)

        # Binary search: max safe lump-sum payment on request_date
        safe_amount = _binary_search_safe_amount(
            ctx, request_date, patterns, min_bal, requested_amount, self.simulator
        )

        # Find earliest date for full payment
        earliest_full = _find_earliest_full_payment(
            ctx, request_date, patterns, min_bal, requested_amount, self.simulator
        )

        return AffordabilityResult(
            requested_amount=requested_amount,
            amount_safe_to_pay=safe_amount,
            earliest_date_for_full_payment=earliest_full,
            min_balance=min_bal,
            baseline_days=baseline,
            recurring_patterns=patterns,
        )


@dataclass
class AffordabilityResult:
    requested_amount: float
    amount_safe_to_pay: float
    earliest_date_for_full_payment: Optional[date]
    min_balance: float
    baseline_days: List[DayEntry]
    recurring_patterns: List[RecurringPattern]

    @property
    def can_pay_now(self) -> bool:
        return self.amount_safe_to_pay >= self.requested_amount

    @property
    def min_balance_in_baseline(self) -> float:
        return min(d.balance_end for d in self.baseline_days)

    def balance_on_day(self, d: date) -> Optional[float]:
        for entry in self.baseline_days:
            if entry.date == d:
                return entry.balance_end
        return None

    def validate_plan(
        self,
        installments: List[Tuple[date, float]],
        min_bal: float,
        simulator: DailyBalanceSimulator,
        ctx: UserContext,
        request_date: date,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that a proposed installment plan never breaches min_balance.
        Returns (is_valid, failure_reason_or_None).
        """
        days = simulator.simulate(ctx, request_date, self.recurring_patterns, installments)
        for day in days:
            if day.balance_end < min_bal:
                shortfall = min_bal - day.balance_end
                return False, (
                    f"Balance breaches minimum by {shortfall:.2f} on {day.date} "
                    f"(balance={day.balance_end:.2f}, min={min_bal:.2f})"
                )
        return True, None


# ─────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────

def _binary_search_safe_amount(
    ctx: UserContext,
    request_date: date,
    patterns: List[RecurringPattern],
    min_bal: float,
    max_amount: float,
    simulator: DailyBalanceSimulator,
    precision: float = 0.01,
) -> float:
    """Binary search for the max lump-sum payable on request_date."""
    lo, hi = 0.0, max_amount
    safe = 0.0

    for _ in range(40):  # 40 iterations gives sub-cent precision
        mid = (lo + hi) / 2
        days = simulator.simulate(ctx, request_date, patterns, [(request_date, mid)])
        if all(d.balance_end >= min_bal for d in days):
            safe = mid
            lo = mid
        else:
            hi = mid
        if hi - lo < precision:
            break

    return round(safe, 2)


def _find_earliest_full_payment(
    ctx: UserContext,
    request_date: date,
    patterns: List[RecurringPattern],
    min_bal: float,
    requested_amount: float,
    simulator: DailyBalanceSimulator,
) -> Optional[date]:
    """First day d in [0..90] where a lump-sum payment of requested_amount is safe."""
    for day_offset in range(91):
        pay_date = request_date + timedelta(days=day_offset)
        days = simulator.simulate(ctx, request_date, patterns, [(pay_date, requested_amount)])
        if all(d.balance_end >= min_bal for d in days):
            return pay_date
    return None  # Not feasible within 90 days


def _parse_date(date_str: Optional[str]) -> date:
    if not date_str:
        return date.today()
    try:
        return date.fromisoformat(str(date_str)[:10])
    except ValueError:
        return date.today()


def _step_recurring(d: date, period_days: float) -> date:
    """
    Advance a recurring occurrence by one period.

    Monthly-ish patterns (26-35 days) step by *calendar month* so the day-of-month
    payday is preserved (a salary on the 15th stays on the 15th). This removes the
    round(30.4)->30 accumulation drift that pushed earliest_date off by 1-3 days.
    Weekly / bi-weekly / other patterns step by whole days as before.
    """
    p = round(period_days)
    if 26 <= p <= 35:
        return _add_one_calendar_month(d)
    return d + timedelta(days=p)


def _add_one_calendar_month(d: date) -> date:
    """Add one calendar month, clamping the day to the target month's last day."""
    year = d.year + (1 if d.month == 12 else 0)
    month = 1 if d.month == 12 else d.month + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _stddev(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def _trimmed_mean(values: List[float], trim: float = 0.1) -> float:
    """Mean after trimming `trim` fraction from each end."""
    n = len(values)
    if n == 0:
        return 0.0
    sorted_vals = sorted(values)
    cut = max(1, int(n * trim))
    trimmed = sorted_vals[cut: n - cut] if n - 2 * cut > 0 else sorted_vals
    return sum(trimmed) / len(trimmed)


def _filter_period_outliers(periods: List[float], tolerance: float = 10.0) -> List[float]:
    """
    Remove period values that deviate more than `tolerance` days from the median.
    Handles one-off bonus payments or duplicate transactions that appear between
    regular monthly/weekly recurring events.
    """
    if not periods:
        return periods
    sorted_p = sorted(periods)
    median = sorted_p[len(sorted_p) // 2]
    filtered = [p for p in periods if abs(p - median) <= tolerance]
    return filtered if filtered else periods  # fallback: return original if all filtered


def _safe_amount(amount) -> float:
    """Return 0.0 for None/NaN amounts (image events not yet extracted)."""
    try:
        v = float(amount)
        return 0.0 if math.isnan(v) else v
    except (TypeError, ValueError):
        return 0.0
