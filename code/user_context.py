"""
user_context.py — Unified UserContext object (Phase 1 completion)

This is the single output of the ingestion layer. The financial simulator
(Phase 2) and the Actor-Critic loop (Phase 3) both operate exclusively on
UserContext objects — never on raw DataFrames.

For each user, UserContext holds:
  - their financial profile
  - their events (currency-normalised to home_currency, delta-patches applied)
  - their messages
  - their image records (with extracted amounts if VLM ran)
  - a dict of payment options keyed by request_id
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional

from .schemas import (
    ExchangeRate,
    FinancialEvent,
    FinancialProfile,
    ImageRecord,
    Message,
    PaymentOption,
    PurchaseRequest,
)
from .exchange_normalizer import ExchangeNormalizer


@dataclass
class UserContext:
    profile: FinancialProfile
    events: List[FinancialEvent]          # patched + currency-normalised
    messages: List[Message]
    images: List[ImageRecord]
    payment_options: Dict[str, List[PaymentOption]]  # request_id -> options
    normalizer: ExchangeNormalizer        # shared across all users
    # Authoritative go-forward salary from a payroll message patch.
    # {"amount": float (home currency), "first_date": date | None}. None when no override.
    salary_override: Optional[dict] = None

    # ---------------------------------------------------------
    # Convenience: events split by status
    # ---------------------------------------------------------

    @property
    def settled_events(self) -> List[FinancialEvent]:
        return [e for e in self.events if e.status == "settled"]

    @property
    def pending_events(self) -> List[FinancialEvent]:
        return [e for e in self.events if e.status == "pending"]

    @property
    def scheduled_events(self) -> List[FinancialEvent]:
        return [e for e in self.events if e.status == "scheduled"]

    @property
    def active_debits(self) -> List[FinancialEvent]:
        """Settled + pending + scheduled debits (things that will cost money)."""
        return [e for e in self.events
                if e.direction == "debit"
                and e.status in ("settled", "pending", "scheduled")]

    @property
    def confirmed_credits(self) -> List[FinancialEvent]:
        """Only settled credits count as usable income."""
        return [e for e in self.events
                if e.direction == "credit"
                and e.status == "settled"]


def build_user_contexts(
    profiles: List[FinancialProfile],
    events_by_user: Dict[str, List[FinancialEvent]],
    messages_by_user: Dict[str, List[Message]],
    images_by_user: Dict[str, List[ImageRecord]],
    payment_options_by_request: Dict[str, List[PaymentOption]],
    requests: List[PurchaseRequest],
    normalizer: ExchangeNormalizer,
) -> Dict[str, UserContext]:
    """
    Build one UserContext per user. Events are currency-normalised to the
    user's home_currency using the provided ExchangeNormalizer.
    """
    # Which payment options belong to which user?
    request_user: Dict[str, str] = {req.request_id: req.user_id for req in requests}

    contexts: Dict[str, UserContext] = {}
    for profile in profiles:
        uid = profile.user_id
        raw_events = events_by_user.get(uid, [])

        # Normalise event amounts to home_currency
        normalised_events: List[FinancialEvent] = []
        for ev in raw_events:
            if ev.currency == profile.home_currency:
                normalised_events.append(ev)
            else:
                try:
                    ref_date = _parse_date(ev.settlement_date or ev.event_date)
                    converted = normalizer.convert(
                        ev.amount, ev.currency, profile.home_currency, ref_date
                    )
                    normalised_events.append(
                        ev.model_copy(update={"amount": converted, "currency": profile.home_currency})
                    )
                except Exception:
                    # Keep original if conversion fails; Phase 2 will log a warning
                    normalised_events.append(ev)

        # Collect payment options for this user's requests
        user_payment_opts: Dict[str, List[PaymentOption]] = {}
        for req in requests:
            if req.user_id == uid:
                user_payment_opts[req.request_id] = payment_options_by_request.get(req.request_id, [])

        contexts[uid] = UserContext(
            profile=profile,
            events=normalised_events,
            messages=messages_by_user.get(uid, []),
            images=images_by_user.get(uid, []),
            payment_options=user_payment_opts,
            normalizer=normalizer,
        )

    return contexts


def _parse_date(date_str: Optional[str]) -> date:
    if not date_str:
        return date.today()
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        except ValueError:
            pass
    return date.today()
