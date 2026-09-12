"""
schemas.py – Pydantic models matching the EXACT column names of every dataset CSV.
All CSV columns are faithfully represented; optional columns use Optional[type].
"""
from __future__ import annotations
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# financial_profiles.csv
# ---------------------------------------------------------------------------
class FinancialProfile(BaseModel):
    user_id: str
    home_currency: str
    current_available_balance: float
    minimum_balance_to_keep: float
    financial_priorities: str                        # pipe-separated, e.g. "education|debt_repayment"
    expense_categories_to_protect: str              # pipe-separated
    expense_categories_user_is_willing_to_reduce: Optional[str] = None
    expense_categories_user_is_willing_to_stop: Optional[str] = None
    payment_methods_user_will_consider: str          # pipe-separated
    max_installment_months: Optional[int] = None

    # ---- convenience helpers -----------------------------------------------
    @property
    def protected_categories(self) -> List[str]:
        return [c.strip() for c in (self.expense_categories_to_protect or "").split("|") if c.strip()]

    @property
    def reducible_categories(self) -> List[str]:
        return [c.strip() for c in (self.expense_categories_user_is_willing_to_reduce or "").split("|") if c.strip()]

    @property
    def stoppable_categories(self) -> List[str]:
        return [c.strip() for c in (self.expense_categories_user_is_willing_to_stop or "").split("|") if c.strip()]

    @property
    def payment_methods(self) -> List[str]:
        return [m.strip() for m in (self.payment_methods_user_will_consider or "").split("|") if m.strip()]

    @property
    def priorities(self) -> List[str]:
        return [p.strip() for p in (self.financial_priorities or "").split("|") if p.strip()]


# ---------------------------------------------------------------------------
# financial_events.csv
# ---------------------------------------------------------------------------
class FinancialEvent(BaseModel):
    event_id: str
    user_id: str
    event_type: str
    description: str
    category: Optional[str] = None
    direction: str          # "debit" | "credit"
    amount: float
    currency: str
    event_date: str
    settlement_date: Optional[str] = None
    status: str             # settled | pending | scheduled | failed | cancelled | unrealized
    linked_event_id: Optional[str] = None
    flexibility: Optional[str] = None   # fixed | flexible | stoppable
    minimum_allowed_amount: Optional[float] = None


# ---------------------------------------------------------------------------
# requests.csv
# ---------------------------------------------------------------------------
class PurchaseRequest(BaseModel):
    request_id: str
    user_id: str
    request_date: str
    request_type: str
    requested_amount: float
    desired_completion_date: Optional[str] = None
    allows_partial_payment: bool
    request_text: str


# ---------------------------------------------------------------------------
# request_payment_options.csv
# ---------------------------------------------------------------------------
class PaymentOption(BaseModel):
    payment_option_id: str
    request_id: str
    payment_method: str         # full_payment | installments | partial_payment | wait
    payment_amount: float
    number_of_payments: int
    first_payment_date: Optional[str] = None
    payment_frequency_days: Optional[int] = None
    financing_fee: Optional[float] = None
    total_payable_amount: float


# ---------------------------------------------------------------------------
# messages.csv
# ---------------------------------------------------------------------------
class Message(BaseModel):
    message_id: str
    user_id: str
    request_id: Optional[str] = None
    related_event_id: Optional[str] = None
    sent_at: str
    source_type: str
    message_text: str


# ---------------------------------------------------------------------------
# images.csv
# ---------------------------------------------------------------------------
class ImageRecord(BaseModel):
    image_id: str
    user_id: str
    request_id: Optional[str] = None
    related_event_id: Optional[str] = None
    # resolved at load time
    image_path: Optional[str] = None
    extracted_amount: Optional[float] = None   # filled by VLM extraction


# ---------------------------------------------------------------------------
# exchange_rates.csv
# ---------------------------------------------------------------------------
class ExchangeRate(BaseModel):
    rate_date: str
    from_currency: str
    to_currency: str
    rate: float


# ---------------------------------------------------------------------------
# output.csv – what we must write
# ---------------------------------------------------------------------------
class OutputRow(BaseModel):
    request_id: str
    amount_safe_to_pay: float
    affordability_status: Literal[
        "affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"
    ]
    recommended_payment_method: Literal[
        "full_payment", "partial_payment", "installments", "wait", "not_recommended"
    ]
    payment_plan: str           # "none" or "YYYY-MM-DD:amount|..."
    earliest_date_for_full_payment: Optional[str] = None
    spending_changes_needed: str   # "none" or "stop:<event_id>|reduce_to:<event_id>:<amount>"
    decision_explanation: str
