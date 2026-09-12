"""
data_loader.py – Load and validate all dataset CSVs into typed Pydantic models.

Each get_*() method returns a clean Python list of the corresponding model.
Pipe-separated string columns (categories, methods, etc.) are kept raw in the
model and exposed via @property helpers on FinancialProfile.

Currency amounts are NOT normalised here — that happens in ExchangeNormalizer
so this loader stays pure / testable.
"""
from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .schemas import (
    ExchangeRate,
    FinancialEvent,
    FinancialProfile,
    ImageRecord,
    Message,
    OutputRow,
    PaymentOption,
    PurchaseRequest,
)


class DataLoader:
    """Loads all CSVs from dataset/ and exposes typed helper methods."""

    def __init__(self, dataset_dir: str | Path | None = None):
        if dataset_dir is None:
            # Default: <repo_root>/dataset/
            dataset_dir = Path(__file__).resolve().parents[1] / "dataset"
        self.base = Path(dataset_dir)

        # Load raw DataFrames once
        self._profiles_df    = pd.read_csv(self.base / "financial_profiles.csv")
        self._events_df      = pd.read_csv(self.base / "financial_events.csv")
        self._requests_df    = pd.read_csv(self.base / "requests.csv")
        self._options_df     = pd.read_csv(self.base / "request_payment_options.csv")
        self._messages_df    = pd.read_csv(self.base / "messages.csv")
        self._images_df      = pd.read_csv(self.base / "images.csv")
        self._rates_df       = pd.read_csv(self.base / "exchange_rates.csv")

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_profiles(self) -> List[FinancialProfile]:
        rows = []
        for _, r in self._profiles_df.iterrows():
            rows.append(FinancialProfile(
                user_id=r["user_id"],
                home_currency=r["home_currency"],
                current_available_balance=float(r["current_available_balance"]),
                minimum_balance_to_keep=float(r["minimum_balance_to_keep"]),
                financial_priorities=str(r["financial_priorities"]) if not _isna(r.get("financial_priorities")) else "",
                expense_categories_to_protect=str(r["expense_categories_to_protect"]) if not _isna(r.get("expense_categories_to_protect")) else "",
                expense_categories_user_is_willing_to_reduce=_opt_str(r.get("expense_categories_user_is_willing_to_reduce")),
                expense_categories_user_is_willing_to_stop=_opt_str(r.get("expense_categories_user_is_willing_to_stop")),
                payment_methods_user_will_consider=str(r["payment_methods_user_will_consider"]) if not _isna(r.get("payment_methods_user_will_consider")) else "",
                max_installment_months=_opt_int(r.get("max_installment_months")),
            ))
        return rows

    def get_profiles_map(self) -> Dict[str, FinancialProfile]:
        return {p.user_id: p for p in self.get_profiles()}

    def get_events(self) -> List[FinancialEvent]:
        rows = []
        for _, r in self._events_df.iterrows():
            rows.append(FinancialEvent(
                event_id=str(r["event_id"]),
                user_id=str(r["user_id"]),
                event_type=str(r["event_type"]),
                description=str(r["description"]),
                category=_opt_str(r.get("category")),
                direction=str(r["direction"]),
                amount=float(r["amount"]),
                currency=str(r["currency"]),
                event_date=str(r["event_date"]),
                settlement_date=_opt_str(r.get("settlement_date")),
                status=str(r["status"]),
                linked_event_id=_opt_str(r.get("linked_event_id")),
                flexibility=_opt_str(r.get("flexibility")),
                minimum_allowed_amount=_opt_float(r.get("minimum_allowed_amount")),
            ))
        return rows

    def get_events_by_user(self) -> Dict[str, List[FinancialEvent]]:
        result: Dict[str, List[FinancialEvent]] = {}
        for ev in self.get_events():
            result.setdefault(ev.user_id, []).append(ev)
        return result

    def get_requests(self) -> List[PurchaseRequest]:
        rows = []
        for _, r in self._requests_df.iterrows():
            rows.append(PurchaseRequest(
                request_id=str(r["request_id"]),
                user_id=str(r["user_id"]),
                request_date=str(r["request_date"]),
                request_type=str(r["request_type"]),
                requested_amount=float(r["requested_amount"]),
                desired_completion_date=_opt_str(r.get("desired_completion_date")),
                allows_partial_payment=bool(str(r.get("allows_partial_payment", "false")).strip().lower() == "true"),
                request_text=str(r["request_text"]),
            ))
        return rows

    def get_requests_map(self) -> Dict[str, PurchaseRequest]:
        return {req.request_id: req for req in self.get_requests()}

    def get_payment_options(self) -> List[PaymentOption]:
        rows = []
        for _, r in self._options_df.iterrows():
            rows.append(PaymentOption(
                payment_option_id=str(r["payment_option_id"]),
                request_id=str(r["request_id"]),
                payment_method=str(r["payment_method"]),
                payment_amount=float(r["payment_amount"]),
                number_of_payments=int(r["number_of_payments"]),
                first_payment_date=_opt_str(r.get("first_payment_date")),
                payment_frequency_days=_opt_int(r.get("payment_frequency_days")),
                financing_fee=_opt_float(r.get("financing_fee")),
                total_payable_amount=float(r["total_payable_amount"]),
            ))
        return rows

    def get_payment_options_by_request(self) -> Dict[str, List[PaymentOption]]:
        result: Dict[str, List[PaymentOption]] = {}
        for opt in self.get_payment_options():
            result.setdefault(opt.request_id, []).append(opt)
        return result

    def get_messages(self) -> List[Message]:
        rows = []
        for _, r in self._messages_df.iterrows():
            rows.append(Message(
                message_id=str(r["message_id"]),
                user_id=str(r["user_id"]),
                request_id=_opt_str(r.get("request_id")),
                related_event_id=_opt_str(r.get("related_event_id")),
                sent_at=str(r["sent_at"]),
                source_type=str(r["source_type"]),
                message_text=str(r["message_text"]),
            ))
        return rows

    def get_messages_by_user(self) -> Dict[str, List[Message]]:
        result: Dict[str, List[Message]] = {}
        for msg in self.get_messages():
            result.setdefault(msg.user_id, []).append(msg)
        return result

    def get_image_records(self) -> List[ImageRecord]:
        rows = []
        images_dir = self.base / "media" / "images"
        for _, r in self._images_df.iterrows():
            img_id = str(r["image_id"])
            img_path = str(images_dir / f"{img_id}.png")
            rows.append(ImageRecord(
                image_id=img_id,
                user_id=str(r["user_id"]),
                request_id=_opt_str(r.get("request_id")),
                related_event_id=_opt_str(r.get("related_event_id")),
                image_path=img_path,
            ))
        return rows

    def get_exchange_rates(self) -> List[ExchangeRate]:
        rows = []
        for _, r in self._rates_df.iterrows():
            rows.append(ExchangeRate(
                rate_date=str(r["rate_date"]),
                from_currency=str(r["from_currency"]),
                to_currency=str(r["to_currency"]),
                rate=float(r["rate"]),
            ))
        return rows

    # ------------------------------------------------------------------
    # Quick summary for debugging
    # ------------------------------------------------------------------
    def summary(self) -> str:
        return (
            f"Profiles: {len(self._profiles_df)} | "
            f"Events: {len(self._events_df)} | "
            f"Requests: {len(self._requests_df)} | "
            f"Options: {len(self._options_df)} | "
            f"Messages: {len(self._messages_df)} | "
            f"Images: {len(self._images_df)} | "
            f"Rates: {len(self._rates_df)}"
        )


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _isna(val) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    return str(val).strip() in ("", "nan", "NaN", "None")

def _opt_str(val) -> Optional[str]:
    return None if _isna(val) else str(val).strip()

def _opt_int(val) -> Optional[int]:
    if _isna(val):
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None

def _opt_float(val) -> Optional[float]:
    if _isna(val):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
