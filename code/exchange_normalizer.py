"""
exchange_normalizer.py – Convert any amount to a target currency using fixed rates.

Rules (from AGENTS.md §6.1):
- Use the row for the event's settlement_date and the stated from/to direction.
- Rates are fixed; no live lookups.
- Supported currencies: ZAR, IDR, EUR, INR, USD.
  The dataset only has pairs relative to USD/EUR, so we build a bidirectional
  lookup at load time and derive cross-rates via USD as base.
"""
from __future__ import annotations
from typing import Dict, Tuple
import pandas as pd
from datetime import date, timedelta


RateKey = Tuple[str, str, str]   # (rate_date_str, from_currency, to_currency)


class ExchangeNormalizer:
    """Load exchange_rates.csv and provide convert(amount, from_cur, to_cur, on_date) → float."""

    def __init__(self, rates_path: str):
        df = pd.read_csv(rates_path)
        df["rate_date"] = pd.to_datetime(df["rate_date"]).dt.date
        self._df = df
        # Build lookup: (date_str, from, to) -> rate
        self._lookup: Dict[RateKey, float] = {}
        for _, row in df.iterrows():
            key = (str(row["rate_date"]), row["from_currency"], row["to_currency"])
            self._lookup[key] = float(row["rate"])
        # Available dates sorted
        self._dates = sorted(df["rate_date"].unique())

    def _nearest_date(self, on: date) -> date:
        """Return the nearest rate_date <= on; fall back to the earliest if none."""
        candidates = [d for d in self._dates if d <= on]
        if candidates:
            return max(candidates)
        return self._dates[0]

    def _direct_rate(self, from_cur: str, to_cur: str, rate_date: date) -> float | None:
        d = self._nearest_date(rate_date)
        key = (str(d), from_cur, to_cur)
        return self._lookup.get(key)

    def convert(self, amount: float, from_cur: str, to_cur: str, on_date: date) -> float:
        """Convert amount from from_cur to to_cur using rates valid on on_date."""
        if from_cur == to_cur:
            return round(amount, 6)

        # Try direct
        rate = self._direct_rate(from_cur, to_cur, on_date)
        if rate is not None:
            return round(amount * rate, 6)

        # Try inverse
        inv = self._direct_rate(to_cur, from_cur, on_date)
        if inv is not None and inv != 0:
            return round(amount / inv, 6)

        # Cross via USD
        # from_cur -> USD
        from_to_usd = self._direct_rate(from_cur, "USD", on_date)
        if from_to_usd is None:
            inv2 = self._direct_rate("USD", from_cur, on_date)
            from_to_usd = (1.0 / inv2) if inv2 else None

        # USD -> to_cur
        usd_to_target = self._direct_rate("USD", to_cur, on_date)
        if usd_to_target is None:
            inv3 = self._direct_rate(to_cur, "USD", on_date)
            usd_to_target = (1.0 / inv3) if inv3 else None

        if from_to_usd and usd_to_target:
            return round(amount * from_to_usd * usd_to_target, 6)

        raise ValueError(
            f"No exchange rate path found: {from_cur} -> {to_cur} on {on_date}"
        )
