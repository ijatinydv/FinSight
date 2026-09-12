"""
tests/test_data_loader.py

Phase 1 validation: ensure all CSVs load without errors and model counts match.
Run: python -m pytest tests/ -v
"""
import sys
from pathlib import Path

# Allow import from `code` package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.data_loader import DataLoader
from code.exchange_normalizer import ExchangeNormalizer
from datetime import date


DATASET = Path(__file__).resolve().parents[1] / "dataset"


def test_profiles_load():
    loader = DataLoader(DATASET)
    profiles = loader.get_profiles()
    assert len(profiles) > 0, "No profiles loaded"
    # Check first profile has required fields
    p = profiles[0]
    assert p.user_id
    assert p.home_currency
    assert p.current_available_balance >= 0
    assert p.minimum_balance_to_keep >= 0
    print(f"  Loaded {len(profiles)} profiles")


def test_events_load():
    loader = DataLoader(DATASET)
    events = loader.get_events()
    assert len(events) > 0, "No events loaded"
    ev = events[0]
    assert ev.event_id
    assert ev.direction in ("debit", "credit")
    assert ev.amount > 0
    print(f"  Loaded {len(events)} events")


def test_requests_load():
    loader = DataLoader(DATASET)
    reqs = loader.get_requests()
    assert len(reqs) > 0, "No requests loaded"
    r = reqs[0]
    assert r.request_id
    assert r.requested_amount > 0
    print(f"  Loaded {len(reqs)} requests")


def test_payment_options_load():
    loader = DataLoader(DATASET)
    opts = loader.get_payment_options()
    assert len(opts) > 0, "No payment options loaded"
    print(f"  Loaded {len(opts)} payment options")


def test_messages_load():
    loader = DataLoader(DATASET)
    msgs = loader.get_messages()
    assert len(msgs) > 0, "No messages loaded"
    print(f"  Loaded {len(msgs)} messages")


def test_images_load():
    loader = DataLoader(DATASET)
    imgs = loader.get_image_records()
    assert len(imgs) > 0, "No image records loaded"
    # All image paths should be constructed correctly
    for img in imgs:
        assert img.image_path and img.image_path.endswith(".png")
    print(f"  Loaded {len(imgs)} image records")


def test_exchange_rates_load():
    loader = DataLoader(DATASET)
    rates = loader.get_exchange_rates()
    assert len(rates) > 0, "No exchange rates loaded"
    print(f"  Loaded {len(rates)} exchange rates")


def test_exchange_normalizer_same_currency():
    norm = ExchangeNormalizer(str(DATASET / "exchange_rates.csv"))
    result = norm.convert(1000.0, "EUR", "EUR", date(2024, 1, 15))
    assert result == 1000.0


def test_exchange_normalizer_eur_to_zar():
    norm = ExchangeNormalizer(str(DATASET / "exchange_rates.csv"))
    result = norm.convert(100.0, "EUR", "ZAR", date(2024, 1, 15))
    # Rate should be 20 EUR->ZAR based on dataset
    assert result == 2000.0, f"Expected 2000.0, got {result}"


def test_events_by_user_grouping():
    loader = DataLoader(DATASET)
    by_user = loader.get_events_by_user()
    assert len(by_user) > 0
    # Each user should have at least 1 event
    for user_id, evs in by_user.items():
        assert len(evs) > 0, f"User {user_id} has no events"


def test_profile_helpers():
    loader = DataLoader(DATASET)
    profiles = loader.get_profiles()
    p = profiles[0]
    # Check pipe-split helpers
    assert isinstance(p.protected_categories, list)
    assert isinstance(p.payment_methods, list)
    assert isinstance(p.priorities, list)


if __name__ == "__main__":
    test_profiles_load()
    test_events_load()
    test_requests_load()
    test_payment_options_load()
    test_messages_load()
    test_images_load()
    test_exchange_rates_load()
    test_exchange_normalizer_same_currency()
    test_exchange_normalizer_eur_to_zar()
    test_events_by_user_grouping()
    test_profile_helpers()
    print("\nAll Phase 1 tests PASSED")
