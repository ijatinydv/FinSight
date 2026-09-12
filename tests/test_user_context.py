"""
tests/test_user_context.py

Verifies that:
1. All 275 UserContext objects build without errors
2. Events are currency-normalised to the user's home_currency
3. UserContext property helpers (settled_events, confirmed_credits, etc.) work correctly
4. Payment options are correctly associated per user

Run: python -m pytest tests/test_user_context.py -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.data_loader import DataLoader
from code.exchange_normalizer import ExchangeNormalizer
from code.user_context import build_user_contexts

DATASET = Path(__file__).resolve().parents[1] / "dataset"


def _build():
    loader = DataLoader(DATASET)
    norm = ExchangeNormalizer(str(DATASET / "exchange_rates.csv"))
    profiles     = loader.get_profiles()
    events_by_u  = loader.get_events_by_user()
    msgs_by_u    = loader.get_messages_by_user()
    images_by_u  = {img.user_id: [] for img in loader.get_image_records()}
    for img in loader.get_image_records():
        images_by_u.setdefault(img.user_id, []).append(img)
    opts_by_req  = loader.get_payment_options_by_request()
    requests     = loader.get_requests()
    return build_user_contexts(profiles, events_by_u, msgs_by_u, images_by_u, opts_by_req, requests, norm)


def test_all_contexts_build():
    ctxs = _build()
    assert len(ctxs) == 275, f"Expected 275 contexts, got {len(ctxs)}"
    print(f"  Built {len(ctxs)} UserContext objects")


def test_events_normalised_to_home_currency():
    ctxs = _build()
    for uid, ctx in ctxs.items():
        for ev in ctx.events:
            assert ev.currency == ctx.profile.home_currency, (
                f"User {uid}: event {ev.event_id} still in {ev.currency}, "
                f"expected {ctx.profile.home_currency}"
            )
    print("  All events normalised to home_currency")


def test_property_helpers():
    ctxs = _build()
    ctx = list(ctxs.values())[0]
    # Just check they return lists without crashing
    assert isinstance(ctx.settled_events, list)
    assert isinstance(ctx.pending_events, list)
    assert isinstance(ctx.scheduled_events, list)
    assert isinstance(ctx.active_debits, list)
    assert isinstance(ctx.confirmed_credits, list)
    print("  Property helpers work")


def test_payment_options_associated():
    ctxs = _build()
    # At least some users should have payment options
    users_with_options = [uid for uid, ctx in ctxs.items() if ctx.payment_options]
    assert len(users_with_options) > 0, "No users have payment options"
    print(f"  {len(users_with_options)} users have payment options")


if __name__ == "__main__":
    test_all_contexts_build()
    test_events_normalised_to_home_currency()
    test_property_helpers()
    test_payment_options_associated()
    print("\nAll UserContext tests PASSED")
