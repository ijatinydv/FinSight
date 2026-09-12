"""Quick validation of the finance engine against the 25 sample requests."""
import sys
from pathlib import Path
from datetime import date
sys.path.insert(0, str(Path(__file__).resolve().parent))

from code.data_loader import DataLoader
from code.exchange_normalizer import ExchangeNormalizer
from code.user_context import build_user_contexts
from code.finance_engine import RecurrenceDetector, DailyBalanceSimulator, AffordabilityCalculator

DATASET = Path("dataset")
loader   = DataLoader(DATASET)
norm     = ExchangeNormalizer(str(DATASET / "exchange_rates.csv"))

profiles    = loader.get_profiles()
events_by_u = loader.get_events_by_user()
msgs_by_u   = loader.get_messages_by_user()
imgs_by_u: dict = {}
for img in loader.get_image_records():
    imgs_by_u.setdefault(img.user_id, []).append(img)
opts_by_req = loader.get_payment_options_by_request()
requests    = loader.get_requests()

ctxs = build_user_contexts(profiles, events_by_u, msgs_by_u, imgs_by_u, opts_by_req, requests, norm)

# Test against sample_01: user_01, 2024-03-03, ZAR 25256, expected amount_safe=25256, affordable_now
sample_reqs = loader._requests_df  # use raw df to get all columns
import pandas as pd
samples = pd.read_csv(DATASET / "sample_requests.csv")

detector  = RecurrenceDetector()
simulator = DailyBalanceSimulator()
calc      = AffordabilityCalculator(simulator, detector)

print("Running affordability calc on 5 sample requests...")
print(f"{'req_id':<12} {'user':<10} {'requested':>12} {'safe_to_pay':>14} {'earliest_full':>15} {'expected_safe':>14} {'match':>6}")
print("-"*85)

for _, row in samples.head(5).iterrows():
    req_id  = row["request_id"]
    user_id = row["user_id"]
    amt     = float(row["requested_amount"])
    req_dt  = date.fromisoformat(row["request_date"])
    exp_safe = float(row["amount_safe_to_pay"])

    ctx = ctxs.get(user_id)
    if not ctx:
        print(f"{req_id:<12} NO CTX")
        continue

    result = calc.compute(ctx, req_dt, amt)
    match = abs(result.amount_safe_to_pay - exp_safe) < 1.0

    earliest_str = str(result.earliest_date_for_full_payment) if result.earliest_date_for_full_payment else "none"
    print(f"{req_id:<12} {user_id:<10} {amt:>12.2f} {result.amount_safe_to_pay:>14.2f} {earliest_str:>15} {exp_safe:>14.2f} {'OK' if match else 'DIFF':>6}")
