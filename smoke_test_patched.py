import sys, logging
from pathlib import Path
from datetime import date
sys.path.insert(0, str(Path(__file__).resolve().parent))
logging.basicConfig(level=logging.WARNING)

from code.data_loader import DataLoader
from code.exchange_normalizer import ExchangeNormalizer
from code.user_context import build_user_contexts
from code.patch_applier import apply_all_patches
from code.finance_engine import RecurrenceDetector, DailyBalanceSimulator, AffordabilityCalculator
import pandas as pd

DATASET = Path("dataset")
loader = DataLoader(DATASET)
norm = ExchangeNormalizer(str(DATASET / "exchange_rates.csv"))
imgs_by_u = {}
for img in loader.get_image_records():
    imgs_by_u.setdefault(img.user_id, []).append(img)

ctxs = build_user_contexts(
    loader.get_profiles(), loader.get_events_by_user(),
    loader.get_messages_by_user(), imgs_by_u,
    loader.get_payment_options_by_request(), loader.get_requests(), norm
)
ctxs = apply_all_patches(ctxs)

samples = pd.read_csv(DATASET / "sample_requests.csv")
det = RecurrenceDetector(); sim = DailyBalanceSimulator(); calc = AffordabilityCalculator(sim, det)

print(f"{'req_id':<12} {'user':<10} {'requested':>14} {'safe_got':>14} {'safe_exp':>14} {'match':>6} {'status_exp'}")
print("-"*100)
ok = 0
for _, row in samples.iterrows():
    ctx = ctxs.get(row["user_id"])
    if not ctx: continue
    req_dt = date.fromisoformat(row["request_date"])
    result = calc.compute(ctx, req_dt, float(row["requested_amount"]))
    exp = float(row["amount_safe_to_pay"])
    match = abs(result.amount_safe_to_pay - exp) < 1.0
    if match: ok += 1
    print(f"{row['request_id']:<12} {row['user_id']:<10} {float(row['requested_amount']):>14.2f} {result.amount_safe_to_pay:>14.2f} {exp:>14.2f} {'OK' if match else 'DIFF':>6} {row['affordability_status']}")
print(f"\nScore: {ok}/25 ({ok/25*100:.0f}%) WITH delta-state patches")
