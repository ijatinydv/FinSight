"""
code/main.py — FinSight Entry Point

Async orchestrator: processes all 251 requests individually using
asyncio + Semaphore(10). Writes dataset/output.csv and agent trace to log.txt.

Usage:
    python -m code.main
    python -m code.main --sample    # run only sample_requests.csv (25 rows)
    python -m code.main --request request_01  # run single request
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from .config import cfg
from .data_loader import DataLoader
from .exchange_normalizer import ExchangeNormalizer
from .user_context import build_user_contexts
from .patch_applier import apply_all_patches
from .schemas import OutputRow, PurchaseRequest
from .agents.reflexion_loop import ReflexionLoop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("finsight.main")

DATASET = cfg.dataset_dir
OUTPUT_CSV = DATASET / "output.csv"
LOG_FILE = Path(__file__).resolve().parents[1] / "log.txt"

OUTPUT_COLUMNS = [
    "request_id", "amount_safe_to_pay", "affordability_status",
    "recommended_payment_method", "payment_plan",
    "earliest_date_for_full_payment", "spending_changes_needed",
    "decision_explanation",
]


async def process_single(
    request: PurchaseRequest,
    contexts: dict,
    msg_patches: dict,
    loop: ReflexionLoop,
    sem: asyncio.Semaphore,
) -> tuple:
    async with sem:
        ctx = contexts.get(request.user_id)
        if not ctx:
            logger.error("No context for user %s (request %s)", request.user_id, request.request_id)
            return None, ""
        # Run in thread pool to avoid blocking event loop (OpenAI calls are sync)
        row, trace = await asyncio.get_event_loop().run_in_executor(
            None, loop.process_request, request, ctx, msg_patches
        )
        logger.info("[%s] done — %s / %s", request.request_id, row.affordability_status, row.recommended_payment_method)
        return row, trace


async def run(requests_to_process: list, contexts: dict, msg_patches: dict) -> list:
    sem = asyncio.Semaphore(10)
    loop = ReflexionLoop()
    tasks = [
        process_single(req, contexts, msg_patches, loop, sem)
        for req in requests_to_process
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="FinSight — Buy or Wait?")
    parser.add_argument("--sample", action="store_true", help="Run only 25 sample requests")
    parser.add_argument("--request", type=str, default=None, help="Run a single request_id")
    args = parser.parse_args()

    logger.info("=== FinSight starting ===")
    logger.info("Model: %s | Base: %s", cfg.model_name, cfg.novita_base_url)

    # ── Load data ─────────────────────────────────────────────
    loader   = DataLoader(DATASET)
    norm     = ExchangeNormalizer(str(DATASET / "exchange_rates.csv"))
    imgs_by_u = {}
    for img in loader.get_image_records():
        imgs_by_u.setdefault(img.user_id, []).append(img)

    all_eval_requests = loader.get_requests()
    all_sample_requests = loader.get_sample_requests()
    all_known_requests = all_eval_requests + all_sample_requests  # both sets for payment option mapping

    contexts = build_user_contexts(
        loader.get_profiles(), loader.get_events_by_user(),
        loader.get_messages_by_user(), imgs_by_u,
        loader.get_payment_options_by_request(), all_known_requests, norm
    )
    contexts = apply_all_patches(contexts)
    logger.info("Loaded %d user contexts (with patches)", len(contexts))

    # ── Load message patches for context building ─────────────
    msg_cache = Path(__file__).resolve().parent / ".cache" / "message_patches.json"
    msg_patches = {}
    if msg_cache.exists():
        with open(msg_cache, encoding="utf-8") as f:
            msg_patches = json.load(f)
    logger.info("Loaded %d message patches", len(msg_patches))

    # ── Select requests ───────────────────────────────────────
    if args.sample:
        # sample_requests.csv has its own rows (request_01..request_25),
        # NOT present in requests.csv — load them directly.
        requests_to_process = all_sample_requests
        logger.info("Running %d sample requests", len(requests_to_process))
    elif args.request:
        requests_to_process = [r for r in all_known_requests if r.request_id == args.request]
        if not requests_to_process:
            logger.error("request_id %s not found in eval or sample requests", args.request)
            sys.exit(1)
    else:
        requests_to_process = all_eval_requests
        logger.info("Running all %d evaluation requests", len(requests_to_process))

    # ── Run async ─────────────────────────────────────────────
    results = asyncio.run(run(requests_to_process, contexts, msg_patches))

    # ── Write output ──────────────────────────────────────────
    rows: list[OutputRow] = []
    traces: list[str] = []
    errors = 0

    for res in results:
        if isinstance(res, Exception):
            logger.error("Request failed: %s", res)
            errors += 1
            continue
        row, trace = res
        if row:
            rows.append(row)
            traces.append(trace)

    # Sort by request_id to match evaluation order
    rows.sort(key=lambda r: r.request_id)

    output_path = OUTPUT_CSV if not args.sample else DATASET / "output_sample.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "request_id": row.request_id,
                "amount_safe_to_pay": row.amount_safe_to_pay,
                "affordability_status": row.affordability_status,
                "recommended_payment_method": row.recommended_payment_method,
                "payment_plan": row.payment_plan,
                "earliest_date_for_full_payment": row.earliest_date_for_full_payment,
                "spending_changes_needed": row.spending_changes_needed,
                "decision_explanation": row.decision_explanation,
            })
    logger.info("Written %d rows to %s", len(rows), output_path)

    # Write traces to log.txt
    ts = datetime.utcnow().isoformat() + "Z"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n\n## {ts} AGENT EXECUTION TRACE — {len(rows)} requests\n")
        for trace in traces:
            f.write(trace)

    if errors:
        logger.warning("%d requests failed", errors)
    logger.info("=== FinSight done: %d/%d succeeded ===", len(rows), len(requests_to_process))
    return rows


if __name__ == "__main__":
    main()
