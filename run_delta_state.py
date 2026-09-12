"""
run_delta_state.py — One-time startup script to:
1. Extract patches from all 215 messages via GLM-5.3-Flash
2. Extract amounts from all 16 images via GLM-5.3-Flash VLM
3. Cache results to code/.cache/delta_patches.json
4. Print a summary report

Run: python run_delta_state.py
Re-runs are safe — cached results are reused.
"""
import json
import sys
import base64
import os
from pathlib import Path

# Make `code` importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from code.data_loader import DataLoader
from code.config import cfg

# Setup
DATASET = Path("dataset")
CACHE_DIR = Path("code/.cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MSG_CACHE  = CACHE_DIR / "message_patches.json"
IMG_CACHE  = CACHE_DIR / "image_amounts.json"

client = OpenAI(
    api_key=cfg.novita_api_key,
    base_url=cfg.novita_base_url,
)
MODEL = cfg.model_name  # zai-org/glm-5.3-flash

loader = DataLoader(DATASET)
messages = loader.get_messages()
images   = loader.get_image_records()
events   = loader.get_events()
event_map = {ev.event_id: ev for ev in events}

# ─────────────────────────────────────────────────────────
# 1. MESSAGE PATCH EXTRACTION
# ─────────────────────────────────────────────────────────
if MSG_CACHE.exists():
    print(f"[MSG] Cache found — loading from {MSG_CACHE}")
    with open(MSG_CACHE) as f:
        msg_patches = json.load(f)
    print(f"[MSG] Loaded {len(msg_patches)} cached message patches")
else:
    print(f"[MSG] Extracting patches from {len(messages)} messages …")
    msg_patches = {}

    SYSTEM_PROMPT = (
        "You are a financial data extraction assistant. "
        "Given a message from an employer, bank, or service provider, extract any amendments to the user's financial situation. "
        "Return ONLY valid JSON with this schema:\n"
        '{"patch_type": "salary_override"|"cancel_event"|"date_change"|"pending_refund"|"informational", '
        '"event_id": "<event_id or null>", '
        '"new_amount": <number or null>, '
        '"new_date": "<YYYY-MM-DD or null>", '
        '"currency": "<ISO code or null>", '
        '"notes": "<brief explanation>"}'
    )

    for msg in messages:
        related_ev = event_map.get(msg.related_event_id or "")
        context = ""
        if related_ev:
            context = (
                f"Linked event: {related_ev.event_id} | {related_ev.description} | "
                f"amount={related_ev.amount} {related_ev.currency} | status={related_ev.status}"
            )

        user_prompt = (
            f"Message ID: {msg.message_id}\n"
            f"Source: {msg.source_type}\n"
            f"Sent at: {msg.sent_at}\n"
            f"User: {msg.user_id}\n"
            + (f"Context: {context}\n" if context else "") +
            f"\nMessage:\n{msg.message_text}"
        )

        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or "{}"
            patch = json.loads(raw)
            patch["source_message_id"] = msg.message_id
            patch["user_id"] = msg.user_id
            patch["request_id"] = msg.request_id
            msg_patches[msg.message_id] = patch
            ptype = patch.get("patch_type", "?")
            print(f"  [{msg.message_id}] {ptype} | amount={patch.get('new_amount')} | event={patch.get('event_id')}")
        except Exception as e:
            print(f"  [{msg.message_id}] ERROR: {e}")
            msg_patches[msg.message_id] = {"patch_type": "informational", "error": str(e)}

    with open(MSG_CACHE, "w", encoding="utf-8") as f:
        json.dump(msg_patches, f, indent=2, ensure_ascii=False)
    print(f"[MSG] Saved {len(msg_patches)} patches to {MSG_CACHE}")

# ─────────────────────────────────────────────────────────
# 2. IMAGE AMOUNT EXTRACTION (VLM)
# ─────────────────────────────────────────────────────────
if IMG_CACHE.exists():
    print(f"\n[IMG] Cache found — loading from {IMG_CACHE}")
    with open(IMG_CACHE) as f:
        img_amounts = json.load(f)
    print(f"[IMG] Loaded {len(img_amounts)} cached image results")
else:
    print(f"\n[IMG] Extracting amounts from {len(images)} images via VLM …")
    img_amounts = {}

    IMG_SYSTEM = (
        "You are a financial document OCR assistant. "
        "Given an image of a payslip, receipt, invoice, or bank document, "
        "extract the primary monetary amount and currency. "
        "Return ONLY valid JSON:\n"
        '{"amount": <number or null>, "currency": "<ISO code or null>", '
        '"document_type": "<payslip|receipt|invoice|other>", '
        '"merchant_or_payer": "<name or null>", '
        '"date": "<YYYY-MM-DD or null>", '
        '"notes": "<brief explanation>"}'
    )

    for img in images:
        img_path = Path(img.image_path) if img.image_path else None
        if not img_path or not img_path.exists():
            print(f"  [{img.image_id}] MISSING file: {img_path}")
            img_amounts[img.image_id] = {"amount": None, "currency": None, "error": "file missing"}
            continue

        related_ev = event_map.get(img.related_event_id or "")
        context_text = ""
        if related_ev:
            context_text = (
                f"This image is linked to event {related_ev.event_id}: "
                f"{related_ev.description}, recorded amount={related_ev.amount} {related_ev.currency}."
            )

        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": IMG_SYSTEM},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"Image ID: {img.image_id} | User: {img.user_id}\n"
                                    + (f"Context: {context_text}\n" if context_text else "") +
                                    "Extract the primary monetary amount visible in this document."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            },
                        ],
                    },
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or "{}"
            result = json.loads(raw)
            result["image_id"] = img.image_id
            result["user_id"]  = img.user_id
            result["event_id"] = img.related_event_id
            img_amounts[img.image_id] = result
            print(f"  [{img.image_id}] amount={result.get('amount')} {result.get('currency')} | {result.get('document_type')} | {result.get('notes','')[:80]}")
        except Exception as e:
            print(f"  [{img.image_id}] ERROR: {e}")
            img_amounts[img.image_id] = {"amount": None, "currency": None, "error": str(e)}

    with open(IMG_CACHE, "w", encoding="utf-8") as f:
        json.dump(img_amounts, f, indent=2, ensure_ascii=False)
    print(f"[IMG] Saved {len(img_amounts)} results to {IMG_CACHE}")

# ─────────────────────────────────────────────────────────
# 3. SUMMARY
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("DELTA STATE EXTRACTION COMPLETE")
print("="*60)

actionable_msgs = [v for v in msg_patches.values() if v.get("patch_type") not in ("informational", None)]
salary_overrides = [v for v in msg_patches.values() if v.get("patch_type") == "salary_override"]
cancels = [v for v in msg_patches.values() if v.get("patch_type") == "cancel_event"]
date_changes = [v for v in msg_patches.values() if v.get("patch_type") == "date_change"]
pending_refunds = [v for v in msg_patches.values() if v.get("patch_type") == "pending_refund"]

print(f"Messages processed : {len(msg_patches)}")
print(f"  salary_override  : {len(salary_overrides)}")
print(f"  cancel_event     : {len(cancels)}")
print(f"  date_change      : {len(date_changes)}")
print(f"  pending_refund   : {len(pending_refunds)}")
print(f"  informational    : {len(msg_patches) - len(actionable_msgs)}")

imgs_with_amount = [v for v in img_amounts.values() if v.get("amount") is not None]
print(f"\nImages processed   : {len(img_amounts)}")
print(f"  amounts extracted: {len(imgs_with_amount)}")

print(f"\nCaches written to  : {CACHE_DIR}/")
print("  message_patches.json")
print("  image_amounts.json")
print("="*60)
