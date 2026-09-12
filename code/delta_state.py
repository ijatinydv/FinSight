"""
delta_state.py – Unified Delta-State Patch Engine

Before the financial simulation runs, this engine:
1. Reads every message and extracts structured patches (salary updates,
   payment date changes, cancellations, confirmations, etc.) via LLM.
2. Reads every image via VLM to extract amounts embedded in payslips,
   receipts, or invoices that are referenced by events.
3. Applies patches to the mutable event list in-place, so the simulation
   always works with a coherent, up-to-date snapshot.

Messages and images are UNTRUSTED evidence – their embedded instructions
never override the challenge rules, but they can amend financial facts.
"""
from __future__ import annotations
import base64
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .schemas import FinancialEvent, ImageRecord, Message

logger = logging.getLogger(__name__)


@dataclass
class Patch:
    """A single mutation to apply to a FinancialEvent (or add a new event)."""
    patch_type: str            # "amend_amount" | "amend_date" | "cancel" | "add_event"
    event_id: Optional[str]    # target event_id (None for add_event)
    new_amount: Optional[float] = None
    new_date: Optional[str] = None
    source: str = ""            # "message" | "image"
    source_id: str = ""
    confidence: float = 1.0


class DeltaStateEngine:
    """
    Apply message and image evidence patches to the event list.
    LLM/VLM calls are lazy and cached to disk.
    """

    CACHE_FILE = ".cache/delta_state_patches.json"

    def __init__(
        self,
        events: List[FinancialEvent],
        messages: List[Message],
        images: List[ImageRecord],
        llm_client,                   # openai-compatible client
        model: str,
        vision_model: str,
        repo_root: Path,
    ):
        self.events = events
        self.messages = messages
        self.images = images
        self.llm = llm_client
        self.model = model
        self.vision_model = vision_model
        self.cache_path = repo_root / self.CACHE_FILE
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._patches: List[Patch] = []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> List[FinancialEvent]:
        """Extract patches from messages + images and return patched events."""
        cached = self._load_cache()
        if cached is not None:
            logger.info("Delta-state: loaded %d patches from cache.", len(cached))
            self._patches = cached
        else:
            logger.info("Delta-state: extracting patches from %d messages, %d images …",
                        len(self.messages), len(self.images))
            self._patches = []
            self._extract_message_patches()
            self._extract_image_patches()
            self._save_cache(self._patches)
            logger.info("Delta-state: %d patches extracted and cached.", len(self._patches))

        return self._apply_patches()

    # ------------------------------------------------------------------
    # Message extraction
    # ------------------------------------------------------------------

    def _extract_message_patches(self) -> None:
        """Ask the LLM to interpret each message and emit structured patches."""
        event_index = {ev.event_id: ev for ev in self.events}

        for msg in self.messages:
            related_ev = event_index.get(msg.related_event_id or "")
            context_snippet = ""
            if related_ev:
                context_snippet = (
                    f"The message is linked to event {related_ev.event_id}: "
                    f"{related_ev.description}, amount {related_ev.amount} {related_ev.currency}, "
                    f"status {related_ev.status}, date {related_ev.event_date}."
                )

            system = (
                "You are a financial data extraction assistant. "
                "Given a message from an employer, service provider, bank, or other source, "
                "extract any amendments to the user's financial events as JSON patches. "
                "Return ONLY valid JSON. If nothing actionable is found, return {\"patches\": []}."
            )
            user_prompt = f"""Message ID: {msg.message_id}
Source type: {msg.source_type}
Sent at: {msg.sent_at}
User: {msg.user_id}
{context_snippet}

Message text:
{msg.message_text}

Extract any of the following if clearly stated:
- salary/income changes (patch_type: amend_amount, event_id: related event or null if new)
- date changes (patch_type: amend_date)
- cancellations (patch_type: cancel)
- confirmations that change a pending status to settled

Return JSON in this schema:
{{
  "patches": [
    {{
      "patch_type": "amend_amount"|"amend_date"|"cancel"|"add_event",
      "event_id": "<event_id or null>",
      "new_amount": <number or null>,
      "new_date": "<YYYY-MM-DD or null>",
      "confidence": <0.0-1.0>
    }}
  ]
}}
"""
            try:
                resp = self.llm.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                raw = resp.choices[0].message.content or "{}"
                data = json.loads(raw)
                for p in data.get("patches", []):
                    self._patches.append(Patch(
                        patch_type=p.get("patch_type", ""),
                        event_id=p.get("event_id"),
                        new_amount=p.get("new_amount"),
                        new_date=p.get("new_date"),
                        source="message",
                        source_id=msg.message_id,
                        confidence=float(p.get("confidence", 1.0)),
                    ))
            except Exception as exc:
                logger.warning("Delta-state: failed to process message %s: %s", msg.message_id, exc)

    # ------------------------------------------------------------------
    # Image extraction
    # ------------------------------------------------------------------

    def _extract_image_patches(self) -> None:
        """Ask the VLM to extract amounts from each image (payslip, receipt, etc.)."""
        event_index = {ev.event_id: ev for ev in self.events}

        for img in self.images:
            if not img.image_path or not Path(img.image_path).exists():
                logger.warning("Delta-state: image file missing: %s", img.image_path)
                continue

            # Encode image to base64
            try:
                with open(img.image_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
            except Exception as exc:
                logger.warning("Delta-state: cannot read image %s: %s", img.image_id, exc)
                continue

            related_ev = event_index.get(img.related_event_id or "")
            context = ""
            if related_ev:
                context = (
                    f"This image is linked to event {related_ev.event_id}: "
                    f"{related_ev.description}, recorded amount {related_ev.amount} {related_ev.currency}."
                )

            system = (
                "You are a financial document OCR assistant. "
                "Given an image (payslip, receipt, bank statement, invoice), "
                "extract any monetary amounts that should amend existing financial records. "
                "Return ONLY valid JSON."
            )
            user_messages = [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Image ID: {img.image_id}\nUser: {img.user_id}\n{context}\n\n"
                                    "Extract any monetary amounts, dates, or status changes visible in this document. "
                                    "Return JSON:\n"
                                    '{"patches": [{"patch_type": "amend_amount"|"amend_date"|"cancel", '
                                    '"event_id": "<id or null>", "new_amount": <number or null>, '
                                    '"new_date": "<YYYY-MM-DD or null>", "confidence": <0.0-1.0>}]}',
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                },
            ]
            try:
                resp = self.llm.chat.completions.create(
                    model=self.vision_model,
                    messages=user_messages,
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                raw = resp.choices[0].message.content or "{}"
                data = json.loads(raw)
                for p in data.get("patches", []):
                    patch = Patch(
                        patch_type=p.get("patch_type", ""),
                        event_id=p.get("event_id"),
                        new_amount=p.get("new_amount"),
                        new_date=p.get("new_date"),
                        source="image",
                        source_id=img.image_id,
                        confidence=float(p.get("confidence", 1.0)),
                    )
                    self._patches.append(patch)
                    # Also update ImageRecord with extracted amount if relevant
                    if patch.new_amount is not None:
                        img.extracted_amount = patch.new_amount
            except Exception as exc:
                logger.warning("Delta-state: failed to process image %s: %s", img.image_id, exc)

    # ------------------------------------------------------------------
    # Apply patches
    # ------------------------------------------------------------------

    def _apply_patches(self) -> List[FinancialEvent]:
        """Apply all collected patches to the events list and return patched copy."""
        event_map: Dict[str, FinancialEvent] = {ev.event_id: ev for ev in self.events}
        patched_events = list(self.events)  # shallow copy; we mutate by rebuilding

        for patch in self._patches:
            if patch.confidence < 0.6:
                logger.debug("Delta-state: skipping low-confidence patch %s (%.2f)", patch.source_id, patch.confidence)
                continue

            ev = event_map.get(patch.event_id or "")
            if ev is None:
                if patch.patch_type == "add_event":
                    pass   # TODO: Phase 2 will add synthetic events
                continue

            if patch.patch_type == "amend_amount" and patch.new_amount is not None:
                logger.info("Delta-state: amend amount %s: %.2f -> %.2f (src=%s)",
                            ev.event_id, ev.amount, patch.new_amount, patch.source_id)
                # Rebuild the event with the updated amount
                updated = ev.model_copy(update={"amount": patch.new_amount})
                event_map[ev.event_id] = updated
                idx = next(i for i, e in enumerate(patched_events) if e.event_id == ev.event_id)
                patched_events[idx] = updated

            elif patch.patch_type == "amend_date" and patch.new_date is not None:
                logger.info("Delta-state: amend date %s: %s -> %s (src=%s)",
                            ev.event_id, ev.event_date, patch.new_date, patch.source_id)
                updated = ev.model_copy(update={"event_date": patch.new_date, "settlement_date": patch.new_date})
                event_map[ev.event_id] = updated
                idx = next(i for i, e in enumerate(patched_events) if e.event_id == ev.event_id)
                patched_events[idx] = updated

            elif patch.patch_type == "cancel":
                logger.info("Delta-state: cancel event %s (src=%s)", ev.event_id, patch.source_id)
                updated = ev.model_copy(update={"status": "cancelled"})
                event_map[ev.event_id] = updated
                idx = next(i for i, e in enumerate(patched_events) if e.event_id == ev.event_id)
                patched_events[idx] = updated

        return patched_events

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _load_cache(self) -> Optional[List[Patch]]:
        if not self.cache_path.exists():
            return None
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [Patch(**p) for p in data]
        except Exception:
            return None

    def _save_cache(self, patches: List[Patch]) -> None:
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump([p.__dict__ for p in patches], f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.warning("Delta-state: could not write cache: %s", exc)
