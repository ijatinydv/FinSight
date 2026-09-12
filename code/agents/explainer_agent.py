"""
agents/explainer_agent.py — Explainer Agent

Generates the final decision_explanation string in the style of sample_requests.csv:
concise (1-2 sentences), grounded in actual numbers, no fluff.
"""
from __future__ import annotations

import logging
from typing import Optional

from openai import OpenAI

from ..config import cfg
from ..schemas import PurchaseRequest
from ..user_context import UserContext
from .strategy_agent import AgentDecisionSchema

logger = logging.getLogger(__name__)

client = OpenAI(api_key=cfg.novita_api_key, base_url=cfg.novita_base_url)
MODEL = cfg.model_name

SYSTEM = """\
You are a concise financial advisor. Write exactly 1-2 sentences explaining a financial decision.
Rules:
- Use the actual currency amounts and dates from the context
- Be specific: state WHY (e.g. "balance would fall below minimum", "salary arrives on X")
- Do NOT use placeholders. Do NOT say "the user". Use direct language.
- Match the style of these examples:
  "Pay ZAR 25,256 in full on 3 March 2024. Balance after payment stays above the ZAR 18,000 minimum."
  "Pay IDR 5,491,000 in full on 15 November 2019. Paying earlier would take the balance below the IDR 2,668,700 minimum."
  "This amount is not affordable within the planning horizon; paying any amount now or later would breach the minimum balance."
Output ONLY the explanation text. No JSON, no markdown.
"""


class ExplainerAgent:
    def explain(
        self,
        request: PurchaseRequest,
        ctx: UserContext,
        decision: AgentDecisionSchema,
    ) -> str:
        """Generate a grounded 1-2 sentence explanation."""
        # If strategy already has a decent explanation (not a fallback note), use it
        existing = decision.decision_explanation.strip()
        if existing and "fallback" not in existing.lower() and len(existing) > 20:
            # Still run through LLM to improve style if needed
            pass

        profile = ctx.profile
        plan_str = (
            " | ".join(f"{p.date}:{p.amount:.2f}" for p in decision.payment_plan)
            if decision.payment_plan else "none"
        )

        user_msg = (
            f"Request: {request.request_type} of {request.requested_amount:.2f} {profile.home_currency} "
            f"on {request.request_date} (deadline {request.desired_completion_date or 'unspecified'})\n"
            f"Decision: status={decision.affordability_status}, method={decision.recommended_payment_method}\n"
            f"amount_safe_to_pay={decision.amount_safe_to_pay:.2f}\n"
            f"earliest_full={decision.earliest_date_for_full_payment or 'N/A'}\n"
            f"payment_plan={plan_str}\n"
            f"spending_changes={decision.spending_changes_needed}\n"
            f"min_balance={profile.minimum_balance_to_keep:.2f} {profile.home_currency}\n"
            f"current_balance={profile.current_available_balance:.2f}\n"
            f"Internal reasoning: {decision.internal_reasoning[:300]}"
        )

        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=200,
            )
            explanation = resp.choices[0].message.content or existing
            return explanation.strip()
        except Exception as e:
            logger.warning("ExplainerAgent failed: %s", e)
            return existing or f"Decision: {decision.affordability_status} ({decision.recommended_payment_method})"
