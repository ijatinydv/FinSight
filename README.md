# FinSight — Buy or Wait? 🤖💸

**HackerRank Orchestrate September 2026 Hackathon Submission**

An AI-powered financial decision agent that determines whether a user should pay in full, pay partially, use installments, wait, or not proceed with a purchase — by reconstructing their complete financial position from structured profiles, historical events, dated exchange rates, and unstructured messages and images.

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- A [Novita AI](https://novita.ai) API key (or any OpenAI-compatible endpoint)

### 2. Install Dependencies

`````bash
pip install -r requirements.txt
`````

### 3. Set Up Environment Variables

Create a .env file in the project root:

`````env
NOVITA_API_KEY=your_api_key_here
NOVITA_BASE_URL=https://api.novita.ai/openai
MODEL_NAME=zai-org/glm-5.3-flash
`````

### 4. Run the Agent

**Full evaluation (generates dataset/output.csv):**
`````bash
python -m code.main
`````

**Sample only (25 requests with known answers for testing):**
`````bash
python -m code.main --sample
`````

**Single request (for debugging):**
`````bash
python -m code.main --request request_42
`````

---

## Project Structure

`````
hackerrank-orchestrate-september26/
├── code/
│   ├── main.py                  # Entry point — async orchestrator
│   ├── config.py                # Pydantic settings (reads .env)
│   ├── schemas.py               # Pydantic data models
│   ├── data_loader.py           # CSV ingestion for all dataset files
│   ├── exchange_normalizer.py   # Fixed-rate foreign currency conversion
│   ├── patch_applier.py         # Applies salary overrides, cancels, image patches
│   ├── user_context.py          # Builds per-user financial state
│   ├── finance_engine.py        # Core deterministic engine
│   ├── spending_optimizer.py    # Greedy spending-change optimizer
│   ├── delta_state.py           # Balance delta computation helpers
│   └── agents/
│       ├── reflexion_loop.py    # Main Actor-Critic orchestration loop
│       ├── strategy_agent.py    # LLM Strategy Agent (Actor)
│       ├── critic_agent.py      # LLM Critic / Risk Validator
│       └── explainer_agent.py   # LLM Explanation Generator
├── dataset/
│   ├── requests.csv             # Evaluation requests (input)
│   ├── output.csv               # Final predictions (generated output)
│   └── ...                      # All other dataset files
├── evaluation/
│   └── usage_report.md          # Token usage and cost report
├── .env                         # API keys (not committed)
├── .gitignore
└── README.md
`````

---

## Architecture & Approach

### Core Philosophy: Deterministic-First, LLM-Second

All safety-critical calculations are **mathematically computed**, not hallucinated. The LLM synthesizes computed facts into the right decision and writes natural language explanations.

`
Dataset Files
     |
     v
DataLoader + PatchApplier
  - Load all CSVs
  - Apply salary overrides from messages (83 events)
  - Apply event cancellations from messages (7 events)
  - Apply amount corrections from receipt images (16 events)
     |
     v
ExchangeNormalizer
  - Convert all amounts to home currency using fixed dated rates
     |
     v
Finance Engine (Deterministic Core)
  |- RecurrenceDetector
  |    - Scans history for recurring patterns (>=3 settled, period sigma < 5 days)
  |    - Trimmed mean of last 6 instances for conservative projection
  |
  |- DailyBalanceSimulator
  |    - Projects 90-day rolling balance from request_date
  |    - Reserves all pending debits, ignores pending credits
  |
  |- Binary Search -> amount_safe_to_pay
  |    - 40-iteration search, checks ALL 90 days
  |    - Final exact-amount boundary check prevents rounding errors
  |
  |- Linear Scan -> earliest_date_for_full_payment
       - Scans forward until full amount is safe
     |
     v
SpendingOptimizer
  - Greedy search over flexible, non-protected events
  - Handles 'stoppable', 'reducible', 'reducible_or_stoppable' flexibility types
  - If spending changes unlock payment: injects deterministic override
     |
     v
ReflexionLoop (LLM Actor-Critic, max 3 iterations)
  |- Strategy Agent (GLM-5.3-Flash)
  |    - Receives computed affordability + payment options + hints + message context
  |    - Proposes complete decision JSON with few-shot GT examples for format matching
  |
  |- Critic Agent
  |    - Validates proposal against all spec rules, feeds back if invalid
  |
  |- Deterministic Post-Processing
  |    - Forces correct earliest_date rules (not_affordable -> "")
  |    - Injects spending changes if LLM missed optimizer hint
  |
  |- Explainer Agent
       - Writes final 1-2 sentence decision_explanation in GT style
`

---

## Key Design Decisions

### 1. Multi-Modal Patch System
Images and messages are parsed before simulation. Receipt images correct event amounts. Payslip messages override salary amounts. Cancellation messages mark events cancelled. All patches are applied before any simulation runs.

### 2. Conservative 90-Day Safety Horizon
The balance must never breach minimum_balance_to_keep on any of the 90 projected days — not just the payment day. Our binary search checks all 90 days per the problem spec.

### 3. Trimmed-Mean Recurrence Projection
Variable expenses (groceries, dining, transport) use a trimmed mean of the last 6 occurrences (dropping top and bottom 10%). This prevents one-off spikes from inflating the recurring estimate.

### 4. Exact-Amount Boundary Check
After 40 binary search iterations, the engine checks whether the exact requested_amount is itself safe, preventing floating-point asymptote rounding (e.g., 620.39 instead of 620.40).

### 5. Installment Plan Determinism
Dates are computed from first_payment_date + payment_frequency_days. Last installment absorbs rounding remainder so total exactly matches total_payable_amount.

### 6. Decision Priority (per spec §6.3)
1. full_payment (affordable_now)
2. installments within deadline (lowest total cost first)
3. partial_payment (only when cheaper — no financing fee)
4. wait (affordable_later)
5. not_recommended

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| LLM Provider | Novita AI (OpenAI-compatible) |
| LLM Model | zai-org/glm-5.3-flash |
| Data Validation | Pydantic v2 |
| Async Concurrency | asyncio + Semaphore(10) |
| Configuration | pydantic-settings + .env |

---

## Token Usage & Cost

See evaluation/usage_report.md for full breakdown.

**Summary (250 evaluation requests):**
- Input tokens: ~690,000
- Output tokens: ~55,000
- Estimated total cost: ~.04
- Average cost per request: ~.00016

---

## Sample Score (25 public examples)

| Metric | Score |
|---|---|
| affordability_status exact match | 20 / 25 |
| recommended_payment_method exact match | 21 / 25 |
| spending_changes_needed exact match | 22 / 25 |
| BOTH status & method correct | 19 / 25 (76%) |
