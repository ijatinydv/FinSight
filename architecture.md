# Buy or Wait? — System Architecture

> Designed for maximum hackathon score: deterministic accuracy + LLM personalization

---

## 1. High-Level Architecture

`
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                                   │
│  requests.csv  financial_profiles.csv  financial_events.csv         │
│  exchange_rates.csv  request_payment_options.csv                    │
│  messages.csv  images.csv  media/images/*.png                       │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     DATA INGESTION LAYER                             │
│                                                                      │
│  DataLoader            ImageExtractor (VLM)    MessageParser        │
│  - Load CSVs           - 16 PNGs               - Salary amendments  │
│  - Normalize currencies- Extract amounts        - Cancellations      │
│  - Exchange rates      - Cache results          - Event amendments   │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│          FINANCIAL STATE RECONSTRUCTION ENGINE                       │
│                                                                      │
│  RecurrenceDetector → CashflowSimulator → AffordabilityCalculator   │
│  - Groups settled      - 90-day balance     - amount_safe_to_pay    │
│    events by cadence     simulation          - earliest_date         │
│  - Projects forward    - Excludes pending   - min_balance enforced  │
│                          credits/cancelled                           │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PLAN SELECTION ENGINE                              │
│                                                                      │
│  EligibilityFilter → PlanEvaluator → SpendingChangeOptimizer        │
│  - Checks user prefs   - Tries all plan types  - Up to 3 changes    │
│  - Installment months  - Verifies each payment - Flexible only      │
│                          keeps balance safe                          │
│                                                                      │
│                      PlanRanker (problem spec priority order)        │
│                      1. Complete by deadline                         │
│                      2. No spending changes                          │
│                      3. Min total cost                               │
│                      4. Earliest start                               │
│                      5. Fewest payments                              │
│                      6. Lowest option_id                             │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                LLM EXPLANATION LAYER (Gemini 1.5 Pro)                │
│  - Batched 10 requests per call                                      │
│  - Generates decision_explanation matching sample style              │
│  - Temperature=0 for determinism                                     │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 VALIDATION & OUTPUT LAYER                             │
│  Validator → OutputWriter                                            │
│  - Constraint checks    - dataset/output.csv                        │
│  - Payment math         - evaluation/usage_report.md                │
└─────────────────────────────────────────────────────────────────────┘
`

## 2. File Structure

`
code/
├── main.py              # Orchestrator
├── data_loader.py       # CSV loading + currency normalization
├── image_extractor.py   # Gemini VLM for 16 PNG images
├── message_parser.py    # Parse salary changes, cancellations
├── finance_engine.py    # 90-day forecast + affordability math
├── plan_selector.py     # Plan ranking + spending changes
├── explainer.py         # LLM explanation generation (batched)
├── validator.py         # Output constraint verification
├── config.py            # Settings, API keys from env
├── evaluation/
│   ├── main.py          # Eval harness
│   └── usage_report.md  # Auto-generated token/cost report
├── requirements.txt
└── README.md
`

## 3. Key Design Decisions

### Deterministic Math + LLM for Language Only
- All financial calculations are deterministic Python
- LLM used only for: image extraction, message classification, explanation text
- This prevents hallucinated numbers while getting quality explanations

### 90-Day Forward Simulation
- Simulate every single day (not just key dates)
- Catches edge cases: mid-month salary enabling a payment, day-after-salary dip

### Conservative Recurrence Detection  
- Only project recurring expense if it appears >= 3x with consistent periodicity
- Never invent income

### Conflict Resolution Priority
1. Explicit cancellation/settlement/amendment
2. Newer record from same source
3. Settled over estimate/forecast
4. Financially safer interpretation

## 4. Currency Conversion
- Use rate_date = closest month-end in exchange_rates.csv on or before settlement_date
- Chain via USD/EUR if direct rate not available

## 5. Spending Changes Strategy
- Only if no safe plan without changes
- Max 3 changes (stop or reduce, not both on same event)
- Only flexible, non-protected events
- Only categories user is willing to change

## 6. Token Budget (~130K total)
- Image extraction (4 batch calls): ~8,800 tokens
- Message parsing (22 batch calls): ~77,000 tokens  
- Explanation generation (26 batch calls): ~44,200 tokens
- Estimated cost: ~.50-1.00
