# FinSight — Evaluation Usage Report

## Run Details

| Field | Value |
|---|---|
| Dataset | `dataset/requests.csv` — 251 requests across 275 user contexts |
| Sample Dataset | `dataset/sample_requests.csv` — 25 requests (public examples) |
| Run Timestamp | 2026-09-13 |
| Run Command | `python -m code.main` |

---

## Model Providers and Names

| Role | Provider | Model | Endpoint |
|---|---|---|---|
| Strategy / Decision Agent | Novita AI | `zai-org/glm-5.3-flash` | `https://api.novita.ai/openai` |
| Explainer Agent | Novita AI | `zai-org/glm-5.3-flash` | `https://api.novita.ai/openai` |
| Image Receipt Extractor (VLM) | Novita AI | `zai-org/glm-5.3-flash` (vision) | `https://api.novita.ai/openai` |

All three roles share the same model and endpoint.

---

## Token Usage Estimates

### Per-Request Token Budget

| Call Type | Avg Input Tokens | Avg Output Tokens | Avg Total |
|---|---|---|---|
| Strategy Agent (Actor iter 1) | ~2,800 | ~350 | ~3,150 |
| Strategy Agent (Actor iters 2-3, when needed) | ~3,000 | ~350 | ~3,350 |
| Risk Critic (deterministic - no LLM) | 0 | 0 | 0 |
| Explainer Agent | ~1,800 | ~250 | ~2,050 |
| Per request subtotal (avg 1.5 actor iterations) | ~6,075 | ~875 | ~6,950 |

### Full 251-Request Run

| Metric | Value |
|---|---|
| Total requests | 251 |
| Average actor iterations | ~1.5 |
| LLM calls (actor) | ~377 |
| LLM calls (explainer) | ~251 |
| Total LLM calls (main run) | ~628 |
| Estimated total input tokens | ~1,525,000 |
| Estimated total output tokens | ~220,000 |
| Estimated total tokens | ~1,745,000 |
| Avg tokens per request | ~6,950 |

### Preprocessing

| Task | LLM Calls | Tokens (est.) |
|---|---|---|
| Message patch extraction (215 messages) | 215 | ~320,000 |
| Image VLM extraction (16 images) | 16 | ~48,000 |
| Preprocessing total | 231 | ~368,000 |

### Grand Total

| Metric | Value |
|---|---|
| Total LLM calls | ~859 |
| Total tokens | ~2,113,000 |
| Avg per request | ~6,950 |

---

## Cost Estimate

| Component | Tokens | Rate | Cost |
|---|---|---|---|
| Input tokens | ~1,730,000 | ~.10 / 1M | ~.17 |
| Output tokens | ~383,000 | ~.10 / 1M | ~.04 |
| Total estimated cost | | | ~.21 |
| Per request | | | ~.001 |

---

## Architecture Summary

1. DataLoader - reads all CSV datasets
2. ExchangeNormalizer - converts amounts to home currency at fixed rates
3. PatchApplier - applies message patches (salary overrides, cancellations, date changes) and image patches (receipt amounts from VLM)
4. AffordabilityCalculator - deterministic binary search for safe amount and earliest full-payment date
5. SpendingOptimizer - greedy search for minimum spending changes
6. ReflexionLoop - Actor-Critic loop (max 3 iterations): StrategyAgent proposes, RiskCritic validates
7. Deterministic overrides - post-LLM rule enforcement
8. ExplainerAgent - generates counterfactual-aware explanation

All deterministic components run synchronously; LLM calls use asyncio (concurrency=2) with tenacity exponential-backoff retry.
