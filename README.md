# FinSight – Buy or Wait? Challenge

## Overview
FinSight is an AI‑powered financial decision agent built to solve the HackerRank **Buy or Wait?** challenge. It reads user financial profiles, events, exchange rates, messages, and images, then decides whether a purchase request is affordable now, later, via a payment plan, or not at all.

## Repository Structure
```
FinSight/
├─ code/                # Core source code
│   ├─ __init__.py
│   ├─ main.py          # CLI entry point
│   ├─ config.py        # Settings (Novita API key, model, paths)
│   ├─ schemas.py       # Pydantic models
│   ├─ data_loader.py   # CSV ingestion & preprocessing
│   └─ ...
├─ dataset/             # Provided CSVs and media
├─ .env                 # Template for API keys (never commit secrets)
├─ requirements.txt     # Python dependencies
├─ .gitignore           # Ignored files
└─ README.md            # This file
```

## Setup
```bash
# Clone the repo (already done)
python -m venv .venv
source .venv/Scripts/activate  # Windows PowerShell
pip install -r requirements.txt
# Copy .env and add your Novita API key
cp .env.example .env   # edit .env with your key
```

## Running the Agent
```bash
python -m code.main
```
The script loads the dataset, builds the delta‑state, runs the Neuro‑Symbolic Reflexion Loop and writes `dataset/output.csv`.

## Testing & Evaluation
Running the benchmark suite (to be added in later phases) will produce a usage report at `code/evaluation/usage_report.md`.

## License & Credits
© 2026 FinSight – built for the HackerRank Orchestrate hackathon.
```
