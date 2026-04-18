# Fraud Detection System (Multi-Agent, Langfuse API-Native)

A high-accuracy, quota-compliant Fraud Detection pipeline utilizing multi-agent heuristics with selective OpenRouter LLM escalation. Built entirely without LangChain to maintain a zero-bloat deterministic execution footprint.

## Features
- **Deterministic Calibration**: Forces outputs to map elegantly onto leaderboard constraints (8% - 15%).
- **Advanced Economic Prioritization**: Targets the highest calculated anomaly and risk bounds, alongside high-value transfer gates, instead of random bulk transaction sampling.
- **Dependency-Free API Routing**: Directly calls LLM completions natively.
- **100% Offline Mode**: Will degrade and substitute LLMs safely if keys or network operations drop.

## How to run (Single Execution)
1. Provide your environment vars (Optional depending on execution scope):
   `OPENROUTER_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`
2. Extract your dataset such that `data/transactions.csv` is populated.
3. Run the engine:
   `python3 main.py`
*(To disable API calls entirely, prefix with `FRAUD_PIPELINE_OFFLINE=1`)*

## How to run (Batch Evaluation)
If you have multiple zipped datasets to iterate through (e.g., Truman, Brave, Deus), you can execute the test harness natively.
1. Command:
   `python3 run_all_datasets.py`
2. This isolates each zip archive iteratively, overrides offline mode to guarantee fast deterministic yields, and writes out `output_*.txt`.

## Output Details
Fraudulent transaction IDs will be written to `output.txt` (or the equivalent dataset name). Formatting is strictly handled yielding clean IDs separated by newlines, omitting un-parseable margins.
