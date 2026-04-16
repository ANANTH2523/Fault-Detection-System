from __future__ import annotations

import argparse
import os

from fraud_system import run_pipeline
from utils import find_input_path, make_synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Production-ready multi-agent fraud detection pipeline.")
    parser.add_argument(
        "--data",
        default="",
        help="Backward-compatible: path to the transactions CSV/JSON. If omitted, --transactions is used (or auto-discovery).",
    )
    parser.add_argument(
        "--transactions",
        default="",
        help="Path to transactions CSV/JSON.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("output", "output.txt"),
        help="Output path for fraudulent transaction IDs.",
    )
    parser.add_argument(
        "--session-id",
        default="",
        help="Optional Langfuse session_id. If omitted, a deterministic session_id is derived from inputs.",
    )
    parser.add_argument(
        "--half-life-days",
        type=float,
        default=7.0,
        help="Time decay half-life (in days) for user behavior memory.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Optional OpenAI model name (LangChain ChatOpenAI).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature (must be low for deterministic output).",
    )
    args = parser.parse_args()

    transactions_path = args.data.strip() or args.transactions.strip()
    if not transactions_path:
        spec = find_input_path()
        transactions_path = spec.path if spec else ""

    # If no input exists, create deterministic synthetic data and run end-to-end.
    if not transactions_path:
        os.makedirs("output", exist_ok=True)
        tmp_tx_path = os.path.join("output", "synthetic_transactions.csv")
        make_synthetic_dataset().to_csv(tmp_tx_path, index=False)
        transactions_path = tmp_tx_path

    run_pipeline(
        transactions_path=transactions_path,
        output_path=args.output,
        session_id=args.session_id.strip(),
        half_life_days=float(args.half_life_days),
        model=args.model.strip() or None,
        temperature=float(args.temperature),
    )


if __name__ == "__main__":
    main()

