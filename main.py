from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import ulid
from dotenv import load_dotenv

from fraud_system import run_pipeline
from utils import find_input_path, make_synthetic_dataset

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Load .env (check parent directory too)
# ──────────────────────────────────────────────
load_dotenv()  # cwd
load_dotenv(Path(__file__).resolve().parent.parent / ".env")  # parent (dataset dir)

# ──────────────────────────────────────────────
# Validate required env vars
# ──────────────────────────────────────────────
_REQUIRED_VARS = ["OPENROUTER_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]
_missing = [v for v in _REQUIRED_VARS if not os.getenv(v)]
if _missing:
    log.error("Missing required environment variable(s): %s", ", ".join(_missing))
    sys.exit(1)


# ──────────────────────────────────────────────
# Auto-discover dataset paths
# ──────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent  # ../  (The Truman Show - train)

def _find_dataset(name: str, extensions: tuple[str, ...] = (".csv", ".json")) -> str:
    """Search DATA_DIR for a file matching *name* with any of the given extensions."""
    for ext in extensions:
        candidate = DATA_DIR / f"{name}{ext}"
        if candidate.exists():
            return str(candidate)
    return ""


def generate_session_id() -> str:
    """Create a unique session ID: {TEAM_NAME}-{ULID}."""
    team = os.getenv("TEAM_NAME", "team").replace(" ", "-")
    session_id = f"{team}-{ulid.new().str}"
    log.info("Generated session ID: %s", session_id)
    return session_id


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
        "--users",
        default="",
        help="Path to users JSON.",
    )
    parser.add_argument(
        "--locations",
        default="",
        help="Path to locations/GPS JSON.",
    )
    parser.add_argument(
        "--conversations",
        default="",
        help="Path to conversations (SMS + mails) JSON.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("output", "output.txt"),
        help="Output path for fraudulent transaction IDs.",
    )
    parser.add_argument(
        "--session-id",
        default="",
        help="Optional Langfuse session_id. If omitted, one is auto-generated as TEAM_NAME-ULID.",
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
        help="Optional LLM model name (via OpenRouter).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature (must be low for deterministic output).",
    )
    args = parser.parse_args()

    # ── Resolve dataset paths (CLI args > auto-discovery) ──
    transactions_path = args.data.strip() or args.transactions.strip() or _find_dataset("transactions")
    users_path        = args.users.strip() or _find_dataset("users")
    locations_path    = args.locations.strip() or _find_dataset("locations")

    # Conversations = sms.json + mails.json  (handled as two separate files by the pipeline)
    conversations_path = args.conversations.strip() or _find_dataset("sms")

    if not transactions_path:
        spec = find_input_path()
        transactions_path = spec.path if spec else ""

    # If no input exists, create deterministic synthetic data and run end-to-end.
    if not transactions_path:
        os.makedirs("output", exist_ok=True)
        tmp_tx_path = os.path.join("output", "synthetic_transactions.csv")
        make_synthetic_dataset().to_csv(tmp_tx_path, index=False)
        transactions_path = tmp_tx_path

    # ── Log discovered datasets ──
    log.info("Transactions : %s", transactions_path or "(not found)")
    log.info("Users        : %s", users_path or "(not found)")
    log.info("Locations    : %s", locations_path or "(not found)")
    log.info("Conversations: %s", conversations_path or "(not found)")

    # ── Session ID ──
    session_id = args.session_id.strip() or generate_session_id()

    run_pipeline(
        transactions_path=transactions_path,
        users_path=users_path,
        gps_path=locations_path,
        conversations_path=conversations_path,
        output_path=args.output,
        session_id=session_id,
        half_life_days=float(args.half_life_days),
        model=args.model.strip() or None,
        temperature=float(args.temperature),
    )

    log.info("Pipeline complete | session: %s", session_id)


if __name__ == "__main__":
    main()

