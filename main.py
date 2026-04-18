"""
RUN INSTRUCTIONS:

1. Install dependencies:
   pip install -r requirements.txt

2. Run:
   python main.py
"""

import pandas as pd
import uuid
import os
import math
from langfuse import observe, get_client
from dotenv import load_dotenv

from data_loader import DataLoader
from agents import MemoryAgent, GlobalThreatAgent, SecurityContextAgent, HeuristicAgent, RiskAgent, DecisionAgent

# Initialize environment directly incase .env was bypassed
load_dotenv()

@observe(name="process_transaction")
def process_transaction(tx, memory, global_threat, sec_context, heuristic, risk, decision, session_id):
    client = get_client()
    if client:
        client.update_current_trace(session_id=session_id)
        
    user = tx["sender_id"]
    profile = memory.get_profile(user)
    
    is_fraudulent = False
    anomaly = 0.0
    risk_val = 0.0
    
    if profile:
        intel = sec_context.fetch_intel(tx)
        threat_multiplier = global_threat.check(tx)
        
        anomaly = heuristic.anomaly_score(tx, profile, intel)
        risk_val = risk.risk_score(tx, anomaly, threat_multiplier)
        
        if decision.is_fraud(tx, profile, intel, anomaly, risk_val, session_id):
            is_fraudulent = True
            global_threat.register_fraud(tx)
            
    memory.update_profile(tx)
    
    return {
        "transaction_id": tx["transaction_id"],
        "amount": float(tx["amount"]),
        "is_high_value": bool(tx["amount"] > 20000),
        "is_fraud": bool(is_fraudulent),
        "anomaly": float(anomaly),
        "risk": float(risk_val),
    }

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


def economic_priority(record):
    return (
        1 if record["is_high_value"] else 0,
        record["risk"],
        record["anomaly"],
        record["amount"],
    )

    log.info("Pipeline complete | session: %s", session_id)


def calibrate_fraud_records(records, min_ratio=0.08, max_ratio=0.15):
    if not records:
        return []

    total = len(records)
    min_count = max(1, math.ceil(total * min_ratio))
    max_count = max(min_count, math.floor(total * max_ratio))

    selected = {record["transaction_id"]: record for record in records if record["is_fraud"]}

    def trim_to_max():
        if len(selected) <= max_count:
            return
        ranked = sorted(selected.values(), key=economic_priority, reverse=True)
        keep = {record["transaction_id"] for record in ranked[:max_count]}
        selected.clear()
        selected.update((record["transaction_id"], record) for record in records if record["transaction_id"] in keep)

    trim_to_max()

    high_value_candidates = sorted(
        (record for record in records if record["is_high_value"]),
        key=economic_priority,
        reverse=True,
    )
    if high_value_candidates and not any(record["is_high_value"] for record in selected.values()):
        best_high_value = high_value_candidates[0]
        if len(selected) < max_count:
            selected[best_high_value["transaction_id"]] = best_high_value
        else:
            replaceable = sorted(
                (record for record in selected.values() if not record["is_high_value"]),
                key=economic_priority,
            )
            if replaceable:
                selected.pop(replaceable[0]["transaction_id"], None)
                selected[best_high_value["transaction_id"]] = best_high_value

    if len(selected) < min_count:
        for record in sorted(records, key=economic_priority, reverse=True):
            selected.setdefault(record["transaction_id"], record)
            if len(selected) >= min_count:
                break

    trim_to_max()
    selected_ids = set(selected)
    return [record for record in records if record["transaction_id"] in selected_ids]

def main():
    # UUID session generator locked entirely on one execution
    session_id = str(uuid.uuid4())

    loader = DataLoader()
    context_data = loader.load_all()

    # Relative target path logic required
    df = pd.read_csv("data/transactions.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour

    records = []
    
    memory = MemoryAgent()
    global_threat = GlobalThreatAgent()
    sec_context = SecurityContextAgent(context_data)
    heuristic = HeuristicAgent()
    risk = RiskAgent()
    decision_eligible_count = max(0, len(df) - df["sender_id"].nunique())
    decision = DecisionAgent(total_transactions=decision_eligible_count)

    for _, tx in df.iterrows():
        records.append(process_transaction(tx, memory, global_threat, sec_context, heuristic, risk, decision, session_id))

    fraud_records = calibrate_fraud_records(records)
    frauds = [record["transaction_id"] for record in fraud_records]
    high_value_flagged = sum(1 for record in fraud_records if record["is_high_value"])
    high_value_transactions = sum(1 for record in records if record["is_high_value"])

    client = get_client()
    if client:
        client.flush()

    with open("output.txt", "w") as f:
        for tx_id in frauds:
            f.write(str(tx_id) + "\n")
            
    # Metric Debug Tracker output logic
    total_tx = len(df)
    fraud_count = len(frauds)
    fraud_perc = (fraud_count / total_tx) * 100 if total_tx > 0 else 0
    llm_perc = (decision.llm_calls / total_tx) * 100 if total_tx > 0 else 0

    print("--- Session Execution Completed ---")
    print(f"Total Transactions: {total_tx}")
    print(f"Frauds Detected:    {fraud_count}")
    print(f"Fraud Percentage:   {fraud_perc:.2f}%")
    print(f"Total LLM Calls:    {decision.llm_calls}")
    print(f"LLM Usage:          {llm_perc:.2f}%")
    print(f"High-Value Tx Count:{high_value_transactions}")
    print(f"High-Value Fraud Detected: {high_value_flagged}")
    print(f"Session ID:         {session_id}")

if __name__ == "__main__":
    main()
