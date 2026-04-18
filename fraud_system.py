from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

from llm import llm_analysis
from utils import preprocess_data, load_dataset, write_output

try:
    from langfuse.decorators import observe
    from langfuse import propagate_attributes, get_client
except Exception:  # pragma: no cover
    def observe(*_args: Any, **_kwargs: Any):  # type: ignore
        def _decorator(fn):
            return fn
        return _decorator

    def propagate_attributes(*_args: Any, **_kwargs: Any):  # type: ignore
        class _DummyCtx:
            def __enter__(self):
                return None
            def __exit__(self, exc_type, exc, tb):
                return False
        return _DummyCtx()

    def get_client() -> Any:  # type: ignore
        return None

@dataclass
class UserProfile:
    # Basic info
    salary_monthly: float = 0.0
    first_name: str = ""
    compromise_timestamps: List[pd.Timestamp] = field(default_factory=list)
    
    # "Impossible travel" tracker
    last_timestamp: Optional[pd.Timestamp] = None
    last_lat: Optional[float] = None
    last_lon: Optional[float] = None


def _deterministic_session_id(seed: str) -> str:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return f"sess_{digest[:40]}"

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return r * c

@observe(name="load_multi_modal_data")
def load_multi_modal_data(
    transactions_path: str,
    users_path: str = "",
    locations_path: str = "",
    sms_path: str = "",
    mails_path: str = "",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    tx_df = pd.read_csv(transactions_path)
    
    # Parse Users
    user_map = {} # sender_iban -> dict(salary, first_name)
    if users_path and os.path.exists(users_path):
        users_df = pd.read_json(users_path)
        for _, row in users_df.iterrows():
            iba = str(row.get('iban', ''))
            user_map[iba] = {
                'salary': float(row.get('salary', 36000)),
                'first_name': str(row.get('first_name', ''))
            }

    # Parse Locations
    locations_df = pd.DataFrame()
    if locations_path and os.path.exists(locations_path):
        # some locations.json might be large, use read_json
        locations_df = pd.read_json(locations_path)
        if 'timestamp' in locations_df.columns:
            locations_df['timestamp'] = pd.to_datetime(locations_df['timestamp'], utc=True)

    # Parse Comms (SMS + Mails)
    comms_list = []
    if sms_path and os.path.exists(sms_path):
        sms_df = pd.read_json(sms_path)
        for _, row in sms_df.iterrows():
            text = str(row.get('sms', ''))
            # Try to grab date assuming "Date: 2087-03-20 13:36:49"
            m = re.search(r"Date:\s*([\d-]+ [\d:]+)", text)
            if m:
                ts = pd.to_datetime(m.group(1), errors="coerce", utc=True)
                if not pd.isna(ts):
                    comms_list.append({"text": text.lower(), "timestamp": ts})
                    
    if mails_path and os.path.exists(mails_path):
        mail_df = pd.read_json(mails_path)
        for _, row in mail_df.iterrows():
            text = str(row.get('mail', ''))
            m = re.search(r"Date:\s*(.+)", text) # Date: Sat, 22 Mar 2087 16:31:10 +0100
            if m:
                ts = pd.to_datetime(m.group(1), errors="coerce", utc=True)
                if not pd.isna(ts):
                    comms_list.append({"text": text.lower(), "timestamp": ts})

    comms_df = pd.DataFrame(comms_list)
    
    return tx_df, {
        "user_map": user_map,
        "locations_df": locations_df,
        "comms_df": comms_df
    }

def extract_features(profile: UserProfile, tx_row: pd.Series, user_meta: Dict[str, Any]) -> Dict[str, float]:
    amount = float(tx_row.get("amount", 0.0) or 0.0)
    ts = pd.to_datetime(tx_row.get("timestamp")) if pd.notna(tx_row.get("timestamp")) else pd.NaT
    sender_id = str(tx_row.get("sender_id", ""))
    
    # 1. Financial Behavior Check: amount vs 50% monthly salary
    financial_risk = 0.0
    if profile.salary_monthly > 0:
        if amount > (profile.salary_monthly * 0.5):
            financial_risk = 1.0 # Exceeds 50% monthly salary

    # 2. Temporal Behavior Check: between 00:00 and 05:59
    temporal_risk = 0.0
    if not pd.isna(ts) and ts.tz_localize(None).hour >= 0 and ts.tz_localize(None).hour < 6:
        temporal_risk = 1.0

    # 3. Communication/Compromise Window Check
    compromise_risk = 0.0
    if not pd.isna(ts) and profile.compromise_timestamps:
        # Was there a compromise in the trailing 72 hours?
        for phish_ts in profile.compromise_timestamps:
            delta = ts - phish_ts
            if pd.Timedelta(hours=0) <= delta <= pd.Timedelta(hours=72):
                compromise_risk = 1.0
                break

    # 4. Geographic Velocity Check
    impossible_travel_risk = 0.0
    locations_df = user_meta.get("locations_df", pd.DataFrame())
    
    # We find the nearest location for this user's biotag around the tx timestamp
    if not locations_df.empty and sender_id and not pd.isna(ts):
        user_locs = locations_df[locations_df['biotag'] == sender_id]
        if not user_locs.empty:
            # find location closest to transaction timestamp but BEFORE or AT transaction
            past_locs = user_locs[user_locs['timestamp'] <= ts]
            if not past_locs.empty:
                nearest_loc = past_locs.sort_values('timestamp', ascending=False).iloc[0]
                
                # Fetch transaction lat/lon (maybe derived from merchant or ip - for our dataset we only have location string)
                # But notice transactions.csv doesn't have lat/lng! Only "Dietzenbach - Dietzenbach Coffee House"
                # If transaction doesn't have explicit lat/lng, we only calculate speed if the transaction had coordinates.
                # Actually, in our pipeline GPS merging is assumed, or we just use locations.json for impossible travel logic!
                # Wait, the user is taking an action "at a location". Let's approximate: 
                # transactions might be e-commerce. Let's just leave this modular and log risk 0 if tx coords don't exist.
                # To accurately do this we'd need merchant coordinates. 
                pass

    return {
        "financial_risk": financial_risk,
        "temporal_risk": temporal_risk,
        "compromise_risk": compromise_risk,
        "impossible_travel_risk": impossible_travel_risk,
    }


def compute_risk_score(features: Dict[str, float]) -> float:
    """
    Combines the deterministic checks.
    """
    # Base risk starts low.
    risk = 0.0
    
    if features.get("compromise_risk", 0.0) == 1.0 and features.get("financial_risk", 0.0) == 1.0:
        return 100.0 # Guaranteed Fraud
        
    if features.get("financial_risk", 0.0) == 1.0 and features.get("temporal_risk", 0.0) == 1.0:
        return 85.0 # High Risk
        
    if features.get("compromise_risk", 0.0) == 1.0:
        risk = max(risk, 75.0)
        
    if features.get("financial_risk", 0.0) == 1.0:
        risk = max(risk, 60.0)
        
    if features.get("temporal_risk", 0.0) == 1.0:
        risk = max(risk, 30.0)
        
    if features.get("impossible_travel_risk", 0.0) == 1.0:
        return 100.0
        
    return risk

def make_decision(risk_score: float, llm_fraud: Optional[bool]) -> bool:
    if risk_score >= 90.0:
        return True
    if risk_score < 50.0:
        return False
    if llm_fraud is None:
        return risk_score >= 75.0
    return bool(llm_fraud)

@observe(name="run_behavioral_fraud_detection")
def run_detection(
    tx_df: pd.DataFrame,
    user_meta: Dict[str, Any],
    session_id: str,
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> pd.DataFrame:

    base = tx_df.copy()
    if 'timestamp' in base.columns:
        base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True)
    base = base.sort_values("timestamp", kind="mergesort")
    
    user_map = user_meta.get("user_map", {})
    comms_df = user_meta.get("comms_df", pd.DataFrame())
    
    # Pre-compute compromise timestamps for all users
    # Search for paypa1, amaz0n, ub3r
    malicious_domains = ["paypa1", "amaz0n", "ub3r"]
    
    profiles: Dict[str, UserProfile] = {}
    decisions = []

    phishing_filter = False
    if not comms_df.empty:
        pattern = "|".join(malicious_domains)
        phishing_mask = comms_df['text'].str.contains(pattern, na=False)
        phishing_comms = comms_df[phishing_mask]

    for _, tx_row in base.iterrows():
        tx_id = str(tx_row.get("transaction_id", ""))
        sender_iban = str(tx_row.get("sender_iban", ""))
        sender_id = str(tx_row.get("sender_id", ""))
        
        # Init profile if missing
        if sender_id not in profiles:
            p = UserProfile()
            if sender_iban in user_map:
                p.salary_monthly = user_map[sender_iban]["salary"] / 12.0
                p.first_name = user_map[sender_iban]["first_name"]
                
                # Setup compromise timestamps
                if not comms_df.empty and p.first_name:
                    my_phish = phishing_comms[phishing_comms['text'].str.contains(p.first_name.lower(), na=False)]
                    p.compromise_timestamps = my_phish['timestamp'].tolist()
            profiles[sender_id] = p
            
        profile = profiles[sender_id]
        
        # Extract behavior features
        features = extract_features(profile, tx_row, user_meta)
        risk_score = compute_risk_score(features)
        
        llm_fraud = None
        # Send to LLM if indeterminate high risk (50 - 89)
        if 50.0 <= risk_score < 90.0:
            tx_details = {
                "transaction_id": tx_id,
                "amount": float(tx_row.get("amount", 0.0)),
                "type": tx_row.get("transaction_type", ""),
                "timestamp": str(tx_row.get("timestamp", "")),
                "risk_features": features,
            }
            llm_fraud = llm_analysis(
                profile_summary=f"Monthly Salary: {profile.salary_monthly}, High Risk Compromise: {features['compromise_risk']}",
                tx_details=tx_details,
                session_id=session_id,
                model=model,
                temperature=temperature,
            )
            
        is_fraud = make_decision(risk_score, llm_fraud)
        
        decisions.append({
            "tx_id": tx_id,
            "user_id": sender_id, # for utils write_output explicitly needs tx_id/is_fraud
            "is_fraud": is_fraud,
            "risk_score": risk_score,
            "anomaly_score": risk_score / 100.0,
        })

    return pd.DataFrame(decisions)


def run_pipeline(
    transactions_path: str,
    users_path: str = "",
    locations_path: str = "",
    sms_path: str = "",
    mails_path: str = "",
    output_path: str = os.path.join("output", "output.txt"),
    session_id: str = "",
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> str:
    # Auto-find aux files if not provided but exists in the same folder as transactions
    dirname = os.path.dirname(transactions_path)
    if not users_path:
        users_path = os.path.join(dirname, "users.json")
    if not locations_path:
        locations_path = os.path.join(dirname, "locations.json")
    if not sms_path:
        sms_path = os.path.join(dirname, "sms.json")
    if not mails_path:
        mails_path = os.path.join(dirname, "mails.json")

    tx_df, user_meta = load_multi_modal_data(
        transactions_path, users_path, locations_path, sms_path, mails_path
    )

    if not session_id:
        session_id = _deterministic_session_id(transactions_path)

    scored_df = run_detection(
        tx_df=tx_df,
        user_meta=user_meta,
        session_id=session_id,
        model=model,
        temperature=temperature,
    )

    write_output(scored_df, output_path=output_path)
    return output_path
