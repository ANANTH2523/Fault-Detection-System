from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

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
    """
    Behavior Memory (time-decayed).

    We update this online in timestamp order so feature extraction is based on
    historical behavior only (i.e., profile from previous transactions).
    """

    # EWMA amount stats
    ema_amount: float = 0.0
    ema_amount_sq: float = 0.0

    # EWMA transaction interval (seconds)
    ema_interval_sec: float = 0.0

    # Time-decayed weights for normalization
    history_weight: float = 0.0
    last_timestamp: Optional[pd.Timestamp] = None

    # Decayed distributions
    location_weights: Dict[str, float] = field(default_factory=dict)
    recipient_weights: Dict[str, float] = field(default_factory=dict)
    hour_weights: np.ndarray = field(default_factory=lambda: np.zeros(24, dtype=float))

    # For "impossible travel" checks
    last_lat: Optional[float] = None
    last_lon: Optional[float] = None
    last_location_id: Optional[str] = None

    def mean_amount(self) -> float:
        return float(self.ema_amount)

    def var_amount(self) -> float:
        # Var can become slightly negative due to floating point error.
        v = float(self.ema_amount_sq - (self.ema_amount**2))
        return max(0.0, v)

    def amount_std(self) -> float:
        return math.sqrt(self.var_amount()) if self.history_weight > 0.0 else 0.0

    def total_location_weight(self) -> float:
        return float(sum(self.location_weights.values()))

    def total_recipient_weight(self) -> float:
        return float(sum(self.recipient_weights.values()))


def _deterministic_session_id(seed: str) -> str:
    # Avoid randomness (uuid) to keep output fully reproducible.
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    # Keep it short enough for Langfuse constraints (<=200 chars).
    return f"sess_{digest[:40]}"


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _log_value_component(amount: float, p50: float, p95: float, p99: float) -> float:
    # Economic impact component based on a robust log scale.
    amount_f = float(amount) if amount is not None else 0.0
    if amount_f <= 0.0:
        return 0.0

    # Prevent log issues for degenerate inputs.
    p50 = max(float(p50), 1e-9)
    p95 = max(float(p95), p50 + 1e-9)
    p99 = max(float(p99), p95 + 1e-9)

    log_amt = math.log1p(amount_f)
    log_50 = math.log1p(p50)
    log_95 = math.log1p(p95)
    log_99 = math.log1p(p99)

    # Piecewise to keep deterministic and intuitive:
    # - below p50 => ~0
    # - between p50 and p95 => ramps
    # - between p95 and p99 => ramps faster
    if log_amt <= log_50:
        return 0.0
    if log_amt <= log_95:
        return _clamp01((log_amt - log_50) / max(1e-9, log_95 - log_50) * 0.7)
    if log_amt <= log_99:
        return _clamp01(0.7 + (log_amt - log_95) / max(1e-9, log_99 - log_95) * 0.3)
    return 1.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # Distance between coordinates (km). Used only when coordinates exist.
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return r * c


def _feature_risk_from_z(z: float) -> float:
    # Convert a z-like deviation into [0,1] risk deterministically.
    # For z=0 => 0; for larger z => saturate.
    z = max(0.0, float(z))
    return 1.0 - math.exp(-(z * z) / 2.0)


@observe(name="load_data")
def load_data(
    transactions_path: str,
    users_path: str = "",
    gps_path: str = "",
    conversations_path: str = "",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load and standardize the datasets.
    Required columns for transactions are handled defensively during preprocess.
    """
    tx_df_raw = load_dataset(transactions_path, kind=None)
    tx_df = preprocess_data(tx_df_raw)

    user_meta: Dict[str, Any] = {
        "users_df": pd.DataFrame(),
        "conversation_df": pd.DataFrame(),
    }

    if users_path:
        try:
            user_meta["users_df"] = preprocess_data(load_dataset(users_path, kind=None))
            # Note: we re-used preprocess_data; user schemas are optional for the challenge.
        except Exception:
            user_meta["users_df"] = pd.DataFrame()

    if conversations_path:
        try:
            user_meta["conversation_df"] = load_dataset(conversations_path, kind=None)
        except Exception:
            user_meta["conversation_df"] = pd.DataFrame()

    # GPS fusion is optional. If transactions already include lat/lon, we keep them.
    if gps_path and (tx_df["lat"].isna().mean() > 0.8 or tx_df["lon"].isna().mean() > 0.8):
        try:
            gps_df = load_dataset(gps_path, kind=None)
            gps_df = preprocess_data(gps_df)
            # Merge on (user_id, time) by nearest timestamp where possible.
            # If it fails, we fall back silently.
            gps_df = gps_df.rename(columns={"lat": "gps_lat", "lon": "gps_lon", "location_id": "gps_location_id"})
            base = tx_df.copy()
            base = base.sort_values(["user_id", "timestamp"], kind="mergesort")
            gps_df = gps_df.sort_values(["user_id", "timestamp"], kind="mergesort")
            merged = pd.merge_asof(
                base,
                gps_df[["user_id", "timestamp", "gps_lat", "gps_lon", "gps_location_id"]],
                on="timestamp",
                by="user_id",
                direction="nearest",
                tolerance=pd.Timedelta("6h"),
            )
            base["lat"] = merged["lat"].where(base["lat"].notna(), merged["gps_lat"])
            base["lon"] = merged["lon"].where(base["lon"].notna(), merged["gps_lon"])
            base["location_id"] = base["location_id"].where(base["lat"].notna() & base["lon"].notna(), merged["gps_location_id"])
            tx_df = base
        except Exception:
            pass

    return tx_df, user_meta


def _decay_factor(dt_seconds: float, half_life_seconds: float) -> float:
    # EW decay: weight *= exp(-dt / half_life)
    half_life_seconds = max(float(half_life_seconds), 1e-9)
    if dt_seconds <= 0.0:
        return 1.0
    return float(math.exp(-dt_seconds / half_life_seconds))


def _update_decayed_distribution(weights: Dict[str, float], key: str, decay: float) -> None:
    # Apply decay to all weights, then add +1 to the current key.
    if key is None:
        return
    for k in list(weights.keys()):
        weights[k] *= decay
        if weights[k] < 1e-6:
            del weights[k]
    weights[str(key)] = weights.get(str(key), 0.0) + 1.0


def update_user_profile(
    profile: UserProfile,
    tx_row: pd.Series,
    half_life_seconds: float,
) -> UserProfile:
    """
    Behavior Memory Agent:
    Update dynamic user profile with time-decayed statistics.
    """
    ts = tx_row.get("timestamp", pd.NaT)
    amount = float(tx_row.get("amount", 0.0) or 0.0)
    location_id = tx_row.get("location_id", None)
    receiver_id = tx_row.get("receiver_id", None)
    lat = tx_row.get("lat", np.nan)
    lon = tx_row.get("lon", np.nan)
    lat_f = None if pd.isna(lat) else float(lat)
    lon_f = None if pd.isna(lon) else float(lon)

    if profile.last_timestamp is None or pd.isna(ts):
        # Initialize profile without decay.
        profile.ema_amount = amount
        profile.ema_amount_sq = amount * amount
        profile.history_weight = 1.0
        profile.last_timestamp = ts if not pd.isna(ts) else None

        if location_id is not None:
            profile.location_weights[str(location_id)] = profile.location_weights.get(str(location_id), 0.0) + 1.0
        if receiver_id is not None:
            profile.recipient_weights[str(receiver_id)] = profile.recipient_weights.get(str(receiver_id), 0.0) + 1.0
        if not pd.isna(ts):
            hour = int(pd.Timestamp(ts).hour)
            profile.hour_weights[hour] += 1.0

        if lat_f is not None and lon_f is not None:
            profile.last_lat = lat_f
            profile.last_lon = lon_f
        if location_id is not None:
            profile.last_location_id = str(location_id)
        return profile

    dt_seconds = 0.0
    try:
        dt_seconds = float((pd.Timestamp(ts) - pd.Timestamp(profile.last_timestamp)).total_seconds())
    except Exception:
        dt_seconds = 0.0

    decay = _decay_factor(dt_seconds, half_life_seconds=half_life_seconds)

    # Update EWMA stats. We incorporate decay and a fixed learning rate via (1-decay).
    learn = 1.0 - decay
    profile.ema_amount = profile.ema_amount * decay + amount * learn
    profile.ema_amount_sq = profile.ema_amount_sq * decay + (amount * amount) * learn
    profile.history_weight = profile.history_weight * decay + learn

    if dt_seconds > 0.0:
        profile.ema_interval_sec = profile.ema_interval_sec * decay + dt_seconds * learn

    # Update time-of-day distribution.
    if not pd.isna(ts):
        hour = int(pd.Timestamp(ts).hour)
        profile.hour_weights *= decay
        profile.hour_weights[hour] += learn

    # Update location and recipient distributions.
    if location_id is not None:
        _update_decayed_distribution(profile.location_weights, str(location_id), decay=decay)
    if receiver_id is not None:
        _update_decayed_distribution(profile.recipient_weights, str(receiver_id), decay=decay)

    # Update last location (for impossible travel).
    if lat_f is not None and lon_f is not None:
        profile.last_lat = lat_f
        profile.last_lon = lon_f
    profile.last_location_id = str(location_id) if location_id is not None else profile.last_location_id
    profile.last_timestamp = ts

    return profile


def extract_features(profile: Optional[UserProfile], tx_row: pd.Series) -> Dict[str, float]:
    """
    Feature Extraction Agent:
    Compute per-transaction behavioral deviations from the user's history.
    """
    if profile is None:
        return {
            "amount_deviation_risk": 0.0,
            "location_novelty_risk": 0.0,
            "time_novelty_risk": 0.0,
            "burst_risk": 0.0,
            "recipient_novelty_risk": 0.0,
            "impossible_travel_risk": 0.0,
        }

    amount = float(tx_row.get("amount", 0.0) or 0.0)
    location_id = tx_row.get("location_id", None)
    receiver_id = tx_row.get("receiver_id", None)
    ts = tx_row.get("timestamp", pd.NaT)

    # Amount deviation
    mean_amt = profile.mean_amount() if profile.history_weight > 0.0 else 0.0
    std_amt = profile.amount_std()
    z = abs(amount - mean_amt) / max(1e-6, std_amt)
    amount_deviation_risk = _feature_risk_from_z(z)

    # Location novelty: 1 - P(location) based on time-decayed counts.
    total_loc = profile.total_location_weight()
    loc_w = float(profile.location_weights.get(str(location_id), 0.0)) if location_id is not None else 0.0
    location_novelty_risk = 1.0 - _clamp01(loc_w / max(1e-9, total_loc))

    # Time novelty: 1 - P(hour)
    if profile.history_weight > 0.0 and not pd.isna(ts):
        hour = int(pd.Timestamp(ts).hour)
        total_hour = float(profile.hour_weights.sum())
        hour_w = float(profile.hour_weights[hour])
        time_novelty_risk = 1.0 - _clamp01(hour_w / max(1e-9, total_hour))
    else:
        time_novelty_risk = 0.0

    # Burstiness: how fast relative to the user's EWMA interval
    burst_risk = 0.0
    if profile.last_timestamp is not None and not pd.isna(ts) and profile.ema_interval_sec > 0.0:
        try:
            dt = float((pd.Timestamp(ts) - pd.Timestamp(profile.last_timestamp)).total_seconds())
            if dt > 0.0:
                ratio = float(profile.ema_interval_sec) / max(1e-6, dt)  # >1 => faster than normal
                # Map ratio to [0,1]
                burst_risk = _clamp01((ratio - 1.0) / 4.0)
        except Exception:
            burst_risk = 0.0

    # Recipient novelty
    total_rec = profile.total_recipient_weight()
    rec_w = float(profile.recipient_weights.get(str(receiver_id), 0.0)) if receiver_id is not None else 0.0
    recipient_novelty_risk = 1.0 - _clamp01(rec_w / max(1e-9, total_rec))

    # Impossible travel: speed between last coords and current coords
    impossible_travel_risk = 0.0
    lat = tx_row.get("lat", np.nan)
    lon = tx_row.get("lon", np.nan)
    lat_f = None if pd.isna(lat) else float(lat)
    lon_f = None if pd.isna(lon) else float(lon)

    if (
        lat_f is not None
        and lon_f is not None
        and profile.last_lat is not None
        and profile.last_lon is not None
        and profile.last_timestamp is not None
        and not pd.isna(ts)
    ):
        try:
            dist_km = _haversine_km(profile.last_lat, profile.last_lon, lat_f, lon_f)
            dt_hours = float((pd.Timestamp(ts) - pd.Timestamp(profile.last_timestamp)).total_seconds()) / 3600.0
            if dt_hours > 0.0:
                speed_kmh = dist_km / dt_hours
                # If speed is implausible, risk rises steeply.
                impossible_travel_risk = _clamp01((speed_kmh - 800.0) / 800.0)
        except Exception:
            impossible_travel_risk = 0.0

    return {
        "amount_deviation_risk": float(amount_deviation_risk),
        "location_novelty_risk": float(location_novelty_risk),
        "time_novelty_risk": float(time_novelty_risk),
        "burst_risk": float(burst_risk),
        "recipient_novelty_risk": float(recipient_novelty_risk),
        "impossible_travel_risk": float(impossible_travel_risk),
    }


def compute_anomaly_score(features: Dict[str, float]) -> float:
    """
    Anomaly Detection Agent (CORE ENGINE):
    Primary fraud signal based on behavioral deviations.
    """
    weights = {
        "amount_deviation_risk": 0.35,
        "location_novelty_risk": 0.20,
        "time_novelty_risk": 0.15,
        "burst_risk": 0.15,
        "recipient_novelty_risk": 0.10,
        "impossible_travel_risk": 0.05,
    }

    score = 0.0
    for k, w in weights.items():
        score += w * float(features.get(k, 0.0))
    return _clamp01(score)


def compute_risk_score(
    anomaly_score: float,
    amount: float,
    global_amount_p50: float,
    global_amount_p95: float,
    global_amount_p99: float,
) -> float:
    """
    Risk Scoring Agent:
    Combine anomaly with transaction value to prioritize high-value fraud.
    Returns a risk score in [0, 100].
    """
    value_component = _log_value_component(amount, global_amount_p50, global_amount_p95, global_amount_p99)

    # Base combination
    risk = 100.0 * (0.60 * float(anomaly_score) + 0.40 * value_component)

    # Economic impact boost for extreme values.
    if amount >= global_amount_p99:
        risk *= 1.25
    elif amount >= global_amount_p95:
        risk *= 1.10

    return float(max(0.0, min(100.0, risk)))


def _user_behavior_summary(profile: UserProfile, user_id: str) -> str:
    mean_amt = profile.mean_amount() if profile.history_weight > 0.0 else 0.0
    std_amt = profile.amount_std() if profile.history_weight > 0.0 else 0.0
    freq = profile.history_weight / max(1.0, profile.ema_interval_sec / 86400.0) if profile.ema_interval_sec > 0.0 else 0.0

    # Top-k locations/recipients (deterministic order by weight then key).
    top_locs = sorted(profile.location_weights.items(), key=lambda x: (-x[1], x[0]))[:3]
    top_recs = sorted(profile.recipient_weights.items(), key=lambda x: (-x[1], x[0]))[:3]
    hour_top = int(np.argmax(profile.hour_weights)) if profile.hour_weights.sum() > 0.0 else -1

    return (
        f"user_id={user_id}; "
        f"avg_amount={mean_amt:.2f}; std_amount={std_amt:.2f}; "
        f"typical_hour={hour_top if hour_top >= 0 else 'unknown'}; "
        f"freq_per_day_est={freq:.3f}; "
        f"top_locations={','.join([f'{k}:{w:.1f}' for k,w in top_locs]) if top_locs else 'none'}; "
        f"top_recipients={','.join([f'{k}:{w:.1f}' for k,w in top_recs]) if top_recs else 'none'}"
    )


def make_decision(
    anomaly_score: float,
    risk_score: float,
    llm_fraud: Optional[bool],
) -> bool:
    """
    Decision Agent:
      - High anomaly -> fraud directly
      - Medium anomaly -> use LLM
      - Low anomaly -> legitimate
    """
    # 1) High anomaly -> fraud directly.
    if anomaly_score >= 0.80:
        return True

    # 2) Very high economic impact -> fraud directly.
    # (Risk scoring already prioritizes high amounts; this makes value-driven fraud robust.)
    if risk_score >= 90.0:
        return True

    # 3) Low anomaly -> legitimate (unless already covered by the economic-impact rule above).
    if anomaly_score < 0.45 and risk_score < 70.0:
        return False

    # 4) Medium/uncertain anomaly -> use LLM (or deterministic fallback if LLM is unavailable).
    if llm_fraud is None:
        return risk_score >= 75.0
    return bool(llm_fraud)


@observe(name="run_fraud_detection")
def run_detection(
    tx_df: pd.DataFrame,
    user_meta: Dict[str, Any],
    session_id: str,
    half_life_days: float = 7.0,
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> pd.DataFrame:
    """
    Full multi-agent pipeline implemented as an online, timestamp-ordered process.
    """
    if tx_df is None or tx_df.empty:
        return pd.DataFrame(columns=["tx_id", "user_id", "is_fraud", "risk_score", "anomaly_score"])

    # Ensure deterministic processing order:
    # - Sort by timestamp (NaT last)
    # - Tie-break by tx_id
    base = tx_df.copy()
    base["timestamp_sort"] = base["timestamp"].fillna(pd.Timestamp("2100-01-01T00:00:00Z"))
    base = base.sort_values(["timestamp_sort", "tx_id"], kind="mergesort")

    amount = pd.to_numeric(base["amount"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    base["amount"] = amount

    # Global amount stats for economic impact scoring.
    global_p50 = float(np.nanquantile(amount, 0.50)) if len(amount) else 0.0
    global_p95 = float(np.nanquantile(amount, 0.95)) if len(amount) else 0.0
    global_p99 = float(np.nanquantile(amount, 0.99)) if len(amount) else 0.0
    # Guard against degenerate quantiles.
    global_p95 = max(global_p95, global_p50 + 1e-9)
    global_p99 = max(global_p99, global_p95 + 1e-9)

    half_life_seconds = max(float(half_life_days) * 86400.0, 1.0)

    profiles: Dict[str, UserProfile] = {}
    decisions: list[Dict[str, Any]] = []

    # Langfuse session propagation for all observed spans.
    # (If Langfuse is not configured, these are effectively no-ops.)
    langfuse = get_client()
    with propagate_attributes(session_id=session_id):
        for _, tx_row in base.iterrows():
            tx_id = str(tx_row.get("tx_id", "")).strip()
            user_id = str(tx_row.get("user_id", "unknown_user")).strip()
            amount_f = float(tx_row.get("amount", 0.0) or 0.0)

            if user_id not in profiles:
                profiles[user_id] = UserProfile()

            profile = profiles[user_id]

            # 1) Feature extraction based on historical profile (before updating with this tx).
            features = extract_features(profile if profile.history_weight > 0.0 else None, tx_row)
            anomaly_score = compute_anomaly_score(features)

            # 2) Risk scoring (economic impact prioritized)
            risk_score = compute_risk_score(
                anomaly_score=anomaly_score,
                amount=amount_f,
                global_amount_p50=global_p50,
                global_amount_p95=global_p95,
                global_amount_p99=global_p99,
            )

            # 3) Selective LLM escalation (only for medium anomaly cases).
            # LLM agent will also consider high-value transactions (it can decide Fraud even if anomaly is moderate).
            llm_fraud: Optional[bool] = None
            high_value = amount_f >= global_p95
            llm_needed = (0.45 <= anomaly_score < 0.80) or (
                high_value and 70.0 <= risk_score < 90.0 and anomaly_score >= 0.35
            )
            if llm_needed:
                # Build a deterministic prompt context.
                profile_summary = _user_behavior_summary(profile, user_id) if profile.history_weight > 0.0 else ""
                tx_details = {
                    "tx_id": tx_id,
                    "amount": amount_f,
                    "type": tx_row.get("tx_type", ""),
                    "timestamp": str(tx_row.get("timestamp", "")),
                    "location_id": tx_row.get("location_id", "unknown_location"),
                    "receiver_id": tx_row.get("receiver_id", "unknown_receiver"),
                    "anomaly_score": float(anomaly_score),
                    "risk_score": float(risk_score),
                }

                # Optional: include user demographics if they were provided.
                # This is intentionally kept short to reduce latency/token usage.
                user_demo = ""
                try:
                    users_df = user_meta.get("users_df", pd.DataFrame())
                    if users_df is not None and not users_df.empty and "user_id" in users_df.columns:
                        demo_row = users_df[users_df["user_id"] == user_id].head(1)
                        if not demo_row.empty:
                            # Pick a few likely demographic columns.
                            col_candidates = [c for c in demo_row.columns if c not in {"tx_id", "amount", "timestamp", "lat", "lon", "location_id"}]
                            kv = []
                            for c in col_candidates[:5]:
                                val = demo_row.iloc[0].get(c, None)
                                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                                    kv.append(f"{c}={val}")
                            if kv:
                                user_demo = "user_demo=" + ",".join(kv)
                except Exception:
                    user_demo = ""

                callback_handler = None
                # llm_analysis function will handle callback + observe if Langfuse is configured.
                llm_fraud = llm_analysis(
                    profile_summary=profile_summary,
                    tx_details=tx_details,
                    session_id=session_id,
                    model=model,
                    temperature=temperature,
                    callback_handler=callback_handler,
                )

            is_fraud = make_decision(
                anomaly_score=float(anomaly_score),
                risk_score=float(risk_score),
                llm_fraud=llm_fraud,
            )

            decisions.append(
                {
                    "tx_id": tx_id,
                    "user_id": user_id,
                    "amount": amount_f,
                    "timestamp": tx_row.get("timestamp", pd.NaT),
                    "location_id": tx_row.get("location_id", "unknown_location"),
                    "risk_score": float(risk_score),
                    "anomaly_score": float(anomaly_score),
                    "llm_fraud": llm_fraud,
                    "is_fraud": bool(is_fraud),
                }
            )

            # 4) Update behavior memory after decision.
            profiles[user_id] = update_user_profile(profile, tx_row, half_life_seconds=half_life_seconds)

    out = pd.DataFrame(decisions)
    return out


def run_pipeline(
    transactions_path: str,
    users_path: str = "",
    gps_path: str = "",
    conversations_path: str = "",
    output_path: str = os.path.join("output", "output.txt"),
    session_id: str = "",
    half_life_days: float = 7.0,
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> str:
    tx_df, user_meta = load_data(
        transactions_path=transactions_path,
        users_path=users_path,
        gps_path=gps_path,
        conversations_path=conversations_path,
    )

    if not session_id:
        seed = "|".join([transactions_path, users_path, gps_path, conversations_path, str(half_life_days), str(model), str(temperature)])
        session_id = _deterministic_session_id(seed)

    scored_df = run_detection(
        tx_df=tx_df,
        user_meta=user_meta,
        session_id=session_id,
        half_life_days=half_life_days,
        model=model,
        temperature=temperature,
    )

    write_output(scored_df, output_path=output_path)
    return output_path

