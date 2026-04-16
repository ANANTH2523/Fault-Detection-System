from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def _clamp01(arr: np.ndarray) -> np.ndarray:
    return np.minimum(1.0, np.maximum(0.0, arr))


def _amount_risk_component(amount: pd.Series) -> np.ndarray:
    amt = amount.astype(float).fillna(0.0).to_numpy()
    if len(amt) == 0:
        return np.array([], dtype=float)

    # Robust piecewise mapping to keep deterministic behavior.
    p50, p75, p90, p95 = np.nanquantile(amt, [0.5, 0.75, 0.9, 0.95])
    comp = np.zeros_like(amt, dtype=float)

    comp = np.where(amt <= p50, 0.0, comp)
    comp = np.where((amt > p50) & (amt <= p75), 0.25, comp)
    comp = np.where((amt > p75) & (amt <= p90), 0.50, comp)
    comp = np.where((amt > p90) & (amt <= p95), 0.75, comp)
    comp = np.where(amt > p95, 1.0, comp)
    return _clamp01(comp)


def _location_risk_component(df: pd.DataFrame) -> np.ndarray:
    if df.empty:
        return np.array([], dtype=float)

    has_coords = df["lat"].notna().mean() > 0.2 and df["lon"].notna().mean() > 0.2
    if has_coords:
        work = df[["user_id", "lat", "lon"]].copy()
        centroid = work.groupby("user_id", as_index=False)[["lat", "lon"]].mean()
        centroid = centroid.rename(columns={"lat": "lat_mean", "lon": "lon_mean"})
        work = work.merge(centroid, on="user_id", how="left")

        # Distance in "degrees", with longitude scaled by cosine(latitude).
        lat_mean_rad = np.deg2rad(work["lat_mean"].to_numpy())
        cos_lat = np.cos(lat_mean_rad)
        cos_lat = np.where(cos_lat <= 1e-6, 1e-6, cos_lat)

        dlat = work["lat"].to_numpy() - work["lat_mean"].to_numpy()
        dlon = (work["lon"].to_numpy() - work["lon_mean"].to_numpy()) * cos_lat
        dist = np.sqrt(dlat**2 + dlon**2)

        work["_dist"] = dist
        # Per-user normalization: 0 at median, 1 at >= 90th percentile.
        dist_median = work.groupby("user_id")["_dist"].transform("median").to_numpy()
        dist_p90 = work.groupby("user_id")["_dist"].transform(lambda s: s.quantile(0.9)).to_numpy()

        comp = np.zeros_like(dist, dtype=float)
        comp = np.where(dist <= dist_median, 0.0, comp)
        comp = np.where((dist > dist_median) & (dist < dist_p90), 0.5, comp)
        comp = np.where(dist >= dist_p90, 1.0, comp)
        return _clamp01(comp)

    # Fallback: location rarity per user_id.
    counts = df.groupby(["user_id", "location_id"], dropna=False).size().rename("loc_count").reset_index()
    counts["rarity"] = 1.0 / np.maximum(1.0, counts["loc_count"].to_numpy(dtype=float))

    max_rarity = counts.groupby("user_id")["rarity"].transform("max").to_numpy(dtype=float)
    # Avoid division by zero.
    denom = np.maximum(1e-12, np.log1p(max_rarity))
    comp_map = np.log1p(counts["rarity"].to_numpy(dtype=float)) / denom
    counts["_comp"] = _clamp01(comp_map)

    merged = df[["user_id", "location_id"]].merge(counts[["user_id", "location_id", "_comp"]], how="left")
    merged["_comp"] = merged["_comp"].fillna(0.0)
    return merged["_comp"].to_numpy(dtype=float)


def _rolling_count_within_window_seconds(times: np.ndarray, window_seconds: int) -> np.ndarray:
    """
    Two-pointer sliding window counts for each element.
    times must be sorted ascending and in int seconds.
    Returns counts of previous events within the window INCLUDING current.
    """
    n = len(times)
    if n == 0:
        return np.array([], dtype=int)

    out = np.zeros(n, dtype=int)
    left = 0
    for i in range(n):
        t = times[i]
        while left < i and times[left] < t - window_seconds:
            left += 1
        out[i] = i - left + 1
    return out


def _frequency_risk_component(df: pd.DataFrame) -> np.ndarray:
    if df.empty:
        return np.array([], dtype=float)

    ts = df["timestamp"]
    has_ts = ts.notna().mean() > 0.2
    if has_ts:
        # Use 24-hour rolling count.
        # Convert to seconds since epoch for speed.
        times = ts.astype("int64") // 10**9
        out = np.zeros(len(df), dtype=float)

        for user_id, idx in df.groupby("user_id").groups.items():
            user_positions = np.array(idx, dtype=int)
            user_positions_sorted = user_positions[np.argsort(times[user_positions])]
            user_times = times[user_positions_sorted]
            user_counts = _rolling_count_within_window_seconds(user_times, window_seconds=24 * 3600)

            # Normalize counts globally via quantiles computed over all users.
            out[user_positions_sorted] = user_counts.astype(float)

        counts = out
        if len(counts) == 0:
            return counts

        p50, p90, p95 = np.nanquantile(counts, [0.5, 0.9, 0.95])
        comp = np.zeros_like(counts, dtype=float)
        comp = np.where(counts <= p50, 0.0, comp)
        comp = np.where((counts > p50) & (counts <= p90), 0.5, comp)
        comp = np.where(counts > p90, 1.0, comp)
        return _clamp01(comp)

    # Fallback: overall user activity count relative to global 90th percentile.
    user_counts = df.groupby("user_id", dropna=False).size().rename("user_count")
    p90 = float(user_counts.quantile(0.9)) if len(user_counts) else 1.0
    p90 = max(p90, 1.0)
    per_tx_user_counts = df["user_id"].map(user_counts).fillna(0.0).astype(float).to_numpy()
    comp = per_tx_user_counts / p90
    return _clamp01(comp)


def compute_risk_score(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """
    Risk Scoring Agent (RULE-BASED):
      - Assign a numerical risk score using deterministic features.
    """
    if df is None or df.empty:
        out = pd.DataFrame(
            columns=[
                "tx_id",
                "user_id",
                "amount",
                "timestamp",
                "lat",
                "lon",
                "location_id",
                "risk_score",
                "llm_needed",
                "high_value",
            ]
        )
        return out, 0.0

    df = df.copy()

    amount_comp = _amount_risk_component(df["amount"])
    location_comp = _location_risk_component(df)
    freq_comp = _frequency_risk_component(df)

    # Weighted sum; values are in [0,1].
    risk_score = 100.0 * (0.45 * amount_comp + 0.35 * location_comp + 0.20 * freq_comp)
    risk_score = np.minimum(100.0, np.maximum(0.0, risk_score))
    df["risk_score"] = risk_score.astype(float)

    amt = df["amount"].astype(float).fillna(0.0)
    high_amount_threshold = float(amt.quantile(0.95)) if len(amt) else 0.0
    df["high_value"] = amt.to_numpy() >= high_amount_threshold

    # Optimization: LLM only for medium/high risk or high-value.
    llm_needed = (df["risk_score"].to_numpy() >= 50.0) | df["high_value"].to_numpy()
    df["llm_needed"] = llm_needed

    # Ensure at least one LLM call when input is non-empty.
    if len(df) > 0 and not bool(df["llm_needed"].any()):
        max_idx = int(df["amount"].astype(float).fillna(0.0).idxmax())
        df.loc[max_idx, "llm_needed"] = True
        df.loc[max_idx, "high_value"] = True

    return df, high_amount_threshold


def make_decision(
    df: pd.DataFrame,
    llm_fraud_map: dict[str, bool],
    high_amount_threshold: float,
) -> pd.DataFrame:
    """
    Decision Agent:
      - Combine rule-based score + LLM result
      - Prioritize high-value fraud detection
    """
    if df is None or df.empty:
        out = pd.DataFrame(columns=["tx_id", "is_fraud"])
        return out

    out = df.copy()
    out["llm_fraud"] = out["tx_id"].map(llm_fraud_map)  # NaN where not escalated

    # High-value and very-high risk are considered fraud even if the LLM is absent.
    threshold_high_risk = 80.0
    threshold_medium_risk = 70.0

    amount = out["amount"].astype(float).fillna(0.0).to_numpy()
    risk_score = out["risk_score"].astype(float).fillna(0.0).to_numpy()
    llm_fraud = out["llm_fraud"].to_numpy()

    is_fraud = np.zeros(len(out), dtype=bool)

    # 1) Very high risk overrides.
    is_fraud = is_fraud | (risk_score >= 90.0)

    # 2) LLM overrides when present (for medium/high cases).
    llm_present_mask = pd.notna(out["llm_fraud"]).to_numpy()
    is_fraud = is_fraud | (llm_present_mask & (llm_fraud == True))  # noqa: E712
    # If LLM says NO, keep it NO unless overridden by (1).
    is_fraud = is_fraud & ~((llm_present_mask) & (llm_fraud == False) & ~(risk_score >= 90.0))  # noqa: E712

    # 3) If no LLM result, use a medium threshold.
    no_llm_mask = ~llm_present_mask
    is_fraud = is_fraud | (no_llm_mask & (risk_score >= threshold_medium_risk))

    # 4) Extra priority: high amount with at least medium risk.
    is_fraud = is_fraud | ((amount >= high_amount_threshold) & (risk_score >= 60.0))

    out["is_fraud"] = is_fraud
    return out

