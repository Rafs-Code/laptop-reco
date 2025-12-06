# ml2/utils.py  (atau sel notebook)
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path.cwd()  # atur jika perlu
ART = ROOT / "ml2" / "artifacts"
MODEL = joblib.load(ART / "model_randomforest.joblib")
DF = pd.read_csv(ROOT / "ml2" / "datasets" / "laptops.csv")

# --- helper: GPU/CPU tier mapping (bisa dikembangkan) ---
GPU_TIERS = {
    "rtx 40": 5, "rtx 30": 4, "rtx 20": 3, "gtx 16": 2, "gtx 10": 1, "integrated": 0
}
CPU_TIERS = {
    "i9":5, "i7":4, "i5":3, "i3":2, "ryzen 9":5, "ryzen 7":4, "ryzen 5":3, "ryzen 3":2, "m1":4, "m2":5
}

def _extract_tier_from_text(text, mapping):
    t = str(text).lower()
    # try matching the most specific keys first
    for key in mapping:
        if key in t:
            return mapping[key]
    return None

def _keyword_score(text, keywords):
    """Return 1.0 if any strong match, 0.75 for partial, 0.0 otherwise."""
    if not keywords:
        return 0.5  # neutral
    txt = str(text).lower()
    score = 0.0
    for kw in keywords:
        kw = kw.lower().strip()
        if not kw:
            continue
        if kw in txt:
            # exact-ish match -> 1.0
            return 1.0
        # partial (e.g. 'rtx' matches 'rtx 3050') handled by above; keep fallback small
        if kw.split()[0] in txt:
            score = max(score, 0.7)
    return score

def _ram_step_score(ram_gb, min_ram):
    """Step scoring: 0 if below half requirement, 0.6 if between half and min, 1 if >= min"""
    if min_ram <= 0:
        return 0.5
    if ram_gb >= min_ram:
        return 1.0
    if ram_gb >= min_ram / 2:
        return 0.6
    return 0.0

def recommend_laptops(df=DF, user_need: dict=None, budget: float=None, top_n: int=5,
                       weights=None):
    """
    Improved recommendation function.
    """
    if user_need is None:
        user_need = {}
    if weights is None:
        weights = {"price":0.4, "gpu":0.3, "cpu":0.15, "ram":0.15}

    table = df.copy().reset_index(drop=True)

    # predict prices robustly
    try:
        preds = MODEL.predict(table.drop(columns=["price_usd"]))
        table["_pred_price"] = preds.astype(float)
    except Exception:
        table["_pred_price"] = table["price_usd"].astype(float)

    min_ram = int(user_need.get("min_ram", 0))
    prefer_gpu = user_need.get("prefer_gpu", [])
    prefer_cpu = user_need.get("prefer_cpu", [])

    # ensure columns exist
    for c in ["ram_gb","gpu","cpu"]:
        if c not in table.columns:
            table[c] = "" if c!="ram_gb" else 0

    # compute component scores
    table["_ram_score"] = table["ram_gb"].apply(lambda r: _ram_step_score(r, min_ram))
    table["_gpu_score"] = table["gpu"].apply(lambda g: _keyword_score(g, prefer_gpu))
    table["_cpu_score"] = table["cpu"].apply(lambda c: _keyword_score(c, prefer_cpu))

    # Tier-boost: if GPU/CPU falls into high-tier mapping, slightly increase score
    def tier_boost(row):
        boost = 0.0
        gt = _extract_tier_from_text(row["gpu"], GPU_TIERS)
        ct = _extract_tier_from_text(row["cpu"], CPU_TIERS)
        if gt is not None and gt >= 4:
            boost += 0.05
        if ct is not None and ct >= 4:
            boost += 0.03
        return boost
    table["_tier_boost"] = table.apply(tier_boost, axis=1)

    # price scoring: normalized closeness (robust)
    if budget:
        # use log scale to reduce impact of big absolute diffs (optional)
        p = table["_pred_price"]
        # avoid zero/negative
        p = p.replace(0, 1.0)
        # distance normalized by (max-min) but with clipping
        span = max(p.max(), budget) - min(p.min(), budget)
        span = max(span, 1.0)
        table["_price_score"] = table["_pred_price"].apply(lambda x: 1.0 - min(1.0, abs(x - budget) / span))
    else:
        table["_price_score"] = 0.5

    # combine
    table["_score_raw"] = (
        weights["price"] * table["_price_score"] +
        weights["gpu"] * table["_gpu_score"] +
        weights["cpu"] * table["_cpu_score"] +
        weights["ram"] * table["_ram_score"]
    )
    table["_score"] = (table["_score_raw"] + table["_tier_boost"]).clip(0.0, 1.0)

    # reason
    def make_reason(row):
        r = []
        r.append(f"Price ~ ${row['_pred_price']:.0f} (data ${row['price_usd']})")
        if row["_gpu_score"] >= 0.99:
            r.append("GPU matches preference")
        elif row["_gpu_score"] >= 0.7:
            r.append("GPU partly suitable")
        else:
            r.append("GPU not prioritized")
        if row["_cpu_score"] >= 0.99:
            r.append("CPU matches preference")
        elif row["_cpu_score"] >= 0.7:
            r.append("CPU partly suitable")
        else:
            r.append("CPU not prioritized")
        if row["_ram_score"] >= 0.99:
            r.append(f"RAM >= {min_ram}GB")
        elif row["_ram_score"] >= 0.6:
            r.append(f"RAM ok ({int(row['ram_gb'])}GB)")
        else:
            r.append(f"Low RAM ({int(row['ram_gb'])}GB)")
        return "; ".join(r)

    table["reason"] = table.apply(make_reason, axis=1)

    out_cols = ["brand","model","cpu","gpu","ram_gb","storage_gb","price_usd","_pred_price","_score","reason"]
    result = table.sort_values("_score", ascending=False).head(top_n)[out_cols].reset_index(drop=True)
    result = result.rename(columns={"_pred_price":"pred_price","_score":"score"})
    result["score"] = result["score"].round(3)
    result["pred_price"] = result["pred_price"].round(2)
    return result
