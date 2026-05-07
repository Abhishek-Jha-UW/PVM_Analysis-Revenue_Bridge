from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PvmConfig:
    period_col: str
    sku_col: str
    revenue_col: str
    units_col: str
    region_col: Optional[str] = None
    period_base: str = ""
    period_cmp: str = ""


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else float("nan")


def aggregate_panel(
    df: pd.DataFrame,
    cfg: PvmConfig,
    *,
    period_value: str,
) -> pd.DataFrame:
    """One row per SKU (and region if configured) for a single period."""
    group_cols: List[str] = [cfg.sku_col]
    if cfg.region_col and cfg.region_col in df.columns:
        group_cols.insert(0, cfg.region_col)

    sub = df[df[cfg.period_col].astype(str) == str(period_value)].copy()
    if sub.empty:
        return pd.DataFrame(columns=group_cols + ["revenue", "units", "price_realization"])

    g = sub.groupby(group_cols, dropna=False, as_index=False).agg(
        revenue=(cfg.revenue_col, "sum"),
        units=(cfg.units_col, "sum"),
    )
    g["price_realization"] = g.apply(lambda r: _safe_div(r["revenue"], r["units"]), axis=1)
    return g


def classify_sku_status(
    base: pd.DataFrame, cmp: pd.DataFrame, key_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into ongoing, new (only in cmp), discontinued (only in base)."""
    b_keys = base[key_cols].drop_duplicates()
    c_keys = cmp[key_cols].drop_duplicates()
    merged = b_keys.merge(c_keys, on=key_cols, how="outer", indicator=True)
    ongoing_keys = merged[merged["_merge"] == "both"][key_cols]
    new_keys = merged[merged["_merge"] == "right_only"][key_cols]
    disc_keys = merged[merged["_merge"] == "left_only"][key_cols]

    on_b = base.merge(ongoing_keys, on=key_cols, how="inner")
    on_c = cmp.merge(ongoing_keys, on=key_cols, how="inner")
    new_c = cmp.merge(new_keys, on=key_cols, how="inner")
    disc_b = base.merge(disc_keys, on=key_cols, how="inner")
    return on_b, on_c, new_c, disc_b


def _line_bridge_ongoing(
    base: pd.DataFrame,
    cmp: pd.DataFrame,
    key_cols: List[str],
) -> pd.DataFrame:
    """
    For matching keys, exact accounting identity per row:
        Δ(pq) = q0·Δp + p0·Δq + Δp·Δq
    Labels:
        price_at_base_qty, volume_at_base_price, interaction
    """
    m = base.merge(
        cmp,
        on=key_cols,
        how="inner",
        suffixes=("_0", "_1"),
    )
    p0 = m["price_realization_0"].astype(float)
    p1 = m["price_realization_1"].astype(float)
    q0 = m["units_0"].astype(float)
    q1 = m["units_1"].astype(float)
    r0 = m["revenue_0"].astype(float)
    r1 = m["revenue_1"].astype(float)

    dp = p1 - p0
    dq = q1 - q0
    price_at_base_qty = q0 * dp
    volume_at_base_price = p0 * dq
    interaction = dp * dq
    delta_revenue = r1 - r0

    out = m[key_cols + ["revenue_0", "revenue_1", "units_0", "units_1", "price_realization_0", "price_realization_1"]].copy()
    out["pvm_status"] = "ongoing"
    out["delta_revenue"] = delta_revenue
    out["price_effect"] = price_at_base_qty
    out["volume_effect"] = volume_at_base_price
    out["interaction_effect"] = interaction
    out["new_sku_effect"] = 0.0
    out["discontinued_sku_effect"] = 0.0
    # reconciliation check (should be ~0)
    out["bridge_residual"] = out["delta_revenue"] - (
        out["price_effect"] + out["volume_effect"] + out["interaction_effect"]
    )
    return out


def build_pvm_bridge(
    df: pd.DataFrame,
    cfg: PvmConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Line-level PVM bridge between cfg.period_base and cfg.period_cmp.
    Returns (detail_dataframe, summary_dict).
    """
    if not cfg.period_base or not cfg.period_cmp:
        raise ValueError("Select baseline and comparison periods.")

    key_cols: List[str] = [cfg.sku_col]
    if cfg.region_col and cfg.region_col in df.columns:
        key_cols.insert(0, cfg.region_col)

    base = aggregate_panel(df, cfg, period_value=cfg.period_base)
    cmp = aggregate_panel(df, cfg, period_value=cfg.period_cmp)

    on_b, on_c, new_c, disc_b = classify_sku_status(base, cmp, key_cols)
    ongoing_detail = _line_bridge_ongoing(on_b, on_c, key_cols)

    new_rows: List[Dict[str, Any]] = []
    for _, r in new_c.iterrows():
        rev = float(r["revenue"])
        u = float(r["units"])
        p = float(r["price_realization"])
        new_rows.append(
            {
                **{k: r[k] for k in key_cols},
                "pvm_status": "new",
                "revenue_0": 0.0,
                "revenue_1": rev,
                "units_0": 0.0,
                "units_1": u,
                "price_realization_0": float("nan"),
                "price_realization_1": p,
                "delta_revenue": rev,
                "price_effect": 0.0,
                "volume_effect": 0.0,
                "interaction_effect": 0.0,
                "new_sku_effect": rev,
                "discontinued_sku_effect": 0.0,
                "bridge_residual": 0.0,
            }
        )

    disc_rows: List[Dict[str, Any]] = []
    for _, r in disc_b.iterrows():
        rev = float(r["revenue"])
        u = float(r["units"])
        p = float(r["price_realization"])
        disc_rows.append(
            {
                **{k: r[k] for k in key_cols},
                "pvm_status": "discontinued",
                "revenue_0": rev,
                "revenue_1": 0.0,
                "units_0": u,
                "units_1": 0.0,
                "price_realization_0": p,
                "price_realization_1": float("nan"),
                "delta_revenue": -rev,
                "price_effect": 0.0,
                "volume_effect": 0.0,
                "interaction_effect": 0.0,
                "new_sku_effect": 0.0,
                "discontinued_sku_effect": -rev,
                "bridge_residual": 0.0,
            }
        )

    parts = [ongoing_detail]
    if new_rows:
        parts.append(pd.DataFrame(new_rows))
    if disc_rows:
        parts.append(pd.DataFrame(disc_rows))

    detail = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def _sum(col: str) -> float:
        return float(detail[col].sum()) if col in detail.columns and not detail.empty else 0.0

    summary: Dict[str, Any] = {
        "period_base": cfg.period_base,
        "period_cmp": cfg.period_cmp,
        "revenue_base": float(base["revenue"].sum()) if not base.empty else 0.0,
        "revenue_cmp": float(cmp["revenue"].sum()) if not cmp.empty else 0.0,
        "delta_revenue_total": _sum("delta_revenue"),
        "price_effect_total": _sum("price_effect"),
        "volume_effect_total": _sum("volume_effect"),
        "interaction_effect_total": _sum("interaction_effect"),
        "new_sku_effect_total": _sum("new_sku_effect"),
        "discontinued_sku_effect_total": _sum("discontinued_sku_effect"),
        "ongoing_lines": int((detail["pvm_status"] == "ongoing").sum()) if not detail.empty else 0,
        "new_lines": int((detail["pvm_status"] == "new").sum()) if not detail.empty else 0,
        "discontinued_lines": int((detail["pvm_status"] == "discontinued").sum()) if not detail.empty else 0,
        "max_abs_bridge_residual": float(detail["bridge_residual"].abs().max()) if not detail.empty else 0.0,
    }
    summary["revenue_cmp_check"] = summary["revenue_base"] + summary["delta_revenue_total"]
    return detail, summary


def rollup_by_dimension(
    detail: pd.DataFrame,
    dim_col: str,
) -> pd.DataFrame:
    """Sum PVM components by a column (e.g. region or sku)."""
    if detail.empty or dim_col not in detail.columns:
        return pd.DataFrame()
    g = detail.groupby(dim_col, dropna=False).agg(
        delta_revenue=("delta_revenue", "sum"),
        price_effect=("price_effect", "sum"),
        volume_effect=("volume_effect", "sum"),
        interaction_effect=("interaction_effect", "sum"),
        new_sku_effect=("new_sku_effect", "sum"),
        discontinued_sku_effect=("discontinued_sku_effect", "sum"),
    ).reset_index()
    total_delta = float(detail["delta_revenue"].sum())
    denom = max(abs(total_delta), 1e-9)
    g["pct_of_total_delta"] = g["delta_revenue"] / denom * 100.0
    return g.sort_values("delta_revenue", ascending=False)


def top_movers(detail: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    if detail.empty:
        return detail
    d = detail.copy()
    d["abs_delta"] = d["delta_revenue"].abs()
    return d.sort_values("abs_delta", ascending=False).head(int(n)).drop(columns=["abs_delta"])


def summary_for_llm(summary: Dict[str, Any], rollup_region: Optional[pd.DataFrame], top_lines: pd.DataFrame) -> Dict[str, Any]:
    """Structured payload for OpenAI — numbers computed here only."""
    payload: Dict[str, Any] = {
        "pvm_summary": summary,
        "top_delta_revenue_lines": top_lines.head(10).to_dict(orient="records") if not top_lines.empty else [],
    }
    if rollup_region is not None and not rollup_region.empty:
        payload["rollup_by_region"] = rollup_region.to_dict(orient="records")
    return payload


def generate_sample_panel(
    *,
    seed: int = 42,
    periods: Sequence[str] = ("2024Q1", "2024Q2"),
) -> pd.DataFrame:
    """Synthetic SKU × region panel for demos (two quarters)."""
    rng = np.random.default_rng(seed)
    regions = ["West", "Central", "East"]
    skus = [f"SKU_{i}" for i in list("ABCDEF")]
    rows: List[Dict[str, Any]] = []
    for period in periods:
        t = 0 if period == periods[0] else 1
        for reg in regions:
            for si, sku in enumerate(skus):
                base_u = max(50, int(rng.poisson(280 + si * 8 + (0 if reg == "West" else 15))))
                u = int(base_u * (1.05 + 0.08 * t + 0.02 * rng.standard_normal()))
                base_p = 8.5 + si * 0.35 + (0.4 if reg == "West" else 0)
                p = base_p * (1.0 + 0.03 * t + 0.015 * rng.standard_normal())
                if t == 1 and rng.random() < 0.15:
                    p *= 0.92
                rev = round(float(p * u), 2)
                rows.append(
                    {
                        "period": period,
                        "region": reg,
                        "sku": sku,
                        "revenue": rev,
                        "units": u,
                    }
                )
    # Drop one SKU in Q2 in one region to show discontinued path
    df = pd.DataFrame(rows)
    mask_drop = (df["period"] == periods[1]) & (df["sku"] == "SKU_F") & (df["region"] == "Central")
    df = df.loc[~mask_drop].reset_index(drop=True)
    # New SKU in Q2
    extra = pd.DataFrame(
        [
            {
                "period": periods[1],
                "region": "East",
                "sku": "SKU_NEW",
                "revenue": round(12.4 * 420, 2),
                "units": 420,
            }
        ]
    )
    return pd.concat([df, extra], ignore_index=True)
