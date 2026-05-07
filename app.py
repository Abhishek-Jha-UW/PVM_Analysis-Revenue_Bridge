from __future__ import annotations

import io
import json
import os
from typing import Any, Dict, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model import (
    PvmConfig,
    build_pvm_bridge,
    generate_sample_panel,
    rollup_by_dimension,
    summary_for_llm,
    top_movers,
)

APP_TITLE = "PVM Analysis — Revenue Bridge"

DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"


def _get_api_key() -> str:
    if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
        k = str(st.secrets["OPENAI_API_KEY"]).strip()
        if k:
            return k
    return os.environ.get("OPENAI_API_KEY", "").strip()


def _openai_client(api_key: str):
    from openai import OpenAI

    return OpenAI(api_key=api_key)


def narrative_with_ai(*, api_key: str, model_name: str, payload: Dict[str, Any]) -> str:
    client = _openai_client(api_key=api_key)
    system = (
        "You are a senior revenue / FP&A analyst.\n"
        "Write an executive summary of the price–volume–mix (PVM) bridge using ONLY the JSON payload.\n"
        "Do not invent numbers, SKUs, or regions. Cite drivers using names from the payload.\n"
        "Cover: headline revenue change, main price vs volume vs interaction vs new/discontinued effects, "
        "and 2–3 caveats (definitions, mix vs interaction, data grain).\n"
    )
    user = "PVM computed results (JSON):\n\n" + json.dumps(payload, indent=2)
    resp = client.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
    )
    text = getattr(resp, "output_text", None)
    if not text:
        raise RuntimeError("Empty response from model.")
    return text.strip()


def _load_uploaded(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded)
    raise ValueError("Upload a CSV or XLSX file.")


def _waterfall_fig(summary: Dict[str, Any]) -> go.Figure:
    base = float(summary["revenue_base"])
    p = float(summary["price_effect_total"])
    v = float(summary["volume_effect_total"])
    x = float(summary["interaction_effect_total"])
    n = float(summary["new_sku_effect_total"])
    d = float(summary["discontinued_sku_effect_total"])
    end = float(summary["revenue_cmp"])

    measures = ["absolute", "relative", "relative", "relative", "relative", "relative", "total"]
    labels = [
        f"Revenue ({summary['period_base']})",
        "Price (at base qty)",
        "Volume (at base price)",
        "Interaction",
        "New SKUs",
        "Discontinued SKUs",
        f"Revenue ({summary['period_cmp']})",
    ]
    ys = [base, p, v, x, n, d, end]
    fig = go.Figure(
        go.Waterfall(
            name="PVM",
            orientation="v",
            measure=measures,
            x=labels,
            y=ys,
            connector={"line": {"color": "rgb(100, 100, 100)"}},
        )
    )
    fig.update_layout(
        title="Revenue bridge (accounting identity on ongoing lines)",
        showlegend=False,
        height=480,
        margin=dict(t=50, b=120),
    )
    return fig


st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption(
    "Line-level revenue bridge: Δ(pq) = q₀·Δp + p₀·Δq + Δp·Δq on ongoing SKU keys, "
    "plus explicit new / discontinued buckets. Numbers are auditable in the detail table."
)

with st.sidebar:
    st.subheader("Data")
    use_sample = st.toggle("Use sample panel (demo)", value=True)
    upload: Optional[Any] = None
    if not use_sample:
        upload = st.file_uploader("Upload CSV / XLSX", type=["csv", "xlsx"])

    st.subheader("OpenAI (optional)")
    use_ai = st.toggle("AI executive narrative", value=True)
    ai_model = st.text_input("Model", value=DEFAULT_OPENAI_MODEL)

    st.markdown("---")
    st.caption("Set `OPENAI_API_KEY` in Streamlit secrets or env for narratives.")

if use_sample:
    if "pvm_df" not in st.session_state or st.session_state.get("pvm_source") != "sample":
        st.session_state["pvm_df"] = generate_sample_panel()
        st.session_state["pvm_source"] = "sample"
else:
    if upload is not None:
        try:
            st.session_state["pvm_df"] = _load_uploaded(upload)
            st.session_state["pvm_source"] = "upload"
        except Exception as e:
            st.error(str(e))

df: Optional[pd.DataFrame] = st.session_state.get("pvm_df")

if df is None or df.empty:
    st.info("Load the sample panel or upload a file to begin.")
    st.stop()

st.subheader("Column mapping")
c1, c2, c3, c4, c5 = st.columns(5)
cols = list(df.columns)
with c1:
    period_col = st.selectbox("Period", options=cols, index=cols.index("period") if "period" in cols else 0)
with c2:
    sku_col = st.selectbox("SKU / product key", options=cols, index=cols.index("sku") if "sku" in cols else 0)
with c3:
    revenue_col = st.selectbox("Revenue", options=cols, index=cols.index("revenue") if "revenue" in cols else 0)
with c4:
    units_col = st.selectbox("Units", options=cols, index=cols.index("units") if "units" in cols else 0)
with c5:
    region_opts = ["(none)"] + cols
    default_r = "region" if "region" in cols else "(none)"
    r_ix = region_opts.index(default_r) if default_r in region_opts else 0
    region_pick = st.selectbox("Region (optional)", options=region_opts, index=r_ix)

region_col = None if region_pick == "(none)" else region_pick

periods = sorted(df[period_col].astype(str).unique().tolist())
if len(periods) < 2:
    st.error("Need at least two distinct periods in the data for a bridge.")
    st.stop()

pc1, pc2 = st.columns(2)
with pc1:
    period_base = st.selectbox("Baseline period", options=periods, index=0)
with pc2:
    default_cmp = 1 if len(periods) > 1 else 0
    period_cmp = st.selectbox("Comparison period", options=periods, index=min(default_cmp, len(periods) - 1))

if period_base == period_cmp:
    st.warning("Pick two different periods.")
    st.stop()

cfg = PvmConfig(
    period_col=period_col,
    sku_col=sku_col,
    revenue_col=revenue_col,
    units_col=units_col,
    region_col=region_col,
    period_base=str(period_base),
    period_cmp=str(period_cmp),
)

if st.button("Run PVM bridge", type="primary"):
    try:
        detail, summary = build_pvm_bridge(df, cfg)
        st.session_state["pvm_detail"] = detail
        st.session_state["pvm_summary"] = summary
    except Exception as e:
        st.error(str(e))

detail = st.session_state.get("pvm_detail")
summary = st.session_state.get("pvm_summary")

if summary and detail is not None:
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Revenue (base)", f"{summary['revenue_base']:,.0f}")
    s2.metric("Revenue (cmp)", f"{summary['revenue_cmp']:,.0f}")
    s3.metric("Δ Revenue", f"{summary['delta_revenue_total']:,.0f}")
    chk = abs(float(summary["revenue_cmp_check"]) - float(summary["revenue_cmp"]))
    s4.metric("Reconciliation |Δ|", f"{chk:,.2f}")

    st.plotly_chart(_waterfall_fig(summary), use_container_width=True)

    tab_detail, tab_region, tab_ai = st.tabs(["Line detail", "Rollup", "AI narrative"])

    with tab_detail:
        st.dataframe(
            detail.round(2),
            use_container_width=True,
            hide_index=True,
        )
        csv_buf = io.StringIO()
        detail.to_csv(csv_buf, index=False)
        st.download_button(
            "Download line-level bridge (CSV)",
            data=csv_buf.getvalue().encode("utf-8"),
            file_name="pvm_line_bridge.csv",
            mime="text/csv",
        )

        st.markdown("**Top movers (|Δ revenue|)**")
        st.dataframe(top_movers(detail, 20).round(2), use_container_width=True, hide_index=True)

    with tab_region:
        if region_col:
            rr = rollup_by_dimension(detail, region_col)
            st.dataframe(rr.round(2), use_container_width=True, hide_index=True)
            if not rr.empty:
                st.bar_chart(rr.set_index(region_col)["delta_revenue"])
        else:
            rs = rollup_by_dimension(detail, sku_col)
            st.caption("No region column — showing roll-up by SKU.")
            st.dataframe(rs.head(40).round(2), use_container_width=True, hide_index=True)

    with tab_ai:
        if not use_ai:
            st.info("Enable AI narrative in the sidebar.")
        else:
            key = _get_api_key()
            if not key:
                st.warning("Add `OPENAI_API_KEY` to secrets to generate a narrative.")
            else:
                rr_df = rollup_by_dimension(detail, region_col) if region_col else None
                payload = summary_for_llm(summary, rr_df, top_movers(detail, 15))
                if st.button("Generate narrative"):
                    with st.spinner("Drafting…"):
                        try:
                            st.session_state["pvm_ai_text"] = narrative_with_ai(
                                api_key=key, model_name=ai_model.strip(), payload=payload
                            )
                        except Exception as e:
                            st.error(str(e))
                if st.session_state.get("pvm_ai_text"):
                    st.markdown(st.session_state["pvm_ai_text"])

    with st.expander("Methodology (read me)"):
        st.markdown(
            """
- **Price (at base qty)** · Σ q₀(p₁−p₀) on ongoing keys.
- **Volume (at base price)** · Σ p₀(q₁−q₀).
- **Interaction** · Σ Δp·Δq (joint price–quantity moves; often grouped with “mix” in conversation — here explicit).
- **New / discontinued** · Full incremental or lost revenue for keys missing in base or comparison period.
- **Price realization** · revenue ÷ units within each period after your aggregation grain.

Residual on ongoing lines should be ~0 (floating point). If not, check duplicate keys in the upload.
            """
        )

elif df is not None:
    st.caption("Map columns and click **Run PVM bridge**.")
