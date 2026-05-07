from __future__ import annotations

import io
import json
import os
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

from model import (
    PvmConfig,
    build_pvm_bridge,
    executive_pvm_headline,
    executive_pvm_table,
    generate_sample_panel,
    pvm_input_template,
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
        "Lead with explicit **Price impact ($)**, **Volume impact ($)**, and **Mix impact ($)** from executive_pvm_usd / executive_pvm_table. "
        "Clarify that mix here is the Cartesian cross-term ΣΔp·Δq (standard PVM third bucket), not a separate econometric mix index.\n"
        "Cover new vs discontinued buckets and 2–3 caveats (data grain, net revenue definition, duplicate keys).\n"
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


def _waterfall_matplotlib(summary: Dict[str, Any]) -> Figure:
    """Compact Matplotlib waterfall (total width controlled via figsize)."""
    base = float(summary["revenue_base"])
    p = float(summary["price_effect_total"])
    v = float(summary["volume_effect_total"])
    x = float(summary["interaction_effect_total"])
    n = float(summary["new_sku_effect_total"])
    d = float(summary["discontinued_sku_effect_total"])
    end = float(summary["revenue_cmp"])

    step_labels = [
        f"Start\n{summary['period_base']}",
        "Price",
        "Volume",
        "Mix",
        "New SKUs",
        "Disc. SKUs",
        f"End\n{summary['period_cmp']}",
    ]
    deltas = [p, v, x, n, d]
    color_total = "#3b6ea5"
    color_up = "#2d8a54"
    color_dn = "#c43c39"

    bottoms: list[float] = [0.0]
    heights: list[float] = [base]
    colors: list[str] = [color_total]

    running = base
    for delta in deltas:
        if delta >= 0:
            bottoms.append(running)
            heights.append(delta)
            colors.append(color_up)
        else:
            bottoms.append(running + delta)
            heights.append(-delta)
            colors.append(color_dn)
        running += delta

    bottoms.append(0.0)
    heights.append(end)
    colors.append(color_total)

    fig, ax = plt.subplots(figsize=(8.2, 3.35), dpi=110)
    x_pos = range(len(step_labels))
    ax.bar(x_pos, heights, bottom=bottoms, color=colors, edgecolor="#333333", linewidth=0.4, width=0.72)
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(step_labels, fontsize=8, rotation=0)
    ax.axhline(0, color="#999999", linewidth=0.6, linestyle="-")
    ax.set_ylabel("Revenue ($)")
    ax.set_title("Revenue bridge — Price · Volume · Mix (+ new / discontinued)", fontsize=10.5, pad=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:,.0f}"))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.22)
    return fig


st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption(
    "Cartesian PVM on ongoing lines: **Δ(pq) = q₀·Δp + p₀·Δq + Δp·Δq** (Price + Volume + **Mix** cross-term), "
    "plus **new** / **discontinued** keys. Below, dollar impacts are stated explicitly for leadership."
)

with st.sidebar:
    st.subheader("Data")
    st.caption("Upload a file **or** use the sample panel when no file is selected.")
    upload = st.file_uploader("Upload CSV / XLSX", type=["csv", "xlsx"], accept_multiple_files=False)
    use_sample = st.toggle("Use sample panel when no upload", value=True)

    st.subheader("OpenAI (optional)")
    use_ai = st.toggle("AI executive narrative", value=True)
    ai_model = st.text_input("Model", value=DEFAULT_OPENAI_MODEL)

    st.markdown("---")
    st.caption("Set `OPENAI_API_KEY` in Streamlit secrets or env for narratives.")

if upload is not None:
    try:
        st.session_state["pvm_df"] = _load_uploaded(upload)
        st.session_state["pvm_source"] = "upload"
    except Exception as e:
        st.error(str(e))
elif use_sample:
    if "pvm_df" not in st.session_state or st.session_state.get("pvm_source") != "sample":
        st.session_state["pvm_df"] = generate_sample_panel()
        st.session_state["pvm_source"] = "sample"
else:
    st.session_state["pvm_df"] = None
    st.session_state["pvm_source"] = None

df: Optional[pd.DataFrame] = st.session_state.get("pvm_df")

if df is None or df.empty:
    st.info("Upload a CSV/XLSX, or turn on **Use sample panel when no upload** in the sidebar.")
    st.stop()

src = st.session_state.get("pvm_source", "unknown")
st.caption(f"**Active dataset:** {'uploaded file' if src == 'upload' else 'built-in sample panel'} · {len(df):,} rows")

tpl = pvm_input_template()
st.download_button(
    label="Download CSV template (period, region, sku, revenue, units)",
    data=tpl.to_csv(index=False).encode("utf-8"),
    file_name="pvm_input_template.csv",
    mime="text/csv",
    help="Fill with your periods and keys; duplicate keys in the same period are summed automatically.",
)

with st.expander("Data preview (raw input)", expanded=False):
    st.dataframe(df.head(150), use_container_width=True, hide_index=True)
    st.caption("Analysis uses this table after column mapping and period filters. Aggregates duplicate keys by summing revenue and units.")

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

run_col, insight_col = st.columns(2)
with run_col:
    run_clicked = st.button("Run PVM bridge", type="primary", use_container_width=True)
with insight_col:
    insight_clicked = st.button(
        "Generate insights",
        type="primary",
        use_container_width=True,
        disabled=not bool(st.session_state.get("pvm_summary")),
        help="Runs after **Run PVM bridge**. Uses your OpenAI key and computed results only.",
    )

if run_clicked:
    try:
        detail, summary = build_pvm_bridge(df, cfg)
        st.session_state["pvm_detail"] = detail
        st.session_state["pvm_summary"] = summary
        st.session_state["pvm_ai_text"] = None
    except Exception as e:
        st.error(str(e))

if insight_clicked:
    if not use_ai:
        st.warning("Turn on **AI executive narrative** in the sidebar.")
    else:
        key = _get_api_key()
        if not key:
            st.warning("Add `OPENAI_API_KEY` to Streamlit secrets or your environment.")
        else:
            d0 = st.session_state.get("pvm_detail")
            s0 = st.session_state.get("pvm_summary")
            if isinstance(d0, pd.DataFrame) and s0:
                rr_df = rollup_by_dimension(d0, region_col) if region_col else None
                payload = summary_for_llm(s0, rr_df, top_movers(d0, 15))
                with st.spinner("Generating insights…"):
                    try:
                        st.session_state["pvm_ai_text"] = narrative_with_ai(
                            api_key=key, model_name=ai_model.strip(), payload=payload
                        )
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

    st.subheader("Executive PVM summary (USD)")
    st.markdown(executive_pvm_headline(summary))
    exec_tbl = executive_pvm_table(summary)
    disp = exec_tbl.copy()
    disp["Amount ($)"] = disp["Amount ($)"].map(lambda v: f"{v:,.2f}")

    def _fmt_pct(v: Any) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        return f"{float(v):.1f}%"

    disp["% of |total Δ|"] = disp["% of |total Δ|"].map(_fmt_pct)
    st.dataframe(disp, use_container_width=True, hide_index=True)
    st.caption(
        "**Mix** = Σ (p₁−p₀)(q₁−q₀) on ongoing lines (captures joint price–quantity moves; "
        "in many board decks this bucket is still called *mix* or *cross*). "
        "It is not a separate share-shift index."
    )

    wf_col_left, wf_col_right = st.columns([1.05, 0.95])
    with wf_col_left:
        fig = _waterfall_matplotlib(summary)
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    if st.session_state.get("pvm_ai_text"):
        st.markdown("**AI insights**")
        st.markdown(st.session_state["pvm_ai_text"])

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
        st.caption("Use the **Generate insights** button above (same style as **Run PVM bridge**). Narratives use only computed JSON from your bridge.")
        if st.session_state.get("pvm_ai_text"):
            st.markdown(st.session_state["pvm_ai_text"])
        elif not use_ai:
            st.info("Enable **AI executive narrative** in the sidebar.")
        elif not _get_api_key():
            st.info("Add `OPENAI_API_KEY` to secrets or environment.")

    with st.expander("Methodology (read me)"):
        st.markdown(
            """
**Cartesian PVM (ongoing keys)** — exact identity per line:  
Δ(pq) = **q₀·Δp** + **p₀·Δq** + **Δp·Δq**

| Label in this app | Formula | Typical FP&A name |
|-------------------|---------|-------------------|
| **Price impact** | Σ q₀·Δp | Price (Laspeyres-style, base qty) |
| **Volume impact** | Σ p₀·Δq | Volume (base price) |
| **Mix impact** | Σ Δp·Δq | Cross / interaction / sometimes “mix” in decks |

**New / discontinued** · Full incremental or lost revenue for keys that exist in only one period.

**Price realization** · revenue ÷ units after aggregation to your chosen grain.

**Template** · Download **CSV template** above; columns are `period`, `region` (optional if you map “(none)”), `sku`, `revenue`, `units`.

Residual on ongoing lines should be ~0 (floating point). If not, check duplicate keys in the upload.
            """
        )

elif df is not None:
    st.caption("Map columns and click **Run PVM bridge**.")
