# PVM Analysis — Revenue Bridge

Professional-style **price–volume–mix (PVM)** tool for **revenue bridges** at a configurable grain (SKU × optional region × period). Built for analysts who need **auditable** decomposition, charts, exports, and an optional **OpenAI** narrative that uses **computed JSON only** (no invented figures).

## Backup

Prior `app.py` / `model.py` snapshots before the executive PVM refresh live in **`Backup/`**.

## What it does

1. **Ingest** weekly/monthly/quarterly panel data: period, SKU, revenue, units, optional region. Use the in-app **CSV template** or `data/pvm_input_template.csv`.
2. **Aggregate** to one row per `(region?, sku)` per period; **price realization** = revenue ÷ units.
3. **Bridge** comparison vs baseline period (Cartesian PVM on ongoing keys):
   - **Price impact** · Σ q₀(p₁−p₀)
   - **Volume impact** · Σ p₀(q₁−q₀)
   - **Mix impact** · Σ Δp·Δq (cross-term; in many decks still called *mix* or *cross*)
   - **New SKUs** / **discontinued SKUs** · full incremental or lost revenue for keys only in one period
4. **Visualize** a compact Matplotlib revenue bridge and roll-ups; **export** line-level CSV.
5. **Optional AI** · executive summary from the structured bridge output (requires `OPENAI_API_KEY`).

## Run locally

```bash
cd "03. PVM Analysis - Revenue Bridge"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Cloud)

- Entrypoint: `app.py`
- Secret: `OPENAI_API_KEY` (optional, for narratives)

## Input shape

| Column   | Role                          |
|----------|-------------------------------|
| period   | Baseline vs comparison labels |
| sku      | Product identifier          |
| revenue  | Net or gross — be consistent |
| units    | Same unit across SKUs       |
| region   | Optional second bridge key   |

Multiple raw rows per period/SKU/region are **summed** before the bridge.

## Old folder

The previous **Price Pack & Promo Simulator** scaffold was removed in favor of this PVM project.
