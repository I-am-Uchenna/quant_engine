# Research & Validation Scripts

Reproducible scripts behind `reports/VALIDATION.pdf`. Every number in the report is
produced by one of these. All market data is pulled live from FRED; no data or
credentials are committed.

## Requirements

```bash
pip install -r requirements-dev.txt      # engine deps
pip install QuantLib autograd            # validation deps
export FRED_API_KEY=your_32_char_fred_key
```

Run each script from the repository root (so `quant_engine` is importable):

```bash
PYTHONPATH=. python research/01_quantlib_reconciliation.py
```

## Contents

| Script | Validates | Key result |
| --- | --- | --- |
| `01_quantlib_reconciliation.py` | European conventions + Bermudan vs QuantLib tree + MC convergence | 0.07 bp European; agreement within MC error; 1/sqrt(N) |
| `02_primal_dual_bounds.py` | Andersen-Broadie primal-dual policy bound | duality gap ~0 with adequate value surface |
| `03_risk_suite_fred.py` | DV01, key-rate, vega, exercise distribution, EPE/PFE | on the live FRED curve |
| `04_parameter_estimation_fred.py` | sigma / a from real SOFR history (realized + OU-MLE) | sigma ~75 bp/yr (recent); a not identifiable |
| `05_parameter_price_impact.py` | price under assumed vs estimated sigma | ~25% price impact |
| `06_curve_pca_fred.py` | PCA of Treasury-curve moves | 87% level, 11% slope |
| `07_g2pp_decorrelation.py` | G2++ curve reproduction + implied vs empirical decorrelation | corr(2Y,30Y): 1F=1.00, G2++=0.77, empirical=0.57 |
| `08_g2pp_quantlib_validation.py` | G2++ swaption vs QuantLib FdG2 | 0.10 bp (within MC error) |
| `09_aad_greeks.py` | reverse-mode AAD Greeks vs bump-and-revalue | match to ~1e-6 |
| `10_aad_scaling.py` | AAD cost vs number of risk factors | reverse pass ~3.3x forward, independent of #inputs |

## Data sources

FRED (Federal Reserve Bank of St. Louis): `SOFR`, `SOFR30/90/180DAYAVG`, `DGS1`...`DGS30`.
Benchmark library: QuantLib 1.42.1. Autodiff: autograd (reverse-mode).
