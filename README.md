# Quant Engine: Hull-White Bermudan Swaption Pricing Framework

Production-grade Bermudan swaption pricing framework with a C++17 quantitative core, pybind11 bindings, authenticated FRED market-data ingestion, distributed Ray execution, gRPC service delivery, Excel integration, Streamlit visualization, and cross-platform wheel builds.

## Executive Summary

This repository implements an end-to-end short-rate derivatives platform centered on the one-factor Hull-White model. The architecture separates low-level stochastic numerics from orchestration and delivery layers:

- **C++17 analytics core** for exact Hull-White simulation, Sobol quasi-Monte Carlo, Brownian bridge construction, and Longstaff-Schwartz regression.
- **Python infrastructure layer** for market-data ingestion, Ray distribution, gRPC microservice delivery, and Excel UDF access.
- **Mathematical invariant tests** covering martingale consistency, Jamshidian convergence, and Bermudan monotonicity.
- **Streamlit dashboard** for live pricing, quantitative reporting, and simulated path visualization.
- **GitHub Actions CI/CD** using cibuildwheel to build and test wheels on Linux, Windows, and macOS.

## Repository Layout

```text
.
|-- .github/workflows/build.yml       # Cross-platform wheel build and invariant test pipeline
|-- .streamlit/                       # Streamlit Cloud configuration and secret template
|-- cpp_core/
|   |-- include/                      # Typed C++ public interfaces
|   `-- src/                          # C++17 implementation and pybind11 bindings
|-- python_layer/
|   |-- data_io.py                    # FRED API + DuckDB market-data ingestion
|   |-- ray_orchestrator.py           # Stateless distributed pricing orchestration
|   |-- grpc_service.py               # Async gRPC pricing service
|   `-- xlwings_udf.py                # Excel UDF integration
|-- tests/test_invariants.py          # Mathematical verification suite
|-- app.py                            # Streamlit dashboard
|-- CMakeLists.txt                    # Native build definition
|-- pyproject.toml                    # PEP 517 wheel build configuration
`-- requirements.txt                  # Runtime, dashboard, and CI dependencies
```

## Quantitative Model

The short rate follows the Hull-White one-factor risk-neutral process:

```math
dr_t = (\theta(t) - a r_t)dt + \sigma dW_t
```

The simulator uses the exact Gaussian transition density:

```math
\mathbb{E}[r_t|\mathcal{F}_s]
= r_s e^{-a(t-s)} + \alpha(t)-\alpha(s)e^{-a(t-s)}
```

```math
\operatorname{Var}(r_t|\mathcal{F}_s)
= \frac{\sigma^2}{2a}\left(1-e^{-2a(t-s)}\right)
```

Early exercise is solved with Longstaff-Schwartz backward induction using weighted Laguerre basis functions and ridge-regularized SVD regression.

## Market Data

The production data path requires a FRED API key. The loader uses:

- FRBNY SOFR and SOFR averages via FRED.
- Federal Reserve H.15 Treasury constant maturity rates via FRED.
- Curve-implied forward-start par swap rates derived from the downloaded curve.

Create a local `.env` file:

```bash
FRED_API_KEY=your_fred_api_key
```

For Streamlit Cloud, configure the same value under **App settings -> Secrets**:

```toml
FRED_API_KEY = "your_fred_api_key"
```

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Build the native extension:

```bash
python -m build --wheel
python -m pip install dist/*.whl
```

Run the mathematical verification suite:

```bash
pytest tests/test_invariants.py -q
```

Run the dashboard:

```bash
streamlit run app.py
```

## Streamlit Cloud Deployment

1. Push this repository to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Select **New app**.
4. Choose this repository and branch `main`.
5. Set the app entrypoint to:

```text
app.py
```

6. Add the FRED API key to Streamlit secrets:

```toml
FRED_API_KEY = "your_fred_api_key"
```

7. Deploy. The platform installs `requirements.txt`, builds the native extension from `pyproject.toml`, and launches the dashboard.

## CI/CD

GitHub Actions builds wheels with `pypa/cibuildwheel` for:

- Ubuntu
- Windows
- macOS
- Python 3.10, 3.11, and 3.12

Each wheel is tested against `tests/test_invariants.py` before artifact upload. Tagged releases publish verified wheels to GitHub Releases.

## Validation Standard

The invariant suite is intentionally model-facing rather than superficial:

- **Martingale property:** verifies discounted zero-coupon bond prices under risk-neutral simulation.
- **Jamshidian convergence:** verifies single-exercise Bermudan valuation against the analytical European swaption reduction.
- **Exercise monotonicity:** verifies the Bermudan value does not decrease as exercise opportunities increase.

## Repository Description

Suggested GitHub description:

```text
Production-grade Hull-White Bermudan swaption pricing engine with C++17 analytics, pybind11, FRED/DuckDB data ingestion, Ray distribution, gRPC delivery, mathematical invariants, CI wheels, and Streamlit visualization.
```
