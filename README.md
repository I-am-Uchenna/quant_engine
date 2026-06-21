# Quant Engine — Hull-White Bermudan Swaption Pricer

An end-to-end Bermudan swaption pricing framework built on the one-factor
Hull-White short-rate model. It pairs a C++17 numerical core with a Python
infrastructure layer for market data, distributed execution, service delivery,
Excel access, and an interactive Streamlit dashboard.

The pricer ships with **two interchangeable backends**:

- a compiled **C++17 core** (via pybind11) for production-grade speed, and
- a pure-**Python reference engine** with identical model mathematics.

The dashboard automatically uses the C++ core when it has been built and
falls back to the Python engine otherwise. This means the app runs anywhere —
including free hosts such as Streamlit Community Cloud, where native code is not
compiled — without any extra setup.

## Highlights

- **C++17 analytics core** — exact Hull-White simulation, Sobol quasi-Monte
  Carlo, Brownian-bridge construction, and Longstaff-Schwartz regression.
- **Python reference engine** — the same model in NumPy/SciPy, used as an
  automatic fallback so pricing never depends on a compiler being present.
- **Python infrastructure** — FRED market-data ingestion, Ray distribution,
  an async gRPC service, and an Excel UDF.
- **Mathematical invariant tests** — martingale consistency, Jamshidian
  convergence, and Bermudan exercise monotonicity.
- **Streamlit dashboard** — live pricing, quantitative reporting, and simulated
  short-rate path visualization.
- **GitHub Actions CI** — cibuildwheel builds and tests wheels on Linux,
  Windows, and macOS.

## Repository Layout

```text
.
|-- .github/workflows/build.yml       # Cross-platform wheel build and invariant test pipeline
|-- .streamlit/                       # Streamlit configuration and secret template
|-- cpp_core/
|   |-- include/                      # Typed C++ public interfaces
|   `-- src/                          # C++17 implementation and pybind11 bindings
|-- python_layer/
|   |-- data_io.py                    # FRED API + DuckDB market-data ingestion; backend loader
|   |-- python_engine.py              # Pure-Python Hull-White reference pricing engine
|   |-- ray_orchestrator.py           # Stateless distributed pricing orchestration
|   |-- grpc_service.py               # Async gRPC pricing service
|   `-- xlwings_udf.py                # Excel UDF integration
|-- tests/test_invariants.py          # Mathematical verification suite
|-- app.py                            # Streamlit dashboard
|-- CMakeLists.txt                    # Native build definition
|-- pyproject.toml                    # PEP 517 wheel build configuration
|-- requirements.txt                  # Dashboard / runtime dependencies (Streamlit Cloud)
`-- requirements-dev.txt              # Full-stack build, distributed, and test dependencies
```

## Quantitative Model

The short rate follows the Hull-White one-factor risk-neutral process:

```math
dr_t = (\theta(t) - a\,r_t)\,dt + \sigma\,dW_t
```

The simulator uses the exact Gaussian transition density:

```math
\mathbb{E}[r_t \mid \mathcal{F}_s]
= r_s e^{-a(t-s)} + \alpha(t) - \alpha(s)e^{-a(t-s)}
```

```math
\operatorname{Var}(r_t \mid \mathcal{F}_s)
= \frac{\sigma^2}{2a}\left(1 - e^{-2a(t-s)}\right)
```

Early exercise is solved by Longstaff-Schwartz backward induction using
weighted Laguerre basis functions and a ridge-regularized normal-equations
solve. Both backends implement the same curve construction (natural cubic
spline), discount-bond reconstruction, and Jamshidian European reduction.

### Pricing Backends

| Backend | Module | Speed | Requires a compiler |
| --- | --- | --- | --- |
| C++17 core | `quant_engine_cpp` (built from `cpp_core/`) | Fast | Yes |
| Python reference | `python_layer/python_engine.py` | Slower | No |

`python_layer.data_io.load_cpp_engine()` resolves the backend in this order:
an installed `quant_engine_cpp` extension, a `quant_engine_cpp` found in a local
`build*` directory, and finally the Python reference engine. The dashboard
displays which backend is active. Quasi-Monte Carlo draws in the Python engine
come from `scipy.stats.qmc.Sobol`, so its prices match the C++ core in
expectation (within Monte Carlo error) rather than draw-for-draw.

## Market Data

Live pricing uses the published US rates curve from FRED and requires a free
FRED API key. The loader reads:

- FRBNY SOFR and SOFR averages,
- Federal Reserve H.15 Treasury constant-maturity rates, and
- curve-implied forward-start par swap rates derived from the downloaded curve.

Create a local `.env` file (it is git-ignored):

```bash
FRED_API_KEY=your_fred_api_key
```

Use only the 32-character key value — not the FRED account URL, a Markdown
link, the `FRED_API_KEY=` prefix, or any placeholder text. For Streamlit Cloud,
add the same value under **App settings → Secrets** as `FRED_API_KEY`. If no key
is configured, the dashboard prompts for one on first load and stores it in the
local `.env` file.

## Local Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
python -m pip install --upgrade pip
```

For the dashboard only:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

This runs against the Python reference engine — no build step required.

### Optional: build the C++ core for maximum speed

```bash
python -m pip install -r requirements-dev.txt
python -m build --wheel
python -m pip install dist/*.whl
```

Once `quant_engine_cpp` is importable, the dashboard and the distributed and
gRPC paths use it automatically.

Run the mathematical verification suite (requires the compiled core):

```bash
pytest tests/test_invariants.py -q
```

## Streamlit Cloud Deployment

1. Push this repository to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Select **New app** and choose this repository and branch `main`.
4. Set the app entrypoint to `app.py`.
5. Add `FRED_API_KEY` under **App settings → Secrets**:

   ```toml
   FRED_API_KEY = "your_fred_api_key"
   ```

6. Deploy.

Streamlit Cloud installs `requirements.txt` and launches the dashboard against
the Python reference engine. It does **not** compile the C++ core; the native
extension is for local and CI builds where a compiler is available.

## CI/CD

GitHub Actions builds wheels with `pypa/cibuildwheel` for Ubuntu, Windows, and
macOS on Python 3.10, 3.11, and 3.12. Each wheel is tested against
`tests/test_invariants.py` before artifact upload, and tagged releases publish
the verified wheels to GitHub Releases.

## Validation Standard

The invariant suite is model-facing rather than superficial:

- **Martingale property** — discounted zero-coupon bond prices are unbiased
  under the risk-neutral simulation.
- **Jamshidian convergence** — single-exercise Bermudan valuation matches the
  analytical European swaption reduction.
- **Exercise monotonicity** — Bermudan value does not decrease as exercise
  opportunities increase.

The Python reference engine satisfies the same three invariants, with the
single-exercise LSMC price reproducing the Jamshidian value to machine
precision in the deterministic (`sigma = 0`) case.
