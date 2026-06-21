# Quant Engine — Hull-White Bermudan Swaption Pricer

Prices Bermudan swaptions under the one-factor Hull-White short-rate model. A
C++17 numerical core does the heavy lifting; a Python layer handles market data,
distribution, delivery, Excel access, and an interactive Streamlit dashboard.

Two interchangeable backends ship with it: a compiled C++17 core (via pybind11)
for speed, and a pure-Python reference engine with the same model mathematics.
The dashboard uses the C++ core when it is available and the Python engine
otherwise, so it runs anywhere — including free hosts that do not compile native
code — with no extra setup.

## What's inside

- **C++17 core** — exact Hull-White simulation, Sobol quasi-Monte Carlo,
  Brownian-bridge construction, Longstaff-Schwartz regression.
- **Python reference engine** — the same model in NumPy/SciPy, used as an
  automatic fallback.
- **Python infrastructure** — FRED market-data ingestion, Ray distribution, an
  async gRPC service, and an Excel UDF.
- **Invariant tests** — martingale consistency, Jamshidian convergence,
  exercise monotonicity.
- **Streamlit dashboard** — live pricing, reporting, and short-rate path plots.
- **CI** — cibuildwheel builds and tests wheels on Linux, Windows, and macOS.

## Layout

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

## Model

The short rate follows the Hull-White one-factor risk-neutral process:

```math
dr_t = (\theta(t) - a\,r_t)\,dt + \sigma\,dW_t
```

The simulator uses the exact Gaussian transition density:

```math
\mathbb{E}[r_t \mid \mathcal{F}_s] = r_s e^{-a(t-s)} + \alpha(t) - \alpha(s)e^{-a(t-s)}
```

```math
\mathrm{Var}(r_t \mid \mathcal{F}_s) = \frac{\sigma^2}{2a}\left(1 - e^{-2a(t-s)}\right)
```

Early exercise uses Longstaff-Schwartz backward induction with weighted Laguerre
basis functions and a ridge-regularized normal-equations solve. Both backends
share the same cubic-spline curve, discount-bond reconstruction, and Jamshidian
European reduction.

### Backends

| Backend | Module | Speed | Needs a compiler |
| --- | --- | --- | --- |
| C++17 core | `quant_engine_cpp` (built from `cpp_core/`) | Fast | Yes |
| Python reference | `python_layer/python_engine.py` | Slower | No |

`load_cpp_engine()` resolves the backend in order: an installed
`quant_engine_cpp` extension, a `quant_engine_cpp` in a local `build*` directory,
then the Python reference engine. The dashboard shows which one is active. The
Python engine draws QMC points from `scipy.stats.qmc.Sobol`, so its prices match
the C++ core in expectation (within Monte Carlo error) rather than draw-for-draw.

## Market data

Live pricing uses the published US rates curve from FRED and needs a free FRED
API key. The loader reads FRBNY SOFR and SOFR averages, Federal Reserve H.15
Treasury constant-maturity rates, and curve-implied forward-start par swap rates
derived from the downloaded curve.

Create a local `.env` file (git-ignored):

```bash
FRED_API_KEY=your_fred_api_key
```

Use the bare 32-character key value — not the account URL, a Markdown link, the
`FRED_API_KEY=` prefix, or placeholder text. For Streamlit Cloud, add the same
value under **App settings → Secrets**. If no key is set, the dashboard prompts
for one on first load and stores it in the local `.env`.

## Local setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
python -m pip install --upgrade pip
```

Run the dashboard against the Python engine (no build step):

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Build the C++ core for full speed:

```bash
python -m pip install -r requirements-dev.txt
python -m build --wheel
python -m pip install dist/*.whl
```

Once `quant_engine_cpp` is importable, the dashboard and the distributed and
gRPC paths use it automatically.

Run the verification suite (requires the compiled core):

```bash
pytest tests/test_invariants.py -q
```

## Streamlit Cloud

1. Push the repository to GitHub.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), select **New
   app** and choose this repository and branch `main`.
3. Set the entrypoint to `app.py`.
4. Add the key under **App settings → Secrets**:

   ```toml
   FRED_API_KEY = "your_fred_api_key"
   ```

5. Deploy.

Streamlit installs `requirements.txt` and starts the dashboard. The app uses
whichever backend is present — the compiled core if the platform builds it,
otherwise the Python reference engine — and shows which one is active.

## CI

GitHub Actions builds wheels with `pypa/cibuildwheel` for Ubuntu, Windows, and
macOS on Python 3.10–3.12. Each wheel is tested against
`tests/test_invariants.py` before upload, and tagged releases publish the
verified wheels to GitHub Releases.

## Validation

The invariant suite checks the model, not just the plumbing:

- **Martingale property** — discounted zero-coupon bond prices are unbiased
  under the risk-neutral simulation.
- **Jamshidian convergence** — single-exercise Bermudan valuation matches the
  analytical European swaption reduction.
- **Exercise monotonicity** — Bermudan value does not fall as exercise
  opportunities increase.

The Python reference engine passes the same three checks; in the deterministic
(`sigma = 0`) case its single-exercise LSMC price reproduces the Jamshidian value
to machine precision.
