# Quant Engine: Hull-White Bermudan Swaption Pricer

Prices and risk-manages Bermudan swaptions under one- and two-factor Gaussian short-rate
models (Hull-White and G2++). A C++17 numerical core does the heavy lifting; a Python layer
handles market data, distribution, delivery, Excel access, and an interactive Streamlit
dashboard. The repository also ships a fully validated, reproducible research package: a
practitioner white paper, independent QuantLib benchmarks on real FRED data, and an automated
number-verification harness.

Two interchangeable backends ship with it: a compiled C++17 core (via pybind11) for speed, and
a pure-Python reference engine with the same model mathematics. The dashboard uses the C++ core
when it is available and the Python engine otherwise, so it runs anywhere, including free hosts
that do not compile native code, with no extra setup. A live instance is at
[quantengine.streamlit.app](https://quantengine.streamlit.app/).

## Validation and white paper

The methods used here are standard and deliberately so; the emphasis is on validating them
rigorously and reproducibly, not on a new model. The full write-up is the practitioner white
paper [`reports/WHITEPAPER.pdf`](reports/WHITEPAPER.pdf), with a terse model-validation record
in [`reports/VALIDATION.pdf`](reports/VALIDATION.pdf).

Headline results, each reproduced by a script in [`research/`](research) and machine-checked by
[`reports/verify_numbers.py`](reports/verify_numbers.py):

| Check | Result |
| --- | --- |
| European price vs QuantLib (Jamshidian) | 0.07 bp of notional |
| Bermudan vs QuantLib trinomial tree | within Monte Carlo error |
| Exercise policy (primal-dual gap) | consistent with zero (near-optimal) |
| G2++ vs QuantLib FdG2 engine | 0.10 bp |
| Volatility from 2,050 real SOFR observations | 25% price impact vs an assumed value |
| AAD Greeks vs bump-and-revalue | match to 1e-6, cost flat in number of factors |
| Number-verification gate | 20 / 20 pass |

Honest scope: parameters are estimated from data or set conventionally, not calibrated to a
live swaption-volatility surface (the principal open item). The white paper's limitations
section states this and the other boundaries plainly.

## Reproducibility

[`research/`](research) holds ten scripts that regenerate every figure and number from public
FRED data and QuantLib. [`reports/verify_numbers.py`](reports/verify_numbers.py) re-derives the
deterministic figures from the engine and QuantLib and checks every value against the paper;
[`reports/SELF_ATTACK.md`](reports/SELF_ATTACK.md) and [`reports/REVIEW.md`](reports/REVIEW.md)
record an adversarial pass and a structured review.

```bash
export FRED_API_KEY=your_32_char_fred_key
pip install -r requirements-dev.txt && pip install QuantLib autograd
PYTHONPATH=. python research/01_quantlib_reconciliation.py
python reports/verify_numbers.py
```

## What's inside

- **C++17 core**: exact Hull-White simulation, Sobol quasi-Monte Carlo, Brownian-bridge
  construction, Longstaff-Schwartz regression.
- **Python reference engine**: the same model in NumPy/SciPy, used as an automatic fallback.
- **Two-factor G2++**: captures curve decorrelation that one factor cannot, validated against
  QuantLib's finite-difference G2 engine.
- **Risk and exposure**: key-rate DV01, vega, exercise probabilities, EPE/PFE, and adjoint
  (AAD) Greeks.
- **Python infrastructure**: FRED market-data ingestion, Ray distribution, an async gRPC
  service, and an Excel UDF.
- **Streamlit dashboard**: live pricing, reporting, and short-rate path plots.
- **Validation package**: white paper, reproducible scripts, and an automated verification gate.

## Layout

```text
.
|-- cpp_core/                         # C++17 core (include/ interfaces, src/ + pybind11)
|-- python_layer/
|   |-- data_io.py                    # FRED + DuckDB ingestion; backend loader
|   |-- python_engine.py              # Pure-Python Hull-White reference engine
|   |-- ray_orchestrator.py           # Distributed pricing orchestration
|   |-- grpc_service.py               # Async gRPC pricing service
|   `-- xlwings_udf.py                # Excel UDF integration
|-- research/                         # 10 reproducible validation scripts (FRED + QuantLib)
|-- reports/                          # White paper, validation report, figures, verification harness
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

Early exercise is solved by Longstaff-Schwartz backward induction with weighted Laguerre basis
functions and a ridge-regularized normal-equations solve. The two-factor G2++ model adds a
second factor for curve decorrelation. See the white paper for the full treatment, including the
primal-dual policy certificate and adjoint Greeks.

### Backends

| Backend | Module | Speed | Needs a compiler |
| --- | --- | --- | --- |
| C++17 core | `quant_engine_cpp` (built from `cpp_core/`) | Fast | Yes |
| Python reference | `python_layer/python_engine.py` | Slower | No |

`load_cpp_engine()` resolves the backend in order: an installed `quant_engine_cpp` extension, a
`quant_engine_cpp` in a local `build*` directory, then the Python reference engine. The dashboard
shows which one is active.

## Market data

Live pricing uses the published US rates curve from FRED and needs a free FRED API key. The
loader reads FRBNY SOFR and SOFR averages, Federal Reserve H.15 Treasury constant-maturity
rates, and curve-implied forward-start par swap rates derived from the downloaded curve.

Create a local `.env` file (git-ignored):

```bash
FRED_API_KEY=your_fred_api_key
```

Use the bare 32-character key value, not the account URL, a Markdown link, the `FRED_API_KEY=`
prefix, or placeholder text. For Streamlit Cloud, add the same value under
**App settings > Secrets**.

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

## Streamlit Cloud

1. Push the repository to GitHub.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), select **New app** and choose
   this repository and branch `main`.
3. Set the entrypoint to `app.py`.
4. Add the key under **App settings > Secrets**:

   ```toml
   FRED_API_KEY = "your_fred_api_key"
   ```

5. Deploy.

Streamlit installs `requirements.txt` and starts the dashboard. The app uses whichever backend
is present (the compiled core if the platform builds it, otherwise the Python reference engine)
and shows which one is active.

## CI

GitHub Actions builds wheels with `pypa/cibuildwheel` for Ubuntu, Windows, and macOS on Python
3.10, 3.11, and 3.12. Each wheel is tested against `tests/test_invariants.py` before upload, and
tagged releases publish the verified wheels to GitHub Releases.
