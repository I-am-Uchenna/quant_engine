# Self-Attack: Adversarial Review of PAPER.tex

Adversarial pass in the E2ER style: deliberately attack the paper's weakest points,
score severity 1 (cosmetic) to 10 (fatal), and record resolution. Findings with
severity >= 7 must be resolved or explicitly documented before the paper proceeds.

| # | Severity | Finding | Resolution |
|---|---|---|---|
| 1 | 7 | **No implied-vol calibration.** Model parameters are historically estimated or conventional, so prices are physical-measure anchors, not market quotes. A referee will ask "calibrated to what?" | **Documented** (Sec. 8, limitation 2) as the deliberate open item; not hidden. Requires a verifiable swaption surface, which is not freely available. |
| 2 | 7 | **Illustrative G2++ parameters.** The decorrelation result uses uncalibrated (a,b,sigma,eta,rho); model corr(2Y,30Y)=0.77 vs empirical 0.57. The model under-decorrelates. | **Stated** (Sec. 4.4, Fig. 3 caption): parameters are illustrative; calibration to the empirical correlation (or vol surface) would close the gap. Claim is qualitative (shape), not a fit. |
| 3 | 6 | **Single-instrument validation.** All numerical validation is on one 2Yx3Y ATM payer. Other strikes, tenors, payer/receiver, and exercise schedules are untested. | **Acknowledged.** The QuantLib reconciliation method is instrument-agnostic; a strike/tenor grid is named as future work. Conclusions are scoped to "within the stated scope." |
| 4 | 5 | **AAD is autograd, not production C++ AAD.** The 40x speedup uses pure-Python reverse-mode timing in a sandbox; absolute numbers are machine-dependent. | **Framed correctly** (Sec. 7): the claim is the *cheap-gradient principle* (reverse cost constant in #inputs), which is implementation-independent; the constant (~3.3x) is reported, not hidden. |
| 5 | 5 | **Stylized swap conventions.** Annual tau=1, NullCalendar, no business-day adjustment. The QuantLib match deliberately uses the same stylization, so it validates the *method*, not market pricing. | **Documented** (Sec. 8, limitation 3). The European reconciliation explicitly isolates conventions (0.07 bp). |
| 6 | 4 | **Two-handed LSMC-vs-tree gap.** The ~0.3 bp gap is attributed both to LS low bias and (via a grid test) to MC noise. | **Clarified**: the gap is within Monte Carlo error; the rigorous policy-quality statement is the primal-dual gap (Sec. 4.3), not the tree comparison. |
| 7 | 4 | **Key-rate buckets do not sum to parallel** (130.98 vs 125.71, ~4%). | **Surfaced honestly** (Sec. 6) with the named cause (cubic-spline non-locality) and the fix (local interpolation / full Jacobian). |
| 8 | 3 | **verify_numbers AAD check uses a simplified stand-in payoff**, not the full Bermudan AAD. | The full AAD validation (1e-6) is in `09_aad_greeks.py`; the gate's stand-in only confirms autograd == finite-difference (rel err 1.7e-6). Noted in the verifier. |
| 9 | 3 | **FRED-dependent numbers are not re-run inside the gate** (network), only checked against the sidecar and paper. | Those numbers are reproduced by the `research/` scripts; the gate covers provenance plus deterministic re-derivation of the engine/QuantLib figures. |

**Outcome:** no unresolved finding at severity >= 8. The two severity-7 items (calibration,
illustrative G2++ parameters) are the paper's central, openly-stated limitations rather than
concealed defects. Proceed to formal review.
