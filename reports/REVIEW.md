# Formal Review: Six Referee Simulations

Referee-simulation reviews in the E2ER style. Each reviewer scores 1-10 on the
dimensions most relevant to their role; the mechanical aggregation (REVIEW_aggregation.json)
applies the deterministic three-rule system.

---

## 1. Mechanism reviewer (is the core claim convincing?) — 7/10
The paper's thesis is that in a mature modelling area, credibility comes from validation
rigour, not model novelty. This is well-executed: independent cross-checks, a primal-dual
optimality proof, and real-data parameterisation. It convinces. It does not clear 8 because
the "engine is correct" claim rests on a single instrument; breadth would raise confidence.

## 2. Technical reviewer (correctness of math and numerics) [weight 1.5] — 8/10
The Hull-White and G2++ formulae are standard and correctly implemented (G2++ reproduces the
curve to 1e-12; both reconcile to QuantLib to <=0.10 bp). The primal-dual diagnostic, that the
naive gap is flat in inner paths and hence value-surface error, is a correct and sophisticated
argument. The `verify_numbers` gate re-derives the deterministic figures. Minor: the AAD
envelope-theorem justification could be stated more formally.

## 3. Methodology / identification reviewer (validity of claims) [weight 1.5] — 7/10
Claims are scoped honestly and the measure caveat (realized vs implied vol) is handled with
unusual care. The 25% price impact is a real, well-identified result. Capped at 7 by the
absence of an implied-vol calibration, which limits "market-consistent" claims, and by the
single-instrument scope.

## 4. Data reviewer (data quality, reproducibility) [weight 1.25] — 8/10
All inputs are real and reproducible (FRED series named and dated; 2,050 SOFR observations).
The realized-vol vs OU-MLE cross-check (agree to 0.1 bp) is a genuine internal validation.
Curve-implied swap rates are correctly labelled as model-implied, not market mid. Strong.

## 5. Literature reviewer (coverage, citations) — 7/10
The canonical references (Hull-White, Longstaff-Schwartz, Andersen-Broadie, Rogers, Brigo-
Mercurio, Jamshidian) are present and correctly used, and SR 11-7 grounds the governance framing.
A fuller discussion of competing approaches (Cheyette, LMM-SABR, deep optimal stopping) would
strengthen positioning.

## 6. Writing reviewer (clarity, precision) — 8/10
Clear, precise, well-structured; figures are labelled and discussed. The limitations section is
a model of honest scoping. Occasional density in the validation sections, but no ambiguity.

---

### Dimension summary

| Reviewer | Weight | Score |
|---|---|---|
| mechanism | 1.00 | 7 |
| technical | 1.50 | 8 |
| methodology/identification | 1.50 | 7 |
| data | 1.25 | 8 |
| literature | 1.00 | 7 |
| writing | 1.00 | 8 |

Weighted average = (7 + 8*1.5 + 7*1.5 + 8*1.25 + 7 + 8) / 7.25 = **7.52 / 10**.
