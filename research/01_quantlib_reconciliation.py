import time, numpy as np, QuantLib as ql
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from quant_engine.python_layer import python_engine as qe

# ---------- shared model / instrument ----------
r_flat, a, sigma = 0.03, 0.05, 0.01
T0 = 2          # option expiry (years)
pay = [3.0,4.0,5.0]          # annual fixed payments of the underlying swap
exr = [2.0,3.0,4.0]          # Bermudan exercise dates
notional = 1.0

# ---------- our engine ----------
t_nodes = np.array([0.,0.5,1,2,3,4,5,7,10], float)
curve = qe.YieldCurve(t_nodes, np.full_like(t_nodes, r_flat))
proc  = qe.HullWhiteProcess(curve, a, sigma)
eng   = qe.LsmcEngine(proc)

# ATM strike from our curve: par rate of the forward swap (start T0, pays at `pay`)
def our_par_rate():
    P = lambda t: np.exp(-r_flat*t)
    ann = sum(1.0*P(t) for t in pay)          # tau=1 annual
    return (P(T0)-P(pay[-1]))/ann
K = our_par_rate()

def our_european():
    spec = qe.BermudanSwaptionSpec(notional, K, qe.SwaptionType.Payer, [T0], pay)
    return eng.european_swaption_jamshidian(spec, float(T0))

def our_lsmc(paths, seed=1):
    spec = qe.BermudanSwaptionSpec(notional, K, qe.SwaptionType.Payer, exr, pay)
    res = eng.price(spec, qe.LsmcSimulationConfig(path_count=paths, seed=seed, ridge_lambda=1e-10))
    return res.price, res.standard_error

# ---------- QuantLib ----------
today = ql.Date(15,6,2026)
ql.Settings.instance().evaluationDate = today
dc_curve = ql.Actual365Fixed()
flat = ql.FlatForward(today, r_flat, dc_curve, ql.Continuous, ql.Annual)
ts = ql.YieldTermStructureHandle(flat)
hw = ql.HullWhite(ts, a, sigma)

cal = ql.NullCalendar()
dc_fix = ql.Thirty360(ql.Thirty360.BondBasis)
idx = ql.IborIndex('idx', ql.Period(1,ql.Years), 0, ql.USDCurrency(), cal,
                   ql.Unadjusted, False, ql.Actual365Fixed(), ts)
start  = cal.advance(today, ql.Period(T0, ql.Years))
mat    = cal.advance(start, ql.Period(len(pay), ql.Years))
fixSch = ql.Schedule(start, mat, ql.Period(1,ql.Years), cal, ql.Unadjusted, ql.Unadjusted,
                     ql.DateGeneration.Forward, False)
fltSch = fixSch
swap = ql.VanillaSwap(ql.VanillaSwap.Payer, notional, fixSch, K, dc_fix,
                      fltSch, idx, 0.0, ql.Actual365Fixed())
swap.setPricingEngine(ql.DiscountingSwapEngine(ts))
ql_fair = swap.fairRate()

# European (single exercise at T0) via Jamshidian
euro_ex = ql.EuropeanExercise(start)
euro_swpt = ql.Swaption(swap, euro_ex)
euro_swpt.setPricingEngine(ql.JamshidianSwaptionEngine(hw, ts))
ql_euro = euro_swpt.NPV()

# Bermudan via tree
berm_dates = [cal.advance(today, ql.Period(int(t), ql.Years)) for t in exr]
berm_ex = ql.BermudanExercise(berm_dates)
berm_swpt = ql.Swaption(swap, berm_ex)
berm_swpt.setPricingEngine(ql.TreeSwaptionEngine(hw, 200))
ql_berm = berm_swpt.NPV()

print(f"ATM strike: ours K={K*100:.4f}%   QuantLib fairRate={ql_fair*100:.4f}%")
print()
print("=== European (single exercise) : isolates conventions ===")
oe = our_european()
print(f"  ours (Jamshidian) : {oe:.6e}")
print(f"  QuantLib (Jamshidian): {ql_euro:.6e}")
print(f"  abs diff: {abs(oe-ql_euro):.2e}  ({abs(oe-ql_euro)/notional*1e4:.2f} bp of notional)")
print()
print("=== Bermudan : LSMC vs QuantLib tree ===")
print(f"  QuantLib tree (200 steps): {ql_berm:.6e}")
print(f"  {'paths':>8} {'LSMC price':>14} {'95% CI half-w':>14} {'vs tree (bp)':>12} {'time(s)':>8}")
for n in [1024,4096,16384,65536]:
    t=time.time(); p,se=our_lsmc(n); dt=time.time()-t
    print(f"  {n:>8} {p:>14.6e} {1.96*se:>14.2e} {(p-ql_berm)/notional*1e4:>12.2f} {dt:>8.2f}")
