import numpy as np, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from quant_engine.python_layer.data_io import MarketDataManager
from quant_engine.python_layer import python_engine as qe
mgr=MarketDataManager(); summ=mgr.ensure_market_data(); arr=mgr.get_curve_arrays(); mgr.close()
times=np.ascontiguousarray(arr.times); zeros=np.ascontiguousarray(arr.zero_rates)
a=0.05; pay=[3.,4.,5.]; exr=[2.,3.,4.]; notional=1_000_000.
curve=qe.YieldCurve(times,zeros); P=lambda t: curve.discount_factor(t)
K=(P(2.)-P(pay[-1]))/sum(P(t) for t in pay)
def price(sig):
    e=qe.LsmcEngine(qe.HullWhiteProcess(curve,a,sig))
    spec=qe.BermudanSwaptionSpec(notional,K,qe.SwaptionType.Payer,exr,pay)
    r=e.price(spec,qe.LsmcSimulationConfig(path_count=131072,seed=12,ridge_lambda=1e-10))
    return r.price, r.standard_error
print(f"Curve: FRED as-of {summ.as_of}.  2Yx3Y ATM payer, K={K*100:.4f}%, a={a} (conventional), notional ${notional:,.0f}\n")
print(f"{'sigma (bp/yr)':>14} {'source':<34} {'price ($)':>12} {'MC se':>8}")
for sig,src in [(0.0100,'assumed (earlier)'),(0.00751,'realized vol, last ~2y (FRED)'),(0.00726,'realized vol, last ~1y (FRED)')]:
    p,se=price(sig); print(f"{sig*1e4:>14.1f} {src:<34} {p:>12,.2f} {se:>8.2f}")
# price uncertainty from sigma estimation error (95% +/-2.4bp at 2y window)
p_lo,_=price(0.00751-0.00024); p_hi,_=price(0.00751+0.00024)
print(f"\nPrice 95% band from sigma estimation error (+/-2.4 bp): [${p_lo:,.0f}, ${p_hi:,.0f}]")
