import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from quant_engine.python_layer.data_io import MarketDataManager
from quant_engine.python_layer import python_engine as qe
WLB,RB=qe._weighted_laguerre_basis, qe._ridge_beta

# ---- REAL market data from FRED (SOFR + H.15 Treasuries) ----
mgr=MarketDataManager(); summ=mgr.ensure_market_data(); arr=mgr.get_curve_arrays(); mgr.close()
times=np.ascontiguousarray(arr.times); zeros=np.ascontiguousarray(arr.zero_rates)
print(f"SOURCE: FRED, as-of {summ.as_of} ({summ.source}); {summ.curve_nodes} curve nodes")
print("Curve (tenor yr : zero %):", ", ".join(f"{t:.3g}:{z*100:.3f}" for t,z in zip(times,zeros)))

a,sigma=0.05,0.01; pay=[3.,4.,5.]; exr=[2.,3.,4.]; notional=1_000_000.; ridge=1e-10
PATHS,SEED=131072,12; bp=1e-4
def build(z,sig): c=qe.YieldCurve(times,z); return c,qe.LsmcEngine(qe.HullWhiteProcess(c,a,sig))
c0,e0=build(zeros,sigma); P=lambda t:c0.discount_factor(t); Kfix=(P(2.)-P(pay[-1]))/sum(P(t) for t in pay)
def price(z,sig,seed=SEED,paths=PATHS):
    c,e=build(z,sig); spec=qe.BermudanSwaptionSpec(notional,Kfix,qe.SwaptionType.Payer,exr,pay)
    r=e.price(spec,qe.LsmcSimulationConfig(path_count=paths,seed=seed,ridge_lambda=ridge)); return r.price,r.standard_error

V0,se0=price(zeros,sigma)
print(f"\n2Yx3Y payer Bermudan, ATM K={Kfix*100:.4f}%, notional ${notional:,.0f}")
print(f"Base price ${V0:,.2f} (MC se ${se0:,.2f})")
Vpar,_=price(zeros+bp,sigma); print(f"Parallel DV01 (+1bp): ${Vpar-V0:,.2f}/bp")
print("Key-rate DV01:")
ks=[]
for i,t in enumerate(times):
    if t<=0: continue
    z=zeros.copy(); z[i]+=bp; Vi,_=price(z,sigma); ks.append(Vi-V0)
    print(f"   {t:>5.3g}y : ${Vi-V0:>10,.2f}")
print(f"   sum buckets ${sum(ks):,.2f} vs parallel ${Vpar-V0:,.2f}")
Vv,_=price(zeros,sigma+bp); print(f"Model vega: ${(Vv-V0)/bp:,.0f}/unit sigma (${Vv-V0:,.2f} per 1bp sigma)")

# ---- exercise distribution + exposure via fitted policy ----
tg=np.array([0.]+exr); ec=len(exr)
spec=qe.BermudanSwaptionSpec(notional,Kfix,qe.SwaptionType.Payer,exr,pay)
proc=qe.HullWhiteProcess(c0,a,sigma); eng=e0
intr=lambda j,r: np.maximum(0.,eng.swap_present_value(float(tg[j]),np.asarray(r,float),spec))
def fit(paths,seed):
    nrm=qe._sobol_standard_normals(ec,paths,seed); R=proc.simulate_short_rate_paths(nrm,tg); I=proc.integrated_short_rates(R,tg)
    bts={}; cf=intr(ec,R[:,ec]); cur=ec
    for ei in range(ec-2,-1,-1):
        tc=ei+1; cf=cf*np.exp(-(I[:,cur]-I[:,tc])); rr=R[:,tc]; iv=intr(tc,rr); itm=iv>0; beta=None; cont=cf.copy()
        if itm.sum()>4:
            B=WLB(rr[itm]); beta=RB(B,cf[itm],ridge); cont=np.zeros_like(cf); cont[itm]=np.maximum(0.,B@beta)
        cf=np.where(itm&(iv>cont),iv,cf); bts[tc]=beta; cur=tc
    return bts
def fwd(paths,seed,bts):
    nrm=qe._sobol_standard_normals(ec,paths,seed); R=proc.simulate_short_rate_paths(nrm,tg); I=proc.integrated_short_rates(R,tg)
    alive=np.ones(paths,bool); stop=np.full(paths,-1); pv=np.zeros(paths); expo=np.zeros((paths,ec+1))
    for j in range(1,ec+1):
        rr=R[:,j]; iv=intr(j,rr); cont=np.zeros(paths)
        if j<ec and bts.get(j) is not None: cont=np.maximum(0.,WLB(rr)@bts[j])
        expo[:,j]=np.where(alive,np.maximum(iv,cont),0.0)
        exn=alive&(iv>cont)&(iv>0); pv=np.where(exn,iv*np.exp(-I[:,j]),pv); stop=np.where(exn,j,stop); alive=alive&~exn
    return pv,stop,expo
bts=fit(65536,1); pv,stop,expo=fwd(200000,99,bts)
print(f"\nForward-policy price ${pv.mean():,.2f} (se ${pv.std(ddof=1)/np.sqrt(len(pv)):,.2f})")
print("Exercise-time distribution:")
for j in range(1,ec+1): print(f"   {exr[j-1]:.0f}y : {100*np.mean(stop==j):5.2f}%")
print(f"   never : {100*np.mean(stop==-1):5.2f}%   P(ever)={100*np.mean(stop!=-1):5.2f}%")
print("Exposure (time-t mark-to-market):")
for j in range(1,ec+1):
    print(f"   {exr[j-1]:.0f}y  EPE ${np.mean(expo[:,j]):>10,.0f}  PFE97.5 ${np.quantile(expo[:,j],0.975):>10,.0f}  P(alive) {100*np.mean(expo[:,j]>0):5.2f}%")
