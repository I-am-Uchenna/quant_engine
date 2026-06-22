import urllib.request, json, numpy as np, sys
import os
KEY=os.environ["FRED_API_KEY"]
def fred(series):
    url=(f"https://api.stlouisfed.org/fred/series/observations?series_id={series}"
         f"&api_key={KEY}&file_type=json&observation_start=2021-01-01")
    req=urllib.request.Request(url, headers={"User-Agent":"quant-engine/0.1"})
    obs=json.loads(urllib.request.urlopen(req,timeout=15).read().decode())["observations"]
    return {o["date"]:float(o["value"]) for o in obs if o["value"] not in (".","")}
tenors=["DGS2","DGS5","DGS10","DGS30"]; labels=["2Y","5Y","10Y","30Y"]
data={}
for t in tenors:
    data[t]=fred(t); print(f"fetched {t}: {len(data[t])} obs", flush=True)
common=sorted(set.intersection(*[set(d) for d in data.values()]))
M=np.array([[data[t][d] for t in tenors] for d in common])
dM=np.diff(M,axis=0)
print(f"\nSOURCE: FRED H.15 {','.join(tenors)}; {len(common)} dates {common[0]}..{common[-1]}")
cov=np.cov(dM.T,ddof=1); w,V=np.linalg.eigh(cov); i=np.argsort(w)[::-1]; w=w[i]; V=V[:,i]; expl=w/w.sum()*100
print("PCs of daily Treasury-curve changes:")
print(f"   {'PC':>2} {'var%':>7} {'cum%':>7}   loadings ("+" ".join(f"{l:>5}" for l in labels)+")")
cum=0
for k in range(len(tenors)):
    cum+=expl[k]; load=V[:,k]; load=load if load[np.argmax(np.abs(load))]>0 else -load
    print(f"   {k+1:>2} {expl[k]:>6.2f}% {cum:>6.2f}%   "+" ".join(f"{x:>5.2f}" for x in load))
print(f"\nPC1 (level) = {expl[0]:.1f}% of variance; a 1-factor model forces this to 100%.")
print(f"PC1+PC2 = {expl[0]+expl[1]:.1f}%. The {expl[1]:.1f}% slope factor is what G2++ adds and 1F cannot.")
