import urllib.request, json, numpy as np, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from quant_engine.python_layer.data_io import MarketDataManager
import os
KEY=os.environ["FRED_API_KEY"]
def fred(s):
    u=(f"https://api.stlouisfed.org/fred/series/observations?series_id={s}&api_key={KEY}"
       f"&file_type=json&observation_start=2021-01-01")
    r=urllib.request.Request(u,headers={"User-Agent":"q/0.1"})
    o=json.loads(urllib.request.urlopen(r,timeout=15).read().decode())["observations"]
    return {x["date"]:float(x["value"]) for x in o if x["value"] not in (".","")}

# ---- G2++ (Brigo-Mercurio) bond reconstruction + curve reproduction ----
mgr=MarketDataManager(); mgr.ensure_market_data(); arr=mgr.get_curve_arrays(); mgr.close()
from quant_engine.python_layer import python_engine as qe
curve=qe.YieldCurve(np.ascontiguousarray(arr.times),np.ascontiguousarray(arr.zero_rates))
PM=lambda T: curve.discount_factor(T)
def Vterm(a,b,sig,eta,rho,tau):
    return (sig**2/a**2)*(tau+(2/a)*np.exp(-a*tau)-(1/(2*a))*np.exp(-2*a*tau)-3/(2*a)) \
         + (eta**2/b**2)*(tau+(2/b)*np.exp(-b*tau)-(1/(2*b))*np.exp(-2*b*tau)-3/(2*b)) \
         + 2*rho*sig*eta/(a*b)*(tau+(np.exp(-a*tau)-1)/a+(np.exp(-b*tau)-1)/b-(np.exp(-(a+b)*tau)-1)/(a+b))
def Pg2(t,T,x,y,p):
    a,b,sig,eta,rho=p; tau=T-t; B=lambda z: (1-np.exp(-z*tau))/z
    A=0.5*(Vterm(*p,tau)-Vterm(*p,T)+Vterm(*p,t))
    return (PM(T)/PM(t))*np.exp(A-B(a)*x-B(b)*y)
p=(0.50,0.05,0.008,0.006,-0.70)   # a,b,sigma,eta,rho (representative; labelled)
err=max(abs(Pg2(0.,T,0.,0.,p)-PM(T)) for T in [1,2,5,10,20])
print(f"G2++ curve reproduction error P(0,T;0,0) vs market: {err:.2e}  (should be ~0)")

# ---- model-implied terminal correlation of zero rates (1F vs G2++) ----
def corr_matrix(p, t, taus):
    a,b,sig,eta,rho=p
    Vx=sig**2*(1-np.exp(-2*a*t))/(2*a); Vy=eta**2*(1-np.exp(-2*b*t))/(2*b)
    Cxy=rho*sig*eta*(1-np.exp(-(a+b)*t))/(a+b); S=np.array([[Vx,Cxy],[Cxy,Vy]])
    Bz=lambda z,tau:(1-np.exp(-z*tau))/z
    U=np.array([[Bz(a,tau)/tau,Bz(b,tau)/tau] for tau in taus])
    C=U@S@U.T; d=np.sqrt(np.diag(C)); return C/np.outer(d,d)
taus=[2,5,10,30]; labels=["2Y","5Y","10Y","30Y"]
Cg2=corr_matrix(p,1.0,taus)

# ---- empirical correlation from real FRED daily changes ----
data={s:fred(s) for s in ["DGS2","DGS5","DGS10","DGS30"]}
common=sorted(set.intersection(*[set(d) for d in data.values()]))
dM=np.diff(np.array([[data[s][d] for s in ["DGS2","DGS5","DGS10","DGS30"]] for d in common]),axis=0)
Cemp=np.corrcoef(dM.T)
def show(name,C):
    print(f"\n{name}:\n        "+" ".join(f"{l:>6}" for l in labels))
    for i,l in enumerate(labels): print(f"   {l:>4} "+" ".join(f"{C[i,j]:>6.2f}" for j in range(len(labels))))
print("\n1-factor Hull-White implied: all correlations = 1.00 (single Brownian, structural).")
show("G2++ implied (a=.5,b=.05,sig=80,eta=60bp,rho=-.7), 1y horizon", Cg2)
show(f"Empirical (real FRED daily changes, {common[0]}..{common[-1]}, n={len(common)})", Cemp)
print(f"\nKey number  corr(2Y,30Y):  1F = 1.00 | G2++ = {Cg2[0,3]:.2f} | empirical = {Cemp[0,3]:.2f}")
