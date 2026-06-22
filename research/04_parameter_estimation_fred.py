import urllib.request, json, numpy as np
import os
KEY=os.environ["FRED_API_KEY"]
def fred(series):
    url=f"https://api.stlouisfed.org/fred/series/observations?series_id={series}&api_key={KEY}&file_type=json"
    req=urllib.request.Request(url, headers={"User-Agent":"quant-engine/0.1"})
    obs=json.loads(urllib.request.urlopen(req,timeout=40).read().decode())["observations"]
    d=[(o["date"],float(o["value"])/100.0) for o in obs if o["value"] not in (".","")]
    return [x[0] for x in d], np.array([x[1] for x in d])

dates, r = fred("SOFR")
dr = np.diff(r); n=len(r); dt=1/252
print(f"SOURCE: FRED series SOFR (daily, overnight). n={n} obs, {dates[0]} -> {dates[-1]}")
print(f"Level range: {r.min()*100:.3f}% to {r.max()*100:.3f}% (max is the 2019 repo spike)")

def realized_sigma(x): 
    d=np.diff(x); return d.std(ddof=1)*np.sqrt(252)
print("\n--- sigma from realized vol of daily changes (robust; measure-invariant) ---")
for lbl,win in [("full history",n),("last 504 (~2y)",505),("last 252 (~1y)",253)]:
    s=realized_sigma(r[-win:]) if win<=n else realized_sigma(r)
    se=s/np.sqrt(2*(min(win,n)-1))          # approx SE of a vol estimate
    print(f"   {lbl:>16}: sigma = {s*1e4:7.1f} bp/yr  (+/- {se*1e4:.1f} bp, ~95%)")

print("\n--- OU/Vasicek MLE (AR(1) exact discretization) ---")
def ou_fit(x):
    a0=x[:-1]; a1=x[1:]; m=len(a0)
    phi=np.cov(a0,a1,ddof=1)[0,1]/np.var(a0,ddof=1)
    c=a1.mean()-phi*a0.mean(); resid=a1-(c+phi*a0); rv=resid.var(ddof=1)
    a=-np.log(phi)/dt; theta=c/(1-phi); sig=np.sqrt(rv*2*a/(1-phi**2))
    # SE of phi (OLS) -> delta method to a
    se_phi=np.sqrt(rv/np.sum((a0-a0.mean())**2)); se_a=se_phi/(phi*dt)
    return a, theta, sig, phi, se_a
for lbl,win in [("full history",n),("last 504 (~2y)",505)]:
    a,th,sig,phi,se_a=ou_fit(r[-win:] if win<=n else r)
    hl=np.log(2)/a
    print(f"   {lbl:>14}: a={a:6.3f}/yr (+/- {se_a:.3f}), half-life={hl:5.2f}y, theta={th*100:5.2f}%, sigma={sig*1e4:6.1f} bp/yr")
print("\nNote: realized-vol sigma is the trustworthy number; OU 'a' from the level series is")
print("fragile because multi-year rates are non-stationary (ZIRP -> hikes -> cuts).")
