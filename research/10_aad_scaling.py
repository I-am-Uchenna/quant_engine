import numpy as np, time, sys
import autograd.numpy as anp
from autograd import grad
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from quant_engine.python_layer.data_io import MarketDataManager

# real curve
mgr=MarketDataManager(); mgr.ensure_market_data(); arr=mgr.get_curve_arrays(); mgr.close()
T=np.ascontiguousarray(arr.times); z0=np.ascontiguousarray(arr.zero_rates)
a=0.05; sig0=0.00751; pay=[3.,4.,5.]; exr=[2.,3.,4.]; notional=1_000_000.; ec=len(exr)
tg=np.array([0.]+exr)

# linear-interp coefficient vectors (zrate(t)=c.z, zprime(t)=d.z) for fixed times
def coefs(t):
    c=np.zeros(len(T)); d=np.zeros(len(T))
    if t<=T[0]: c[0]=1.0
    elif t>=T[-1]: c[-1]=1.0
    else:
        i=int(np.searchsorted(T,t)-1); seg=T[i+1]-T[i]; w=(t-T[i])/seg
        c[i]=1-w; c[i+1]=w; d[i]=-1/seg; d[i+1]=1/seg
    return c,d
times_needed=sorted(set([0.0]+exr+pay)); CO={t:coefs(t) for t in times_needed}
def zr(t,z): return anp.dot(CO[t][0],z)
def zp(t,z): return anp.dot(CO[t][1],z)
def Pm(t,z): return anp.exp(-zr(t,z)*t)
def fwd(t,z): return zr(t,z)+t*zp(t,z)
def alpha(t,z,s): return fwd(t,z)+(s*s/(2*a*a))*(1-np.exp(-a*t))**2
def dbond(t,Tm,r,z,s):
    B=(1-np.exp(-a*(Tm-t)))/a; conv=(s*s/(4*a))*(1-np.exp(-2*a*t))*B*B
    A=(Pm(Tm,z)/Pm(t,z))*anp.exp(B*fwd(t,z)-conv); return A*anp.exp(-B*r)
def swap_pv(t,r,z,s):
    fix=0.0; last=1.0; prev=t
    for pt in pay:
        if pt<=t+1e-12: continue
        disc=dbond(t,pt,r,z,s); fix=fix+0.04*(pt-prev)*disc; last=disc; prev=pt
    return notional*((1-last)-fix)   # note: K replaced below

# generate fixed normals (CRN) and fit policy at base via numpy
rng=np.random.default_rng(7); N=40000; Z=rng.standard_normal((N,ec))
# ATM strike from real curve
Pf=lambda t: float(np.exp(-np.dot(coefs(t)[0],z0)*t)); K=(Pf(2.)-Pf(pay[-1]))/sum(Pf(t) for t in pay)
def swap_pv_np(t,r):  # numpy version with strike K for policy fit
    fix=np.zeros_like(r); last=np.ones_like(r); prev=t
    for pt in pay:
        if pt<=t+1e-12: continue
        B=(1-np.exp(-a*(pt-t)))/a; conv=(sig0**2/(4*a))*(1-np.exp(-2*a*t))*B*B
        A=(Pf(pt)/Pf(t))*np.exp(B*(np.dot(coefs(t)[1],z0)*t+np.dot(coefs(t)[0],z0))-conv); disc=A*np.exp(-B*r)
        fix=fix+K*(pt-prev)*disc; last=disc; prev=pt
    return notional*((1-last)-fix)
def sim_np(z,s):
    r=np.zeros((N,ec+1)); r[:,0]=float(np.dot(coefs(0.)[0],z))
    for j in range(1,ec+1):
        dt=tg[j]-tg[j-1]; dec=np.exp(-a*dt); sd=s*np.sqrt((1-np.exp(-2*a*dt))/(2*a))
        al_s=np.dot(coefs(tg[j-1])[1],z)*tg[j-1]+np.dot(coefs(tg[j-1])[0],z)+(s*s/(2*a*a))*(1-np.exp(-a*tg[j-1]))**2
        al_t=np.dot(coefs(tg[j])[1],z)*tg[j]+np.dot(coefs(tg[j])[0],z)+(s*s/(2*a*a))*(1-np.exp(-a*tg[j]))**2
        r[:,j]=r[:,j-1]*dec+al_t-al_s*dec+sd*Z[:,j-1]
    integ=np.zeros((N,ec+1))
    for j in range(1,ec+1): integ[:,j]=integ[:,j-1]+0.5*(r[:,j-1]+r[:,j])*(tg[j]-tg[j-1])
    return r,integ
from quant_engine.python_layer import python_engine as qe
WLB,RB=qe._weighted_laguerre_basis,qe._ridge_beta
r_np,integ_np=sim_np(z0,sig0)
# backward fit betas
cf=np.maximum(0.,swap_pv_np(exr[-1],r_np[:,ec])); cur=ec; betas={}
for ei in range(ec-2,-1,-1):
    tc=ei+1; cf=cf*np.exp(-(integ_np[:,cur]-integ_np[:,tc])); rr=r_np[:,tc]; iv=np.maximum(0.,swap_pv_np(exr[tc-1],rr)); itm=iv>0
    beta=None; cont=cf.copy()
    if itm.sum()>4: B=WLB(rr[itm]); beta=RB(B,cf[itm],1e-10); cont=np.zeros_like(cf); cont[itm]=np.maximum(0.,B@beta)
    cf=np.where(itm&(iv>cont),iv,cf); betas[tc]=beta; cur=tc
# forward to get stop col
alive=np.ones(N,bool); stop=np.full(N,-1)
for j in range(1,ec+1):
    rr=r_np[:,j]; iv=np.maximum(0.,swap_pv_np(exr[j-1],rr)); cont=np.zeros(N)
    if j<ec and betas.get(j) is not None: cont=np.maximum(0.,WLB(rr)@betas[j])
    exn=alive&(iv>cont)&(iv>0); stop=np.where(exn,j,stop); alive=alive&~exn
masks=[(stop==j).astype(float) for j in range(1,ec+1)]

# ---- AAD price as smooth fn of (z, sigma), fixed policy ----
def price(z,s):
    r=[anp.dot(CO[0.0][0],z)*anp.ones(N)]
    for j in range(1,ec+1):
        dt=tg[j]-tg[j-1]; dec=np.exp(-a*dt); sd=s*np.sqrt((1-np.exp(-2*a*dt))/(2*a))
        r.append(r[j-1]*dec+alpha(tg[j],z,s)-alpha(tg[j-1],z,s)*dec+sd*Z[:,j-1])
    integ=[anp.zeros(N)]
    for j in range(1,ec+1): integ.append(integ[j-1]+0.5*(r[j-1]+r[j])*(tg[j]-tg[j-1]))
    tot=0.0
    for j in range(1,ec+1):
        sv=notional*((1-_last(tg[j],r[j],z,s))-_fix(tg[j],r[j],z,s))
        tot=tot+anp.sum(masks[j-1]*anp.exp(-integ[j])*anp.maximum(0.,sv))
    return tot/N
def _fix(t,r,z,s):
    fix=0.0; prev=t
    for pt in pay:
        if pt<=t+1e-12: continue
        fix=fix+K*(pt-prev)*dbond(t,pt,r,z,s); prev=pt
    return fix
def _last(t,r,z,s):
    last=anp.ones(N)
    for pt in pay:
        if pt<=t+1e-12: continue
        last=dbond(t,pt,r,z,s)
    return last


import time
def pt(theta): return price(theta[:-1], theta[-1])
def bench(label):
    theta0=np.concatenate([z0,[sig0]]); V=pt(theta0)
    t=time.time()
    for _ in range(3): pt(theta0)
    fwd=(time.time()-t)/3
    t=time.time(); g=grad(pt)(theta0); rev=time.time()-t
    h=1e-6; t=time.time()
    for i in range(len(theta0)): (pt(theta0+h*np.eye(len(theta0))[i])-V)/h
    allbump=time.time()-t
    print(f"   factors={len(theta0):>3} | 1 fwd={fwd*1e3:6.1f}ms | AAD reverse={rev*1e3:6.1f}ms (={rev/fwd:4.1f}x fwd) | bump-all={allbump*1e3:7.1f}ms | AAD speedup {allbump/rev:5.1f}x")
print("AAD vs bump-and-revalue scaling (reverse-pass cost ~ independent of #inputs):")
bench("14")
base_T=T.copy(); base_z=z0.copy()
for nn in [30,60,120]:
    Tn=np.unique(np.concatenate([base_T,np.linspace(base_T[0],base_T[-1],nn)])); zn=np.interp(Tn,base_T,base_z)
    globals()['T']=Tn; globals()['z0']=zn; globals()['CO']={t:coefs(t) for t in sorted(set([0.0]+exr+pay))}
    bench(str(len(Tn)))
