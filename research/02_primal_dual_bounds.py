import numpy as np, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from quant_engine.python_layer import python_engine as qe
WLB, RB = qe._weighted_laguerre_basis, qe._ridge_beta
r_flat, a, sigma = 0.03, 0.05, 0.01
pay=[3.,4.,5.]; exr=[2.,3.,4.]; notional=1.; ridge=1e-10
t_nodes=np.array([0.,.5,1,2,3,4,5,7,10],float)
curve=qe.YieldCurve(t_nodes,np.full_like(t_nodes,r_flat)); proc=qe.HullWhiteProcess(curve,a,sigma); eng=qe.LsmcEngine(proc)
P=lambda t: np.exp(-r_flat*t); Kstr=(P(2.)-P(pay[-1]))/sum(P(t) for t in pay)
spec=qe.BermudanSwaptionSpec(notional,Kstr,qe.SwaptionType.Payer,exr,pay)
time_grid=np.array([0.]+exr); Tn=time_grid.size; ex_count=len(exr); QL_TREE=1.576184e-2
intrinsic=lambda col,r: np.maximum(0., eng.swap_present_value(float(time_grid[col]), np.asarray(r,float), spec))

def rich(r,mu,sd):
    xs=(np.asarray(r,float)-mu)/sd
    return np.column_stack([xs**k for k in range(6)])   # standardized degree-5 poly

def primal(paths, seed):
    nrm=qe._sobol_standard_normals(ex_count,paths,seed)
    rates=proc.simulate_short_rate_paths(nrm,time_grid); integ=proc.integrated_short_rates(rates,time_grid)
    pol={}; rv={}; final=ex_count; cf=intrinsic(final,rates[:,final]); cur=final
    for ei in range(ex_count-2,-1,-1):
        tc=ei+1; cf=cf*np.exp(-(integ[:,cur]-integ[:,tc])); rr=rates[:,tc]; intr=intrinsic(tc,rr); itm=intr>0
        if itm.sum()>4:
            B=WLB(rr[itm]); beta=RB(B,cf[itm],ridge); cont=np.zeros_like(cf); cont[itm]=np.maximum(0.,B@beta)
        else:
            beta=None; cont=cf.copy()
        # rich value-surface fit over ALL paths (for the dual)
        mu,sd=float(rr.mean()),float(rr.std()+1e-12); Br=rich(rr,mu,sd)
        rbeta=np.linalg.solve(Br.T@Br+1e-8*np.eye(6), Br.T@cf)
        pol[tc]=(beta,); rv[tc]=(mu,sd,rbeta)
        cf=np.where(itm&(intr>cont),intr,cf); cur=tc
    pv=cf*np.exp(-integ[:,cur]); return float(pv.mean()), float(pv.std(ddof=1)/np.sqrt(paths)), pol, rv

def Vhat(col,r,pol,rv,kind):
    intr=intrinsic(col,r)
    if col==ex_count: return intr
    if kind=='laguerre':
        beta=pol[col][0]
        if beta is None: return intr
        return np.maximum(intr,np.maximum(0.,WLB(r)@beta))
    mu,sd,rbeta=rv[col]; return np.maximum(intr,np.maximum(0.,rich(r,mu,sd)@rbeta))

def dual(outerN,innerM,pol,rv,kind,seed=7):
    rng=np.random.default_rng(seed); nrm=rng.standard_normal((outerN,ex_count))
    rates=proc.simulate_short_rate_paths(nrm,time_grid); integ=proc.integrated_short_rates(rates,time_grid); D=np.exp(-integ)
    M=np.zeros(outerN); dmax=np.zeros(outerN)
    for j in range(1,Tn):
        s,t=float(time_grid[j-1]),float(time_grid[j]); dt=t-s
        decay=np.exp(-a*dt); sd_=np.sqrt(max(0.,(sigma**2/(2*a))*(1-np.exp(-2*a*dt)))); als,alt=proc.alpha(s),proc.alpha(t)
        rp,rc=rates[:,j-1],rates[:,j]; z=rng.standard_normal((outerN,innerM))
        r_in=rp[:,None]*decay+alt-als*decay+sd_*z
        ce=np.mean(np.exp(-0.5*(rp[:,None]+r_in)*dt)*Vhat(j,r_in.ravel(),pol,rv,kind).reshape(outerN,innerM),axis=1)
        M=M+D[:,j]*Vhat(j,rc,pol,rv,kind)-D[:,j-1]*ce
        dmax=np.maximum(dmax,D[:,j]*intrinsic(j,rc)-M)
    return float(dmax.mean()), float(dmax.std(ddof=1)/np.sqrt(outerN))

L,seL,pol,rv=primal(65536,1)
Ul,sl=dual(6000,1500,pol,rv,'laguerre'); Ur,sr=dual(6000,1500,pol,rv,'rich')
print(f"Primal lower L          : {L*1e2:.4f}%   (+/- {1.96*seL*1e4:.2f} bp)")
print(f"QuantLib tree (truth)   : {QL_TREE*1e2:.4f}%")
print(f"Dual U, 4-term Laguerre : {Ul*1e2:.4f}%   gap to tree {(Ul-QL_TREE)*1e4:.2f} bp,  bracket width {(Ul-L)*1e4:.2f} bp")
print(f"Dual U, deg-5 poly      : {Ur*1e2:.4f}%   gap to tree {(Ur-QL_TREE)*1e4:.2f} bp,  bracket width {(Ur-L)*1e4:.2f} bp")
print(f"Tree in [L, U_rich]?    : {L <= QL_TREE <= Ur}")
