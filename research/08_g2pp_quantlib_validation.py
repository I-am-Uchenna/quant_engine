import numpy as np, QuantLib as ql
r0,a,b,sig,eta,rho=0.03,0.50,0.05,0.008,0.006,-0.70
pay=[3.,4.,5.]; T=2.0
PM=lambda t: np.exp(-r0*t); fM=lambda t: r0
def Vt(tau):
    return (sig**2/a**2)*(tau+(2/a)*np.exp(-a*tau)-(1/(2*a))*np.exp(-2*a*tau)-3/(2*a)) \
         + (eta**2/b**2)*(tau+(2/b)*np.exp(-b*tau)-(1/(2*b))*np.exp(-2*b*tau)-3/(2*b)) \
         + 2*rho*sig*eta/(a*b)*(tau+(np.exp(-a*tau)-1)/a+(np.exp(-b*tau)-1)/b-(np.exp(-(a+b)*tau)-1)/(a+b))
def Pg2(t,Tm,x,y):
    tau=Tm-t; A=0.5*(Vt(tau)-Vt(Tm)+Vt(t)); Ba=(1-np.exp(-a*tau))/a; Bb=(1-np.exp(-b*tau))/b
    return (PM(Tm)/PM(t))*np.exp(A-Ba*x-Bb*y)
K=(PM(T)-PM(pay[-1]))/sum(PM(t) for t in pay)
def phi(t): return fM(t)+(sig**2/(2*a**2))*(1-np.exp(-a*t))**2+(eta**2/(2*b**2))*(1-np.exp(-b*t))**2+rho*sig*eta/(a*b)*(1-np.exp(-a*t))*(1-np.exp(-b*t))

# --- my G2++ European payer swaption via risk-neutral MC ---
def mc(paths=200000, steps=48, seed=1):
    rng=np.random.default_rng(seed); dt=T/steps; tg=np.linspace(0,T,steps+1)
    vx=sig**2*(1-np.exp(-2*a*dt))/(2*a); vy=eta**2*(1-np.exp(-2*b*dt))/(2*b)
    cxy=rho*sig*eta*(1-np.exp(-(a+b)*dt))/(a+b); L=np.linalg.cholesky([[vx,cxy],[cxy,vy]])
    x=np.zeros(paths); y=np.zeros(paths); integ=np.zeros(paths); rprev=x+y+phi(0.0)
    for i in range(1,steps+1):
        z=rng.standard_normal((paths,2))@L.T
        x=x*np.exp(-a*dt)+z[:,0]; y=y*np.exp(-b*dt)+z[:,1]
        rcur=x+y+phi(tg[i]); integ+=0.5*(rprev+rcur)*dt; rprev=rcur
    Vsw=1-Pg2(T,pay[-1],x,y)-K*sum(Pg2(T,ti,x,y) for ti in pay)
    pv=np.exp(-integ)*np.maximum(Vsw,0.0); return pv.mean(), pv.std(ddof=1)/np.sqrt(paths)

# --- QuantLib G2 reference ---
today=ql.Date(15,6,2026); ql.Settings.instance().evaluationDate=today
ts=ql.YieldTermStructureHandle(ql.FlatForward(today,r0,ql.Actual365Fixed(),ql.Continuous,ql.Annual))
g2=ql.G2(ts,a,sig,b,eta,rho)
cal=ql.NullCalendar(); idx=ql.IborIndex('i',ql.Period(1,ql.Years),0,ql.USDCurrency(),cal,ql.Unadjusted,False,ql.Actual365Fixed(),ts)
start=cal.advance(today,ql.Period(2,ql.Years)); mat=cal.advance(start,ql.Period(3,ql.Years))
sch=ql.Schedule(start,mat,ql.Period(1,ql.Years),cal,ql.Unadjusted,ql.Unadjusted,ql.DateGeneration.Forward,False)
swap=ql.VanillaSwap(ql.VanillaSwap.Payer,1.0,sch,K,ql.Thirty360(ql.Thirty360.BondBasis),sch,idx,0.0,ql.Actual365Fixed())
swpt=ql.Swaption(swap,ql.EuropeanExercise(start)); swpt.setPricingEngine(ql.FdG2SwaptionEngine(g2))
qlp=swpt.NPV()
m,se=mc()
print(f"G2++ European 2Yx3Y ATM payer (a={a},b={b},sig={sig*1e4:.0f},eta={eta*1e4:.0f}bp,rho={rho}):")
print(f"   my G2++ MC          : {m:.6e}  (+/- {1.96*se:.1e})")
print(f"   QuantLib FdG2 engine: {qlp:.6e}")
print(f"   diff: {abs(m-qlp)/1*1e4:.2f} bp of notional  ({'within MC error' if abs(m-qlp)<2*se else 'check'})")
