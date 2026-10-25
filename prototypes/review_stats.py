import numpy as np
from math import sqrt
from statistics import NormalDist
N=NormalDist()
rng=np.random.default_rng(0)
def sim_pairs(n,p1,delta,pd,reps=20000):
    # paired binary: P(01)=b, P(10)=c ; delta = p2-p1 = b - c ; pd = b + c
    b=(pd+delta)/2; c=(pd-delta)/2
    p11=p1-c; p00=1-p11-b-c
    assert min(b,c,p11,p00)>=0,(b,c,p11,p00)
    cat=rng.choice(4,size=(reps,n),p=[p00,b,c,p11])  # 1: arm2 wins only, 2: arm1 wins only
    nb=(cat==1).sum(1); nc=(cat==2).sum(1)
    d=(nb-nc)/n
    se=np.sqrt(np.maximum((nb+nc)/n - d**2,1e-12)/n)
    return d,se,nb,nc
print("T-1.9 non-inferiority (bound>-10pts), Wald one-sided 95%, p1=0.5")
for pd in (0.32,0.40,0.49):
    for delta in (0.0,-0.03,-0.05):
        for n in (250,350):
            d,se,_,_=sim_pairs(n,0.5,delta,pd)
            print(f" pd={pd:.2f} true_delta={delta:+.2f} n={n}: P(pass)={np.mean(d-1.645*se>-0.10):.2f}")
print("Screen n=50 (point >= -15): P(pass)")
for delta in (0,-0.10,-0.20):
    d,se,_,_=sim_pairs(50,0.5,delta,0.45)
    print(f" true_delta={delta:+.2f}: {np.mean(d>=-0.15):.2f}")
print("SP-1 rule n=50, pd=0.45: P(branch)")
for delta in (0,-0.05,-0.10,-0.20):
    d,se,_,_=sim_pairs(50,0.5,delta,0.45)
    print(f" true={delta:+.2f}: v1(>=-10)={np.mean(d>=-0.10):.2f} sweep={np.mean((d>=-0.20)&(d<-0.10)):.2f} stop(<-20)={np.mean(d<-0.20):.2f}")
print("P(C3 >= C4) at n=50 when C4 truly better by x (pd 0.45)")
for x in (0,0.05,0.10):
    d,se,_,_=sim_pairs(50,0.45,x,0.45)   # d = C4 - C3
    print(f" C4-C3 true={x:+.2f}: P(C3>=C4)={np.mean(d<=0):.2f}")
# MDE two-sided 0.05 power 0.8 for paired, pd=0.45
for n in (50,250,350):
    print("MDE n",n, round((1.96+0.8416)*sqrt(0.45/n)*100,1),"pts")
