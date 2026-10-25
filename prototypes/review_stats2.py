import numpy as np
rng=np.random.default_rng(1)
def run(n=250,delta=0.0,reps=20000,arms=3,add=0):
    p=rng.beta(2,2,size=(reps,n+add))          # scene difficulty, E[disc]=0.4
    c1=rng.random((reps,n+add))<p
    passes=[];incon=[]
    for a in range(arms):
        c4=rng.random((reps,n+add))<np.clip(p+delta,0,1)
        def bound(m):
            d=(c4[:,:m].astype(int)-c1[:,:m].astype(int))
            mean=d.mean(1); se=d.std(1,ddof=0)/np.sqrt(m)
            return mean, mean-1.645*se
        mean,lb=bound(n)
        ok=lb>-0.10
        if add:
            redo=(~ok)&(mean>=-0.05)
            m2,lb2=bound(n+add)
            ok=ok|(redo&(lb2>-0.10))
        passes.append(ok)
    P=np.array(passes)
    return P[0].mean(), P.all(0).mean()
for delta in (0.0,-0.05,-0.10):
    s,j=run(delta=delta)
    s2,j2=run(delta=delta,add=100)
    print(f"true delta {delta:+.2f}: single-arm pass {s:.2f}, all-3 pass {j:.2f} | with +100 inconclusive rule: single {s2:.3f}, all-3 {j2:.2f}")
