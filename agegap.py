"""Age-marginal DP gap (old vs young selection rate, rule q), lambda=8, natural, seeds 42-46."""
import numpy as np, json, ccprep as C
from engine import *
import sys; SC=sys.argv[1]; rows=[]
NSEED=int(__import__('os').environ.get('NSEED','5'))
for seed in range(42,42+NSEED):
    for name,arm,lam in (("fedavg","naive",0.0),("sexonly","sex",8),("ageonly","age",8),("marg","marg",8),("naive","naive",8)):
        cl,Xte,yte,ig=C.prep(seed,SC,arm)
        grp={"naive":"agesex"}.get(arm,arm); T,lr=sched(grp,lam,"dp")
        m=train_fed(Logit,cl,dict(T=T,E=3,lr=lr,l2=1e-4,nmin=20,lam=lam,pen="dp",mode="naive"),seed); p=sigmoid(Logit.fwd(m.p,Xte)[0])
        yh=(p>=np.quantile(p,1-yte.mean())).astype(int); old=np.array([str(g).startswith("old") for g in ig])
        dpa=abs(yh[old].mean()-yh[~old].mean()); fem=np.array([str(g).endswith("_F") for g in ig]); dps=abs(yh[fem].mean()-yh[~fem].mean())
        rows.append(dict(seed=seed,model=name,DP_age=float(dpa),DP_sex=float(dps))); print(rows[-1],flush=True)
        json.dump(rows,open("res4/agegap_"+SC+("_20" if NSEED>5 else "")+".json","w"))
