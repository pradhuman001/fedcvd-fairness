"""Age coefficient (standardized) of the intersectional-penalty LR vs lambda, natural, seeds 42-46; EO_int under rule (q)."""
import numpy as np, json, ccprep as C
from engine import *
rows=[]
for seed in range(42,47):
    cl,Xte,yte,ig=C.prep(seed,"natural","naive")
    for lam in (0.0,0.5,1,2,4,8):
        T,lr=sched("agesex",lam,"dp")
        m=train_fed(Logit,cl,dict(T=T,E=3,lr=lr,l2=1e-4,nmin=20,lam=lam,pen="dp",mode="naive"),seed)
        w=np.array(m.p["w"]).ravel(); p=sigmoid(Logit.fwd(m.p,Xte)[0])
        yh=(p>=np.quantile(p,1-yte.mean())).astype(int); dp,eo,mt,_=gaps(yte,yh,ig)
        rows.append(dict(seed=seed,lam=lam,age=float(w[0]),sex=float(w[6]),EO_int=float(eo),minTPR=float(mt)))
        json.dump(rows,open("res4/agecoef.json","w"))
import pandas as pd
d=pd.DataFrame(rows);print(d.groupby("lam")[["age","sex","EO_int"]].agg(["mean","std"]).round(3).to_string())
