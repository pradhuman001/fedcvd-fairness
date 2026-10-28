"""Extended convergence evidence: production schedule vs. half learning rate / double rounds, for all arms (penalty types, oversample, group re-weight, EO penalty).
capped seeds 42-46 (all arms) and natural seeds 42,43 (lambda=8 core arms)."""
import numpy as np, pandas as pd, json, sys, run2 as R
from engine import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
res=[]
def prep(seed,scen,arm):
    tr,tt,parts=R.split(seed,scen,150,"random"); sc=StandardScaler().fit(tr[FEATS].values)
    cats4=sorted(set(make_groups(tr,"agesex",55))); cl=[]
    for j,c in enumerate(parts):
        c=c.copy(); grp="agesex" if arm in ("naive","oversample","w_group","eo","over_label") else arm
        if arm=="sex": g=make_groups(c,"sex"); cats=["F","M"]; S=onehot(g,cats)
        elif arm=="age": S=onehot(np.where(c["age"].values>=55,"old","young"),["young","old"])
        elif arm=="marg": S=np.hstack([onehot(make_groups(c,"sex"),["F","M"]),onehot(np.where(c["age"].values>=55,"old","young"),["young","old"])])
        else:
            c["g"]=make_groups(c,"agesex",55)
            if arm=="oversample": c=oversample(c,"g",np.random.default_rng(seed*1000+j),add_noise=True)
            if arm=="over_label": c=oversample(c.assign(g="all"),"g",np.random.default_rng(seed*1000+j)); c["g"]=make_groups(c,"agesex",55)
            S=onehot(c["g"].values,cats4)
        d=dict(X=sc.transform(c[FEATS].values),y=c.target.values.astype(float),S=S)
        if arm=="w_group": d["sw"]=cell_weights(c,"g","group")
        cl.append(d)
    return cl,sc.transform(tt[FEATS].values),tt.target.values,make_groups(tt,"agesex",55)
def run(seed,scen,arm,lam,pen="dp"):
    cl,Xte,yte,ig=prep(seed,scen,arm)
    grp={"naive":"agesex","oversample":"agesex","w_group":"agesex","eo":"agesex","over_label":"agesex"}.get(arm,arm)
    T,lr=sched(grp,lam,pen); out={}
    for tag,(T_,lr_) in (("prod",(T,lr)),("half",(2*T,lr/2))):
        m=train_fed(Logit,cl,dict(T=T_,E=3,lr=lr_,l2=1e-4,nmin=20,lam=lam,pen=pen,mode="naive"),seed); p=sigmoid(Logit.fwd(m.p,Xte)[0])
        yh=(p>=np.quantile(p,1-yte.mean())).astype(int); dp,eo,_,_=gaps(yte,yh,ig)
        out[tag]=(float(roc_auc_score(yte,p)),float(dp),float(eo))
    r=dict(seed=seed,scen=scen,arm=arm,lam=lam,pen=pen,prod=out["prod"],half=out["half"]); res.append(r); print(r,flush=True)
    pass

for seed in range(42,47):
    for lam in (8,16): run(seed,"natural","over_label",lam)
