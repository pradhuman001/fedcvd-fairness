"""Convergence of PENALIZED models: LR (T=80 vs 300, lr .15) and MLP (T=50 vs 150, lr .25); AUROC/DP/EO on seeds 42,43."""
import numpy as np, pandas as pd, run2 as R
from engine import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
rows=[]
for scen in ["natural","capped"]:
  for seed in (42,43):
    tr,tt,parts=R.split(seed,scen,150,"random"); sc=StandardScaler().fit(tr[FEATS].values)
    Xte=sc.transform(tt[FEATS].values); yte=tt.target.values; ig=make_groups(tt,"agesex",55); cats=sorted(set(make_groups(tr,"agesex",55)))
    for over in (False,True):
        cl=[]
        for j,c in enumerate(parts):
            c=c.copy(); c["g"]=make_groups(c,"agesex",55)
            if over: c=oversample(c,"g",np.random.default_rng(seed*1000+j))
            cl.append(dict(X=sc.transform(c[FEATS].values),y=c.target.values.astype(float),S=onehot(c["g"].values,cats)))
        for model,M,cfgs in [("lr",Logit,[(80,.15),(300,.15)]),("mlp",MLP,[(50,.25),(150,.25)])]:
            for lam in (1,8,16):
                if model=="mlp" and lam==16: continue
                for T,lr in cfgs:
                    m=train_fed(M,cl,dict(T=T,E=3,lr=lr,l2=1e-4,nmin=20,lam=lam,pen="dp",mode="naive"),seed)
                    p=sigmoid(M.fwd(m.p,Xte)[0])
                    if not np.isfinite(p).all(): rows.append(dict(scen=scen,seed=seed,over=over,model=model,lam=lam,T=T,diverged=1)); continue
                    yh=(p>=np.quantile(p,1-yte.mean())).astype(int); dp,eo,_,_=gaps(yte,yh,ig)
                    rows.append(dict(scen=scen,seed=seed,over=over,model=model,lam=lam,T=T,AUROC=roc_auc_score(yte,p),DP_int=dp,EO_int=eo))
    pd.DataFrame(rows).to_csv("res3/converge.csv",index=False); print(scen,seed,"done",flush=True)
