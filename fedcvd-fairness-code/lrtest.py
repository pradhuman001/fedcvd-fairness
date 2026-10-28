import numpy as np, pandas as pd, run2 as R, time
from engine import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
seed=42; tr,tt,parts=R.split(seed,"natural",150,"random"); sc=StandardScaler().fit(tr[FEATS].values)
Xte=sc.transform(tt[FEATS].values); yte=tt.target.values; ig=make_groups(tt,"agesex",55); cats=sorted(set(make_groups(tr,"agesex",55)))
cl=[]
for c in parts:
    c=c.copy(); c["g"]=make_groups(c,"agesex",55); cl.append(dict(X=sc.transform(c[FEATS].values),y=c.target.values.astype(float),S=onehot(c["g"].values,cats)))
for T,lr in [(100,.3),(150,.3),(100,.4)]:
  for lam in (1,8,16):
    t=time.time(); m=train_fed(Logit,cl,dict(T=T,E=3,lr=lr,l2=1e-4,nmin=20,lam=lam,pen="dp",mode="naive"),seed); p=sigmoid(Logit.fwd(m.p,Xte)[0])
    ok=np.isfinite(p).all()
    if ok: yh=(p>=np.quantile(p,1-yte.mean())).astype(int); dp,eo,_,_=gaps(yte,yh,ig); print(T,lr,lam,round(roc_auc_score(yte,p),3),round(dp,3),round(eo,3),round(time.time()-t),"s",flush=True)
    else: print(T,lr,lam,"diverged",flush=True)
