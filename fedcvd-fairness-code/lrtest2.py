import numpy as np, run2 as R, time
from engine import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
seed=42; tr,tt,parts=R.split(seed,"capped",150,"random"); sc=StandardScaler().fit(tr[FEATS].values)
Xte=sc.transform(tt[FEATS].values); yte=tt.target.values; ig=make_groups(tt,"agesex",55); sx=make_groups(tt,"sex")
for grp in ("sex","marg","age"):
  cl=[]
  for c in parts:
    if grp=="sex": S=onehot(make_groups(c,"sex"),["F","M"])
    elif grp=="age": S=onehot(np.where(c["age"].values>=55,"old","young"),["young","old"])
    else: S=np.hstack([onehot(make_groups(c,"sex"),["F","M"]),onehot(np.where(c["age"].values>=55,"old","young"),["young","old"])])
    cl.append(dict(X=sc.transform(c[FEATS].values),y=c.target.values.astype(float),S=S))
  for lam in (4,8,16):
    for lr,T in [(.3,100),(.15,200),(.075,400),(.0375,800)]:
        m=train_fed(Logit,cl,dict(T=T,E=3,lr=lr,l2=1e-4,nmin=20,lam=lam,pen="dp",mode="naive"),seed); p=sigmoid(Logit.fwd(m.p,Xte)[0])
        yh=(p>=np.quantile(p,1-yte.mean())).astype(int); dp,eo,_,_=gaps(yte,yh,ig); dps,_,_,_=gaps(yte,yh,sx)
        print(grp,lam,lr,T,"AUC",round(roc_auc_score(yte,p),3),"DPsex",round(dps,3),"DPint",round(dp,3),"EOint",round(eo,3),flush=True)
