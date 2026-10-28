"""Convergence evidence for the penalized LR models: production schedule vs. half learning rate / double rounds (capped seed 42,43)."""
import numpy as np, run2 as R, json
from engine import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
res=[]
for seed in (42,43):
  tr,tt,parts=R.split(seed,"capped",150,"random"); sc=StandardScaler().fit(tr[FEATS].values)
  Xte=sc.transform(tt[FEATS].values); yte=tt.target.values; ig=make_groups(tt,"agesex",55); sx=make_groups(tt,"sex")
  for grp in ("agesex","sex","age","marg"):
    cl=[]
    for c in parts:
      if grp=="sex": S=onehot(make_groups(c,"sex"),["F","M"])
      elif grp=="age": S=onehot(np.where(c["age"].values>=55,"old","young"),["young","old"])
      elif grp=="marg": S=np.hstack([onehot(make_groups(c,"sex"),["F","M"]),onehot(np.where(c["age"].values>=55,"old","young"),["young","old"])])
      else: S=onehot(make_groups(c,"agesex",55),sorted(set(make_groups(tr,"agesex",55))))
      cl.append(dict(X=sc.transform(c[FEATS].values),y=c.target.values.astype(float),S=S))
    for lam in (1,8,16):
      T,lr=sched(grp,lam); out={}
      for tag,(T_,lr_) in (("prod",(T,lr)),("half",(2*T,lr/2))):
        m=train_fed(Logit,cl,dict(T=T_,E=3,lr=lr_,l2=1e-4,nmin=20,lam=lam,pen="dp",mode="naive"),seed); p=sigmoid(Logit.fwd(m.p,Xte)[0])
        yh=(p>=np.quantile(p,1-yte.mean())).astype(int); dp,eo,_,_=gaps(yte,yh,ig); dps,_,_,_=gaps(yte,yh,sx)
        out[tag]=(roc_auc_score(yte,p),dp,eo,dps)
      res.append(dict(seed=seed,grp=grp,lam=lam,prod=out["prod"],half=out["half"]))
      print(seed,grp,lam,[round(v,3) for v in out["prod"]],[round(v,3) for v in out["half"]],flush=True)
json.dump(res,open("res4/convcheck.json","w"))
