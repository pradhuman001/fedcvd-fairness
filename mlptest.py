import numpy as np, run2 as R, sys
from engine import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
rows=[]
for seed in (42,43):
  tr,tt,parts=R.split(seed,"capped",150,"random"); sc=StandardScaler().fit(tr[FEATS].values)
  Xte=sc.transform(tt[FEATS].values); yte=tt.target.values; ig=make_groups(tt,"agesex",55)
  cl=[dict(X=sc.transform(c[FEATS].values),y=c.target.values.astype(float),S=onehot(make_groups(c,"agesex",55),sorted(set(make_groups(tr,"agesex",55))))) for c in parts]
  for lam in (0,1,8,16):
    for T,lr in [(50,.25),(150,.25),(400,.25),(400,.1)]:
      m,nh,ok=train_safe(MLP,cl,dict(T=T,E=3,lr=lr,l2=1e-4,nmin=20,lam=lam,pen="dp",mode="naive"),seed)
      p=sigmoid(MLP.fwd(m.p,Xte)[0]); yh=(p>=np.quantile(p,1-yte.mean())).astype(int); dp,eo,_,_=gaps(yte,yh,ig)
      print(seed,"lam",lam,"T",T,"lr",lr,"AUC",round(roc_auc_score(yte,p),3),"DPint",round(dp,3),"EOint",round(eo,3),"ok",ok,"halv",nh,flush=True)
