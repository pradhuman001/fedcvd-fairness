import numpy as np, run2 as R
from engine import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
seed=42; tr,tt,parts=R.split(seed,"capped",150,"random"); sc=StandardScaler().fit(tr[FEATS].values)
Xte=sc.transform(tt[FEATS].values); yte=tt.target.values; ig=make_groups(tt,"agesex",55)
cats=sorted(set(make_groups(tr,"agesex",55)))
cl=[dict(X=sc.transform(c[FEATS].values),y=c.target.values.astype(float),S=onehot(make_groups(c,"agesex",55),cats)) for c in parts]
for lam in (1,4,8,16,32,64,128):
  for T,lr in [(100,.3),(300,.15),(600,.075),(1200,.0375)]:
    m=train_fed(Logit,cl,dict(T=T,E=3,lr=lr,l2=1e-4,nmin=20,lam=lam,pen="eo",mode="naive"),seed); p=sigmoid(Logit.fwd(m.p,Xte)[0])
    if not np.isfinite(p).all(): print(lam,T,lr,"NaN"); continue
    yh=(p>=np.quantile(p,1-yte.mean())).astype(int); dp,eo,_,_=gaps(yte,yh,ig)
    if not np.isfinite(p).all(): print(lam,T,lr,"NaN"); continue
    print(lam,T,lr,"AUC",round(roc_auc_score(yte,p),3),"DP",round(dp,3),"EO",round(eo,3),flush=True)
