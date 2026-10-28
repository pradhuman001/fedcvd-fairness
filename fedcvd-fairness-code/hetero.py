import numpy as np, pandas as pd, run2 as R, json
from engine import make_groups
out={}
for scen in ["natural","dir03","dir"]:
    tr,tt,parts=R.split(42,scen,150,"random")
    G=np.array(sorted(set(make_groups(tr,"agesex",55).astype(str))))
    glob=pd.Series(make_groups(tr,"agesex",55).astype(str)).value_counts(normalize=True).reindex(G).values
    prev=[p.target.mean() for p in parts]; tv=[]
    for p in parts:
        m=pd.Series(make_groups(p,"agesex",55).astype(str)).value_counts(normalize=True).reindex(G).fillna(0).values
        tv.append(0.5*np.abs(m-glob).sum())
    out[scen]=dict(K=len(parts),n_med=float(np.median([len(p) for p in parts])),prev_sd=float(np.std(prev)),prev_min=float(min(prev)),prev_max=float(max(prev)),tv_group_mean=float(np.mean(tv)),tv_group_max=float(np.max(tv)))
    print(scen,{k:round(v,3) for k,v in out[scen].items()})
json.dump(out,open("res4/hetero.json","w"),indent=1)
