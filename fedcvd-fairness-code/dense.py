import pandas as pd, numpy as np, json
import analysis as A
core=pd.read_csv("res4/core_natural.csv"); ctl=pd.read_csv("res4/controls_natural.csv"); de=pd.read_csv("res4/dense_natural.csv")
cols=["seed","method","lam","AUROC","DP_int","EO_int"]
base=pd.concat([core[core.method.isin(["naive","oversample"])][cols], ctl[ctl.method.isin(["over_label","w_group"])][cols], de[cols]])
base=base.drop_duplicates(["seed","method","lam"])
# controls also contain naive/oversample at same lam: prefer core (already dropped dups by order)
out={}
def run(name,lams):
    d=base[base.lam.isin(lams)]
    for meth in ("oversample","over_label","w_group"):
        for met in ("EO_int","DP_int"):
            o=A.offsets(d,meth,"naive",met); m,lo,hi=A.ci_boot(o)[:3] if len(A.ci_boot(o))>=3 else (None,)*3
            out[f"{name}|{meth}|{met}"]=dict(n=len(o),mean=float(o.mean()),lo=float(lo),hi=float(hi),pos=int((o>0).sum()),min=float(o.min()),max=float(o.max()),sd=float(o.std(ddof=1)))
run("core",[1,4,8,16]); run("dense",[0.5,1,2,4,8,16]); run("dense_strong",[0.5,1,2,4,8,16])
json.dump(out,open("res4/dense.json","w"),indent=1)
for k,v in out.items():
    if k.startswith(("core","dense|")): print(k,{a:(round(b,3) if isinstance(b,float) else b) for a,b in v.items()})
# append dense-grid rows to the control table (natural columns only)
def cell(k):
    v=out[k]; f=lambda x:(f"{x:+.3f}").replace("-","$-$")
    return f"{f(v['mean'])} [{f(v['lo'])}, {f(v['hi'])}]"
t=open("paper/tab_ctl.tex").read()
if "dense" not in t:
    rows=[f"Oversample, dense natural grid & {cell('dense|oversample|DP_int')} & {cell('dense|oversample|EO_int')} & -- & -- \\\\",
          f"Label-balance, dense natural grid & {cell('dense|over_label|DP_int')} & {cell('dense|over_label|EO_int')} & -- & -- \\\\",
          f"Loss reweight: groups only, dense natural grid & {cell('dense|w_group|DP_int')} & {cell('dense|w_group|EO_int')} & -- & -- \\\\"]
    open("paper/tab_ctl.tex","w").write(t.rstrip("\n")+"\n"+"\n".join(rows))
