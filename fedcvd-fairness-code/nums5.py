"""Equal-lambda table, paired oversample - label-balance differences, cost at EO-optimal lambda."""
import pandas as pd, numpy as np, json
import analysis as A
out={}
L=[r"\begin{tabular}{@{}lcccccc@{}}",r"\toprule",r"& $\lambda$ & \multicolumn{2}{c}{Oversample} & \multicolumn{2}{c}{Naive penalty} & $\Delta$EO\\",r"& & AUROC & EO$_{int}$ & AUROC & EO$_{int}$ & (os $-$ naive)\\",r"\midrule"]
for sc in ("natural","capped"):
    c=A.load("core",sc); 
    L.append(rf"\multicolumn{{7}}{{l}}{{\emph{{{sc.capitalize()} ({c.seed.nunique()} seeds)}}}}\\")
    for lam in sorted(set(c.lam)-{0.0}):
        o=c[(c.method=="oversample")&(c.lam==lam)].set_index("seed"); n=c[(c.method=="naive")&(c.lam==lam)].set_index("seed")
        s=o.index.intersection(n.index)
        if len(s)<5: continue
        d=(o.loc[s,"EO_int"]-n.loc[s,"EO_int"]).values; m,lo,hi=A.ci_boot(d)[:3]
        out[f"eqlam|{sc}|{lam}"]=dict(os_auc=float(o.loc[s,"AUROC"].mean()),os_eo=float(o.loc[s,"EO_int"].mean()),nv_auc=float(n.loc[s,"AUROC"].mean()),nv_eo=float(n.loc[s,"EO_int"].mean()),d=float(m),lo=float(lo),hi=float(hi))
        f=lambda x:f"{x:+.3f}".replace("-","$-$")
        L.append(f"& {lam:g} & {o.loc[s,'AUROC'].mean():.3f} & {o.loc[s,'EO_int'].mean():.3f} & {n.loc[s,'AUROC'].mean():.3f} & {n.loc[s,'EO_int'].mean():.3f} & {f(m)} [{f(lo)}, {f(hi)}]\\\\")
    if sc=="natural": L.append(r"\midrule")
L+=[r"\bottomrule",r"\end{tabular}"]
open("paper/tab5_eqlam.tex","w").write("\n".join(L))
# paired oversample - label-balance offsets (dense/core frontiers as in the primary tests)
def share(sc,met="EO_int"):
    core=A.load("core",sc);ctl=A.load("controls",sc)
    o=A.offsets(core,"oversample","naive",met); l=A.offsets(ctl,"over_label","naive",met)
    sd_o=sorted(core[core.method=="oversample"].seed.unique())
    # offsets() drops seeds in order of groupby; rebuild per-seed dict
    def per(df,meth):
        r={}
        for s,d in df.groupby("seed"):
            fr=d[d.method=="naive"].sort_values("AUROC")[["AUROC",met]].values; m=d[d.method==meth]
            of=[row[met]-A.interp_gap(fr,row["AUROC"]) for _,row in m.iterrows()]; of=[x for x in of if not np.isnan(x)]
            if of: r[s]=np.mean(of)
        return r
    po,pl=per(core,"oversample"),per(ctl,"over_label"); S=sorted(set(po)&set(pl))
    d=np.array([po[s]-pl[s] for s in S]); m,lo,hi=A.ci_boot(d)[:3]
    return dict(n=len(S),os=float(np.mean([po[s] for s in S])),ol=float(np.mean([pl[s] for s in S])),diff=float(m),lo=float(lo),hi=float(hi))
for sc in ("natural","capped"):
    for met in ("EO_int","DP_int"): out[f"share|{sc}|{met}"]=share(sc,met)
json.dump(out,open("res4/nums5.json","w"),indent=1)
for k,v in out.items():
    if k.startswith("share"): print(k,{a:round(b,3) for a,b in v.items()})
