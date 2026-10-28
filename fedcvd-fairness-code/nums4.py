import pandas as pd, numpy as np, json
cols=["AUROC","DP_int_fx","EO_int_fx","minTPR_fx"]
def f(x): return f"{x:.3f}"
L=[r"\begin{tabular}{@{}lccccc@{}}",r"\toprule","Method & AUROC & DP$_{int}$ & EO$_{int}$ & EO range (min--max) & min TPR\\\\",r"\midrule"]
out={}
for sc in("natural","capped"):
    p=pd.read_csv(f"res4/post_{sc}.csv");e=pd.read_csv(f"res4/eopost_{sc}.csv")
    n=e.seed.nunique()
    L.append(rf"\multicolumn{{6}}{{l}}{{\emph{{{sc.capitalize()} ({n} seeds)}}}}\\")
    rows=[("FedAvg",p[p.method=="fedavg"]),("FedAvg + equal-TPR thr.",p[p.method=="fedavg__thrEO"]),("FedAvg + equal-sel. thr.",p[p.method=="fedavg__thrDP"])]
    for l in (0.5,1,4,16): rows.append((rf"EO penalty $\lambda$={l:g}",e[(e.method=="eo_pen")&(e.lam==l)]))
    for name,d in rows:
        d=d[d.seed.isin(e.seed.unique())]
        m=d[cols].mean(); s=d[cols].std(ddof=1)
        rg=f"{d.EO_int_fx.min():.2f}--{d.EO_int_fx.max():.2f}"
        L.append(f"{name} & "+" & ".join(f(m[c]) for c in cols[:3])+f" & {rg} & "+f(m[cols[3]])+r"\\")
        out[f"{sc}|{name}"]={c:float(m[c]) for c in cols}|{"EOsd":float(s["EO_int_fx"])}
    L.append(r"\midrule")
L[-1]=r"\bottomrule";L.append(r"\end{tabular}")
open("paper/tab4_eo.tex","w").write("\n".join(L).replace("nat &","fx &").replace("cap &","fx &"))
json.dump(out,open("res4/nums4.json","w"),indent=1)
# convergence
import os
d=[r for r in json.load(open("res4/convcheck2.json")) if r["scen"]!="natural"]+(json.load(open("res4/convcheck3.json")) if os.path.exists("res4/convcheck3.json") else [])+(json.load(open("res4/convcheck4.json")) if os.path.exists("res4/convcheck4.json") else []);import collections
g=collections.defaultdict(list)
for r in d: g[(r['scen'],r['pen'])].append([abs(a-b) for a,b in zip(r['prod'],r['half'])])
C=[r"\begin{tabular}{@{}llccccc@{}}",r"\toprule","Scenario & Penalty & $n$ & $\\overline{|\\Delta|}$ AUROC & $\\overline{|\\Delta|}$ DP$_{int}$ & $\\overline{|\\Delta|}$ EO$_{int}$ & max $|\\Delta|$ EO$_{int}$\\\\",r"\midrule"]
for (sc,pen),v in g.items():
    v=np.array(v)
    if pen=="eo" and (v[:,2]>=0.5).any():
        vc=v[v[:,2]<0.5]; C.append(f"{sc} & EO (excl.\\ collapsed fit) & {len(vc)} & {vc[:,0].mean():.4f} & {vc[:,1].mean():.4f} & {vc[:,2].mean():.4f} & {vc[:,2].max():.3f}\\\\")
    C.append(f"{sc} & {pen.upper()} & {len(v)} & {v[:,0].mean():.4f} & {v[:,1].mean():.4f} & {v[:,2].mean():.4f} & {v[:,2].max():.3f}\\\\")
for sc in ("capped","natural"):
    for nm,sel in (("oversample",lambda r:r['arm']=="oversample"),("other DP arms",lambda r:r['arm']!="oversample" and r['pen']=="dp")):
        v=np.array([[abs(a-b) for a,b in zip(r['prod'],r['half'])] for r in d if r['scen']==sc and sel(r)])
        if len(v): C.append(f"{sc} & DP: {nm} & {len(v)} & {v[:,0].mean():.4f} & {v[:,1].mean():.4f} & {v[:,2].mean():.4f} & {v[:,2].max():.3f}\\\\")
C+= [r"\bottomrule",r"\end{tabular}"]
open("paper/tab3_conv2.tex","w").write("\n".join(C))
