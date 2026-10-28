import numpy as np, pandas as pd, json, analysis as A, analysis2 as B
N={}
def f3(x): return ("$-$" if x<0 else "+")+f"{abs(x):.3f}" if False else (f"{x:.3f}".replace("-","$-$"))
def sg(x): return ("$-$%.3f"%abs(x)) if x<0 else ("+%.3f"%x)
def ci(t): m,lo,hi,p=t; return f"{sg(m)} [{sg(lo)}, {sg(hi)}]"
fine={s:pd.read_csv(f"res4/fine_{s}.csv") for s in ("capped","natural")}
# A masking at lambda=8
L=[]
import json as _j, pandas as _pd
AG={sc:_pd.DataFrame(_j.load(open("res4/agegap_capped_20.json" if sc=="capped" else f"res4/agegap_{sc}.json"))).groupby("model").DP_age.mean() for sc in ("natural","capped")}
names=[("fedavg","FedAvg",0.0),("sexonly","Sex-only",8),("ageonly","Age-only",8),("marg","Sex+age marginal",8),("naive","Intersectional",8)]
for sc in ("natural","capped"):
    d=fine[sc]; ns=d.seed.nunique()
    L.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{sc.capitalize()} ({ns} seeds)}}}}\\\\")
    fa=d[d.method=="fedavg"].mean(numeric_only=True)
    for m,nm,l in names:
        r=d[(d.method==m)&(d.lam==l)].mean(numeric_only=True)
        red=lambda k:"--" if m=="fedavg" else f"{100*(1-r[k]/fa[k]):.0f}\\%"
        L.append(f"{nm} & {r.AUROC:.3f} & {r.DP_sex:.3f} & {AG[sc][m]:.3f} & {r.DP_int:.3f} & {red('DP_int')} & {r.EO_int:.3f} & {red('EO_int')} \\\\")
        N[f"mask|{sc}|{m}"]=dict(AUROC=r.AUROC,DP_sex=r.DP_sex,DP_int=r.DP_int,EO_int=r.EO_int,redDP=1-r.DP_int/fa.DP_int,redEO=1-r.EO_int/fa.EO_int,redDPsex=1-r.DP_sex/fa.DP_sex)
    if sc=="natural": L.append("\\midrule")
open("paper/tab3_mask.tex","w").write("\n".join(L))
# B per-lambda frontier
L=[]
lams=[0.1,0.25,0.5,1,2,4,8,16]
for sc in ("capped","natural"):
    d=fine[sc]; L.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{sc.capitalize()} ({d.seed.nunique()} seeds)}}}}\\\\")
    for l in lams:
        row=[]
        for m in ("naive","marg","sexonly"):
            r=d[(d.method==m)&(d.lam==l)].mean(numeric_only=True)
            row+= [r.AUROC, r.DP_int, r.EO_int]
            N[f"fine|{sc}|{m}|{l}"]=dict(AUROC=r.AUROC,DP_int=r.DP_int,EO_int=r.EO_int)
        L.append(f"{l:g} & "+" & ".join(f"{v:.3f}" for v in row)+" \\\\")
    if sc=="capped": L.append("\\midrule")
open("paper/tab3_fine.tex","w").write("\n".join(L))
# EO(lam=8)-EO(lam=1) paired, naive; and EO(0.5)
for sc in ("capped","natural"):
    d=fine[sc]; a=d[(d.method=="naive")&(d.lam==8)].set_index("seed").EO_int; b=d[(d.method=="naive")&(d.lam==1)].set_index("seed").EO_int
    c=d[(d.method=="naive")&(d.lam==0.5)].set_index("seed").EO_int; z=d[(d.method=="fedavg")].set_index("seed").EO_int
    N[f"u|{sc}|eo8-eo1"]=A.ci_boot((a-b).values); N[f"u|{sc}|eo1-eo0.5"]=A.ci_boot((b-c).values); N[f"u|{sc}|eo1-eo0"]=A.ci_boot((b-z).values)
    for me in ("marg",):
        for met in ("DP_int","EO_int"): N[f"off|{sc}|{me}|{met}"]=B.offs_ci(d,me,met)
    for me in ("oversample","w_group"):
        for met in ("DP_int","EO_int"): N[f"off|{sc}|{me}|{met}"]=B.offs_ci(d,me,met)
# D thresholds vs penalty at small lambda (fx rule)
L=[]
for sc in ("natural","capped"):
    p=pd.read_csv(f"res4/post_{sc}.csv"); L.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{sc.capitalize()} ({p.seed.nunique()} seeds)}}}}\\\\")
    rows=[("FedAvg + equal-TPR thresholds","fedavg__thrEO",0.0),("\\quad (val.\\ split)","fedval__thrEO",0.0)]+[(f"Naive penalty $\\lambda{{=}}{l:g}$","naive",l) for l in (1,2,4,8)]
    for nm,m,l in rows:
        d=p[(p.method==m)&(p.lam==l)] if m!="naive" else p[(p.method=="naive")&(p.lam==l)]
        if len(d)==0: continue
        r=d.mean(numeric_only=True)
        L.append(f"{nm} & {r.AUROC:.3f} & {r.DP_int_fx:.3f} & {r.EO_int_fx:.3f} & {r.sens_fx:.3f} & {r.ppv_fx:.3f} \\\\")
        N[f"thr|{sc}|{m}|{l}"]=dict(AUROC=r.AUROC,DP=r.DP_int_fx,EO=r.EO_int_fx,sens=r.sens_fx,ppv=r.ppv_fx)
    t=p[p.method=="fedavg__thrEO"].set_index("seed").EO_int_fx
    for l in (1,2,4,8):
        n_=p[(p.method=="naive")&(p.lam==l)].set_index("seed").EO_int_fx
        if len(n_): N[f"thrdiff|{sc}|{l}"]=A.ci_boot((t-n_).values)
    if sc=="natural": L.append("\\midrule")
open("paper/tab3_thr.tex","w").write("\n".join(L))
# F global weights
L=[]
for s,nm in (("dir03","Dirichlet $\\alpha{=}1$"),("dir","Dirichlet $\\alpha{=}0.3$")):
    d=pd.read_csv(f"res4/glob_{s}.csv")
    for m,mn in (("w_group","local group counts"),("w_group_glob","global group counts")):
        r=[B.offs_ci(d,m,met) for met in ("DP_int","EO_int")]
        N[f"glob|{s}|{m}"]=r
        L.append(f"{nm} & {mn} & {ci(r[0])} & {ci(r[1])} \\\\")
open("paper/tab3_glob.tex","w").write("\n".join(L))
# G group-wise AUROC/CITL
o=json.load(open("res4/opts.json")); L=[]
for sc in ("natural","capped"):
    L.append(f"\\multicolumn{{9}}{{l}}{{\\emph{{{sc.capitalize()}}}}}\\\\")
    for tag,nm in (("fedavg|0.0","FedAvg"),("naive|8","Naive penalty $\\lambda{=}8$"),("oversample|8","Oversample $\\lambda{=}8$")):
        a=o[f"{sc}|{tag}|grpAUROC"]; c=o[f"{sc}|{tag}|grpCITL"]
        L.append(f"{nm} & "+" & ".join(f"{v:.2f}" for v in a)+" & "+" & ".join(f"{v:+.3f}".replace("-","$-$") for v in c)+" \\\\")
        N[f"grp|{sc}|{tag}"]=dict(auroc=a,citl=c)
    if sc=="natural": L.append("\\midrule")
open("paper/tab3_grp.tex","w").write("\n".join(L))
json.dump(N,open("res4/nums3.json","w"),indent=1,default=float)
for k,v in N.items():
    if k.startswith(("u|","off|","thrdiff","glob")): print(k,v)
# EG appendix table
L=[]
for sc in ("natural","capped"):
    own=pd.read_csv(f"res3/eg_{sc}.csv"); fl=pd.read_csv(f"res4/fairlearn/fairlearn_eg_{sc}.csv"); fl=fl[fl.kind=="expected"]
    L.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{sc.capitalize()} ({fl.seed.nunique()} seeds)}}}}\\\\")
    for src,d in (("own",own),("Fairlearn",fl)):
        for (kind,eps),x in d.groupby(["method","eps"],sort=False):
            k="DP" if kind=="eg_dp" else "EO"
            L.append(f"{src} EG, {k}-constr., $\\epsilon{{=}}{eps}$ & {x.AUROC.mean():.3f} & {x.DP_int.mean():.3f} & {x.EO_int.mean():.3f} & {x.sens.mean():.3f} & {x.sel_rate.mean():.3f} \\\\")
    if sc=="natural": L.append("\\midrule")
open("paper/tab3_eg.tex","w").write("\n".join(L))
