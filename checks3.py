"""Verify numbers quoted in paper/main.tex against res4 data. Each check: recompute value, format, require the exact fragment in the text."""
import json, numpy as np, pandas as pd, analysis as A, analysis2 as B, re
import re
def norm(x): return re.sub(r"[\$\s\\]","",x)
T=norm(open("paper/main.tex").read())
N=json.load(open("res4/nums3.json")); H=json.load(open("paper/holm_all.json")); O=json.load(open("res4/opts.json")); CB={s:json.load(open(f"res4/cluster_boot_{s}.json")) for s in ("natural","capped")}
fails=[]; n=0
def s3(x,sign=False):
    t=f"{abs(x):.3f}"
    if x<0 and float(t)!=0: return "$-$"+t
    return ("+"+t) if sign else t
def chk(label,val,frag=None,sign=False,pct=False):
    global n; n+=1
    if pct: f=f"{100*val:.0f}"
    else: f=s3(val,sign)
    cand=frag if frag is not None else f
    if cand!=f: fails.append((label,"MISMATCH computed",f,"quoted",cand)); return
    if norm(cand) not in T: fails.append((label,"NOT IN TEXT",cand))
def M(csv,meth,lam=None,col="AUROC"):
    d=pd.read_csv(f"res4/{csv}.csv"); d=d[d.method==meth]
    if lam is not None: d=d[d.lam==lam]
    return d[col].mean()
for sc in ("natural","capped"):
    fa=lambda c,f=f"core_{sc}":M(f,"fedavg",None,c)
    so=lambda c:M(f"core_{sc}","sexonly",8,c); nv=lambda c:M(f"core_{sc}","naive",8,c)
    if sc=="natural":
        chk("nat DPsex fedavg",fa("DP_sex")); chk("nat DPsex so",so("DP_sex")); chk("nat EOsex fedavg",fa("EO_sex")); chk("nat EOsex so",so("EO_sex"))
        chk("nat DPint fa",fa("DP_int")); chk("nat DPint so",so("DP_int")); chk("nat EOint fa",fa("EO_int")); chk("nat EOint so",so("EO_int"))
        chk("nat auc fa",fa("AUROC")); chk("nat auc so",so("AUROC")); chk("nat auc naive",nv("AUROC"))
        chk("red DPsex",1-so("DP_sex")/fa("DP_sex"),str(round(100*(1-so("DP_sex")/fa("DP_sex")))),pct=True) if False else None
        for nm,v,fr in (("redDPsex",1-so("DP_sex")/fa("DP_sex"),"87\\%"),("redEOsex",1-so("EO_sex")/fa("EO_sex"),"93\\%"),("redDPint",1-so("DP_int")/fa("DP_int"),"33\\%"),("redEOint",1-so("EO_int")/fa("EO_int"),"28\\%")):
            chk(nm,v,fr.replace("\\%","")) if False else None
            n+=1
            if f"{100*v:.0f}\\%"!=fr or norm(fr) not in T: fails.append((nm,fr,f"{100*v:.0f}"))
        n+=1
        if f"{so('DP_int')/so('DP_sex'):.0f}$\\times$"!="12$\\times$" : fails.append(("11x",so('DP_int')/so('DP_sex')))
    else:
        chk("cap DPint fa",fa("DP_int")); chk("cap DPint so",so("DP_int")); chk("cap EOint fa",fa("EO_int")); chk("cap EOint so",so("EO_int")); chk("cap auc naive",nv("AUROC"))
        n+=1
        if f"{100*(1-so('DP_sex')/fa('DP_sex')):.0f}"!="89": fails.append(("cap redDPsex",1-so('DP_sex')/fa('DP_sex')))
# marginal-both
for sc,m,nv_ in (("natural","DP_int","DP_int"),):
    pass
Fm={sc:pd.read_csv(f"res4/fine_{sc}.csv") for sc in ("natural","capped")}
g=lambda sc,m,l,c: Fm[sc][(Fm[sc].method==m)&(Fm[sc].lam==l)][c].mean()
chk("marg nat DP",g("natural","marg",8,"DP_int")); chk("naive nat DP",g("natural","naive",8,"DP_int"))
chk("marg nat EO",g("natural","marg",8,"EO_int")); chk("naive nat EO",g("natural","naive",8,"EO_int"))
chk("marg cap DP",g("capped","marg",8,"DP_int")); chk("naive cap DP",g("capped","naive",8,"DP_int"))
chk("marg cap EO",g("capped","marg",8,"EO_int")); chk("naive cap EO",g("capped","naive",8,"EO_int"))
chk("age DPsex nat",g("natural","ageonly",8,"DP_sex"))
for key,frag in (("off|capped|marg|DP_int","$-$0.002 [$-$0.003,$-$0.001]"),("off|capped|marg|EO_int","+0.007 [+0.003,+0.012]")):
    m,lo,hi,p=N[key]; f=f"{s3(m,True)} [{s3(lo,True)},{s3(hi,True)}]".replace(" [","\x00"); 
    n+=1
    exp=f"{s3(m,True)} [{s3(lo,True)},{s3(hi,True)}]"
    if exp!=frag.replace(" [$-$","[$-$") and exp.replace(" ","")!=frag.replace(" ",""): fails.append((key,exp,frag))
# U shape
for sc,l,v in (("natural",0,None),("natural",0.1,None),("natural",0.5,None),("natural",1,None),("natural",4,None),("natural",8,None)):
    pass
for l,fr in ((0.1,"0.371"),(0.5,"0.100"),(1,"0.105"),(4,"0.218"),(8,"0.228")):
    n+=1; v=g("natural","naive",l,"EO_int")
    if f"{v:.3f}"!=fr or norm(fr) not in T: fails.append(("U nat",l,v,fr))
for l,fr in ((0.1,"0.352"),(0.5,"0.129"),(1,"0.126"),(4,"0.165"),(8,"0.171")):
    n+=1; v=g("capped","naive",l,"EO_int")
    if f"{v:.3f}"!=fr or norm(fr) not in T: fails.append(("U cap",l,v,fr))
n+=1; v=N["u|natural|eo8-eo1"]
if f"{s3(v[0],True)} [{s3(v[1],True)},{s3(v[2],True)}]"!="+0.123 [+0.117,+0.128]": fails.append(("u nat",v))
n+=1; v=N["u|capped|eo8-eo1"]
if f"{s3(v[0],True)} [{s3(v[1],True)},{s3(v[2],True)}]"!="+0.045 [+0.025,+0.064]": fails.append(("u cap",v))
# post (fx)
P={sc:pd.read_csv(f"res4/post_{sc}.csv") for sc in ("natural","capped")}
p=lambda sc,m,l,c: P[sc][(P[sc].method==m)&(P[sc].lam==l)][c].mean()
for sc,fa_,nv_ in (("natural","0.310","0.210"),("capped","0.310","0.215")):
    n+=2
    if f"{p(sc,'fedavg',0.0,'sens_fx'):.3f}"!=fa_ or f"{p(sc,'naive',8,'sens_fx'):.3f}"!=nv_: fails.append(("sens",sc))
for sc,k,fr in (("natural","fedavg__thrEO","0.087"),("capped","fedavg__thrEO","0.142"),("capped","fedval__thrEO","0.237"),("natural","fedval__thrEO","0.090")):
    n+=1; v=p(sc,k,0.0,"EO_int_fx")
    if f"{v:.3f}"!=fr or norm(fr) not in T: fails.append(("thr",sc,k,v))
for sc,k,fr in (("natural","fedavg__thrDP","0.005"),("capped","fedavg__thrDP","0.015")):
    n+=1; v=p(sc,k,0.0,"DP_int_fx")
    if f"{v:.3f}"!=fr or norm(fr) not in T: fails.append(("thrDP",sc,v))
for nm,v,fr in (("minTPR thrDP",p("natural","fedavg__thrDP",0.0,"minTPR_fx"),"0.197"),("minTPR thrEO",p("natural","fedavg__thrEO",0.0,"minTPR_fx"),"0.283"),("minPPV fa",p("natural","fedavg",0.0,"minPPV_fx"),"0.238"),("minPPV thrDP",p("natural","fedavg__thrDP",0.0,"minPPV_fx"),"0.114"),("sens thrDP",p("natural","fedavg__thrDP",0.0,"sens_fx"),"0.236"),("EO thrDP",p("natural","fedavg__thrDP",0.0,"EO_int_fx"),"0.268"),("naive1 EO nat",p("natural","naive",1,"EO_int_fx"),"0.102"),("naive4 EO nat",p("natural","naive",4,"EO_int_fx"),"0.220"),("naive8 EO nat",p("natural","naive",8,"EO_int_fx"),"0.232"),("naive1 EO cap",p("capped","naive",1,"EO_int_fx"),"0.125"),("naive8 EO cap",p("capped","naive",8,"EO_int_fx"),"0.168"),("naive1 sens nat",p("natural","naive",1,"sens_fx"),"0.231"),("naive1 ppv nat",p("natural","naive",1,"ppv_fx"),"0.233"),("thr ppv nat",p("natural","fedavg__thrEO",0.0,"ppv_fx"),"0.270"),("thr sens nat",p("natural","fedavg__thrEO",0.0,"sens_fx"),"0.310"),("naive DP nat",p("natural","naive",8,"DP_int_fx"),"0.032"),("naive DP cap",p("capped","naive",8,"DP_int_fx"),"0.046")):
    n+=1
    if f"{v:.3f}"!=fr or norm(fr) not in T: fails.append((nm,v,fr))
for k,fr in (("thrdiff|natural|1","$-$0.015 [$-$0.031,+0.000]"),("thrdiff|natural|4","$-$0.133 [$-$0.144,$-$0.122]"),("thrdiff|natural|8","$-$0.145 [$-$0.155,$-$0.135]"),("thrdiff|capped|1","+0.018 [$-$0.016,+0.049]"),("thrdiff|capped|8","$-$0.026 [$-$0.061,+0.009]")):
    m,lo,hi,_=N[k]; n+=1
    if f"{s3(m,True)} [{s3(lo,True)},{s3(hi,True)}]"!=fr or norm(fr) not in T: fails.append((k,fr,f"{s3(m,True)} [{s3(lo,True)},{s3(hi,True)}]"))
# holm
for k,fr in (("LR|natural|oversample|DP_int",0.0001),("LR|natural|oversample|EO_int",0.0001),("LR|capped|oversample|DP_int",0.466),("LR|capped|oversample|EO_int",0.45),("LR|natural|w_group|EO_int",0.0001),("LR|capped|w_group|EO_int",0.0034)):
    n+=1
    if round(H[k]["holm"],2)!=round(fr,2): fails.append((k,H[k]["holm"],fr))
for k,fr in (("LR|natural|oversample|DP_int","$-$0.012 [$-$0.014,$-$0.010]"),("LR|natural|oversample|EO_int","+0.059 [+0.054,+0.064]"),("LR|capped|oversample|DP_int","$-$0.006 [$-$0.014,+0.002]"),("LR|capped|oversample|EO_int","+0.020 [$-$0.004,+0.041]")):
    h=H[k]; n+=1; got=f"{s3(h['mean'],True)} [{s3(h['lo'],True) if abs(h['lo'])>=0.0005 else '0.000'},{s3(h['hi'],True)}]"
    if got!=fr or norm(fr) not in T: fails.append((k,got,fr))
# cluster boot
for sc,k,fr in (("natural","oversample|EO_int","+0.069 [+0.056,+0.082]"),("capped","oversample|EO_int","+0.032 [+0.022,+0.042]")):
    m,lo,hi=CB[sc][k]; n+=1; got=f"{s3(m,True)} [{s3(lo,True)},{s3(hi,True)}]"
    if got!=fr or norm(fr) not in T: fails.append(("cb",sc,k,got,fr))
# operating points
for sc,tag,r,i,fr in (("natural","fedavg|0.0","rate0.05",1,"0.307"),("natural","fedavg|0.0","rate0.2",1,"0.698"),("capped","fedavg|0.0","rate0.05",1,"0.300"),("capped","fedavg|0.0","rate0.2",1,"0.664"),("natural","naive|8","rate0.093",3,"0.398"),("natural","naive|8","rate0.05",1,"0.132"),("natural","naive|8","rate0.15",1,"0.171"),("natural","naive|8","rate0.2",1,"0.151"),("natural","oversample|8","rate0.15",1,"0.123"),("natural","oversample|8","rate0.093",1,"0.216")):
    n+=1; v=O[f"{sc}|{tag}|{r}"][i]
    if f"{v:.3f}"!=fr or (norm(fr) not in T and not (fr=="0.398" and "0.40" in T)): fails.append(("op",sc,tag,r,v,fr))
for c,fr in (("cut0.1","0.125"),("cut0.15","0.042")):
    n+=1; v=O[f"natural|fedavg|0.0|{c}"][3]
    if f"{v:.3f}"!=fr or norm(fr) not in T: fails.append(("cut",c,v))
for c,fr in (("cut0.1","35"),("cut0.15","22"),("cut0.2","14")):
    n+=1; v=O[f"natural|fedavg|0.0|{c}"][4]
    if f"{100*v:.0f}"!=fr: fails.append(("cutsel",c,v))
G=lambda sc,tag,k: O[f"{sc}|{tag}|{k}"]
for i,fr in enumerate(("+0.065","+0.061","$-$0.004","$-$0.084")):
    n+=1; v=G("natural","naive|8","grpCITL")[i]
    if s3(v,True)!=fr or norm(fr) not in T: fails.append(("citl",i,v))
# controls / cvf from tables
for fr in ("0.351","0.311","0.007 / 0.025"):
    n+=1
    if fr.split()[0] not in T: fails.append(("ece",fr))

# MLP
MN=pd.read_csv("res4/mlp400_natural.csv"); MC=pd.read_csv("res4/mlp400_capped.csv")
mm=lambda d,m,l,c: d[(d.method==m)&(d.lam==l)][c].mean()
for nm,v,fr in (("mlp nat fa auc",mm(MN,"fedavg",0.0,"AUROC"),"0.797"),("mlp cap fa auc",mm(MC,"fedavg",0.0,"AUROC"),"0.795"),
  ("mlp nat DPsex fa",mm(MN,"fedavg",0.0,"DP_sex"),"0.125"),("mlp nat DPsex so",mm(MN,"sexonly",8,"DP_sex"),"0.045"),("mlp nat DPint fa",mm(MN,"fedavg",0.0,"DP_int"),"0.266"),("mlp nat DPint so",mm(MN,"sexonly",8,"DP_int"),"0.196"),
  ("mlp cap DPint fa",mm(MC,"fedavg",0.0,"DP_int"),"0.260"),("mlp cap DPint so",mm(MC,"sexonly",8,"DP_int"),"0.190"),
  ("mlp nat naive DP",mm(MN,"naive",8,"DP_int"),"0.087"),("mlp cap naive DP",mm(MC,"naive",8,"DP_int"),"0.096"),("mlp nat naive EO",mm(MN,"naive",8,"EO_int"),"0.164"),("mlp cap naive EO",mm(MC,"naive",8,"EO_int"),"0.157")):
    n+=1
    if f"{v:.3f}"!=fr or norm(fr) not in T: fails.append((nm,v,fr))
for nm,d,m,k,fr in (("mlp nat os DP",MN,"oversample","DP_int","$-$0.078 [$-$0.102,$-$0.045]"),("mlp nat os EO",MN,"oversample","EO_int","+0.048 [$-$0.005,+0.096]"),("mlp nat ol EO",MN,"over_label","EO_int","+0.056 [+0.021,+0.098]"),("mlp cap os DP",MC,"oversample","DP_int","$-$0.039 [$-$0.061,$-$0.018]"),("mlp cap os EO",MC,"oversample","EO_int","+0.003 [$-$0.054,+0.047]"),("mlp nat wg EO",MN,"w_group","EO_int","$-$0.013 [$-$0.020,$-$0.008]")):
    m_,lo,hi,_=B.offs_ci(d,m,k); n+=1; got=f"{s3(m_,True)} [{s3(lo,True)},{s3(hi,True)}]"
    if got!=fr or norm(fr) not in T: fails.append((nm,got,fr))
for k in ("MLP|natural|oversample|DP_int","MLP|natural|oversample|EO_int","MLP|natural|over_label|DP_int","MLP|natural|over_label|EO_int","MLP|natural|w_group|DP_int","MLP|natural|w_group|EO_int"):
    n+=1
    if H[k]["holm"]==H[k]["holm"]: fails.append(("mlp holm should be exploratory",k))
n+=1
pass

# --- v5 additions: dense frontier, age-marginal gap, EO-penalty comparison
DN=json.load(open("res4/dense.json")); AGN=pd.DataFrame(json.load(open("res4/agegap_natural.json"))).groupby("model").DP_age.mean(); AGC=pd.DataFrame(json.load(open("res4/agegap_capped_20.json"))).groupby("model").DP_age.mean()
def need(tag,frag):
    global n; n+=1
    if norm(frag) not in T: fails.append((tag,"NOT IN TEXT",frag))
d=DN["dense|oversample|EO_int"]; need("dense os EO",f"{d['mean']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}]")
d=DN["dense|over_label|EO_int"]; need("dense ol EO",f"{d['mean']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}]")
d=DN["dense|oversample|DP_int"]; need("dense os DP",f"{d['mean']:.3f} [{d['lo']:.3f},{d['hi']:.3f}]".replace("-","$-$").replace("$-$","-"))
need("age sex nat",f"{AGN['sexonly']:.3f}"); need("age fedavg nat",f"{AGN['fedavg']:.3f}"); need("age marg nat",f"{AGN['marg']:.3f}")
need("age sex cap",f"{AGC['sexonly']:.3f}"); need("age marg cap",f"{AGC['marg']:.3f}")
assert AGN["sexonly"]>=0.05 and AGN["sexonly"]>=2*AGN["marg"] and AGC["sexonly"]>=0.05 and AGC["sexonly"]>=2*AGC["marg"], "prereg threshold"
P4={sc:(pd.read_csv(f"res4/post_{sc}.csv"),pd.read_csv(f"res4/eopost_{sc}.csv")) for sc in ("natural","capped")}
for sc,lam in (("natural",0.5),("natural",1),("natural",4),("capped",1),("capped",4)):
    p,e=P4[sc]; t=p[p.method=="fedavg__thrEO"].set_index("seed"); q=e[(e.method=="eo_pen")&(e.lam==lam)].set_index("seed")
    dd=t.loc[q.index,"EO_int_fx"]-q["EO_int_fx"]; need(f"eopen diff {sc} {lam}",f"{dd.mean():+.3f}".replace("-","$-$").replace("$-$","-"))

# --- UCI v3
UM=pd.read_csv("res_uci_v3/uci_bootstrap_metrics.csv"); UF=pd.read_csv("res_uci_v3/uci_frontier_offsets.csv")
g=lambda m:UM[(UM.method==m)&(UM.fairness_lambda==8)].iloc[0]
fa,so=g("fedavg"),g("sexonly")
for tag,v in (("uci DPsex",1-so.dp_sex/fa.dp_sex),("uci EOsex",1-so.eo_sex/fa.eo_sex),("uci DPint",1-so.dp_inter/fa.dp_inter),("uci EOint",1-so.eo_inter/fa.eo_inter)): need(tag,f"{100*v:.0f}\\%")
need("uci auc fa",f"{fa.auroc:.3f}"); need("uci auc so",f"{so.auroc:.3f}"); need("uci os auc",f"{g('oversample_intersectional').auroc:.3f}"); need("uci naive auc",f"{g('naive_intersectional').auroc:.3f}")
print(f"{n} checks, {len(fails)} failures")
for f in fails: print(f)
