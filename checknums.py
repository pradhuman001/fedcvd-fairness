import json, numpy as np, pandas as pd
J=json.load(open("paper/round2.json")); H=json.load(open("paper/holm_all.json")); O=json.load(open("paper/offsets.json")); Rm=json.load(open("results.json"))
HE=json.load(open("res4/hetero.json")); CAP=json.load(open("paper/capabl.json"))
bad=[]; n=0
def chk(lab, typed, actual, tol=0.0006):
    global n; n+=1
    if actual is None or np.isnan(actual) or abs(typed-actual)>tol: bad.append((lab,typed,actual))
M=lambda s,m,k: Rm[s]["main"][m][k][0]
# masking
chk("nat sexDP fedavg",0.107,M("natural","fedavg","DP_sex")); chk("nat sexDP sexonly",0.013,M("natural","sexonly","DP_sex"))
chk("nat sexEO fedavg",0.251,M("natural","fedavg","EO_sex")); chk("nat sexEO sexonly",0.014,M("natural","sexonly","EO_sex"))
chk("nat DPint fed",0.245,M("natural","fedavg","DP_int")); chk("nat DPint sex",0.169,M("natural","sexonly","DP_int"))
chk("nat EOint fed",0.438,M("natural","fedavg","EO_int")); chk("nat EOint sex",0.327,M("natural","sexonly","EO_int"))
chk("nat auroc fed",0.795,M("natural","fedavg","AUROC")); chk("nat auroc sex",0.787,M("natural","sexonly","AUROC"))
chk("red DP sex",88,100*(1-M("natural","sexonly","DP_sex")/M("natural","fedavg","DP_sex")),1); chk("red EO sex",94,100*(1-M("natural","sexonly","EO_sex")/M("natural","fedavg","EO_sex")),1)
chk("red DP int",31,100*(1-M("natural","sexonly","DP_int")/M("natural","fedavg","DP_int")),1); chk("red EO int",25,100*(1-M("natural","sexonly","EO_int")/M("natural","fedavg","EO_int")),1)
chk("ratio 13",13,M("natural","sexonly","DP_int")/M("natural","sexonly","DP_sex"),0.6)
chk("cap DPint fed",0.243,M("capped","fedavg","DP_int")); chk("cap DPint sex",0.163,M("capped","sexonly","DP_int")); chk("cap EOint fed",0.437,M("capped","fedavg","EO_int")); chk("cap EOint sex",0.308,M("capped","sexonly","EO_int"))
chk("naive auroc nat",0.680,M("natural","naive","AUROC")); chk("naive auroc cap",0.683,M("capped","naive","AUROC"))
mm=pd.read_csv("res4/mlp400_natural.csv"); x=mm[(mm.lam==0)|(mm.lam==8)].groupby("method").mean(numeric_only=True)
chk("mlp sexonly sexDP",0.115,x.loc["sexonly","DP_sex"]); chk("mlp fedavg sexDP",0.114,x.loc["fedavg","DP_sex"])
# sens
chk("sens fedavg",0.303,J["post|natural|fedavg|sens_fx"]); chk("sens naive",0.214,J["post|natural|naive|sens_fx"])
chk("fedavg central auroc nat",0.795,J["central|natural|fedavg_auroc"]); chk("fedavg central auroc cap",0.795,J["central|capped|fedavg_auroc"])
# thresholds
for sc,dp,eo in [("natural",(0.006,0.036),(0.083,0.215)),("capped",(0.015,0.051),(0.150,0.154))]:
    chk(sc+" thrDP",dp[0],J[f"post|{sc}|fedavg__thrDP|DP_int_fx"]); chk(sc+" naive DP",dp[1],J[f"post|{sc}|naive|DP_int_fx"])
    chk(sc+" thrEO",eo[0],J[f"post|{sc}|fedavg__thrEO|EO_int_fx"]); chk(sc+" naive EO",eo[1],J[f"post|{sc}|naive|EO_int_fx"])
d=J["postdiff|natural|fedavg__thrDP-naive|DP_int_fx"]; chk("pd dp",-0.030,d[0]); chk("pd dp lo",-0.032,d[1]); chk("pd dp hi",-0.028,d[2])
d=J["postdiff|natural|fedavg__thrEO-naive|EO_int_fx"]; chk("pd eo",-0.132,d[0]); chk("pd eo lo",-0.145,d[1]); chk("pd eo hi",-0.118,d[2])
d=J["postdiff|capped|fedavg__thrEO-naive|EO_int_fx"]; chk("pd cap eo",-0.004,d[0]); chk("lo",-0.038,d[1]); chk("hi",0.031,d[2])
d=J["postdiff|capped|fedval__thrEO-naive|EO_int_fx"]; chk("pd val eo",0.073,d[0]); chk("lo",0.034,d[1]); chk("hi",0.116,d[2])
chk("thrEO sens",0.304,J["post|natural|fedavg__thrEO|sens_fx"]); chk("thrEO ppv",0.273,J["post|natural|fedavg__thrEO|ppv_fx"]); chk("naive ppv",0.217,J["post|natural|naive|ppv_fx"])
chk("w_group EO fx nat",0.207,J["post|natural|w_group|EO_int_fx"])
chk("minTPR fedavg",0.0,J["post|natural|fedavg|minTPR_fx"]); chk("thrDP minTPR",0.196,J["post|natural|fedavg__thrDP|minTPR_fx"])
chk("thrDP ppv",0.238,J["post|natural|fedavg__thrDP|ppv_fx"]); chk("minPPV fed",0.233,J["post|natural|fedavg|minPPV_fx"]); chk("minPPV thrDP",0.112,J["post|natural|fedavg__thrDP|minPPV_fx"]); chk("thrDP EO",0.264,J["post|natural|fedavg__thrDP|EO_int_fx"])
chk("fedval nat eo",0.088,J["post|natural|fedval__thrEO|EO_int_fx"]); chk("fedval cap eo",0.227,J["post|capped|fedval__thrEO|EO_int_fx"])
# oversampling
for sc,a,b in [("natural",-0.029,0.076),("capped",-0.013,0.045)]:
    chk(sc+" over DP",a,H[f"LR|{sc}|oversample|DP_int"]["mean"]); chk(sc+" over EO",b,H[f"LR|{sc}|oversample|EO_int"]["mean"])
chk("holm nat max",0.0024,max(H[f"LR|natural|{m}|{k}"]["holm"] for m in["oversample"] for k in["DP_int","EO_int"]),0.0002)
chk("holm cap EO",0.029,H["LR|capped|oversample|EO_int"]["holm"],0.001); chk("holm cap DP",0.070,H["LR|capped|oversample|DP_int"]["holm"],0.001)
chk("holm mlp nat EO",0.15,H["MLP|natural|oversample|EO_int"]["holm"],0.01); chk("mlp lab EO nat holm",0.034,H["MLP|natural|over_label|EO_int"]["holm"],0.001); chk("mlp wg cap holm",0.034,H["MLP|capped|w_group|EO_int"]["holm"],0.003)
chk("wg nat EO",-0.014,H["LR|natural|w_group|EO_int"]["mean"]); chk("wg cap EO",-0.008,H["LR|capped|w_group|EO_int"]["mean"]); chk("wg nat DP",0.003,H["LR|natural|w_group|DP_int"]["mean"])
chk("wg holm",0.0024,max(H["LR|natural|w_group|EO_int"]["holm"],H["LR|capped|w_group|EO_int"]["holm"]),0.0002)
chk("perlam naive EO nat1",0.085,J["perlam|natural|naive|EO_int|1.0"]); chk("perlam naive EO nat16",0.221,J["perlam|natural|naive|EO_int|16.0"])
for lam in ["1.0","4.0","8.0","16.0"]: chk("perlam nat over EO>0 "+lam,1,float(J[f"perlam|natural|oversample|EO_int|{lam}"][1]>0))
for lam in ["2.0","4.0","8.0","16.0"]: chk("perlam cap over EO>0 "+lam,1,float(J[f"perlam|capped|oversample|EO_int|{lam}"][1]>0))
chk("perlam cap1 ns",1,float(J["perlam|capped|oversample|EO_int|1.0"][1]<0<J["perlam|capped|oversample|EO_int|1.0"][2]))
# cvf
c=lambda k: J[k][0]
chk("cvf nat DP c",-0.033,c("cvf|natural|oversample|DP_int|central")); chk("cvf cap DP c",-0.012,c("cvf|capped|oversample|DP_int|central"))
chk("cvf nat EO c",0.067,c("cvf|natural|oversample|EO_int|central")); chk("cvf cap EO c",0.038,c("cvf|capped|oversample|EO_int|central"))
d=J["cvf|natural|oversample|EO_int|diff"]; chk("diff EO",0.010,d[0]); chk("lo",0.006,d[1]); chk("hi",0.013,d[2]); chk("diff DP nat",0.004,c("cvf|natural|oversample|DP_int|diff"))
d=J["cvf|capped|oversample|EO_int|diff"]; chk("diff cap EO",0.007,d[0]); chk("lo",-0.012,d[1]); chk("hi",0.027,d[2])
# controls
oc=lambda s,k,m: O[s]["off"][f"{k}|{m}"][0]
chk("lab DP",-0.014,oc("natural","ctl_over_label","DP_int")); chk("lab EO",0.058,oc("natural","ctl_over_label","EO_int"))
chk("wlab DP",-0.018,oc("natural","ctl_w_label","DP_int")); chk("wlab EO",0.050,oc("natural","ctl_w_label","EO_int")); chk("wcell DP",-0.011,oc("natural","ctl_w_cell","DP_int")); chk("wcell EO",0.048,oc("natural","ctl_w_cell","EO_int"))
# calibration
chk("ece fed",0.014,J["post|natural|fedavg|ECE"]); chk("ece over nat",0.352,J["post|natural|oversample|ECE"]); chk("ece over cap",0.314,J["post|capped|oversample|ECE"])
chk("recal nat",0.006,J["post|natural|oversample|ECE_recal"]); chk("recal cap",0.027,J["post|capped|oversample|ECE_recal"])
# mlp
chk("mlp fedavg auroc nat",0.794,J["mlp|natural|fedavg|AUROC"]); chk("mlp fedavg auroc cap",0.793,J["mlp|capped|fedavg|AUROC"])
for sc,vals in [("natural",(-0.030,0.030)),("capped",(-0.008,0.007))]:
    chk("mlp over DP "+sc,vals[0],H[f"MLP|{sc}|oversample|DP_int"]["mean"]); chk("mlp over EO "+sc,vals[1],H[f"MLP|{sc}|oversample|EO_int"]["mean"])
chk("mlp lab EO nat",0.036,H["MLP|natural|over_label|EO_int"]["mean"]); chk("mlp lab DP nat",-0.020,H["MLP|natural|over_label|DP_int"]["mean"]); chk("mlp lab DP cap",-0.022,H["MLP|capped|over_label|DP_int"]["mean"])
chk("mlp wg EO nat",-0.012,H["MLP|natural|w_group|EO_int"]["mean"]); chk("mlp wg EO cap",-0.006,H["MLP|capped|w_group|EO_int"]["mean"])
chk("mlp nat EO lo",0.003,H["MLP|natural|oversample|EO_int"]["lo"]); chk("mlp nat EO hi",0.054,H["MLP|natural|oversample|EO_int"]["hi"])
# LSO
for k,t in [("natural|oversample|DP_int_fx",-0.037),("natural|oversample|EO_int_fx",0.062),("capped|oversample|DP_int_fx",-0.011),("capped|oversample|EO_int_fx",0.018),("natural|over_label|EO_int_fx",0.051),("capped|over_label|EO_int_fx",0.038),("natural|w_group|EO_int_fx",-0.016) if False else ("natural|w_group|EO_int_fx",-0.016),("capped|w_group|EO_int_fx",-0.005)]:
    sc,m,met=k.split("|"); chk("lso "+k,t,J[f"lso|{sc}|{m}|{met}"][0] if f"lso|{sc}|{m}|{met}" in J else J[f"lso|{sc}|{ {'w_group':'w_group'}.get(m,m)}|{met}"][0])
d=J["lso|capped|oversample|EO_int_fx"]; chk("lso lo",-0.007,d[1]); chk("lso hi",0.042,d[2]); d=J["lso|capped|oversample|DP_int_fx"]; chk("lso dp lo",-0.016,d[1]); chk("hi",-0.005,d[2])
# cluster
for k,v in [("natural|oversample|DP_int",(-0.029,-0.033,-0.025)),("natural|oversample|EO_int",(0.078,0.064,0.092)),("capped|oversample|DP_int",(-0.013,-0.015,-0.011)),("capped|oversample|EO_int",(0.045,0.033,0.057)),("natural|w_group|EO_int",(-0.014,-0.019,-0.010)),("capped|w_group|EO_int",(-0.008,-0.011,-0.005))]:
    for i in range(3): chk("cluster "+k+str(i),v[i],J["cluster|"+k][i])
# shrink
mx=max(abs(O[s]["off"][f"{k}|{m}"][0]) for s in O for k in ["shrink","cut45_shrink","cut65_shrink"] for m in ["DP_int","EO_int"]); chk("shrink max",0.004,mx,0.0006)
# cutoffs
chk("cut45 nat EO",0.007,oc("natural","cut45_oversample","EO_int")); chk("cut45 cap EO",0.000,oc("capped","cut45_oversample","EO_int"),0.0006); chk("cut65 nat",0.091,oc("natural","cut65_oversample","EO_int")); chk("cut65 cap",0.066,oc("capped","cut65_oversample","EO_int"))
chk("age3 nat",0.122,oc("natural","age3_oversample","EO_int")); chk("age3 cap",0.033,oc("capped","age3_oversample","EO_int")); chk("cut45 lo",-0.006,O["natural"]["off"]["cut45_oversample|EO_int"][1]); chk("hi",0.018,O["natural"]["off"]["cut45_oversample|EO_int"][2])
# hetero
chk("prev sd",0.019,HE["natural"]["prev_sd"]); chk("tv",0.059,HE["natural"]["tv_group_mean"]); chk("prev min",0.049,HE["natural"]["prev_min"]); chk("prev max",0.150,HE["natural"]["prev_max"])
chk("dir03 sd",0.071,HE["dir03"]["prev_sd"]); chk("dir03 tv",0.282,HE["dir03"]["tv_group_mean"]); chk("dir sd",0.144,HE["dir"]["prev_sd"]); chk("dir tv",0.420,HE["dir"]["tv_group_mean"])

def dd(k): return J[k]
for sc,m,met,t in [("dir03","oversample","EO_int",0.070),("dir03","oversample","DP_int",-0.030),("dir03","w_group","EO_int",0.010),("dir","w_group","EO_int",0.052),("dir","oversample","EO_int",0.041)]:
    chk(f"dir {sc} {m} {met} fed",t,J[f"dir|{sc}|{m}|{met}|fed"][0])
chk("dir03 over EO cen",0.070,J["dir|dir03|oversample|EO_int|cen"][0]); chk("dir03 over DP cen",-0.033,J["dir|dir03|oversample|DP_int|cen"][0]); chk("dir03 wg EO cen",-0.015,J["dir|dir03|w_group|EO_int|cen"][0]); chk("dir wg EO cen",-0.015,J["dir|dir|w_group|EO_int|cen"][0]); chk("dir over EO cen",0.072,J["dir|dir|oversample|EO_int|cen"][0])
d=J["dir|dir03|w_group|EO_int|fed"]; chk("lo",-0.000,d[1]); chk("hi",0.021,d[2]); d=J["dir|dir03|w_group|EO_int|diff"]; chk("diff",0.026,d[0]); chk("lo",0.015,d[1]); chk("hi",0.036,d[2])
d=J["dir|dir|w_group|EO_int|fed"]; chk("lo",0.032,d[1]); chk("hi",0.076,d[2]); d=J["dir|dir|oversample|EO_int|diff"]; chk("diff",-0.030,d[0]); chk("lo",-0.052,d[1]); chk("hi",-0.008,d[2])
for cap,m,met,v in [(75,"oversample","EO_int",(0.034,0.009,0.056)),(300,"oversample","EO_int",(0.063,0.042,0.082))]:
    for i in range(3): chk(f"cap{cap}{i}",v[i],CAP[f"cap{cap}|{m}|{met}"][i])
chk("cap150 EO",0.045,H["LR|capped|oversample|EO_int"]["mean"]); chk("nat EO",0.076,H["LR|natural|oversample|EO_int"]["mean"])
eg={}
for sc in ["natural","capped"]:
    d=pd.read_csv(f"res3/eg_{sc}.csv"); g=d.groupby(["method","eps"]).mean(numeric_only=True)
    eg[sc]=g
allg=pd.concat(eg.values())
chk("eg dp min",0.016,allg.loc["eg_dp","DP_int"].min()); chk("eg dp max",0.042,allg.loc["eg_dp","DP_int"].max())
chk("eg eo min",0.056,allg.loc["eg_eo","EO_int"].min()); chk("eg eo max",0.098,allg.loc["eg_eo","EO_int"].max())
chk("eg sel min",0.38,allg.sel_rate.min(),0.01); chk("eg sel max",0.45,allg.sel_rate.max(),0.01); chk("eg auc min",0.64,allg.AUROC.min(),0.005); chk("eg auc max",0.69,allg.AUROC.max(),0.005)
ppv=0.093*allg.sens/allg.sel_rate; chk("eg ppv min",0.12,ppv.min(),0.006); chk("eg ppv max",0.15,ppv.max(),0.006)
chk("mlp EO hi",0.054,H["MLP|natural|oversample|EO_int"]["hi"])

print("checked",n); print("MISMATCHES:",*bad,sep="\n")
