import json, numpy as np, os
import analysis as A
import make_results as R  # builds figs + results.json on import? (guarded by __main__)

os.makedirs("paper", exist_ok=True)
res = {}
for scen in ["natural", "capped"]:
    res[scen] = {"main": R.table_main(scen), "off": R.offs_table(scen)}


def f3(x):
    return f"{x:.3f}"


def cell(o, bold_thresh=True):
    m, lo, hi, p = o
    s = f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
    return s.replace("-", "$-$")


def main_table():
    names = [("fedavg", "FedAvg"), ("sexonly", "Sex-only"), ("naive", "Naive inter."),
             ("oversample", "Oversample"), ("shrink", "Shrinkage")]
    L = []
    for scen, title in [("natural", "Natural (52 states, $\\sim$6K each; 20 seeds)"),
                        ("capped", "Capped ($n{=}150$ per client; 20 seeds)")]:
        t = res[scen]["main"]
        L.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{title}}}}}\\\\")
        for k, nm in names:
            r = t[k]
            L.append(f"{nm} & " + " & ".join(f"{r[c][0]:.3f}$\\pm${r[c][1]:.3f}" for c in ["AUROC", "DP_sex", "DP_int", "EO_sex", "EO_int"]) + " \\\\")
        L.append("\\midrule" if scen == "natural" else "")
    open("paper/tab_main.tex", "w").write("\n".join(L))


def off_table():
    rows = [("Oversample", "oversample"), ("Shrinkage", "shrink")]
    L = []
    for nm, k in rows:
        parts = []
        for scen in ["natural", "capped"]:
            o = res[scen]["off"]
            for met in ["DP_int", "EO_int"]:
                parts.append(cell(o[(k, met)]))
        L.append(f"{nm} & " + " & ".join(parts) + " \\\\")
    open("paper/tab_off.tex", "w").write("\n".join(L))


def ctl_table():
    rows = [("Oversample (cells, noise)", "ctl_oversample"), ("\\quad orig.\\ $n_k$ in FedAvg", "ctl_over_origagg"),
            ("\\quad no noise (pure duplication)", "ctl_over_nonoise"),
            ("Label-balance by duplication only", "ctl_over_label"),
            ("Loss reweight: labels only", "ctl_w_label"), ("Loss reweight: (group,label) cells", "ctl_w_cell"),
            ("Loss reweight: groups only", "ctl_w_group")]
    L = []
    for nm, k in rows:
        parts = []
        for scen in ["natural", "capped"]:
            o = res[scen]["off"]
            for met in ["DP_int", "EO_int"]:
                parts.append(cell(o[(k, met)]) if (k, met) in o else "--")
        L.append(f"{nm} & " + " & ".join(parts) + " \\\\")
    open("paper/tab_ctl.tex", "w").write("\n".join(L))


def rob_table():
    rows = [("Age cut 45: oversample", "cut45_oversample"), ("Age cut 45: shrinkage", "cut45_shrink"),
            ("Age cut 65: oversample", "cut65_oversample"), ("Age cut 65: shrinkage", "cut65_shrink"),
            ("3 age bands $\\times$ sex: oversample", "age3_oversample")]
    L = []
    for nm, k in rows:
        parts = []
        for scen in ["natural", "capped"]:
            o = res[scen]["off"]
            for met in ["DP_int", "EO_int"]:
                parts.append(cell(o[(k, met)]) if (k, met) in o else "--")
        L.append(f"{nm} & " + " & ".join(parts) + " \\\\")
    open("paper/tab_rob.tex", "w").write("\n".join(L))


def mlp_table():
    names=[("fedavg","FedAvg"),("sexonly","Sex-only"),("naive","Naive inter."),("oversample","Oversample"),("shrink","Shrinkage")]
    L=[]
    for scen,title in [("natural","Natural"),("capped","Capped")]:
        d=A.load("mlp",scen)
        L.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{title}}}}}\\\\")
        for k,nm in names:
            x=d[(d.method==k)&((d.lam==8)|(k=="fedavg"))]
            L.append(f"{nm} & "+" & ".join(f"{x[c].mean():.3f}" for c in ["AUROC","DP_sex","DP_int","EO_sex","EO_int"])+" \\\\")
    open("paper/tab_mlp.tex","w").write("\n".join(L))


def holm_report():
    fam = []
    for scen in ["natural", "capped"]:
        for k in ["oversample"]:
            for met in ["DP_int", "EO_int"]:
                fam.append((scen, k, met, res[scen]["off"][(k, met)]))
        for met in ["DP_int", "EO_int"]:
            fam.append((scen, "w_group", met, res[scen]["off"][("ctl_w_group", met)]))
    adj = A.holm([f[3][3] for f in fam])
    for f, a in zip(fam, adj):
        print(f[:3], f"p={f[3][3]:.4f} holm={a:.4f}")


if __name__ == "__main__":
    main_table(); off_table(); ctl_table(); rob_table(); holm_report()
    json.dump({s: {"off": {f"{k[0]}|{k[1]}": [float(x) for x in v] for k, v in res[s]["off"].items()}} for s in res},
              open("paper/offsets.json", "w"), indent=1)
    print("tables ok")
