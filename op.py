"""Operating-point sensitivity from saved test predictions (res4/preds): gaps at several selection rates / fixed risk cut-offs,
group-wise AUROC and calibration-in-the-large. Output res4/opts.json + paper/tab2_op.tex"""
import glob, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from engine import gaps
GN = ["young F", "young M", "old F", "old M"]
out = {}; L = []
def grp(z): return np.where(z["age"] >= 55, 2, 0) + z["sex"].astype(int)
for sc in ["natural", "capped"]:
    acc = {}
    for f in sorted(glob.glob(f"res4/preds/{sc}_*.npz")):
        z = np.load(f, allow_pickle=True); y = z["y"].astype(int); g = grp(z); tags = list(z["tags"])
        for tag in ["fedavg|0.0", "naive|8", "oversample|8", "w_group|8"]:
            if tag not in tags: continue
            p = z["P"][tags.index(tag)].astype(float)
            for rate in (0.05, 0.093, 0.15, 0.20):
                yh = (p >= np.quantile(p, 1 - rate)).astype(int)
                dp, eo, mt, tg = gaps(y, yh, g)
                tpr = [yh[(g == a) & (y == 1)].mean() for a in range(4)]
                acc.setdefault((tag, f"rate{rate}"), []).append([dp, eo, yh[y == 1].mean(), tpr[0]])
            if tag == "fedavg|0.0":
                for cut in (0.10, 0.15, 0.20):
                    yh = (p >= cut).astype(int)
                    if yh.sum() == 0: continue
                    dp, eo, mt, tg = gaps(y, yh, g); tpr = [yh[(g == a) & (y == 1)].mean() for a in range(4)]
                    acc.setdefault((tag, f"cut{cut}"), []).append([dp, eo, yh[y == 1].mean(), tpr[0], yh.mean()])
            acc.setdefault((tag, "grpAUROC"), []).append([roc_auc_score(y[g == a], p[g == a]) for a in range(4)])
            acc.setdefault((tag, "grpCITL"), []).append([p[g == a].mean() - y[g == a].mean() for a in range(4)])
    for (tag, k), v in acc.items():
        out[f"{sc}|{tag}|{k}"] = np.mean(v, 0).tolist(); out[f"{sc}|{tag}|{k}|sd"] = np.std(v, 0).tolist()
    L.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{sc.capitalize()}}}}}\\\\")
    for tag, nm in [("fedavg|0.0", "FedAvg"), ("naive|8", "Naive penalty $\\lambda{=}8$"), ("oversample|8", "Oversample $\\lambda{=}8$")]:
        cells = []
        for rate in ("rate0.05", "rate0.093", "rate0.15", "rate0.2"):
            k = f"{sc}|{tag}|{rate}"
            cells.append(f"{out[k][0]:.3f} / {out[k][1]:.3f}" if k in out else "--")
        L.append(f"{nm} & " + " & ".join(cells) + " \\\\")
    if sc == "natural": L.append("\\midrule")
open("paper/tab2_op.tex", "w").write("\n".join(L)); json.dump(out, open("res4/opts.json", "w"), indent=1)
for sc in ["natural", "capped"]:
    for tag in ["fedavg|0.0", "naive|8"]:
        print(sc, tag, "grpAUROC", np.round(out[f"{sc}|{tag}|grpAUROC"], 3), "CITL", np.round(out[f"{sc}|{tag}|grpCITL"], 3))
    for k in ["rate0.05", "rate0.093", "rate0.15", "rate0.2", "cut0.1", "cut0.15", "cut0.2"]:
        kk = f"{sc}|fedavg|0.0|{k}"
        if kk in out: print(sc, k, np.round(out[kk], 3))
