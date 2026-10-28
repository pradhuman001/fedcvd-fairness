"""Tables for the revised paper (round 2). Writes paper/tab2_*.tex and paper/round2.json (all numbers quoted in prose)."""
import json, os, numpy as np, pandas as pd
import analysis as A
import analysis2 as B

os.makedirs("paper", exist_ok=True)
J = {}
SC = ["natural", "capped"]


def fmt(o):
    m, lo, hi = o[:3]
    return (f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]").replace("-", "$-$")


def store(key, o):
    J[key] = [float(x) for x in o]


# ---------- central vs federated
rows = []
for sc in SC:
    cv = B.central_vs_fed(sc)
    for _, r in cv.iterrows():
        store(f"cvf|{sc}|{r.method}|{r.metric}|central", (r.central, r.central_lo, r.central_hi, np.nan))
        store(f"cvf|{sc}|{r.method}|{r.metric}|fed", (r.fed, r.fed_lo, r.fed_hi, np.nan))
        store(f"cvf|{sc}|{r.method}|{r.metric}|diff", (r.diff_fed_minus_central, r.diff_lo, r.diff_hi, np.nan))
L = []
for m, nm in [("oversample", "Oversample"), ("over_label", "Label-balance (dup.)"), ("w_group", "Group re-weight")]:
    for met, mn in [("DP_int", "DP"), ("EO_int", "EO")]:
        cells = []
        for sc in SC:
            cells += [fmt(J[f"cvf|{sc}|{m}|{met}|central"]), fmt(J[f"cvf|{sc}|{m}|{met}|fed"]),
                      fmt(J[f"cvf|{sc}|{m}|{met}|diff"])]
        L.append(f"{nm} ({mn}) & " + " & ".join(cells) + " \\\\")
open("paper/tab2_cvf.tex", "w").write("\n".join(L))

# ---------- baselines (fixed train-score threshold) + calibration, lambda=8
import glob
def recal_ece(scen):
    """ECE after a logit intercept shift so that mean(p) equals the nominal training prevalence (monotone: DP/EO unchanged)."""
    from scipy.optimize import brentq
    out = {}
    for f in sorted(glob.glob(f"res4/preds/{scen}_*.npz")):
        z = np.load(f, allow_pickle=True); y = z["y"].astype(float)
        for ti, tag in enumerate(z["tags"]):
            p = np.clip(z["P"][ti].astype(float), 1e-6, 1 - 1e-6); lg = np.log(p / (1 - p))
            tgt = 0.093
            try:
                sh = brentq(lambda c: (1 / (1 + np.exp(-(lg + c)))).mean() - tgt, -20, 20)
            except ValueError:
                continue
            q = 1 / (1 + np.exp(-(lg + sh)))
            bins = np.minimum((q * 10).astype(int), 9)
            e = sum((bins == k).mean() * abs(q[bins == k].mean() - y[bins == k].mean()) for k in range(10) if (bins == k).any())
            out.setdefault(tag, []).append(e)
    return {k: float(np.mean(v)) for k, v in out.items()}

names = [("fedavg", "FedAvg"), ("fedavg__thrDP", "\\quad + group thr.\\ (equal selection)"),
         ("fedavg__thrEO", "\\quad + group thr.\\ (equal TPR)"), ("fedval__thrEO", "\\quad + group thr.\\ (equal TPR, val.\\ split)"),
         ("sexonly", "Sex-only penalty"), ("naive", "Naive inter. penalty"), ("oversample", "Oversample"), ("w_group", "Group re-weight")]
def ci(x):
    o = A.ci_boot(np.asarray(x)); return f"{o[0]:.3f} [{o[1]:.3f}, {o[2]:.3f}]"
L = []
for sc in SC:
    d = pd.read_csv(f"res4/post_{sc}.csv")
    rec = recal_ece(sc)
    L.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{sc.capitalize()} ({d.seed.nunique()} seeds)}}}}\\\\")
    for k, nm in names:
        x = d[(d.method == k) & ((d.lam == 8) | (d.lam == 0))].sort_values("seed")
        g = lambda c: x[c].mean()
        ece = f"{g('ECE'):.3f}" if not np.isnan(g("ECE")) else "--"
        rc = rec.get("fedavg|0.0" if k == "fedavg" else f"{k}|8", rec.get(f"{k}|8.0", np.nan))
        rcs = f"{rc:.3f}" if not np.isnan(rc) else "--"
        L.append(f"{nm} & {g('AUROC'):.3f} & {ci(x.DP_int_fx)} & {ci(x.EO_int_fx)} & {g('sens_fx'):.3f} & {g('ppv_fx'):.3f} & {g('minPPV_fx'):.3f} & {ece} / {rcs} \\\\")
        for c in ["AUROC", "DP_int_fx", "EO_int_fx", "TPRgap_int_fx", "FPRgap_int_fx", "CITL_int", "ECE", "sens_fx", "ppv_fx", "minPPV_fx", "minTPR_fx"]:
            J[f"post|{sc}|{k}|{c}"] = float(g(c))
        J[f"post|{sc}|{k}|ECE_recal"] = float(rc)
    # paired differences (threshold rule minus penalty), fixed rule
    def paired(a, b, c):
        xa = d[(d.method == a) & ((d.lam == 8) | (d.lam == 0))].sort_values("seed")[c].values
        xb = d[(d.method == b) & ((d.lam == 8) | (d.lam == 0))].sort_values("seed")[c].values
        return A.ci_boot(xa - xb)
    for a, b, c in [("fedavg__thrDP", "naive", "DP_int_fx"), ("fedavg__thrEO", "naive", "EO_int_fx"), ("fedavg__thrEO", "w_group", "EO_int_fx"),
                    ("fedval__thrEO", "naive", "EO_int_fx"), ("fedval__thrDP", "naive", "DP_int_fx"), ("fedavg__thrEO", "naive", "sens_fx"),
                    ("fedavg__thrEO", "naive", "ppv_fx")]:
        J[f"postdiff|{sc}|{a}-{b}|{c}"] = [float(v) for v in paired(a, b, c)]
    if sc == "natural":
        L.append("\\midrule")
open("paper/tab2_post.tex", "w").write("\n".join(L))
for sc in SC:
    d = pd.read_csv(f"res4/post_{sc}.csv")
    fr = d[d.method == "naive"].groupby("lam")[["AUROC", "DP_int_fx", "EO_int_fx"]].mean()
    for lam, r in fr.iterrows():
        for c in fr.columns:
            J[f"frontier|{sc}|naive|{lam}|{c}"] = float(r[c])

# ---------- leave-states-out offsets (fixed-threshold metrics)
rowsl = [("oversample", "Oversample"), ("over_label", "Label-balance (dup.)"), ("w_label", "Loss re-weight: labels"),
         ("w_cell", "Loss re-weight: cells"), ("w_group", "Loss re-weight: groups"), ("shrink", "Shrinkage")]
L = []
for k, nm in rowsl:
    cells = []
    for sc in SC:
        f = f"res4/lso_{sc}.csv"
        if not os.path.exists(f):
            cells += ["--", "--"]
            continue
        d = pd.read_csv(f)
        for met in ["DP_int_fx", "EO_int_fx"]:
            o = A.ci_boot(A.offsets(d, k, metric=met))
            store(f"lso|{sc}|{k}|{met}", o)
            cells.append(fmt(o))
    L.append(f"{nm} & " + " & ".join(cells) + " \\\\")
open("paper/tab2_lso.tex", "w").write("\n".join(L))
for sc in SC:
    f = f"res4/lso_{sc}.csv"
    if os.path.exists(f):
        d = pd.read_csv(f)
        J[f"lso|{sc}|nseeds"] = int(d.seed.nunique())
        for m in ["fedavg", "sexonly", "naive", "oversample"]:
            x = d[(d.method == m) & ((d.lam == 8) | (d.lam == 0))]
            for c in ["AUROC", "DP_sex_fx", "DP_int_fx", "EO_int_fx"]:
                J[f"lsomain|{sc}|{m}|{c}"] = float(x[c].mean())

# ---------- cluster bootstrap
for sc in SC:
    f = f"res4/cluster_boot_{sc}.json"
    if os.path.exists(f):
        for k, v in json.load(open(f)).items():
            J[f"cluster|{sc}|{k}"] = v

# ---------- MLP (properly trained)
L = []
for sc in SC:
    f = f"res4/mlp400_{sc}.csv"
    if not os.path.exists(f):
        continue
    d = pd.read_csv(f)
    L.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{sc.capitalize()} ({d.seed.nunique()} seeds)}}}}\\\\")
    for k, nm in [("fedavg", "FedAvg"), ("sexonly", "Sex-only"), ("naive", "Naive inter."), ("oversample", "Oversample"),
                  ("w_group", "Group re-weight")]:
        x = d[(d.method == k) & ((d.lam == 8) | (d.lam == 0))]
        L.append(f"{nm} & " + " & ".join(f"{x[c].mean():.3f}" for c in ["AUROC", "DP_sex", "DP_int", "EO_sex", "EO_int"]) + " \\\\")
        for c in ["AUROC", "DP_sex", "DP_int", "EO_sex", "EO_int"]:
            J[f"mlp|{sc}|{k}|{c}"] = float(x[c].mean())
    if sc == "natural":
        L.append("\\midrule")
    for k in ["oversample", "over_label", "w_group"]:
        for met in ["DP_int", "EO_int"]:
            store(f"mlpoff|{sc}|{k}|{met}", A.ci_boot(A.offsets(d, k, metric=met)))
open("paper/tab2_mlp.tex", "w").write("\n".join(L))

json.dump(J, open("paper/round2.json", "w"), indent=1)
print("tables2 ok", len(J))

# ---------- MLP offsets table + Holm across all primary tests
from analysis import holm
fam = []
import make_tables as MT
for sc in SC:
    for k, kk in [("oversample", "oversample"), ("over_label", "dense_over_label"), ("w_group", "dense_w_group")]:
        for met in ["DP_int", "EO_int"]:
            fam.append((f"LR|{sc}|{k}|{met}", MT.res[sc]["off"][(kk, met)]))
    d = pd.read_csv(f"res4/mlp400_{sc}.csv")
    d = d[d.diverged.isna()] if "diverged" in d else d
    for k in ["oversample", "over_label", "w_group"]:
        for met in ["DP_int", "EO_int"]:
            fam.append((f"MLP|{sc}|{k}|{met}", A.ci_boot(A.offsets(d, k, metric=met))))
famLR = [f for f in fam if f[0].startswith("LR|")]; famMLP = [f for f in fam if f[0].startswith("MLP|")]
adj = holm([f[1][3] for f in famLR])  # Holm family: the 12 logistic-regression primary tests (MLP tests are exploratory, uncorrected)
H = {f[0]: dict(mean=float(f[1][0]), lo=float(f[1][1]), hi=float(f[1][2]), p=float(f[1][3]), holm=float(a)) for f, a in zip(famLR, adj)}
H.update({f[0]: dict(mean=float(f[1][0]), lo=float(f[1][1]), hi=float(f[1][2]), p=float(f[1][3]), holm=float("nan")) for f in famMLP})
json.dump(H, open("paper/holm_all.json", "w"), indent=1)
names = {"oversample": "Oversample", "over_label": "Label-balance (dup.)", "w_group": "Group re-weight"}
L = []
for k in ["oversample", "over_label", "w_group"]:
    cells = []
    for sc in SC:
        for met in ["DP_int", "EO_int"]:
            h = H[f"MLP|{sc}|{k}|{met}"]
            cells.append(fmt((h["mean"], h["lo"], h["hi"])))
    L.append(f"{names[k]} & " + " & ".join(cells) + " \\\\")
open("paper/tab2_mlpoff.tex", "w").write("\n".join(L))
for sc in SC:
    d = pd.read_csv(f"res4/mlp400_{sc}.csv")
    if "diverged" in d:
        dv = d[d.diverged == 1]
        J[f"mlp|{sc}|diverged"] = {f"{m}|{l}": int(n) for (m, l), n in dv.groupby(["method", "lam"]).size().items()}
for sc in SC:
    c = pd.read_csv(f"res4/central_{sc}.csv")
    J[f"central|{sc}|fedavg_auroc"] = float(c[c.method == "fedavg"].AUROC.mean())
    J[f"central|{sc}|naive8_auroc"] = float(c[(c.method == "naive") & (c.lam == 8)].AUROC.mean())
json.dump(J, open("paper/round2.json", "w"), indent=1)
print("holm ok")
for k, v in H.items(): print(k, round(v["mean"],3), round(v["p"],4), round(v["holm"],4))
print({k: v for k, v in J.items() if "diverged" in k or k.startswith("central|")})

# ---------- per-lambda operating-point offsets (oversample / w_group vs naive frontier at the method's own AUROC)
def perlam(scen, method, met):
    d = pd.concat([A.load("core", scen), A.load("controls", scen)] + ([pd.read_csv("res4/dense_natural.csv")] if scen == "natural" else []))
    d = d.drop_duplicates(["seed", "method", "lam"])
    out = {}
    for s, ds in d.groupby("seed"):
        r = ds[ds.method == "naive"].sort_values("AUROC")
        fr = r[["AUROC", met]].values
        for _, row in ds[ds.method == method].iterrows():
            o = row[met] - A.interp_gap(fr, row["AUROC"])
            out.setdefault(row["lam"], []).append(o)
    # report a per-lambda equal-AUROC offset only if >= half of the seeds lie inside the naive AUROC range
    return {l: (A.ci_boot(np.array(v)) if np.sum(~np.isnan(np.array(v, float))) >= 0.5 * len(v) else (np.nan,) * 4) for l, v in sorted(out.items())}
L = []
for sc in SC:
    lams = sorted(set(A.load("core", sc).lam) - {0.0})
    if sc == "natural": lams = sorted(set(lams) | {0.5, 2})
    for l in lams:
        cells = []
        for m in ["oversample", "w_group"]:
            for met in ["DP_int", "EO_int"]:
                o = perlam(sc, m, met).get(l)
                store(f"perlam|{sc}|{m}|{met}|{l}", o) if o is not None else None
                cells.append(fmt(o) if o is not None and not np.isnan(o[0]) else "--")
        dd = A.load("core", sc)
        if sc == "natural": dd = pd.concat([dd, pd.read_csv("res4/dense_natural.csv")])
        nv = dd[(dd.method == "naive") & (dd.lam == l)]
        J[f"perlam|{sc}|naive|EO_int|{l}"] = float(nv.EO_int.mean()); J[f"perlam|{sc}|naive|AUROC|{l}"] = float(nv.AUROC.mean())
        L.append(f"{sc[:3].capitalize()}. {l:g} & {nv.AUROC.mean():.3f} & {nv.EO_int.mean():.3f} & " + " & ".join(cells) + " \\\\")
    if sc == "natural":
        L.append("\\midrule")
open("paper/tab2_perlam.tex", "w").write("\n".join(L))

# ---------- Dirichlet partitions: federated vs central offsets
L = []
for scn, nm in [("dir03", "Dirichlet $\\alpha{=}1$"), ("dir", "Dirichlet $\\alpha{=}0.3$")]:
    ff, cf = f"res4/fedsub_{scn}.csv", f"res4/centsub_{scn}.csv"
    if not (os.path.exists(ff) and os.path.exists(cf)):
        continue
    fd, cd = pd.read_csv(ff), pd.read_csv(cf)
    for m, mn in [("oversample", "Oversample"), ("over_label", "Label-balance"), ("w_group", "Group re-weight")]:
        cells = []
        for met in ["DP_int", "EO_int"]:
            pf, pc = B.per_seed_offsets(fd, m, met), B.per_seed_offsets(cd, m, met)
            sd = sorted(set(pf) & set(pc))
            of, oc = A.ci_boot([pf[s] for s in sd]), A.ci_boot([pc[s] for s in sd])
            od = A.ci_boot([pf[s] - pc[s] for s in sd])
            for tag, o in (("fed", of), ("cen", oc), ("diff", od)):
                store(f"dir|{scn}|{m}|{met}|{tag}", o)
            cells += [fmt(of), fmt(oc), fmt(od)]
        L.append(f"{nm} & {mn} & " + " & ".join(cells) + " \\\\")
    J[f"dir|{scn}|fedavg_auroc_fed"] = float(fd[fd.method == "fedavg"].AUROC.mean())
    J[f"dir|{scn}|naive8_auroc_fed"] = float(fd[(fd.method == "naive") & (fd.lam == 8)].AUROC.mean())
    J[f"dir|{scn}|fedavg_auroc_cen"] = float(cd[cd.method == "fedavg"].AUROC.mean())
open("paper/tab2_dir.tex", "w").write("\n".join(L))

# ---------- exponentiated-gradient baseline
L = []
for sc in SC:
    f = f"res3/eg_{sc}.csv"
    if not os.path.exists(f):
        continue
    d = pd.read_csv(f)
    L.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{sc.capitalize()} ({d.seed.nunique()} seeds)}}}}\\\\")
    for (kind, eps), x in d.groupby(["method", "eps"]):
        L.append(f"EG ({kind[3:].upper()}-constrained, $\\epsilon{{=}}{eps}$) & {x.AUROC.mean():.3f} & {x.DP_int.mean():.3f} & {x.EO_int.mean():.3f} & {x.sens.mean():.3f} & {x.sel_rate.mean():.3f} \\\\")
        for c in ["AUROC", "DP_int", "EO_int", "sens", "sel_rate"]:
            J[f"eg|{sc}|{kind}|{eps}|{c}"] = float(x[c].mean())
    if sc == "natural":
        L.append("\\midrule")
open("paper/tab2_eg.tex", "w").write("\n".join(L))
json.dump(J, open("paper/round2.json", "w"), indent=1)
print("tables3 ok")
