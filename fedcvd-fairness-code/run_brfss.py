"""BRFSS experiment runner. Usage: python run_brfss.py <suite> <scenario> <seeds> <out.csv>
suites: core | controls | eo | mlp | agecut
"""
import sys, time, numpy as np, pandas as pd
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
from engine import *

DF = pd.read_csv("brfss_cvd_clean.csv")
import os
CV = os.environ.get('CONV')
BASE = dict(T=100, E=3, lr=0.3, l2=1e-4, nmin=20) if CV == '2' else (dict(T=80, E=3, lr=0.15, l2=1e-4, nmin=20) if CV else dict(T=25, E=3, lr=0.1, l2=1e-4, nmin=20))


def spec(name, lam, **kw):
    s = dict(name=name, lam=lam, grp="agesex", pen="dp", mode="naive", over=None, sw=None,
             model="lr", orig_agg=False, cut=55)
    s.update(kw)
    return s


def suite_specs(suite, scen):
    L = [1, 4, 8, 16] if scen == "natural" else [1, 2, 4, 8, 16]
    out = []
    if suite == "core":
        out.append(spec("fedavg", 0.0, grp=None))
        for l in L:
            out += [spec("sexonly", l, grp="sex"), spec("naive", l), spec("oversample", l, over="cell"),
                    spec("shrink", l, mode="shrink")]
    if suite == "dense":  # extra lambdas for the dense naive frontier (natural), same arms as controls
        for l in [0.5, 2]:
            out += [spec("naive", l), spec("oversample", l, over="cell"), spec("w_group", l, sw="group"),
                    spec("over_label", l, over="label")]
    if suite == "dense05":  # lambda=0.5 only (capped dense frontier; lambda=2 already in core/controls)
        out += [spec("naive", 0.5), spec("oversample", 0.5, over="cell"), spec("w_group", 0.5, sw="group"),
                spec("over_label", 0.5, over="label")]
    if suite == "controls":  # lam fixed 8 plus frontier lams for sw variants
        for l in L:
            out += [spec("naive", l), spec("oversample", l, over="cell"),
                    spec("over_origagg", l, over="cell", orig_agg=True),
                    spec("over_nonoise", l, over="cell_nonoise"),
                    spec("w_cell", l, sw="cell"), spec("w_label", l, sw="label"), spec("w_group", l, sw="group"),
                    spec("over_label", l, over="label")]
    if suite == "eo":
        for l in L:
            out += [spec("naive", l), spec("eo_pen", l, pen="eo"), spec("oversample", l, over="cell"),
                    spec("eo_pen_over", l, pen="eo", over="cell"), spec("eo_pen_shrink", l, pen="eo", mode="shrink")]
    if suite == "eoext":
        LL = [8, 16, 32, 64, 128, 256]
        for l in LL:
            out += [spec("naive", l), spec("eo_pen", l, pen="eo"), spec("eo_pen_shrink", l, pen="eo", mode="shrink")]
    if suite == "capabl":
        for l in L:
            out += [spec("naive", l), spec("oversample", l, over="cell"), spec("w_group", l, sw="group"),
                    spec("over_label", l, over="label")]
    if suite == "mlp":
        out.append(spec("fedavg", 0.0, grp=None, model="mlp"))
        for l in L:
            out += [spec("naive", l, model="mlp"), spec("oversample", l, model="mlp", over="cell"),
                    spec("sexonly", l, model="mlp", grp="sex"), spec("shrink", l, model="mlp", mode="shrink")]
    if suite == "agecut":
        for cut in (45, 65):
            out.append(spec("fedavg", 0.0, grp=None, cut=cut))
            for l in L:
                out += [spec("naive", l, cut=cut), spec("oversample", l, cut=cut, over="cell"),
                        spec("shrink", l, cut=cut, mode="shrink")]
        for l in L:
            out += [spec("naive", l, grp="age3sex", cut="age3"), spec("oversample", l, grp="age3sex", cut="age3", over="cell")]
    return out


def run(args):
    suite, scen, cap, seed = args
    rng = np.random.default_rng(seed)
    n = len(DF)
    test_idx = rng.choice(n, size=int(n * 0.15), replace=False)
    mask = np.ones(n, bool)
    mask[test_idx] = False
    te, tr = DF.iloc[test_idx].reset_index(drop=True), DF[mask].reset_index(drop=True)
    sc = StandardScaler().fit(tr[FEATS].values)
    prev = tr["target"].mean()
    counts = tr["state_fips"].value_counts()
    states = sorted(counts[counts >= 500].index)
    cap_rng = np.random.default_rng(seed)
    parts = []
    for s in states:
        c = tr[tr["state_fips"] == s].reset_index(drop=True)
        if scen == "capped" and len(c) > cap:
            c = c.loc[cap_rng.choice(c.index, size=cap, replace=False)].reset_index(drop=True)
        parts.append(c)
    rows = []
    ytest = te["target"].values.astype(int)
    Xte = sc.transform(te[FEATS].values)
    sex_te = make_groups(te, "sex")
    cache = {}
    for sp in suite_specs(suite, scen):
        cut = 55 if sp["cut"] == "age3" else sp["cut"]
        ev_g = make_groups(te, "agesex", cut) if sp["cut"] != "age3" else make_groups(te, "age3sex")
        gk = sp["grp"] if sp["grp"] else "agesex"
        key = (gk, cut, sp["over"], sp["sw"])
        if key not in cache:
            cl = []
            for j, c in enumerate(parts):
                c = c.copy()
                c["g"] = make_groups(c, gk, cut if gk == "agesex" else 55)
                r2 = np.random.default_rng(seed * 1000 + j)
                if sp["over"] in ("cell", "cell_nonoise"):
                    c = oversample(c, "g", r2, add_noise=(sp["over"] == "cell"))
                elif sp["over"] == "label":  # equalise labels only, per client
                    c = oversample(c.assign(g="all"), "g", r2)
                    c["g"] = make_groups(c, gk, cut if gk == "agesex" else 55)
                cl.append(c)
            cache[key] = cl
        cl = cache[key]
        cats = sorted(set(np.concatenate([c["g"].to_numpy(dtype=object) for c in cl]).astype(str)))
        clients = []
        for c in cl:
            d = dict(X=sc.transform(c[FEATS].values), y=c["target"].values.astype(float),
                     S=onehot(c["g"].values, cats) if sp["grp"] else None)
            if sp["sw"]:
                d["sw"] = cell_weights(c, "g", sp["sw"])
            clients.append(d)
        cfg = dict(BASE, lam=sp["lam"], pen=sp["pen"], mode=sp["mode"])
        if CV == "2" and sp["model"] == "lr":
            Ts, lrs = sched(sp["grp"], sp["lam"], sp["pen"]); cfg.update(T=Ts, lr=lrs)
        if sp["orig_agg"]:
            cfg["agg_sizes"] = [len(c) for c in parts]
        M = MLP if sp["model"] == "mlp" else Logit
        model, nhalve, okfit = train_safe(M, clients, cfg, seed)
        proba = sigmoid(M.fwd(model.p, Xte)[0])
        m = evaluate(ytest, proba, prev, sex_te, ev_g)
        m.update(suite=suite, scenario=scen, cap=cap if scen == "capped" else 0, seed=seed, method=sp["name"],
                 lam=sp["lam"], grp=sp["grp"], cut=str(sp["cut"]), model=sp["model"], halvings=nhalve, fit_ok=int(okfit))
        rows.append(m)
    return rows


if __name__ == "__main__":
    suite, scen, seeds, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    cap = int(sys.argv[5]) if len(sys.argv) > 5 else 150
    seeds = [int(s) for s in seeds.split(",")]
    t = time.time()
    with Pool(2) as p:
        res = p.map(run, [(suite, scen, cap, s) for s in seeds])
    pd.DataFrame([r for rr in res for r in rr]).to_csv(out, index=False)
    print("done", out, round(time.time() - t), "s")
