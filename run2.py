"""Second-round experiments (reviewer-driven). Usage: python run2.py <suite> <scenario> <seeds> <out.csv> [cap]
suites:
  central : centralized matched control (one pooled client, 75 gradient steps = 25 rounds x 3 epochs)
  mlp50   : properly trained MLP (T=50, lr=0.25 -> FedAvg AUROC ~ LR), frontier over lambda
  post    : baselines incl. group-threshold post-processing + calibration; saves test predictions for cluster bootstrap
  lso     : leave-states-out evaluation (held-out states, train-fixed score threshold)
scenario: natural | capped.  Splits in 'central','mlp50','post' are identical to run_brfss.py (same seeds).
"""
import sys, os, time, numpy as np, pandas as pd
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import engine as E
from engine import *

DF = pd.read_csv("brfss_cvd_clean.csv")
CV = os.environ.get('CONV')
BASE = dict(T=100, E=3, lr=0.3, l2=1e-4, nmin=20) if CV == '2' else (dict(T=80, E=3, lr=0.15, l2=1e-4, nmin=20) if CV else dict(T=25, E=3, lr=0.1, l2=1e-4, nmin=20))
PD = os.environ.get('PREDS', 'res2/preds')
STATES = sorted(DF.state_fips.value_counts()[lambda s: s >= 500].index)


def spec(name, lam, **kw):
    s = dict(name=name, lam=lam, grp="agesex", pen="dp", mode="naive", over=None, sw=None, model="lr",
             central=False, T=BASE['T'], lr=BASE['lr'], thr=False)
    s.update(kw)
    return s


def specs(suite, scen):
    L = [1, 4, 8, 16] if scen in ("natural", "dir", "dir03") else [1, 2, 4, 8, 16]
    o = []
    if suite == "centdense":  # dense-grid extra lambdas, centralized control (natural), same arms as the federated dense suite
        for l in [0.5, 2]:
            o += [spec("naive", l, central=True), spec("oversample", l, over="cell", central=True),
                  spec("w_group", l, sw="group", central=True), spec("over_label", l, over="label", central=True)]
    if suite == "central":
        o.append(spec("fedavg", 0.0, grp=None, central=True))
        for l in L:
            o += [spec("sexonly", l, grp="sex", central=True), spec("naive", l, central=True),
                  spec("oversample", l, over="cell", central=True), spec("w_group", l, sw="group", central=True),
                  spec("over_label", l, over="label", central=True)]
    if suite == "fine":
        L = [0.1, 0.25, 0.5, 1, 2, 4, 8, 16]
        o.append(spec("fedavg", 0.0, grp=None))
        for l in L:
            o += [spec("sexonly", l, grp="sex"), spec("ageonly", l, grp="age"), spec("marg", l, grp="marg"), spec("naive", l),
                  spec("oversample", l, over="cell"), spec("w_group", l, sw="group")]
    if suite == "glob":
        o.append(spec("fedavg", 0.0, grp=None))
        for l in L:
            o += [spec("naive", l), spec("w_group", l, sw="group"), spec("w_group_glob", l, sw="group_global")]
    if suite in ("fedsub", "centsub"):
        c = suite == "centsub"
        o.append(spec("fedavg", 0.0, grp=None, central=c))
        for l in L:
            o += [spec("naive", l, central=c), spec("oversample", l, over="cell", central=c),
                  spec("w_group", l, sw="group", central=c), spec("over_label", l, over="label", central=c)]
    if suite in ("mlp50", "mlp400"):
        kw = dict(model="mlp", T=50, lr=0.25) if suite == "mlp50" else dict(model="mlp", T=400, lr=0.1)
        o.append(spec("fedavg", 0.0, grp=None, **kw))
        for l in L:
            o += [spec("sexonly", l, grp="sex", **kw), spec("naive", l, **kw), spec("oversample", l, over="cell", **kw),
                  spec("w_group", l, sw="group", **kw), spec("over_label", l, over="label", **kw)]
    if suite == "eopost":
        o.append(spec("fedavg", 0.0, grp=None))
        for l in (0.5, 1, 2, 4, 8, 16, 32):
            o.append(spec("eo_pen", l, pen="eo"))
    if suite in ("post", "lso"):
        o.append(spec("fedavg", 0.0, grp=None, thr=(suite == "post")))
        for l in L:
            o += [spec("sexonly", l, grp="sex"), spec("naive", l, thr=(suite == "post" and l == 8)),
                  spec("oversample", l, over="cell", thr=(suite == "post" and l == 8)),
                  spec("w_group", l, sw="group"), spec("over_label", l, over="label")]
        if suite == "lso":
            for l in L:
                o += [spec("shrink", l, mode="shrink"), spec("w_label", l, sw="label"), spec("w_cell", l, sw="cell")]
    return o


def split(seed, scen, cap, mode):
    n = len(DF)
    rng = np.random.default_rng(seed)
    if mode == "random":
        te = rng.choice(n, size=int(n * 0.15), replace=False)
        m = np.ones(n, bool)
        m[te] = False
        tr, tt = DF[m].reset_index(drop=True), DF.iloc[te].reset_index(drop=True)
        states = sorted(tr.state_fips.value_counts()[lambda s: s >= 500].index)
    else:  # leave-states-out: hold out 15% of states entirely
        hold = rng.choice(STATES, size=int(round(0.15 * len(STATES))), replace=False)
        tt = DF[DF.state_fips.isin(hold)].reset_index(drop=True)
        tr = DF[~DF.state_fips.isin(hold)].reset_index(drop=True)
        states = [s for s in STATES if s not in set(hold)]
    cr = np.random.default_rng(seed)
    if scen in ("dir", "dir03"):  # synthetic label+group-skew partition of the same training data into K clients
        alpha = 0.3 if scen == "dir" else 1.0
        K = len(states)
        g = make_groups(tr, "agesex", 55).astype(str)
        cell = pd.Series(g) + "_" + tr.target.astype(str).values
        assign = np.zeros(len(tr), int)
        for cv in cell.unique():
            idx = np.where(cell.values == cv)[0]
            pr = cr.dirichlet(np.full(K, alpha))
            assign[idx] = cr.choice(K, size=len(idx), p=pr)
        parts = [tr[assign == k].reset_index(drop=True) for k in range(K)]
        parts = [p for p in parts if len(p) >= 200]
        return tr, tt, parts
    parts = []
    for s in states:
        c = tr[tr.state_fips == s].reset_index(drop=True)
        if scen == "capped" and len(c) > cap:
            c = c.loc[cr.choice(c.index, size=cap, replace=False)].reset_index(drop=True)
        parts.append(c)
    return tr, tt, parts


def pred_metrics(y, yh, sex, ig):
    dps, eos, _, _ = gaps(y, yh, sex)
    dpi, eoi, mt, tg = gaps(y, yh, ig)
    # separate TPR / FPR gaps (intersectional)
    tpr = [yh[(ig == g) & (y == 1)].mean() for g in np.unique(ig)]
    fpr = [yh[(ig == g) & (y == 0)].mean() for g in np.unique(ig)]
    ppvs = [y[(ig == g) & (yh == 1)].mean() if ((ig == g) & (yh == 1)).any() else np.nan for g in np.unique(ig)]
    return dict(DP_sex=dps, EO_sex=eos, DP_int=dpi, EO_int=eoi, TPRgap_int=max(tpr) - min(tpr), FPRgap_int=max(fpr) - min(fpr),
                minTPR=min(tpr), sens=yh[y == 1].mean(), ppv=(y[yh == 1].mean() if yh.any() else np.nan), minPPV=np.nanmin(ppvs))


def calib(y, p, ig):
    citl = max(abs(p[ig == g].mean() - y[ig == g].mean()) for g in np.unique(ig))
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum((bins == b).mean() * abs(p[bins == b].mean() - y[bins == b].mean()) for b in range(10) if (bins == b).any())
    return dict(CITL_int=citl, ECE=ece)


def run(args):
    suite, scen, cap, seed = args
    mode = "lso" if suite == "lso" else "random"
    tr, tt, parts = split(seed, scen, cap, mode)
    sc = StandardScaler().fit(tr[FEATS].values)
    prev_q = tr.target.mean()
    Xte = sc.transform(tt[FEATS].values)
    yte = tt.target.values.astype(int)
    sex_te = make_groups(tt, "sex")
    ig_te = make_groups(tt, "agesex", 55)
    pool = pd.concat(parts, ignore_index=True)
    Xpool = sc.transform(pool[FEATS].values)
    ypool = pool.target.values.astype(int)
    ig_pool = make_groups(pool, "agesex", 55)
    rows, preds, thrq = [], {}, {}
    cache = {}
    for sp in specs(suite, scen):
        gk = "agesex" if sp["grp"] in ("marg", "age") else (sp["grp"] if sp["grp"] else "agesex")
        key = (gk, sp["over"], sp["sw"], sp["central"])
        if key not in cache:
            src = [pool] if sp["central"] else parts
            cl = []
            for j, c in enumerate(src):
                c = c.copy()
                c["g"] = make_groups(c, gk, 55)
                r2 = np.random.default_rng(seed * 1000 + j)
                if sp["over"] == "cell":
                    c = oversample(c, "g", r2)
                elif sp["over"] == "label":
                    c0 = c.g.values
                    c = oversample(c.assign(g="all"), "g", r2)
                    c["g"] = make_groups(c, gk, 55)
                cl.append(c)
            cache[key] = cl
        cl = cache[key]
        cats = sorted(set(np.concatenate([c["g"].to_numpy(dtype=object) for c in cl]).astype(str)))
        clients = []
        if sp["sw"] == "group_global":  # weights from POOLED group counts (needs cross-client counts), split back per client
            wall = cell_weights(pd.concat(cl, ignore_index=True), "g", "group")
            cuts = np.cumsum([len(c) for c in cl])[:-1]
            wsplit = np.split(wall, cuts)
        for j, c in enumerate(cl):
            if sp["grp"] == "marg":
                S = np.hstack([onehot(make_groups(c, "sex"), ["F", "M"]), onehot(np.where(c["age"].values >= 55, "old", "young"), ["young", "old"])])
            elif sp["grp"] == "age":
                S = onehot(np.where(c["age"].values >= 55, "old", "young"), ["young", "old"])
            else:
                S = onehot(c["g"].values, cats) if sp["grp"] else None
            d = dict(X=sc.transform(c[FEATS].values), y=c.target.values.astype(float), S=S)
            if sp["sw"] == "group_global":
                d["sw"] = wsplit[j]
            elif sp["sw"]:
                d["sw"] = cell_weights(c, "g", sp["sw"])
            clients.append(d)
        T, Ee = (1, BASE['T'] * BASE['E']) if sp["central"] else (sp["T"], BASE["E"])
        lr_ = sp["lr"]
        if CV == "2" and sp["model"] == "lr":
            Ts, lr_ = sched(sp.get("grp"), sp["lam"], sp["pen"])
            if sp["central"]:
                T, Ee = 1, Ts * BASE['E']
            else:
                T = Ts
        cfg = dict(BASE, T=T, E=Ee, lr=lr_, lam=sp["lam"], pen=sp["pen"], mode=sp["mode"])
        M = MLP if sp["model"] == "mlp" else Logit
        model, nhalve, okfit = train_safe(M, clients, cfg, seed)
        p = sigmoid(M.fwd(model.p, Xte)[0])
        ptr = sigmoid(M.fwd(model.p, Xpool)[0])
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(ptr))):  # optimisation diverged -> record, exclude from frontiers
            rows.append(dict(suite=suite, scenario=scen, seed=seed, method=sp["name"], lam=sp["lam"], model=sp["model"], diverged=1))
            continue
        # (q) top-p of held-out scores at training prevalence (as in the main paper)
        thq = np.quantile(p, 1 - prev_q)
        yq = (p >= thq).astype(int)
        # (fx) fixed score threshold from TRAINING scores at training prevalence (deployable)
        thf = np.quantile(ptr, 1 - ypool.mean())
        yf = (p >= thf).astype(int)
        base = dict(suite=suite, scenario=scen, seed=seed, method=sp["name"], lam=sp["lam"], model=sp["model"],
                    AUROC=roc_auc_score(yte, p), PRAUC=average_precision_score(yte, p))
        base.update(halvings=nhalve, fit_ok=int(okfit))
        r = dict(base)
        r.update(pred_metrics(yte, yq, sex_te, ig_te))
        r.update({k + "_fx": v for k, v in pred_metrics(yte, yf, sex_te, ig_te).items()})
        r.update(calib(yte, p, ig_te))
        r["pos_rate_fx"] = yf.mean()
        rows.append(r)
        tag = f"{sp['name']}|{sp['lam']}"
        preds[tag] = p.astype(np.float32)
        thrq[tag] = thq
        if sp["thr"]:  # group-threshold post-processing on top of this model, train-fixed thresholds
            tpr_t = (ptr[ypool == 1] >= thf).mean()
            sel_t = (ptr >= thf).mean()
            for nm, kind in [("thrEO", "eo"), ("thrDP", "dp")]:
                t = {}
                for g in np.unique(ig_pool):
                    if kind == "eo":
                        t[g] = np.quantile(ptr[(ig_pool == g) & (ypool == 1)], 1 - tpr_t)
                    else:
                        t[g] = np.quantile(ptr[ig_pool == g], 1 - sel_t)
                thr_vec = np.array([t[g] for g in ig_te])
                yp = (p >= thr_vec).astype(int)
                r2 = dict(base)
                r2["method"] = sp["name"] + "__" + nm
                r2.update({k + "_fx": v for k, v in pred_metrics(yte, yp, sex_te, ig_te).items()})
                r2["pos_rate_fx"] = yp.mean()
                rows.append(r2)
    if suite == "post":  # group thresholds fitted on a held-out validation split (model trained without it)
        rv = np.random.default_rng(seed + 777)
        ptr_parts, val_parts = [], []
        for c in parts:
            mk = rv.random(len(c)) < 0.2
            val_parts.append(c[mk].reset_index(drop=True)); ptr_parts.append(c[~mk].reset_index(drop=True))
        val = pd.concat(val_parts, ignore_index=True)
        clv = [dict(X=sc.transform(c[FEATS].values), y=c.target.values.astype(float), S=None) for c in ptr_parts]
        mv = train_fed(Logit, clv, dict(BASE, lam=0.0, pen="dp", mode="naive"), seed)
        pv = sigmoid(Logit.fwd(mv.p, sc.transform(val[FEATS].values))[0])
        pt = sigmoid(Logit.fwd(mv.p, Xte)[0])
        yv = val.target.values.astype(int); igv = make_groups(val, "agesex", 55)
        thv = np.quantile(pv, 1 - yv.mean())
        base = dict(suite=suite, scenario=scen, seed=seed, lam=0.0, model="lr", AUROC=roc_auc_score(yte, pt), PRAUC=average_precision_score(yte, pt))
        r0 = dict(base, method="fedval"); r0.update({k + "_fx": v for k, v in pred_metrics(yte, (pt >= thv).astype(int), sex_te, ig_te).items()})
        rows.append(r0)
        tpr_t = (pv[yv == 1] >= thv).mean(); sel_t = (pv >= thv).mean()
        for nm, kind in [("thrEO", "eo"), ("thrDP", "dp")]:
            t = {g: (np.quantile(pv[(igv == g) & (yv == 1)], 1 - tpr_t) if kind == "eo" else np.quantile(pv[igv == g], 1 - sel_t)) for g in np.unique(igv)}
            yp = (pt >= np.array([t[g] for g in ig_te])).astype(int)
            r2 = dict(base, method="fedval__" + nm); r2.update({k + "_fx": v for k, v in pred_metrics(yte, yp, sex_te, ig_te).items()})
            rows.append(r2)
    if suite == "post":
        os.makedirs(PD, exist_ok=True)
        st = tt.state_fips.values
        np.savez_compressed(f"{PD}/{scen}_{seed}.npz", y=yte, state=st, sex=(tt.sex.values == 1), age=tt.age.values,
                            tags=np.array(list(preds)), P=np.stack(list(preds.values())),
                            thq=np.array([thrq[k] for k in preds]))
    return rows


if __name__ == "__main__":
    suite, scen, seeds, out = sys.argv[1:5]
    cap = int(sys.argv[5]) if len(sys.argv) > 5 else 150
    seeds = [int(s) for s in seeds.split(",")]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    t = time.time()
    with Pool(2) as p:
        res = p.map(run, [(suite, scen, cap, s) for s in seeds])
    pd.DataFrame([r for rr in res for r in rr]).to_csv(out, index=False)
    print("done", out, round(time.time() - t), "s", flush=True)
