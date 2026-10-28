"""Own implementation of the exponentiated-gradient reduction (Agarwal et al., 2018) for DP / EO constraints on the 4 age x sex groups.
Centralized in-processing reference baseline (fairlearn is not installable here). Usage: python eg.py <scenario> <seeds> <out.csv> [cap]"""
import sys, os, numpy as np, pandas as pd
from multiprocessing import Pool
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import run2 as R
from engine import FEATS, make_groups

def coefs(g, y, kind):
    n = len(y); G = np.unique(g); C = []
    for a in G:
        if kind == "dp":
            m = (g == a); v = m / m.sum() - 1.0 / n
            C += [v, -v]
        else:
            for yy in (0, 1):
                m = (g == a) & (y == yy); b = (y == yy)
                v = m / max(m.sum(), 1) - b / b.sum()
                C += [v, -v]
    return np.array(C)  # (K, n): gamma_k(h) = C[k] @ h

def eg(X, y, g, kind, eps, T=40, B=None, eta=2.0, cap_rows=None, rng=None):
    n = len(y); C = coefs(g, y, kind); K = len(C); B = B or 1.0 / eps
    theta = np.zeros(K); H = []; Hbar = np.zeros(n)
    for t in range(T):
        e = np.exp(theta - theta.max()); lam = B * e / (np.exp(-theta.max()) + e.sum())
        s = lam @ C                                   # per-sample cost of predicting 1 from the constraints
        wn = np.where(y == 1, 0.5 / y.sum(), 0.5 / (n - y.sum()))  # class-balanced error (plain 0-1 error collapses to the all-negative rule at 9% prevalence)
        c1 = (1 - y) * wn + s; c0 = y * wn
        lab = (c0 > c1).astype(int); w = np.abs(c0 - c1)
        if lab.min() == lab.max():
            h = np.full(n, float(lab[0])); clf = ("const", lab[0])
        else:
            clf = LogisticRegression(C=1e4, max_iter=200).fit(X, lab, sample_weight=w * n)
            h = clf.predict(X).astype(float)
        H.append(clf); Hbar = (Hbar * t + h) / (t + 1)
        viol = C @ Hbar
        theta = theta + eta * (viol - eps) 
    return H

def predict(H, X):
    out = np.zeros(len(X))
    for clf in H:
        out += (np.full(len(X), float(clf[1])) if isinstance(clf, tuple) else clf.predict(X))
    return out / len(H)

def run(args):
    scen, cap, seed = args
    tr, tt, parts = R.split(seed, scen, cap, "random")
    pool = pd.concat(parts, ignore_index=True)
    sc = StandardScaler().fit(tr[FEATS].values)
    if len(pool) > 30000:  # natural scenario: train the reduction on a 30K random subsample of the pooled training data
        pool = pool.sample(30000, random_state=seed)
    X = sc.transform(pool[FEATS].values); y = pool.target.values.astype(int); g = make_groups(pool, "agesex", 55)
    Xt = sc.transform(tt[FEATS].values); yt = tt.target.values.astype(int); gt = make_groups(tt, "agesex", 55)
    rows = []
    for kind in ("dp", "eo"):
        for eps, B in ((0.05, 2), (0.02, 5), (0.01, 10)):
            H = eg(X, y, g, kind, eps, T=80, B=B, eta=0.5)
            ph = predict(H, Xt)
            sel = [ph[gt == a].mean() for a in np.unique(gt)]
            tpr = [ph[(gt == a) & (yt == 1)].mean() for a in np.unique(gt)]
            fpr = [ph[(gt == a) & (yt == 0)].mean() for a in np.unique(gt)]
            rows.append(dict(scenario=scen, seed=seed, method=f"eg_{kind}", eps=eps, AUROC=roc_auc_score(yt, ph),
                             DP_int=max(sel) - min(sel), B=B, EO_int=max(max(tpr) - min(tpr), max(fpr) - min(fpr)),
                             sens=ph[yt == 1].mean(), sel_rate=ph.mean(), prev=yt.mean()))
    return rows

if __name__ == "__main__":
    scen, seeds, out = sys.argv[1:4]; cap = int(sys.argv[4]) if len(sys.argv) > 4 else 150
    with Pool(2) as p:
        res = p.map(run, [(scen, cap, int(s)) for s in seeds.split(",")])
    pd.DataFrame([r for rr in res for r in rr]).to_csv(out, index=False); print("done", out)
