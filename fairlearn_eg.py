"""Fairlearn ExponentiatedGradient reference on BRFSS (Colab / local).
Setup:  pip install fairlearn scikit-learn pandas numpy scipy
Upload next to this file: brfss_cvd_clean.csv, engine.py, run2.py  (from code/ in the package)
Run:    python fairlearn_eg.py          (capped: 20 seeds, small/fast; natural: 10 seeds, 30K-row subsample)
Output: fairlearn_eg_capped.csv, fairlearn_eg_natural.csv  -> send these back.
EG is class-balanced via sample_weight (plain 0-1 error collapses to all-negative at 9.3% prevalence). Reported as a REFERENCE only:
it is centralized and its operating point (selection rate) differs from the 9.3%-prevalence rules used elsewhere."""
import sys, time, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
import run2 as R
from engine import FEATS, make_groups

def rates(p, y, g):
    G = np.unique(g)
    sel = [p[g == a].mean() for a in G]
    tpr = [p[(g == a) & (y == 1)].mean() for a in G]
    fpr = [p[(g == a) & (y == 0)].mean() for a in G]
    return dict(DP_int=max(sel) - min(sel), EO_int=max(max(tpr) - min(tpr), max(fpr) - min(fpr)),
                sens=p[y == 1].mean(), sel_rate=p.mean(), minTPR=min(tpr))

def one(scen, seed, cap=150):
    tr, tt, parts = R.split(seed, scen, cap, "random")
    pool = pd.concat(parts, ignore_index=True)
    if len(pool) > 30000:
        pool = pool.sample(30000, random_state=seed)
    sc = StandardScaler().fit(tr[FEATS].values)
    X, y, g = sc.transform(pool[FEATS].values), pool.target.values.astype(int), make_groups(pool, "agesex", 55)
    Xt, yt, gt = sc.transform(tt[FEATS].values), tt.target.values.astype(int), make_groups(tt, "agesex", 55)
    w = np.where(y == 1, 0.5 / y.sum(), 0.5 / (len(y) - y.sum())) * len(y)
    ib = np.random.default_rng(seed).choice(len(y), size=len(y), replace=True, p=w / w.sum())  # class-balanced resample (fairlearn>=0.13 rejects sample_weight)
    X, y, g = X[ib], y[ib], np.asarray(g)[ib]
    rows = []
    for name, C in (("eg_dp", DemographicParity), ("eg_eo", EqualizedOdds)):
        for eps in (0.05, 0.02, 0.01):
            m = ExponentiatedGradient(LogisticRegression(max_iter=300), constraints=C(difference_bound=eps), max_iter=50)
            m.fit(X, y, sensitive_features=g)
            pmf = m._pmf_predict(Xt)[:, 1]                       # expected positive-prediction probability
            hard = m.predict(Xt, random_state=seed).astype(float)  # one random draw of the randomized classifier
            for kind, p in (("expected", pmf), ("drawn", hard)):
                r = dict(scenario=scen, seed=seed, method=name, eps=eps, kind=kind, AUROC=roc_auc_score(yt, pmf))
                r.update(rates(p, yt, gt)); rows.append(r)
    return rows

if __name__ == "__main__":
    for scen, seeds in (("capped", range(42, 62)), ("natural", range(42, 52))):
        out, t = [], time.time()
        for s in seeds:
            out += one(scen, s); print(scen, s, round(time.time() - t), "s", flush=True)
            pd.DataFrame(out).to_csv(f"fairlearn_eg_{scen}.csv", index=False)
