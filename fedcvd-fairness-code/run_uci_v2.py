"""UCI extension: same protocol as experiment_uci.py (20 repeats x stratified 5-fold, site x label strata,
pooled OOF), with the controls added in the revised paper. Writes npz in the format bootstrap_uci.py expects."""
import numpy as np, pandas as pd, os, sys
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import engine as E
from engine import *

FE = ["age", "sex", "trestbps", "thalach", "exang", "oldpeak", "cp_2", "cp_3", "cp_4", "restecg_1", "restecg_2"]
SITES = ["cleveland", "hungarian", "switzerland", "va"]
GROUPS = ["old_female", "old_male", "young_female", "young_male"]
E.NUMERIC = ["age", "trestbps", "thalach", "oldpeak"]
df = pd.read_csv("uci_heart_clean.csv")
df["g"] = np.where(df.age >= 55, "old", "young") + "_" + np.where(df.sex == 1, "male", "female")
n = len(df)
y_all = df.target.values.astype(int)
METHODS = {  # name: dict(pen, over, sw)
    "naive_intersectional": dict(pen="dp", over=None, sw=None),
    "oversample_intersectional": dict(pen="dp", over="cell", sw=None),
    "over_label": dict(pen="dp", over="label", sw=None),
    "w_label": dict(pen="dp", over=None, sw="label"),
    "w_cell": dict(pen="dp", over=None, sw="cell"),
    "w_group": dict(pen="dp", over=None, sw="group"),
    "eo_pen": dict(pen="eo", over=None, sw=None),
}
LAMS = [1.0, 4.0, 8.0, 16.0]
SEEDS = list(range(42, 42 + int(os.environ.get('NS', '20'))))
mnames = list(METHODS)
P = np.zeros((len(LAMS), len(SEEDS), len(mnames), n), np.float32)
Y = np.zeros(P.shape, np.int8)
strat = df.site + "_" + df.target.astype(str)
for si, seed in enumerate(SEEDS):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, te in skf.split(df, strat):
        dtr, dte = df.iloc[tr], df.iloc[te]
        sc = StandardScaler().fit(dtr[FE].values)
        Xte = sc.transform(dte[FE].values)
        prev = dtr.target.mean()
        for mi, (mn, mc) in enumerate(METHODS.items()):
            cl = []
            for j, s in enumerate(SITES):
                c = dtr[dtr.site == s].reset_index(drop=True)
                r2 = np.random.default_rng(seed * 1000 + j)
                if mc["over"] == "cell":
                    c = E.oversample(c, "g", r2)
                elif mc["over"] == "label":
                    c = E.oversample(c.assign(g0=c.g).assign(g="all"), "g", r2)
                    c["g"] = c.g0
                cl.append(c)
            clients = []
            for c in cl:
                d = dict(X=sc.transform(c[FE].values), y=c.target.values.astype(float), S=onehot(c.g.values, GROUPS))
                if mc["sw"]:
                    d["sw"] = cell_weights(c, "g", mc["sw"])
                clients.append(d)
            for li, lam in enumerate(LAMS):
                Ts, lrs = sched("agesex", lam, mc["pen"])
                cfg = dict(T=Ts, E=3, lr=lrs, l2=1e-4, nmin=20, lam=lam, pen=mc["pen"], mode="naive")
                m, _nh, _ok = train_safe(Logit, clients, cfg, seed)
                p = sigmoid(Logit.fwd(m.p, Xte)[0])
                thr = np.quantile(p, 1 - prev)
                P[li, si, mi, te] = p
                Y[li, si, mi, te] = (p >= thr)
    print("seed", seed, flush=True)
os.makedirs("res_uci_v2", exist_ok=True)
np.savez_compressed("res_uci_v2/oof_predictions_full.npz", proba=P, pred=Y, lambdas=np.array(LAMS),
                    seeds=np.array(SEEDS), methods=np.array(mnames))
# summary_full.csv for the bootstrap script's self-test
rows = []
sex = df.sex.values
for li, lam in enumerate(LAMS):
    for mi, mn in enumerate(mnames):
        r = []
        for si in range(len(SEEDS)):
            yh = Y[li, si, mi]
            a = roc_auc_score(y_all, P[li, si, mi])
            dps, eos, _, _ = gaps(y_all, yh, sex)
            dpi, eoi, _, _ = gaps(y_all, yh, df.g.values)
            r.append([a, dps, eos, dpi, eoi])
        r = np.array(r)
        rows.append(dict(fairness_lambda=lam, method=mn, AUROC_mean=r[:, 0].mean(), DP_sex_mean=r[:, 1].mean(),
                         EO_sex_mean=r[:, 2].mean(), DP_inter_mean=r[:, 3].mean(), EO_inter_mean=r[:, 4].mean()))
pd.DataFrame(rows).round(4).to_csv("res_uci_v2/summary_full.csv", index=False)
print("done")
