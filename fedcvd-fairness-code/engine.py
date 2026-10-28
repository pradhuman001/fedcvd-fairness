"""Federated fairness engine (logistic regression + 1-hidden-layer MLP), numpy only.
Penalty variants: 'dp' (covariance, all samples), 'eo' (covariance within positives and negatives),
shrinkage (blend local cov with pooled cov), sample weights (cell/label/group), oversampling.
"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

FEATS = ["age", "bmi", "high_bp", "high_chol", "diabetes", "smoker", "sex"]
NUMERIC = ["age"]  # matches original data.py (FEATURE_COLS_NUMERIC ∩ BRFSS columns = age only)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


# ---------------------------------------------------------------- groups
def make_groups(df, kind="agesex", age_cut=55):
    sex = np.where(df["sex"].values == 1, "M", "F")
    if kind == "sex":
        return sex
    if kind == "agesex":
        return np.where(df["age"].values >= age_cut, "old_", "young_") + sex
    if kind == "age3sex":  # 3 age bands x sex
        a = df["age"].values
        band = np.where(a < 45, "a1_", np.where(a < 65, "a2_", "a3_"))
        return np.char.add(band, sex)
    raise ValueError(kind)


def onehot(g, cats):
    g = np.asarray(g, dtype=object).astype(str)
    return (g[:, None] == np.asarray(cats, dtype=object).astype(str)[None, :]).astype(float)


# ---------------------------------------------------------------- penalty grads
def cov_grad_z(z, S, mask=None):
    """Gradient wrt z of sum_g cov_g^2 (cov over samples in mask). Returns (grad_z, covs)."""
    n = len(z)
    if mask is None:
        idx = slice(None)
        m = n
    else:
        idx = mask
        m = int(mask.sum())
    gz = np.zeros(n)
    covs = np.zeros(S.shape[1])
    if m < 2:
        return gz, covs
    zz = z[idx]
    zc = zz - zz.mean()
    Ss = S[idx]
    Sc = Ss - Ss.mean(0)
    covs = (Sc * zc[:, None]).mean(0)
    g_sub = (2.0 * covs[None, :] * Sc).sum(1) / m
    if mask is None:
        gz = g_sub
    else:
        gz[idx] = g_sub
    return gz, covs


def pooled_covs(clients, fwd, mode):
    """Server-side pooled covariance per group (simulated secure aggregation)."""
    Z = np.concatenate([fwd(c["X"]) for c in clients])
    S = np.concatenate([c["S"] for c in clients])
    Y = np.concatenate([c["y"] for c in clients])
    out = []
    masks = [None] if mode == "dp" else [Y == 1, Y == 0]
    for mk in masks:
        zz = Z if mk is None else Z[mk]
        Ss = S if mk is None else S[mk]
        out.append(((Ss - Ss.mean(0)) * (zz - zz.mean())[:, None]).mean(0))
    return out


# ---------------------------------------------------------------- models
class Logit:
    kind = "lr"

    def __init__(self, d, seed):
        r = np.random.default_rng(seed)
        self.p = {"w": r.normal(0, 0.01, d), "b": np.zeros(1)}

    @staticmethod
    def fwd(p, X):
        return X @ p["w"] + p["b"][0], None

    @staticmethod
    def back(p, X, cache, gz):
        return {"w": X.T @ gz, "b": np.array([gz.sum()])}


class MLP:
    kind = "mlp"

    def __init__(self, d, seed, h=16):
        r = np.random.default_rng(seed)
        self.p = {"W1": r.normal(0, 1 / np.sqrt(d), (d, h)), "b1": np.zeros(h),
                  "w2": r.normal(0, 1 / np.sqrt(h), h), "b2": np.zeros(1)}

    @staticmethod
    def fwd(p, X):
        H = np.tanh(X @ p["W1"] + p["b1"])
        return H @ p["w2"] + p["b2"][0], H

    @staticmethod
    def back(p, X, H, gz):
        gH = np.outer(gz, p["w2"]) * (1 - H ** 2)
        return {"W1": X.T @ gH, "b1": gH.sum(0), "w2": H.T @ gz, "b2": np.array([gz.sum()])}


# ---------------------------------------------------------------- local training
def local_train(M, p0, c, cfg, glob):
    p = {k: v.copy() for k, v in p0.items()}
    X, y, S, sw = c["X"], c["y"], c["S"], c.get("sw")
    n = len(y)
    for _ in range(cfg["E"]):
        z, cache = M.fwd(p, X)
        err = sigmoid(z) - y
        if sw is not None:
            err = err * sw
        gz = err / n
        lam = cfg["lam"]
        if lam > 0 and S is not None:
            pen = np.zeros(n)
            if cfg["pen"] == "dp":
                g1, cv = cov_grad_z(z, S)
                if cfg["mode"] == "shrink":
                    g1 = shrink_grad(z, S, None, cv, c, glob[0], cfg["nmin"])
                pen += g1
            else:  # eo
                for j, mk in enumerate([y == 1, y == 0]):
                    g1, cv = cov_grad_z(z, S, mk)
                    if cfg["mode"] == "shrink":
                        g1 = shrink_grad(z, S, mk, cv, c, glob[j], cfg["nmin"])
                    pen += g1
            gz = gz + lam * pen
        g = M.back(p, X, cache, gz)
        for k in p:
            p[k] = p[k] - cfg["lr"] * (g[k] + (cfg["l2"] * p[k] if k.startswith("w") or k == "W1" else 0))
    return p


def shrink_grad(z, S, mask, cov_loc, c, cov_glob, nmin):
    """Heuristic update of the paper: blended covariance used as coefficient."""
    n = len(z)
    idx = slice(None) if mask is None else mask
    m = n if mask is None else int(mask.sum())
    gz = np.zeros(n)
    if m < 2:
        return gz
    Ss = S[idx]
    ng = Ss.sum(0)
    a = ng / (ng + nmin)
    cu = a * cov_loc + (1 - a) * cov_glob
    Sc = Ss - Ss.mean(0)
    g_sub = (2.0 * cu[None, :] * Sc).sum(1) / m
    if mask is None:
        gz = g_sub
    else:
        gz[idx] = g_sub
    return gz


def train_fed(M, clients, cfg, seed):
    model = M(clients[0]["X"].shape[1], seed)
    sizes = np.array([len(c["y"]) for c in clients], float)
    if cfg.get("agg_sizes") is not None:
        sizes = np.asarray(cfg["agg_sizes"], float)
    wts = sizes / sizes.sum()
    for _ in range(cfg["T"]):
        glob = None
        if cfg["mode"] == "shrink" and cfg["lam"] > 0:
            glob = pooled_covs(clients, lambda X: M.fwd(model.p, X)[0], cfg["pen"])
        new = None
        for c, w in zip(clients, wts):
            pl = local_train(M, model.p, c, cfg, glob)
            if new is None:
                new = {k: w * v for k, v in pl.items()}
            else:
                for k in new:
                    new[k] += w * pl[k]
        model.p = new
    return model


# ---------------------------------------------------------------- data prep
def oversample(df, gcol, rng, noise=0.05, add_noise=True):
    cnt = df.groupby([gcol, "target"]).size()
    mx = cnt.max()
    parts = [df]
    for (g, y), n in cnt.items():
        if n >= mx:
            continue
        sub = df[(df[gcol] == g) & (df["target"] == y)]
        s = sub.iloc[rng.integers(0, len(sub), mx - n)].copy()
        if add_noise:
            sd = sub[NUMERIC].std(ddof=0).fillna(0).values
            s[NUMERIC] = s[NUMERIC].values + rng.normal(0, noise, (len(s), len(NUMERIC))) * sd
        parts.append(s)
    return pd.concat(parts).reset_index(drop=True)


def cell_weights(df, gcol, kind):
    """Sample weights equalising (group,label) cells / labels / groups without duplication."""
    if kind == "cell":
        key = df[gcol].astype(str) + "_" + df["target"].astype(str)
    elif kind == "label":
        key = df["target"].astype(str)
    elif kind == "group":
        key = df[gcol].astype(str)
    cnt = key.map(key.value_counts()).values.astype(float)
    w = 1.0 / cnt
    return w * len(df) / w.sum()


# ---------------------------------------------------------------- metrics
def gaps(y, yhat, g):
    rates, tprs, fprs = [], [], []
    for u in np.unique(g):
        m = g == u
        rates.append(yhat[m].mean())
        p, q = m & (y == 1), m & (y == 0)
        tprs.append(yhat[p].mean() if p.any() else np.nan)
        fprs.append(yhat[q].mean() if q.any() else np.nan)
    dp = max(rates) - min(rates)
    eo = max(np.nanmax(tprs) - np.nanmin(tprs), np.nanmax(fprs) - np.nanmin(fprs))
    return dp, eo, np.nanmin(tprs), np.nanmax(tprs) - np.nanmin(tprs)


def evaluate(y, proba, thr_prev, sex, ig):
    thr = np.quantile(proba, 1 - thr_prev)
    yh = (proba >= thr).astype(int)
    dps, eos, _, _ = gaps(y, yh, sex)
    dpi, eoi, mintpr, tprgap = gaps(y, yh, ig)
    return dict(AUROC=roc_auc_score(y, proba), PRAUC=average_precision_score(y, proba),
                DP_sex=dps, EO_sex=eos, DP_int=dpi, EO_int=eoi, minTPR=mintpr, TPRgap=tprgap)


def train_safe(M, clients, cfg, seed, tries=3):
    """train_fed with divergence guard: if the pooled TRAINING AUROC is < 0.55 or scores are non-finite (unstable step size for a stiff
    penalty), retry with half the learning rate and twice the rounds. Returns (model, number_of_halvings, fit_ok)."""
    from sklearn.metrics import roc_auc_score
    X = np.vstack([c["X"] for c in clients]); y = np.concatenate([c["y"] for c in clients])
    cfg = dict(cfg)
    for k in range(tries + 1):
        m = train_fed(M, clients, cfg, seed)
        with np.errstate(all="ignore"):
            z = M.fwd(m.p, X)[0]
        ok = bool(np.isfinite(z).all() and roc_auc_score(y, z) >= 0.55)
        if ok or k == tries:
            return m, k, ok
        cfg["lr"] = cfg["lr"] / 2; cfg["T"] = cfg["T"] * 2


def sched(grp, lam, pen="dp"):
    """CONV=2 schedule for logistic models: (rounds T, learning rate). Penalties on few overlapping/marginal indicators (sex, age, marg) are stiffer."""
    if pen == "eo":  # covariance EO penalty is stiffer: lr 0.3 diverges at lambda>=8, lr .15 at 16, lr .075 at 32
        if lam <= 8: return 300, 0.15
        if lam <= 16: return 600, 0.075
        return 1200, 0.0375
    if grp in ("sex", "age", "marg"):
        if lam <= 4: return 100, 0.3
        if lam <= 8: return (300, 0.1) if grp == "marg" else (200, 0.15)
        return (600, 0.05) if grp == "marg" else (400, 0.075)
    return (300, 0.15) if lam > 8 else (100, 0.3)
