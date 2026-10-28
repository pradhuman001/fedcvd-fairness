import numpy as np, pandas as pd, glob, json, os
rng = np.random.default_rng(0)


DENSE_FILES = {"natural": "res4/dense_natural.csv", "capped": "res4/dense05_capped.csv"}


def load(suite, scen):
    f = f"{os.environ.get('RES','res4')}/{suite}_{scen}.csv"
    d = pd.read_csv(f) if os.path.exists(f) else None
    if d is not None and suite in ("core", "controls") and os.environ.get("DENSE", "1") == "1" and os.path.exists(DENSE_FILES[scen]):
        # dense frontier: add lambda = 0.5 (and 2 in natural) for the arms each suite carries; core and controls overlap on naive/oversample (bit-identical rows)
        x = pd.read_csv(DENSE_FILES[scen])
        x = x[x.method.isin(["naive", "oversample"])] if suite == "core" else x
        d = pd.concat([d, x]).drop_duplicates(["seed", "method", "lam"] + (["model"] if "model" in d.columns and "model" in x.columns else [])).reset_index(drop=True)
    return d


def ci_boot(x, B=10000):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan, np.nan
    idx = np.random.default_rng(12345).integers(0, len(x), (B, len(x)))
    m = x[idx].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    # two-sided sign-flip (paired permutation) test of mean = 0; the bootstrap sign probability was anti-conservative
    r = np.random.default_rng(777)
    n = len(x)
    if n <= 16:
        import itertools
        sg = np.array(list(itertools.product([-1, 1], repeat=n)), float)
    else:
        sg = r.choice([-1.0, 1.0], size=(200000, n))
    pm = np.abs((sg * x).mean(1))
    p = (pm >= abs(x.mean()) - 1e-15).mean()
    return x.mean(), lo, hi, max(p, 1.0 / len(sg))


def interp_gap(fr, a):
    """fr: array of (auroc, gap) sorted by auroc; linear interpolation, nan if outside range."""
    if a < fr[0, 0] or a > fr[-1, 0]:
        return np.nan
    return float(np.interp(a, fr[:, 0], fr[:, 1]))


def offsets(df, method, ref="naive", metric="DP_int", ref_grp=None, meth_grp=None):
    """Per-seed mean offset (method gap - reference-frontier gap at equal AUROC), points in range only."""
    out = []
    for s, d in df.groupby("seed"):
        r = d[(d.method == ref)].sort_values("AUROC")
        m = d[(d.method == method)]
        if len(r) < 2 or len(m) == 0:
            continue
        fr = r[["AUROC", metric]].values
        offs = [row[metric] - interp_gap(fr, row["AUROC"]) for _, row in m.iterrows()]
        offs = [o for o in offs if not np.isnan(o)]
        if offs:
            out.append(np.mean(offs))
    return np.array(out)


def summ(df, lam, methods=None, cols=("AUROC", "PRAUC", "DP_sex", "EO_sex", "DP_int", "EO_int")):
    d = df[(df.lam == lam) | (df.method == "fedavg")]
    if methods:
        d = d[d.method.isin(methods)]
    g = d.groupby("method")[list(cols)].agg(["mean", "std"])
    return g


def holm(ps):
    ps = np.asarray(ps)
    o = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0
    for i, k in enumerate(o):
        run = max(run, (len(ps) - i) * ps[k])
        adj[k] = min(1, run)
    return adj
