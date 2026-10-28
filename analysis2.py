"""Analysis of second-round results: central-vs-federated, MLP50, post-processing/calibration, leave-states-out, cluster bootstrap."""
import numpy as np, pandas as pd, glob, os, json
import analysis as A

rng = np.random.default_rng(1)


def per_seed_offsets(df, method, metric, ref="naive"):
    """dict seed -> mean offset at equal AUROC vs the naive frontier (in-range points only)."""
    out = {}
    for s, d in df.groupby("seed"):
        r = d[d.method == ref].sort_values("AUROC")
        m = d[d.method == method]
        if len(r) < 2 or len(m) == 0:
            continue
        fr = r[["AUROC", metric]].values
        offs = [row[metric] - A.interp_gap(fr, row["AUROC"]) for _, row in m.iterrows()]
        offs = [o for o in offs if not np.isnan(o)]
        if offs:
            out[s] = np.mean(offs)
    return out


def offs_ci(df, method, metric, ref="naive"):
    v = np.array(list(per_seed_offsets(df, method, metric, ref).values()))
    return A.ci_boot(v)


def central_vs_fed(scen):
    c = pd.read_csv(f"res4/central_{scen}.csv")
    f = pd.concat([pd.read_csv(f"res4/core_{scen}.csv"), pd.read_csv(f"res4/controls_{scen}.csv")])
    f = f[f.method.isin(["naive", "oversample", "w_group", "over_label"])]
    f = f.drop_duplicates(["seed", "method", "lam"])  # core and controls both contain naive/oversample rows (bit-identical)
    if scen == "natural":  # dense frontier: add lambda = 0.5, 2 (federated and centralized, same seeds)
        c = pd.concat([c, pd.read_csv("res4/centdense_natural.csv")])
        f = pd.concat([f, pd.read_csv("res4/dense_natural.csv")]).drop_duplicates(["seed", "method", "lam"])
    rows = []
    for m in ["oversample", "over_label", "w_group"]:
        for met in ["DP_int", "EO_int"]:
            pc, pf = per_seed_offsets(c, m, met), per_seed_offsets(f, m, met)
            seeds = sorted(set(pc) & set(pf))
            diff = np.array([pf[s] - pc[s] for s in seeds])
            rows.append(dict(method=m, metric=met, central=A.ci_boot(list(pc.values()))[0],
                             central_lo=A.ci_boot(list(pc.values()))[1], central_hi=A.ci_boot(list(pc.values()))[2],
                             fed=A.ci_boot(list(pf.values()))[0], fed_lo=A.ci_boot(list(pf.values()))[1],
                             fed_hi=A.ci_boot(list(pf.values()))[2],
                             diff_fed_minus_central=A.ci_boot(diff)[0], diff_lo=A.ci_boot(diff)[1], diff_hi=A.ci_boot(diff)[2]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- cluster bootstrap over states
def wauc(score_sorted_idx, y, W):
    ys = y[score_sorted_idx].astype(np.float32)
    Ws = W[:, score_sorted_idx]
    neg = Ws * (1 - ys)
    pos = Ws * ys
    nb = np.cumsum(neg, 1) - neg
    # ties ignored (scores continuous)
    return (pos * nb).sum(1) / (pos.sum(1) * neg.sum(1))


def wfair(yh, y, G, W):
    """weighted DP/EO over groups G (n,k one-hot float32); W (B,n)."""
    yh = yh.astype(np.float32)
    den = W @ G
    den_p = W @ (y[:, None] * G)
    den_n = W @ ((1 - y)[:, None] * G)
    rate = (W @ (yh[:, None] * G)) / den
    tpr = (W @ ((yh * y)[:, None] * G)) / den_p
    fpr = (W @ ((yh * (1 - y))[:, None] * G)) / den_n
    g = lambda x: np.nanmax(x, 1) - np.nanmin(x, 1)
    return g(rate), np.maximum(g(tpr), g(fpr))


def cluster_bootstrap(scen, B=400, methods=("oversample", "w_group", "over_label")):
    files = sorted(glob.glob(f"res4/preds/{scen}_*.npz"))
    if not files:
        return None
    # one shared bootstrap over states (all seeds' test sets contain all ~52 states in 'random' split)
    first = np.load(files[0])
    states = np.unique(first["state"])
    cnt = rng.multinomial(len(states), np.full(len(states), 1 / len(states)), size=B).astype(np.float32)  # (B, S)
    sidx = {s: i for i, s in enumerate(states)}
    stats = {(m, met): np.zeros(B) for m in methods for met in ("DP_int", "EO_int")}
    nseeds = 0
    for f in files:
        z = np.load(f, allow_pickle=True)
        y = z["y"].astype(np.float32)
        W = cnt[:, [sidx[s] for s in z["state"]]]  # (B, n)
        old = z["age"] >= 55
        grp = np.where(old, 2, 0) + z["sex"].astype(int)
        G = np.eye(4, dtype=np.float32)[grp]
        tags = list(z["tags"])
        pts = {}
        for ti, tag in enumerate(tags):
            name, lam = tag.split("|")
            p = z["P"][ti]
            order = np.argsort(p, kind="mergesort")
            au = wauc(order, y, W)
            yh = (p >= z["thq"][ti]).astype(np.float32)
            dp, eo = wfair(yh, y, G, W)
            pts[(name, float(lam))] = (au, dp, eo)
        refs = sorted([k for k in pts if k[0] == "naive"], key=lambda k: k[1])
        for m in methods:
            mk = [k for k in pts if k[0] == m]
            for met, idx in (("DP_int", 1), ("EO_int", 2)):
                offs = np.full((B, len(mk)), np.nan)
                for j, k in enumerate(mk):
                    for b in range(B):
                        ra = np.array([pts[r][0][b] for r in refs])
                        rg = np.array([pts[r][idx][b] for r in refs])
                        o = np.argsort(ra)
                        a = pts[k][0][b]
                        if ra[o][0] <= a <= ra[o][-1]:
                            offs[b, j] = pts[k][idx][b] - np.interp(a, ra[o], rg[o])
                with np.errstate(all="ignore"):
                    stats[(m, met)] += np.nanmean(offs, 1)
        nseeds += 1
    out = {}
    for k, v in stats.items():
        v = v / nseeds
        lo, hi = np.nanpercentile(v, [2.5, 97.5])
        out[k] = (float(np.nanmean(v)), float(lo), float(hi))
    return out, nseeds
