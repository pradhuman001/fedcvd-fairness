"""
bootstrap_uci.py
-----------------
Patient-level bootstrap confidence intervals for the UCI results, computed from
the saved out-of-fold predictions (results_uci/oof_predictions_full.npz).

Procedure
  * One bootstrap replicate = one resample of the 851 patients (with replacement,
    implemented as multinomial weights). The SAME resample is applied to every
    method, lambda and CV repeat, so differences between methods are PAIRED.
  * For each replicate the metric is averaged over the 20 CV repeats.
  * 95% CI = 2.5 / 97.5 percentiles over B replicates.

What the CI does and does not cover
  * It captures uncertainty from WHICH patients are in the data (test-side
    sampling of the pooled out-of-fold predictions).
  * It does NOT refit the models on each resample, so training-set sampling
    variability beyond the 20 CV repeats is not included. State this in the paper.

Outputs (in --outdir):
  uci_bootstrap_metrics.csv   per (lambda, method): point estimate + 95% CI
  uci_bootstrap_diffs.csv     paired differences vs naive_intersectional + CI
  uci_frontier_offsets.csv    offset from the naive accuracy-fairness frontier
                              at equal AUROC (negative = better than naive)
"""
from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd

from data_uci import load_uci, add_intersectional_group, GROUPS

METRICS = ["auroc", "dp_inter", "eo_inter", "dp_sex", "eo_sex"]


def indicator(codes, k):
    return np.eye(k, dtype=np.float32)[codes]


def gap(rate):                       # rate: (B, K) -> (B,)  max-min ignoring NaN
    return np.nanmax(rate, axis=1) - np.nanmin(rate, axis=1)


def weighted_fairness(pred, y, W, I, den_all, den_pos, den_neg):
    """DP and EO gaps for every bootstrap weight vector in W (B, n).
    Same definitions as fairness_utils (EO = max of TPR gap and FPR gap)."""
    pred = pred.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = (W @ (pred[:, None] * I)) / den_all
        tpr = (W @ ((pred * y)[:, None] * I)) / den_pos
        fpr = (W @ ((pred * (1 - y))[:, None] * I)) / den_neg
    return gap(rate), np.maximum(gap(tpr), gap(fpr))


def weighted_auc(score, y, W):
    order = np.argsort(score, kind="mergesort")
    ys = y[order].astype(np.float32)
    Ws = W[:, order]
    neg = Ws * (1 - ys)
    pos = Ws * ys
    neg_before = np.cumsum(neg, axis=1) - neg
    return (pos * neg_before).sum(1) / (pos.sum(1) * neg.sum(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="uci_heart_clean.csv")
    ap.add_argument("--npz", default="results_uci/oof_predictions_full.npz")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--outdir", default="results_uci")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()

    df = add_intersectional_group(load_uci(a.csv)).reset_index(drop=True)
    d = np.load(a.npz, allow_pickle=True)
    P, Y = d["proba"], d["pred"]                    # (L, S, M, n)
    lambdas, methods = list(d["lambdas"]), list(d["methods"])
    L, S, M, n = P.shape
    y = df["target"].values.astype(np.float32)

    I_int = indicator(pd.Categorical(df["intersect_group"], categories=GROUPS).codes, len(GROUPS))
    I_sex = indicator(df["sex"].values.astype(int), 2)
    rng = np.random.default_rng(a.seed)

    def run(W):
        """Return dict metric -> array (L, M, B) averaged over CV repeats."""
        B = W.shape[0]
        dens = {}
        for name, I in (("int", I_int), ("sex", I_sex)):
            dens[name] = (W @ I, W @ (y[:, None] * I), W @ ((1 - y)[:, None] * I))
        out = {k: np.zeros((L, M, B), dtype=np.float64) for k in METRICS}
        for li in range(L):
            for mi in range(M):
                acc = {k: np.zeros(B) for k in METRICS}
                for si in range(S):
                    p, s = Y[li, si, mi], P[li, si, mi]
                    dp_i, eo_i = weighted_fairness(p, y, W, I_int, *dens["int"])
                    dp_s, eo_s = weighted_fairness(p, y, W, I_sex, *dens["sex"])
                    acc["dp_inter"] += dp_i; acc["eo_inter"] += eo_i
                    acc["dp_sex"] += dp_s;   acc["eo_sex"] += eo_s
                    acc["auroc"] += weighted_auc(s, y, W)
                for k in METRICS:
                    out[k][li, mi] = acc[k] / S
        return out

    # point estimates (all weights = 1) + self-test against the original summary
    point = run(np.ones((1, n), dtype=np.float32))
    summ = pd.read_csv(os.path.join(a.outdir, "summary_full.csv"))
    chk = []
    for li, lam in enumerate(lambdas):
        for mi, m in enumerate(methods):
            r = summ[(summ.fairness_lambda == lam) & (summ.method == m)].iloc[0]
            chk.append([abs(point["dp_inter"][li, mi, 0] - r.DP_inter_mean),
                        abs(point["eo_inter"][li, mi, 0] - r.EO_inter_mean),
                        abs(point["dp_sex"][li, mi, 0] - r.DP_sex_mean),
                        abs(point["eo_sex"][li, mi, 0] - r.EO_sex_mean),
                        abs(point["auroc"][li, mi, 0] - r.AUROC_mean)])
    chk = np.array(chk)
    print("SELF-TEST vs summary_full.csv (max abs diff):",
          dict(zip(["DP_inter", "EO_inter", "DP_sex", "EO_sex", "AUROC"], chk.max(0).round(5))))
    assert chk[:, :4].max() < 5e-4 and chk[:, 4].max() < 0.1, "vectorised metrics disagree with original code"

    # bootstrap
    W = rng.multinomial(n, np.full(n, 1.0 / n), size=a.B).astype(np.float32)
    boot = run(W)

    ci = lambda x: np.percentile(x, [2.5, 97.5], axis=-1)

    # 1) metrics with CI
    rows = []
    for li, lam in enumerate(lambdas):
        for mi, m in enumerate(methods):
            row = {"fairness_lambda": lam, "method": m}
            for k in METRICS:
                lo, hi = ci(boot[k][li, mi])
                row[f"{k}"] = round(float(point[k][li, mi, 0]), 4)
                row[f"{k}_lo"], row[f"{k}_hi"] = round(float(lo), 4), round(float(hi), 4)
            rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(a.outdir, "uci_bootstrap_metrics.csv"), index=False)

    # 2) paired differences vs naive_intersectional
    ni = methods.index("naive_intersectional")
    rows = []
    for li, lam in enumerate(lambdas):
        for mi, m in enumerate(methods):
            if mi == ni:
                continue
            row = {"fairness_lambda": lam, "method": m}
            for k in ["auroc", "dp_inter", "eo_inter"]:
                diff = boot[k][li, mi] - boot[k][li, ni]
                lo, hi = ci(diff)
                row[f"d_{k}"] = round(float(point[k][li, mi, 0] - point[k][li, ni, 0]), 4)
                row[f"d_{k}_lo"], row[f"d_{k}_hi"] = round(float(lo), 4), round(float(hi), 4)
                row[f"d_{k}_excl0"] = bool(lo > 0 or hi < 0)
            rows.append(row)
    diffs = pd.DataFrame(rows)
    diffs.to_csv(os.path.join(a.outdir, "uci_bootstrap_diffs.csv"), index=False)

    # 3) frontier offsets at equal AUROC (naive curve over lambda is the reference)
    naive_auc_pt = point["auroc"][:, ni, 0]
    lo_pt, hi_pt = naive_auc_pt.min(), naive_auc_pt.max()
    rows = []
    for fam, members in {"oversample": ["oversample_intersectional"], "over_label": ["over_label"], "w_label": ["w_label"], "w_cell": ["w_cell"], "w_group": ["w_group"], "eo_pen": ["eo_pen"]}.items():
        for k in ["dp_inter", "eo_inter"]:
            pts = [(li, methods.index(m)) for m in members for li in range(L)
                   if lo_pt <= point["auroc"][li, methods.index(m), 0] <= hi_pt]
            if not pts:
                continue
            off_b = np.zeros(a.B)
            for b in range(a.B):
                x = boot["auroc"][:, ni, b]
                o = np.argsort(x)
                f = boot[k][:, ni, b]
                offs = [boot[k][li, mi, b] - np.interp(boot["auroc"][li, mi, b], x[o], f[o]) for li, mi in pts]
                off_b[b] = np.mean(offs)
            xp = point["auroc"][:, ni, 0]; op = np.argsort(xp)
            off_pt = np.mean([point[k][li, mi, 0] - np.interp(point["auroc"][li, mi, 0], xp[op], point[k][:, ni, 0][op])
                              for li, mi in pts])
            lo, hi = np.percentile(off_b, [2.5, 97.5])
            rows.append({"family": fam, "metric": k, "n_points": len(pts), "offset": round(float(off_pt), 4),
                         "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
                         "ci_excludes_0": bool(lo > 0 or hi < 0)})
    fr = pd.DataFrame(rows)
    fr.to_csv(os.path.join(a.outdir, "uci_frontier_offsets.csv"), index=False)

    pd.set_option("display.width", 220)
    print("\nFRONTIER OFFSETS at equal AUROC (negative = better than naive penalty):")
    print(fr.to_string(index=False))
    print("\nPAIRED DIFFERENCES vs naive_intersectional (lambda=4 and 8):")
    print(diffs[diffs.fairness_lambda.isin([4.0, 8.0])][
        ["fairness_lambda", "method", "d_dp_inter", "d_dp_inter_lo", "d_dp_inter_hi",
         "d_eo_inter", "d_eo_inter_lo", "d_eo_inter_hi"]].to_string(index=False))
    sx = diffs[diffs.method == "single_attr_sex"]
    print("\nSINGLE-ATTRIBUTE (sex) vs naive_intersectional, intersectional gaps:")
    print(sx[["fairness_lambda", "d_dp_inter", "d_dp_inter_lo", "d_dp_inter_hi",
              "d_eo_inter", "d_eo_inter_lo", "d_eo_inter_hi"]].to_string(index=False))


if __name__ == "__main__":
    main()
