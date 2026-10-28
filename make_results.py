"""Build all tables (json) and figures from res/*.csv."""
import numpy as np, pandas as pd, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import analysis as A

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 8.5, "legend.frameon": False, "figure.dpi": 200,
                     "font.family": "serif"})
C = {"fedavg": "#444444", "sexonly": "#E69F00", "naive": "#0072B2", "oversample": "#D55E00",
     "shrink": "#009E73", "w_group": "#CC79A7", "over_label": "#7B3294", "eo_pen": "#56B4E9"}
LAB = {"fedavg": "FedAvg", "sexonly": "Sex-only", "naive": "Naive inter.", "oversample": "Oversample",
       "shrink": "Shrinkage", "w_group": "Group-reweight", "over_label": "Label-balance",
       "eo_pen": "EO penalty"}
os.makedirs("figs", exist_ok=True)
OUT = {}


def fmt(m, s):
    return f"{m:.3f} ± {s:.3f}"


def table_main(scen):
    d = A.load("core", scen)
    if d is None:
        return None
    rows = {}
    for m in ["fedavg", "sexonly", "naive", "oversample", "shrink"]:
        x = d[(d.method == m) & ((d.lam == 8) | (m == "fedavg"))]
        rows[m] = {k: (x[k].mean(), x[k].std()) for k in ["AUROC", "PRAUC", "DP_sex", "EO_sex", "DP_int", "EO_int"]}
        rows[m]["n"] = int(x.seed.nunique())
    return rows


def offs_table(scen):
    out = {}
    d = A.load("core", scen)
    if d is not None:
        for m in ["oversample", "shrink"]:
            for met in ["DP_int", "EO_int"]:
                out[(m, met)] = A.ci_boot(A.offsets(d, m, metric=met))
    c = A.load("controls", scen)
    if c is not None:
        # all control rows use the common core lambda points (dense-only lambdas 0.5, 2 excluded)
        # so rows are comparable; the dense-grid rows are added separately by dense.py
        cc = c[~(c.lam.isin([0.5, 2]) & (c.method != 'naive'))]
        for m in ["oversample", "over_origagg", "over_nonoise", "over_label", "w_cell", "w_label", "w_group"]:
            for met in ["DP_int", "EO_int"]:
                out[("ctl_" + m, met)] = A.ci_boot(A.offsets(cc, m, metric=met))
                out[("dense_" + m, met)] = A.ci_boot(A.offsets(c, m, metric=met))  # full (dense) grid
    e = A.load("eo", scen)
    if e is not None:
        for m in ["eo_pen", "eo_pen_over", "eo_pen_shrink"]:
            for met in ["DP_int", "EO_int"]:
                out[("eo_" + m, met)] = A.ci_boot(A.offsets(e, m, metric=met))
    x = A.load("eoext", scen)
    if x is not None:
        for m in ["eo_pen", "eo_pen_shrink"]:
            for met in ["DP_int", "EO_int"]:
                out[("eoext_" + m, met)] = A.ci_boot(A.offsets(x, m, metric=met))
    for suite, ms in [("mlp", ["oversample", "shrink", "sexonly"])]:
        z = A.load(suite, scen)
        if z is not None:
            for m in ms:
                for met in ["DP_int", "EO_int"]:
                    out[("mlp_" + m, met)] = A.ci_boot(A.offsets(z, m, metric=met))
    g = A.load("agecut", scen)
    if g is not None:
        for cut in ["45", "65"]:
            gg = g[(g.cut == cut) | (g.cut == int(cut))]
            gg = gg[gg.grp == "agesex"]
            for m in ["oversample", "shrink"]:
                for met in ["DP_int", "EO_int"]:
                    out[(f"cut{cut}_{m}", met)] = A.ci_boot(A.offsets(gg, m, metric=met))
        g3 = g[g.grp == "age3sex"]
        if len(g3):
            for met in ["DP_int", "EO_int"]:
                out[("age3_oversample", met)] = A.ci_boot(A.offsets(g3, "oversample", metric=met))
    return out


def fig_mask(scen):
    d = A.load("core", scen)
    if d is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(3.4, 1.9), sharey=True)
    for a, met, t in zip(ax, ["DP", "EO"], ["Demographic parity gap", "Equalized-odds gap"]):
        vals = []
        for m in ["fedavg", "sexonly"]:
            x = d[(d.method == m) & ((d.lam == 8) | (m == "fedavg"))]
            vals.append((x[f"{met}_sex"].mean(), x[f"{met}_int"].mean(), x[f"{met}_sex"].std(), x[f"{met}_int"].std()))
        xs = np.arange(2)
        a.bar(xs - 0.2, [v[0] for v in vals], 0.38, yerr=[v[2] for v in vals], color="#9ecae1", label="2 sex groups", capsize=1.5, lw=0.5)
        a.bar(xs + 0.2, [v[1] for v in vals], 0.38, yerr=[v[3] for v in vals], color="#08519c", label="4 age×sex groups", capsize=1.5, lw=0.5)
        a.set_xticks(xs)
        a.set_xticklabels(["FedAvg", "Sex-only\n(λ=8)"])
        a.set_title(t)
    ax[0].legend(fontsize=6.5, loc="upper right")
    ax[0].set_ylabel("gap (lower = fairer)")
    fig.tight_layout()
    fig.savefig(f"figs/mask_{scen}.pdf")
    plt.close(fig)


def fig_frontier(scen):
    d = A.load("controls", scen)
    if d is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.3))
    for a, met, t in zip(ax, ["DP_int", "EO_int"], ["AUROC vs. intersectional DP gap", "AUROC vs. intersectional EO gap"]):
        for m in ["naive", "oversample", "w_group", "over_label"]:
            g = d[d.method == m].groupby("lam")[["AUROC", met]].mean().sort_index()
            a.plot(g["AUROC"], g[met], "-o", ms=3, lw=1, color=C[m], label=LAB[m])
        a.set_xlabel("AUROC")
        a.set_ylabel("gap")
        a.set_title(t)
    ax[1].legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(f"figs/frontier_{scen}.pdf")
    plt.close(fig)


def fig_controls(scen, off):
    keys = [("ctl_oversample", "Oversample\n(cells)"), ("ctl_over_origagg", "…orig. $n_k$\nweights"),
            ("ctl_over_nonoise", "…no noise"), ("ctl_over_label", "Label-balance\nonly (dup.)"),
            ("ctl_w_label", "Label\nreweight"), ("ctl_w_cell", "Cell\nreweight"), ("ctl_w_group", "Group\nreweight")]
    keys = [k for k in keys if (k[0], "EO_int") in off]
    if not keys:
        return
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5), sharex=True)
    for a, met, t in zip(ax, ["EO_int", "DP_int"], ["EO offset vs. naive (equal AUROC)", "DP offset vs. naive (equal AUROC)"]):
        for i, (k, lab) in enumerate(keys):
            m, lo, hi, _ = off[(k, met)]
            a.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="o", color="#D55E00" if m > 0 else "#0072B2", capsize=2, ms=3.5, lw=1)
        a.axhline(0, color="k", lw=0.6)
        a.set_xticks(range(len(keys)))
        a.set_xticklabels([k[1].replace("\n"," ") for k in keys], fontsize=6, rotation=30, ha="right")
        a.set_title(t)
    fig.tight_layout()
    fig.savefig(f"figs/controls_{scen}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    for scen in ["natural", "capped"]:
        OUT[scen] = {"main": table_main(scen)}
        off = offs_table(scen)
        OUT[scen]["offsets"] = {f"{k[0]}|{k[1]}": [float(v) for v in vals] for k, vals in off.items()}
        fig_mask(scen)
        fig_frontier(scen)
        fig_controls(scen, off)
    json.dump(OUT, open("results.json", "w"), indent=1, default=lambda o: float(o))
    print("ok")
