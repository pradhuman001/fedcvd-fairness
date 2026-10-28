import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import analysis as A
plt.rcParams.update({"font.size":8,"font.family":"serif","axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False})
c=A.load("controls","natural")
fig,ax=plt.subplots(1,2,figsize=(7,2.8))
cols={"naive":"#0072B2","oversample":"#D55E00","over_label":"#7B3294"}
lab={"naive":"Naive penalty","oversample":"Oversample","over_label":"Label-balance"}
for j,met in enumerate(["EO_int","DP_int"]):
    for m in cols:
        d=c[c.method==m].groupby("lam")[["AUROC",met]].mean().sort_index()
        d=d[d.index.isin([0.5,1,2,4,8,16])]
        ax[j].plot(d.AUROC,d[met],"o-",ms=3,lw=1,color=cols[m],label=lab[m])
        if j==0:
            for l,r in d.iterrows():
                if m=="naive": ax[j].annotate(f"{l:g}",(r.AUROC,r[met]),fontsize=6,xytext=(2,3),textcoords="offset points")
    ax[j].set_xlabel("AUROC"); ax[j].set_ylabel(met.replace("_int","$_{int}$"))
ax[0].set_title("EO gap (labels: naive $\\lambda$)"); ax[1].set_title("DP gap"); ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig("figs/frontier_dense_natural.pdf")
