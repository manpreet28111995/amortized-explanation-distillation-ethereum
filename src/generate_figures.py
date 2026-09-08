"""Generate all publication figures for Idea 8 from results/*. 300 dpi PNG + PDF."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 300, "savefig.bbox": "tight"})
C = {"SHAP": "#0072B2", "LIME": "#D55E00", "sur": "#009E73"}

res = pd.read_csv("results/main_raw.csv")
swp = pd.read_csv("results/sweep_raw.csv")
sca = np.load("results/scatter.npz")


def save(fig, name):
    fig.savefig(f"figures/{name}.png")
    fig.savefig(f"figures/{name}.pdf")
    plt.close(fig)
    print("saved", name)


# ---- Fig 1: latency (log scale) ----------------------------------------------
fig, ax = plt.subplots(figsize=(4.8, 3.3))
labels, vals, errs, cols = [], [], [], []
for t in ("SHAP", "LIME"):
    r = res[res.teacher == t]
    labels += [f"Kernel{t}\n(teacher)" if t == "SHAP" else f"{t}\n(teacher)",
               f"{t}\nsurrogate"]
    vals += [r.teacher_lat_s.mean() * 1e3, r.surrogate_lat_s.mean() * 1e3]
    errs += [r.teacher_lat_s.std() * 1e3, r.surrogate_lat_s.std() * 1e3]
    cols += [C[t], C["sur"]]
bars = ax.bar(labels, vals, yerr=errs, capsize=3, color=cols)
ax.set_yscale("log")
ax.set_ylabel("Latency per explanation (ms, log)")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v * 1.25,
            f"{v:.3g}", ha="center", fontsize=8)
sp_s = res[res.teacher == "SHAP"].speedup.mean()
sp_l = res[res.teacher == "LIME"].speedup.mean()
ax.set_title(f"Mean speed-up: {sp_s:,.0f}$\\times$ (SHAP), "
             f"{sp_l:,.0f}$\\times$ (LIME)", fontsize=9)
save(fig, "fig1_latency")

# ---- Fig 2: amortization sweep -----------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
for ax, met, ttl in zip(axes, ("topk_overlap", "r2"),
                        ("Top-3 overlap with teacher", "$R^2$ vs teacher")):
    for t in ("SHAP", "LIME"):
        g = swp[swp.teacher == t].groupby("n_teacher")[met]
        ax.errorbar(g.mean().index, g.mean(), yerr=g.std(), color=C[t],
                    marker="o", capsize=3, label=f"{t} surrogate")
    ax.set_xlabel("Teacher explanations used for training")
    ax.set_title(ttl, fontsize=10)
axes[0].legend(fontsize=8)
save(fig, "fig2_amortization")

# ---- Fig 3: teacher vs surrogate SHAP scatter ----------------------------------
fig, ax = plt.subplots(figsize=(4.0, 3.6))
t, p = sca["truth"].ravel(), sca["pred"].ravel()
hb = ax.hexbin(t, p, gridsize=60, cmap="Blues", mincnt=1, bins="log")
lim = np.percentile(np.abs(t), 99.5)
ax.plot([-lim, lim], [-lim, lim], "r--", lw=1)
r = np.corrcoef(t, p)[0, 1]
ax.set_xlabel("Teacher KernelSHAP value")
ax.set_ylabel("Surrogate prediction")
ax.set_title(f"Pearson r = {r:.3f}  (all seeds, held-out)", fontsize=9)
fig.colorbar(hb, ax=ax, label="log count")
save(fig, "fig3_scatter")

# ---- Fig 4: break-even total cost ---------------------------------------------
fig, ax = plt.subplots(figsize=(4.8, 3.3))
r = res[res.teacher == "SHAP"]
lat_t = r.teacher_lat_s.mean()
lat_s = r.surrogate_lat_s.mean()
train = r.surrogate_train_s.mean()
setup = 300 * lat_t + train                      # 300 teacher labels + training
n = np.arange(0, 20001, 50)
ax.plot(n, n * lat_t, color=C["SHAP"], label="Direct KernelSHAP")
ax.plot(n, setup + n * lat_s, color=C["sur"],
        label="Amortized surrogate (incl. setup)")
be = setup / max(lat_t - lat_s, 1e-9)
ax.axvline(be, color="k", ls=":", lw=1)
ax.annotate(f"break-even\nN $\\approx$ {be:,.0f}", (be, setup * 1.6),
            fontsize=8, ha="left")
ax.set_xlabel("Number of explanations served")
ax.set_ylabel("Total compute time (s)")
ax.legend(fontsize=8)
save(fig, "fig4_breakeven")

# ---- Fig 5: fidelity distributions across seeds --------------------------------
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
for ax, met, ttl in zip(axes, ("sign_agree", "topk_overlap", "kendall_tau"),
                        ("Sign agreement", "Top-3 overlap", "Kendall's $\\tau$")):
    data = [res[res.teacher == t][met] for t in ("SHAP", "LIME")]
    bp = ax.boxplot(data, tick_labels=["SHAP", "LIME"], patch_artist=True,
                    widths=0.5)
    for p, t in zip(bp["boxes"], ("SHAP", "LIME")):
        p.set_facecolor(C[t]); p.set_alpha(0.6)
    ax.set_title(ttl, fontsize=10)
save(fig, "fig5_fidelity_dist")

# ---- Fig 6: methodology pipeline diagram ----------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 2.9))
ax.axis("off")
top = [("Ethereum accounts\n9,841 (22.1% fraud)", 0.03),
       ("GBM fraud\nclassifier\n(top-20 features)", 0.21),
       ("High-risk alert\nstream (400/seed)", 0.39)]
for txt, x in top:
    ax.add_patch(plt.Rectangle((x, 0.55), 0.14, 0.38, fc="#E8F0FE",
                               ec="#0072B2", lw=1.2))
    ax.text(x + 0.07, 0.74, txt, ha="center", va="center", fontsize=7.5)
    if x < 0.39:
        ax.annotate("", xy=(x + 0.18, 0.74), xytext=(x + 0.14, 0.74),
                    arrowprops=dict(arrowstyle="->", color="#333"))
off = [("Teacher explanations\nKernelSHAP (300 samp.)\nLIME (1000 pert.)", 0.585, "#FFF3E0", "#D55E00"),
       ("Amortized MLP surrogates\n$g_\\theta: x \\mapsto \\phi(x)$\n(128–128)", 0.585, "#E8F5E9", "#009E73")]
ax.add_patch(plt.Rectangle((0.585, 0.55), 0.17, 0.38, fc="#FFF3E0", ec="#D55E00", lw=1.2))
ax.text(0.67, 0.74, "Teacher explanations\nKernelSHAP / LIME\n(offline, slow)",
        ha="center", va="center", fontsize=7.5)
ax.add_patch(plt.Rectangle((0.585, 0.06), 0.17, 0.38, fc="#E8F5E9", ec="#009E73", lw=1.2))
ax.text(0.67, 0.25, "Amortized MLP surrogate\n$g_\\theta: x \\mapsto \\hat\\phi(x)$\n(online, fast)",
        ha="center", va="center", fontsize=7.5)
ax.add_patch(plt.Rectangle((0.82, 0.3), 0.155, 0.38, fc="#F3E5F5", ec="#6A1B9A", lw=1.2))
ax.text(0.8975, 0.49, "Benchmarks:\nlatency, fidelity,\nbreak-even", ha="center",
        va="center", fontsize=7.5)
ax.annotate("", xy=(0.585, 0.74), xytext=(0.53, 0.74),
            arrowprops=dict(arrowstyle="->", color="#333"))
ax.annotate("distill", xy=(0.67, 0.44), xytext=(0.67, 0.55),
            arrowprops=dict(arrowstyle="->", color="#333"), fontsize=7.5,
            ha="center")
ax.annotate("", xy=(0.82, 0.49), xytext=(0.755, 0.4),
            arrowprops=dict(arrowstyle="->", color="#333"))
ax.annotate("", xy=(0.82, 0.55), xytext=(0.755, 0.7),
            arrowprops=dict(arrowstyle="->", color="#333"))
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
save(fig, "fig0_pipeline")

print("All Idea-8 figures done.")
