"""
Distilling Explanations for Ultra-Low-Latency Streaming Inference
==========================================================================
Full experimental pipeline on the REAL Ethereum Fraud Detection dataset
(transaction_dataset.csv, Kaggle: vagifa/ethereum-frauddetection-dataset;
features from Farrugia et al., 2020).

Pipeline:
  1. Gradient-boosting fraud classifier over account-level features.
  2. "Teacher" explanations for high-risk accounts:
        KernelSHAP (nsamples=300) and LIME (1000 perturbations).
  3. Amortized MLP surrogates  x -> explanation vector,
     trained on {50, 100, 200, 300} teacher explanations (amortization sweep).
  4. Benchmarks (10 seeds, mean +/- std):
        latency & throughput  |  fidelity: MSE, R^2, sign agreement,
        top-k overlap, Kendall tau  |  local-accuracy (efficiency) gap
        |  break-even analysis of total explanation cost.

Outputs: results/*.csv, figures/*.png (300 dpi).

Usage:
  python idea8_experiments.py --csv transaction_dataset.csv
"""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import shap
from lime import lime_tabular

warnings.filterwarnings("ignore")

N_SEEDS = 10
SEEDS = list(range(N_SEEDS))
N_FEATURES = 20            # KernelSHAP tractability: top-20 by GBM importance
N_TEACHER_POOL = 300       # teacher explanations generated per seed
TEACHER_SIZES = [50, 100, 200, 300]
N_EVAL = 100               # held-out explanations for fidelity evaluation
TOP_K = 3
SHAP_NSAMPLES = 300
LIME_NSAMPLES = 1000

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)


def load_ethereum(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    y = df["FLAG"].astype(int).values
    X = (df.drop(columns=["Unnamed: 0", "Index", "Address", "FLAG"],
                 errors="ignore")
           .select_dtypes(include=[np.number]).fillna(0))
    X = X.loc[:, X.std() > 0]
    print(f"Ethereum accounts: {len(y)}, fraud rate {y.mean():.3f}, "
          f"{X.shape[1]} numeric features (pre-selection)")
    return X, y


def teacher_shap(model, bg, X, base_val):
    f = lambda z: model.predict_proba(z)[:, 1]
    ex = shap.KernelExplainer(f, bg)
    t0 = time.perf_counter()
    v = np.asarray(ex.shap_values(X, nsamples=SHAP_NSAMPLES, silent=True))
    lat = (time.perf_counter() - t0) / len(X)
    v = v[0] if v.ndim == 3 else v
    return v, lat, float(ex.expected_value)


def teacher_lime(model, X_train, X, feats):
    ex = lime_tabular.LimeTabularExplainer(
        X_train, feature_names=feats, mode="classification",
        discretize_continuous=True, random_state=0)
    out = []
    t0 = time.perf_counter()
    for x in X:
        e = ex.explain_instance(x, model.predict_proba,
                                num_features=len(feats),
                                num_samples=LIME_NSAMPLES)
        w = np.zeros(len(feats))
        for fid, wt in e.as_map()[1]:
            w[fid] = wt
        out.append(w)
    lat = (time.perf_counter() - t0) / len(X)
    return np.array(out), lat


def fidelity_metrics(truth, pred, k=TOP_K):
    mse = float(np.mean((truth - pred) ** 2))
    r2 = float(r2_score(truth.ravel(), pred.ravel()))
    sign = float(np.mean(np.sign(truth) == np.sign(pred)))
    taus, ovl = [], []
    for t, p in zip(truth, pred):
        taus.append(kendalltau(np.abs(t), np.abs(p))[0])
        tk = set(np.argsort(-np.abs(t))[:k])
        pk = set(np.argsort(-np.abs(p))[:k])
        ovl.append(len(tk & pk) / k)
    return dict(mse=mse, r2=r2, sign_agree=sign,
                topk_overlap=float(np.mean(ovl)),
                kendall_tau=float(np.nanmean(taus)))


def run(csv_path, seeds=SEEDS):
    X_df, y = load_ethereum(csv_path)
    rows, sweep_rows, scatter_store = [], [], []

    for seed in seeds:
        t_seed = time.time()
        Xtr_df, Xte_df, ytr, yte = train_test_split(
            X_df, y, test_size=0.3, random_state=seed, stratify=y)

        # feature selection: top-20 by GBM importance (fit on train only)
        sel_model = GradientBoostingClassifier(random_state=seed).fit(Xtr_df, ytr)
        top = np.argsort(-sel_model.feature_importances_)[:N_FEATURES]
        feats = [X_df.columns[i] for i in top]

        sc = StandardScaler().fit(Xtr_df.iloc[:, top])
        Xtr = sc.transform(Xtr_df.iloc[:, top])
        Xte = sc.transform(Xte_df.iloc[:, top])

        model = GradientBoostingClassifier(random_state=seed).fit(Xtr, ytr)
        p_te = model.predict_proba(Xte)[:, 1]
        auc, ap = roc_auc_score(yte, p_te), average_precision_score(yte, p_te)

        # explanation pool: highest-risk test accounts (alert stream)
        pool = np.argsort(-p_te)[:N_TEACHER_POOL + N_EVAL]
        Xp = Xte[pool]
        bg = KMeans(n_clusters=10, n_init=3,
                    random_state=seed).fit(
                        Xtr[np.random.default_rng(seed).choice(
                            len(Xtr), 1000, replace=False)]).cluster_centers_

        shap_t, shap_lat, base = teacher_shap(model, bg, Xp, None)
        lime_t, lime_lat = teacher_lime(model, Xtr, Xp, feats)

        # random train/eval split of the alert pool (avoids rank-induced shift)
        perm = np.random.default_rng(seed).permutation(len(Xp))
        tr_i, te_i = perm[:N_TEACHER_POOL], perm[N_TEACHER_POOL:]

        # ---- amortization sweep --------------------------------------------
        for n_teach in TEACHER_SIZES:
            for tname, T in [("SHAP", shap_t), ("LIME", lime_t)]:
                t0 = time.perf_counter()
                mu, sd = T[tr_i[:n_teach]].mean(0), T[tr_i[:n_teach]].std(0) + 1e-8
                sur = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=800,
                                   alpha=1e-3, random_state=seed)
                sur.fit(Xp[tr_i[:n_teach]], (T[tr_i[:n_teach]] - mu) / sd)
                train_t = time.perf_counter() - t0
                t0 = time.perf_counter()
                pred = sur.predict(Xp[te_i]) * sd + mu
                sur_lat = (time.perf_counter() - t0) / N_EVAL
                m = fidelity_metrics(T[te_i], pred)
                m.update(seed=seed, teacher=tname, n_teacher=n_teach,
                         surrogate_train_s=train_t, surrogate_lat_s=sur_lat)
                sweep_rows.append(m)
                if n_teach == N_TEACHER_POOL:
                    if tname == "SHAP":
                        # local accuracy (efficiency) gap of surrogate SHAP
                        f_alert = model.predict_proba(Xp[te_i])[:, 1]
                        gap_t = np.abs(shap_t[te_i].sum(1) + base - f_alert)
                        gap_s = np.abs(pred.sum(1) + base - f_alert)
                        eff_t, eff_s = float(gap_t.mean()), float(gap_s.mean())
                        scatter_store.append(
                            dict(seed=seed, truth=shap_t[te_i], pred=pred))
                    else:
                        eff_t = eff_s = np.nan
                    rows.append(dict(seed=seed, teacher=tname, model_auc=auc,
                                     model_ap=ap,
                                     teacher_lat_s=shap_lat if tname == "SHAP" else lime_lat,
                                     surrogate_lat_s=sur_lat,
                                     surrogate_train_s=train_t,
                                     speedup=(shap_lat if tname == "SHAP" else lime_lat) / max(sur_lat, 1e-9),
                                     eff_gap_teacher=eff_t, eff_gap_surrogate=eff_s,
                                     **{k: m[k] for k in
                                        ("mse", "r2", "sign_agree",
                                         "topk_overlap", "kendall_tau")}))
        print(f"[seed {seed}] AUC={auc:.3f} "
              f"SHAP {shap_lat*1e3:.0f}ms LIME {lime_lat*1e3:.0f}ms "
              f"({time.time()-t_seed:.0f}s)")

    tag = f"s{seeds[0]}_{seeds[-1]}"
    res = pd.DataFrame(rows)
    swp = pd.DataFrame(sweep_rows)
    res.to_csv(f"results/main_raw_{tag}.csv", index=False)
    swp.to_csv(f"results/sweep_raw_{tag}.csv", index=False)
    np.savez(f"results/scatter_{tag}.npz",
             truth=np.vstack([s["truth"] for s in scatter_store]),
             pred=np.vstack([s["pred"] for s in scatter_store]))
    print(res.groupby("teacher").mean(numeric_only=True))
    return res, swp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()
    run(args.csv, args.seeds)
