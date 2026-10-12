"""
06_clustering_eval.py — FASE 4: Clustering & Evaluasi per Eksperimen
=====================================================================
Untuk setiap EXP_?.npy:
  - Jalankan K-Means dengan k = 2..6
  - Hitung Silhouette Score (SS), Davies-Bouldin Index (DBI),
    Calinski-Harabasz Index (CHI)
  - Tentukan k optimal per EXP
  - Simpan comparison_metrics.csv
  - Buat visualisasi heatmap SS dan barplot DBI

Referensi metrik evaluasi: Kurniawan et al. (2026) — Paper 8
"""

import sys
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    OUTPUTS_PREPROC, OUTPUTS_DATA, OUTPUTS_REPORTS,
    PLOTS_COMPARISON, PLOTS_PCA, PLOT_DPI,
    EXPERIMENT_CONFIGS, SEED,
)

K_RANGE = range(2, 7)   # k = 2, 3, 4, 5, 6


# ── Helpers ───────────────────────────────────────────────────────────────────

def print_header(title: str):
    bar = "═" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def print_sub(title: str):
    print(f"\n  ── {title}")


def load_exp(exp_id: str) -> tuple:
    """Load X.npy dan meta.json untuk satu eksperimen."""
    npy_path  = OUTPUTS_PREPROC / f"{exp_id}.npy"
    meta_path = OUTPUTS_PREPROC / f"{exp_id}_meta.json"

    if not npy_path.exists():
        raise FileNotFoundError(f"Tidak ditemukan: {npy_path}")

    X    = np.load(npy_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return X, meta


def run_kmeans_sweep(X: np.ndarray, exp_id: str) -> list:
    """Jalankan K-Means untuk setiap k, return list of result dicts."""
    results = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=20, max_iter=500)
        labels = km.fit_predict(X)

        # Hitung metrik
        ss  = silhouette_score(X, labels) if len(set(labels)) > 1 else 0.0
        dbi = davies_bouldin_score(X, labels) if len(set(labels)) > 1 else np.inf
        chi = calinski_harabasz_score(X, labels) if len(set(labels)) > 1 else 0.0

        results.append({
            "exp_id":   exp_id,
            "k":        k,
            "SS":       round(ss,  6),
            "DBI":      round(dbi, 6),
            "CHI":      round(chi, 4),
            "inertia":  round(km.inertia_, 4),
        })
        print(f"    k={k}: SS={ss:.4f}  DBI={dbi:.4f}  CHI={chi:.1f}")

    return results


def find_optimal_k(results: list) -> dict:
    """
    Tentukan k optimal per eksperimen berdasarkan SS tertinggi.
    Kembalikan dict ringkasan.
    """
    best = max(results, key=lambda r: r["SS"])
    return {
        "exp_id":    best["exp_id"],
        "best_k":    best["k"],
        "best_SS":   best["SS"],
        "best_DBI":  best["DBI"],
        "best_CHI":  best["CHI"],
    }


# ── Plot: Heatmap SS ──────────────────────────────────────────────────────────

def plot_ss_heatmap(df_metrics: pd.DataFrame):
    """Heatmap SS × (EXP × k)."""
    pivot = df_metrics.pivot(index="exp_id", columns="k", values="SS")

    exp_order = list(EXPERIMENT_CONFIGS.keys())
    pivot = pivot.reindex([e for e in exp_order if e in pivot.index])

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, cmap="YlGn", aspect="auto",
                   vmin=0, vmax=max(0.5, pivot.values.max()))

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"k={k}" for k in pivot.columns], fontsize=11)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = "black" if val < 0.5 * pivot.values.max() else "white"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Silhouette Score")
    ax.set_title("Silhouette Score — Perbandingan Semua Eksperimen\n"
                 "(lebih tinggi = cluster lebih baik)", fontweight="bold", pad=12)
    ax.set_xlabel("Jumlah Cluster (k)", fontsize=11)
    ax.set_ylabel("Eksperimen", fontsize=11)
    plt.tight_layout()

    PLOTS_COMPARISON.mkdir(parents=True, exist_ok=True)
    out = PLOTS_COMPARISON / "heatmap_SS.png"
    fig.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 {out.name}")


# ── Plot: Bar DBI per EXP ─────────────────────────────────────────────────────

def plot_dbi_barplot(df_best: pd.DataFrame):
    """Bar chart DBI di k optimal per EXP."""
    df_plot = df_best.sort_values("best_DBI")
    colors  = ["steelblue" if v < df_best["best_DBI"].median() else "salmon"
               for v in df_plot["best_DBI"]]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.barh(df_plot["exp_id"], df_plot["best_DBI"],
                   color=colors, edgecolor="white", alpha=0.85)

    for bar, val in zip(bars, df_plot["best_DBI"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    ax.axvline(x=df_best["best_DBI"].median(), color="gray",
               linestyle="--", lw=1.5, label="Median")
    ax.set_xlabel("Davies-Bouldin Index (lebih rendah lebih baik)", fontsize=11)
    ax.set_title("Davies-Bouldin Index pada k Optimal per Eksperimen",
                 fontweight="bold", pad=10)
    ax.legend()
    plt.tight_layout()

    out = PLOTS_COMPARISON / "barplot_DBI.png"
    fig.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 {out.name}")


# ── Plot: SS line per k ───────────────────────────────────────────────────────

def plot_ss_lines(df_metrics: pd.DataFrame):
    """Line plot SS vs k untuk semua EXP."""
    fig, ax = plt.subplots(figsize=(11, 5))

    cmap = plt.get_cmap("tab10")
    exp_ids = df_metrics["exp_id"].unique()
    for i, exp_id in enumerate(exp_ids):
        sub = df_metrics[df_metrics["exp_id"] == exp_id].sort_values("k")
        cfg = EXPERIMENT_CONFIGS.get(exp_id, {})
        label = cfg.get("label", exp_id)
        ax.plot(sub["k"], sub["SS"], "o-", lw=2, ms=6,
                color=cmap(i), label=f"{exp_id}: {label}")

    ax.set_xlabel("Jumlah Cluster (k)", fontsize=11)
    ax.set_ylabel("Silhouette Score", fontsize=11)
    ax.set_title("Silhouette Score vs k — Semua Eksperimen",
                 fontweight="bold", pad=10)
    ax.set_xticks(list(K_RANGE))
    ax.legend(fontsize=7.5, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = PLOTS_COMPARISON / "lines_SS_vs_k.png"
    fig.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 {out.name}")


# ── Plot: Cluster scatter per EXP (k optimal) ─────────────────────────────────

def plot_cluster_scatter(X: np.ndarray, labels: np.ndarray,
                         profile: pd.DataFrame, exp_id: str, k: int):
    """Scatter PC1 vs PC2 diwarnai oleh cluster dan oleh label kondisi."""
    if X.shape[1] < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel kiri: warna per cluster
    cmap_cl = plt.get_cmap("tab10", k)
    sc0 = axes[0].scatter(X[:, 0], X[:, 1], c=labels, cmap=cmap_cl,
                          alpha=0.5, s=15, edgecolors="none",
                          vmin=-0.5, vmax=k - 0.5)
    plt.colorbar(sc0, ax=axes[0], ticks=range(k), label="Cluster")
    axes[0].set_title(f"Cluster Label (k={k})", fontweight="bold")
    axes[0].set_xlabel("Dim-1")
    axes[0].set_ylabel("Dim-2")

    # Panel kanan: warna per kondisi cooler (paling informatif)
    sc1 = axes[1].scatter(X[:, 0], X[:, 1], c=profile["cooler_condition"].values,
                          cmap="viridis", alpha=0.5, s=15, edgecolors="none")
    plt.colorbar(sc1, ax=axes[1], label="Cooler Condition")
    axes[1].set_title("Cooler Condition Label (validasi)", fontweight="bold")
    axes[1].set_xlabel("Dim-1")

    fig.suptitle(f"{exp_id} — Cluster Scatter (k={k})\n"
                 f"{EXPERIMENT_CONFIGS.get(exp_id, {}).get('label', '')}",
                 fontweight="bold", fontsize=11)
    plt.tight_layout()

    PLOTS_COMPARISON.mkdir(parents=True, exist_ok=True)
    out = PLOTS_COMPARISON / f"scatter_cluster_{exp_id}_k{k}.png"
    fig.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUTS_REPORTS.mkdir(parents=True, exist_ok=True)
    PLOTS_COMPARISON.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print("\n" + "█" * 60)
    print("  06_clustering_eval.py — Clustering & Evaluasi")
    print("█" * 60)

    profile = pd.read_csv(OUTPUTS_DATA / "profile.csv")

    all_results = []
    best_per_exp = []

    for exp_id in EXPERIMENT_CONFIGS:
        print_header(f"{exp_id}: {EXPERIMENT_CONFIGS[exp_id]['label']}")
        try:
            X, meta = load_exp(exp_id)
            print(f"  Shape: {X.shape}")
            print_sub("K-Means sweep")
            results = run_kmeans_sweep(X, exp_id)
            all_results.extend(results)

            opt = find_optimal_k(results)
            best_per_exp.append(opt)
            print(f"\n  ★ k optimal = {opt['best_k']}  "
                  f"(SS={opt['best_SS']:.4f}, DBI={opt['best_DBI']:.4f}, "
                  f"CHI={opt['best_CHI']:.1f})")

            # Scatter plot di k optimal
            km_best = KMeans(n_clusters=opt["best_k"], random_state=SEED,
                             n_init=20, max_iter=500)
            labels_best = km_best.fit_predict(X)
            plot_cluster_scatter(X, labels_best, profile, exp_id, opt["best_k"])

        except Exception as e:
            print(f"  ❌ GAGAL: {e}")
            best_per_exp.append({"exp_id": exp_id, "best_k": None,
                                  "best_SS": None, "best_DBI": None,
                                  "best_CHI": None})

    # ── Simpan CSV ───────────────────────────────────────────────────────────
    df_all  = pd.DataFrame(all_results)
    df_best = pd.DataFrame(best_per_exp)

    metrics_path = OUTPUTS_REPORTS / "comparison_metrics.csv"
    best_path    = OUTPUTS_REPORTS / "best_k_per_exp.csv"
    df_all.to_csv(metrics_path, index=False)
    df_best.to_csv(best_path,   index=False)
    print(f"\n  💾 {metrics_path.name}  [{df_all.shape}]")
    print(f"  💾 {best_path.name}  [{df_best.shape}]")

    # ── Visualisasi ──────────────────────────────────────────────────────────
    print_header("Visualisasi Perbandingan")
    plot_ss_heatmap(df_all)
    plot_dbi_barplot(df_best.dropna())
    plot_ss_lines(df_all)

    # ── Tabel Ringkasan ──────────────────────────────────────────────────────
    print_header("Ringkasan Hasil Clustering")
    print(f"\n  {'EXP':<8} {'Label':<32} {'k*':<5} {'SS':<8} {'DBI':<8} {'CHI'}")
    print(f"  {'─'*8} {'─'*32} {'─'*5} {'─'*8} {'─'*8} {'─'*8}")
    df_sorted = df_best.dropna().sort_values("best_SS", ascending=False)
    for _, row in df_sorted.iterrows():
        lbl = EXPERIMENT_CONFIGS.get(row["exp_id"], {}).get("label", "")[:31]
        print(f"  {row['exp_id']:<8} {lbl:<32} {int(row['best_k']):<5} "
              f"{row['best_SS']:<8.4f} {row['best_DBI']:<8.4f} {row['best_CHI']:.1f}")

    winner = df_sorted.iloc[0]
    print(f"\n  🏆 Eksperimen terbaik: {winner['exp_id']}  "
          f"(SS={winner['best_SS']:.4f} pada k={int(winner['best_k'])})")

    elapsed = time.time() - t_start
    print(f"\n  ⏱  Total: {elapsed:.1f} detik")
    print(f"  📂 Output di: {OUTPUTS_REPORTS}")
    print(f"  📂 Plots di: {PLOTS_COMPARISON}\n")


if __name__ == "__main__":
    main()
