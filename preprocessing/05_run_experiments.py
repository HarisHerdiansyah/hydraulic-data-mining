"""
05_run_experiments.py — FASE 3: Normalisasi + PCA per Eksperimen
=================================================================
Membaca 4 feature CSV dari Phase 2, lalu untuk setiap EXP:
  1. Load feature matrix yang sesuai
  2. Correlation filter (hapus r > 0.95)
  3. Normalisasi (Robust / Standard / MinMax)
  4. PCA (threshold dinamis atau skip)
  5. Simpan EXP_?.npy + EXP_?_meta.json

Output: outputs/data/preprocessed/EXP_?.npy (siap clustering)
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    OUTPUTS_FEATURES, OUTPUTS_PREPROC, OUTPUTS_DATA,
    PLOTS_PCA, PLOTS_COMPARISON, PLOT_DPI,
    CORR_THRESHOLD, NZV_THRESHOLD, SEED,
    EXPERIMENT_CONFIGS,
)

# ── Mapping feature_set key → file CSV ────────────────────────────────────────
FEATURE_SET_FILES = {
    "stats":               "features_stats.csv",
    "stats_fft":           "features_stats_fft.csv",
    "stats_fft_13cluster": "features_stats_fft_13cl.csv",
    "ewt":                 "features_ewt.csv",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def print_header(title: str):
    bar = "═" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def print_sub(title: str):
    print(f"\n  ── {title}")


def load_feature_matrix(feature_set: str) -> pd.DataFrame:
    fname = FEATURE_SET_FILES[feature_set]
    fpath = OUTPUTS_FEATURES / fname
    if not fpath.exists():
        raise FileNotFoundError(
            f"Feature file tidak ditemukan: {fpath}\n"
            f"Jalankan 03_feature_extraction.py terlebih dahulu."
        )
    df = pd.read_csv(fpath)
    print(f"  Feature file : {fname}  {df.shape}")
    return df


def apply_corr_filter(X: np.ndarray, col_names: list,
                      threshold: float = CORR_THRESHOLD,
                      min_features: int = 50) -> tuple:
    """
    Hapus salah satu dari setiap pasangan fitur dengan korelasi > threshold.
    Hanya dijalankan jika n_features > min_features.
    """
    n = X.shape[1]
    if n <= min_features:
        print(f"  Correlation filter: dilewati (n_features={n} ≤ {min_features})")
        return X, col_names

    corr = np.corrcoef(X.T)
    to_drop = set()
    for i in range(n):
        if i in to_drop:
            continue
        for j in range(i + 1, n):
            if j in to_drop:
                continue
            if abs(corr[i, j]) > threshold:
                to_drop.add(j)

    keep_mask = [i for i in range(n) if i not in to_drop]
    X_filtered = X[:, keep_mask]
    names_filtered = [col_names[i] for i in keep_mask]
    print(f"  Correlation filter (r>{threshold}): {n} → {X_filtered.shape[1]} fitur "
          f"(dihapus: {len(to_drop)})")
    return X_filtered, names_filtered


def apply_scaler(X: np.ndarray, scaler_type: str) -> np.ndarray:
    if scaler_type == "robust":
        scaler = RobustScaler()
    elif scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler(feature_range=(0, 1))
    else:
        raise ValueError(f"Scaler tidak dikenal: {scaler_type}")

    X_scaled = scaler.fit_transform(X)
    print(f"  Scaler : {scaler_type} → shape={X_scaled.shape}")
    return X_scaled


def apply_pca(X_scaled: np.ndarray, threshold: float | None,
              exp_id: str) -> tuple:
    """
    Jalankan PCA dengan threshold EVR. Jika threshold=None, skip PCA.
    Returns (X_out, n_components, actual_evr)
    """
    if threshold is None:
        print(f"  PCA : DILEWATI (threshold=None)")
        return X_scaled, X_scaled.shape[1], None

    # Fit full PCA dulu untuk analisis
    pca_full = PCA(random_state=SEED)
    pca_full.fit(X_scaled)
    cumul_evr = np.cumsum(pca_full.explained_variance_ratio_)

    k = int(np.argmax(cumul_evr >= threshold)) + 1
    k = max(k, 2)  # minimal 2 komponen untuk visualisasi

    actual_evr = float(cumul_evr[k - 1])
    print(f"  PCA {int(threshold*100)}% EVR : {X_scaled.shape[1]} → {k} komponen "
          f"(EVR aktual: {actual_evr*100:.2f}%)")

    # Transform final
    pca_final = PCA(n_components=k, random_state=SEED)
    X_pca = pca_final.fit_transform(X_scaled)

    # Simpan scree plot
    _plot_scree(pca_full.explained_variance_ratio_, cumul_evr, k, threshold, exp_id)

    return X_pca, k, actual_evr


def _plot_scree(evr: np.ndarray, cumul_evr: np.ndarray,
                k: int, threshold: float, exp_id: str):
    PLOTS_PCA.mkdir(parents=True, exist_ok=True)
    n_show = min(50, len(evr))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].bar(range(1, n_show + 1), evr[:n_show] * 100,
                color="steelblue", alpha=0.8, edgecolor="white")
    axes[0].axvline(x=k, color="crimson", linestyle="--", lw=2, label=f"K={k}")
    axes[0].set_title(f"Explained Variance / Komponen", fontweight="bold")
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("EVR (%)")
    axes[0].legend()

    axes[1].plot(range(1, len(cumul_evr) + 1), cumul_evr * 100,
                 "o-", color="steelblue", lw=1.5, ms=3)
    axes[1].axhline(y=threshold * 100, color="crimson", linestyle="--",
                    lw=1.5, label=f"{int(threshold*100)}% threshold")
    axes[1].axvline(x=k, color="crimson", linestyle=":", lw=1.5)
    axes[1].set_title("Cumulative Explained Variance", fontweight="bold")
    axes[1].set_xlabel("Jumlah Komponen")
    axes[1].set_ylabel("Cumulative EVR (%)")
    axes[1].set_xlim(1, min(80, len(cumul_evr)))
    axes[1].legend()

    fig.suptitle(f"PCA Scree Plot — {exp_id}", fontweight="bold")
    plt.tight_layout()
    out = PLOTS_PCA / f"scree_{exp_id}.png"
    fig.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 Scree plot: {out.name}")


def plot_pca_2d(X_out: np.ndarray, profile: pd.DataFrame, exp_id: str):
    """Scatter PC1 vs PC2 diwarnai per kondisi komponen."""
    if X_out.shape[1] < 2:
        return
    PLOTS_PCA.mkdir(parents=True, exist_ok=True)

    conditions = [
        ("cooler_condition", "viridis"),
        ("valve_condition",  "plasma"),
        ("pump_leakage",     "coolwarm"),
        ("accumulator",      "RdYlGn"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, (cond, cmap) in zip(axes.flatten(), conditions):
        sc = ax.scatter(X_out[:, 0], X_out[:, 1],
                        c=profile[cond].values, cmap=cmap,
                        alpha=0.5, s=15, edgecolors="none")
        plt.colorbar(sc, ax=ax, shrink=0.85)
        ax.set_title(cond.replace("_", " ").title(), fontweight="bold")
        ax.set_xlabel("PC1" if X_out.shape[1] >= 2 else "Dim1")
        ax.set_ylabel("PC2" if X_out.shape[1] >= 2 else "Dim2")

    fig.suptitle(f"2D Projection — {exp_id}\n(Label untuk validasi eksplorasi)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = PLOTS_PCA / f"scatter_{exp_id}.png"
    fig.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 Scatter plot: {out.name}")


# ── Main loop per eksperimen ──────────────────────────────────────────────────

def run_experiment(exp_id: str, cfg: dict, profile: pd.DataFrame) -> dict:
    print_header(f"{exp_id}: {cfg['label']}")
    t0 = time.time()

    # 1. Load feature matrix
    print_sub("1. Load Feature Matrix")
    df = load_feature_matrix(cfg["feature_set"])
    n_raw = df.shape[1]
    col_names = list(df.columns)
    X = df.values.astype(np.float64)

    # 2. Correlation filter
    print_sub("2. Correlation Filter")
    X, col_names = apply_corr_filter(X, col_names)
    n_after_corr = X.shape[1]

    # 3. Normalisasi
    print_sub(f"3. Normalisasi ({cfg['scaler']})")
    X_scaled = apply_scaler(X, cfg["scaler"])

    # 4. PCA
    print_sub(f"4. PCA (threshold={cfg['pca_threshold']})")
    X_out, n_components, actual_evr = apply_pca(
        X_scaled, cfg["pca_threshold"], exp_id
    )

    # 5. Simpan output
    print_sub("5. Simpan Output")
    OUTPUTS_PREPROC.mkdir(parents=True, exist_ok=True)

    npy_path = OUTPUTS_PREPROC / f"{exp_id}.npy"
    np.save(npy_path, X_out)
    print(f"  ✅ {npy_path.name}  shape={X_out.shape}")

    meta = {
        "exp_id":             exp_id,
        "label":              cfg["label"],
        "description":        cfg["description"],
        "papers":             cfg["papers"],
        "feature_set":        cfg["feature_set"],
        "outlier_cap":        cfg["outlier_cap"],
        "scaler":             cfg["scaler"],
        "pca_threshold":      cfg["pca_threshold"],
        "n_features_raw":     n_raw,
        "n_features_corr_filter": n_after_corr,
        "n_components_pca":   n_components,
        "pca_actual_evr":     round(actual_evr, 6) if actual_evr else None,
        "output_shape":       list(X_out.shape),
        "elapsed_sec":        round(time.time() - t0, 2),
    }
    meta_path = OUTPUTS_PREPROC / f"{exp_id}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  ✅ {meta_path.name}")

    # 6. Plot 2D projection
    print_sub("6. Plot 2D Projection")
    plot_pca_2d(X_out, profile, exp_id)

    elapsed = time.time() - t0
    print(f"\n  ⏱  {exp_id} selesai dalam {elapsed:.1f} detik")
    print(f"     {n_raw} raw fitur → {n_after_corr} (corr) → {n_components} komponen")

    return meta


def main():
    t_start = time.time()

    print("\n" + "█" * 60)
    print("  05_run_experiments.py — Multi-Experiment Preprocessing")
    print("█" * 60)

    # Load profile (shared)
    profile = pd.read_csv(OUTPUTS_DATA / "profile.csv")
    print(f"\nProfile dimuat: {profile.shape}")

    summary_rows = []

    for exp_id, cfg in EXPERIMENT_CONFIGS.items():
        try:
            meta = run_experiment(exp_id, cfg, profile)
            summary_rows.append({
                "exp_id":         meta["exp_id"],
                "label":          meta["label"],
                "feature_set":    meta["feature_set"],
                "scaler":         meta["scaler"],
                "pca_threshold":  meta["pca_threshold"],
                "n_raw_features": meta["n_features_raw"],
                "n_corr_filter":  meta["n_features_corr_filter"],
                "n_components":   meta["n_components_pca"],
                "pca_evr":        meta["pca_actual_evr"],
                "output_shape":   str(meta["output_shape"]),
                "elapsed_sec":    meta["elapsed_sec"],
                "status":         "OK",
            })
        except Exception as e:
            print(f"\n  ❌ {exp_id} GAGAL: {e}")
            summary_rows.append({
                "exp_id": exp_id, "status": f"ERROR: {e}"
            })

    # Simpan ringkasan
    OUTPUTS_PREPROC.mkdir(parents=True, exist_ok=True)
    df_summary = pd.DataFrame(summary_rows)
    summary_path = OUTPUTS_PREPROC / "experiments_summary.csv"
    df_summary.to_csv(summary_path, index=False)

    # Print tabel ringkasan
    print_header("Ringkasan Semua Eksperimen")
    print(f"{'EXP':<8} {'Feature Set':<28} {'Scaler':<10} {'PCA':<6} {'n_comp':<8} {'Shape'}")
    print("─" * 80)
    for _, row in df_summary.iterrows():
        if row.get("status", "OK") != "OK":
            print(f"  {row['exp_id']:<6} ❌ {row['status']}")
            continue
        pca_val = row['pca_threshold']
        pca_str = f"{int(pca_val*100)}%" if pd.notna(pca_val) and pca_val else "None"
        print(f"  {row['exp_id']:<6} {str(row['feature_set']):<28} "
              f"{str(row['scaler']):<10} {pca_str:<6} "
              f"{int(row['n_components']):<8} {row['output_shape']}")

    total = time.time() - t_start
    print(f"\n  ⏱  Total waktu semua eksperimen: {total/60:.1f} menit")
    print(f"  📂 Output di: {OUTPUTS_PREPROC}")
    print(f"  → Lanjut ke: clustering & evaluasi\n")


if __name__ == "__main__":
    main()
