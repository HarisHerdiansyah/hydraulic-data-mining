"""
04_normalize_pca.py — FASE 3 & 4: Normalisasi + Reduksi Dimensi PCA
====================================================================
Tugas:
  - RobustScaler (median centering + IQR normalization)
  - PCA dengan threshold 90% explained variance
  - Scree plot + cumulative EVR
  - Visualisasi 2D per kondisi label
  - Analisis PCA loadings
  - Simpan ke outputs/data/:
      data_scaled.csv            (setelah RobustScaler)
      data_pca.csv               (siap clustering)
      pca_loadings.csv           (loading per fitur)

Referensi:
  - RobustScaler : Shaikh et al. (2025)
  - PCA          : Silva & Maitelli (2025)
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    OUTPUTS_DATA, PLOTS_PCA, PLOT_DPI,
    PCA_EVR_THRESHOLD, PCA_MIN_COMPONENTS, SEED
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def print_header(title: str):
    bar = "─" * 55
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


def load_inputs():
    df_feat   = pd.read_csv(OUTPUTS_DATA / "feature_matrix.csv")
    df_label  = pd.read_csv(OUTPUTS_DATA / "feature_matrix_labeled.csv")
    profile   = df_label[["cooler_condition", "valve_condition",
                           "pump_leakage", "accumulator", "stable_flag"]]
    return df_feat, profile


def save_fig(fig: plt.Figure, filename: str):
    path = PLOTS_PCA / filename
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 Saved: {path.name}")


# ── FASE 3: Normalisasi ───────────────────────────────────────────────────────

def normalize(df: pd.DataFrame) -> tuple[np.ndarray, RobustScaler]:
    print_header("FASE 3 — Normalisasi: RobustScaler")

    # Verifikasi tidak ada NaN/Inf
    assert not df.isnull().any().any(),    "Masih ada NaN di feature matrix!"
    assert not np.isinf(df.values).any(), "Masih ada Inf di feature matrix!"

    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(df.values)

    print(f"  Input shape  : {df.shape}")
    print(f"  NaN setelah scaling : {np.isnan(X_scaled).sum()}  ✅")
    print(f"  Inf setelah scaling : {np.isinf(X_scaled).sum()}  ✅")

    return X_scaled, scaler


def plot_scaling_comparison(df_feat: pd.DataFrame,
                             X_scaled: np.ndarray,
                             feature_names: list[str]):
    """Before vs After scaling untuk 5 fitur representatif."""
    sample_feats = ["PS1_mean", "EPS1_mean", "TS1_mean", "FS1_mean", "VS1_mean"]
    # Filter hanya fitur yang ada
    sample_feats = [f for f in sample_feats if f in df_feat.columns]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for feat in sample_feats:
        axes[0].hist(df_feat[feat], bins=40, alpha=0.45, density=True, label=feat)
    axes[0].set_title("SEBELUM Normalisasi", fontweight="bold")
    axes[0].set_xlabel("Nilai")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=8)

    for i, feat in enumerate(sample_feats):
        col_idx = list(df_feat.columns).index(feat)
        axes[1].hist(X_scaled[:, col_idx], bins=40, alpha=0.45,
                     density=True, label=feat)
    axes[1].set_title("SESUDAH RobustScaler", fontweight="bold")
    axes[1].set_xlabel("Nilai Ternormalisasi")
    axes[1].set_ylabel("Density")
    axes[1].legend(fontsize=8)

    fig.suptitle(
        "Efek RobustScaler — Distribusi Fitur\n"
        "(Referensi: Shaikh et al., 2025)",
        fontweight="bold"
    )
    plt.tight_layout()
    save_fig(fig, "01_scaling_comparison.png")


# ── FASE 4: PCA ───────────────────────────────────────────────────────────────

def run_pca(X_scaled: np.ndarray,
            threshold: float = PCA_EVR_THRESHOLD,
            min_components: int = PCA_MIN_COMPONENTS
            ) -> tuple[np.ndarray, PCA, int]:

    print_header(f"FASE 4 — PCA  (threshold: {int(threshold*100)}% EVR, min: {min_components} komponen)")

    # Verifikasi final sebelum PCA
    assert not np.isnan(X_scaled).any(), "NaN ditemukan sebelum PCA!"
    assert not np.isinf(X_scaled).any(), "Inf ditemukan sebelum PCA!"

    pca_full = PCA(random_state=SEED)
    pca_full.fit(X_scaled)

    evr        = pca_full.explained_variance_ratio_
    cumul_evr  = np.cumsum(evr)
    k_by_evr   = int(np.argmax(cumul_evr >= threshold)) + 1
    k_selected = max(k_by_evr, min_components)   # pastikan minimal min_components

    print(f"  Total fitur original : {X_scaled.shape[1]}")
    print(f"  Komponen untuk  85%  : {int(np.argmax(cumul_evr >= 0.85)) + 1}")
    print(f"  Komponen untuk  90%  : {k_by_evr}")
    print(f"  Komponen untuk  95%  : {int(np.argmax(cumul_evr >= 0.95)) + 1}")
    if k_selected > k_by_evr:
        print(f"  ⚠️  K dari EVR ({k_by_evr}) < minimum ({min_components}) → dipakai {k_selected}")
    print(f"  Komponen dipilih     : {k_selected}  ← DIPILIH")
    print(f"  Variance dijelaskan  : {cumul_evr[k_selected-1]*100:.2f}%")
    print(f"  Rasio reduksi        : {X_scaled.shape[1]} → {k_selected} "
          f"({100*(1-k_selected/X_scaled.shape[1]):.1f}% pengurangan)")

    # Transform final
    pca_final = PCA(n_components=k_selected, random_state=SEED)
    X_pca     = pca_final.fit_transform(X_scaled)

    print(f"\n  Shape output PCA : {X_pca.shape}  ✅")
    return X_pca, pca_final, k_selected, evr, cumul_evr


def plot_scree(evr: np.ndarray, cumul_evr: np.ndarray, k: int):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n_show = min(40, len(evr))
    axes[0].bar(range(1, n_show + 1), evr[:n_show] * 100,
                color="steelblue", alpha=0.8, edgecolor="white")
    axes[0].axvline(x=k, color="crimson", linestyle="--", lw=2,
                    label=f"K={k} (90%)")
    axes[0].set_title(f"Explained Variance per Komponen (top {n_show})",
                      fontweight="bold")
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance (%)")
    axes[0].legend()

    axes[1].plot(range(1, len(cumul_evr) + 1), cumul_evr * 100,
                 color="steelblue", lw=2, marker=".")
    for thresh, col, lab in [
        (85, "orange", "85%"),
        (90, "crimson", "90% ← dipilih"),
        (95, "seagreen", "95%"),
    ]:
        axes[1].axhline(y=thresh, color=col, linestyle="--", lw=1.5, label=lab)
    axes[1].axvline(x=k, color="crimson", linestyle=":", lw=1.5)
    axes[1].set_title("Cumulative Explained Variance", fontweight="bold")
    axes[1].set_xlabel("Jumlah Komponen")
    axes[1].set_ylabel("Cumulative EVR (%)")
    axes[1].set_xlim(1, min(80, len(cumul_evr)))
    axes[1].legend()

    fig.suptitle(
        "PCA — Scree Plot & Cumulative Explained Variance\n"
        "(Referensi: Silva & Maitelli, 2025)",
        fontweight="bold"
    )
    plt.tight_layout()
    save_fig(fig, "02_scree_plot.png")


def plot_pca_2d(X_pca: np.ndarray, profile: pd.DataFrame):
    """Scatter PC1 vs PC2 diwarnai per kondisi. Skip jika K < 2."""
    if X_pca.shape[1] < 2:
        print("  ⚠️  K < 2 — visualisasi 2D dilewati. Gunakan PCA_MIN_COMPONENTS >= 2.")
        return

    conditions = [
        ("cooler_condition", "viridis"),
        ("valve_condition",  "plasma"),
        ("pump_leakage",     "coolwarm"),
        ("accumulator",      "RdYlGn"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, (cond, cmap) in zip(axes.flatten(), conditions):
        sc = ax.scatter(
            X_pca[:, 0], X_pca[:, 1],
            c=profile[cond].values, cmap=cmap,
            alpha=0.6, s=20, edgecolors="none"
        )
        plt.colorbar(sc, ax=ax, shrink=0.85)
        ax.set_title(cond.replace("_", " ").title(), fontweight="bold")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    fig.suptitle(
        "PCA 2D — Struktur Awal Cluster per Kondisi Komponen\n"
        "(Label hanya untuk validasi eksplorasi)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    save_fig(fig, "03_pca_2d_by_condition.png")


def plot_loadings(pca: PCA, feature_names: list[str]):
    """Top 15 loading untuk PC1 dan PC2."""
    loadings = pd.DataFrame(
        pca.components_[:2].T,
        index=feature_names,
        columns=["PC1", "PC2"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, pc in zip(axes, ["PC1", "PC2"]):
        top   = loadings[pc].abs().nlargest(15)
        vals  = loadings.loc[top.index[::-1], pc]
        colors = ["crimson" if v < 0 else "steelblue" for v in vals]
        ax.barh(top.index[::-1], vals, color=colors, alpha=0.8)
        ax.axvline(x=0, color="black", lw=0.8)
        ax.set_title(f"Top 15 Loading — {pc}", fontweight="bold")
        ax.set_xlabel("Loading Value")

    fig.suptitle(
        "PCA Loadings — Fitur Paling Berkontribusi\n"
        "(Referensi: Shaikh et al., 2025)",
        fontweight="bold"
    )
    plt.tight_layout()
    save_fig(fig, "04_pca_loadings.png")

    # Simpan loading lengkap
    loadings.to_csv(OUTPUTS_DATA / "pca_loadings.csv")
    print(f"\n  Top 10 kontribusi ke PC1:")
    top10 = loadings["PC1"].abs().nlargest(10)
    for feat, val in top10.items():
        direction = "+" if loadings.loc[feat, "PC1"] >= 0 else "-"
        print(f"    {direction}{val:.4f}  {feat}")

    return loadings


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    PLOTS_PCA.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("\n" + "=" * 55)
    print("  04_normalize_pca.py — Normalisasi + PCA")
    print("=" * 55)

    df_feat, profile = load_inputs()
    feature_names    = list(df_feat.columns)

    # ── FASE 3: RobustScaler ─────────────────────────────────────────────
    X_scaled, scaler = normalize(df_feat)
    plot_scaling_comparison(df_feat, X_scaled, feature_names)

    # Simpan data ternormalisasi
    df_scaled = pd.DataFrame(X_scaled, columns=feature_names)
    df_scaled.to_csv(OUTPUTS_DATA / "data_scaled.csv", index=False)
    print(f"  data_scaled.csv  → {df_scaled.shape}")

    # ── FASE 4: PCA ──────────────────────────────────────────────────────
    X_pca, pca_final, k, evr, cumul_evr = run_pca(
        X_scaled, threshold=PCA_EVR_THRESHOLD
    )

    plot_scree(evr, cumul_evr, k)
    plot_pca_2d(X_pca, profile)
    plot_loadings(pca_final, feature_names)

    # Simpan data PCA (output akhir untuk clustering)
    pc_cols = [f"PC{i+1}" for i in range(k)]
    df_pca  = pd.DataFrame(X_pca, columns=pc_cols)
    df_pca.to_csv(OUTPUTS_DATA / "data_pca.csv", index=False)

    # Simpan juga versi dengan label
    df_pca_labeled = pd.concat(
        [df_pca.reset_index(drop=True), profile.reset_index(drop=True)], axis=1
    )
    df_pca_labeled.to_csv(OUTPUTS_DATA / "data_pca_labeled.csv", index=False)

    elapsed = time.time() - t0

    print_header("Ringkasan Pipeline Lengkap")
    print(f"  Raw input         : 2205 × 43.680 atribut")
    print(f"  Feature matrix    : 2205 × {len(feature_names)} fitur")
    print(f"  Setelah scaling   : 2205 × {df_scaled.shape[1]}")
    print(f"  Setelah PCA (90%) : 2205 × {k}  ← SIAP CLUSTERING")
    print(f"\n  ⏱  Selesai dalam {elapsed:.2f} detik")
    print(f"\n  📂 Output disimpan di: outputs/data/")
    print(f"     data_pca.csv       → input model clustering")
    print(f"     data_pca_labeled.csv → dengan label (untuk validasi)")
    print(f"\n  ✅ Pipeline preprocessing selesai!\n")


if __name__ == "__main__":
    main()
