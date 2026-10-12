"""
02_eda.py — FASE 1B: Exploratory Data Analysis
===============================================
Tugas:
  - Statistik deskriptif per sensor
  - Histogram + KDE per sensor
  - Visualisasi time series dalam siklus (sampel)
  - Correlation matrix + hierarchical clustering dendrogram
  - Boxplot distribusi per kondisi label
  - Deteksi outlier + analisis stable flag

Semua plot disimpan ke outputs/plots/eda/

Referensi: Silva & Maitelli (2025) — EDA sebagai langkah awal kritis
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    SENSOR_INFO, SENSORS_100HZ, SENSORS_1HZ, PROFILE_COLS,
    OUTPUTS_DATA, PLOTS_EDA, PLOT_DPI
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def print_header(title: str):
    bar = "─" * 55
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


def load_inputs():
    data = np.load(OUTPUTS_DATA / "sensors.npz")
    sensors = {name: data[name] for name in data.files}
    profile = pd.read_csv(OUTPUTS_DATA / "profile.csv")
    return sensors, profile


def save_fig(fig: plt.Figure, filename: str):
    path = PLOTS_EDA / filename
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 Saved: {path.name}")


# ── 1. Statistik Deskriptif ────────────────────────────────────────────────────

def descriptive_stats(sensors: dict) -> pd.DataFrame:
    print_header("1. Statistik Deskriptif per Sensor")
    rows = []
    for name, arr in sensors.items():
        vals = arr.mean(axis=1)   # mean per siklus
        q1, q3 = np.percentile(vals, [25, 75])
        rows.append({
            "Sensor":  name,
            "Hz":      SENSOR_INFO[name][0],
            "Mean":    round(vals.mean(), 4),
            "Std":     round(vals.std(), 4),
            "Min":     round(vals.min(), 4),
            "Q1":      round(q1, 4),
            "Median":  round(np.median(vals), 4),
            "Q3":      round(q3, 4),
            "Max":     round(vals.max(), 4),
            "IQR":     round(q3 - q1, 4),
            "Range":   round(vals.max() - vals.min(), 4),
        })
    df = pd.DataFrame(rows).set_index("Sensor")

    # Tampilkan dan simpan ke CSV
    print(df.to_string())
    df.to_csv(OUTPUTS_DATA / "eda_descriptive_stats.csv")

    print(f"\n  Sensor dengan range terlebar (top 5):")
    top5 = df["Range"].sort_values(ascending=False).head(5)
    for s, v in top5.items():
        print(f"    {s:<6} → {v}")

    return df


# ── 2. Histogram + KDE ────────────────────────────────────────────────────────

def plot_histogram_kde(sensors: dict):
    print_header("2. Histogram + KDE per Sensor")
    cycle_means = {name: sensors[name].mean(axis=1) for name in sensors}

    fig, axes = plt.subplots(4, 5, figsize=(22, 15))
    axes = axes.flatten()

    for i, (name, vals) in enumerate(cycle_means.items()):
        ax = axes[i]
        ax.hist(vals, bins=40, density=True, alpha=0.45,
                color="steelblue", edgecolor="white")
        kde_x = np.linspace(vals.min(), vals.max(), 300)
        kde   = stats.gaussian_kde(vals)
        ax.plot(kde_x, kde(kde_x), color="crimson", lw=2)
        ax.set_title(f"{name}  ({SENSOR_INFO[name][0]} Hz)", fontweight="bold")
        ax.set_xlabel("Mean per Siklus")
        ax.set_ylabel("Density")

    for j in range(len(sensors), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Distribusi Mean per Siklus — Histogram + KDE\n"
        "(Referensi: Silva & Maitelli, 2025)",
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    save_fig(fig, "01_histogram_kde.png")


# ── 3. Time Series dalam Siklus ───────────────────────────────────────────────

def plot_time_series_samples(sensors: dict):
    print_header("3. Time Series dalam Siklus (Sampel)")
    sample_cycles = [0, 700, 1800]
    colors_s = ["steelblue", "darkorange", "seagreen"]
    labels_s = [f"Siklus {c}" for c in sample_cycles]

    # Sensor 100 Hz
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    for ax, sname in zip(axes, ["PS1", "EPS1"]):
        t = np.arange(6000) / 100
        for cyc, col, lab in zip(sample_cycles, colors_s, labels_s):
            ax.plot(t, sensors[sname][cyc], color=col,
                    alpha=0.8, lw=0.7, label=lab)
        ax.set_title(f"{sname} (100 Hz)")
        ax.set_xlabel("Waktu (detik)")
        ax.set_ylabel("Nilai")
        ax.legend()
    fig.suptitle("Time Series dalam Siklus — Sensor 100 Hz",
                 fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "02a_timeseries_100hz.png")

    # Sensor 1 Hz
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    for ax, sname in zip(axes, ["TS1", "VS1"]):
        t = np.arange(60)
        for cyc, col, lab in zip(sample_cycles, colors_s, labels_s):
            ax.plot(t, sensors[sname][cyc], color=col, alpha=0.8,
                    lw=1.5, marker="o", ms=3, label=lab)
        ax.set_title(f"{sname} (1 Hz)")
        ax.set_xlabel("Waktu (detik)")
        ax.set_ylabel("Nilai")
        ax.legend()
    fig.suptitle("Time Series dalam Siklus — Sensor 1 Hz",
                 fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "02b_timeseries_1hz.png")


# ── 4. Correlation Matrix + Hierarchical Clustering ──────────────────────────

def plot_correlation(sensors: dict) -> pd.DataFrame:
    print_header("4. Correlation Matrix + Hierarchical Clustering")

    mean_matrix = pd.DataFrame(
        {name: sensors[name].mean(axis=1) for name in sensors}
    )
    corr = mean_matrix.corr()

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Heatmap
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0,
        linewidths=0.5, ax=axes[0], cbar_kws={"shrink": 0.8}
    )
    axes[0].set_title("Correlation Matrix Antar Sensor",
                      fontweight="bold")

    # Dendrogram
    dist_arr = (1 - corr.abs()).values.copy()
    np.fill_diagonal(dist_arr, 0)
    lnk = linkage(squareform(np.clip(dist_arr, 0, None)), method="ward")
    dendrogram(
        lnk, labels=corr.columns.tolist(),
        ax=axes[1], orientation="right",
        leaf_font_size=11, color_threshold=0.4
    )
    axes[1].set_title("Hierarchical Clustering Sensor\n(berdasarkan Korelasi)",
                      fontweight="bold")
    axes[1].set_xlabel("Distance  (1 − |r|)")

    fig.suptitle(
        "Korelasi Antar Sensor + Hierarchical Clustering\n"
        "(Referensi: Silva & Maitelli, 2025)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    save_fig(fig, "03_correlation_hclust.png")

    # Laporan pasangan sangat berkorelasi
    pairs = [
        (corr.columns[i], corr.columns[j], round(corr.iloc[i, j], 4))
        for i in range(len(corr.columns))
        for j in range(i + 1, len(corr.columns))
        if abs(corr.iloc[i, j]) > 0.90
    ]
    if pairs:
        df_pairs = pd.DataFrame(pairs, columns=["Sensor A", "Sensor B", "r"])
        print("\n  Pasangan dengan |r| > 0.90 (kandidat redundan):")
        print(df_pairs.to_string(index=False))
        df_pairs.to_csv(OUTPUTS_DATA / "eda_high_corr_pairs.csv", index=False)
    else:
        print("\n  Tidak ada pasangan sensor dengan |r| > 0.90")

    corr.to_csv(OUTPUTS_DATA / "eda_correlation_matrix.csv")
    return corr


# ── 5. Boxplot per Kondisi Label ──────────────────────────────────────────────

def plot_boxplots_by_condition(sensors: dict, profile: pd.DataFrame):
    print_header("5. Distribusi Sensor per Kondisi Label")

    sensor_rep = {
        "cooler_condition": ["PS3", "TS1", "CE"],
        "valve_condition":  ["PS1", "FS1", "EPS1"],
        "pump_leakage":     ["PS2", "FS2", "VS1"],
        "accumulator":      ["PS4", "PS5", "SE"],
    }

    # Mapping keterangan teks sesuai isi file profile.txt
    label_map = {
        "cooler_condition": {3: "Failure\n(3%)", 20: "Reduced\n(20%)", 100: "Full\n(100%)"},
        "valve_condition":  {73: "Failure\n(73%)", 80: "Severe\n(80%)", 90: "Small Lag\n(90%)", 100: "Optimal\n(100%)"},
        "pump_leakage":     {0: "None\n(0)", 1: "Weak\n(1)", 2: "Severe\n(2)"},
        "accumulator":      {90: "Failure\n(90 bar)", 100: "Severe\n(100 bar)", 115: "Slight\n(115 bar)", 130: "Optimal\n(130 bar)"},
    }

    for cond, slist in sensor_rep.items():
        unique_vals = sorted(profile[cond].unique())
        fig, axes = plt.subplots(1, len(slist), figsize=(6 * len(slist), 5))

        for ax, sname in zip(axes, slist):
            data_plot   = [
                sensors[sname][profile[cond].values == v].mean(axis=1)
                for v in unique_vals
            ]
            tick_labels = [label_map[cond].get(v, str(v)) for v in unique_vals]
            bp = ax.boxplot(
                data_plot, tick_labels=tick_labels, patch_artist=True,
                boxprops=dict(facecolor="steelblue", alpha=0.7),
                medianprops=dict(color="crimson", lw=2)
            )
            ax.set_title(sname, fontweight="bold")
            ax.set_xlabel(cond.replace("_", " ").title())
            ax.set_ylabel("Mean per Siklus")

        fig.suptitle(
            f"Distribusi Sensor — {cond.replace('_', ' ').title()}\n"
            "(Label digunakan HANYA untuk validasi eksplorasi)",
            fontweight="bold"
        )
        plt.tight_layout()
        save_fig(fig, f"04_boxplot_{cond}.png")


# ── 6. Outlier + Stable Flag ──────────────────────────────────────────────────

def plot_outlier_stable_flag(sensors: dict, profile: pd.DataFrame):
    print_header("6. Deteksi Outlier + Pengaruh Stable Flag")

    rows = []
    for name, arr in sensors.items():
        vals = arr.mean(axis=1)
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        n_out = int(((vals < q1 - 1.5 * iqr) | (vals > q3 + 1.5 * iqr)).sum())
        rows.append({
            "Sensor": name,
            "N_outlier_siklus": n_out,
            "Persen": round(100 * n_out / len(vals), 2),
        })

    df_out = (
        pd.DataFrame(rows)
        .set_index("Sensor")
        .sort_values("Persen", ascending=False)
    )
    print(df_out.to_string())
    df_out.to_csv(OUTPUTS_DATA / "eda_outlier_summary.csv")

    stable_mask = profile["stable_flag"].values == 1
    print(f"\n  Siklus stable_flag=1 : {stable_mask.sum()} (dipertahankan)")
    print(f"  Siklus stable_flag=0 : {(~stable_mask).sum()}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, sname in zip(axes, ["PS1", "TS1"]):
        ax.hist(sensors[sname][~stable_mask].mean(axis=1),
                bins=30, alpha=0.6, label="stable=0", color="steelblue")
        ax.hist(sensors[sname][stable_mask].mean(axis=1),
                bins=30, alpha=0.6, label="stable=1", color="crimson")
        ax.set_title(sname, fontweight="bold")
        ax.set_xlabel("Mean per Siklus")
        ax.set_ylabel("Count")
        ax.legend()

    fig.suptitle("Pengaruh Stable Flag terhadap Distribusi Sensor",
                 fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "05_stable_flag.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    PLOTS_EDA.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("\n" + "=" * 55)
    print("  02_eda.py — Exploratory Data Analysis")
    print("=" * 55)

    sensors, profile = load_inputs()

    descriptive_stats(sensors)
    plot_histogram_kde(sensors)
    plot_time_series_samples(sensors)
    plot_correlation(sensors)
    plot_boxplots_by_condition(sensors, profile)
    plot_outlier_stable_flag(sensors, profile)

    elapsed = time.time() - t0
    print(f"\n  ⏱  EDA selesai dalam {elapsed:.2f} detik")
    print(f"  📂 Plot disimpan di: outputs/plots/eda/")
    print(f"  → Lanjut ke: python 03_feature_extraction.py\n")


if __name__ == "__main__":
    main()
