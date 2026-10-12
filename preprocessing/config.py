"""
config.py — Konfigurasi terpusat untuk seluruh pipeline
Semua path dan parameter preprocessing didefinisikan di sini.
"""

from pathlib import Path

# ── Root project (satu level di atas folder preprocessing/) ──────────────────
ROOT = Path(__file__).resolve().parent.parent

# ── Input paths ───────────────────────────────────────────────────────────────
DATASET_PATH = ROOT / "libraries" / "dataset"
PROFILE_PATH = ROOT / "libraries" / "profile.txt"

# ── Output paths ──────────────────────────────────────────────────────────────
OUTPUTS_DATA      = ROOT / "outputs" / "data"
OUTPUTS_FEATURES  = ROOT / "outputs" / "data" / "features"
OUTPUTS_PREPROC   = ROOT / "outputs" / "data" / "preprocessed"
OUTPUTS_REPORTS   = ROOT / "outputs" / "reports"
PLOTS_EDA         = ROOT / "outputs" / "plots" / "eda"
PLOTS_FEAT        = ROOT / "outputs" / "plots" / "features"
PLOTS_PCA         = ROOT / "outputs" / "plots" / "pca"
PLOTS_COMPARISON  = ROOT / "outputs" / "plots" / "comparison"

# ── Sensor metadata ───────────────────────────────────────────────────────────
SENSOR_INFO = {
    # name : (sampling_hz, n_cols_per_cycle)
    "PS1":  (100, 6000), "PS2":  (100, 6000), "PS3":  (100, 6000),
    "PS4":  (100, 6000), "PS5":  (100, 6000), "PS6":  (100, 6000),
    "EPS1": (100, 6000),
    "FS1":  (10,  600),  "FS2":  (10,  600),
    "TS1":  (1,   60),   "TS2":  (1,   60),
    "TS3":  (1,   60),   "TS4":  (1,   60),
    "VS1":  (1,   60),
    "CE":   (1,   60),   "CP":   (1,   60),   "SE":   (1,   60),
}

SENSORS_100HZ = [s for s, (hz, _) in SENSOR_INFO.items() if hz == 100]
SENSORS_10HZ  = [s for s, (hz, _) in SENSOR_INFO.items() if hz == 10]
SENSORS_1HZ   = [s for s, (hz, _) in SENSOR_INFO.items() if hz == 1]

N_CYCLES = 2205

PROFILE_COLS = [
    "cooler_condition",
    "valve_condition",
    "pump_leakage",
    "accumulator",
    "stable_flag",
]

# ── Preprocessing parameters ──────────────────────────────────────────────────
OUTLIER_LOWER_PCTILE = 1    # percentile bawah untuk outlier capping
OUTLIER_UPPER_PCTILE = 99   # percentile atas

NZV_THRESHOLD      = 1e-10  # near-zero variance threshold
CORR_THRESHOLD     = 0.95   # hapus fitur dengan korelasi > nilai ini
SEED               = 42

# PCA defaults (dioverride per-EXP oleh EXPERIMENT_CONFIGS)
PCA_EVR_THRESHOLD  = 0.90
PCA_MIN_COMPONENTS = 10

# ── Feature extraction parameters ─────────────────────────────────────────────
ACF_LAGS        = (1, 10)   # lag untuk fitur autokorelasi
FFT_LOW_BAND    = (0, 10)   # Hz — band frekuensi rendah
FFT_HIGH_BAND   = (10, 50)  # Hz — band frekuensi tinggi
N_CLUSTERS_PS1  = 13        # jumlah sub-interval temporal berbasis PS1 (Paper 3)
EWT_N_SSC       = 3         # jumlah Sub-Signal Components dari EWT (Paper 4)

# ── Plot style ────────────────────────────────────────────────────────────────
PLOT_DPI = 150

# ── Experiment Configurations ─────────────────────────────────────────────────
# Setiap eksperimen mendefinisikan:
#   feature_set   : kunci ke fungsi ekstraksi fitur
#   outlier_cap   : apakah capping P1-P99 diterapkan
#   scaler        : 'robust' | 'standard' | 'minmax'
#   pca_threshold : float (0-1) atau None (tanpa PCA)
#   papers        : referensi paper pendukung
EXPERIMENT_CONFIGS = {
    "EXP_A": {
        "label":         "Baseline — Stats Dasar",
        "feature_set":   "stats",
        "outlier_cap":   True,
        "scaler":        "robust",
        "pca_threshold": 0.95,
        "papers":        ["Shaikh2025", "Silva2025"],
        "description":   "15 fitur statistik × 17 sensor, RobustScaler, PCA 95%",
    },
    "EXP_B": {
        "label":         "Stats + FFT Spectral",
        "feature_set":   "stats_fft",
        "outlier_cap":   True,
        "scaler":        "robust",
        "pca_threshold": 0.95,
        "papers":        ["Shaikh2025", "Chen2023"],
        "description":   "Stats + fitur spektral FFT untuk sensor 100Hz/10Hz, RobustScaler, PCA 95%",
    },
    "EXP_C": {
        "label":         "Stats + FFT + 13-Cluster PS1",
        "feature_set":   "stats_fft_13cluster",
        "outlier_cap":   True,
        "scaler":        "robust",
        "pca_threshold": 0.95,
        "papers":        ["Kim2022"],
        "description":   "13 sub-interval temporal berbasis PS1, stats per cluster, RobustScaler, PCA 95%",
    },
    "EXP_D": {
        "label":         "Stats + FFT — Z-score",
        "feature_set":   "stats_fft",
        "outlier_cap":   True,
        "scaler":        "standard",
        "pca_threshold": 0.95,
        "papers":        ["Wang2025", "Kurniawan2026"],
        "description":   "Identik EXP-B tapi StandardScaler (Z-score), PCA 95%",
    },
    "EXP_E": {
        "label":         "Stats + FFT — Min-Max",
        "feature_set":   "stats_fft",
        "outlier_cap":   True,
        "scaler":        "minmax",
        "pca_threshold": 0.95,
        "papers":        ["Silva2025"],
        "description":   "Identik EXP-B tapi MinMaxScaler [0,1], PCA 95%",
    },
    "EXP_F": {
        "label":         "EWT + 17 Fitur Time-Domain",
        "feature_set":   "ewt",
        "outlier_cap":   True,
        "scaler":        "robust",
        "pca_threshold": 0.95,
        "papers":        ["Chen2023"],
        "description":   "EWT decomposition + 17 fitur per SSC untuk sensor 100Hz, RobustScaler, PCA 95%",
    },
    "EXP_G": {
        "label":         "Stats + FFT — PCA 85%",
        "feature_set":   "stats_fft",
        "outlier_cap":   True,
        "scaler":        "robust",
        "pca_threshold": 0.85,
        "papers":        ["variasi"],
        "description":   "Identik EXP-B tapi PCA lebih agresif (85% variance)",
    },
    "EXP_H": {
        "label":         "Stats + FFT — Tanpa PCA",
        "feature_set":   "stats_fft",
        "outlier_cap":   True,
        "scaler":        "robust",
        "pca_threshold": None,    # None = skip PCA
        "papers":        ["kontrol"],
        "description":   "Identik EXP-B tapi tanpa PCA (kontrol dimensi)",
    },
}
