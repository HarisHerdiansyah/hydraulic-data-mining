"""
03_feature_extraction.py — FASE 2: Multi Feature Set Extraction
================================================================
Menghasilkan 4 CSV feature matrix untuk 8 eksperimen:

  features_stats.csv          → EXP-A  (255 fitur)
  features_stats_fft.csv      → EXP-B, D, E, G, H  (~294 fitur)
  features_stats_fft_13cl.csv → EXP-C  (~600 fitur)
  features_ewt.csv            → EXP-F  (~507 fitur)

Referensi:
  - Stats + Trend + FFT : Shaikh et al. (2025) — Paper 1
  - 13 cluster temporal : Kim & Heo (2022) — Paper 3 [ZeMA identik]
  - EWT + 17 fitur      : Chen et al. (2023) — Paper 4
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.fft import fft, rfft, rfftfreq, fftfreq
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    SENSOR_INFO, SENSORS_100HZ, SENSORS_10HZ, SENSORS_1HZ,
    N_CYCLES, OUTPUTS_DATA, OUTPUTS_FEATURES, PLOTS_FEAT, PLOT_DPI,
    ACF_LAGS, FFT_LOW_BAND, FFT_HIGH_BAND,
    NZV_THRESHOLD, N_CLUSTERS_PS1, EWT_N_SSC, SEED,
)


# ══════════════════════════════════════════════════════════════════
# UTILS
# ══════════════════════════════════════════════════════════════════

def print_header(title: str):
    bar = "─" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def load_sensors() -> dict:
    data = np.load(OUTPUTS_DATA / "sensors.npz")
    return {name: data[name] for name in data.files}


def safe_skew(arr_2d: np.ndarray) -> np.ndarray:
    return np.nan_to_num(sp_stats.skew(arr_2d, axis=1), nan=0.0)


def safe_kurtosis(arr_2d: np.ndarray) -> np.ndarray:
    return np.nan_to_num(sp_stats.kurtosis(arr_2d, axis=1), nan=0.0)


def nan_filter(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Median imputation + NZV filter."""
    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        print(f"  ⚠ {label}: {nan_count} NaN ditemukan → median imputation")
        imp = SimpleImputer(strategy="median")
        df = pd.DataFrame(imp.fit_transform(df.values), columns=df.columns)

    nzv = df.var()[df.var() < NZV_THRESHOLD].index.tolist()
    if nzv:
        print(f"  ⚠ {label}: {len(nzv)} fitur NZV dihapus")
        df = df.drop(columns=nzv)

    print(f"  ✅ {label}: shape={df.shape}, NaN={df.isnull().sum().sum()}")
    return df


# ══════════════════════════════════════════════════════════════════
# FEATURE SET A — STATS DASAR (15 fitur × 17 sensor = 255)
# ══════════════════════════════════════════════════════════════════

def extract_stats(arr: np.ndarray, name: str) -> dict:
    """
    15 fitur statistik per sensor per siklus.
    Referensi: Kim & Heo (2022), Shaikh (2025), Chen (2023)
    """
    q1  = np.percentile(arr, 25, axis=1)
    q3  = np.percentile(arr, 75, axis=1)
    n_t = arr.shape[1]
    t   = np.arange(n_t, dtype=np.float64)
    t_c = t - t.mean()
    t_var = (t_c ** 2).sum()
    arr_c  = arr - arr.mean(axis=1, keepdims=True)
    slopes = np.where(
        arr.std(axis=1) == 0, 0.0,
        np.nan_to_num((arr_c * t_c).sum(axis=1) / t_var, nan=0.0)
    )
    argmax_norm = np.argmax(arr, axis=1) / n_t
    argmin_norm = np.argmin(arr, axis=1) / n_t

    return {
        f"{name}_mean":     arr.mean(axis=1),
        f"{name}_std":      arr.std(axis=1),
        f"{name}_min":      arr.min(axis=1),
        f"{name}_max":      arr.max(axis=1),
        f"{name}_median":   np.median(arr, axis=1),
        f"{name}_q1":       q1,
        f"{name}_q3":       q3,
        f"{name}_iqr":      q3 - q1,
        f"{name}_var":      arr.var(axis=1),
        f"{name}_skew":     safe_skew(arr),
        f"{name}_kurt":     safe_kurtosis(arr),
        f"{name}_ptp":      arr.max(axis=1) - arr.min(axis=1),     # peak-to-peak
        f"{name}_rms":      np.sqrt((arr ** 2).mean(axis=1)),
        f"{name}_slope":    slopes,
        f"{name}_argmax":   argmax_norm,
        # argmin tidak dimasukkan agar simetris (15 fitur genap)
    }


def build_feature_set_stats(sensors: dict) -> pd.DataFrame:
    print_header("Feature Set A — Stats Dasar (15 × 17 = 255 fitur)")
    feats = {}
    for name, arr in sensors.items():
        feats.update(extract_stats(arr, name))
        print(f"    {name}: 15 fitur ✓")
    df = pd.DataFrame(feats)
    print(f"  Raw shape: {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════
# FEATURE SET B — STATS + FFT SPECTRAL
# ══════════════════════════════════════════════════════════════════

def extract_fft_spectral(arr: np.ndarray, name: str, fs: int) -> dict:
    """
    5 fitur spektral FFT per sensor.
    Sensor 100 Hz: 5 fitur (dominant freq, total energy, band low, band high, spectral entropy)
    Sensor 10 Hz : 2 fitur (dominant freq, total energy)
    Referensi: Shaikh (2025) — Paper 1, Chen (2023) — Paper 4
    """
    n_cycles, n_t = arr.shape
    freqs_pos = fftfreq(n_t, d=1.0 / fs)[: n_t // 2]
    low_mask  = (freqs_pos >= FFT_LOW_BAND[0])  & (freqs_pos <= FFT_LOW_BAND[1])
    high_mask = (freqs_pos >  FFT_HIGH_BAND[0]) & (freqs_pos <= FFT_HIGH_BAND[1])

    dom_freq    = np.zeros(n_cycles)
    spec_energy = np.zeros(n_cycles)
    e_low       = np.zeros(n_cycles)
    e_high      = np.zeros(n_cycles)
    spec_ent    = np.zeros(n_cycles)

    for i in range(n_cycles):
        mag = np.abs(fft(arr[i]))[: n_t // 2]
        dom_freq[i]    = freqs_pos[np.argmax(mag)]
        spec_energy[i] = (mag ** 2).sum()

        if fs >= 100:
            e_low[i]    = (mag[low_mask]  ** 2).sum()
            e_high[i]   = (mag[high_mask] ** 2).sum()
            # Spectral entropy (Paper 4: ENT)
            psd = mag ** 2
            psd_norm = psd / (psd.sum() + 1e-12)
            spec_ent[i] = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))

    result = {
        f"{name}_fft_dom_freq":    np.nan_to_num(dom_freq),
        f"{name}_fft_energy":      np.nan_to_num(spec_energy),
    }
    if fs >= 100:
        result[f"{name}_fft_e_low"]  = np.nan_to_num(e_low)
        result[f"{name}_fft_e_high"] = np.nan_to_num(e_high)
        result[f"{name}_fft_entropy"] = np.nan_to_num(spec_ent)

    return result


def build_feature_set_stats_fft(sensors: dict) -> pd.DataFrame:
    print_header("Feature Set B — Stats + FFT Spectral")

    # Stats (semua sensor)
    feats = {}
    for name, arr in sensors.items():
        feats.update(extract_stats(arr, name))

    # FFT (100 Hz dan 10 Hz saja)
    print("  FFT sensor 100 Hz:")
    for name in SENSORS_100HZ:
        fs = SENSOR_INFO[name][0]
        print(f"    {name} ...", end=" ", flush=True)
        feats.update(extract_fft_spectral(sensors[name], name, fs))
        print("✓")

    print("  FFT sensor 10 Hz:")
    for name in SENSORS_10HZ:
        fs = SENSOR_INFO[name][0]
        print(f"    {name} ...", end=" ", flush=True)
        feats.update(extract_fft_spectral(sensors[name], name, fs))
        print("✓")

    df = pd.DataFrame(feats)
    n_stats = 15 * len(sensors)
    n_fft   = 5 * len(SENSORS_100HZ) + 2 * len(SENSORS_10HZ)
    print(f"\n  Stats fitur : {n_stats}")
    print(f"  FFT fitur   : {n_fft}")
    print(f"  Raw shape   : {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════
# FEATURE SET C — STATS + FFT + 13-CLUSTER TEMPORAL PS1
# ══════════════════════════════════════════════════════════════════

def detect_phase_boundaries(template: np.ndarray, n_phases: int = 13) -> list:
    """
    Deteksi n_phases sub-interval temporal dari sinyal template PS1.
    Strategi: cari local minima/maxima pada template; jika tidak cukup,
    fallback ke equal-interval division.
    Referensi: Kim & Heo (2022) — Paper 3
    """
    n = len(template)

    # Cari local extrema (peaks dan valleys)
    peaks,   _ = find_peaks(template,  prominence=template.std() * 0.3)
    valleys, _ = find_peaks(-template, prominence=template.std() * 0.3)

    boundaries = sorted(set([0] + list(peaks) + list(valleys) + [n - 1]))

    if len(boundaries) - 1 >= n_phases:
        # Pilih n_phases boundary yang paling merata
        step = (len(boundaries) - 1) / n_phases
        selected = [boundaries[int(round(i * step))] for i in range(n_phases + 1)]
        selected[-1] = n - 1
        return sorted(set(selected))
    else:
        # Fallback: equal-interval
        print(f"    ⚠ Hanya {len(boundaries)-1} extrema ditemukan → equal-interval fallback")
        return [int(round(i * n / n_phases)) for i in range(n_phases + 1)]


def extract_cluster_stats(arr_slice: np.ndarray, name: str, cl_id: int,
                           hz: int) -> dict:
    """
    Ekstraksi fitur per sub-interval (cluster temporal).
    - Distribusi: mean, std, min, max, q1, q3, var  (selalu)
    - Shape: slope, ptp                              (selalu)
    - Skew/Kurt: hanya jika hz >= 10 (cukup data point)
    Referensi: Kim & Heo (2022) — Paper 3
    """
    prefix = f"{name}_cl{cl_id:02d}"
    n_t = arr_slice.shape[1]

    result = {
        f"{prefix}_mean": arr_slice.mean(axis=1),
        f"{prefix}_std":  arr_slice.std(axis=1),
        f"{prefix}_min":  arr_slice.min(axis=1),
        f"{prefix}_max":  arr_slice.max(axis=1),
        f"{prefix}_q1":   np.percentile(arr_slice, 25, axis=1),
        f"{prefix}_q3":   np.percentile(arr_slice, 75, axis=1),
        f"{prefix}_var":  arr_slice.var(axis=1),
        f"{prefix}_ptp":  arr_slice.max(axis=1) - arr_slice.min(axis=1),
    }

    # Slope
    if n_t >= 2:
        t   = np.arange(n_t, dtype=np.float64)
        t_c = t - t.mean()
        t_var = (t_c ** 2).sum() + 1e-12
        arr_c  = arr_slice - arr_slice.mean(axis=1, keepdims=True)
        result[f"{prefix}_slope"] = np.nan_to_num(
            (arr_c * t_c).sum(axis=1) / t_var, nan=0.0
        )

    # Skewness & kurtosis hanya untuk sensor ≥ 10 Hz (cukup data point per cluster)
    if hz >= 10:
        result[f"{prefix}_skew"] = safe_skew(arr_slice)
        result[f"{prefix}_kurt"] = safe_kurtosis(arr_slice)

    return result


def build_feature_set_stats_fft_13cluster(sensors: dict) -> pd.DataFrame:
    print_header("Feature Set C — Stats + FFT + 13-Cluster Temporal PS1")

    # Deteksi boundaries dari rata-rata PS1
    template_ps1 = sensors["PS1"].mean(axis=0)
    boundaries = detect_phase_boundaries(template_ps1, n_phases=N_CLUSTERS_PS1)
    n_actual = len(boundaries) - 1
    print(f"  Jumlah sub-interval terdeteksi: {n_actual}")
    print(f"  Boundaries (index): {boundaries[:5]} ... {boundaries[-3:]}")

    feats = {}

    # --- Per-cluster stats untuk semua sensor ---
    for name, arr in sensors.items():
        hz  = SENSOR_INFO[name][0]
        n_t = arr.shape[1]

        for cl_id, (b_start, b_end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            # Sesuaikan boundary ke resolusi sensor (100 Hz→6000, 10 Hz→600, 1 Hz→60)
            scale = n_t / 6000  # PS1 punya 6000 points; scale ke sensor ini
            s = max(0,    int(round(b_start * scale)))
            e = min(n_t,  int(round(b_end   * scale)))
            if e <= s:
                e = s + 1

            arr_slice = arr[:, s:e]
            feats.update(extract_cluster_stats(arr_slice, name, cl_id, hz))

        print(f"    {name}: {n_actual} cluster ✓")

    # --- FFT global (sama seperti Feature Set B) ---
    print("\n  FFT global (100 Hz + 10 Hz):")
    for name in SENSORS_100HZ + SENSORS_10HZ:
        fs = SENSOR_INFO[name][0]
        print(f"    {name} ...", end=" ", flush=True)
        feats.update(extract_fft_spectral(sensors[name], name, fs))
        print("✓")

    df = pd.DataFrame(feats)
    print(f"\n  Raw shape: {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════
# FEATURE SET F — EWT + 17 FITUR TIME-DOMAIN
# ══════════════════════════════════════════════════════════════════

def bandpass_filter(x: np.ndarray, low: float, high: float,
                    fs: int, order: int = 4) -> np.ndarray:
    """Butterworth bandpass filter."""
    nyq = fs / 2.0
    low_n  = max(low  / nyq, 1e-4)
    high_n = min(high / nyq, 1 - 1e-4)
    if low_n >= high_n:
        return x
    b, a = butter(order, [low_n, high_n], btype="band")
    return filtfilt(b, a, x)


def lowpass_filter(x: np.ndarray, cutoff: float,
                   fs: int, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    c   = min(cutoff / nyq, 1 - 1e-4)
    b, a = butter(order, c, btype="low")
    return filtfilt(b, a, x)


def highpass_filter(x: np.ndarray, cutoff: float,
                    fs: int, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    c   = max(cutoff / nyq, 1e-4)
    b, a = butter(order, c, btype="high")
    return filtfilt(b, a, x)


def get_ewt_boundaries(x: np.ndarray, fs: int, n_ssc: int = 3) -> list:
    """
    Simplified EWT: deteksi boundary frekuensi dari spektrum.
    Mengembalikan list of (f_low, f_high) untuk setiap SSC.
    Referensi: Chen et al. (2023) — Paper 4
    """
    n = len(x)
    freqs = rfftfreq(n, d=1.0 / fs)
    mag   = np.abs(rfft(x))

    # Cari peaks pada spektrum
    min_dist = max(1, len(freqs) // 20)
    peaks, _ = find_peaks(mag, distance=min_dist,
                          height=np.percentile(mag, 60))

    if len(peaks) < n_ssc - 1:
        # Fallback: bagi spektrum secara equal
        max_f = freqs[-1]
        boundaries = [i * max_f / n_ssc for i in range(n_ssc + 1)]
    else:
        # Pilih n_ssc-1 peaks terbesar sebagai boundary
        top_peaks = sorted(peaks[np.argsort(mag[peaks])[::-1]][:n_ssc - 1])
        boundaries = [0.0] + [freqs[p] for p in top_peaks] + [freqs[-1]]

    # Konversi ke list of intervals
    intervals = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
    return intervals[:n_ssc]


def extract_17_features(x: np.ndarray, prefix: str) -> dict:
    """
    17 fitur time-domain dari sinyal x (Paper 4: Chen et al. 2023).
    """
    eps = 1e-12
    n   = len(x)
    mean_abs = np.mean(np.abs(x)) + eps

    # Zero-crossing count
    zcc = int(np.sum(np.diff(np.sign(x)) != 0))

    # Square root amplitude (F8)
    sqrt_amp = (np.mean(np.sqrt(np.abs(x) + eps))) ** 2

    # RMS / Mean Square Amplitude (F10)
    rms = np.sqrt(np.mean(x ** 2))

    # Peak value (F11)
    peak = np.max(np.abs(x))

    # PSD for entropy
    mag   = np.abs(rfft(x))
    psd   = mag ** 2
    psd_n = psd / (psd.sum() + eps)
    ent   = -np.sum(psd_n * np.log2(psd_n + eps))

    return {
        f"{prefix}_ene":   np.sum(x ** 2),                         # F1: Energy
        f"{prefix}_ent":   float(ent),                              # F2: Spectral Entropy
        f"{prefix}_zcc":   float(zcc),                              # F3: Zero-Crossing Count
        f"{prefix}_mean":  float(np.mean(x)),                      # F4: Mean
        f"{prefix}_std":   float(np.std(x)),                       # F5: Std
        f"{prefix}_var":   float(np.var(x)),                       # F6: Variance
        f"{prefix}_ptp":   float(np.max(x) - np.min(x)),          # F7: Peak-to-Peak
        f"{prefix}_sqrtamp": float(sqrt_amp),                      # F8: Sqrt Amplitude
        f"{prefix}_avgamp":  float(mean_abs),                      # F9: Average Amplitude
        f"{prefix}_rms":   float(rms),                             # F10: RMS
        f"{prefix}_peak":  float(peak),                            # F11: Peak Value
        f"{prefix}_wf":    float(rms / mean_abs),                  # F12: Waveform Index
        f"{prefix}_cf":    float(peak / mean_abs),                 # F13: Peak Index (Crest Factor)
        f"{prefix}_if":    float(peak / sqrt_amp),                 # F14: Impulsion Index
        f"{prefix}_clf":   float(peak / (sqrt_amp ** 2 + eps)),    # F15: Clearance Factor
        f"{prefix}_skew":  float(sp_stats.skew(x)),                # F16: Skewness
        f"{prefix}_kurt":  float(sp_stats.kurtosis(x)),            # F17: Kurtosis
    }


def build_feature_set_ewt(sensors: dict) -> pd.DataFrame:
    print_header(f"Feature Set F — EWT + 17 Fitur Time-Domain")
    print(f"  EWT SSC per sensor: {EWT_N_SSC}")

    all_rows = []

    for cyc_idx in range(N_CYCLES):
        if cyc_idx % 200 == 0:
            print(f"  Siklus {cyc_idx}/{N_CYCLES} ...", flush=True)

        row = {}

        # --- Sensor 100 Hz: EWT decomposition → 17 fitur per SSC ---
        for name in SENSORS_100HZ:
            fs  = SENSOR_INFO[name][0]
            x   = sensors[name][cyc_idx]

            intervals = get_ewt_boundaries(x, fs=fs, n_ssc=EWT_N_SSC)
            for ssc_id, (f_low, f_high) in enumerate(intervals):
                prefix = f"{name}_ssc{ssc_id}"
                if f_low < 1e-3:
                    ssc = lowpass_filter(x, f_high, fs)
                elif f_high >= fs / 2 - 1:
                    ssc = highpass_filter(x, f_low, fs)
                else:
                    ssc = bandpass_filter(x, f_low, f_high, fs)

                row.update(extract_17_features(ssc, prefix))

        # --- Sensor 10 Hz dan 1 Hz: Stats dasar saja (tidak EWT) ---
        for name in SENSORS_10HZ + SENSORS_1HZ:
            arr = sensors[name][cyc_idx]
            pref = name
            row[f"{pref}_mean"]   = float(np.mean(arr))
            row[f"{pref}_std"]    = float(np.std(arr))
            row[f"{pref}_min"]    = float(np.min(arr))
            row[f"{pref}_max"]    = float(np.max(arr))
            row[f"{pref}_median"] = float(np.median(arr))
            row[f"{pref}_q1"]     = float(np.percentile(arr, 25))
            row[f"{pref}_q3"]     = float(np.percentile(arr, 75))
            row[f"{pref}_iqr"]    = float(np.percentile(arr, 75) - np.percentile(arr, 25))
            row[f"{pref}_var"]    = float(np.var(arr))
            row[f"{pref}_skew"]   = float(sp_stats.skew(arr))
            row[f"{pref}_kurt"]   = float(sp_stats.kurtosis(arr))
            row[f"{pref}_ptp"]    = float(np.max(arr) - np.min(arr))
            row[f"{pref}_rms"]    = float(np.sqrt(np.mean(arr ** 2)))
            row[f"{pref}_slope"]  = float(np.polyfit(np.arange(len(arr)), arr, 1)[0])

        all_rows.append(row)

    df = pd.DataFrame(all_rows)

    n_ewt_feats  = len(SENSORS_100HZ) * EWT_N_SSC * 17
    n_stat_feats = (len(SENSORS_10HZ) + len(SENSORS_1HZ)) * 14
    print(f"\n  EWT fitur   : {n_ewt_feats}  (7 sensor × {EWT_N_SSC} SSC × 17 fitur)")
    print(f"  Stats fitur : {n_stat_feats}  (non-100Hz sensor)")
    print(f"  Raw shape   : {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    OUTPUTS_FEATURES.mkdir(parents=True, exist_ok=True)
    PLOTS_FEAT.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    print("\n" + "=" * 60)
    print("  03_feature_extraction.py — Multi Feature Set")
    print("=" * 60)

    sensors = load_sensors()
    print(f"  Sensor dimuat: {list(sensors.keys())}")

    results = {}

    # ── Feature Set A: Stats ──────────────────────────────────────
    t0 = time.time()
    df_A = build_feature_set_stats(sensors)
    df_A = nan_filter(df_A, "FeatureSet-A")
    out_A = OUTPUTS_FEATURES / "features_stats.csv"
    df_A.to_csv(out_A, index=False)
    results["A"] = df_A.shape
    print(f"  💾 Disimpan: {out_A.name}  [{df_A.shape}]  ({time.time()-t0:.1f}s)")

    # ── Feature Set B: Stats + FFT ────────────────────────────────
    t0 = time.time()
    df_B = build_feature_set_stats_fft(sensors)
    df_B = nan_filter(df_B, "FeatureSet-B")
    out_B = OUTPUTS_FEATURES / "features_stats_fft.csv"
    df_B.to_csv(out_B, index=False)
    results["B"] = df_B.shape
    print(f"  💾 Disimpan: {out_B.name}  [{df_B.shape}]  ({time.time()-t0:.1f}s)")

    # ── Feature Set C: Stats + FFT + 13-Cluster ───────────────────
    t0 = time.time()
    df_C = build_feature_set_stats_fft_13cluster(sensors)
    df_C = nan_filter(df_C, "FeatureSet-C")
    out_C = OUTPUTS_FEATURES / "features_stats_fft_13cl.csv"
    df_C.to_csv(out_C, index=False)
    results["C"] = df_C.shape
    print(f"  💾 Disimpan: {out_C.name}  [{df_C.shape}]  ({time.time()-t0:.1f}s)")

    # ── Feature Set F: EWT ───────────────────────────────────────
    t0 = time.time()
    df_F = build_feature_set_ewt(sensors)
    df_F = nan_filter(df_F, "FeatureSet-F")
    out_F = OUTPUTS_FEATURES / "features_ewt.csv"
    df_F.to_csv(out_F, index=False)
    results["F"] = df_F.shape
    print(f"  💾 Disimpan: {out_F.name}  [{df_F.shape}]  ({time.time()-t0:.1f}s)")

    # ── Ringkasan ─────────────────────────────────────────────────
    print_header("Ringkasan Feature Extraction")
    print(f"  {'Feature Set':<30} {'Shape'}")
    print(f"  {'─'*30} {'─'*15}")
    names = {
        "A": "Stats (EXP-A)",
        "B": "Stats+FFT (EXP-B,D,E,G,H)",
        "C": "Stats+FFT+13Cluster (EXP-C)",
        "F": "EWT+17feat (EXP-F)",
    }
    for k, shape in results.items():
        print(f"  {names[k]:<30} {str(shape)}")

    elapsed = time.time() - t_total
    print(f"\n  ⏱  Total waktu: {elapsed:.1f} detik")
    print(f"  → Lanjut ke: python 05_run_experiments.py\n")


if __name__ == "__main__":
    main()
