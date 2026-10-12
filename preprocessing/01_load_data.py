"""
01_load_data.py — FASE 1A: Loading & Verifikasi Data
=====================================================
Tugas:
  - Load semua 17 file sensor dari libraries/dataset/
  - Load profile.txt (label kondisi)
  - Verifikasi dimensi dan tipe data
  - Simpan ke outputs/data/ sebagai:
      sensors.npz          (semua array sensor)
      profile.csv          (label kondisi)

Referensi: Silva & Maitelli (2025) — ETL step pertama
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATASET_PATH, PROFILE_PATH, SENSOR_INFO, PROFILE_COLS,
    N_CYCLES, OUTPUTS_DATA
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def print_header(title: str):
    bar = "=" * 55
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


def load_sensors() -> dict[str, np.ndarray]:
    """Load tiap file sensor; return dict {name: array(2205, T)}."""
    sensors = {}
    print_header("FASE 1A — Loading Sensor Data")

    for name, (hz, expected_cols) in SENSOR_INFO.items():
        fpath = DATASET_PATH / f"{name}.txt"
        if not fpath.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {fpath}")

        arr = pd.read_csv(fpath, sep="\t", header=None).values.astype(np.float64)

        if arr.shape != (N_CYCLES, expected_cols):
            raise ValueError(
                f"{name}: shape tidak sesuai! "
                f"Got {arr.shape}, expected ({N_CYCLES}, {expected_cols})"
            )

        sensors[name] = arr
        print(f"  ✅ {name:<6}  shape={arr.shape}  "
              f"[{hz} Hz | {expected_cols} pts/siklus]")

    return sensors


def load_profile() -> pd.DataFrame:
    """Load profile.txt sebagai DataFrame."""
    profile = pd.read_csv(
        PROFILE_PATH, sep="\t", header=None, names=PROFILE_COLS
    )
    assert profile.shape == (N_CYCLES, len(PROFILE_COLS)), \
        f"profile shape mismatch: {profile.shape}"
    return profile


def print_profile_summary(profile: pd.DataFrame):
    """Cetak distribusi setiap label kondisi."""
    print_header("Distribusi Label Kondisi (profile.txt)")
    for col in PROFILE_COLS:
        print(f"\n  [{col}]")
        vc = profile[col].value_counts().sort_index()
        for val, cnt in vc.items():
            print(f"    {val:>6}  →  {cnt:>4} siklus")


def save_outputs(sensors: dict, profile: pd.DataFrame):
    """Simpan data ke outputs/data/."""
    OUTPUTS_DATA.mkdir(parents=True, exist_ok=True)

    # Simpan semua sensor dalam satu .npz
    sensor_path = OUTPUTS_DATA / "sensors.npz"
    np.savez_compressed(sensor_path, **sensors)

    # Simpan profile
    profile_path = OUTPUTS_DATA / "profile.csv"
    profile.to_csv(profile_path, index=False)

    print_header("Output Disimpan")
    print(f"  sensors.npz  →  {sensor_path}")
    print(f"  profile.csv  →  {profile_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    sensors = load_sensors()
    profile = load_profile()

    # Missing values check
    nan_count = sum(np.isnan(arr).sum() for arr in sensors.values())
    print(f"\n  Missing values di sensor data: {nan_count}")
    print(f"  Missing values di profile    : {profile.isnull().sum().sum()}")

    print_profile_summary(profile)
    save_outputs(sensors, profile)

    elapsed = time.time() - t0
    print(f"\n  ⏱  Selesai dalam {elapsed:.2f} detik")
    print(f"  → Lanjut ke: python 02_eda.py\n")


if __name__ == "__main__":
    main()
