"""
run_pipeline.py — Orchestrator: jalankan semua fase secara berurutan
====================================================================
Cara pakai:
  python run_pipeline.py            # jalankan semua fase (01 → 06)
  python run_pipeline.py --from 03  # mulai dari fase tertentu
  python run_pipeline.py --only 06  # jalankan satu fase saja

Setiap fase mencetak status dan waktu eksekusi.
"""

import sys
import time
import argparse
import subprocess
from pathlib import Path

PREPROCESSING_DIR = Path(__file__).resolve().parent / "preprocessing"

PHASES = {
    "01": ("01_load_data.py",          "Load & Verifikasi Data"),
    "02": ("02_eda.py",                "Exploratory Data Analysis"),
    "03": ("03_feature_extraction.py", "Feature Extraction (Stats, FFT, 13-Cl, EWT)"),
    "04": ("04_normalize_pca.py",      "Normalisasi + PCA (Baseline)"),
    "05": ("05_run_experiments.py",    "Multi-Experiment Pipelines (EXP_A–EXP_H)"),
    "06": ("06_clustering_eval.py",    "K-Means Clustering & Evaluasi Metrik"),
}


def run_phase(phase_id: str):
    script, desc = PHASES[phase_id]
    script_path  = PREPROCESSING_DIR / script

    print(f"\n{'='*60}")
    print(f"  FASE {phase_id} — {desc}")
    print(f"{'='*60}")

    t0  = time.time()
    ret = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PREPROCESSING_DIR.parent)
    )
    elapsed = time.time() - t0

    if ret.returncode != 0:
        print(f"\n  ❌ Fase {phase_id} gagal (exit code {ret.returncode})")
        sys.exit(ret.returncode)

    print(f"  ✅ Fase {phase_id} selesai dalam {elapsed:.1f} detik")


def main():
    parser = argparse.ArgumentParser(
        description="Jalankan pipeline preprocessing hydraulic dataset"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--from", dest="from_phase", metavar="N",
        help="Mulai dari fase N (contoh: --from 03)"
    )
    group.add_argument(
        "--only", dest="only_phase", metavar="N",
        help="Jalankan satu fase saja (contoh: --only 02)"
    )
    args = parser.parse_args()

    t_total = time.time()
    print("\n" + "=" * 60)
    print("  HYDRAULIC DATASET — PREPROCESSING PIPELINE")
    print("  Referensi: Shaikh et al. (2025) | Silva & Maitelli (2025)")
    print("=" * 60)

    if args.only_phase:
        if args.only_phase not in PHASES:
            print(f"  ❌ Fase tidak dikenal: {args.only_phase}")
            print(f"  Fase tersedia: {list(PHASES.keys())}")
            sys.exit(1)
        run_phase(args.only_phase)

    else:
        start_from = args.from_phase or "01"
        if start_from not in PHASES:
            print(f"  ❌ Fase tidak dikenal: {start_from}")
            sys.exit(1)

        phase_ids = sorted(PHASES.keys())
        phases_to_run = [p for p in phase_ids if p >= start_from]

        for phase_id in phases_to_run:
            run_phase(phase_id)

    total_elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  🏁 Pipeline selesai dalam {total_elapsed:.1f} detik")
    print(f"  📂 Output tersimpan di: outputs/")
    print(f"     data/  → file CSV & NPZ")
    print(f"     plots/ → semua visualisasi PNG")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
