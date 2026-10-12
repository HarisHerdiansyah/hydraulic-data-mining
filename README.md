# Condition Monitoring of Hydraulic Systems: Multi-Pipeline Preprocessing & Clustering Analysis

Repositori ini memuat implementasi modular untuk pra-pemrosesan data (*preprocessing*), ekstraksi fitur deret waktu (*time-series feature extraction*), reduksi dimensi (*PCA*), dan evaluasi klasterisasi (*unsupervised clustering*) pada dataset **Condition Monitoring of Hydraulic Systems** (ZeMA / UCI Machine Learning Repository).

Proyek ini membandingkan **8 rancangan pipeline eksperimen** (EXP_A hingga EXP_H) untuk menganalisis konfigurasi representasi fitur terbaik bagi pengelompokan kondisi operasional sistem hidraulik.

---

## 1. Struktur Repositori

```text
.
├── preprocessing/                 # Skrip modular pipeline data
│   ├── config.py                  # Konfigurasi terpusat (path, parameter, konstanta)
│   ├── 01_load_data.py            # Fase 01: Ingestion & validasi data sensor
│   ├── 02_eda.py                  # Fase 02: Exploratory Data Analysis & visualisasi distribusi
│   ├── 03_feature_extraction.py   # Fase 03: Ekstraksi fitur (Stats, FFT, 13-Cl, EWT)
│   ├── 04_normalize_pca.py        # Fase 04: Normalisasi RobustScaler & PCA baseline
│   ├── 05_run_experiments.py      # Fase 05: Eksekusi 8 variasi pipeline (EXP_A–EXP_H)
│   ├── 06_clustering_eval.py      # Fase 06: Klasterisasi K-Means & evaluasi metrik
│   └── 07_export_npy_to_csv.py    # Utilitas: Ekspor matriks array (.npy) ke CSV
├── outputs/
│   ├── plots/                     # Visualisasi hasil grafik (PNG)
│   │   ├── eda/                   # Boxplot per label, dendrogram, korelasi sensor
│   │   ├── features/              # Spektrum FFT dan dekomposisi fitur
│   │   ├── pca/                   # Scree plot, variance rasio, 2D PCA projection
│   │   └── comparison/            # Heatmap Silhouette score & perbandingan DBI
│   └── reports/                   # Tabel metrik kuantitatif (CSV)
│       ├── comparison_metrics.csv # Metrik Silhouette, DBI, CHI (k = 2..6)
│       ├── best_k_per_exp.csv     # Ringkasan k optimal tiap eksperimen
│       └── experiments_summary.csv# Ringkasan dimensi fitur sebelum & sesudah PCA
├── libraries/
│   ├── dataset/                   # Direktori penempatan raw dataset (.txt)
│   │   └── README.md              # Petunjuk tautan unduh data resmi UCI
│   ├── description.txt            # Metadata sensor hidraulik
│   ├── documentation.txt          # Dokumentasi teknis rig uji
│   └── profile.txt                # Label target operasional (2.205 siklus)
├── run_pipeline.py                # Runner utama untuk eksekusi otomatis end-to-end
├── requirements.txt               # Daftar pustaka / dependensi Python
├── .gitignore                     # Filter pengabaian cache, file biner, dan dataset besar
└── README.md                      # Dokumentasi utama proyek
```

---

## 2. Instalasi dan Persiapan Lingkungan

Proyek ini dibangun menggunakan Python 3.10+ (atau kompatibel dengan Python 3.8+).

1. **Clone atau Ekstrak Repositori:**
   ```bash
   git clone <repo-url>
   cd hyrdraulic
   ```

2. **Buat Virtual Environment (Direkomendasikan):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Di Linux / macOS
   # .venv\Scripts\activate   # Di Windows
   ```

3. **Instal Dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 3. Persiapan Dataset

Dataset yang digunakan adalah **Condition Monitoring of Hydraulic Systems** (Helwig et al., 2018):
- **Sumber Unduh:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems)
- Unduh dan ekstrak seluruh berkas sensor `.txt` (`PS1.txt` s/d `PS6.txt`, `EPS1.txt`, `FS1.txt`, `FS2.txt`, `TS1.txt` s/d `TS4.txt`, `VS1.txt`, `CE.txt`, `CP.txt`, `SE.txt`) ke dalam folder:
  ```text
  libraries/dataset/
  ```
- Pastikan berkas label `profile.txt` berada di folder:
  ```text
  libraries/profile.txt
  ```

---

## 4. Cara Menjalankan Pipeline

Seluruh alur kerja dirancang untuk dapat direproduksi secara penuh (*fully reproducible*) menggunakan orchestrator `run_pipeline.py`.

### A. Menjalankan Seluruh Fase Sekaligus (End-to-End)
```bash
python run_pipeline.py
```
Perintah ini akan mengeksekusi secara berurutan:
1. **Fase 01:** Membaca 17 berkas sensor, memverifikasi ketiadaan *missing value*, dan membentuk `sensors.npz`.
2. **Fase 02:** Melakukan analisis statistik deskriptif, korelasi antar-sensor, dan mendeteksi titik ekstrem (*EDA*).
3. **Fase 03:** Mengekstrak representasi fitur deret waktu (Statistik, Tren, FFT, 13-Klaster temporal, dan EWT).
4. **Fase 04:** Menjalankan normalisasi `RobustScaler` dan baseline PCA.
5. **Fase 05:** Memproses 8 konfigurasi eksperimen perlakuan data (EXP_A hingga EXP_H) dan menyimpannya.
6. **Fase 06:** Menjalankan klasterisasi K-Means ($k=2 \dots 6$), menghitung *Silhouette Score*, *Davies-Bouldin Index*, dan *Calinski-Harabasz Index*.

### B. Menjalankan Mulai dari Fase Tertentu
Jika fase ekstraksi fitur sudah pernah dijalankan, Anda dapat memulai langsung dari fase tertentu:
```bash
python run_pipeline.py --from 05
```

### C. Menjalankan Satu Fase Saja
```bash
python run_pipeline.py --only 06
```

---

## 5. Ringkasan Desain Eksperimen (EXP_A – EXP_H)

| Eksperimen | Basis Fitur | Filter Korelasi | Metode Skalasi | Target PCA | Jumlah Komponen |
|---|---|---|---|---|---|
| **EXP_A** | Statistik Dasar (Mean, Std, IQR, Skew, dll.) | \|r\| > 0,95 | RobustScaler | 95% EVR | 2 Komponen |
| **EXP_B** | Statistik Dasar + Spektral FFT | \|r\| > 0,95 | RobustScaler | 95% EVR | 2 Komponen |
| **EXP_C** | Statistik + FFT + 13-Klaster Temporal (PS1) | \|r\| > 0,95 | RobustScaler | 95% EVR | 2 Komponen |
| **EXP_D** | Statistik + FFT | \|r\| > 0,95 | StandardScaler (Z-score) | 95% EVR | 46 Komponen |
| **EXP_E** | Statistik + FFT | \|r\| > 0,95 | MinMaxScaler | 95% EVR | 31 Komponen |
| **EXP_F** | EWT (*Empirical Wavelet Transform*) + Time Domain | \|r\| > 0,95 | RobustScaler | 95% EVR | 2 Komponen |
| **EXP_G** | Statistik + FFT | \|r\| > 0,95 | RobustScaler | 85% EVR | 2 Komponen |
| **EXP_H** | Statistik + FFT | \|r\| > 0,95 | RobustScaler | *Tanpa PCA* | 115 Fitur Asli |

---

## 6. Hasil dan Artefak Output

Setiap keluaran otomatis disimpan ke direktori `outputs/`:
- **`outputs/plots/comparison/`**:
  - `01_heatmap_silhouette.png`: Visualisasi skor Silhouette pada seluruh variasi eksperimen dan klaster $k$.
  - `02_barplot_dbi.png`: Perbandingan separasi klaster berdasarkan Davies-Bouldin Index.
  - `03_pca_clusters_best.png`: Proyeksi bidang 2D klaster terbaik.
- **`outputs/reports/`**:
  - `comparison_metrics.csv`: Data kuantitatif evaluasi K-Means per nilai $k$.
  - `best_k_per_exp.csv`: Nilai $k$ optimal yang direkomendasikan untuk masing-masing eksperimen.
  - `experiments_summary.csv`: Ringkasan dimensi data dan rasio variansi terjelaskan.

---

## 7. Referensi Metodologi

1. **Dataset ZeMA:** Helwig, N., Pignanelli, E., & Schütze, A. (2018). *Condition Monitoring of Hydraulic Systems Data Set*. ZeMA Test Rig, Saarland University.
2. **Robust Preprocessing & FFT:** Shaikh, S. A., et al. (2025). *Unsupervised Anomaly Detection on Industrial Pump Sensors*. Machine Learning with Applications.
3. **Reduksi Dimensi PCA & Korelasi:** Silva, F., & Maitelli, C. (2025). *Anomaly Detection in Water Pumping Systems*. Sensors, 25(11), 3532.
4. **Segmentasi Temporal Sub-siklus:** Kim, J., & Heo, S. (2022). *Feature Extraction for Hydraulic IoT Sensors*. Sensors, 22(7), 2479.
5. **Dekomposisi Sinyal EWT:** Chen, X., et al. (2023). *Fault Diagnosis of Hydraulic System Based on Improved Empirical Wavelet Transform*. Journal of Mechanical Engineering.
6. **Evaluasi Klasterisasi:** Kurniawan, R., et al. (2026). *Comparative Study of Clustering Algorithms on Industrial Big Data*.
