# Dataset: Condition Monitoring of Hydraulic Systems

## Sumber & Tautan Unduh
Dataset ini berasal dari ZeMA test rig dan di-host secara publik di **UCI Machine Learning Repository**:
- URL: https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems
- Referensi: Helwig, N., Pignanelli, E., & Schütze, A. (2018). *Condition Monitoring of Hydraulic Systems*. UCI Machine Learning Repository.

## Petunjuk Penempatan File
Ekstrak file `.txt` sensor ke dalam direktori ini (`libraries/dataset/`):
- **Tekanan (100 Hz):** `PS1.txt`, `PS2.txt`, `PS3.txt`, `PS4.txt`, `PS5.txt`, `PS6.txt`
- **Daya Motor (100 Hz):** `EPS1.txt`
- **Laju Aliran (10 Hz):** `FS1.txt`, `FS2.txt`
- **Temperatur (1 Hz):** `TS1.txt`, `TS2.txt`, `TS3.txt`, `TS4.txt`
- **Getaran (1 Hz):** `VS1.txt`
- **Efisiensi (1 Hz):** `CE.txt`, `CP.txt`, `SE.txt`

Pastikan file label target `profile.txt` berada di folder `libraries/profile.txt`.
