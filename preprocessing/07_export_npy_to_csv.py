"""
Ekspor EXP_A..EXP_H (.npy) ke CSV berkolom + label kondisi dari profile.csv.
Output: outputs/data/preprocessed_csv/EXP_X.csv
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "outputs" / "data" / "preprocessed"
DST = ROOT / "outputs" / "data" / "preprocessed_csv"
DST.mkdir(parents=True, exist_ok=True)

profile = pd.read_csv(ROOT / "outputs" / "data" / "profile.csv")

for npy in sorted(SRC.glob("EXP_?.npy")):
    exp = npy.stem
    X = np.load(npy)
    meta = json.loads((SRC / f"{exp}_meta.json").read_text())
    prefix = "PC" if meta.get("pca_threshold") is not None else "feat_"
    cols = [f"{prefix}{i + 1}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df.insert(0, "cycle_id", range(len(df)))
    assert len(df) == len(profile), f"{exp}: jumlah baris tidak cocok"
    df = pd.concat([df, profile.reset_index(drop=True)], axis=1)
    df.to_csv(DST / f"{exp}.csv", index=False)
    print(f"{exp}: {df.shape} -> {DST / (exp + '.csv')}")
