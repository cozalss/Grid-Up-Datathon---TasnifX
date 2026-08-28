"""v93 denetimi -- adim 1: butun olculmus gonderimleri log1p uzayina yukle ve onbellekle."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
SUB = KOK / "submissions"
ONB = Path(__file__).resolve().parent / "onbellek"
ONB.mkdir(exist_ok=True)

# (etiket, dosya, public skor, kaggle ref, kaggle UTC damgasi)
OLCUMLER = [
    ("v2", "tuketim_v2.csv", 1.16143, 55668648, "2026-08-21 12:08:18"),
    ("v7", "tuketim_v7.csv", 1.16922, 55668961, "2026-08-21 12:29:47"),
    ("v15", "tuketim_v15.csv", 1.03910, 55684049, "2026-08-22 05:32:05"),
    ("v16", "tuketim_v16.csv", 1.06605, 55684082, "2026-08-22 05:35:06"),
    ("v18", "tuketim_v18.csv", 1.03370, 55688907, "2026-08-22 10:26:34"),
    ("v25", "tuketim_v25_hedge.csv", 1.04820, 55707720, "2026-08-23 06:07:20"),
    ("v27", "tuketim_v27_v18hedge.csv", 1.03362, 55707804, "2026-08-23 06:13:46"),
    ("v30", "tuketim_v30_buzme.csv", 1.02639, 55717274, "2026-08-23 14:26:55"),
    ("v46", "tuketim_v46_gun.csv", 1.02448, 55732647, "2026-08-24 04:19:17"),
    ("v44", "tuketim_v44_v27yeni.csv", 1.03053, 55732790, "2026-08-24 04:27:24"),
    ("v47", "tuketim_v47_eskison.csv", 1.01750, 55732850, "2026-08-24 04:30:49"),
    ("v50", "tuketim_v50_nihai30.csv", 1.01686, 55755676, "2026-08-25 00:00:45"),
    ("v55", "tuketim_v55_gunolcek.csv", 1.01591, 55755748, "2026-08-25 00:01:53"),
    ("v67", "tuketim_v67_c1335_olay.csv", 1.01548, 55780927, "2026-08-26 00:02:32"),
    ("v73", "tuketim_v73_soguk_gun160.csv", 1.01538, 55780987, "2026-08-26 00:03:52"),
    ("v79", "tuketim_v79_S3.csv", 1.01556, 55781080, "2026-08-26 00:06:13"),
    ("v80", "tuketim_v80_optimum.csv", 1.01341, 55811381, "2026-08-27 06:02:09"),
    ("v81", "tuketim_v81_sicak08.csv", 1.01429, 55811392, "2026-08-27 06:02:59"),
    ("v83", "tuketim_v83_sicak_optimum.csv", 1.01318, 55811502, "2026-08-27 06:09:27"),
]
ADAYLAR = [
    ("v93", "tuketim_v93_gram_optimum.csv", None, None, None),
    ("v85", "tuketim_v85_gram_rank2.csv", None, None, None),
]


def sha256(yol: Path) -> str:
    h = hashlib.sha256()
    with open(yol, "rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest().upper()


def main() -> None:
    ref_id = None
    kayit = {}
    for etiket, ad, skor, ref, damga in OLCUMLER + ADAYLAR:
        yol = SUB / ad
        npy = ONB / f"{etiket}.npy"
        st = yol.stat()
        if npy.exists():
            lg = np.load(npy)
        else:
            df = pd.read_csv(yol)
            kol = df.columns[1]
            ids = df["id"].to_numpy()
            if ref_id is None:
                np.save(ONB / "_ids.npy", ids)
                ref_id = ids
            else:
                if ref_id is None:
                    ref_id = np.load(ONB / "_ids.npy")
                assert np.array_equal(ids, ref_id), f"{etiket}: id sirasi FARKLI"
            lg = np.log1p(df[kol].to_numpy(dtype=np.float64))
            np.save(npy, lg)
        if ref_id is None and (ONB / "_ids.npy").exists():
            ref_id = np.load(ONB / "_ids.npy")
        kayit[etiket] = {
            "dosya": ad,
            "skor": skor,
            "ref": ref,
            "kaggle_utc": damga,
            "sha256": sha256(yol),
            "mtime_yerel": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
            "boyut": st.st_size,
            "n": int(lg.size),
            "min_log": float(lg.min()),
            "max_log": float(lg.max()),
            "negatif_tahmin": int((np.expm1(lg) < 0).sum()),
            "nan": int(np.isnan(lg).sum()),
        }
        print(
            f"{etiket:5s} n={lg.size} min={lg.min():.6f} max={lg.max():.4f} sha={kayit[etiket]['sha256'][:12]}"
        )
    (Path(__file__).resolve().parent / "envanter.json").write_text(
        json.dumps(kayit, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
