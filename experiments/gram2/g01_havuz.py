"""G1 -- Gonderim havuzunu kur, sha256 dogrula, log1p matrisini onbellege al.

Kaggle'a HICBIR SEY GONDERMEZ. Yalnizca yerel dosyalari okur.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

KOK = Path(__file__).resolve().parents[2]
GON = KOK / "submissions"
CIK = Path(__file__).resolve().parent

# Kaggle "competitions submissions" ciktisindan (2026-08-28 04:30) BIREBIR.
# gun1_baseline.csv DISLANDI (farkli format: hedef kolonu / R00320 id'leri).
# v55 iki kez gonderildi (55755748/55755749), ayni dosya ayni skor -> tek kayit.
HAVUZ = [
    # (ad, dosya, public skor, kaggle ref, tarih)
    ("v2", "tuketim_v2.csv", 1.16143, "55668648", "2026-08-21 12:08"),
    ("v7", "tuketim_v7.csv", 1.16922, "55668961", "2026-08-21 12:29"),
    ("v15", "tuketim_v15.csv", 1.03910, "55684049", "2026-08-22 05:32"),
    ("v16", "tuketim_v16.csv", 1.06605, "55684082", "2026-08-22 05:35"),
    ("v18", "tuketim_v18.csv", 1.03370, "55688907", "2026-08-22 10:26"),
    ("v25", "tuketim_v25_hedge.csv", 1.04820, "55707720", "2026-08-23 06:07"),
    ("v27", "tuketim_v27_v18hedge.csv", 1.03362, "55707804", "2026-08-23 06:13"),
    ("v30", "tuketim_v30_buzme.csv", 1.02639, "55717274", "2026-08-23 14:26"),
    ("v46", "tuketim_v46_gun.csv", 1.02448, "55732647", "2026-08-24 04:19"),
    ("v44", "tuketim_v44_v27yeni.csv", 1.03053, "55732790", "2026-08-24 04:27"),
    ("v47", "tuketim_v47_eskison.csv", 1.01750, "55732850", "2026-08-24 04:30"),
    ("v50", "tuketim_v50_nihai30.csv", 1.01686, "55755676", "2026-08-25 00:00"),
    ("v55", "tuketim_v55_gunolcek.csv", 1.01591, "55755749", "2026-08-25 00:01"),
    ("v67", "tuketim_v67_c1335_olay.csv", 1.01548, "55780927", "2026-08-26 00:02"),
    ("v73", "tuketim_v73_soguk_gun160.csv", 1.01538, "55780987", "2026-08-26 00:03"),
    ("v79", "tuketim_v79_S3.csv", 1.01556, "55781080", "2026-08-26 00:06"),
    ("v80", "tuketim_v80_optimum.csv", 1.01341, "55811381", "2026-08-27 06:02"),
    ("v81", "tuketim_v81_sicak08.csv", 1.01429, "55811392", "2026-08-27 06:02"),
    ("v83", "tuketim_v83_sicak_optimum.csv", 1.01318, "55811502", "2026-08-27 06:09"),
    ("v101", "tuketim_v101_hepsi.csv", 1.01614, "55833361", "2026-08-28 04:16"),
    ("v102", "tuketim_v102_kappa_optimum.csv", 1.00553, "55833415", "2026-08-28 04:19"),
]

KILIT = {  # reports/gram_rank2.json icindeki sha256 kilitleri
    "v18": "FB4E2A4C52A8432D556C6272CEFEA14AD0DA16285AB9B56B47730103EF5BBFD3",
    "v27": "51822FF1472D32138030C9DD53019D972DD69535B318326C56D97E4AB65CBA06",
    "v30": "1764CD9C1D69F273025794606B032E944F1505AFC1239316FDBA27582E395636",
    "v44": "5F8B7F257C3BEB786E43B0425E040748B00A817A34CAF7E01ECFB1B0084051A1",
    "v46": "2C23F5FD63858F8150EEB3CDA0BA2325421028D03DE91272962EE995CD7810B3",
    "v47": "72DE024A7D77CE8BB28EB258661182512CDE7A9814214A6CA46B14B2082631E0",
    "v50": "706EEF87869EDE9FFE52B6614809F53C4C5C748041BE940B051F4936F9D68BC4",
    "v73": "08673F271EE8257BBC323FF17228289A31A55F6F6E6BC081794B1A82FDC9DEB8",
    "v81": "ED9B792B7FB8B448D5C5AC2EB28B12DDDA5CE114F80966C8EB29383E98137A35",
    "v83": "F482A9DEEB771BF6D17B9271B9D11190B8FB495D28388D35E5A6C28CAC108041",
}


def sha256(yol: Path) -> str:
    h = hashlib.sha256()
    with open(yol, "rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest().upper()


def yukle() -> tuple[list[str], np.ndarray, np.ndarray, pd.Index]:
    """log1p matrisini (K x N) dondurur. Onbellek: g01_X.npy"""
    npy = CIK / "g01_X.npy"
    meta = CIK / "g01_meta.json"
    adlar = [h[0] for h in HAVUZ]
    skorlar = np.array([h[2] for h in HAVUZ], dtype="float64")
    ids = pd.read_csv(GON / HAVUZ[0][1], usecols=["id"])["id"]
    if npy.exists() and meta.exists():
        m = json.loads(meta.read_text(encoding="utf-8"))
        if m["adlar"] == adlar:
            return adlar, np.load(npy), skorlar, pd.Index(ids)

    X = np.empty((len(HAVUZ), len(ids)), dtype="float64")
    for k, (ad, dosya, _s, _r, _t) in enumerate(HAVUZ):
        d = pd.read_csv(GON / dosya)
        assert list(d.columns) == ["id", "tuketim"], (ad, list(d.columns))
        assert (d["id"].values == ids.values).all(), f"{ad}: id sirasi farkli"
        X[k] = np.log1p(d["tuketim"].to_numpy(dtype="float64"))
    np.save(npy, X)
    meta.write_text(json.dumps({"adlar": adlar, "n": int(len(ids))}), encoding="utf-8")
    return adlar, X, skorlar, pd.Index(ids)


def main() -> None:
    print("=" * 78)
    print("G1 -- HAVUZ DENETIMI")
    print("=" * 78)
    kotu = 0
    for ad, dosya, skor, ref, tarih in HAVUZ:
        yol = GON / dosya
        var = yol.exists()
        h = sha256(yol) if var else "-"
        kilit = KILIT.get(ad)
        durum = (
            "yok"
            if not var
            else ("KILIT-OK" if kilit and h == kilit else ("KILIT-BOZUK" if kilit else "kilitsiz"))
        )
        if durum in ("yok", "KILIT-BOZUK"):
            kotu += 1
        print(f"{ad:>5} {tarih}  {skor:.5f}  ref={ref}  {durum:>11}  {h[:16]}")
    print(f"\nsha256 uyusmazligi/eksik: {kotu}")

    adlar, X, skorlar, ids = yukle()
    print(f"\nyuklendi: K={X.shape[0]} dosya, N={X.shape[1]} satir")
    print(f"log1p araligi: [{X.min():.6f}, {X.max():.6f}]  NaN={int(np.isnan(X).sum())}")


if __name__ == "__main__":
    main()
