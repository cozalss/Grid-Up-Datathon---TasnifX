"""H8j -- v71'de ISTENEN olcek gercekten ULASILDI mi?

NEDEN
-----
H7 ajani gercek bir kusur sinifi buldu: ``son_islem_gunolcek.py`` istenen c'yi
tam uygulamiyor -- ``np.clip(...,0,None)`` sifira yakin tahminleri kirpiyor ve
ULASILAN olcek istenenden kucuk kaliyor (v66: istenen 1,335, ulasilan 1,3241;
v55: istenen 1,492, ulasilan 1,4760). Betigin %3'luk kapisi bunu SESSIZ geciyor.

Ayni kusur v71'de de olabilir: soguk tarafta 7.163 sifir tahmin var ve
expm1(log1p(0) + negatif delta) < 0 -> 0'a kirpiliyor, yani o satirlarda
istenen kayma UYGULANMIYOR.

BU BETIK
--------
v67 (giris) ve v71 (cikis) soguk gun eksenlerini AYNI protokolle olcup
    ulasilan_c = sigma(v71 gun ekseni) / sigma(v67 gun ekseni)
oranini verir. Istenen 2,20'ye ne kadar yakin?

Ayrica H7'nin dersini uygular: TASINABILIR sabit c DEGIL, HEDEF GENLIK'tir.
Soguk taraf icin hedef genlik capadan geliyordu:
    S*_soguk = sigma_gercek(2025 Nis-Tem, dogmus trafolar) x kor
             = 0,3829 x 0,8971 = 0,3435   <- gun ekseni std'sinin varmasi gereken yer
Ulasilan genlik bunun neresinde?
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
MIN_YAS, MIN_GUN = 7, 60
ISTENEN_C = 2.20
SIGMA_GERCEK = 0.3829
KOR = 0.8971


def iki_yonlu(v, bi, gi, nb, ng, tur=400):
    mu = float(v.mean())
    a = np.zeros(nb)
    b = np.zeros(ng)
    cb = np.maximum(np.bincount(bi, minlength=nb), 1)
    cg = np.maximum(np.bincount(gi, minlength=ng), 1)
    for _ in range(tur):
        a = np.bincount(bi, v - mu - b[gi], minlength=nb) / cb
        b = np.bincount(gi, v - mu - a[bi], minlength=ng) / cg
        b -= b.mean()
    return b


def main() -> int:
    tr_tanim = set(
        pd.read_csv(KOK / "data/raw/train.csv", usecols=["tanim"], dtype={"tanim": str})[
            "tanim"
        ].unique()
    )
    te = pd.read_csv(
        KOK / "data/raw/test.csv",
        usecols=["id", "tanim", "tarih"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
    )
    tc = te[~te["tanim"].isin(tr_tanim)].reset_index(drop=True)
    idx = (~te["tanim"].isin(tr_tanim)).to_numpy()

    ilk = tc.groupby("tanim")["tarih"].transform("min")
    yas = (tc["tarih"] - ilk).dt.days.to_numpy()
    say = tc.groupby("tanim")["tanim"].transform("size").to_numpy()
    temiz = (yas >= MIN_YAS) & (say >= MIN_GUN)
    t = tc.loc[temiz].reset_index(drop=True)
    bi, _ = pd.factorize(t["tanim"])
    gi, _ = pd.factorize(t["tarih"])
    nb, ng = int(bi.max()) + 1, int(gi.max()) + 1
    n_d = np.bincount(gi, minlength=ng).astype(float)

    def gun_std(ad: str) -> tuple[float, np.ndarray, int]:
        d = pd.read_csv(KOK / "submissions" / ad)
        assert (d["id"].values == te["id"].values).all(), ad
        lg = np.log1p(np.clip(d["tuketim"].to_numpy(dtype="float64"), 0, None))
        sf = int((d["tuketim"].to_numpy()[idx] == 0).sum())
        b = iki_yonlu(lg[idx][temiz], bi, gi, nb, ng)
        bc = b - float(np.dot(n_d, b) / n_d.sum())
        return float(np.sqrt(np.dot(n_d, bc**2) / n_d.sum())), bc, sf

    s67, b67, z67 = gun_std("tuketim_v67_c1335_olay.csv")
    s71, b71, z71 = gun_std("tuketim_v71_soguk_gun.csv")

    print("SOGUK GUN EKSENI GENLIGI (satir-agirlikli, trafo etkisi cikarilmis, T3)")
    print(f"  v67 (giris)  sigma {s67:.4f}   sifir tahmin {z67:,}")
    print(f"  v71 (cikis)  sigma {s71:.4f}   sifir tahmin {z71:,}")
    print(f"\n  ISTENEN c   {ISTENEN_C:.4f}")
    print(f"  ULASILAN c  {s71 / s67:.4f}   (fark {(s71 / s67) / ISTENEN_C - 1:+.2%})")

    hedef = SIGMA_GERCEK * KOR
    print("\nH7 DERSI -- tasinabilir sabit c DEGIL, HEDEF GENLIK S*")
    print(f"  S*_soguk = sigma_gercek x kor = {SIGMA_GERCEK:.4f} x {KOR:.4f} = {hedef:.4f}")
    print(f"  v67 genligi {s67:.4f}  ({s67 / hedef:.1%} of S*)")
    print(f"  v71 genligi {s71:.4f}  ({s71 / hedef:.1%} of S*)")
    print(f"  S*'a tam varmak icin gereken c = {hedef / s67:.4f}")

    kor = float(np.corrcoef(b67, b71)[0, 1])
    print(
        f"\n  gun profili korelasyonu v67 vs v71: {kor:+.6f}  (1,000000 olmali "
        f"-- yalniz GENLIK degismeli, SEKIL degil)"
    )
    if kor < 0.9999:
        print("  UYARI: sekil degismis -- kirpma profili bozuyor olabilir")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
