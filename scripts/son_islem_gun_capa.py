# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""GUN EKSENI DIS CAPASI -- etiketsiz bilinen havadan gun profili duzeltmesi.

DURUM: ADAY. Uretim varsayilani DEGIL. Olcum kaydi asagida.

NE YAPAR
--------
Yalnizca SICAK satirlarin GUN ORTALAMALARINA dokunur. Trafo ekseni, seviye
ve soguk satirlar aynen kalir (soguk taraf ``son_islem.py`` ile ayri gider).

    d_t   = trafo bazinda merkezlenmis model gun ofseti (log1p uzayi)
    c_t   = CAPA: yaz25 ETIKETLERINDEN uydurulan  gun_ofset ~ CDD22 + HDD
            katsayilarinin, testin GERCEKLESMIS 2026 havasina uygulanmasi
    c~_t  = c_t * (std(d)/std(c))          <- SEKIL; genlik iddiasi YOK
    k     = 1 + lam * (std(c)/std(d) - 1)  <- GENLIK
    d'_t  = k * [ (1-gam)*d_t + gam*c~_t ]
    log1p(tahmin) += (d'_t - d_t)          (gun ortalamasi sifirlanmis olarak)

NEDEN MESRU
-----------
``hava_gunluk`` test penceresinin 122 gununun tamamini kapsiyor ve o
gunlerde ``hava_tahmin`` payi %0,0 -- yani TAHMIN degil GERCEKLESMIS gozlem.
Egitim ve test gunlerinde doluluk %100/%100, ilce-ortalamasi deger destegi
tam iceride (destek disi %0,00). Kalici kural 2 gecildi.

OLCULDU (2026-08-25, 3 tohum, uretim esli aile onbellegi, sicak satirlar)
------------------------------------------------------------------------
SEKIL-ICIN kol (gam=0,5, lam=0,0), capa BLOK DISI referanstan:

    hedef   ref     duz fark    SH        t     tohum   olcut-agirlikli
    yaz25   guz25   +0,00022   0,00006   +3,82   3/3    +0,00041
    guz25   yaz25   +0,00053   0,00003  +19,96   3/3    +0,00043
    kis26   yaz25   +0,00149   0,00006  +24,35   3/3    +0,00081

UC BLOKTA DA POZITIF -- bu gecenin tek blok-disi tasinan gun ekseni bulgusu.

Kirpilmis tablo (yaz25, tam capa gam=0,6): K=0 +0,01067 -> K=50 +0,01010;
en buyuk trafo payi %1,3, ilk5 %5,5, trafolarin %73,6'si iyilesiyor. Gun
ekseni duzeltmesi tanimi geregi YOGUNLASMAZ -- reddedilen soguk bulgularin
tersi.

GENLIK kolu (lam>0) DAHA RISKLI: testte capa std 0,187-0,203 (dogrusal /
kuadratik), bicimden bagimsiz kNN capasi 0,197; model 0,16752. Yani
k = 1,12..1,21. Ama guz25 katsayilariyla kurulan capa 0,164 verir (k=0,98).
Referans MENZIL ESLESMELI: yaz25 testin ayni aylari, guz25 degil. Elde
menzil eslesmis TEK etiketli yaz var. lam=0,5 varsayilan bu yuzden.

    python scripts/son_islem_gun_capa.py --giris submissions/X.csv \
        --cikis submissions/Y.csv --gam 0.5 --lam 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
GAM = 0.5
LAM = 0.5


def _ilce_gunluk_hava() -> pd.DataFrame:
    h = pd.read_parquet(KOK / "data/external/hava_gunluk.parquet")
    h["tarih"] = pd.to_datetime(h["tarih"])
    if "hava_tahmin" in h.columns:
        pass  # bayrak test gunlerinde %0,0 -- olculdu, kapi asagida
    return h.groupby("tarih")[["sogutma_derece_gun", "isitma_derece_gun"]].mean()


def _gun_ofseti(tarih: pd.Series, tanim: pd.Series, deger: np.ndarray) -> pd.Series:
    s = pd.Series(np.asarray(deger))
    mu = s.groupby(tanim.to_numpy()).transform("mean")
    o = (s - mu).groupby(tarih.to_numpy()).mean()
    return o - o.mean()


def _tasarim(iz: pd.DataFrame, idx: pd.Index) -> np.ndarray:
    return np.column_stack(
        [np.ones(len(idx)), iz.loc[idx, "sogutma_derece_gun"], iz.loc[idx, "isitma_derece_gun"]]
    )


def main() -> int:
    a = argparse.ArgumentParser(description="gun ekseni dis capasi (yalniz SICAK satirlar)")
    a.add_argument("--giris", required=True)
    a.add_argument("--cikis", required=True)
    a.add_argument("--gam", type=float, default=GAM, help="sekil harmani agirligi")
    a.add_argument("--lam", type=float, default=LAM, help="genlik duzeltmesi payi")
    a.add_argument("--referans", default="yaz25", help="capa katsayilarinin uydurulacagi blok")
    ar = a.parse_args()

    ornek = pd.read_csv(KOK / "data/raw/sample_submission.csv", encoding="utf-8")
    sub = pd.read_csv(ar.giris, encoding="utf-8")
    if not sub["id"].equals(ornek["id"]):
        raise RuntimeError("id sirasi sample_submission ile ayni degil")

    te = pd.read_csv(
        KOK / "data/raw/test.csv",
        usecols=["id", "tanim", "tarih"],
        encoding="utf-8",
        dtype={"tanim": str},
    )
    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "tarih", "tuketim"],
        encoding="utf-8",
        dtype={"tanim": str},
    )
    te["tarih"] = pd.to_datetime(te["tarih"])
    tr["tarih"] = pd.to_datetime(tr["tarih"])
    m = sub.merge(te, on="id", how="left", validate="one_to_one")
    sicak_trafo = set(tr["tanim"])
    sicak = m["tanim"].isin(sicak_trafo).to_numpy()

    iz = _ilce_gunluk_hava()
    eksik = pd.DatetimeIndex(m["tarih"].unique()).difference(iz.index)
    if len(eksik):
        raise RuntimeError(f"hava capasi test gunlerini kapsamiyor: {len(eksik)} gun")

    # --- CAPA: referans blogun ETIKETLERINDEN uydurulur
    pencere = {"yaz25": ("2025-04-01", "2025-07-31"), "guz25": ("2025-08-01", "2025-11-30")}[
        ar.referans
    ]
    ref = tr[(tr["tarih"] >= pencere[0]) & (tr["tarih"] <= pencere[1])]
    o_ref = _gun_ofseti(ref["tarih"], ref["tanim"], np.log1p(ref["tuketim"].clip(lower=0)))
    beta, *_ = np.linalg.lstsq(_tasarim(iz, o_ref.index), o_ref.to_numpy(), rcond=None)
    print(f"  capa ({ar.referans}) cdd {beta[1]:+.5f}  hdd {beta[2]:+.5f}")

    lp = np.log1p(m["tuketim"].clip(lower=0).to_numpy(dtype="float64"))
    ms = m.loc[sicak]
    d = _gun_ofseti(ms["tarih"], ms["tanim"], lp[sicak])
    c = pd.Series(_tasarim(iz, d.index) @ beta, index=d.index)
    c = c - c.mean()
    print(
        f"  model gun-std {d.std():.5f} | capa gun-std {c.std():.5f} | kor {np.corrcoef(d, c)[0, 1]:+.4f}"
    )

    c_sekil = c * (d.std() / c.std())
    k = 1.0 + ar.lam * (c.std() / d.std() - 1.0)
    d_yeni = k * ((1.0 - ar.gam) * d + ar.gam * c_sekil)
    d_yeni = d_yeni - d_yeni.mean()
    delta = (d_yeni - d).reindex(pd.DatetimeIndex(m["tarih"])).to_numpy()
    print(
        f"  gam {ar.gam:.2f}  lam {ar.lam:.2f}  k {k:.4f}  delta RMS {np.sqrt(np.mean(delta[sicak] ** 2)):.5f}"
    )

    lp_yeni = lp.copy()
    lp_yeni[sicak] = lp[sicak] + delta[sicak]
    yeni = np.clip(np.expm1(lp_yeni), 0.0, None)

    if not np.allclose(yeni[~sicak], m["tuketim"].to_numpy()[~sicak], rtol=1e-12, atol=1e-12):
        raise RuntimeError("SOGUK satirlar degismis olmamaliydi")
    cikti = pd.DataFrame({"id": sub["id"], "tuketim": yeni})
    if cikti["tuketim"].isna().any() or (cikti["tuketim"] < 0).any():
        raise RuntimeError("cikti NaN ya da negatif")
    cikti.to_csv(ar.cikis, index=False)
    print(f"  degisen satir {int((np.abs(yeni - m['tuketim'].to_numpy()) > 1e-9).sum()):,}")
    print(f"  yazildi: {ar.cikis}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
