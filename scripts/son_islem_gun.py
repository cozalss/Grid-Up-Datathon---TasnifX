"""SON ISLEM (gun korumali): soguk tahminlere ZAMAN EKSENI BOZULMADAN buzme.

``son_islem.py``nin iki kusurunu birden kapatir. Yeniden egitim YOK.

KUSUR 1 -- ZAMAN EKSENI EZILIYORDU
----------------------------------
Eski kurgu ofsetin TAMAMINI tek bir sabite (tahminin kendi genel ortalamasi)
dogru cekiyordu. Ofsetin iki ekseni var ve buzme ikisini ayirmiyordu:

    trafo ekseni   -> asiri yayilmis, BUZULMELI
    zaman ekseni   -> gercek mevsim rampasi, KORUNMALI

Beta ``kis26`` (Ara-Mar) uzerinde ayarlandi ve orada ay ekseni varyansi
0,00113 -- kis ofseti duz oldugu icin zaman eksenini ezmek BEDAVA gorundu.
Test penceresinin (Nis-Tem) mevsimsel ikizinde ay ekseni varyansi 0,15298:

    **test penceresinde zaman ekseni 136 KAT guclu.**

Olculdu (gonderim dosyalari uzerinde, 158.369 soguk satir):

    ay   v27 ofset   v30 ofset   2025 ayni ay gercek
    04     +0,1219     +0,2611          +0,0408
    05     +0,1169     +0,2581          +0,0056
    06     +0,4138     +0,4363          +0,4706
    07     +0,7642     +0,6465          +0,9517

    gun-ortalamalarinin std'si  0,3003 -> 0,1802   (tam olarak x0,60 = beta)

v27'nin May->Tem rampasi zaten gercegin altindaydi (0,647 vs 0,946); buzme
onu 0,389'a indirip gercekten UZAKLASTIRDI.

Bu betik gun ortalamasini AYNEN korur: buzme yalnizca gun ICINDEKI trafolar
arasi yayilmaya uygulanir.

KUSUR 2 -- HEDEF COK KABA
-------------------------
Genel ortalama, bir tahminin cekilebilecegi en kaba hedef; trafonun ilcesini
ve kVA kademesini yok sayar. Efron-Morris (1975) James-Stein'i tam bu yonde
genellestirir: hedef ne kadar bilgiliyse buzmenin yanlilik maliyeti o kadar
kucuktur. Hedefe egitimden turetilen bir HUCRE ETKISI eklenir.

Hucre yapisi ``deney_taban_ince.py`` ile UC BLOKTA tarandi. Kazanan: ilce x
kova, EBEVEYN ``ilce`` (``kova`` degil -- ilce tek basina kovadan iyi, seyrek
hucre oraya dusmeli), 24 kova, M ~ 2000.

KURGU
-----
    r        = log1p(tahmin) - log1p(guc)
    gun_ort  = o GUNUN soguk satirlarindaki r ortalamasi   (modelden: model
               tarihi ve trendi biliyor)
    hucre    = ampirik-Bayes ilce x kova ofset ortalamasi  (egitimden)
    etki     = hucre - o gunun soguk satirlarindaki hucre ortalamasi
    taban    = gun_ort + etki
    r'       = taban + beta * (r - taban)

Gun ortalamasi TAM olarak korunur (etki gun icinde merkezlendigi icin).

OLCULDU (2026-08-23, ``deney_soguk_taban.py``, kis26 soguk, 3 tohum):

    URETIM (v30: kendi ortalamasi, beta=0,60)        1,83979
    GUN korumali, hucresiz,        beta=0,30         1,83438
    GUN + ilce x kova M=2000,      beta=0,25         1,83114   <- SECILEN
    GUN + ilce x kova M=5000,      beta=0,25         1,83079
    (kis26 ust siniri: OLS ile uydurulmus afin kalibrasyon 1,80850)

kis26'da gun korumasi neredeyse BEDAVA (kis ofseti duz) -- kazanci testte
ortaya cikar ve orada olculemez. Dogru yapisal duzeltmenin imzasi budur.

SOGUK TANIMI: test trafosunun ``tanim`` kodu ``train.csv``de HIC gecmiyor.

    python scripts/son_islem_gun.py --giris submissions/tuketim_v27_v18hedge.csv \
        --cikis submissions/tuketim_v31.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]

#: Buzme katsayisi. kis26'da 0,20-0,30 arasi duz; 0,25 secildi.
BETA = 0.25

#: kVA kademesi sayisi. Kenarlar YALNIZ egitimden turetilir ve teste AYNI
#: kenarlarla uygulanir -- iki tarafta ayri hesaplamak kovalari kaydirir.
KOVA_SAYISI = 24

#: Ampirik-Bayes onsel agirligi: n satirlik hucre n/(n+M) agirlik alir.
#: Ana etkiler (ilce, kova) icin hafif, hucre icin guclu duzlestirme.
M_ANA = 200.0
M_HUCRE = 2000.0


def _kova(guc: np.ndarray, kenar: np.ndarray) -> np.ndarray:
    lg = np.log1p(np.clip(guc, 0.0, None))
    return np.clip(np.searchsorted(kenar, lg, side="right") - 1, 0, KOVA_SAYISI - 1)


def _eb(anahtar_e: np.ndarray, ofs_e: np.ndarray, anahtar_h: np.ndarray,
        ebeveyn: np.ndarray, m_once: float) -> np.ndarray:
    """Ampirik-Bayes hucre ortalamasi: (hucre_toplami + M*ebeveyn) / (n + M)."""
    s = pd.Series(ofs_e).groupby(anahtar_e).agg(["sum", "count"])
    top = np.nan_to_num(pd.Series(s["sum"]).reindex(anahtar_h).to_numpy(dtype="float64"), nan=0.0)
    n = np.nan_to_num(pd.Series(s["count"]).reindex(anahtar_h).to_numpy(dtype="float64"), nan=0.0)
    return (top + m_once * ebeveyn) / (n + m_once)


def hucre_etkisi(tr: pd.DataFrame, hedef: pd.DataFrame) -> np.ndarray:
    """Egitimden ilce x kova ofset ortalamasi (ham, merkezlenmemis)."""
    ofs = (np.log1p(tr["tuketim"].clip(lower=0.0).to_numpy(dtype="float64"))
           - np.log1p(tr["guc"].to_numpy(dtype="float64")))
    lg_e = np.log1p(tr["guc"].to_numpy(dtype="float64"))
    kenar = np.linspace(float(lg_e.min()), float(lg_e.max()) + 1e-9, KOVA_SAYISI + 1)
    kv_e = _kova(tr["guc"].to_numpy(dtype="float64"), kenar)
    kv_h = _kova(hedef["guc"].to_numpy(dtype="float64"), kenar)
    il_e = tr["lokasyon"].astype(str).to_numpy()
    il_h = hedef["lokasyon"].astype(str).to_numpy()

    genel = np.full(len(hedef), float(ofs.mean()))
    ilce = _eb(il_e, ofs, il_h, genel, M_ANA)
    anahtar_e = pd.Series(il_e).to_numpy() + "|" + pd.Series(kv_e).astype(str).to_numpy()
    anahtar_h = pd.Series(il_h).to_numpy() + "|" + pd.Series(kv_h).astype(str).to_numpy()
    return _eb(anahtar_e, ofs, anahtar_h, ilce, M_HUCRE)


def main() -> int:
    a = argparse.ArgumentParser(description="soguk buzme -- gun ekseni korumali")
    a.add_argument("--giris", required=True)
    a.add_argument("--cikis", required=True)
    a.add_argument("--beta", type=float, default=BETA)
    a.add_argument("--hucresiz", action="store_true",
                   help="hucre etkisini kapat (yalniz gun korumasi)")
    ar = a.parse_args()

    ornek = pd.read_csv(KOK / "data/raw/sample_submission.csv", encoding="utf-8")
    te = pd.read_csv(KOK / "data/raw/test.csv", usecols=["id", "tanim", "guc", "tarih", "lokasyon"],
                     encoding="utf-8", dtype={"tanim": str})
    tr = pd.read_csv(KOK / "data/raw/train.csv", usecols=["tanim", "guc", "tuketim", "lokasyon"],
                     encoding="utf-8", dtype={"tanim": str})
    giris = pd.read_csv(KOK / ar.giris, encoding="utf-8")

    m = ornek[["id"]].merge(giris, on="id", how="left").merge(te, on="id", how="left")
    if m["tuketim"].isna().any() or m["guc"].isna().any():
        raise RuntimeError("giris dosyasi ile ornek gonderim id kumesi ortusmuyor")

    soguk = ~m["tanim"].isin(set(tr["tanim"])).to_numpy()
    log_guc = np.log1p(m["guc"].to_numpy(dtype="float64"))
    ham = m["tuketim"].to_numpy(dtype="float64")
    r = np.log1p(ham) - log_guc
    gun = m["tarih"].to_numpy()

    # --- taban: gun ortalamasi (modelden) + hucre etkisi (egitimden) ---
    gun_s = pd.Series(gun[soguk])
    gun_ort = pd.Series(r[soguk]).groupby(gun_s.to_numpy()).transform("mean").to_numpy()
    if ar.hucresiz:
        etki = np.zeros(int(soguk.sum()))
    else:
        hucre = hucre_etkisi(tr, m)[soguk]
        # GUN ICINDE merkezle -- boylece gun ortalamasi TAM korunur
        etki = hucre - pd.Series(hucre).groupby(gun_s.to_numpy()).transform("mean").to_numpy()
    taban = gun_ort + etki

    r_yeni = r.copy()
    r_yeni[soguk] = taban + ar.beta * (r[soguk] - taban)
    yeni = ham.copy()  # sicak satirlar gidis-donusumsuz, birebir kopya
    yeni[soguk] = np.clip(np.expm1(r_yeni[soguk] + log_guc[soguk]), 0.0, None)

    # --- guvenlik kapilari ---
    if not np.array_equal(yeni[~soguk], ham[~soguk]):
        raise RuntimeError("SICAK satirlar degismis olmamaliydi")
    onceki = pd.Series(r[soguk]).groupby(gun_s.to_numpy()).mean()
    sonraki = pd.Series(r_yeni[soguk]).groupby(gun_s.to_numpy()).mean()
    sapma = float(np.abs(onceki - sonraki).max())
    if sapma > 1e-9:
        raise RuntimeError(f"gun ortalamasi korunmadi: en buyuk sapma {sapma:.3e}")
    if not np.isfinite(yeni).all() or (yeni < 0).any():
        raise RuntimeError("cikti NaN/sonsuz/negatif iceriyor")

    n_s = int(soguk.sum())
    hucre_ad = "KAPALI" if ar.hucresiz else f"ilce x kova (M={M_HUCRE:.0f})"
    trafo_sayisi = int(m.loc[soguk, "tanim"].nunique())
    print(f"  beta {ar.beta:.2f}  hucre {hucre_ad}")
    print(f"  soguk {n_s:,} satir (%{100 * n_s / len(m):.2f}), {trafo_sayisi:,} trafo")
    print(f"  gun ortalamasi korundu (en buyuk sapma {sapma:.2e})")
    print(f"  soguk ofset std   {r[soguk].std():.5f} -> {r_yeni[soguk].std():.5f}")
    print(f"  gun-ort std       {onceki.std():.5f} -> {sonraki.std():.5f}   (DEGISMEMELI)")
    ay = pd.to_datetime(m["tarih"]).dt.month.to_numpy()
    print("   ay   once    sonra")
    for k in np.unique(ay):
        d = soguk & (ay == k)
        print(f"   {k:2d}  {r[d].mean():+.4f}  {r_yeni[d].mean():+.4f}")

    cikti = pd.DataFrame({"id": m["id"], "tuketim": yeni})
    yol = KOK / ar.cikis
    yol.parent.mkdir(parents=True, exist_ok=True)
    cikti.to_csv(yol, index=False, encoding="utf-8")
    print(f"  yazildi: {ar.cikis}  ({len(cikti):,} satir)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
