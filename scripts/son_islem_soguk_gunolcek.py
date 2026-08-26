"""SOGUK GUN EKSENI OLCEKLEMESI -- soguk satirlarin gun rampasi AZ yayilmis.

BULGU (docs/45 tik 1)
---------------------
``son_islem_gunolcek.py`` gun ekseni genligini duzeltir ve LB'de DOGRULANMIS
tek yapisal kazanctir (1,01750 -> 1,01591). Ama yalnizca SICAK satirlara
uygulaniyor: v50 -> v55 -> v66 zincirinde SOGUK satirlarin degisen sayisi
TAM SIFIR (0/158.369, ``scripts/h8_soguk_gun_ekseni.py``).

Soguk satirlar test satirlarinin %22'si ama MSE'nin %63'u:
    RMSE_soguk  1,713      RMSE_sicak  0,700

MEVSIMSEL IKIZ OLCUMU
---------------------
``data/interim/gun_ekseni/yaz25_*`` = 2025 Nis-Tem'de DOGMUS 678 trafo
(%100). Model RMSE 1,58; test soguk tarafi 1,71 -- gercek ikiz.

    panel                          c* (etiketli)   dMSE      tohum
    yaz25 T0 ham                       2,127      -0,0556    6/6  t=-28,2
    yaz25 T3 (ilk 7 gun atildi,
              >=60 gunluk trafo)       2,781      -0,0675    6/6  t=-39,1
    guz25 T3                           0,673      -0,0041    3/3  t= -5,9

Temizlik bulguyu GUCLENDIRDI -> dogum/giris eseri DEGIL. Isaret bloklar
arasi ters, cunku gun ekseni genligi MEVSIMSELDIR -- sicak tarafta da ayni
desen olculmustu (yaz25 2,65 | guz25 0,75 | kis26 0,70).

Mevsime bagli bir parametre icin "iki ortusmeyen kesme ayni isareti versin"
kapisi MANTIKEN saglanamaz. O sinifta gecerli kapi:
    (a) capa ETIKETSIZ olacak (test etiketi kullanilmaz -- kural 5)
    (b) capa formulu ETIKETLI optimumu URETEBILDIGI ISPATLANMIS olacak

(b) SINANDI (``scripts/h8g_soguk_capa.py`` Adim 1):
    panel     c_capa   c_etiketli   oran
    yaz25      2,885      2,781     1,037   <- TESTIN MEVSIMI, %4 hata
    guz25      0,938      0,673     1,394
Sapma iki blokta da AYNI YONDE (capa >= etiketli) -> buzme guvenli taraf.

CAPA -- test etiketi KULLANILMADI
---------------------------------
    sigma_gercek  0,4537   2025 Nis-Tem'de dogmus trafolar, GERCEK (train)
    sigma_model   0,1645   2026 test soguk, sampiyon tahmini
                           1.823 trafo / 139.166 satir
    oran          2,7580   korelasyon +0,9122 (110 ortak gun-of-year)
    c_capa        2,5159   yaz25 kalibrasyonuyla (1,037) duzeltilmis 2,43

Test soguk rejimi ikizin neredeyse birebir kopyasi (ikizde sigma 0,4516/0,14,
korelasyon 0,90). Bu, projede gorulmus EN GUCLU tasinabilirlik kaniti.

SECILEN c = 2,20
----------------
capa 2,516 -> kalibrasyon 2,43 -> yogunlasma riski icin %15 buzme.
Kuadratik bu bantta DUZ: c in [2,0; 2,5] boyunca panel dMSE -0,052..-0,056.

    beklenen test dMSE = p_soguk * dMSE_panel = 0,22159 * (-0,0553) = -0,01225

KIRPMA TABLOSU (kural 1), yaz25 T0, c=2,1:
    K       0        1        5       10       25       50
    dMSE  -0,0553  -0,0493  -0,0388  -0,0300  -0,0081  +0,0218
    kazanan trafo 446/678 (%65,8)
Isaret genis, buyukluk yogun. K=25'te (%3,7 kirpma) hala iyilestiriyor
(t=-3,10). "4 trafo" patolojisi DEGIL -- genlik duzeltmelerinin beklenen imzasi.

RISK: c*=1 olsaydi (etki yok) maliyet ~ +0,014 test MSE. Capa kaniti
(olculen oran 2,758 + korelasyon 0,912) bunu cok dusuk olasilikli kiliyor.

NASIL UYGULANIR -- SEVIYE SIZDIRMAZ
-----------------------------------
Gun profili ``b_gun`` TEMIZ alt kumeden (ilk 7 gun atilmis, >=60 gunluk
trafolar) cikarilir; boylece DOGUM/OLAY dususleri profile girmez ve
buyutulmez. Sonra TUM soguk satirlara gun duzeyinde sabit olarak yazilir:

    log1p(yeni) = log1p(eski) + (c - 1) * b_gun[gun]

``b_gun`` ortalamasi tanim geregi 0 -> GENEL SEVIYE DEGISMEZ. Trafo ekseni
ve gun ici yapi da degismez; yalnizca gun ekseni GENLIGI.

Sicak satirlara DOKUNULMAZ (onlarin c*=1,335'i zaten uygulandi).

    uv run python scripts/son_islem_soguk_gunolcek.py \
        --giris submissions/tuketim_v67_c1335_olay.csv \
        --cikis submissions/tuketim_v71_soguk_gun.csv [--c 2.20]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
C_VARSAYILAN = 2.20
MIN_YAS, MIN_GUN = 7, 60  # temiz profil alt kumesi
P_SOGUK = 0.22159
BEKLENEN_SATIR = 714688


def iki_yonlu(
    v: np.ndarray, bi: np.ndarray, gi: np.ndarray, nb: int, ng: int, tur: int = 400
) -> np.ndarray:
    """Iki yonlu sabit etki; GUN bilesenini dondurur (ortalamasi 0)."""
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
    a = argparse.ArgumentParser(description="soguk gun ekseni olceklemesi")
    a.add_argument("--giris", required=True)
    a.add_argument("--cikis", required=True)
    a.add_argument("--c", type=float, default=C_VARSAYILAN)
    ar = a.parse_args()

    giris = KOK / ar.giris if not Path(ar.giris).is_absolute() else Path(ar.giris)
    cikis = KOK / ar.cikis if not Path(ar.cikis).is_absolute() else Path(ar.cikis)

    te = pd.read_csv(
        KOK / "data/raw/test.csv",
        usecols=["id", "tanim", "tarih"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
    )
    d = pd.read_csv(giris)
    if len(d) != BEKLENEN_SATIR:
        raise SystemExit(f"giris {len(d)} satir, {BEKLENEN_SATIR} bekleniyordu")
    if not (d["id"].values == te["id"].values).all():
        raise SystemExit("giris id sirasi test.csv ile ayni degil")

    tr_tanim = set(
        pd.read_csv(KOK / "data/raw/train.csv", usecols=["tanim"], dtype={"tanim": str})[
            "tanim"
        ].unique()
    )
    soguk = (~te["tanim"].isin(tr_tanim)).to_numpy()
    print(
        f"giris {giris.name}   soguk {soguk.sum():,} / {len(d):,} "
        f"({soguk.mean():.4f}; beklenen {P_SOGUK})"
    )

    lg = np.log1p(np.clip(d["tuketim"].to_numpy(dtype="float64"), 0, None))

    # --- TEMIZ alt kume uzerinde gun profili
    sc = te.loc[soguk].copy()
    sc["lg"] = lg[soguk]
    ilk = sc.groupby("tanim")["tarih"].transform("min")
    yas = (sc["tarih"] - ilk).dt.days.to_numpy()
    say = sc.groupby("tanim")["tanim"].transform("size").to_numpy()
    temiz = (yas >= MIN_YAS) & (say >= MIN_GUN)
    t = sc.loc[temiz]
    print(
        f"temiz profil alt kumesi: {temiz.sum():,} satir, "
        f"{t.tanim.nunique():,} trafo, {t.tarih.nunique()} gun"
    )

    bi, _ = pd.factorize(t["tanim"])
    gi, gun = pd.factorize(t["tarih"])
    b = iki_yonlu(t["lg"].to_numpy(dtype="float64"), bi, gi, int(bi.max()) + 1, int(gi.max()) + 1)
    profil = pd.Series(b, index=pd.Index(gun, name="tarih")).sort_index()

    # --- TUM soguk satirlara uygula
    ek = sc["tarih"].map(profil).to_numpy(dtype="float64")
    if np.isnan(ek).any():
        eksik = int(np.isnan(ek).sum())
        print(f"  UYARI: {eksik:,} soguk satirin gunu temiz profilde yok -> 0 yazildi")
        ek = np.nan_to_num(ek)

    # SATIR-AGIRLIKLI MERKEZLEME -- mudahale SEVIYEYI degistirmesin (h8h).
    # Profil GUNLER uzerinde ortalamasi 0'dir, ama soguk satirlar gunlere
    # ESIT dagilmaz (2026-05-11'de 1.326 trafoluk toplu katilim var), bu
    # yuzden satir ortalamasi SIFIR DEGIL. Merkezlemeden uygulanirsa
    # mudahale gizlice +0,0714'luk bir SEVIYE kaymasi tasir ve LB probuyla
    # cozulen b_soguk ile karisir. h8h olctu: kazancin %95,5'i GENLIK,
    # %4,5'i seviye -- ve seviye sizintisi temizlenince kazanc BUYUDU
    # (-0,0556 -> -0,0792). Seviye AYRI knob, AYRI kalir.
    ek = ek - float(ek.mean())
    print(
        f"gun profili (satir-merkezli): std {ek.std():.4f}  "
        f"ort {ek.mean():+.2e}  min {ek.min():+.3f}  max {ek.max():+.3f}"
    )

    yeni = lg.copy()
    yeni[soguk] = lg[soguk] + (ar.c - 1.0) * ek
    cikti = np.expm1(yeni)
    cikti = np.clip(cikti, 0.0, None)

    # --- KAPILAR
    fark = yeni - lg
    print(f"\nc = {ar.c}")
    print(
        f"  degisen satir      {int((np.abs(fark) > 1e-12).sum()):,} "
        f"(soguk {int((np.abs(fark[soguk]) > 1e-12).sum()):,} / "
        f"sicak {int((np.abs(fark[~soguk]) > 1e-12).sum()):,})"
    )
    print(f"  SICAK dokunulmadi  {bool((np.abs(fark[~soguk]) < 1e-12).all())}")
    print(f"  soguk ort kayma    {fark[soguk].mean():+.6f}   (0 olmali -- seviye sizmaz)")
    print(f"  soguk std kayma    {fark[soguk].std():.6f}")
    print(f"  soguk maxabs kayma {np.abs(fark[soguk]).max():.6f}")
    print(
        f"  genel ort log1p    {lg.mean():.6f} -> {yeni.mean():.6f} "
        f"(fark {yeni.mean() - lg.mean():+.2e})"
    )
    print(
        f"  NaN {int(np.isnan(cikti).sum())}  negatif {int((cikti < 0).sum())}  "
        f"sifir {int((cikti == 0).sum())}"
    )

    if abs(float(fark[soguk].mean())) > 1e-3:
        raise SystemExit("seviye sizintisi: soguk ortalama kayma 0 degil")
    if not bool((np.abs(fark[~soguk]) < 1e-12).all()):
        raise SystemExit("sicak satirlar degismis")

    cikis.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": d["id"], "tuketim": cikti}).to_csv(cikis, index=False)
    print(f"\nyazildi: {cikis}")
    print(f"beklenen test dMSE ~ {P_SOGUK * -0.0553:+.5f} (yaz25 ikiz paneli, c=2,1 olcumu)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
