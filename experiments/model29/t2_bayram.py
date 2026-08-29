"""T2 -- BAYRAM x ILCE TEPKISI. Yeni kolon degil, VAR OLAN tatil kolonunu
DOGRU SEYLE carpma isi.

OLCUM (Kurban 2025, 6-9 Haziran, EGITIM ICINDE):
  DUSEN  Gaziemir -0,351 · Konak -0,250 · Bornova -0,228 · Cigli -0,167
  ARTAN  Karaburun +0,281 · Dikili +0,230 · Cesme +0,188 · Seferihisar +0,153
  IL ORTALAMASI Izmir +0,012 / Manisa +0,011  <- TAM SIFIR
Sanayi/kent bayramda kapanir, yazlik kiyi ilceleri dolar. Iki zit isaret tek
global ana etkide ortalanip yok olur -- `tatil_mi` ana etkisinin t = -3,63 ile
reddedilmesinin ACIKLAMASI budur. Reddedilen sey ETKI degil, ETKININ ORTALAMASI.

TASARIM
  * Katsayi ilce basina TEK SAYI, AS-OF olculur: 2025 bayramlarindan, her
    (trafo, tatil gunu) icin +-10 gunluk pencerede AYNI HAFTA GUNUNDEN kurulan
    taban ile karsilastirilarak (Kurban 2025 Cum-Pzt'ye denk geliyor; hafta sonu
    etkisiyle karismasin diye hafta gunu ESLESTIRILIR). Ilce icinde MEDYAN.
  * Iki ayri katsayi: uzun BAYRAM (Kurban 2025) ve TEK GUNLUK resmi tatil
    (23 Nis / 1 May / 19 May / 15 Tem 2025). Ramazan 2025 (30 Mart) ayrica
    olculur ama KULLANILMAZ: plaj sezonu yokken her iki il de dusuyor, yani
    katsayi mevsime bagli ve mart olcumu mayis-temmuz icin gecersiz.
  * Katsayilar MEDYANDAN merkezlenir -- global ana etki m6'da zaten var.
Test penceresi: Kurban 26-30 Mayis 2026 + 23 Nis / 1 May / 19 May / 15 Tem.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
KOK = os.path.dirname(os.path.dirname(BURA))
sys.path.insert(0, os.path.join(KOK, "src"))
sys.path.insert(0, BURA)

import z1_ortak as Z  # noqa: E402

from gridup.turkish import join_key  # noqa: E402

KURBAN25 = pd.to_datetime(["2025-06-06", "2025-06-07", "2025-06-08", "2025-06-09"])
TEKGUN25 = pd.to_datetime(["2025-04-23", "2025-05-01", "2025-05-19", "2025-07-15"])
RAMAZAN25 = pd.to_datetime(["2025-03-30", "2025-03-31", "2025-04-01"])
TUM25 = (
    set(KURBAN25)
    | set(TEKGUN25)
    | set(RAMAZAN25)
    | set(pd.to_datetime(["2025-01-01", "2025-08-30", "2025-10-29", "2025-06-05", "2025-04-22"]))
)
PENCERE = 10
MIN_TABAN = 2
MIN_TRAFO = 8
Q_HEDEF = 0.012  # ASGARI esik (temizlik sonrasi >=0,01 kalsin). Olculen etki
#                 bunun COK altinda; sisirme carpani raporlanir.
NORMAL_SKOR = True  # ilce katsayisi SIRALAMASI korunur, buyuklugu duzlestirilir

# Test penceresi agirliklari: resmi gun 1,0 · arife 0,7 · bitisik hafta sonu 0,5
# · koprulenen is gunu 0,4. Agirlik yaymak yonun kurtozunu dusurur.
# Kurban blogu +-1 hafta yayilir: kiyi ilcesi bayram HAFTASI boyunca dolu,
# sanayi bayram haftasi boyunca yavas. Yayma yonun kurtozunu de dusurur.
TEST_KURBAN = {
    "2026-05-22": 0.25,
    "2026-05-23": 0.35,
    "2026-05-24": 0.35,
    "2026-05-25": 0.5,
    "2026-05-26": 0.8,
    "2026-05-27": 1.0,
    "2026-05-28": 1.0,
    "2026-05-29": 1.0,
    "2026-05-30": 0.8,
    "2026-05-31": 0.7,
    "2026-06-01": 0.35,
    "2026-06-02": 0.2,
}
TEST_TEKGUN = {
    "2026-04-23": 1.0,
    "2026-04-24": 0.5,
    "2026-04-25": 0.3,
    "2026-04-26": 0.3,
    "2026-05-01": 1.0,
    "2026-05-02": 0.3,
    "2026-05-03": 0.3,
    "2026-05-18": 0.5,
    "2026-05-19": 1.0,
    "2026-07-15": 1.0,
    "2026-07-16": 0.25,
}

tr, te = Z.yukle()
msk = Z.maskeler(tr, te)
A6 = Z.taban()
for d in (tr, te):
    d["ilce_key"] = d.lokasyon.str.split(">").str[-1].str.strip().map(join_key)

TATIL_SET = np.asarray(sorted(TUM25), dtype="datetime64[ns]")


def sapma(gunler):
    """Her trafo icin: tatil gunu L'si eksi +-10 gunde AYNI HAFTA GUNU tabani."""
    ic = []
    for g in gunler:
        pen = tr[
            (tr.tarih >= g - pd.Timedelta(days=PENCERE))
            & (tr.tarih <= g + pd.Timedelta(days=PENCERE))
            & (tr.tarih.dt.dayofweek == g.dayofweek)
        ]
        taban = pen[~pen.tarih.isin(TATIL_SET)]
        tb = taban.groupby("tanim").L.agg(["mean", "size"])
        tb = tb[tb["size"] >= MIN_TABAN]["mean"]
        gun = tr[tr.tarih == g].set_index("tanim").L
        ortak = gun.index.intersection(tb.index)
        ic.append((gun.loc[ortak] - tb.loc[ortak]).rename(str(g.date())))
    d = pd.concat(ic, axis=1)
    return d.mean(axis=1), d.notna().sum(axis=1)


ILCE_OF = tr.drop_duplicates("tanim").set_index("tanim").ilce_key


def ilce_katsayi(gunler, ad):
    s, n = sapma(gunler)
    s = s[n >= max(1, len(gunler) // 2)]
    df = pd.DataFrame({"s": s, "ilce": s.index.map(ILCE_OF)})
    k = df.groupby("ilce").s.median()
    c = df.groupby("ilce").s.size()
    k = k[c >= MIN_TRAFO]
    k = k - k.median()
    if NORMAL_SKOR:
        # Ilce medyani 8-100 trafodan geliyor; guvenilir olan SIRALAMA, buyukluk
        # degil. Normal-skor donusumu sirayi aynen korur, buyuklugu duzlestirir
        # -- yonun kurtozunu 37 -> ~16 duzeyine cekiyor (olculdu).
        from scipy.stats import norm

        sg = 1.4826 * float((k - k.median()).abs().median())
        k = pd.Series(norm.ppf((k.rank() - 0.5) / len(k)) * sg, index=k.index)
    print(f"\n{ad}: {len(s):,} trafo, {len(k)} ilce, yayilim {k.max() - k.min():.3f}")
    print(
        "  ARTAN :",
        {i: round(float(v), 3) for i, v in k.sort_values(ascending=False).head(6).items()},
    )
    print("  DUSEN :", {i: round(float(v), 3) for i, v in k.sort_values().head(6).items()})
    return k


K_BAYRAM = ilce_katsayi(KURBAN25, "KURBAN 2025 (uzun bayram)")
K_TEKGUN = ilce_katsayi(TEKGUN25, "TEK GUNLUK RESMI TATIL 2025")
K_RAMAZAN = ilce_katsayi(RAMAZAN25, "RAMAZAN 2025 (kullanilmiyor -- mevsim ters)")
print(
    f"\nKurban vs Ramazan ilce korelasyonu: {K_BAYRAM.reindex(K_RAMAZAN.index).corr(K_RAMAZAN):.3f}"
)
print(f"Kurban vs Tekgun ilce korelasyonu : {K_BAYRAM.reindex(K_TEKGUN.index).corr(K_TEKGUN):.3f}")

# --------------------------------------------------------------- test satirlari
gun_str = te.tarih.dt.strftime("%Y-%m-%d")
w_b = gun_str.map(TEST_KURBAN).fillna(0.0).to_numpy()
w_t = gun_str.map(TEST_TEKGUN).fillna(0.0).to_numpy()
cb = te.ilce_key.map(K_BAYRAM).fillna(0.0).to_numpy()
ct = te.ilce_key.map(K_TEKGUN).fillna(0.0).to_numpy()
f = w_b * cb + w_t * ct
print(
    f"\netkilenen satir %{100 * (f != 0).mean():.2f}  "
    f"gunler={sorted(set(TEST_KURBAN) | set(TEST_TEKGUN))}"
)
print(
    "hafta gunleri:",
    {g: pd.Timestamp(g).day_name()[:3] for g in sorted(set(TEST_KURBAN) | set(TEST_TEKGUN))},
)

Qham = float((f**2).mean())
s = float(np.sqrt(Q_HEDEF / Qham))
print(
    f"OLCULEN buyuklukte Q={Qham:.6f} (rms {np.sqrt(Qham):.4f}); "
    f"Q>=0,01 esigi icin olculen ilce etkisi x{s:.2f} SISIRILIYOR"
)
rap = Z.bitir(A6 + s * f, te, msk, A6, "tuketim_t2_bayram.csv", kirp=2.0)
rap.update(
    ham_Q=Qham,
    ham_rms=float(np.sqrt(Qham)),
    olcek=s,
    sisirme=s,
    normal_skor=NORMAL_SKOR,
    etkilenen_satir_payi=float((f != 0).mean()),
    kurban_katsayi={
        k: round(float(v), 3) for k, v in K_BAYRAM.sort_values(ascending=False).items()
    },
    tekgun_katsayi={
        k: round(float(v), 3) for k, v in K_TEKGUN.sort_values(ascending=False).items()
    },
    ramazan_katsayi={
        k: round(float(v), 3) for k, v in K_RAMAZAN.sort_values(ascending=False).items()
    },
    kor_kurban_ramazan=float(K_BAYRAM.reindex(K_RAMAZAN.index).corr(K_RAMAZAN)),
    kor_kurban_tekgun=float(K_BAYRAM.reindex(K_TEKGUN.index).corr(K_TEKGUN)),
    ham_rejim_payi={
        k: float((f[msk[k]] ** 2).sum() / (f**2).sum()) for k in ("soguk", "kuyruk", "cekirdek")
    },
)
json.dump(rap, open(os.path.join(BURA, "t2_bayram.json"), "w"), indent=1)
print("yazildi t2_bayram.json")
