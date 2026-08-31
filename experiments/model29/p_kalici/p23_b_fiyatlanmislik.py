"""p23-B: PARTI ZATEN FIYATLANMIS MI?

Uretim adayi p21_harman311_olu50.csv'nin SOGUK satirlarinda:
parti-soguk tahmin seviyesi ile parti-olmayan soguk tahmin seviyesini
AYNI kVA x ilce hucresi icinde karsilastir.

Mevsim karisikligini onlemek icin iki gorunum:
  (1) tarih >= 2026-05-11 kisiti (iki grup ayni pencere)
  (2) kVA x ilce x AY eslesmesi (duyarlilik)
Ince hucreler kirpilir: her iki grupta da n >= 20 sart.

Ek: p06_test_soguk_aile.npy (cat/xgb/lgbm) ile ayni olcum -- soguk uzman
aileleri parti trafolarini nasil konumlandiriyor.
"""

import json
import os

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
PK = os.path.join(KOK, "experiments/model29/p_kalici")
AC = os.path.join(PK, "aday_csv")
JSON_YOL = os.path.join(PK, "p23_parti.json")

test = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"), dtype={"tanim": str})
p21 = pd.read_csv(os.path.join(AC, "p21_harman311_olu50.csv"))
assert (p21["id"].to_numpy() == test["id"].to_numpy()).all()

soguk = np.load(os.path.join(AC, "p06_test_soguk_maske.npy"))
m_parti = np.load(os.path.join(AC, "p23_parti_soguk_maske.npy"))
assert not (m_parti & ~soguk).any()

test["ilce"] = test["lokasyon"].str.split(">").str[-1]
test["ay"] = test["tarih"].str.slice(0, 7)
test["log_p21"] = np.log1p(p21["tuketim"].to_numpy())

# soguk aile tahminleri (log uzayinda) -- soguk satirlarin test sirasinda
aile = np.load(os.path.join(AC, "p06_test_soguk_aile.npy"))
assert aile.shape == (int(soguk.sum()), 3)
for j, ad in enumerate(["cat", "xgb", "lgbm"]):
    kol = np.full(len(test), np.nan)
    kol[soguk] = aile[:, j]
    test[f"log_{ad}"] = kol

S = test[soguk].copy()
S["parti"] = m_parti[soguk]


def hucre_farki(df, hucre_kolonlari, deger, n_esik=20):
    """Hucre ici (parti - diger) ortalama log farki; agirlik = hucredeki parti satiri."""
    g = df.groupby(hucre_kolonlari + ["parti"], observed=True)[deger].agg(["mean", "size"])
    g = g.unstack("parti")
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g = g.dropna()
    if len(g) == 0:
        return None
    g = g[(g["size_True"] >= n_esik) & (g["size_False"] >= n_esik)]
    if len(g) == 0:
        return None
    fark = g["mean_True"] - g["mean_False"]
    w = g["size_True"]
    return {
        "hucre_sayisi": int(len(g)),
        "kapsanan_parti_satiri": int(w.sum()),
        "agirlikli_fark": round(float((fark * w).sum() / w.sum()), 4),
        "medyan_fark": round(float(fark.median()), 4),
    }


sonuc = {}

# (1) ortak pencere: tarih >= 2026-05-11
P = S[S["tarih"] >= "2026-05-11"]
sonuc["pencere_0511_sonrasi"] = {
    "parti_satir": int(P["parti"].sum()),
    "diger_soguk_satir": int((~P["parti"]).sum()),
    "p21_kva_ilce": hucre_farki(P, ["guc", "ilce"], "log_p21"),
    "p21_yalniz_kva": hucre_farki(P, ["guc"], "log_p21"),
}
for ad in ["cat", "xgb", "lgbm"]:
    sonuc["pencere_0511_sonrasi"][f"{ad}_kva_ilce"] = hucre_farki(P, ["guc", "ilce"], f"log_{ad}")

# (2) ay eslesmeli (tum soguk satirlar, hucre = kVA x ilce x ay)
sonuc["ay_eslesmeli"] = {
    "p21_kva_ilce_ay": hucre_farki(S, ["guc", "ilce", "ay"], "log_p21"),
    "p21_kva_ay": hucre_farki(S, ["guc", "ay"], "log_p21"),
}

# (3) kVA bazinda ayrinti (iddiadaki 400/1000/1250 ile kiyas; pencere 05-11+)
kva_detay = {}
for kva in [250, 400, 630, 800, 1000, 1250, 1600]:
    alt = P[P["guc"] == kva]
    if len(alt) == 0:
        continue
    r = hucre_farki(alt, ["ilce"], "log_p21")
    if r is not None:
        kva_detay[str(kva)] = r["agirlikli_fark"]
sonuc["p21_kva_detay_ilce_ici"] = kva_detay
sonuc["iddia_gercek_tuketim_farki"] = {"1000": -0.381, "400": -0.443, "1250": -0.497}

# ham seviyeler (kontrol icin)
sonuc["ham_seviye_0511_sonrasi"] = {
    "parti_ort_log_p21": round(float(P.loc[P["parti"], "log_p21"].mean()), 4),
    "diger_ort_log_p21": round(float(P.loc[~P["parti"], "log_p21"].mean()), 4),
}

R = {}
if os.path.exists(JSON_YOL):
    with open(JSON_YOL, encoding="utf-8") as fh:
        R = json.load(fh)
R["adim2_fiyatlanmislik"] = sonuc
with open(JSON_YOL, "w", encoding="utf-8") as fh:
    json.dump(R, fh, indent=1, ensure_ascii=False)

print(json.dumps(sonuc, indent=1, ensure_ascii=False))
