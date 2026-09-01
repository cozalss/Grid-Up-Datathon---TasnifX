"""p32 ORTAK: ezberden bagimsiz son-islem katmanlari icin ortak tezgah.

TEZGAH SECIMI
-------------
p24_b_olc.veri_kur() kullanilir: SICAK taraf URETIM harmani (cat 3, xgb 1,
lgbm 1, sinir_agi 1.4), soguk taraf SABIT uretim sogugu, kohort agirligi
test (p_gun_sayisi x kVA x ufuk) hucre dagilimina gore. Bu, p02_duzeltme'nin
esit-harman tezgahindan farklidir ve MEMORY'deki
"CV tezgahi uretim harmanini olcmuyor" uyarisina uyar.

GECMIS OZNITELIKLERI
--------------------
Her blok icin kesim = blogun ilk gunu; test icin kesim = 2026-04-01.
Kesimden ONCEKI ham gunluk kayitlardan trafo basina:
  - kuyruk_sifir_serisi: kesimden geriye kesintisiz tuketim<=0 gun sayisi
    (KAYIT bazinda, takvim bosluklari atlanir)
  - sifir_orani: tum gecmiste tuketim<=0 orani
  - gecmis_gun: kayit sayisi
  - son_W_max: son W kayitta maks tuketim

UYARI: train.csv 2025-01-01'de basliyor. Blok basina GECMIS UZUNLUGU:
  yaz25 (kesim 2025-04-01)  ~90 gun
  guz25 (kesim 2025-08-01) ~212 gun
  kis26 (kesim 2025-12-01) ~334 gun
  TEST  (kesim 2026-04-01) ~455 gun
Yani W=365 penceresi YALNIZ test'te olculebilir; yaz25'te W>90 anlamsiz.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

KOK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BURA = os.path.dirname(os.path.abspath(__file__))
PK = os.path.join(BURA, "p_kalici")
AC = os.path.join(PK, "aday_csv")

BLOKLAR = ("yaz25", "guz25", "kis26")
TOHUMLAR = (1000, 1001, 1002)
KESIM = {
    "yaz25": "2025-04-01",
    "guz25": "2025-08-01",
    "kis26": "2025-12-01",
    "TEST": "2026-04-01",
}


def _ham() -> pd.DataFrame:
    tr = pd.read_csv(
        os.path.join(KOK, "data", "raw", "train.csv"), usecols=["tanim", "tarih", "tuketim"]
    )
    tr["tarih"] = pd.to_datetime(tr["tarih"])
    tr["tanim"] = tr["tanim"].astype(str)
    return tr.sort_values(["tanim", "tarih"], kind="mergesort").reset_index(drop=True)


def gecmis_ozet(pencereler: tuple[int, ...]) -> dict[str, pd.DataFrame]:
    """kesim adi -> trafo indeksli DataFrame (sifir_orani, gecmis_gun,
    kuyruk_serisi, sonW_max_<W> ...)."""
    tr = _ham()
    out: dict[str, pd.DataFrame] = {}
    for ad, kes in KESIM.items():
        g = tr[tr["tarih"] < pd.Timestamp(kes)]
        z = (g["tuketim"].to_numpy("float64") <= 0).astype("int8")
        t = g["tanim"].to_numpy()
        df = pd.DataFrame({"tanim": t, "z": z, "x": g["tuketim"].to_numpy("float64")})
        gb = df.groupby("tanim", sort=True)
        res = pd.DataFrame(
            {
                "sifir_orani": gb["z"].mean(),
                "gecmis_gun": gb["z"].size().astype("float64"),
            }
        )
        # kuyruk kesintisiz sifir serisi: sondan geriye
        def _seri(s: np.ndarray) -> int:
            k = 0
            for v in s[::-1]:
                if v == 1:
                    k += 1
                else:
                    break
            return k

        res["kuyruk_serisi"] = gb["z"].apply(lambda s: _seri(s.to_numpy())).astype("float64")
        for W in pencereler:
            res[f"sonmax_{W}"] = gb["x"].apply(
                lambda s, W=W: float(np.max(s.to_numpy()[-W:])) if len(s) else np.nan
            )
        out[ad] = res
    return out


def olu_maske(
    ozet: pd.DataFrame, tanimlar: np.ndarray, W: int | str, sifir_esik: float = 0.99
) -> np.ndarray:
    """Kural: son W kayitta max<=0 VE gecmis sifir orani>=esik VE
    kuyruk kesintisiz sifir serisi>=W.  W='tum' -> tum gecmis."""
    idx = pd.Index(tanimlar)
    if W == "tum":
        seri = ozet["kuyruk_serisi"].reindex(idx).to_numpy("float64")
        gg = ozet["gecmis_gun"].reindex(idx).to_numpy("float64")
        kos_pencere = seri >= gg  # tum gecmis sifir
        kos_seri = np.ones_like(kos_pencere, dtype=bool)
    else:
        sm = ozet[f"sonmax_{W}"].reindex(idx).to_numpy("float64")
        seri = ozet["kuyruk_serisi"].reindex(idx).to_numpy("float64")
        gg = ozet["gecmis_gun"].reindex(idx).to_numpy("float64")
        kos_pencere = (sm <= 0) & (gg >= W)
        kos_seri = seri >= W
    so = ozet["sifir_orani"].reindex(idx).to_numpy("float64")
    m = kos_pencere & kos_seri & (so >= sifir_esik)
    return np.nan_to_num(m, nan=0.0).astype(bool)
