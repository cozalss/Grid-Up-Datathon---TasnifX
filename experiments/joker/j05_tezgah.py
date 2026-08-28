"""J5 - birlesik uc blok tezgahi (sicak + soguk, uretim son islemiyle).

Onbelleklerden okur, FIT YOK. Cikti: her blok icin
  ly  (gercek log1p),  lp  (uretim tahmini log1p),  ve tani kolonlari.

Dogrulama capasi: kis26 soguk RMSLE ~= 1.82141 (butunluk_son_islem.py BEKLENEN).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(r"C:\Users\Cem\Desktop\Datahon_Laptop\Grid-Up-Datathon---TasnifX")
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

import son_islem_gun as si  # noqa: E402

BLOKLAR = ("yaz25", "guz25", "kis26")
AILELER = ("cat", "xgb", "lgbm")
AGIRLIK = (3.0, 1.0, 1.0)
TOHUMLAR = (1000, 1001, 1002)
IC = KOK / "data" / "interim" / "deney"

KOLONLAR = [
    "_blok",
    "soguk_mu",
    "tanim",
    "guc",
    "tarih",
    "tuketim",
    "lokasyon",
    "ufuk_gun",
    "t_kuyruk_sifir",
    "t_olu_mu",
    "t_gun_sayisi",
    "t_log_ort",
]


def _uygula(r, hucre, gun, ay, a, b, m_gun):
    """son_islem_gun.main cebirinin birebir kopyasi (butunluk_son_islem.py'den)."""

    def gruplu(v, anahtar):
        return pd.Series(v).groupby(anahtar).transform("mean").to_numpy()

    n_gun = pd.Series(gun).groupby(gun).transform("size").to_numpy().astype("float64")
    w = n_gun / (n_gun + m_gun) if m_gun > 0 else np.ones_like(n_gun)
    seviye = w * gruplu(r, gun) + (1.0 - w) * gruplu(r, ay)
    seviye = seviye - gruplu(seviye, ay) + gruplu(r, ay)
    h_ref = w * gruplu(hucre, gun) + (1.0 - w) * gruplu(hucre, ay)
    etki = hucre - h_ref
    etki = etki - gruplu(etki, ay)
    return seviye + a * etki + b * (r - seviye)


def bloklari_kur(bloklar=BLOKLAR):
    eg = pd.read_parquet(IC / "egitim.parquet", columns=KOLONLAR)
    ham = pd.read_csv(KOK / "data/raw/train.csv", encoding="utf-8", dtype={"tanim": str})
    ham["t"] = pd.to_datetime(ham["tarih"])
    zs = np.load(IC / "sicak_tahmin.npz")
    pay = sum(AGIRLIK)
    cikti = {}
    for b in bloklar:
        dog = eg[eg["_blok"] == b].reset_index(drop=True)
        soguk = (dog["soguk_mu"] == 1).to_numpy()
        # --- SICAK ---
        tohum_log = [
            sum(AGIRLIK[i] * zs[f"{b}_{t}_{a}"] for i, a in enumerate(AILELER)) / pay
            for t in TOHUMLAR
        ]
        lp_sicak = np.mean(tohum_log, axis=0)
        # --- SOGUK ---
        zc = np.load(IC / f"soguk_tahmin_{b}.npz")
        dgc = dog[soguk]
        log_guc = np.log1p(dgc["guc"].to_numpy(dtype="float64"))
        gun = pd.to_datetime(dgc["tarih"]).to_numpy()
        ay = pd.to_datetime(dgc["tarih"]).dt.to_period("M").astype(str).to_numpy()
        blok_bas = pd.Timestamp(gun.min())
        kaynak = ham[(ham["t"] >= si.TABLO_BASLANGIC) & (ham["t"] < blok_bas)]
        hedef = pd.DataFrame({"guc": dgc["guc"].to_numpy(), "lokasyon": dgc["lokasyon"].to_numpy()})
        hucre = si.hucre_etkisi(kaynak, hedef)
        soguk_tohum = [t for t in TOHUMLAR if f"{t}_cat" in zc]
        lp_c = []
        for t in soguk_tohum:
            r = zc[f"{t}_cat"] - log_guc
            lp_c.append(_uygula(r, hucre, gun, ay, si.A_HUCRE, si.B_MODEL, si.M_GUN) + log_guc)
        lp_soguk = np.mean(lp_c, axis=0)
        # --- BIRLESTIR ---
        lp = np.empty(len(dog), dtype="float64")
        lp[~soguk] = lp_sicak
        lp[soguk] = lp_soguk
        dog["lp"] = lp
        dog["soguk"] = soguk
        dog["ly"] = np.log1p(dog["tuketim"].to_numpy(dtype="float64").clip(min=0.0))
        cikti[b] = dog
    return cikti


def mse(d, lp=None):
    """np.clip(np.expm1(.),0,None) kirpmasini ICEREN olcut."""
    v = d["lp"].to_numpy() if lp is None else lp
    tahmin = np.clip(np.expm1(v), 0.0, None)
    e = np.log1p(tahmin) - d["ly"].to_numpy()
    return float((e * e).mean())


if __name__ == "__main__":
    bl = bloklari_kur()
    print(
        f"{'blok':8}{'n':>9}{'soguk':>8}{'MSE':>10}{'RMSLE':>9}{'sicakMSE':>10}{'sogukRMSLE':>12}"
    )
    for b, d in bl.items():
        m = mse(d)
        s = d["soguk"].to_numpy()
        e = np.log1p(np.clip(np.expm1(d["lp"].to_numpy()), 0, None)) - d["ly"].to_numpy()
        print(
            f"{b:8}{len(d):9,}{int(s.sum()):8,}{m:10.5f}{np.sqrt(m):9.5f}"
            f"{(e[~s] ** 2).mean():10.5f}{np.sqrt((e[s] ** 2).mean()):12.5f}"
        )
    print("\ncapa: kis26 soguk RMSLE beklenen ~1.82141 (butunluk_son_islem.py)")
