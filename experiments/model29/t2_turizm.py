"""T2 -- TURIZM kolonlari: yalnizca TAKVIMLE DIK bilesenler.

NEDEN BOYLE
-----------
Panelde SADECE IKI IL var (izmir, manisa). Bu yuzden ham il-aylik bir seri
(geceleme, doluluk, gelis) = f(il, ay) demektir ve modelin ZATEN sahip oldugu
`il` (2 seviyeli kategori) x `ay` etkilesiminin icindedir. deney_takvim'in
"dis veri aslinda takvim olcuyor" hukmu burada BIREBIR gecerlidir.

Bu yuzden ham kolonlar KULLANILMAZ. Yalnizca iki bilesen alinir:
  1) ILCE KIRILIMI -- ilce_pay (KTB yillik ilce bulteni, 1 YIL gecikmeli) ve
     onun il aylik profiliyle carpimi. 45 ilce x 4 ay = 180 hucre; `ilce`
     kategorisi x `ay` etkilesimiyle teorik olarak ortusur ama modelin 15 aylik
     egitim verisinden bunu ogrenmesi imkansizdir -- bu bir ONSEL enjeksiyonudur.
  2) YILLAR ARASI ORAN (YoY), ay ortalamasi CIKARILMIS. Demean sart, cunku
     2025-07'de KTB kapsami sicradi (rejim 2 -> 3): test penceresindeki HER YoY
     rejim sinirini asiyor. Iki ilde ortak olan kapsam sicramasi demean ile
     dusuyor; geriye izmir-manisa AYRISMASI kaliyor (2026 yazinda izmir yukari,
     manisa asagi).

SIZINTI: aylik seride lag >= 2 ay (tourism.MIN_LAG_MONTHS), yillik ilce
tablosunda lag 1 yil. Test satiri 2026-07 icin en taze aylik donem 2026-05'tir.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
KOK = os.path.dirname(os.path.dirname(BURA))
sys.path.insert(0, os.path.join(KOK, "src"))
DIS = os.path.join(KOK, "data", "external")

sys.path.insert(0, BURA)
from gridup.features.tourism import MIN_LAG_MONTHS  # noqa: E402

LAG_AY = 2
assert LAG_AY >= MIN_LAG_MONTHS

# Uzun BAYRAM tatilleri (3-4 gunluk): tatil beldesinde konut yuku firlar,
# sanayi/ticari yuk duser. Tek gunluk resmi tatillerden AYRI tutulur.
BAYRAM = set(
    pd.to_datetime(
        [
            "2025-03-30",
            "2025-03-31",
            "2025-04-01",  # Ramazan 2025
            "2025-06-06",
            "2025-06-07",
            "2025-06-08",
            "2025-06-09",  # Kurban 2025
            "2026-03-20",
            "2026-03-21",
            "2026-03-22",  # Ramazan 2026
            "2026-05-27",
            "2026-05-28",
            "2026-05-29",  # Kurban 2026  <- test ortasi
        ]
    )
)

TZ_KOL = [
    "tz_ilce_pay",
    "tz_ilce_ay_yog",
    "tz_ilce_gec_l12",
    "tz_yoy_gec_dm",
    "tz_yoy_dol_dm",
    "tz_ilce_yoy_gec",
    "tz_ilce_yoy_dol",
    "tz_rejim_kirilma",
]
TH_KOL = [
    "th_tarim",
    "th_yerlesim",
    "th_bayram",
    "th_tatil_yog",
    "th_tatil_pay",
    "th_tatil_tarim",
    "th_tatil_yerlesim",
    "th_bayram_yog",
    "th_bayram_tarim",
    "th_bayram_yerlesim",
]


def _ai(yil, ay):
    return np.asarray(yil, dtype=np.int64) * 12 + (np.asarray(ay, dtype=np.int64) - 1)


class Turizm:
    """Aylik il serisi + yillik ilce serisi + statik arazi ortusu."""

    def __init__(self):
        t = pd.read_parquet(os.path.join(DIS, "turizm_aylik_il.parquet"))
        t = t[t.kapsam == "isletme_basit"].copy()
        t["il_key"] = t["il_key"].astype(object)
        t["ai"] = _ai(t.yil, t.ay)
        self.gec = t.set_index(["il_key", "ai"]).geceleme.astype(float)
        self.dol = t.set_index(["il_key", "ai"]).doluluk.astype(float)
        self.rej = t.set_index(["il_key", "ai"]).kapsam_rejimi.astype(float)

        # YoY ve ULUSAL demean -- KAYNAK TABLODA, 81 il uzerinden hesaplanir.
        # (Yalnizca 2 ilin ortalamasini cikarmak kapsam sicramasini temizlemez;
        #  81 ilin ortalamasi o donemin ulusal/kapsam bilesenidir.)
        y = t[["il_key", "ai", "yil", "ay", "geceleme", "doluluk"]].copy()
        onc = y.assign(ai=y.ai + 12).rename(
            columns={"geceleme": "geceleme_o", "doluluk": "doluluk_o"}
        )[["il_key", "ai", "geceleme_o", "doluluk_o"]]
        y = y.merge(onc, on=["il_key", "ai"], how="left")
        with np.errstate(divide="ignore", invalid="ignore"):
            y["yoy_gec"] = np.where(y.geceleme_o > 0, y.geceleme / y.geceleme_o, np.nan)
            y["yoy_dol"] = np.where(y.doluluk_o > 0, y.doluluk / y.doluluk_o, np.nan)
        for k in ("yoy_gec", "yoy_dol"):
            y[k + "_dm"] = y[k] - y.groupby("ai")[k].transform("mean")
        self.yoy_gec_dm = y.set_index(["il_key", "ai"]).yoy_gec_dm.astype(float)
        self.yoy_dol_dm = y.set_index(["il_key", "ai"]).yoy_dol_dm.astype(float)
        self.ulusal_yoy = y.groupby("ai").yoy_gec.mean()

        # il x yil-ay-payi (kaynak YILIN 12 ayindaki pay; yalnizca tam yillar)
        tam = t.groupby(["il_key", "yil"]).ay.transform("nunique").eq(12)
        tt = t[tam]
        top = tt.groupby(["il_key", "yil"]).geceleme.transform("sum")
        self.ay_payi = (
            tt.assign(p=tt.geceleme / top).set_index(["il_key", "yil", "ay"]).p.astype(float)
        )
        self.tam_yillar = sorted(tt.yil.unique().astype(int).tolist())

        # ilce payi (yillik bulten) -- il ici pay
        g = pd.read_parquet(os.path.join(DIS, "turizm_geceleme.parquet")).copy()
        g["ilce_key"] = g["ilce_key"].astype(object)
        g["il_key"] = g["il_key"].astype(object)
        ilt = g.groupby(["il_key", "yil"]).geceleme.transform("sum")
        self.ilce_payi = g.assign(p=g.geceleme / ilt).set_index(["ilce_key", "yil"]).p.astype(float)
        self.ilce_yillar = sorted(g.yil.unique().astype(int).tolist())

        a = pd.read_parquet(os.path.join(DIS, "arazi_ortusu_ilce.parquet"))
        a["ilce_key"] = a["ilce_key"].astype(object)
        self.arazi = a.set_index("ilce_key")[["tarim_orani", "yerlesim_orani"]].astype(float)

    # ------------------------------------------------------------------ yardim
    def _kaynak_yil(self, yil, mevcut):
        m = np.asarray(sorted(mevcut))
        y = np.asarray(yil, dtype=np.int64) - 1
        idx = np.searchsorted(m, y, side="right") - 1
        idx = np.clip(idx, 0, len(m) - 1)
        return m[idx]

    def _seri(self, seri, il, ai):
        return seri.reindex(pd.MultiIndex.from_arrays([il, ai])).to_numpy(dtype=float)

    # ------------------------------------------------------------------ ana API
    def kolonlar(self, il_key, ilce_key, tarih, tatil_bayrak):
        """meta dizilerinden turizm kolonlarini uretir (dict of np.float32)."""
        il = np.asarray(il_key, dtype=object)
        ilce = np.asarray(ilce_key, dtype=object)
        t = pd.to_datetime(pd.Series(tarih))
        yil = t.dt.year.to_numpy()
        ay = t.dt.month.to_numpy()
        ai = _ai(yil, ay)

        a2 = ai - LAG_AY  # yayimlanmis en taze donem
        r2 = self._seri(self.rej, il, a2)
        r2p = self._seri(self.rej, il, a2 - 12)
        g12 = self._seri(self.gec, il, ai - 12)
        yg_dm = self._seri(self.yoy_gec_dm, il, a2)
        yd_dm = self._seri(self.yoy_dol_dm, il, a2)

        ky = self._kaynak_yil(yil, self.tam_yillar)
        ap = self.ay_payi.reindex(pd.MultiIndex.from_arrays([il, ky, ay])).to_numpy(dtype=float)

        kiy = self._kaynak_yil(yil, self.ilce_yillar)
        ip = self.ilce_payi.reindex(pd.MultiIndex.from_arrays([ilce, kiy])).to_numpy(dtype=float)
        # KTB listelemedigi ilcede belgeli konaklama YOK -> 0 (tourism.py kurali)
        ip = np.where(np.isnan(ip), 0.0, ip)

        o = {}
        o["tz_ilce_pay"] = ip
        o["tz_ilce_ay_yog"] = ip * ap
        o["tz_ilce_gec_l12"] = np.log1p(np.clip(ip * g12, 0, None))
        o["tz_yoy_gec_dm"] = yg_dm
        o["tz_yoy_dol_dm"] = yd_dm
        o["tz_ilce_yoy_gec"] = ip * yg_dm
        o["tz_ilce_yoy_dol"] = ip * yd_dm
        o["tz_rejim_kirilma"] = (r2 != r2p).astype(float)

        ar = self.arazi.reindex(ilce)
        tar = ar.tarim_orani.to_numpy(dtype=float)
        yer = ar.yerlesim_orani.to_numpy(dtype=float)
        tb = np.asarray(tatil_bayrak, dtype=float)
        bay = np.isin(t.to_numpy(), np.asarray(sorted(BAYRAM), dtype="datetime64[ns]")).astype(
            float
        )
        o["th_tarim"] = tar
        o["th_yerlesim"] = yer
        o["th_bayram"] = bay
        o["th_tatil_yog"] = tb * o["tz_ilce_ay_yog"]
        o["th_tatil_pay"] = tb * ip
        o["th_tatil_tarim"] = tb * tar
        o["th_tatil_yerlesim"] = tb * yer
        o["th_bayram_yog"] = bay * o["tz_ilce_ay_yog"]
        o["th_bayram_tarim"] = bay * tar
        o["th_bayram_yerlesim"] = bay * yer
        return {k: v.astype(np.float32) for k, v in o.items()}
