"""HUCRE TABLOSU HANGI AYLARDAN KURULMALI? -- onceki yilin ayni aylari mi?

NEDEN
-----
``son_islem_gun.py`` ilce x kova ofset tablosunu egitimin TUM aylarindan
kuruyor. Ama hedef pencere Nisan-Temmuz. Bagimsiz bir olcum, uretimde
kullanilan "tum aylar" tablosunun yaz tablosuyla korelasyonunun yalnizca
0,7559 oldugunu ve yazdan %27 daha genis oldugunu gosterdi. Yani tablonun
SEKLI mevsime gore degisiyor.

Bu dogrudan olculebilir gorunmuyordu: kis26 (Ara-Mar) icin egitim parcasi
Nis-Kas 2025 ve icinde kis ayi yok. AMA ham ``train.csv`` Ocak 2025'te
basliyor ve **Ocak-Mart 2025 hicbir bloga ait degil** -- blok verisi
Nisan 2025'te basliyor. Yani kis26 icin "onceki yilin ayni aylari"
tablosu KURULABILIR ve tamamen fold-disidir.

Bu, teste yapacagimiz seyin birebir provasi:
    kis26 (Ara2025-Mar2026)  <- tablo: Oca-Mar 2025   (ayni aylar, onceki yil)
    TEST  (Nis-Tem 2026)     <- tablo: Nis-Tem 2025   (ayni aylar, onceki yil)

SIZINTI: tablo kaynagi hedef blogun tarih araligiyla HIC kesismez; kod
bunu acikca kontrol eder.

Karar mercii kis26. Model tahminleri onbellekten (cat-only), a/b agirliklari
``deney_ikili_agirlik.py`` ile secilen degerler.

    python scripts/deney_tablo_penceresi.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

import deney as d  # noqa: E402
import deney_ileri as di  # noqa: E402
import tuketim_model as tm  # noqa: E402

from gridup.reporting import satir_tamponlu_cikti  # noqa: E402

BLOK = "kis26"
TOHUMLAR = (1000, 1001, 1002)
ONBELLEK = KOK / "data" / "interim" / "deney" / f"soguk_tahmin_{BLOK}.npz"
KAYIT = KOK / "experiments" / "tablo_penceresi.jsonl"
KOVA_SAYISI = 24
M_ANA = 200.0
M_HUCRE = 2000.0
A_HUCRE = 0.55
B_MODEL = 0.25

#: Tablo kaynagi pencereleri. kis26 = 2025-12-01..2026-03-31.
PENCERELER = (
    ("tum (blok oncesi)", "2025-01-01", "2025-11-30"),
    ("ayni ay onceki yil", "2025-01-01", "2025-03-31"),
    ("son 3 ay", "2025-09-01", "2025-11-30"),
    ("son 6 ay", "2025-06-01", "2025-11-30"),
)


def _eb(anahtar_e, ofs_e, anahtar_h, ebeveyn, m_once):  # noqa: ANN001, ANN201
    s = pd.Series(ofs_e).groupby(anahtar_e).agg(["sum", "count"])
    top = np.nan_to_num(pd.Series(s["sum"]).reindex(anahtar_h).to_numpy(dtype="float64"), nan=0.0)
    n = np.nan_to_num(pd.Series(s["count"]).reindex(anahtar_h).to_numpy(dtype="float64"), nan=0.0)
    return (top + m_once * ebeveyn) / (n + m_once)


def tablo_kur(kaynak: pd.DataFrame, hedef_guc: np.ndarray, hedef_ilce: np.ndarray) -> np.ndarray:
    ofs = (np.log1p(kaynak["tuketim"].clip(lower=0.0).to_numpy(dtype="float64"))
           - np.log1p(kaynak["guc"].to_numpy(dtype="float64")))
    lg_e = np.log1p(kaynak["guc"].to_numpy(dtype="float64"))
    kenar = np.linspace(float(lg_e.min()), float(lg_e.max()) + 1e-9, KOVA_SAYISI + 1)
    kv_e = np.clip(np.searchsorted(kenar, lg_e, side="right") - 1, 0, KOVA_SAYISI - 1)
    kv_h = np.clip(
        np.searchsorted(kenar, np.log1p(hedef_guc), side="right") - 1,
        0, KOVA_SAYISI - 1)
    il_e = kaynak["lokasyon"].astype(str).to_numpy()
    genel = np.full(len(hedef_guc), float(ofs.mean()))
    ilce = _eb(il_e, ofs, hedef_ilce, genel, M_ANA)
    ae = pd.Series(il_e).to_numpy() + "|" + pd.Series(kv_e).astype(str).to_numpy()
    ah = pd.Series(hedef_ilce).to_numpy() + "|" + pd.Series(kv_h).astype(str).to_numpy()
    return _eb(ae, ofs, ah, ilce, M_HUCRE)


def main() -> int:
    satir_tamponlu_cikti()
    t0 = time.time()
    print("=" * 92)
    print(f"HUCRE TABLOSU PENCERESI  --  {BLOK} (a={A_HUCRE}, b={B_MODEL})")
    print("=" * 92)

    egitim, test = d.cerceveleri_kur()
    tm.kategorik_kodla(egitim, test)
    parca, dogrulama, gercek, soguk = di.blok_parcalari(egitim, BLOK)
    y = gercek[soguk]
    dg = dogrulama[soguk]
    log_guc = np.log1p(dg["guc"].to_numpy(dtype="float64"))
    tarih = pd.to_datetime(dg["tarih"]).to_numpy()
    blok_bas, blok_son = pd.Timestamp(tarih.min()), pd.Timestamp(tarih.max())
    print(f"  {BLOK}: {blok_bas.date()} .. {blok_son.date()}, {len(y):,} soguk satir")

    ham = pd.read_csv(KOK / "data/raw/train.csv", encoding="utf-8", dtype={"tanim": str})
    ham["t"] = pd.to_datetime(ham["tarih"])
    hedef_guc = dg["guc"].to_numpy(dtype="float64")
    hedef_ilce = dg["lokasyon"].astype(str).to_numpy()

    z = np.load(ONBELLEK)
    cat = {t: z[f"{t}_cat"] for t in TOHUMLAR}

    def grup_ort(v, k):  # noqa: ANN001, ANN202
        return pd.Series(v).groupby(k).transform("mean").to_numpy()

    def skorla(ofs):  # noqa: ANN001, ANN202
        return tm.rmsle(y, np.clip(np.expm1(ofs + log_guc), 0.0, None))

    kayitlar = []
    print(f"\n  {'pencere':22}{'kaynak satir':>13}{'tablo std':>11}{'RMSLE':>10}")
    for ad, bas, son in PENCERELER:
        kaynak = ham[(ham["t"] >= bas) & (ham["t"] <= son)]
        if pd.Timestamp(son) >= blok_bas:
            raise RuntimeError(f"{ad}: kaynak hedef blokla kesisiyor -- SIZINTI")
        hucre = tablo_kur(kaynak, hedef_guc, hedef_ilce)
        sk = []
        for t in TOHUMLAR:
            m = cat[t] - log_guc
            gun = grup_ort(m, tarih)
            etki = hucre - grup_ort(hucre, tarih)
            sk.append(skorla(gun + A_HUCRE * etki + B_MODEL * (m - gun)))
        o = float(np.mean(sk))
        kayitlar.append({"pencere": ad, "n": int(len(kaynak)), "std": float(hucre.std()),
                         "rmsle": o})
        print(f"  {ad:22}{len(kaynak):13,}{hucre.std():11.4f}{o:10.5f}")

    taban = kayitlar[0]["rmsle"]
    en_iyi = min(kayitlar, key=lambda k: k["rmsle"])
    print(f"\n  TABAN (tum aylar): {taban:.5f}")
    print(f"  EN IYI: {en_iyi['pencere']}  {en_iyi['rmsle']:.5f}   "
          f"kazanc {taban - en_iyi['rmsle']:+.5f}")
    print(f"  genel skora tahmini etki {-(taban - en_iyi['rmsle']) * 0.377:+.5f}")

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        for k in kayitlar:
            fh.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
