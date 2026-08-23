"""SOGUK TABAN DOGRULAMASI: hedef secimi kis26'ya ozgu bir tesadduf mu?

``deney_soguk_taban.py`` kis26'da su siralamayi buldu (soguk RMSLE):

    ilce   1,82314   <- en iyi taban
    kova   1,82670
    genel  1,82919
    ilce x kova 1,84148
    URETIM (tahminin kendi ortalamasi, beta=0,60)  1,83979

Bu betik iki soru sorar:

  1) Siralama UC BLOKTA da ayni mi?
     Kritik nokta: BURADA EZBER KANALI YOK. docs/35'teki kirlilik MODELIN
     trafo kimligini ezberlemesinden geliyor; saf grup ortalamasinda
     ezberlenecek bir sey yok. Yani yaz25 ve guz25 bu KARSILASTIRMA icin
     GECERLI -- kis26 doktrini modele bakan olcumler icindir.

  2) Taban hangi PENCEREDEN kurulmali?
     tum parca / son 120 gun / son 60 gun. Test icin de ayni soru gecerli
     ve orada mevsim eslesmesi de mumkun (Nisan-Temmuz 2025 var).

Model yok, fit yok -- saniyeler surer.

    python scripts/deney_taban_dogrula.py
"""

from __future__ import annotations

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

BLOKLAR = ("yaz25", "guz25", "kis26")
PENCERELER = ((None, "tum"), (120, "son120g"), (60, "son60g"))
M_DEGERLERI = (50.0, 200.0, 1000.0)


def _eb(anahtar_e: np.ndarray, ofs_e: np.ndarray, anahtar_h: np.ndarray,
        ebeveyn: np.ndarray, m_once: float) -> np.ndarray:
    s = pd.Series(ofs_e).groupby(anahtar_e).agg(["sum", "count"])
    top = np.nan_to_num(pd.Series(s["sum"]).reindex(anahtar_h).to_numpy(dtype="float64"), nan=0.0)
    n = np.nan_to_num(pd.Series(s["count"]).reindex(anahtar_h).to_numpy(dtype="float64"), nan=0.0)
    return (top + m_once * ebeveyn) / (n + m_once)


def _metin(v: np.ndarray) -> np.ndarray:
    """Anahtar birlestirmek icin metne cevir."""
    return pd.Series(v).astype(str).to_numpy()


def main() -> int:
    satir_tamponlu_cikti()
    t0 = time.time()
    print("=" * 92)
    print("SOGUK TABAN DOGRULAMASI  --  uc blok, model yok")
    print("=" * 92)

    egitim, test = d.cerceveleri_kur()

    print(f"\n{'blok':7} {'pencere':9} {'M':>6}  {'genel':>8} {'kova':>8} "
          f"{'ilce':>8} {'ilcexkova':>9}")
    for blok in BLOKLAR:
        parca, dogrulama, gercek, soguk = di.blok_parcalari(egitim, blok)
        dg = dogrulama[soguk]
        y = gercek[soguk]
        log_guc = np.log1p(dg["guc"].to_numpy(dtype="float64"))
        blok_bas = pd.to_datetime(dogrulama["tarih"]).min()

        p_tarih = pd.to_datetime(parca["tarih"])
        for gun, p_ad in PENCERELER:
            if gun is None:
                alt = parca
            else:
                alt = parca[(p_tarih >= blok_bas - pd.Timedelta(days=gun)) & (p_tarih < blok_bas)]
                if len(alt) < 10_000:
                    print(f"{blok:7} {p_ad:9} {'':>6}  (pencere cok seyrek: "
                          f"{len(alt):,} satir) ATLANDI")
                    continue
            of_e = (np.log1p(alt[tm.HEDEF].clip(lower=0.0).to_numpy(dtype="float64"))
                    - np.log1p(alt["guc"].to_numpy(dtype="float64")))
            kenar = np.linspace(float(np.log1p(alt["guc"]).min()),
                                float(np.log1p(alt["guc"]).max()) + 1e-9, 25)
            kv_e = np.clip(np.searchsorted(
                kenar, np.log1p(alt["guc"].to_numpy()), side="right") - 1, 0, 23)
            kv_h = np.clip(np.searchsorted(kenar, np.log1p(dg["guc"].to_numpy()), side="right") - 1,
                0, 23)
            il_e = alt["ilce_key"].to_numpy()
            il_h = dg["ilce_key"].to_numpy()

            for m_once in M_DEGERLERI:
                genel = np.full(len(dg), float(of_e.mean()))
                kova = _eb(kv_e, of_e, kv_h, genel, m_once)
                ilce = _eb(il_e, of_e, il_h, genel, m_once)
                ik = _eb(
                    _metin(il_e) + "|" + _metin(kv_e),
                    of_e,
                    _metin(il_h) + "|" + _metin(kv_h),
                    kova, m_once,
                )
                sk = [tm.rmsle(y, np.clip(np.expm1(v + log_guc), 0.0, None))
                      for v in (genel, kova, ilce, ik)]
                print(f"{blok:7} {p_ad:9} {m_once:6.0f}  "
                      + "  ".join(f"{v:8.5f}" for v in sk[:3]) + f"  {sk[3]:9.5f}")
        print()

    print(f"TAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
