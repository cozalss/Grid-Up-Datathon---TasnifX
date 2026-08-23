"""SOGUK TABAN INCE AYARI: ilce x kova hucresi, duzlestirme ve ebeveyn secimi.

``deney_taban_dogrula.py`` ilce x kova hucresinin GUCLU duzlestirmeyle uc
blokta da en iyi taban oldugunu gosterdi (M=1000). Iki soru kaldi:

  1) M nerede tepe yapiyor? (200 -> 1000 arasi hala iniyordu)
  2) EBEVEYN kim olmali? Bugun ``kova``, ama tek basina ``ilce`` ``kova``dan
     iyi. Uc aday:
        kova      (bugunku)
        ilce
        toplamsal  ilce + kova - genel   (iki ana etkinin toplamsal modeli;
                   hucre seyrekse buraya duser, yoksa hucrenin kendisine)

Model yok, fit yok. Uc blokta da olculur -- saf grup ortalamasinda
ezberlenecek bir sey olmadigi icin yaz25/guz25 de gecerlidir.

    python scripts/deney_taban_ince.py
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
M_DEGERLERI = (200.0, 500.0, 1000.0, 2000.0, 5000.0, 20000.0)
KOVA_SAYILARI = (8, 16, 24)
EBEVEYNLER = ("kova", "ilce", "toplamsal")


def _eb(anahtar_e: np.ndarray, ofs_e: np.ndarray, anahtar_h: np.ndarray,
        ebeveyn: np.ndarray, m_once: float) -> np.ndarray:
    s = pd.Series(ofs_e).groupby(anahtar_e).agg(["sum", "count"])
    top = np.nan_to_num(pd.Series(s["sum"]).reindex(anahtar_h).to_numpy(dtype="float64"), nan=0.0)
    n = np.nan_to_num(pd.Series(s["count"]).reindex(anahtar_h).to_numpy(dtype="float64"), nan=0.0)
    return (top + m_once * ebeveyn) / (n + m_once)


def main() -> int:
    satir_tamponlu_cikti()
    t0 = time.time()
    print("=" * 96)
    print("SOGUK TABAN INCE AYARI  --  ilce x kova: M, kova sayisi, ebeveyn")
    print("=" * 96)

    egitim, _ = d.cerceveleri_kur()
    en_iyi_genel: dict[tuple[int, str, float], list[float]] = {}

    for blok in BLOKLAR:
        parca, dogrulama, gercek, soguk = di.blok_parcalari(egitim, blok)
        dg = dogrulama[soguk]
        y = gercek[soguk]
        log_guc = np.log1p(dg["guc"].to_numpy(dtype="float64"))
        of_e = (np.log1p(parca[tm.HEDEF].clip(lower=0.0).to_numpy(dtype="float64"))
                - np.log1p(parca["guc"].to_numpy(dtype="float64")))
        lg_e = np.log1p(parca["guc"].to_numpy(dtype="float64"))
        lg_h = np.log1p(dg["guc"].to_numpy(dtype="float64"))
        il_e = parca["ilce_key"].to_numpy()
        il_h = dg["ilce_key"].to_numpy()
        genel = np.full(len(dg), float(of_e.mean()))
        genel_sk = tm.rmsle(y, np.clip(np.expm1(genel + log_guc), 0.0, None))
        print(f"\n{blok}  ({len(y):,} soguk satir)   genel taban {genel_sk:.5f}")
        print(f"  {'kova#':>5} {'ebeveyn':>10} " + " ".join(f"{m:>9.0f}" for m in M_DEGERLERI))

        for k_sayi in KOVA_SAYILARI:
            kenar = np.linspace(float(lg_e.min()), float(lg_e.max()) + 1e-9, k_sayi + 1)
            kv_e = np.clip(np.searchsorted(kenar, lg_e, side="right") - 1, 0, k_sayi - 1)
            kv_h = np.clip(np.searchsorted(kenar, lg_h, side="right") - 1, 0, k_sayi - 1)
            anahtar_e = (pd.Series(il_e).astype(str).to_numpy() + "|"
                         + pd.Series(kv_e).astype(str).to_numpy())
            anahtar_h = (pd.Series(il_h).astype(str).to_numpy() + "|"
                         + pd.Series(kv_h).astype(str).to_numpy())
            # ana etkiler hep ayni M ile (sabit, hafif duzlestirme)
            kova = _eb(kv_e, of_e, kv_h, genel, 200.0)
            ilce = _eb(il_e, of_e, il_h, genel, 200.0)
            ebeveyn_tablosu = {"kova": kova, "ilce": ilce, "toplamsal": ilce + kova - genel}
            for eb_ad in EBEVEYNLER:
                satir = []
                for m_once in M_DEGERLERI:
                    v = _eb(anahtar_e, of_e, anahtar_h, ebeveyn_tablosu[eb_ad], m_once)
                    sk = tm.rmsle(y, np.clip(np.expm1(v + log_guc), 0.0, None))
                    satir.append(sk)
                    en_iyi_genel.setdefault((k_sayi, eb_ad, m_once), []).append(sk)
                print(f"  {k_sayi:5d} {eb_ad:>10} " + " ".join(f"{v:9.5f}" for v in satir))

    print("\n" + "=" * 96)
    print("UC BLOK ORTALAMASI (dusuk = iyi)")
    print("=" * 96)
    sirali = sorted(en_iyi_genel.items(), key=lambda kv: float(np.mean(kv[1])))
    for (k_sayi, eb_ad, m_once), sk in sirali[:12]:
        print(f"  kova={k_sayi:2d} ebeveyn={eb_ad:10} M={m_once:7.0f}   "
              f"ort {np.mean(sk):.5f}   [" + " ".join(f"{v:.5f}" for v in sk) + "]")
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
