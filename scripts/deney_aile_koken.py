"""EK KOKENLER x AILE, ve HARMAN AGIRLIGI YENIDEN.

IKI ACIK SORU
-------------
1. Ek kokenlerin degeri YALNIZCA CatBoost'ta olculdu (``deney_koken2``,
   ``deney_koken_rejim``). Ama uretim uc aile harmanliyor ve ek kokenler
   UCUNE birden veriliyor. xgb ve lgbm de kazaniyor mu, yoksa biri zarar
   mi goruyor? Zarar goruyorsa cozum aile basina koken kontrolu.

2. ``AILE_AGIRLIKLARI = 3/1/1`` ek kokenlerden ONCE olculmustu. Ek kokenler
   CatBoost'u guclendirdiyse (sicak 0,80675 -> 0,79848) optimum agirlik
   kaymis olabilir.

TASARIM -- TAHMIN ONBELLEGI
---------------------------
Her (blok, aile, kol) icin log-tahminler BIR KEZ uretilip saklaniyor.
Sonra agirlik izgarasi bedavaya taranabiliyor: fit sayisi agirlik
sayisindan bagimsiz. Bu, 3/1/1 kararini yeniden acmanin en ucuz yolu.

Tek tohum: agirlik egrisi tohum gurultusune gore duz (onceki olcumde
2/1/1 ile 4/1/1 arasi 0,001), ve uc blok ortalamasi zaten yumusatiyor.
Aile-koken karsilastirmasi icin de yon yeterli.

Sicak uzmani yapilandirmasi (maske 0,15), SICAK satirlarda skorlaniyor --
soguk uzmani ek koken ZATEN almiyor (olculdu: -0,03273, t=-2,59).

Fit: 3 blok x 2 kol x 3 aile = 18 model ~ 36 dakika.

    python scripts/deney_aile_koken.py
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

SICAK_MASKE = 0.15
TOHUM = 1000
CAT_USTYAZIM: dict[str, object] = {"random_strength": 4.0, "l2_leaf_reg": 1.0, "depth": 6}
AILELER = ("cat", "xgb", "lgbm")

#: Taranacak (cat, xgb, lgbm) agirliklari. Uretim 3/1/1.
AGIRLIKLAR: tuple[tuple[float, float, float], ...] = (
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 1),
    (2, 1, 1),
    (3, 1, 1),
    (4, 1, 1),
    (6, 1, 1),
    (3, 2, 1),
    (3, 1, 2),
    (2, 1, 0),
    (2, 0, 1),
)

KAYIT = KOK / "experiments" / "aile_koken.jsonl"


def main() -> int:
    satir_tamponlu_cikti()
    t_bas = time.time()
    print("=" * 100)
    print("EK KOKENLER x AILE  +  HARMAN AGIRLIGI YENIDEN")
    print("=" * 100)

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    kolonlar = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)

    ek = d._ek_kokenler_kur(False)
    # Onbellekte 11 koken var (yogunlastirma denemesinden); uretimdeki ALTIYA indir.
    ek = ek[ek["_blok"].isin([a for a, _, _ in tm.EK_KOKENLER])]
    ortak = [k for k in egitim.columns if k in ek.columns]
    genis = pd.concat([egitim[ortak], ek[ortak]], ignore_index=True)
    for k in tm.KATEGORIK:
        genis[k] = pd.Categorical(genis[k], categories=egitim[k].cat.categories)
    print(f"  ana {len(egitim):,} satir -> ek kokenli {len(genis):,} ({len(tm.EK_KOKENLER)} koken)")

    # (kol, blok, aile) -> log tahmin;  (blok) -> (gercek, sicak maskesi)
    onbellek: dict[tuple[str, str, str], np.ndarray] = {}
    hedefler: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for kol_ad, kaynak, ayikla in (("ANA", egitim, False), ("EK KOKENLI", genis, True)):
        for b in tm.BLOKLAR:
            dogrulama = egitim[egitim["_blok"] == b.ad]
            kalan = tm.kokenleri_ayikla(kaynak, b.ad) if ayikla else kaynak[kaynak["_blok"] != b.ad]
            gercek = dogrulama[tm.HEDEF].to_numpy()
            sic = (dogrulama["soguk_mu"] == 0).to_numpy()
            hedefler[b.ad] = (gercek, sic)
            maskeli = d.soguk_maskele(kalan, kolonlar, SICAK_MASKE, TOHUM)
            for aile in AILELER:
                t0 = time.time()
                ust = CAT_USTYAZIM if aile == "cat" else {}
                onbellek[(kol_ad, b.ad, aile)] = di.egit_tahmin(
                    aile, maskeli, dogrulama, kolonlar, TOHUM, **ust
                )
                print(f"    {kol_ad:11} {b.ad:6} {aile:5} bitti ({time.time() - t0:.0f} sn)")

    def skor(kol_ad: str, w: tuple[float, float, float]) -> float:
        toplam = sum(w)
        blok = []
        for b in tm.BLOKLAR:
            gercek, sic = hedefler[b.ad]
            log_t = (
                sum(wi * onbellek[(kol_ad, b.ad, a)] for a, wi in zip(AILELER, w, strict=True))
                / toplam
            )
            t = np.clip(np.expm1(log_t), 0.0, None)
            blok.append(tm.rmsle(gercek[sic], t[sic]))
        return float(np.mean(blok))

    print("\n  --- AILE BASINA: ek kokenler kime yariyor ---")
    print(f"  {'aile':6}{'ANA':>10}{'EK KOKENLI':>13}{'fark':>10}")
    kayitlar = []
    for i, aile in enumerate(AILELER):
        w = tuple(1.0 if j == i else 0.0 for j in range(3))
        a, e = skor("ANA", w), skor("EK KOKENLI", w)  # type: ignore[arg-type]
        print(f"  {aile:6}{a:>10.5f}{e:>13.5f}{a - e:>+10.5f}")
        kayitlar.append({"tur": "aile", "aile": aile, "ana": a, "ek": e, "fark": a - e})

    print("\n  --- HARMAN AGIRLIKLARI ---")
    print(f"  {'cat/xgb/lgbm':14}{'ANA':>10}{'EK KOKENLI':>13}")
    en_iyi = {}
    for w in AGIRLIKLAR:
        a, e = skor("ANA", w), skor("EK KOKENLI", w)
        isaret = "  <- URETIM" if w == (3, 1, 1) else ""
        print(f"  {str(w):14}{a:>10.5f}{e:>13.5f}{isaret}")
        for kol_ad, v in (("ANA", a), ("EK KOKENLI", e)):
            if kol_ad not in en_iyi or v < en_iyi[kol_ad][1]:
                en_iyi[kol_ad] = (w, v)
        kayitlar.append({"tur": "agirlik", "w": list(w), "ana": a, "ek": e})

    print()
    for kol_ad, (w, v) in en_iyi.items():
        uretim = skor(kol_ad, (3, 1, 1))
        print(
            f"  {kol_ad:11} en iyi {str(w):12} {v:.5f}   uretim 3/1/1 {uretim:.5f}"
            f"   fark {uretim - v:+.5f}"
        )
    print("\n  UYARI: 12 hucrelik izgaradan en iyisini secmek ASIRI UYDURMADIR.")
    print("  Uretim agirligini ancak fark tohum gurultusunun (~0,007) USTUNDEYSE degistir.")

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as f:
        for k in kayitlar:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"\nTAMAM  {(time.time() - t_bas) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
