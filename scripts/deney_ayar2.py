"""CatBoost ayarlari -- HER REJIM KENDI ICINDE.

NEDEN YENIDEN
-------------
``deney_ileri.py --deney ayar`` yazildi ama yanlis rejimde koşuyordu:
maske %22,16'da, yani yonlendirmeden SONRA artik var olmayan bir
yapilandirmada. Uretimde iki uzman var ve ikisi bambaska problemler
cozuyor:

  * SICAK uzmani: maske %15, 144 oznitelik, gecmisi olan trafolar,
    hedef artiginin std'si ~1,8. Sicak satirlarda olculmeli.
  * SOGUK uzmani: maske %100, ~125 canli oznitelik (butun ``t_*``
    sabit-NaN), hicbir gecmis yok. Soguk satirlarda olculmeli.

Bir hiperparametrenin optimumu probleme baglidir. Iki rejimi tek sayida
birlestirip taramak, ikisinin ortalamasina uygun ama hicbirine uygun
olmayan bir ayar bulur.

Ilk ipucu var: ``deney_soguk_uzman.py`` soguk uzmaninda derinlik 6'nin
d5'ten 0,0065 iyi oldugunu olctu (esik alti ama yon tutarli). Sicak
tarafta d5 taranmis ve d8 kotu cikmisti -- yani iki rejim zaten farkli
yerde.

TARANAN EKSENLER -- ucu de HIC dokunulmamis:
  * ``l2_leaf_reg=3,0`` CatBoost'un KENDI varsayilani; "ayarlandi"
    denmesine ragmen hic oynatilmamis.
  * ``bootstrap_type`` sessizce ``MVS``; Bernoulli/Bayesian hicbir arama
    uzayina girmemis.
  * ``random_strength`` bolme SECIMINI duzenler, denenmemis.
Derinlik yalnizca SOGUK tarafta taranir (sicakta zaten duz cikti).

ESLESTIRILMIS: her aday TABAN ile ayni tohumlari VE ayni maskelenmis
cerceveleri kullanir. Tohum gurultusunun buyuk kismi ikisinde de ayni
yonde hareket eder ve fark uzerinden duser.

Fit: soguk 8 aday + sicak 6 aday = 14 x 3 blok x 3 tohum = 126 ~ 75 dakika.

Calistirma::

    python scripts/deney_ayar2.py            # ikisi de
    python scripts/deney_ayar2.py --rejim soguk
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

import deney as d  # noqa: E402
import deney_ileri as di  # noqa: E402
import tuketim_model as tm  # noqa: E402

from gridup.reporting import satir_tamponlu_cikti  # noqa: E402

SOGUK_ADAYLAR: tuple[tuple[str, dict[str, object]], ...] = (
    ("TABAN d5", {}),
    ("d6", {"depth": 6}),
    ("d7", {"depth": 7}),
    ("l2=1", {"l2_leaf_reg": 1.0}),
    ("l2=10", {"l2_leaf_reg": 10.0}),
    ("bootstrap=Bayesian", {"bootstrap_type": "Bayesian", "bagging_temperature": 1.0}),
    ("bootstrap=Bernoulli0.7", {"bootstrap_type": "Bernoulli", "subsample": 0.7}),
    ("random_strength=4", {"random_strength": 4.0}),
)

SICAK_ADAYLAR: tuple[tuple[str, dict[str, object]], ...] = (
    ("TABAN d5", {}),
    ("l2=1", {"l2_leaf_reg": 1.0}),
    ("l2=10", {"l2_leaf_reg": 10.0}),
    ("bootstrap=Bayesian", {"bootstrap_type": "Bayesian", "bagging_temperature": 1.0}),
    ("bootstrap=Bernoulli0.7", {"bootstrap_type": "Bernoulli", "subsample": 0.7}),
    ("random_strength=4", {"random_strength": 4.0}),
)


def rejim_tara(
    egitim,  # noqa: ANN001 - pd.DataFrame
    kolonlar: list[str],
    rejim: str,
    maske_orani: float,
    adaylar: tuple[tuple[str, dict[str, object]], ...],
) -> None:
    print("\n" + "=" * 100)
    print(f"{rejim.upper()} UZMANI  (maske {maske_orani})  --  yalnizca {rejim} satirlarda olculur")
    print("=" * 100)

    parcalar = {}
    for b in tm.BLOKLAR:
        kalan, dogrulama, gercek, soguk = di.blok_parcalari(egitim, b.ad)
        parcalar[b.ad] = (kalan, dogrulama, gercek, soguk)

    # Maskelenmis cerceveler BIR KEZ kurulur ve butun adaylar paylasir.
    # Esleştirme boyle saglanir: adaylar arasindaki tek fark hiperparametre.
    maskeli: dict[tuple[str, int], object] = {}
    for b in tm.BLOKLAR:
        for tohum in di.TOHUMLAR:
            maskeli[(b.ad, tohum)] = d.soguk_maskele(
                parcalar[b.ad][0], kolonlar, maske_orani, tohum
            )

    taban = None
    for ad, ustyazim in adaylar:
        t0 = time.time()
        skorlar = {}
        for b in tm.BLOKLAR:
            _, dogrulama, gercek, soguk = parcalar[b.ad]
            secim = soguk if rejim == "soguk" else ~soguk
            tahminler = [
                di.egit_tahmin(
                    "cat", maskeli[(b.ad, tohum)], dogrulama, kolonlar, tohum, **ustyazim
                )
                for tohum in di.TOHUMLAR
            ]
            t = np.clip(np.expm1(np.mean(tahminler, axis=0)), 0.0, None)
            skorlar[b.ad] = tm.rmsle(gercek[secim], t[secim])
        genel = float(np.mean(list(skorlar.values())))
        if taban is None:
            taban = genel
            isaret = "TABAN"
        else:
            fark = taban - genel
            isaret = f"{fark:+.5f} " + ("GECTI" if fark > di.ESIK else "esik alti")
        detay = "  ".join(f"{k} {v:.4f}" for k, v in skorlar.items())
        sure = f"({time.time() - t0:.0f} sn)"
        print(f"  {ad:24} {rejim.upper():5} {genel:.5f}  {isaret:20} {detay}  {sure}")


def main() -> int:
    satir_tamponlu_cikti()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rejim", choices=("soguk", "sicak", "ikisi"), default="ikisi")
    args = ap.parse_args()

    t_bas = time.time()
    print("=" * 100)
    print("CatBoost AYARLARI -- HER REJIM KENDI ICINDE (eslestirilmis, 3 tohum torbalanmis)")
    print("=" * 100)
    egitim, test = d.cerceveleri_kur()
    kolonlar = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    tm.kategorik_kodla(egitim, test)
    print(f"  egitim {len(egitim):,} satir | {len(kolonlar)} oznitelik | esik {di.ESIK:.5f}")

    # Soguk ONCE: hatanin %59'u orada, ve zaman biterse orasi kalsin.
    if args.rejim in ("soguk", "ikisi"):
        rejim_tara(egitim, kolonlar, "soguk", 1.0, SOGUK_ADAYLAR)
    if args.rejim in ("sicak", "ikisi"):
        rejim_tara(egitim, kolonlar, "sicak", 0.15, SICAK_ADAYLAR)

    print(f"\nTAMAM  {(time.time() - t_bas) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
