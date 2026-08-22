"""SICAK UZMANI TARAMASI -- kaldiraci en yuksek dilim.

NEDEN SICAK
-----------
Gecenin butun kazanci SOGUK tarafa gitti (sicak 0,7979 -> 0,7962, yani
hicbir sey). Ama kaldirac hesabi tersini soyluyor::

    genel = sqrt(0,222 * soguk^2 + 0,778 * sicak^2)

    d(genel)/d(soguk) = 0,350   ->  0,01 soguk kazanci = 0,0035 genel
    d(genel)/d(sicak) = 0,590   ->  0,01 sicak kazanci = 0,0059 genel

Yani ayni buyuklukte bir sicak kazanci genel skora **1,7 kat** fazla
geciyor, cunku satirlarin %78'i orada. Birinciyi (1,03170) gecmek icin
gereken 0,0074 genel kazanc:

    soguk tarafta  -0,0211   (zor: tavan 0,48, biz 1,66, ve 9 acidan
                              denenip reddedildi)
    sicak tarafta  -0,0125   (hic taranmadi)

HANGI PARAMETRELER
------------------
``docs/23-olcumler`` §6 uc parametreyi "hic dokunulmamis" diye kaydetti:

* ``l2_leaf_reg=3.0`` -- CatBoost'un KENDI varsayilani; "ayarlandi"
  denmesine ragmen hic oynatilmamis
* ``bootstrap_type`` -- sessizce ``MVS`` (varsayilan)
* ``random_strength`` -- sonradan sicak icin 4,0'e ayarlandi, TABANDA var

Buna ``rsm`` (kolon ornekleme, 105 kolonda 0,75) ve derinlik eklendi.
``langevin`` (SGLB, CatBoost 0.21) ayri bir aday: gradyana gurultu
ekleyerek konveks olmayan kayiplarda kaliteyi artirmak icin tasarlandi;
yalnizca CPU'da calisiyor, bizde CPU var.

TASARIM -- ESLENIK OLCUM
------------------------
Maskeli cerceveler (blok x tohum) BIR KEZ kuruluyor ve butun adaylar
ayni cerceveleri goruyor. Yani adaylar arasindaki fark maskeleme
gurultusunu TASIMIYOR; olcum eslenik. Tohumlar arasi sapma da
raporlaniyor, boylece "esik alti" hukmu goz karariyla degil sayiyla
veriliyor.

Yalnizca SICAK satirlarda skorlaniyor -- soguk uzmani ayri model ve bu
tarama ona dokunmuyor.

Fit: 7 aday x 3 blok x 3 tohum = 63 CatBoost ~ 40 dakika.

    python scripts/deney_sicak.py
"""

from __future__ import annotations

import json
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

#: Uretimdeki sicak uzmaninin maske orani.
SICAK_MASKE = 0.15

#: Uretimdeki sicak uzmaninin CatBoost ustyazimi -- her adayin TABANI.
TABAN_USTYAZIM: dict[str, object] = {"random_strength": 4.0}

#: d(genel)/d(sicak): 0,01 sicak kazanci genel skora ne kadar geciyor.
SICAK_KALDIRAC = 0.590

#: Adaylar: (ad, TABAN_USTYAZIM uzerine eklenecekler).
ADAYLAR: tuple[tuple[str, dict[str, object]], ...] = (
    ("TABAN (uretim, rs=4)", {}),
    ("l2_leaf_reg=10", {"l2_leaf_reg": 10.0}),
    ("l2_leaf_reg=1", {"l2_leaf_reg": 1.0}),
    ("depth=6", {"depth": 6}),
    ("rsm=0,55", {"rsm": 0.55}),
    ("bootstrap Bernoulli", {"bootstrap_type": "Bernoulli", "subsample": 0.8}),
    ("langevin (SGLB)", {"langevin": True}),
)

KAYIT = KOK / "experiments" / "sicak_tarama.jsonl"


def _blok_skoru(
    parcalar: tuple,
    maskeli: dict,
    kolonlar: list[str],
    ustyazim: dict[str, object],
    blok: str,
) -> tuple[float, float]:
    """Bir blokta sicak RMSLE ve tohumlar arasi sapma dondurur."""
    _, dogrulama, gercek, soguk = parcalar
    tekil = []
    tahminler = []
    for tohum in di.TOHUMLAR:
        log_t = di.egit_tahmin(
            "cat", maskeli[(blok, tohum)], dogrulama, kolonlar, tohum, **ustyazim
        )
        tahminler.append(log_t)
        tek = np.clip(np.expm1(log_t), 0.0, None)
        tekil.append(tm.rmsle(gercek[~soguk], tek[~soguk]))
    torbali = np.clip(np.expm1(np.mean(tahminler, axis=0)), 0.0, None)
    return tm.rmsle(gercek[~soguk], torbali[~soguk]), float(np.std(tekil))


def main() -> int:
    satir_tamponlu_cikti()
    t_bas = time.time()
    print("=" * 100)
    print("SICAK UZMANI TARAMASI -- maske %15, SICAK satirlarda skorlaniyor")
    print("=" * 100)

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    kolonlar = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)
    print(f"  onbellek {len(tum)} kolon | uretim seti {len(kolonlar)} kolon")
    print(f"  taban ustyazim: {TABAN_USTYAZIM}")

    parcalar = {b.ad: di.blok_parcalari(egitim, b.ad) for b in tm.BLOKLAR}
    for b in tm.BLOKLAR:
        _, dogrulama, _, soguk = parcalar[b.ad]
        print(f"    {b.ad:6} dogrulama {len(dogrulama):>7,} satir | sicak {(~soguk).sum():>7,}")

    print("\n  maskeli cerceveler kuruluyor (aday-bagimsiz, ESLENIK olcum)...")
    t0 = time.time()
    maskeli = {}
    for b in tm.BLOKLAR:
        for tohum in di.TOHUMLAR:
            maskeli[(b.ad, tohum)] = d.soguk_maskele(
                parcalar[b.ad][0], kolonlar, SICAK_MASKE, tohum
            )
    print(f"  hazir ({time.time() - t0:.0f} sn)")

    print(f"\n{'aday':24} {'SICAK':>8} {'fark':>9} {'genel etki':>11}  bloklar")
    print("-" * 100)

    taban = None
    kayitlar = []
    for ad, ek in ADAYLAR:
        t0 = time.time()
        ustyazim = {**TABAN_USTYAZIM, **ek}
        skorlar, sapmalar = {}, {}
        for b in tm.BLOKLAR:
            skorlar[b.ad], sapmalar[b.ad] = _blok_skoru(
                parcalar[b.ad], maskeli, kolonlar, ustyazim, b.ad
            )
        genel = float(np.mean(list(skorlar.values())))
        gurultu = float(np.mean(list(sapmalar.values())))
        if taban is None:
            taban = genel
            fark_s, etki_s = "TABAN", ""
        else:
            fark = taban - genel
            fark_s = f"{fark:+.5f}"
            etki_s = f"{fark * SICAK_KALDIRAC:+.5f}"
        detay = " ".join(f"{k} {v:.4f}" for k, v in skorlar.items())
        print(
            f"{ad:24} {genel:8.5f} {fark_s:>9} {etki_s:>11}  {detay}"
            f"  (tohum sapmasi {gurultu:.4f}, {time.time() - t0:.0f} sn)"
        )
        kayitlar.append(
            {
                "aday": ad,
                "ustyazim": {k: str(v) for k, v in ustyazim.items()},
                "sicak": genel,
                "bloklar": skorlar,
                "tohum_sapmasi": gurultu,
                "genel_etki": None if taban == genel else (taban - genel) * SICAK_KALDIRAC,
            }
        )

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as f:
        for k in kayitlar:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"\n  kayit: {KAYIT}")

    en_iyi = min(kayitlar, key=lambda k: k["sicak"])
    print(f"\n  EN IYI: {en_iyi['aday']}  sicak {en_iyi['sicak']:.5f}")
    if en_iyi["aday"] != ADAYLAR[0][0]:
        kazanc = kayitlar[0]["sicak"] - en_iyi["sicak"]
        print(f"  sicak kazanc {kazanc:+.5f} -> genel {kazanc * SICAK_KALDIRAC:+.5f}")
        print(f"  tohum gurultusu {en_iyi['tohum_sapmasi']:.4f} -- kazanc bunun ustunde mi?")
    print(f"\nTAMAM  {(time.time() - t_bas) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
