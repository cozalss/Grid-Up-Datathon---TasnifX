r"""HAM ve MERKEZLI olcum -- kurgusal yanliligi gercek kazanctan ayirir.

NEDEN BU ALET GEREKLI
---------------------
2026-08-22'de olculdu (``deney_sekil_mekanizma.py``): sekiz kolonu atmak
ham skorda 0,018 kazandiriyordu ama yanlilik giderilince 0,0017
KAYBETTIRIYORDU. Kazancin tamami blogun kendi sapmasini kucultmekten
geliyordu.

Nedeni kurgusal ve genel::

    yaz25 TEK yaz blogu, guz25 TEK sonbahar blogu, kis26 TEK kis blogu.
    Bir blok dislandiginda model o mevsimin etiketini HIC gormemis olur.
    Olculen ortalama artiklar: yaz25 +0,100  guz25 -0,367  kis26 +0,200

Modeli ifadesizlestiren her degisiklik (kolon atmak, duzenlileştirmeyi
artirmak, kapasiteyi kismak) tahminleri ortalamaya buzer, bu da
gorulmemis-mevsim sapmasini kucultur ve CV'de HAKSIZ YERE iyi gorunur.
Tersi de dogru: kapasiteyi artiran her degisiklik haksiz yere cezalanir.

Uretimde bu sapma yok -- son model uc mevsimi de gorur.

OZDESLIK
--------
    MSE = Var(artik) + ortalama(artik)^2
          \_________/   \______________/
            SACILIM         YANLILIK

MERKEZLI skor = RMSLE(artik - artik.mean()), yani SACILIM. Kurgusal
yanliliktan arindirilmis oldugu icin uretime tasinabilir olan kisim budur.

OKUMA KURALI
    ham + / merkezli +   -> GERCEK, al
    ham + / merkezli -   -> KURGUSAL, reddet          (8 kolon boyleydi)
    ham - / merkezli +   -> gercek ama CV maskeliyor, dikkatle al
    ham - / merkezli -   -> gercekten kotu, reddet

ADAYLAR
    1  TABAN            uretim: 105 kolon, rs=4
    2  l2=1 + d6        kapasiteyi ARTIRAN degisiklik; ham +0,00226 (3/3
                        blok) olculmustu, yani CV cezasina RAGMEN pozitif
    3  GENIS (144)      yalin setin kendisi bir buzulmeydi (39 kolon
                        atildi, ham -0,0095 kazandirdi ve URETIME ALINDI).
                        Ayni suphe ona da dusuyor: gercek miydi?

Fit: 3 aday x 3 blok x 3 tohum = 27 CatBoost ~ 35 dakika.

    python scripts/deney_merkezli.py
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

SICAK_MASKE = 0.15
TABAN_USTYAZIM: dict[str, object] = {"random_strength": 4.0}

#: Yalin setin cikardigi onekler. GENIS aday bunlari GERI KOYAR -- ama
#: t_mevsim_* ve nufus ailesi haric: ikisi de ayrica olculup reddedildi,
#: geri koymak iki soruyu birbirine karistirmak olurdu.
YALIN_ONEKLER = ("tk_", "tatil", "ramazan", "p_", "g_", "gp_")

KAYIT = KOK / "experiments" / "merkezli.jsonl"


def main() -> int:
    satir_tamponlu_cikti()
    t_bas = time.time()
    print("=" * 100)
    print("HAM ve MERKEZLI -- kurgusal yanlilik gercek kazanctan ayriliyor")
    print("=" * 100)

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    uretim = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    genis = [
        k
        for k in tum
        if not k.startswith(tuple(o for o in tm.YALIN_CIKARILAN if o not in YALIN_ONEKLER))
    ]
    tm.kategorik_kodla(egitim, test)
    print(f"  onbellek {len(tum)} | uretim {len(uretim)} | genis {len(genis)}")

    adaylar: tuple[tuple[str, list[str], dict[str, object]], ...] = (
        ("TABAN (105, rs=4)", uretim, {}),
        ("l2=1 + d6", uretim, {"l2_leaf_reg": 1.0, "depth": 6}),
        (f"GENIS ({len(genis)})", genis, {}),
    )

    parcalar = {b.ad: di.blok_parcalari(egitim, b.ad) for b in tm.BLOKLAR}
    maskeli: dict[tuple[str, int, int], object] = {}
    for kol in (uretim, genis):
        for b in tm.BLOKLAR:
            for tohum in di.TOHUMLAR:
                maskeli[(b.ad, tohum, len(kol))] = d.soguk_maskele(
                    parcalar[b.ad][0], kol, SICAK_MASKE, tohum
                )

    sonuc: dict[str, dict[str, tuple[float, float, float]]] = {}
    for ad, kol, ek in adaylar:
        t0 = time.time()
        ustyazim = {**TABAN_USTYAZIM, **ek}
        sonuc[ad] = {}
        for b in tm.BLOKLAR:
            _, dogrulama, gercek, soguk = parcalar[b.ad]
            sic = ~soguk
            ln_y = np.log1p(gercek[sic])
            tahminler = []
            for tohum in di.TOHUMLAR:
                log_t = di.egit_tahmin(
                    "cat",
                    maskeli[(b.ad, tohum, len(kol))],
                    dogrulama,
                    kol,
                    tohum,
                    **ustyazim,
                )
                tahminler.append(np.log1p(np.clip(np.expm1(log_t), 0.0, None))[sic])
            artik = ln_y - np.mean(tahminler, axis=0)
            yanlilik = float(artik.mean())
            sonuc[ad][b.ad] = (
                float(np.sqrt((artik**2).mean())),
                float(np.sqrt(((artik - yanlilik) ** 2).mean())),
                yanlilik,
            )
        print(f"  {ad:20} bitti ({time.time() - t0:.0f} sn)")

    taban_ad = adaylar[0][0]
    taban = sonuc[taban_ad]
    print(f"\n  {'aday':20} {'HAM fark':>10} {'MERKEZLI fark':>15} {'hukum':>28}")
    print("-" * 100)
    ham0 = float(np.mean([taban[b.ad][0] for b in tm.BLOKLAR]))
    mrk0 = float(np.mean([taban[b.ad][1] for b in tm.BLOKLAR]))
    print(f"  {taban_ad:20} {ham0:>10.5f} {mrk0:>15.5f}   (mutlak)")
    kayitlar = []
    for ad, _, _ in adaylar[1:]:
        hf = ham0 - float(np.mean([sonuc[ad][b.ad][0] for b in tm.BLOKLAR]))
        mf = mrk0 - float(np.mean([sonuc[ad][b.ad][1] for b in tm.BLOKLAR]))
        if hf > 0 and mf > 0:
            hukum = "GERCEK -- al"
        elif hf > 0:
            hukum = "KURGUSAL -- reddet"
        elif mf > 0:
            hukum = "gercek ama CV maskeliyor"
        else:
            hukum = "kotu -- reddet"
        print(f"  {ad:20} {hf:>+10.5f} {mf:>+15.5f}   {hukum:>28}")
        for b in tm.BLOKLAR:
            h, m, y = sonuc[ad][b.ad]
            print(
                f"    {b.ad:>16} {taban[b.ad][0] - h:>+10.5f} {taban[b.ad][1] - m:>+15.5f}"
                f"   yanlilik {taban[b.ad][2]:+.4f} -> {y:+.4f}"
            )
        kayitlar.append({"aday": ad, "ham_fark": hf, "merkezli_fark": mf, "hukum": hukum})

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as f:
        for k in kayitlar:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")

    print(f"\nTAMAM  {(time.time() - t_bas) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
