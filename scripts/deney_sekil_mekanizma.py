"""8 KOLONUN KAZANCI GERCEK MI, KURGUSAL MI.

BULGU
-----
``deney_kacan.py`` (2026-08-22): sekiz kolonu atmak sicak skoru 0,01833
iyilestiriyor, t=-1,99. Genel skora ~0,0065, yani birinciyle aramizdaki
0,0074'un %88'i. Alinmaya deger buyuklukte.

AMA DESEN SUPHELI
-----------------
    blok    TABAN     -HEPSI    fark
    yaz25   0,8195    0,8206   +0,0011   <- test ikizi, KOTULESTI
    guz25   0,8319    0,7778   -0,0541   <- kazancin TAMAMI burada
    kis26   0,7856    0,7836   -0,0020

Kazancin tamami guz25'te. Ve ``teshis_ufuk.py`` ayni gun olctu: guz25 en
buyuk KURGUSAL yanliligi tasiyan blok (ortalama artik -0,3268, yani model
sonbahari sistematik olarak fazla tahmin ediyor). Nedeni kurgusal: guz25
TEK sonbahar blogu, dislandiginda model sonbahar etiketini hic gormemis
oluyor.

IKI ACIKLAMA
    (a) Sekiz kolon gercekten zararli  -> uretimde de kazandirir
    (b) Onlari atmak guz25'in kurgusal yanliligini tesadufen azaltiyor
        -> uretimde model UC MEVSIMI DE gordugu icin o yanlilik yok,
           kazanc buharlasir

AYRIM
-----
Her blokta iki skor olculur::

    HAM        RMSLE(artik)                 -- oldugu gibi
    MERKEZLI   RMSLE(artik - artik.mean())  -- blogun kendi yanliligi giderilmis

Kareli hatada bu ozdeslik tam: MSE = Var(artik) + ortalama^2. Yani
MERKEZLI skor, yanliliktan arindirilmis SACILIM'i olcer.

    Kazanc HAM'da var, MERKEZLI'de yok   -> (b), kurgusal, REDDET
    Kazanc MERKEZLI'de de var            -> (a), gercek, AL

Fit: 2 aday x 3 blok x 3 tohum = 18 CatBoost ~ 25 dakika.

    python scripts/deney_sekil_mekanizma.py
"""

from __future__ import annotations

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
USTYAZIM: dict[str, object] = {"random_strength": 4.0}

SEKIZ = (
    "t_gy_log_ort",
    "t_gy_sifir_orani",
    "t_gy_gun",
    "t_yayilma",
    "t_kayma",
    "t_hg_genligi",
    "ozet_pencere_gun",
    "t_doluluk",
)
UC_SEKIL = ("t_yayilma", "t_kayma", "t_hg_genligi")

ADAYLAR: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("TABAN (105)", ()),
    ("-sekil (3)", UC_SEKIL),
    ("-HEPSI (8)", SEKIZ),
)


def main() -> int:
    satir_tamponlu_cikti()
    t_bas = time.time()
    print("=" * 104)
    print("8 KOLON -- kazanc GERCEK mi (sacilim) yoksa KURGUSAL mi (yanlilik)")
    print("=" * 104)

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    kolonlar = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)

    parcalar = {b.ad: di.blok_parcalari(egitim, b.ad) for b in tm.BLOKLAR}
    maskeli = {}
    for b in tm.BLOKLAR:
        for tohum in di.TOHUMLAR:
            maskeli[(b.ad, tohum)] = d.soguk_maskele(
                parcalar[b.ad][0], kolonlar, SICAK_MASKE, tohum
            )

    # aday -> blok -> (ham, merkezli, yanlilik)
    sonuc: dict[str, dict[str, tuple[float, float, float]]] = {}
    for ad, cikar in ADAYLAR:
        t0 = time.time()
        alt = [k for k in kolonlar if k not in cikar]
        sonuc[ad] = {}
        for b in tm.BLOKLAR:
            _, dogrulama, gercek, soguk = parcalar[b.ad]
            sic = ~soguk
            ln_y = np.log1p(gercek[sic])
            tahminler = []
            for tohum in di.TOHUMLAR:
                log_t = di.egit_tahmin("cat", maskeli[(b.ad, tohum)], dogrulama, alt, tohum)
                tahminler.append(np.log1p(np.clip(np.expm1(log_t), 0.0, None))[sic])
            # tohumlar log uzayinda torbalanir -- uretimdeki tahminci bu
            artik = ln_y - np.mean(tahminler, axis=0)
            ham = float(np.sqrt((artik**2).mean()))
            yanlilik = float(artik.mean())
            merkezli = float(np.sqrt(((artik - yanlilik) ** 2).mean()))
            sonuc[ad][b.ad] = (ham, merkezli, yanlilik)
        print(f"  {ad:14} bitti ({time.time() - t0:.0f} sn)")

    taban = sonuc[ADAYLAR[0][0]]
    for ad, _ in ADAYLAR[1:]:
        print(f"\n  --- {ad} vs TABAN ---")
        print(
            f"  {'blok':>8} {'HAM fark':>10} {'MERKEZLI fark':>15} "
            f"{'yanlilik taban':>15} {'yanlilik aday':>14}"
        )
        for b in tm.BLOKLAR:
            h0, m0, y0 = taban[b.ad]
            h1, m1, y1 = sonuc[ad][b.ad]
            print(f"  {b.ad:>8} {h1 - h0:>+10.5f} {m1 - m0:>+15.5f} {y0:>+15.4f} {y1:>+14.4f}")
        hf = float(np.mean([sonuc[ad][b.ad][0] - taban[b.ad][0] for b in tm.BLOKLAR]))
        mf = float(np.mean([sonuc[ad][b.ad][1] - taban[b.ad][1] for b in tm.BLOKLAR]))
        print(f"  {'ORTALAMA':>8} {hf:>+10.5f} {mf:>+15.5f}")
        pay = mf / hf if abs(hf) > 1e-9 else 0.0
        print(
            f"  kazancin SACILIM'dan gelen payi: %{100 * pay:.0f}"
            f"  (kalani blogun kendi yanliligindan -- URETIME TASINMAZ)"
        )

    print(f"\nTAMAM  {(time.time() - t_bas) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
