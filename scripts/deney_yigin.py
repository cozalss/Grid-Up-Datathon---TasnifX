"""TAM YIGIN -- esik alti kazanclarin BIRLIKTE olcumu.

NEDEN
-----
2026-08-21 gecesi dort degisiklik olculdu ve YALNIZCA BIRI esigi gecti:

    rejim yonlendirmesi        +0,0177   ALINDI
    soguk uzmani derinlik 7    +0,0151 (soguk satirlarda)   esik alti
    yalin set 105 kolon        +0,0095   esik alti
    sicak random_strength=4    +0,0038   esik alti

Uc tanesi tek tek reddedildi. Ama BIRLIKTE hic olculmedi.

Toplamak varsayimdir: uc degisiklik ayni hatayi mi duzeltiyor (kazanclar
ortusur, toplam < parcalarin toplami) yoksa farkli hatalari mi (toplanir)?
Bu ancak olculerek bilinir. Ve dogru test tam olarak budur -- TEK BIR SAYI,
ya esigi gecer ya gecmez. Alt-esik kazanclari "birikir" diye tek tek
almak, bu disiplinin engellemek icin var oldugu seyin ta kendisidir.

Olculen yigin:
  * yonlendirme: sicak satirlar maske %15'ten, soguk satirlar maske %100'den
  * soguk uzmani derinlik 7 (d5 yerine) -- tepe noktasi, bes derinlikte
    tek tepeli egri: 1,75955 / 1,75360 / 1,74443 / 1,74698 / 1,74592
  * sicak uzmani random_strength=4
  * yalin oznitelik seti: takvim + panel_yapisi + grup_seviye + grup_profil
    aileleri cikarilmis (144 -> 105 kolon)
  * harman 3/1/1, uc tohum torbalanmis

Kiyas: ayni tezgahta olculmus mevcut yapilandirma 1,08143.

Fit: 2 yapilandirma x 3 blok x 2 rejim x 3 aile x 3 tohum = 108 ~ 70 dakika.
Taban da yeniden olculuyor -- ESLESTIRME icin sart, cunku yalin set
kolon kumesini degistiriyor ve kiyasin ayni kosuda olmasi gerekir.

Calistirma::

    python scripts/deney_yigin.py
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

AILELER = ("cat", "xgb", "lgbm")
AGIRLIK = {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0}

#: Yalin sette cikarilan aileler. Ablasyon bunlari olculemez ilan etmisti;
#: takvim ailesi CIKARILINCA skor iyilesiyordu (iki bagimsiz olcumde).
YALIN_CIKAR = ("takvim", "panel_yapisi", "grup_seviye", "grup_profil")

#: (ad, sicak ustyazim, soguk ustyazim, yalin mi)
YAPILANDIRMALAR = (
    ("MEVCUT (d5, tam 144 kolon)", {}, {}, False),
    ("YIGIN (soguk d7 + sicak rs4 + yalin 105)", {"random_strength": 4.0}, {"depth": 7}, True),
)


def main() -> int:
    satir_tamponlu_cikti()
    t_bas = time.time()
    print("=" * 100)
    print("TAM YIGIN -- esik alti kazanclar BIRLIKTE")
    print("=" * 100)
    egitim, test = d.cerceveleri_kur()
    kolonlar = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    tm.kategorik_kodla(egitim, test)
    at = {k for a in YALIN_CIKAR for k in d._aile_kolonlari(kolonlar, a)}
    yalin = [k for k in kolonlar if k not in at]
    print(f"  egitim {len(egitim):,} satir | tam {len(kolonlar)} kolon | yalin {len(yalin)} kolon")
    print(f"  esik {di.ESIK:.5f}   kiyas: ayni tezgahta mevcut yapilandirma 1,08143")

    parcalar = {}
    for b in tm.BLOKLAR:
        parcalar[b.ad] = di.blok_parcalari(egitim, b.ad)

    taban = None
    for ad, s_ust, c_ust, yalin_mi in YAPILANDIRMALAR:
        t0 = time.time()
        alt = yalin if yalin_mi else kolonlar
        blok_skorlari = {}
        for b in tm.BLOKLAR:
            kalan, dogrulama, gercek, soguk = parcalar[b.ad]
            # rejim -> aile -> tohum ortalamasi
            birlesik = np.zeros(len(dogrulama), dtype="float64")
            for rejim, oran, ust in (("sicak", 0.15, s_ust), ("soguk", 1.0, c_ust)):
                secim = soguk if rejim == "soguk" else ~soguk
                if not secim.any():
                    continue
                hedef = dogrulama.loc[secim]
                toplam = sum(AGIRLIK.values())
                karisim = np.zeros(len(hedef), dtype="float64")
                for tohum in di.TOHUMLAR:
                    maskeli = d.soguk_maskele(kalan, alt, oran, tohum)
                    for a in AILELER:
                        karisim += (AGIRLIK[a] / toplam / len(di.TOHUMLAR)) * di.egit_tahmin(
                            a, maskeli, hedef, alt, tohum, **ust
                        )
                    del maskeli
                birlesik[secim] = karisim
            blok_skorlari[b.ad] = [di.skorla(gercek, soguk, birlesik)]
        genel = di.yazdir(ad, blok_skorlari)
        di.kaydet(ad, blok_skorlari, {"deney": "yigin"})
        if taban is None:
            taban = genel
        else:
            fark = taban - genel
            print(f"      fark {fark:+.5f}   {'GECTI' if fark > di.ESIK else 'ESIK ALTI'}")
        print(f"      ({time.time() - t0:.0f} sn)")

    print(f"\nTAMAM  {(time.time() - t_bas) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
