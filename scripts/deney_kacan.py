"""ABLASYONDAN KACAN KOLONLAR -- kapsama acigini kapatir.

BULGU
-----
``deney.AILELER`` sozlugu onek eslesmesiyle calisiyor ve **hicbir kapsama
testi yok**. Denetlendiginde 144 kolonun 12'sinin hicbir aileye girmedigi
gorüldü:

    guc, il_key, bolge, soguk_mu          <- yapisal, aileye girmemesi dogru
    t_gy_log_ort, t_gy_sifir_orani, t_gy_gun
    t_yayilma, t_kayma, t_hg_genligi
    ozet_pencere_gun, t_doluluk           <- 8 kolon GERCEKTEN kacti

Yani "136 kolonun 19'u is yapiyor" ablasyonu bu sekizini HIC sinamadi.
Faydali mi zararli mi, bilinmiyor.

Bu, ayni gece bulunan ikinci kapsama hatasi: ``hava`` ailesinin onekleri
"isitma_derece"/"sogutma_derece" idi ama o isimde kolon yoktu, ve 10 CDD
kolonu ablasyondan kacmisti.

NEDEN SICAK REJIMDE OLCULUYOR
Sekizinin de gecmisten turemis (``t_gy_*``, ``t_kayma``, ``t_yayilma``,
``t_hg_genligi``, ``t_doluluk``) ya da pencere yapisina ait
(``ozet_pencere_gun``) olmasi, onlari SOGUK uzmani icin sabit-NaN yapar --
orada zaten olu. Etkileri yalnizca SICAK uzmaninda olculebilir.

``t_gy_*`` DOLULUK ORANI (sicak satirlar) -- olculdu:
    yaz25 %0,0   guz25 %0,0   kis26 %58,0   TEST %52,6
Yani kis26 test'e YAKIN doluluktu. ``t_ay_sapma``daki gibi %5-e-karsi-%47
degil; model bu kolonu kis26'da ogrenebiliyor.

Fit: 5 aday x 3 blok x 3 tohum = 45 CatBoost ~ 26 dakika.

Calistirma::

    python scripts/deney_kacan.py
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

GECEN_YAZ = ("t_gy_log_ort", "t_gy_sifir_orani", "t_gy_gun")
SEKIL = ("t_yayilma", "t_kayma", "t_hg_genligi")
PENCERE = ("ozet_pencere_gun", "t_doluluk")

ADAYLAR: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("TABAN (144 kolon)", ()),
    ("-t_gy_* (gecen yaz, 3)", GECEN_YAZ),
    ("-sekil (yayilma/kayma/genlik, 3)", SEKIL),
    ("-pencere (2)", PENCERE),
    ("-HEPSI (8)", GECEN_YAZ + SEKIL + PENCERE),
)


def main() -> int:
    satir_tamponlu_cikti()
    t_bas = time.time()
    print("=" * 100)
    print("ABLASYONDAN KACAN KOLONLAR -- sicak uzmani (maske %15), sicak satirlarda")
    print("=" * 100)
    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    # URETIM SETI (105). Taban tam set olsaydi, sekizinin degeri atilmis 46
    # kolonun VARLIGINDA olculurdu ve uretime tasinmazdi.
    kolonlar = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)

    kapsanan = set()
    for onek in d.AILELER.values():
        kapsanan |= {k for k in kolonlar if k.startswith(onek)}
    kacan = [k for k in kolonlar if k not in kapsanan]
    print(
        f"  onbellek {len(tum)} | URETIM SETI {len(kolonlar)} | ablasyonun kapsadigi"
        f" {len(kapsanan)} | KACAN {len(kacan)}"
    )
    print(f"  kacanlar: {kacan}")
    print("  hukum: eslenik t testi, |t| >= 2 (9 hucre: 3 blok x 3 tohum)")

    parcalar = {b.ad: di.blok_parcalari(egitim, b.ad) for b in tm.BLOKLAR}
    maskeli = {}
    for b in tm.BLOKLAR:
        for tohum in di.TOHUMLAR:
            maskeli[(b.ad, tohum)] = d.soguk_maskele(parcalar[b.ad][0], kolonlar, 0.15, tohum)

    taban_tekil: dict[tuple[str, int], float] = {}
    for ad, cikar in ADAYLAR:
        t0 = time.time()
        alt = [k for k in kolonlar if k not in cikar]
        skorlar, tekil = {}, {}
        for b in tm.BLOKLAR:
            _, dogrulama, gercek, soguk = parcalar[b.ad]
            tahminler = []
            for tohum in di.TOHUMLAR:
                log_t = di.egit_tahmin("cat", maskeli[(b.ad, tohum)], dogrulama, alt, tohum)
                tahminler.append(log_t)
                tek = np.clip(np.expm1(log_t), 0.0, None)
                tekil[(b.ad, tohum)] = tm.rmsle(gercek[~soguk], tek[~soguk])
            t = np.clip(np.expm1(np.mean(tahminler, axis=0)), 0.0, None)
            skorlar[b.ad] = tm.rmsle(gercek[~soguk], t[~soguk])
        genel = float(np.mean(list(skorlar.values())))
        if not taban_tekil:
            taban_tekil = tekil
            isaret = "TABAN"
        else:
            # pozitif fark = kolonlar FAYDALI (cikarinca skor kotulesti)
            farklar = np.array([tekil[k] - taban_tekil[k] for k in taban_tekil])
            ort = float(farklar.mean())
            sh = float(farklar.std(ddof=1) / np.sqrt(len(farklar)))
            t_deger = ort / sh if sh > 0 else 0.0
            hukum = ("FAYDALI" if ort > 0 else "ZARARLI") if abs(t_deger) >= 2.0 else "olculemez"
            isaret = f"{ort:+.5f} t={t_deger:+.2f} {hukum}"
        detay = "  ".join(f"{k} {v:.4f}" for k, v in skorlar.items())
        sure = f"({time.time() - t0:.0f} sn)"
        print(f"  {ad:34} [{len(alt):3d}] SICAK {genel:.5f}  {isaret:28} {detay}  {sure}")

    print("\n  pozitif fark = kolonlar FAYDALI (cikarinca skor kotulesti)")
    print("  negatif fark = kolonlar ZARARLI (cikarinca skor iyilesti)")
    print(f"\nTAMAM  {(time.time() - t_bas) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
