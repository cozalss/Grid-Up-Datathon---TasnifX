"""EK KOKENLER hangi REJIME yariyor -- sicak mi, soguk mu.

NEDEN
-----
``deney_koken2.py`` ek kokenlerin +0,00782 (t=+3,01) kazandirdigini
olctu. Ama o olcum YONLENDIRMESIZ tek bir CatBoost'la yapildi (maske
0,15, butun satirlar). Uretimde iki ayri uzman var::

    sicak   maske 0,15   -> t_* kolonlarinin %85'i DOLU
    soguk   maske 1,00   -> t_* kolonlarinin TAMAMI NaN

Uretim kosusunda (v17, 2026-08-22) genel skor berabere kaldi ama bloklar
yer degistirdi ve kayip soguk tarafta gorundu.

HIPOTEZ
-------
Ek kokenler AYNI (trafo, gun) satirini farkli ozet pencereleriyle tekrar
gosteriyor. Sicak uzmani icin bu GERCEK veri artirmadir: ayni etiket,
gercekten farkli ``t_*`` ozetleriyle geliyor.

Soguk uzmani icin degil. Maske 1,00'da butun ``t_*`` NaN'lanir, yani ayni
satirin kopyalari arasinda geriye yalnizca ``ozet_pencere_gun``,
``t_doluluk`` ve ``ufuk_gun`` farki kalir; hedef ise BIREBIR ayni. Bu veri
artirma degil, KOPYA COGALTMA -- ve etiketleri tarih boyunca yeniden
agirliklandirir.

Dogruysa cozum basit: ek kokenleri YALNIZCA sicak uzmaninda kullan.

TASARIM
-------
Iki rejim ayri ayri, kendi uretim ayarlariyla, kendi satirlarinda::

    SICAK   maske 0,15   rs=4, l2=1, d6   ->  sicak satirlarda skorla
    SOGUK   maske 1,00   d7               ->  soguk satirlarda skorla

Her rejimde ANA (3 blok) ve EK KOKENLI karsilastirilir. Eslenik: ayni
blok, ayni tohum. Dogrulamada ``kokenleri_ayikla`` -- ortusme sizintidir.

Fit: 2 rejim x 2 aday x 3 blok x 3 tohum = 36 CatBoost ~ 45 dakika.

    python scripts/deney_koken_rejim.py
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

#: (ad, maske orani, CatBoost ustyazimi, soguk satirlarda mi skorlanacak)
REJIMLER: tuple[tuple[str, float, dict[str, object], bool], ...] = (
    ("SICAK", 0.15, {"random_strength": 4.0, "l2_leaf_reg": 1.0, "depth": 6}, False),
    ("SOGUK", 1.00, {"depth": 7}, True),
)

KAYIT = KOK / "experiments" / "koken_rejim.jsonl"


def main() -> int:
    satir_tamponlu_cikti()
    t_bas = time.time()
    print("=" * 100)
    print("EK KOKENLER x REJIM -- hangi uzmana yariyor")
    print("=" * 100)

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    kolonlar = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)

    ek = d._ek_kokenler_kur(False)
    eksik = [k for k in kolonlar if k not in ek.columns]
    if eksik:
        print(f"  UYARI: ek koken onbellegi BAYAT, {len(eksik)} kolon dusuruluyor")
        kolonlar = [k for k in kolonlar if k in ek.columns]
    ortak = [k for k in egitim.columns if k in ek.columns]
    genis = pd.concat([egitim[ortak], ek[ortak]], ignore_index=True)
    for k in tm.KATEGORIK:
        genis[k] = pd.Categorical(genis[k], categories=egitim[k].cat.categories)
    print(f"  {len(kolonlar)} kolon | ana {len(egitim):,} -> ek kokenli {len(genis):,}")

    kayitlar = []
    for rejim, maske, ustyazim, soguk_tarafi in REJIMLER:
        print(f"\n  ===== {rejim} (maske {maske}, {ustyazim}) =====")
        tekil: dict[str, dict[tuple[str, int], float]] = {}
        for ad, kaynak, ayikla in (("ANA", egitim, False), ("EK KOKENLI", genis, True)):
            t0 = time.time()
            tekil[ad] = {}
            blok_skor = {}
            for b in tm.BLOKLAR:
                dogrulama = egitim[egitim["_blok"] == b.ad]
                kalan = (
                    tm.kokenleri_ayikla(kaynak, b.ad) if ayikla else kaynak[kaynak["_blok"] != b.ad]
                )
                gercek = dogrulama[tm.HEDEF].to_numpy()
                soguk = (dogrulama["soguk_mu"] == 1).to_numpy()
                secim = soguk if soguk_tarafi else ~soguk
                log_tahminler = []
                for tohum in di.TOHUMLAR:
                    maskeli = d.soguk_maskele(kalan, kolonlar, maske, tohum)
                    log_t = di.egit_tahmin("cat", maskeli, dogrulama, kolonlar, tohum, **ustyazim)
                    log_tahminler.append(log_t)
                    tek = np.clip(np.expm1(log_t), 0.0, None)
                    tekil[ad][(b.ad, tohum)] = tm.rmsle(gercek[secim], tek[secim])
                harman = np.clip(np.expm1(np.mean(log_tahminler, axis=0)), 0.0, None)
                blok_skor[b.ad] = tm.rmsle(gercek[secim], harman[secim])
            ort = float(np.mean(list(blok_skor.values())))
            detay = "  ".join(f"{k} {v:.5f}" for k, v in blok_skor.items())
            print(f"    {ad:12} {ort:.5f}   {detay}  ({time.time() - t0:.0f} sn)")

        farklar = np.array([tekil["ANA"][k] - tekil["EK KOKENLI"][k] for k in tekil["ANA"]])
        o, sh = float(farklar.mean()), float(farklar.std(ddof=1) / np.sqrt(len(farklar)))
        t_deger = o / sh if sh > 0 else 0.0
        hukum = (
            "EK KOKEN KAZANDIRIYOR"
            if t_deger >= 2
            else ("EK KOKEN ZARAR VERIYOR" if t_deger <= -2 else "olculemedi")
        )
        print(f"    ESLENIK FARK (ANA - EK) {o:+.5f}  SH {sh:.5f}  t {t_deger:+.2f}   {hukum}")
        for b in tm.BLOKLAR:
            f = np.array(
                [tekil["ANA"][(b.ad, t)] - tekil["EK KOKENLI"][(b.ad, t)] for t in di.TOHUMLAR]
            )
            print(f"      {b.ad:6} {f.mean():+.5f}")
        kayitlar.append({"rejim": rejim, "fark": o, "sh": sh, "t": t_deger, "hukum": hukum})

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as f:
        for k in kayitlar:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")

    print(f"\nTAMAM  {(time.time() - t_bas) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
