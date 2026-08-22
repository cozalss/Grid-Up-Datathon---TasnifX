"""SOGUK UZMANINA GRUP PROFILLERI -- yalin setin gomdugu yedek seviyeler.

NEDEN
-----
``g_*`` kolonlari ``trafo.py``de ACIKCA soguk trafolar icin tasarlanmis::

    "SOGUK trafolarin dusecegi yedek seviyeler. Trafo gecmisi yoksa geriye
     ne kaliyor: kurulu guc ve konum. Olculdu -- guc kovasi tek basina
     log-varyansin %26,2'sini, ilce %15,5'ini acikliyor. Ikisinin KESISIMI,
     ikisinin toplamindan fazlasini tasir."

        g_guc_kova        ordinal kVA kademesi
        g_kova_log_ort    kova ortalamasi
        g_ilce_log_ort    ilce ortalamasi
        g_ilce_kova_ort   ilce x kova ortalamasi (asil yedek)
        g_ilce_kova_n     o hucredeki egitim satiri sayisi (guven olcusu)

Sonra YALIN SET (2026-08-21 gecesi) onlari 39 kolonluk bir DEMET icinde
atti. O olcum YONLENDIRME ONCESI tek model uzerinde yapildi; soguk uzmani
(maske 1,00) diye bir sey henuz yoktu. Rejim bazinda hic olculmedi.

Ve soguk uzmaninin elinde HICBIR sey yok: butun ``t_*`` NaN. Grup
ortalamalari onun icin tek seviye kaynagi.

SIZINTI DENETIMI -- yapildi
    blok_kur:  profil_kaynak = tam_egitim[~etiket_maske]  (hedef blogun
               etiketleri HARIC)
    test_kur:  profil_kaynak = tam_egitim (test zaten sonrasinda)
Yani grup ortalamalari fold-disi. Temiz.

Adaylar (hepsi v20'nin soguk yapilandirmasi ustune: maske 1,00, d7,
+tk_haftanin_gunu, +tk_hafta_sonu)::

    TABAN       105 + 2 hafta kolonu
    +g_         5 grup seviyesi
    +gp_        3 grup profili (ilce x ay, ilce x haftagunu, kova x ay)
    +g_ +gp_    8

Fit: 4 aday x 3 blok x 3 tohum = 36 CatBoost ~ 55 dakika.

    python scripts/deney_soguk_grup.py
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

HAFTA = ("tk_haftanin_gunu", "tk_hafta_sonu")
GRUP = ("g_guc_kova", "g_kova_log_ort", "g_ilce_log_ort", "g_ilce_kova_ort", "g_ilce_kova_n")
PROFIL = ("gp_ilce_ay", "gp_ilce_hg", "gp_kova_ay")

ADAYLAR: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("TABAN (v20 soguk)", ()),
    ("+g_ (5)", GRUP),
    ("+gp_ (3)", PROFIL),
    ("+g_ +gp_ (8)", GRUP + PROFIL),
)

USTYAZIM: dict[str, object] = {"depth": 7}
SOGUK_MASKE = 1.00
KAYIT = KOK / "experiments" / "soguk_grup.jsonl"


def main() -> int:
    satir_tamponlu_cikti()
    t_bas = time.time()
    print("=" * 100)
    print("SOGUK UZMANINA GRUP PROFILLERI -- soguk satirlarda, v20 tabani")
    print("=" * 100)

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    taban = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)] + list(HAFTA)
    hepsi = list(GRUP + PROFIL)
    eksik = [k for k in hepsi + list(HAFTA) if k not in tum]
    if eksik:
        raise RuntimeError(f"onbellekte olmayan kolon: {eksik}")
    tm.kategorik_kodla(egitim, test)
    print(f"  taban {len(taban)} kolon (105 + 2 hafta)")

    parcalar = {b.ad: di.blok_parcalari(egitim, b.ad) for b in tm.BLOKLAR}
    maskeli = {}
    for b in tm.BLOKLAR:
        for tohum in di.TOHUMLAR:
            maskeli[(b.ad, tohum)] = d.soguk_maskele(
                parcalar[b.ad][0], taban + hepsi, SOGUK_MASKE, tohum
            )

    tekil: dict[str, dict[tuple[str, int], float]] = {}
    for ad, ek in ADAYLAR:
        t0 = time.time()
        kol = taban + list(ek)
        tekil[ad] = {}
        blok_skor = {}
        for b in tm.BLOKLAR:
            _, dogrulama, gercek, soguk = parcalar[b.ad]
            log_tahminler = []
            for tohum in di.TOHUMLAR:
                log_t = di.egit_tahmin(
                    "cat", maskeli[(b.ad, tohum)], dogrulama, kol, tohum, **USTYAZIM
                )
                log_tahminler.append(log_t)
                tek = np.clip(np.expm1(log_t), 0.0, None)
                tekil[ad][(b.ad, tohum)] = tm.rmsle(gercek[soguk], tek[soguk])
            harman = np.clip(np.expm1(np.mean(log_tahminler, axis=0)), 0.0, None)
            blok_skor[b.ad] = tm.rmsle(gercek[soguk], harman[soguk])
        ort = float(np.mean(list(blok_skor.values())))
        detay = "  ".join(f"{k} {v:.5f}" for k, v in blok_skor.items())
        print(f"  {ad:20} SOGUK {ort:.5f}   {detay}  ({time.time() - t0:.0f} sn)")

    taban_ad = ADAYLAR[0][0]
    kayitlar = []
    for ad, _ in ADAYLAR[1:]:
        f = np.array([tekil[taban_ad][k] - tekil[ad][k] for k in tekil[taban_ad]])
        o, sh = float(f.mean()), float(f.std(ddof=1) / np.sqrt(len(f)))
        t_d = o / sh if sh > 0 else 0.0
        hukum = ("KAZANDIRIYOR" if o > 0 else "ZARARLI") if abs(t_d) >= 2 else "esik alti"
        print(f"\n  {ad} vs TABAN: {o:+.5f}  SH {sh:.5f}  t {t_d:+.2f}   {hukum}")
        for b in tm.BLOKLAR:
            bb = np.array([tekil[taban_ad][(b.ad, t)] - tekil[ad][(b.ad, t)]
                           for t in di.TOHUMLAR])
            print(f"      {b.ad:6} {bb.mean():+.5f}  ({(bb > 0).sum()}/{len(bb)} tohum pozitif)")
        print(f"      genel skora etkisi {o * 0.350:+.5f}")
        kayitlar.append({"aday": ad, "fark": o, "sh": sh, "t": t_d, "hukum": hukum})

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        for k in kayitlar:
            fh.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"\nTAMAM  {(time.time() - t_bas) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
