"""URETIM ESLI AILE TAHMIN ONBELLEGI -- bir kez ode, sonra bedava sor.

NEDEN BU BETIK (loop 2026-08-24 sonunda yazildi, KOSULMADI)
-----------------------------------------------------------
Bugun uc kez ayni duvara carpildi: bir harman/agirlik sorusunu yanitlamak
TAM KOSU gerektiriyor.

1. ``data/interim/deney/sicak_tahmin.npz`` **ek_kokensiz** uretilmis
   (``deney_sicak_agirlik.py`` ``blok_parcalari`` kullaniyor), uretim sicak
   uzmani ise ``ek_koken: True``. Onbellek cat=0,80675 veriyor ve bu
   ``tuketim_model.py:844``teki "SICAK ANA 0,80675 -> EK 0,79848" satirinin
   ANA kolu. Aile siralamasi iki kolda TERS: ANA'da cat en iyi, EK'te xgb.
   Yani o onbellekten cikan hicbir sicak harman hukmu uretime tasinmaz.

2. ``sinir_agi`` HICBIR onbellekte yok. ``deney_sicak_agirlik.py:18`` bunu
   acikca yaziyor: "sinir_agi izgaraya GIREMEZ (tek fit ~20 dakika, 27 fit
   imkansiz)". Sonuc: uretimdeki ``sinir_agi: 1,4`` agirligi hicbir olcum
   kaydinda GECMIYOR -- izgaraya hic girmemis.

3. Agin A1-A5 ablasyonlari ``sinir_agi.py:796-798``de CLI bayragi olarak
   TANIMLI ama ``experiments/`` altinda tek sonuc dosyasi yok. Yani ag tek
   commit'te (317cfc7) uretime girdi ve hic ayarlanmadi.

Bu betik o duvari bir kez yikar: uretim ESLI (ek kokenli, ayni maske, ayni
ustyazim) aile tahminlerini diske yazar. Sonrasinda harman agirligi, ag
agirligi ve A5 ablasyonu SAF ARITMETIK olur.

MALIYET VE NEDEN BU LOOPTA KOSULMADI
------------------------------------
Blok x tohum basina: cat+xgb+lgbm ~2,5 dk (2,86M satir) + ag ~24 dk = ~32 dk.
3 blok x 3 tohum = **~4,7 saat**. 24 Agustos loopunda CPU 15->30 tohum
kuyruguna baglanmisti (garantili -0,00166) ve bu is oraya sigmadi.

KESINTIYE DAYANIKLI
-------------------
Her (blok, tohum, aile) parcasi TAMAMLANDIGI ANDA diske yazilir ve yeniden
kosuda ATLANIR. 4,7 saatlik bir isi tek parca halinde kosmak, ortasinda bir
kesinti olurs hepsini kaybetmek demektir.

    python scripts/aile_onbellegi.py                # eksikleri tamamlar
    python scripts/aile_onbellegi.py --aile cat,xgb,lgbm   # once ucuz olanlar
    python scripts/aile_onbellegi.py --durum        # ne var ne yok, kosmadan
"""

from __future__ import annotations

import argparse
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

DIZIN = KOK / "data" / "interim" / "aile_onbellek"
TOHUMLAR = (1000, 1001, 1002)
AILELER = ("cat", "xgb", "lgbm", "sinir_agi")

#: URETIM sicak uzmani (``REJIM_AYARLARI['sicak']``). Bu sozluk uretimden
#: SAPARSA onbellek gecersizdir -- degistirilirse dizin silinmeli.
SICAK_MASKE = 0.15
SICAK_CAT = {"random_strength": 4.0, "l2_leaf_reg": 1.0, "depth": 6}


def parca_yolu(blok: str, tohum: int, aile: str, etiket: str) -> Path:
    return DIZIN / f"{blok}_{tohum}_{aile}_{etiket}.npy"


def _ag_a5(parca, dogrulama, kolonlar, tohum):  # noqa: ANN001, ANN202
    """A5 ablasyonu: gostergeler QuantileTransformer'dan GECMEZ.

    ``ayri_gosterge`` ``tm.aile_tahmini`` uzerinden gecirilemiyor (o dal
    ``SinirAgi(tohum=..., hizli=...)`` diye kuruyor), bu yuzden ag burada
    ELDEN kuruluyor. Maskeleme ve kapasite ofseti uretim dalinin BIREBIR
    aynisi -- tek fark bayrak.
    """
    from sinir_agi import SinirAgi

    maskeli = tm.soguk_maskele(parca, kolonlar, tohum, SICAK_MASKE)
    y = tm.ofsetli_hedef(maskeli)
    ag = SinirAgi(tohum=tohum, hizli=False, ayri_gosterge=True)
    ag.fit(maskeli[kolonlar], y)
    return tm.ofseti_geri_ekle(ag.predict(dogrulama[kolonlar]), dogrulama)


def durum_yaz(etiket: str) -> None:
    print(f"\n  ONBELLEK DURUMU  ({DIZIN})")
    print(f"  {'blok':>8}{'tohum':>8}" + "".join(f"{a:>12}" for a in AILELER))
    eksik = 0
    for b in tm.BLOKLAR:
        for t in TOHUMLAR:
            print(f"  {b.ad:>8}{t:>8}", end="")
            for a in AILELER:
                var = parca_yolu(b.ad, t, a, etiket).exists()
                print(f"{'VAR' if var else 'eksik':>12}", end="")
                eksik += 0 if var else 1
            print()
    print(f"\n  eksik parca: {eksik} / {len(tm.BLOKLAR) * len(TOHUMLAR) * len(AILELER)}")


def main() -> int:
    satir_tamponlu_cikti()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aile", default=",".join(AILELER), help="virgulle ayrilmis aile listesi")
    ap.add_argument("--etiket", default="uretim", help="onbellek kolu adi (A5 icin 'a5' gibi)")
    ap.add_argument("--ayri-gosterge", action="store_true", help="agda A5 ablasyonu")
    ap.add_argument("--durum", action="store_true", help="yalnizca durum bas, kosma")
    ar = ap.parse_args()
    istenen = tuple(a.strip() for a in ar.aile.split(",") if a.strip())

    DIZIN.mkdir(parents=True, exist_ok=True)
    if ar.durum:
        durum_yaz(ar.etiket)
        return 0

    t_bas = time.time()
    print("=" * 96)
    print(f"URETIM ESLI AILE ONBELLEGI -- kol '{ar.etiket}', aileler {istenen}")
    print("=" * 96)

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    kol = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)

    # URETIM egitim seti: ek kokenli (REJIM_AYARLARI['sicak']['ek_koken']=True).
    ek = d._ek_kokenler_kur(False)
    ek = ek[ek["_blok"].isin([a for a, _, _ in tm.EK_KOKENLER])]
    ortak = [k for k in egitim.columns if k in ek.columns]
    genis = pd.concat([egitim[ortak], ek[ortak]], ignore_index=True)
    for k in tm.KATEGORIK:
        genis[k] = pd.Categorical(genis[k], categories=egitim[k].cat.categories)
    print(f"  ana {len(egitim):,} -> EK KOKENLI {len(genis):,} satir")

    for b in tm.BLOKLAR:
        _, dogrulama, gercek, soguk = di.blok_parcalari(egitim, b.ad)
        sicak = ~soguk
        # Hedef blokla KESISEN kokenler atilir -- ortusme dogrulamada sizinti.
        parca = tm.kokenleri_ayikla(genis, b.ad)
        gercek_yol = DIZIN / f"{b.ad}_gercek.npy"
        if not gercek_yol.exists():
            np.save(gercek_yol, gercek[sicak])
        print(f"\n  {b.ad}  egitim {len(parca):,}  dogrulama sicak {int(sicak.sum()):,}")

        for tohum in TOHUMLAR:
            gerekli = [a for a in istenen if not parca_yolu(b.ad, tohum, a, ar.etiket).exists()]
            if not gerekli:
                print(f"    tohum {tohum}: hepsi onbellekte, atlandi")
                continue
            for aile in gerekli:
                t0 = time.time()
                if aile == "sinir_agi" and ar.ayri_gosterge:
                    log_t = _ag_a5(parca, dogrulama, kol, tohum)
                else:
                    # URETIM YOLU. ``tm.aile_tahmini`` maskelemeyi ve kapasite
                    # ofsetini KENDI icinde yapiyor; sinir agini da bu yol
                    # tasiyor (``di.aile_modeli`` onu tanimaz, ValueError atar).
                    # Dort aileyi de ayni yoldan gecirmek sart -- yoksa tezgah
                    # yine uretimden ayrilir (bkz. docs/40 §3).
                    log_t = tm.aile_tahmini(
                        aile,
                        parca,
                        dogrulama,
                        kol,
                        tohum,
                        hizli=False,
                        maske_orani=SICAK_MASKE,
                        cat_ustyazim=SICAK_CAT if aile == "cat" else None,
                    )
                v = log_t[sicak] if log_t.shape[0] == soguk.size else log_t
                np.save(parca_yolu(b.ad, tohum, aile, ar.etiket), v.astype("float32"))
                print(f"    tohum {tohum}  {aile:10} yazildi  ({time.time() - t0:.0f} sn)")

    durum_yaz(ar.etiket)
    print(f"\nTAMAM  {(time.time() - t_bas) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
