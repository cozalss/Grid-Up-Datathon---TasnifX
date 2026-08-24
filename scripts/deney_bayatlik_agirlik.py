"""KOVARYAT KAYMA DUZELTMESI: bayat satirlari EGITIMDE yukselt.

BULGU (``deney_olcut_kayma.py`` + dagilim sayimi)
------------------------------------------------
Uretim egitim setinin bayatlik dagilimi testinkinden 15-16 KAT sapiyor:

    kova    EGITIM+EK   TEST     p_test/p_egitim
    0 gun     97,78     84,47     0,86
    1-6        0,51      7,76    15,2      <-
    7-29       0,71      0,34     0,48
    30-89      0,59      0,80     1,36
    90+        0,41      6,63    16,2      <-

Ve o iki kova sicak MSLE'nin %34'unu tasiyor (satirlarin %14,4'u):

    kis26 sicak   0 gun 0,748   1-6 1,041   90+ 1,585

Yani model, hatanin ucte birini ureten satir ailesini egitimde %0,9
oraninda goruyor ve onlari gurultu sayip taze satirlara gore optimize
ediyor.

NEDEN SON ISLEM DEGIL DE EGITIM
-------------------------------
Son islemle kaydirma DENENDI ve CURUDU (``deney_bayatlik.py``): kova basina
yanlilik buyuk (t=+11..+34) ama "0 gun" kovasinda ISARET bloktan bloga
donuyor (yaz25 +0,139, guz25 -0,356, kis26 +0,191) -- bu bir bayatlik
etkisi degil, MEVSIMSEL SEVIYE etkisi. Blok-disi kaydirma 0/3 kaybetti.
Ayni hastalik dun gece ``son_islem_gun.py``i LB'de curutmustu.

Bu deney farkli bir sinifta: agirliklar bir dogrulama blogunun SEVIYESINDEN
degil, DAGILIM SAYIMLARINDAN geliyor. Sayimlar gercek test cercevesinden
okunuyor, etiket gerekmiyor. Tohum ortalamasi da boyle model-disi bir
nicelikten turemisti ve LB'de ONGORDUGUNU getirmisti (docs/39 §6).

PROTOKOL
--------
Eslenik: her (blok, tohum) ciftinde iki kol AYNI maskelenmis egitim
parcasini gorur, tek fark ornek agirligi. Uc blok da SICAK satirlar icin
mesru (ezber kanali yalnizca sogugu kirletir, docs/35).

    python scripts/deney_bayatlik_agirlik.py --tohum 2
"""

from __future__ import annotations

import argparse
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
import olcut as ol  # noqa: E402
import tuketim_model as tm  # noqa: E402

from gridup.reporting import satir_tamponlu_cikti  # noqa: E402

KAYIT = KOK / "experiments" / "bayatlik_agirlik.jsonl"
ETIKET = ("0 gun", "1-6", "7-29", "30-89", "90+")

#: Uretimdeki sicak uzman ayarlari. ``thread_count`` arka plandaki tohum
#: kuyruguna cekirdek birakmak icin kisilir -- skoru degistirmez.
SICAK_USTYAZIM: dict[str, object] = {
    "random_strength": 4.0,
    "l2_leaf_reg": 1.0,
    "depth": 6,
    "thread_count": 5,
}
MASKE = 0.15

#: Agirlik tavani. 16x zaten buyuk; tavan tek bir kovanin kaybi ele
#: gecirmesini engeller.
TAVAN = 20.0

#: ``d(genel)/d(sicak)`` -- bkz. ``deney_bayatlik.py``.
SICAK_KATSAYI = 0.7784 * 0.74263 / 1.07907


def kova_agirliklari(egitim_cercevesi, test_sicak) -> tuple[np.ndarray, np.ndarray]:  # noqa: ANN001
    """Kova basina ``p_test / p_egitim``. ETIKET GEREKTIRMEZ."""
    ke = ol._kova(
        egitim_cercevesi["t_son_kayit_yasi"].to_numpy(dtype="float64"), ol.BAYATLIK_KENARLARI
    )
    kt = ol._kova(test_sicak["t_son_kayit_yasi"].to_numpy(dtype="float64"), ol.BAYATLIK_KENARLARI)
    pe = np.array([(ke == i).mean() for i in range(5)])
    pt = np.array([(kt == i).mean() for i in range(5)])
    oran = np.where(pe > 0, pt / np.maximum(pe, 1e-12), 1.0)
    return np.clip(oran, 1.0 / TAVAN, TAVAN), pe


def satir_agirligi(cerceve, oran) -> np.ndarray:  # noqa: ANN001
    """Maskelenmis egitim satirlarina agirlik.

    ``t_son_kayit_yasi`` NaN olan satirlar (dogal soguk + maskelenmisler)
    agirlik 1,0 alir: onlar bayatlik ekseninde YOK, maske duzenlileyicisi
    olarak duruyorlar ve kovaryat kayma duzeltmesi onlari ilgilendirmez.
    """
    k = ol._kova(cerceve["t_son_kayit_yasi"].to_numpy(dtype="float64"), ol.BAYATLIK_KENARLARI)
    w = np.where(k < 0, 1.0, oran[np.clip(k, 0, 4)])
    return w / w.mean()


def main() -> int:
    satir_tamponlu_cikti()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tohum", type=int, default=2, help="blok basina tohum sayisi")
    ar = ap.parse_args()
    tohumlar = tuple(1000 + i for i in range(ar.tohum))

    t_bas = time.time()
    print("=" * 104)
    print("BAYATLIK ONEM AGIRLIKLANDIRMASI -- egitimde kovaryat kayma duzeltmesi")
    print("=" * 104)

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    kol = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)
    guc_kenar = ol.guc_kenarlari(test)
    te_s = test[test["soguk_mu"] != 1]

    oran, pe = kova_agirliklari(egitim[egitim["soguk_mu"] != 1], te_s)
    print(f"\n  {'kova':>8}{'p_egitim %':>12}{'p_test %':>10}{'AGIRLIK':>10}")
    kt = ol._kova(te_s["t_son_kayit_yasi"].to_numpy(dtype="float64"), ol.BAYATLIK_KENARLARI)
    for i, ad in enumerate(ETIKET):
        print(f"  {ad:>8}{100 * pe[i]:12.2f}{100 * (kt == i).mean():10.2f}{oran[i]:10.2f}")

    # Kollar: (ad, us). ``us`` onem agirligini YUMUSATIR -- w**us. Tam
    # duzeltme (us=1) yansizdir ama varyansi buyuktur; us=0,5 klasik
    # "tempered importance weighting": yanliligin yarisini geri alir,
    # varyansin cogunu kirpar. Hangisinin daha iyi oldugu OLCULUR.
    kollar = (("TABAN", 0.0), ("AGIRLIK", 1.0), ("YUMUSAK", 0.5))
    kol_adlari = [k for k, _ in kollar]
    tekil: dict[str, dict] = {k: {} for k in kol_adlari}
    kova_skor: dict[str, dict] = {k: {} for k in kol_adlari}

    for b in tm.BLOKLAR:
        parca, dogrulama, gercek, soguk = di.blok_parcalari(egitim, b.ad)
        sicak = ~soguk
        dsicak = dogrulama[~soguk]
        y = gercek[sicak]
        kk = ol._kova(dsicak["t_son_kayit_yasi"].to_numpy(dtype="float64"), ol.BAYATLIK_KENARLARI)
        w_test, tani = ol.test_agirliklari(dsicak, te_s, guc_kenar, eksenler=("bayatlik",))
        print(f"\n  {b.ad}  sicak {len(y):,} satir  ESS %{100 * tani['ess_orani']:.0f}")

        for tohum in tohumlar:
            maskeli = d.soguk_maskele(parca, kol, MASKE, tohum)
            w_egitim = satir_agirligi(maskeli, oran)
            for kolad, us in kollar:
                t0 = time.time()
                a = None if us == 0.0 else (w_egitim**us) / (w_egitim**us).mean()
                log_t = di.egit_tahmin(
                    "cat", maskeli, dogrulama, kol, tohum, agirlik=a, **SICAK_USTYAZIM
                )
                tahmin = np.clip(
                    np.expm1(log_t[sicak] if log_t.shape[0] == soguk.size else log_t), 0.0, None
                )
                duz = ol.agirlikli_rmsle(y, tahmin)
                agir = ol.agirlikli_rmsle(y, tahmin, w_test)
                tekil[kolad][(b.ad, tohum)] = (duz, agir)
                kova_skor[kolad][(b.ad, tohum)] = [
                    ol.agirlikli_rmsle(y[kk == i], tahmin[kk == i])
                    if (kk == i).sum() >= 30
                    else np.nan
                    for i in range(5)
                ]
                print(
                    f"    tohum {tohum}  {kolad:8}  duz {duz:.5f}  agirlikli {agir:.5f}"
                    f"   ({time.time() - t0:.0f} sn)"
                )

    # ------------------------------------------------------------- hukum
    print("\n" + "-" * 104)
    print("KOVA BASINA (butun blok x tohum ortalamasi)")
    print("-" * 104)
    print(
        f"  {'kova':>8}{'TABAN':>10}{'AGIRLIK':>10}{'YUMUSAK':>10}"
        f"{'f-agir':>10}{'f-yumu':>10}{'test %':>10}"
    )
    for i, ad in enumerate(ETIKET):
        a = np.nanmean([v[i] for v in kova_skor["TABAN"].values()])
        b_ = np.nanmean([v[i] for v in kova_skor["AGIRLIK"].values()])
        c_ = np.nanmean([v[i] for v in kova_skor["YUMUSAK"].values()])
        print(
            f"  {ad:>8}{a:10.5f}{b_:10.5f}{c_:10.5f}"
            f"{a - b_:+10.5f}{a - c_:+10.5f}{100 * (kt == i).mean():10.2f}"
        )

    print("\n" + "-" * 104)
    print("ESLENIK FARK (TABAN - AGIRLIK; pozitif = agirliklandirma IYI)")
    print("-" * 104)
    kayitlar = []
    for kolad in ("AGIRLIK", "YUMUSAK"):
        print(f"\n  === {kolad} ===")
        for idx, ad in ((0, "duz"), (1, "teste agirliklandirilmis")):
            f = np.array([tekil["TABAN"][k][idx] - tekil[kolad][k][idx] for k in tekil["TABAN"]])
            o = float(f.mean())
            sh = float(f.std(ddof=1) / np.sqrt(len(f))) if len(f) > 1 else 0.0
            t_d = o / sh if sh > 0 else 0.0
            hukum = "AL" if t_d >= 2 else ("REDDET" if t_d <= -2 else "esik alti")
            print(f"  {ad:26} fark {o:+.5f}  SH {sh:.5f}  t {t_d:+.2f}   {hukum}")
            for b in tm.BLOKLAR:
                bb = np.array(
                    [
                        tekil["TABAN"][(b.ad, t)][idx] - tekil[kolad][(b.ad, t)][idx]
                        for t in tohumlar
                    ]
                )
                print(f"     {b.ad:6} {bb.mean():+.5f}  ({(bb > 0).sum()}/{len(bb)} tohum kazanc)")
            if idx == 1:
                print(f"     genel skora tahmini etki {-o * SICAK_KATSAYI:+.5f}")
            kayitlar.append(
                {"kol": kolad, "olcut": ad, "fark": o, "sh": sh, "t": t_d, "hukum": hukum}
            )

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        for k in kayitlar:
            fh.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"\nTAMAM  {(time.time() - t_bas) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
