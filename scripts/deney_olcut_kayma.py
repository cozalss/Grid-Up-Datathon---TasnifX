"""OLCUTUN KENDISINI SINA: bayatlik kaymasi + sicak harmanin torbalama bagimliligi.

IKI SORU BIRDEN, IKISI DE ONBELLEKTEN (egitim YOK, belirlenimci)
---------------------------------------------------------------

SORU 1 -- BAYATLIK KAYMASI NE KADAR BUYUK?
docs/39 §8 sunu olcmus ama uzerine hicbir sey yapilmamis:

    t_son_kayit_yasi >= 1   TEST sicak %15,5   CV %1,7
    kis26 sicak skoru       0,77882 -> 0,87811

Burada eksen tabakalara ayrilip hatanin nerede toplandigi cikarilir ve
``olcut.py`` ile testin ORTAK dagilimina agirliklandirilmis skor hesaplanir.

SORU 2 -- SICAK HARMAN AGIRLIGI YANLIS OLCULMUS OLABILIR.
``tuketim_model.py`` satir 874-882: izgara (2,2,1)'i (3,1,1)'den 0,0027 iyi
buldu, uretim dogrulamasi REDDETTI. Gerekce: "izgara 3 tohum TORBALANMIS
tahminler uzerindeydi, uretimin dogrulama adimi ise TEK tohum (42)".

Ama URETIM 15 TOHUM ORTALIYOR. Yani GONDERILEN sey torbalanmis; tek tohumlu
dogrulama gonderileni olcmuyor. Reddetme gerekcesi tersine donmus olabilir.

Burada ayni onbellek uzerinde k=1 ve k=3 icin optimal agirlik AYRI hesaplanir.
Optimum k ile kayiyorsa, 15 tohumlu uretim icin dogru cevap k buyuk taraftir.

    python scripts/deney_olcut_kayma.py
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
import olcut as ol  # noqa: E402
import tuketim_model as tm  # noqa: E402

from gridup.reporting import satir_tamponlu_cikti  # noqa: E402

ONBELLEK = KOK / "data" / "interim" / "deney" / "sicak_tahmin.npz"
KAYIT = KOK / "experiments" / "olcut_kayma.jsonl"
TOHUMLAR = (1000, 1001, 1002)
AILELER = ("cat", "xgb", "lgbm")

#: Taranacak sicak harman agirliklari (cat, xgb, lgbm).
HARMANLAR = (
    (3.0, 1.0, 1.0),  # URETIM
    (2.0, 2.0, 1.0),  # izgaranin sectigi, uretim dogrulamasinin reddettigi
    (3.0, 3.0, 1.0),
    (1.0, 1.0, 1.0),
    (0.0, 1.0, 0.0),  # yalniz xgb -- tek basina en iyi aile
    (1.0, 2.0, 1.0),
    (1.0, 1.0, 0.0),
)


def _blend(z, blok: str, tohumlar, w) -> np.ndarray:  # noqa: ANN001
    """Log uzayinda tohum-ortalamali aile harmani."""
    toplam = sum(w)
    yig = []
    for t in tohumlar:
        h = sum(wi * z[f"{blok}_{t}_{a}"] for a, wi in zip(AILELER, w, strict=True)) / toplam
        yig.append(h)
    return np.mean(yig, axis=0)


def main() -> int:
    satir_tamponlu_cikti()
    t0 = time.time()
    print("=" * 100)
    print("OLCUT SINAMASI -- bayatlik kaymasi + sicak harmanin torbalama bagimliligi")
    print("=" * 100)

    egitim, test = d.cerceveleri_kur()
    tm.kategorik_kodla(egitim, test)
    guc_kenar = ol.guc_kenarlari(test)
    z = np.load(ONBELLEK)

    # ---------------------------------------------------------- SORU 1
    print("\n" + "-" * 100)
    print("SORU 1  BAYATLIK DAGILIMI (t_son_kayit_yasi), SICAK satirlar")
    print("-" * 100)
    etiket = ("0 gun", "1-6", "7-29", "30-89", "90+")
    te_sicak = test[test["soguk_mu"] != 1]
    tk = ol._kova(te_sicak["t_son_kayit_yasi"].to_numpy(dtype="float64"), ol.BAYATLIK_KENARLARI)
    tp = np.array([(tk == i).mean() for i in range(5)])
    print(f"  {'kova':>8}{'TEST %':>10}", end="")
    for b in tm.BLOKLAR:
        print(f"{b.ad + ' %':>11}", end="")
    print(f"{'TEST/CV':>10}")

    blok_paylar = {}
    for b in tm.BLOKLAR:
        _, dogrulama, _, soguk = di.blok_parcalari(egitim, b.ad)
        dsicak = dogrulama[~soguk]
        kk = ol._kova(dsicak["t_son_kayit_yasi"].to_numpy(dtype="float64"), ol.BAYATLIK_KENARLARI)
        blok_paylar[b.ad] = np.array([(kk == i).mean() for i in range(5)])

    for i, ad in enumerate(etiket):
        print(f"  {ad:>8}{100 * tp[i]:10.2f}", end="")
        for b in tm.BLOKLAR:
            print(f"{100 * blok_paylar[b.ad][i]:11.2f}", end="")
        ort_cv = np.mean([blok_paylar[b.ad][i] for b in tm.BLOKLAR])
        oran = tp[i] / ort_cv if ort_cv > 1e-9 else np.inf
        print(f"{oran:10.1f}x")

    # ------------------------------------------------- SORU 1b: hata nerede
    print("\n" + "-" * 100)
    print("SORU 1b  SICAK HATA BAYATLIGA GORE TABAKALI  (uretim harmani 3/1/1, 3 tohum)")
    print("-" * 100)
    kayitlar = []
    for b in tm.BLOKLAR:
        _, dogrulama, gercek, soguk = di.blok_parcalari(egitim, b.ad)
        sicak = ~soguk
        n_blok = int(soguk.size)
        p = _blend(z, b.ad, TOHUMLAR, (3.0, 1.0, 1.0))
        if p.shape[0] == n_blok:
            p_sicak, y_sicak = p[sicak], gercek[sicak]
            dsicak = dogrulama[~soguk]
        elif p.shape[0] == int(sicak.sum()):
            p_sicak, y_sicak = p, gercek[sicak]
            dsicak = dogrulama[~soguk]
        else:
            raise RuntimeError(
                f"{b.ad}: onbellek {p.shape[0]} != blok {n_blok} / sicak {sicak.sum()}"
            )
        tahmin = np.clip(np.expm1(p_sicak), 0.0, None)
        kk = ol._kova(dsicak["t_son_kayit_yasi"].to_numpy(dtype="float64"), ol.BAYATLIK_KENARLARI)
        duz = ol.agirlikli_rmsle(y_sicak, tahmin)
        print(f"\n  {b.ad}   sicak {len(y_sicak):,} satir   duz RMSLE {duz:.5f}")
        print(
            f"  {'kova':>8}{'satir':>9}{'CV pay':>8}{'RMSLE':>9}{'TEST pay':>10}{'MSLE katki':>12}"
        )
        for i, ad in enumerate(etiket):
            m = kk == i
            if m.sum() == 0:
                print(f"  {ad:>8}{0:9d}{0.0:9.2f}{'--':>9}{100 * tp[i]:11.2f}{'--':>12}")
                continue
            r = ol.agirlikli_rmsle(y_sicak[m], tahmin[m])
            katki = tp[i] * r**2
            print(
                f"  {ad:>8}{int(m.sum()):9,}{100 * m.mean():9.2f}{r:9.5f}"
                f"{100 * tp[i]:11.2f}{katki:12.5f}"
            )
            kayitlar.append(
                {"blok": b.ad, "kova": ad, "n": int(m.sum()), "rmsle": r, "test_pay": float(tp[i])}
            )

    # ------------------------------------------- SORU 1c: agirliklandirilmis skor
    print("\n" + "-" * 100)
    print("SORU 1c  TESTE AGIRLIKLANDIRILMIS SICAK SKOR (uretim harmani)")
    print("-" * 100)
    print(
        f"  {'blok':>8}{'duz':>10}{'bayatlik':>10}{'ortak':>10}"
        f"{'ESS%':>8}{'kirp%':>8}{'kapsyok%':>10}"
    )
    agirlik_onbellek = {}
    for b in tm.BLOKLAR:
        _, dogrulama, gercek, soguk = di.blok_parcalari(egitim, b.ad)
        sicak = ~soguk
        dsicak, y_sicak = dogrulama[~soguk], gercek[sicak]
        p = _blend(z, b.ad, TOHUMLAR, (3.0, 1.0, 1.0))
        p_sicak = p[sicak] if p.shape[0] == soguk.size else p
        tahmin = np.clip(np.expm1(p_sicak), 0.0, None)
        te_s = test[test["soguk_mu"] != 1]
        w1, t1 = ol.test_agirliklari(dsicak, te_s, guc_kenar, eksenler=("bayatlik",))
        w3, t3 = ol.test_agirliklari(dsicak, te_s, guc_kenar, eksenler=("bayatlik", "guc", "ufuk"))
        agirlik_onbellek[b.ad] = (w1, w3, sicak, y_sicak)
        print(
            f"  {b.ad:>8}{ol.agirlikli_rmsle(y_sicak, tahmin):10.5f}"
            f"{ol.agirlikli_rmsle(y_sicak, tahmin, w1):10.5f}"
            f"{ol.agirlikli_rmsle(y_sicak, tahmin, w3):10.5f}"
            f"{100 * t3['ess_orani']:8.1f}{100 * t3['kirpilan']:8.2f}"
            f"{100 * t3['kapsanmayan']:10.2f}"
        )

    # ---------------------------------------------------------- SORU 2
    print("\n" + "-" * 100)
    print("SORU 2  SICAK HARMAN: optimum agirlik TORBALAMA ile kayiyor mu?")
    print("-" * 100)
    print("  k=1: tek tohum (1000).  k=3: uc tohum log uzayinda ortalanmis (uretim gibi).")
    print(f"\n  {'harman':>12}{'k=1 duz':>10}{'k=3 duz':>10}{'k=1 agir':>11}{'k=3 agir':>11}")
    sonuc = []
    for w in HARMANLAR:
        satir = {"harman": w}
        for k, tohumlar in ((1, (1000,)), (3, TOHUMLAR)):
            duz_kare, agir_kare, agir_tp = 0.0, 0.0, 0.0
            for b in tm.BLOKLAR:
                w1, _w3, sicak, y_sicak = agirlik_onbellek[b.ad]
                p = _blend(z, b.ad, tohumlar, w)
                p_sicak = p[sicak] if p.shape[0] == sicak.size else p
                tahmin = np.clip(np.expm1(p_sicak), 0.0, None)
                duz_kare += ol.agirlikli_rmsle(y_sicak, tahmin) ** 2
                agir_kare += ol.agirlikli_rmsle(y_sicak, tahmin, w1) ** 2 * w1.sum()
                agir_tp += w1.sum()
            satir[f"duz{k}"] = float(np.sqrt(duz_kare / len(tm.BLOKLAR)))
            satir[f"agir{k}"] = float(np.sqrt(agir_kare / agir_tp))
        etik = "/".join(str(int(x)) for x in w)
        bayrak = "  <- URETIM" if w == (3.0, 1.0, 1.0) else ""
        print(
            f"  {etik:>12}{satir['duz1']:10.5f}{satir['duz3']:10.5f}"
            f"{satir['agir1']:11.5f}{satir['agir3']:11.5f}{bayrak}"
        )
        sonuc.append(satir)

    print("\n  HUKUM")
    for anahtar, ad in (
        ("duz1", "k=1 duz"),
        ("duz3", "k=3 duz"),
        ("agir1", "k=1 agirlikli"),
        ("agir3", "k=3 agirlikli"),
    ):
        en_iyi = min(sonuc, key=lambda s: s[anahtar])
        uretim = next(s for s in sonuc if s["harman"] == (3.0, 1.0, 1.0))
        etik = "/".join(str(int(x)) for x in en_iyi["harman"])
        print(
            f"    {ad:>14}: en iyi {etik:>8}  {en_iyi[anahtar]:.5f}   "
            f"uretim(3/1/1) {uretim[anahtar]:.5f}   fark {uretim[anahtar] - en_iyi[anahtar]:+.5f}"
        )

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        for k in kayitlar:
            fh.write(json.dumps({"tur": "tabaka", **k}, ensure_ascii=False) + "\n")
        for s in sonuc:
            fh.write(
                json.dumps(
                    {
                        "tur": "harman",
                        **{k: (list(v) if isinstance(v, tuple) else v) for k, v in s.items()},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
