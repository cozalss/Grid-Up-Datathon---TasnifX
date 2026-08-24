"""SICAK HARMAN -- URETIM ESLI ONBELLEK UZERINDE, saf aritmetik.

NEDEN AYRI BIR BETIK
--------------------
``deney_olcut_kayma.py`` sicak harmani ``sicak_tahmin.npz`` uzerinde puanladi
ve "3/1/1 dort olcutte de en iyi" dedi. Sonra o onbellegin **ek_kokensiz**
oldugu ortaya cikti (docs/40 §3): uretim sicak uzmani ``ek_koken: True`` ile
2,86M satir goruyor, onbellek ise 1,04M. ek_koken aileleri ESIT OLMAYAN
olcude gucludiriyor (cat +0,0083, lgbm +0,0171, xgb +0,0327), yani aile
siralamasi iki kolda TERS.

Bu betik ayni sorulari ``scripts/aile_onbellegi.py``nin urettigi URETIM ESLI
onbellek uzerinde sorar. Egitim yok; her sey diskteki tahminlerden.

NE OLCULUYOR
------------
1. Tek aile skorlari -- siralama gercekten donuyor mu?
2. cat/xgb/lgbm harman izgarasi, k=1 ve k=3 torbalamada ayri.
3. ``sinir_agi`` onbellekte varsa: DORT uyeli izgara ve ag agirliginin
   taranmasi. Uretimdeki ``sinir_agi: 1,4`` hicbir olcum kaydinda gecmiyor
   -- izgaraya hic girmemis (``deney_sicak_agirlik.py:18``).

Skor iki olcutle birden: ham RMSLE ve ``olcut.py`` ile TESTE
AGIRLIKLANDIRILMIS RMSLE (bayatlik ekseni).

    python scripts/deney_harman_uretim.py
"""

from __future__ import annotations

import itertools
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

DIZIN = KOK / "data" / "interim" / "aile_onbellek"
KAYIT = KOK / "experiments" / "harman_uretim.jsonl"
TOHUMLAR = (1000, 1001, 1002)
ETIKET = "uretim"

#: ``d(genel)/d(sicak)`` -- bkz. docs/40.
SICAK_KATSAYI = 0.7784 * 0.74263 / 1.07907


def yukle(aileler: tuple[str, ...]) -> dict:
    veri = {}
    for b in tm.BLOKLAR:
        g = DIZIN / f"{b.ad}_gercek.npy"
        if not g.exists():
            raise SystemExit(f"onbellek eksik: {g}\n  once: python scripts/aile_onbellegi.py")
        blok = {"gercek": np.load(g)}
        for t in TOHUMLAR:
            for a in aileler:
                y = DIZIN / f"{b.ad}_{t}_{a}_{ETIKET}.npy"
                if not y.exists():
                    raise SystemExit(f"onbellek eksik: {y}")
                blok[(t, a)] = np.load(y).astype("float64")
        veri[b.ad] = blok
    return veri


def main() -> int:
    satir_tamponlu_cikti()
    t0 = time.time()
    print("=" * 100)
    print("SICAK HARMAN -- URETIM ESLI (ek kokenli) onbellek")
    print("=" * 100)

    agli = all(
        (DIZIN / f"{b.ad}_{t}_sinir_agi_{ETIKET}.npy").exists()
        for b in tm.BLOKLAR
        for t in TOHUMLAR
    )
    aileler = ("cat", "xgb", "lgbm") + (("sinir_agi",) if agli else ())
    print(f"  aileler: {aileler}" + ("" if agli else "   (sinir agi onbellegi HENUZ YOK)"))
    veri = yukle(aileler)

    egitim, test = d.cerceveleri_kur()
    tm.kategorik_kodla(egitim, test)
    guc_kenar = ol.guc_kenarlari(test)
    te_s = test[test["soguk_mu"] != 1]
    agirlik = {}
    for b in tm.BLOKLAR:
        _, dogrulama, _, soguk = di.blok_parcalari(egitim, b.ad)
        w, _ = ol.test_agirliklari(dogrulama[~soguk], te_s, guc_kenar, eksenler=("bayatlik",))
        agirlik[b.ad] = w

    def skorla(agr, tohumlar):  # noqa: ANN001, ANN202
        """(ham, agirlikli) -- bloklar satir sayisiyla havuzlanmis."""
        top = sum(agr)
        ham_k, ag_k, n_top, w_top = 0.0, 0.0, 0, 0.0
        for b in tm.BLOKLAR:
            v = veri[b.ad]
            yig = [
                sum(wi * v[(t, a)] for a, wi in zip(aileler, agr, strict=True)) / top
                for t in tohumlar
            ]
            tahmin = np.clip(np.expm1(np.mean(yig, axis=0)), 0.0, None)
            y, w = v["gercek"], agirlik[b.ad]
            ham_k += ol.agirlikli_rmsle(y, tahmin) ** 2 * len(y)
            n_top += len(y)
            ag_k += ol.agirlikli_rmsle(y, tahmin, w) ** 2 * w.sum()
            w_top += w.sum()
        return float(np.sqrt(ham_k / n_top)), float(np.sqrt(ag_k / w_top))

    # ------------------------------------------------------ tek aile
    print("\n" + "-" * 100)
    print("1) TEK AILE (k=3 torbalanmis) -- siralama ek_koken ile donuyor mu?")
    print("-" * 100)
    print(f"  {'aile':>12}{'ham':>10}{'agirlikli':>12}")
    for i, a in enumerate(aileler):
        agr = tuple(1.0 if j == i else 0.0 for j in range(len(aileler)))
        h, g = skorla(agr, TOHUMLAR)
        print(f"  {a:>12}{h:10.5f}{g:12.5f}")

    # -------------------------------------------------------- izgara
    print("\n" + "-" * 100)
    print("2) HARMAN IZGARASI")
    print("-" * 100)
    if agli:
        temel = [(3.0, 1.0, 1.0), (2.0, 2.0, 1.0), (3.0, 3.0, 1.0), (1.0, 1.0, 1.0)]
        ag_agirliklari = (0.0, 0.7, 1.4, 2.2, 3.0, 4.0)
        adaylar = [(*t, w) for t in temel for w in ag_agirliklari]
    else:
        adaylar = [t for t in itertools.product((0.0, 1.0, 2.0, 3.0), repeat=3) if sum(t) > 0]
    uretim = (3.0, 1.0, 1.0, 1.4) if agli else (3.0, 1.0, 1.0)

    sonuc = []
    for agr in adaylar:
        h1, g1 = skorla(agr, (1000,))
        h3, g3 = skorla(agr, TOHUMLAR)
        sonuc.append({"harman": list(agr), "ham1": h1, "ag1": g1, "ham3": h3, "ag3": g3})

    u = next(s for s in sonuc if tuple(s["harman"]) == uretim)
    print(
        f"  {'harman':>18}{'k1 ham':>10}{'k3 ham':>10}"
        f"{'k1 agir':>10}{'k3 agir':>10}{'k3agir-U':>11}"
    )
    for s in sorted(sonuc, key=lambda s: s["ag3"])[:12]:
        etik = "/".join(f"{x:g}" for x in s["harman"])
        bayrak = "  <- URETIM" if tuple(s["harman"]) == uretim else ""
        print(
            f"  {etik:>18}{s['ham1']:10.5f}{s['ham3']:10.5f}"
            f"{s['ag1']:10.5f}{s['ag3']:10.5f}{s['ag3'] - u['ag3']:+11.5f}{bayrak}"
        )

    # ----------------------------------------------- blok tutarliligi
    # Havuzlanmis skor tek basina ALDATIR: uretim harman degisikligi 2026-08-22'de
    # tam boyle secildi (izgara 0,0027 iyi dedi) ve URETIM DOGRULAMASINDA uc
    # blokta da kotu cikti (tuketim_model.py:875-879). Bir aday ancak UC BLOKTA
    # DA uretimi geciyorsa hipotez olur.
    print("\n" + "-" * 100)
    print("3) BLOK TUTARLILIGI (teste agirliklandirilmis, k=3) -- uc blokta da mi kazaniyor?")
    print("-" * 100)
    print(f"  {'harman':>18}", end="")
    for b in tm.BLOKLAR:
        print(f"{b.ad:>11}", end="")
    print(f"{'kazanan blok':>14}")
    en_iyi_uc = sorted(sonuc, key=lambda s: s["ag3"])[:3]
    for s in [*en_iyi_uc, u]:
        agr = tuple(s["harman"])
        etik = "/".join(f"{x:g}" for x in agr)
        kazanan = 0
        print(f"  {etik:>18}", end="")
        for b in tm.BLOKLAR:
            v, w = veri[b.ad], agirlik[b.ad]
            top = sum(agr)
            yig = [
                sum(wi * v[(t, a)] for a, wi in zip(aileler, agr, strict=True)) / top
                for t in TOHUMLAR
            ]
            tahmin = np.clip(np.expm1(np.mean(yig, axis=0)), 0.0, None)
            skor = ol.agirlikli_rmsle(v["gercek"], tahmin, w)
            yigu = [
                sum(wi * v[(t, a)] for a, wi in zip(aileler, uretim, strict=True)) / sum(uretim)
                for t in TOHUMLAR
            ]
            skoru = ol.agirlikli_rmsle(
                v["gercek"], np.clip(np.expm1(np.mean(yigu, axis=0)), 0.0, None), w
            )
            kazanan += 1 if skor < skoru else 0
            print(f"{skor:11.5f}", end="")
        etiket_k = "URETIM" if agr == uretim else f"{kazanan}/3"
        print(f"{etiket_k:>14}")

    print("\n  HUKUM")
    for anahtar, ad in (("ham3", "k=3 ham"), ("ag3", "k=3 teste agirliklandirilmis")):
        en = min(sonuc, key=lambda s: s[anahtar])
        etik = "/".join(f"{x:g}" for x in en["harman"])
        fark = u[anahtar] - en[anahtar]
        print(
            f"    {ad:30} en iyi {etik:>14} {en[anahtar]:.5f}   "
            f"uretim {u[anahtar]:.5f}   fark {fark:+.5f}"
        )
        if anahtar == "ag3":
            print(f"    {'':30} genel skora tahmini etki {-fark * SICAK_KATSAYI:+.5f}")

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        for s in sonuc:
            fh.write(json.dumps({"agli": agli, **s}, ensure_ascii=False) + "\n")
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
