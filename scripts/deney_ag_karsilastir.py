"""SINIR AGI: agirligi, A5 ablasyonu ve toplulukta gercek degeri.

NEDEN
-----
Ag uretime TEK COMMIT ile girdi (317cfc7) ve hic ayarlanmadi. Uc somut bosluk:

1. ``sinir_agi: 1,4`` agirligi HICBIR olcum kaydinda gecmiyor. Sebebi
   ``deney_sicak_agirlik.py:18``te yazili: "sinir_agi izgaraya GIREMEZ (tek
   fit ~20 dakika, 27 fit imkansiz)". Yani izgaraya hic girmemis.

2. A1-A5 ablasyonlari ``sinir_agi.py:796-798``de CLI bayragi olarak TANIMLI
   ama ``experiments/`` altinda tek sonuc dosyasi yok. A5 ozellikle supheli:
   ``ayri_gosterge`` varsayilani ``False``, yani ``SimpleImputer(add_indicator=
   True)``in urettigi IKILI gosterge kolonlari ``QuantileTransformer``dan
   geciyor. Kuantil donusumu ikili kolonlarda anlamsizdir.

3. Agin harmana kattigi sey DOGRULUK mu CESITLILIK mi hic ayrilmadi.
   Krogh & Vedelsby (NeurIPS 1994) log uzayinda bir OZDESLIK verir:

       ortalama_uye_hatasi - AYRISMA = harman_hatasi

   Ayrisma = uyelerin harman etrafindaki agirlikli yayilimi. Bu betik onu
   dogrudan hesaplar: ag TEK BASINA kotu olsa bile ayrismayi buyutuyorsa
   agirligi artmali.

Hepsi ``scripts/aile_onbellegi.py``nin urettigi URETIM ESLI (ek kokenli)
onbellekten okunur -- egitim YOK, saf aritmetik.

    python scripts/aile_onbellegi.py --aile sinir_agi                    # uretim kolu
    python scripts/aile_onbellegi.py --aile sinir_agi --etiket a5 --ayri-gosterge
    python scripts/deney_ag_karsilastir.py
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

DIZIN = KOK / "data" / "interim" / "aile_onbellek"
KAYIT = KOK / "experiments" / "ag_karsilastir.jsonl"
TOHUMLAR = (1000, 1001, 1002)
GBDT = ("cat", "xgb", "lgbm")
GBDT_AGIRLIK = (3.0, 1.0, 1.0)
URETIM_AG = 1.4
AG_TARAMA = (0.0, 0.5, 1.0, 1.4, 2.0, 2.6, 3.4, 4.5, 6.0)
SICAK_KATSAYI = 0.7784 * 0.74263 / 1.07907


def _oku(blok: str, tohum: int, aile: str, etiket: str) -> np.ndarray | None:
    y = DIZIN / f"{blok}_{tohum}_{aile}_{etiket}.npy"
    return np.load(y).astype("float64") if y.exists() else None


def main() -> int:
    satir_tamponlu_cikti()
    t0 = time.time()
    print("=" * 100)
    print("SINIR AGI -- agirlik, A5 ablasyonu, toplulukta degeri")
    print("=" * 100)

    egitim, test = d.cerceveleri_kur()
    tm.kategorik_kodla(egitim, test)
    guc_kenar = ol.guc_kenarlari(test)
    te_s = test[test["soguk_mu"] != 1]

    veri = {}
    for b in tm.BLOKLAR:
        _, dogrulama, _, soguk = di.blok_parcalari(egitim, b.ad)
        w, _ = ol.test_agirliklari(dogrulama[~soguk], te_s, guc_kenar, eksenler=("bayatlik",))
        blok = {"gercek": np.load(DIZIN / f"{b.ad}_gercek.npy"), "w": w}
        for t in TOHUMLAR:
            for a in GBDT:
                blok[(t, a)] = _oku(b.ad, t, a, "uretim")
            blok[(t, "ag")] = _oku(b.ad, t, "sinir_agi", "uretim")
            blok[(t, "ag_a5")] = _oku(b.ad, t, "sinir_agi", "a5")
        veri[b.ad] = blok

    hazir = {
        ad: all(veri[b.ad][(t, ad)] is not None for b in tm.BLOKLAR for t in TOHUMLAR)
        for ad in ("ag", "ag_a5")
    }
    print(f"  uretim agi hazir: {hazir['ag']}   A5 agi hazir: {hazir['ag_a5']}")
    if not hazir["ag"]:
        eksik = [
            f"{b.ad}/{t}" for b in tm.BLOKLAR for t in TOHUMLAR if veri[b.ad][(t, "ag")] is None
        ]
        print(f"  URETIM AGI EKSIK: {eksik[:6]}{' ...' if len(eksik) > 6 else ''}")
        print("  once: python scripts/aile_onbellegi.py --aile sinir_agi")
        return 1

    def gbdt_log(blok: str, tohum: int) -> np.ndarray:
        top = sum(GBDT_AGIRLIK)
        return (
            sum(w * veri[blok][(tohum, a)] for a, w in zip(GBDT, GBDT_AGIRLIK, strict=True)) / top
        )

    def skorla(ag_adi: str | None, ag_w: float, blok: str | None = None):  # noqa: ANN202
        bloklar = [b.ad for b in tm.BLOKLAR] if blok is None else [blok]
        k_ham, k_ag, n_top, w_top = 0.0, 0.0, 0, 0.0
        for bad in bloklar:
            v = veri[bad]
            yig = []
            for t in TOHUMLAR:
                g = gbdt_log(bad, t)
                if ag_adi is None or ag_w == 0.0:
                    yig.append(g)
                else:
                    top = sum(GBDT_AGIRLIK) + ag_w
                    yig.append((g * sum(GBDT_AGIRLIK) + ag_w * v[(t, ag_adi)]) / top)
            tahmin = np.clip(np.expm1(np.mean(yig, axis=0)), 0.0, None)
            y, w = v["gercek"], v["w"]
            k_ham += ol.agirlikli_rmsle(y, tahmin) ** 2 * len(y)
            n_top += len(y)
            k_ag += ol.agirlikli_rmsle(y, tahmin, w) ** 2 * w.sum()
            w_top += w.sum()
        return float(np.sqrt(k_ham / n_top)), float(np.sqrt(k_ag / w_top))

    kayitlar = []

    # ------------------------------------------------ 1) agirlik taramasi
    print("\n" + "-" * 100)
    print("1) AG AGIRLIGI TARAMASI (GBDT 3/1/1 sabit) -- uretim 1,4 izgaraya hic girmemisti")
    print("-" * 100)
    print(f"  {'ag w':>7}{'ham':>10}{'agirlikli':>12}", end="")
    for b in tm.BLOKLAR:
        print(f"{b.ad:>11}", end="")
    print()
    for w in AG_TARAMA:
        h, g = skorla("ag", w)
        print(f"  {w:7.1f}{h:10.5f}{g:12.5f}", end="")
        blok_skor = []
        for b in tm.BLOKLAR:
            _, gb = skorla("ag", w, b.ad)
            blok_skor.append(gb)
            print(f"{gb:11.5f}", end="")
        print("  <- URETIM" if abs(w - URETIM_AG) < 1e-9 else "")
        kayitlar.append({"tur": "agirlik", "w": w, "ham": h, "agirlikli": g, "blok": blok_skor})

    u = next(k for k in kayitlar if k["tur"] == "agirlik" and abs(k["w"] - URETIM_AG) < 1e-9)
    en = min((k for k in kayitlar if k["tur"] == "agirlik"), key=lambda k: k["agirlikli"])
    print(f"\n  en iyi w={en['w']:.1f}  {en['agirlikli']:.5f}   uretim w=1,4 {u['agirlikli']:.5f}")
    fark = u["agirlikli"] - en["agirlikli"]
    print(f"  fark {fark:+.5f}   genel skora tahmini etki {-fark * SICAK_KATSAYI:+.5f}")
    kazanan = sum(1 for i in range(len(tm.BLOKLAR)) if en["blok"][i] < u["blok"][i])
    print(f"  BLOK TUTARLILIGI: {kazanan}/{len(tm.BLOKLAR)}")
    if kazanan < len(tm.BLOKLAR):
        print("  UYARI: uc blokta birden kazanmiyor -- 2026-08-22 dersi, hipotez SAYILMAZ.")

    # ------------------------------------------------------ 2) A5 ablasyonu
    if hazir["ag_a5"]:
        print("\n" + "-" * 100)
        print("2) A5 ABLASYONU (gostergeler QuantileTransformer'dan GECMEZ)")
        print("-" * 100)
        print(f"  {'ag w':>7}{'URETIM ag':>12}{'A5 ag':>10}{'fark':>10}")
        for w in (1.0, 1.4, 2.0, 2.6):
            _, gu = skorla("ag", w)
            _, ga = skorla("ag_a5", w)
            print(f"  {w:7.1f}{gu:12.5f}{ga:10.5f}{gu - ga:+10.5f}")
            kayitlar.append({"tur": "a5", "w": w, "uretim": gu, "a5": ga})
        print(f"\n  {'blok':>8}{'URETIM (w=1,4)':>16}{'A5 (w=1,4)':>13}{'fark':>10}")
        a5_kazanan = 0
        for b in tm.BLOKLAR:
            _, gu = skorla("ag", URETIM_AG, b.ad)
            _, ga = skorla("ag_a5", URETIM_AG, b.ad)
            a5_kazanan += 1 if ga < gu else 0
            print(f"  {b.ad:>8}{gu:16.5f}{ga:13.5f}{gu - ga:+10.5f}")
        print(f"  BLOK TUTARLILIGI: A5 {a5_kazanan}/{len(tm.BLOKLAR)}")
    else:
        print("\n  A5 onbellegi henuz yok -- 2. bolum atlandi.")

    # ------------------------------ 3) Krogh-Vedelsby: dogruluk mu cesitlilik mi
    print("\n" + "-" * 100)
    print("3) AGIN TOPLULUGA KATTIGI: dogruluk mu, cesitlilik mi?")
    print("-" * 100)
    print(
        "  Krogh & Vedelsby: ortalama_uye_hatasi - AYRISMA = harman_hatasi (log uzayinda ozdeslik)"
    )
    print(
        f"\n  {'blok':>8}{'ag tek':>10}{'GBDT harman':>13}{'AYRISMA(GBDT)':>15}{'AYRISMA(+ag)':>14}"
    )
    for b in tm.BLOKLAR:
        v = veri[b.ad]
        glog = np.log1p(np.clip(v["gercek"], 0.0, None))
        ag_tek = np.mean([v[(t, "ag")] for t in TOHUMLAR], axis=0)
        g_har = np.mean([gbdt_log(b.ad, t) for t in TOHUMLAR], axis=0)
        uyeler = [np.mean([veri[b.ad][(t, a)] for t in TOHUMLAR], axis=0) for a in GBDT]
        agr = np.array(GBDT_AGIRLIK) / sum(GBDT_AGIRLIK)
        ayr_g = float(sum(w * np.mean((m - g_har) ** 2) for w, m in zip(agr, uyeler, strict=True)))
        tam_agr = np.array([*GBDT_AGIRLIK, URETIM_AG])
        tam_agr = tam_agr / tam_agr.sum()
        tam_uye = [*uyeler, ag_tek]
        h_tam = sum(w * m for w, m in zip(tam_agr, tam_uye, strict=True))
        ayr_t = float(
            sum(w * np.mean((m - h_tam) ** 2) for w, m in zip(tam_agr, tam_uye, strict=True))
        )
        print(
            f"  {b.ad:>8}{float(np.sqrt(np.mean((glog - ag_tek) ** 2))):15.5f}"
            f"{float(np.sqrt(np.mean((glog - g_har) ** 2))):13.5f}{ayr_g:15.5f}{ayr_t:14.5f}"
        )
        kayitlar.append({"tur": "kv", "blok": b.ad, "ayrisma_gbdt": ayr_g, "ayrisma_agli": ayr_t})
    print("\n  AYRISMA buyudukce harman ortalama uyeden daha COK kazanir.")

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        for k in kayitlar:
            fh.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
