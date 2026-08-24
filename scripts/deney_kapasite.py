"""SICAK UZMAN KAPASITESI: ayar 1,04M satirda tarandi, uretim 2,86M goruyor.

BULGU
-----
``tuketim_model.aile_modeli`` CatBoost tabani ``iterations=250``,
``learning_rate=0,05``, ``depth=5``; sicak uzman ustyazimi ``depth=6``,
``l2_leaf_reg=1``, ``random_strength=4``.

Bu ayarlar ``deney_ayar2.py`` ile tarandi ve o betik (satir 101)
``di.blok_parcalari`` kullaniyor -- yani **ek kokensiz** 1,04M satirlik kol.
Uretim sicak uzmani ise ``ek_koken: True`` ile **2,86M** satir goruyor.

Bugun ayni rig uyusmazliginin aile SIRALAMASINI cevirdigi olculdu (docs/40 §3):
ek koken aileleri esit olmayan olcude gucludiriyor (cat +0,0083, lgbm +0,0171,
xgb +0,0327). Kapasite parametreleri (agac sayisi, derinlik) veri boyutuna en
duyarli olanlardir: uc kat veride 250 agac yetersiz uydurma adayidir.

SOGUK UZMAN BU SORUNDAN MUAF: ``ek_koken: False``, yani onun tarama rig'i
uretimle zaten eslesiyor. Sorun YALNIZCA sicak tarafta.

PROTOKOL
--------
Uretim esli (ek kokenli) egitim seti, uretim maskesi ve ustyazimi. Her
(blok, tohum, kol) tahmini diske yazilir -- kesintiye dayanikli ve sonraki
sorular bedava. Hukum ``olcut.py`` ile TESTE AGIRLIKLANDIRILMIS skor uzerinde
ve **blok kirilimi ile** verilir: bugun havuzlanmis skor uc kez kandirdi.

    python scripts/deney_kapasite.py
    python scripts/deney_kapasite.py --kollar 250,500      # yalniz ikisi
"""

from __future__ import annotations

import argparse
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
import olcut as ol  # noqa: E402
import tuketim_model as tm  # noqa: E402

from gridup.reporting import satir_tamponlu_cikti  # noqa: E402

DIZIN = KOK / "data" / "interim" / "aile_onbellek"
KAYIT = KOK / "experiments" / "kapasite.jsonl"
TOHUMLAR = (1000, 1001, 1002)
GBDT = ("cat", "xgb", "lgbm")
GBDT_AGIRLIK = (3.0, 1.0, 1.0)
AG_AGIRLIK = 1.4
MASKE = 0.15

#: Kollar: ad -> CatBoost ustyazimi. "250" URETIM.
KOLLAR: dict[str, dict[str, object]] = {
    "250": {"iterations": 250, "depth": 6},
    "500": {"iterations": 500, "depth": 6},
    "900": {"iterations": 900, "depth": 6},
    "500d7": {"iterations": 500, "depth": 7},
}
ORTAK_CAT: dict[str, object] = {"random_strength": 4.0, "l2_leaf_reg": 1.0}
SICAK_KATSAYI = 0.7784 * 0.74263 / 1.07907


def main() -> int:
    satir_tamponlu_cikti()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kollar", default=",".join(KOLLAR), help="virgulle ayrilmis kol adlari")
    ar = ap.parse_args()
    istenen = [k.strip() for k in ar.kollar.split(",") if k.strip() in KOLLAR]

    t0 = time.time()
    print("=" * 100)
    print("SICAK UZMAN KAPASITESI -- ayar 1,04M'de tarandi, uretim 2,86M goruyor")
    print("=" * 100)

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    kol = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)
    guc_kenar = ol.guc_kenarlari(test)
    te_s = test[test["soguk_mu"] != 1]

    ek = d._ek_kokenler_kur(False)
    ek = ek[ek["_blok"].isin([a for a, _, _ in tm.EK_KOKENLER])]
    ortak = [k for k in egitim.columns if k in ek.columns]
    genis = pd.concat([egitim[ortak], ek[ortak]], ignore_index=True)
    for k in tm.KATEGORIK:
        genis[k] = pd.Categorical(genis[k], categories=egitim[k].cat.categories)
    print(f"  ana {len(egitim):,} -> EK KOKENLI {len(genis):,} satir   kollar: {istenen}")

    veri = {}
    for b in tm.BLOKLAR:
        _, dogrulama, _, soguk = di.blok_parcalari(egitim, b.ad)
        sicak = ~soguk
        parca = tm.kokenleri_ayikla(genis, b.ad)
        w, _ = ol.test_agirliklari(dogrulama[~soguk], te_s, guc_kenar, eksenler=("bayatlik",))
        blok = {"gercek": np.load(DIZIN / f"{b.ad}_gercek.npy"), "w": w}
        for t in TOHUMLAR:
            for a in ("xgb", "lgbm", "sinir_agi"):
                blok[(t, a)] = np.load(DIZIN / f"{b.ad}_{t}_{a}_uretim.npy").astype("float64")
        print(f"\n  {b.ad}  egitim {len(parca):,}  sicak {int(sicak.sum()):,}")
        for t in TOHUMLAR:
            maskeli = None
            for kol_ad in istenen:
                yol = DIZIN / f"{b.ad}_{t}_cat_kap{kol_ad}.npy"
                if yol.exists():
                    blok[(t, kol_ad)] = np.load(yol).astype("float64")
                    continue
                if maskeli is None:
                    maskeli = d.soguk_maskele(parca, kol, MASKE, t)
                t1 = time.time()
                ust = {**ORTAK_CAT, **KOLLAR[kol_ad]}
                log_t = di.egit_tahmin("cat", maskeli, dogrulama, kol, t, **ust)
                v = log_t[sicak] if log_t.shape[0] == soguk.size else log_t
                np.save(yol, v.astype("float32"))
                blok[(t, kol_ad)] = v.astype("float64")
                print(f"    tohum {t}  cat/{kol_ad:6} ({time.time() - t1:.0f} sn)")
        veri[b.ad] = blok

    def skorla(kol_ad: str, blok: str | None = None):  # noqa: ANN202
        bloklar = [b.ad for b in tm.BLOKLAR] if blok is None else [blok]
        k_ag, w_top = 0.0, 0.0
        top = sum(GBDT_AGIRLIK) + AG_AGIRLIK
        for bad in bloklar:
            v = veri[bad]
            yig = []
            for t in TOHUMLAR:
                pay = (
                    GBDT_AGIRLIK[0] * v[(t, kol_ad)]
                    + GBDT_AGIRLIK[1] * v[(t, "xgb")]
                    + GBDT_AGIRLIK[2] * v[(t, "lgbm")]
                    + AG_AGIRLIK * v[(t, "sinir_agi")]
                )
                yig.append(pay / top)
            tahmin = np.clip(np.expm1(np.mean(yig, axis=0)), 0.0, None)
            y, w = v["gercek"], v["w"]
            k_ag += ol.agirlikli_rmsle(y, tahmin, w) ** 2 * w.sum()
            w_top += w.sum()
        return float(np.sqrt(k_ag / w_top))

    print("\n" + "-" * 100)
    print("HUKUM (teste agirliklandirilmis, uretim harmani icinde)")
    print("-" * 100)
    print(f"  {'kol':>8}{'agirlikli':>12}", end="")
    for b in tm.BLOKLAR:
        print(f"{b.ad:>11}", end="")
    print(f"{'250e gore':>12}")
    kayitlar = []
    taban = skorla("250") if "250" in istenen else None
    for kol_ad in istenen:
        g = skorla(kol_ad)
        blok_skor = [skorla(kol_ad, b.ad) for b in tm.BLOKLAR]
        fark = (taban - g) if taban is not None else 0.0
        print(
            f"  {kol_ad:>8}{g:12.5f}" + "".join(f"{x:11.5f}" for x in blok_skor) + f"{fark:+12.5f}",
            end="",
        )
        print("  <- URETIM" if kol_ad == "250" else "")
        kayitlar.append({"kol": kol_ad, "agirlikli": g, "blok": blok_skor})

    if taban is not None and len(kayitlar) > 1:
        tb = next(k for k in kayitlar if k["kol"] == "250")
        en = min(kayitlar, key=lambda k: k["agirlikli"])
        fark = tb["agirlikli"] - en["agirlikli"]
        kazanan = sum(1 for i in range(len(tm.BLOKLAR)) if en["blok"][i] < tb["blok"][i])
        print(f"\n  en iyi {en['kol']}  {en['agirlikli']:.5f}   uretim {tb['agirlikli']:.5f}")
        print(f"  fark {fark:+.5f}   genel skora tahmini etki {-fark * SICAK_KATSAYI:+.5f}")
        print(f"  BLOK TUTARLILIGI: {kazanan}/{len(tm.BLOKLAR)}")
        if kazanan == len(tm.BLOKLAR):
            print("  UC BLOKTA DA KAZANIYOR -- hipotez GECERLI, uretim kosusuna deger.")
        else:
            print("  UYARI: uc blokta birden kazanmiyor -- hipotez SAYILMAZ.")

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        for k in kayitlar:
            fh.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
