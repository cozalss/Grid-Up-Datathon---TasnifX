"""5. UYE: ayni GBDT, HAM ``log1p(y)`` hedefiyle -- cesitlilik icin.

NEDEN
-----
``deney_ileri.egit_tahmin`` docstring'i (satir 226-231) sunu yaziyor:

    ``ofset=False`` ayni modeli HAM ``log1p(y)`` hedefiyle egitir. Tek
    basina daha kotu (olculdu: ofset -0,0352) ama harmanda CESITLILIK
    kaynagi: ASHRAE 1.'si tam bunu yapmis -- ``log1p(y/m2)`` ile egitilmis
    fazladan bir uye ekleyip "topluluga cesitlilik katti ve skoru ~0,002
    iyilestirdi" demis.

Bayrak VAR, gerekce YAZILI, ama tek olcum "TEK BASINA daha kotu" -- yani
tam da docstring'in uyardigi hata yapilmis. **Harman uyesi olarak hic
olculmedi.**

Bugunku Krogh-Vedelsby olcumu bunu tam olarak destekliyor
(``deney_ag_karsilastir.py`` §3): sinir agi tek basina GBDT harmanindan cok
daha kotu (kis26 0,873 vs 0,739) ama AYRISMAYI IKIYE KATLIYOR ve harmani
0,90065 -> 0,89718 cekiyor. Yani bu toplulukta cesitlilik, doğrulukdan daha
degerli -- ve elimizde bedava bir cesitlilik kaynagi duruyor.

TASARIM
-------
Iki kol da AYNI fonksiyondan (``di.egit_tahmin``) gecer; tek fark ``ofset``
bayragi. OFSETLI kol ayrica RIG SINAMASIDIR: uretim yolunun
(``tm.aile_tahmini``) urettigi onbellekle ayni sayiyi vermeli. Bugun iki kez
tezgah-uretim uyusmazligi yanlis yone cekti, ucuncusune izin yok.

    python scripts/deney_ofsetsiz_uye.py
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
import olcut as ol  # noqa: E402
import tuketim_model as tm  # noqa: E402

from gridup.reporting import satir_tamponlu_cikti  # noqa: E402

DIZIN = KOK / "data" / "interim" / "aile_onbellek"
KAYIT = KOK / "experiments" / "ofsetsiz_uye.jsonl"
TOHUMLAR = (1000, 1001, 1002)
GBDT = ("cat", "xgb", "lgbm")
GBDT_AGIRLIK = (3.0, 1.0, 1.0)
AG_AGIRLIK = 1.4
MASKE = 0.15
SICAK_CAT: dict[str, object] = {"random_strength": 4.0, "l2_leaf_reg": 1.0, "depth": 6}
YENI_AGIRLIKLAR = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.5)
SICAK_KATSAYI = 0.7784 * 0.74263 / 1.07907


def main() -> int:
    satir_tamponlu_cikti()
    t0 = time.time()
    print("=" * 100)
    print("5. UYE: ayni CatBoost, HAM log1p(y) hedefi -- cesitlilik icin")
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
    print(f"  ana {len(egitim):,} -> EK KOKENLI {len(genis):,} satir")

    veri = {}
    for b in tm.BLOKLAR:
        _, dogrulama, _, soguk = di.blok_parcalari(egitim, b.ad)
        sicak = ~soguk
        parca = tm.kokenleri_ayikla(genis, b.ad)
        w, _ = ol.test_agirliklari(dogrulama[~soguk], te_s, guc_kenar, eksenler=("bayatlik",))
        blok = {"gercek": np.load(DIZIN / f"{b.ad}_gercek.npy"), "w": w}
        for t in TOHUMLAR:
            for a in (*GBDT, "sinir_agi"):
                blok[(t, a)] = np.load(DIZIN / f"{b.ad}_{t}_{a}_uretim.npy").astype("float64")
        print(f"\n  {b.ad}  egitim {len(parca):,}  sicak {int(sicak.sum()):,}")
        for t in TOHUMLAR:
            maskeli = d.soguk_maskele(parca, kol, MASKE, t)
            for etiket, ofs in (("ofsetli", True), ("ofsetsiz", False)):
                yol = DIZIN / f"{b.ad}_{t}_cat_{etiket}.npy"
                if yol.exists():
                    blok[(t, etiket)] = np.load(yol).astype("float64")
                    continue
                t1 = time.time()
                log_t = di.egit_tahmin("cat", maskeli, dogrulama, kol, t, ofset=ofs, **SICAK_CAT)
                v = log_t[sicak] if log_t.shape[0] == soguk.size else log_t
                np.save(yol, v.astype("float32"))
                blok[(t, etiket)] = v.astype("float64")
                print(f"    tohum {t}  cat/{etiket:9} ({time.time() - t1:.0f} sn)")
        veri[b.ad] = blok

    # ------------------------------------------------------ RIG SINAMASI
    print("\n" + "-" * 100)
    print("RIG SINAMASI: di.egit_tahmin(ofset=True) == tm.aile_tahmini onbellegi mi?")
    print("-" * 100)
    print(f"  {'blok':>8}{'ort |fark|':>13}{'maks |fark|':>13}{'hukum':>10}")
    rig_tamam = True
    for b in tm.BLOKLAR:
        v = veri[b.ad]
        f = np.concatenate([np.abs(v[(t, "ofsetli")] - v[(t, "cat")]) for t in TOHUMLAR])
        ok = float(f.mean()) < 1e-6
        rig_tamam &= ok
        print(f"  {b.ad:>8}{f.mean():13.2e}{f.max():13.2e}{'TAMAM' if ok else 'SAPMA':>10}")
    if not rig_tamam:
        print("  NOT: iki yol ayni sayiyi vermiyor. Karsilastirma yine de ESLENIK")
        print("       (iki kol da di.egit_tahmin'den), ama uretim harmanina tasima")
        print("       yorumu bu sapma kadar zayiflar.")

    # ----------------------------------------------------- 5. uye taramasi
    def skorla(yeni_w: float, blok: str | None = None):  # noqa: ANN202
        bloklar = [b.ad for b in tm.BLOKLAR] if blok is None else [blok]
        k_ag, w_top = 0.0, 0.0
        for bad in bloklar:
            v = veri[bad]
            yig = []
            for t in TOHUMLAR:
                pay = (
                    sum(w * v[(t, a)] for a, w in zip(GBDT, GBDT_AGIRLIK, strict=True))
                    + AG_AGIRLIK * v[(t, "sinir_agi")]
                )
                top = sum(GBDT_AGIRLIK) + AG_AGIRLIK
                if yeni_w > 0:
                    pay = pay + yeni_w * v[(t, "ofsetsiz")]
                    top = top + yeni_w
                yig.append(pay / top)
            tahmin = np.clip(np.expm1(np.mean(yig, axis=0)), 0.0, None)
            y, w = v["gercek"], v["w"]
            k_ag += ol.agirlikli_rmsle(y, tahmin, w) ** 2 * w.sum()
            w_top += w.sum()
        return float(np.sqrt(k_ag / w_top))

    print("\n" + "-" * 100)
    print("5. UYE AGIRLIGI (uretim harmani 3/1/1 + ag 1,4 sabit)")
    print("-" * 100)
    print(f"  {'yeni w':>8}{'agirlikli':>12}", end="")
    for b in tm.BLOKLAR:
        print(f"{b.ad:>11}", end="")
    print()
    kayitlar = []
    for w in YENI_AGIRLIKLAR:
        g = skorla(w)
        blok_skor = [skorla(w, b.ad) for b in tm.BLOKLAR]
        print(f"  {w:8.1f}{g:12.5f}" + "".join(f"{x:11.5f}" for x in blok_skor), end="")
        print("  <- URETIM (5. uye YOK)" if w == 0.0 else "")
        kayitlar.append({"w": w, "agirlikli": g, "blok": blok_skor})

    taban = kayitlar[0]
    en = min(kayitlar, key=lambda k: k["agirlikli"])
    fark = taban["agirlikli"] - en["agirlikli"]
    kazanan = sum(1 for i in range(len(tm.BLOKLAR)) if en["blok"][i] < taban["blok"][i])
    print(f"\n  en iyi w={en['w']:.1f}  {en['agirlikli']:.5f}   uretim {taban['agirlikli']:.5f}")
    print(f"  fark {fark:+.5f}   genel skora tahmini etki {-fark * SICAK_KATSAYI:+.5f}")
    print(f"  BLOK TUTARLILIGI: {kazanan}/{len(tm.BLOKLAR)}")
    if kazanan < len(tm.BLOKLAR):
        print("  UYARI: uc blokta birden kazanmiyor -- hipotez SAYILMAZ.")

    # ------------------------------------------------- Krogh-Vedelsby
    print("\n" + "-" * 100)
    print("AYRISMA: 5. uye toplulugun cesitliligini ne kadar buyutuyor?")
    print("-" * 100)
    print(f"  {'blok':>8}{'5.uye tek':>12}{'AYR(uretim)':>14}{'AYR(+5.uye)':>14}")
    for b in tm.BLOKLAR:
        v = veri[b.ad]
        glog = np.log1p(np.clip(v["gercek"], 0.0, None))
        uye = [np.mean([v[(t, a)] for t in TOHUMLAR], axis=0) for a in (*GBDT, "sinir_agi")]
        yeni = np.mean([v[(t, "ofsetsiz")] for t in TOHUMLAR], axis=0)
        for etiket, uyeler, agr in (
            ("uretim", uye, [*GBDT_AGIRLIK, AG_AGIRLIK]),
            ("yeni", [*uye, yeni], [*GBDT_AGIRLIK, AG_AGIRLIK, en["w"] or 1.0]),
        ):
            a = np.array(agr, dtype="float64")
            a = a / a.sum()
            har = sum(wi * m for wi, m in zip(a, uyeler, strict=True))
            ayr = float(sum(wi * np.mean((m - har) ** 2) for wi, m in zip(a, uyeler, strict=True)))
            if etiket == "uretim":
                ayr_u = ayr
            else:
                ayr_y = ayr
        tek = float(np.sqrt(np.mean((glog - yeni) ** 2)))
        print(f"  {b.ad:>8}{tek:12.5f}{ayr_u:14.5f}{ayr_y:14.5f}")

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        for k in kayitlar:
            fh.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
