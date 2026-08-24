"""pg10'un DIGER AILELERDEKI etkisi -- olctugum sey gonderdigim sey mi?

BULUNAN BOSLUK
--------------
``ek_kolon`` uretimde rejimin BUTUN ailelerine uygulanir: ``rejim_tahmini``
icinde ``kol = kolonlar + ek_kolon`` kurulur ve o liste cat, xgb, lgbm ve
sinir agina birlikte gider.

Ama ``deney_pg_maske.py`` yalnizca **cat**'i yeni kolonlarla egitti; xgb,
lgbm ve ag 105 kolonluk ``uretim`` onbelleginden geldi. Yani olculen sey
(+0,0102, t=+3,59) "cat 115 kolon, digerleri 105" karisimidir -- uretimde
ise dordu de 115 kolon gorecek.

Bu, bugun UC KEZ tuzaga dusuren durumun ta kendisi: olcum tezgahi uretimden
sapiyor. Fark lehte de olabilir aleyhte de; TAHMIN EDILMEZ, OLCULUR.

MALIYET: xgb ve lgbm ucuz (9'ar fit). SINIR AGI PAHALI (~20 dk x 9 = 3 saat)
ve bu gecenin butcesine sigmaz -- o yuzden ag 105 kolonluk onbellekten
gelmeye devam eder ve KALAN BELIRSIZLIK olarak kayda gecer. Agin harmandaki
payi 1,4/6,4 = %22.

    python scripts/deney_pg10_aileler.py
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
KAYIT = KOK / "experiments" / "pg10_aileler.jsonl"
TOHUMLAR = (1000, 1001, 1002)
GBDT_AGIRLIK = (3.0, 1.0, 1.0)
AG_AGIRLIK = 1.4
MASKE = 0.15
SICAK_KATSAYI = 0.7784 * 0.74263 / 1.07907


def main() -> int:
    satir_tamponlu_cikti()
    t0 = time.time()
    print("=" * 100)
    print("pg10'un DIGER AILELERDEKI etkisi -- xgb ve lgbm de 115 kolon gormeli")
    print("=" * 100)

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    taban_kol = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    ek10 = list(tm.REJIM_AYARLARI["sicak"]["ek_kolon"])  # type: ignore[index]
    kol115 = taban_kol + ek10
    tm.kategorik_kodla(egitim, test)
    guc_kenar = ol.guc_kenarlari(test)
    te_s = test[test["soguk_mu"] != 1]
    print(f"  taban {len(taban_kol)} kolon  ->  pg10 {len(kol115)} kolon")

    ek = d._ek_kokenler_kur(False)
    ek = ek[ek["_blok"].isin([a for a, _, _ in tm.EK_KOKENLER])]
    ortak = [k for k in egitim.columns if k in ek.columns]
    genis = pd.concat([egitim[ortak], ek[ortak]], ignore_index=True)
    for k in tm.KATEGORIK:
        genis[k] = pd.Categorical(genis[k], categories=egitim[k].cat.categories)

    veri = {}
    for b in tm.BLOKLAR:
        _, dogrulama, _, soguk = di.blok_parcalari(egitim, b.ad)
        sicak = ~soguk
        parca = tm.kokenleri_ayikla(genis, b.ad)
        w, _ = ol.test_agirliklari(dogrulama[~soguk], te_s, guc_kenar, eksenler=("bayatlik",))
        blok = {"gercek": np.load(DIZIN / f"{b.ad}_gercek.npy"), "w": w}
        for t in TOHUMLAR:
            for a in ("cat", "xgb", "lgbm", "sinir_agi"):
                blok[(t, a, 105)] = np.load(DIZIN / f"{b.ad}_{t}_{a}_uretim.npy").astype("float64")
            blok[(t, "cat", 115)] = np.load(DIZIN / f"{b.ad}_{t}_cat_pg10.npy").astype("float64")
        print(f"\n  {b.ad}  egitim {len(parca):,}  sicak {int(sicak.sum()):,}")
        for t in TOHUMLAR:
            for aile in ("xgb", "lgbm"):
                yol = DIZIN / f"{b.ad}_{t}_{aile}_pg10.npy"
                if yol.exists():
                    blok[(t, aile, 115)] = np.load(yol).astype("float64")
                    continue
                t1 = time.time()
                maskeli = d.soguk_maskele(parca, kol115, MASKE, t)
                log_t = di.egit_tahmin(aile, maskeli, dogrulama, kol115, t)
                v = log_t[sicak] if log_t.shape[0] == soguk.size else log_t
                np.save(yol, v.astype("float32"))
                blok[(t, aile, 115)] = v.astype("float64")
                print(f"    tohum {t}  {aile:5} 115 kolon ({time.time() - t1:.0f} sn)")
        veri[b.ad] = blok

    def skor(bad: str, tohum: int, cat_k: int, gbdt_k: int) -> float:
        v = veri[bad]
        top = sum(GBDT_AGIRLIK) + AG_AGIRLIK
        pay = (
            GBDT_AGIRLIK[0] * v[(tohum, "cat", cat_k)]
            + GBDT_AGIRLIK[1] * v[(tohum, "xgb", gbdt_k)]
            + GBDT_AGIRLIK[2] * v[(tohum, "lgbm", gbdt_k)]
            + AG_AGIRLIK * v[(tohum, "sinir_agi", 105)]
        )
        return ol.agirlikli_rmsle(v["gercek"], np.clip(np.expm1(pay / top), 0.0, None), v["w"])

    ciftler = [(b.ad, t) for b in tm.BLOKLAR for t in TOHUMLAR]
    kurgular = {
        "URETIM (hepsi 105)": (105, 105),
        "OLCULEN (cat 115)": (115, 105),
        "GERCEK (cat+xgb+lgbm 115)": (115, 115),
    }
    skorlar = {ad: {c: skor(*c, *k) for c in ciftler} for ad, k in kurgular.items()}

    print("\n" + "-" * 100)
    print("UC KURGU (teste agirliklandirilmis; ag her zaman 105 kolon)")
    print("-" * 100)
    print(f"  {'kurgu':>26}{'ortalama':>11}", end="")
    for b in tm.BLOKLAR:
        print(f"{b.ad:>11}", end="")
    print()
    for ad in kurgular:
        s = skorlar[ad]
        blok_ort = [np.mean([s[(b.ad, t)] for t in TOHUMLAR]) for b in tm.BLOKLAR]
        print(
            f"  {ad:>26}{np.mean(list(s.values())):11.5f}" + "".join(f"{x:11.5f}" for x in blok_ort)
        )

    print("\n" + "-" * 100)
    print("ESLENIK FARK (URETIM - aday; POZITIF = aday IYI)")
    print("-" * 100)
    kayitlar = []
    taban = skorlar["URETIM (hepsi 105)"]
    for ad in ("OLCULEN (cat 115)", "GERCEK (cat+xgb+lgbm 115)"):
        s = skorlar[ad]
        f = np.array([taban[c] - s[c] for c in ciftler])
        ort, sh = float(f.mean()), float(f.std(ddof=1) / np.sqrt(len(f)))
        t_d = ort / sh if sh > 0 else 0.0
        blok_ort = {
            b.ad: float(np.mean([taban[(b.ad, t)] - s[(b.ad, t)] for t in TOHUMLAR]))
            for b in tm.BLOKLAR
        }
        uc = all(v > 0 for v in blok_ort.values())
        print(
            f"  {ad:>26}  fark {ort:+.5f}  SH {sh:.5f}  t {t_d:+.2f}  "
            f"{(f > 0).sum()}/{len(f)}  uc blok {'EVET' if uc else 'HAYIR'}"
            f"  genel {-ort * SICAK_KATSAYI:+.5f}"
        )
        for b in tm.BLOKLAR:
            print(f"       {b.ad:8} {blok_ort[b.ad]:+.5f}")
        kayitlar.append({"kurgu": ad, "fark": ort, "sh": sh, "t": t_d, "blok": blok_ort})

    o = skorlar["OLCULEN (cat 115)"]
    g = skorlar["GERCEK (cat+xgb+lgbm 115)"]
    d_f = np.array([o[c] - g[c] for c in ciftler])
    print(
        f"\n  BOSLUGUN KENDISI (olculen -> gercek): {d_f.mean():+.5f}"
        f"  SH {d_f.std(ddof=1) / np.sqrt(len(d_f)):.5f}"
        f"  ({(d_f > 0).sum()}/{len(d_f)} gercek daha iyi)"
    )
    print("  Pozitifse xgb/lgbm de kolonlardan yararlaniyor ve uretim OLCULENDEN IYI olur.")
    print("  Negatifse uretim olculenden KOTU olur ve karar gozden gecirilmeli.")
    print("  KALAN BELIRSIZLIK: sinir agi (harmanin %22'si) hala 105 kolonluk onbellekten.")

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        for k in kayitlar:
            fh.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
