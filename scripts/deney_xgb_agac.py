"""xgb AGAC SAYISI -- 400 hic xgb icin olculmedi, LightGBM egrisinden kopyalandi.

BULGU (olcum kaydi denetimi, 2026-08-24 aksami)
-----------------------------------------------
``SABIT_AGAC = 400`` (``tuketim_model.py:803``) xgb'nin ``n_estimators``i
olarak kullaniliyor (:1125). Ama 400 sayisi bir **LightGBM** egrisinden
geliyor (``deney.py:22-25``: "egri 200-1000 agac arasinda duz") ve o olcum
su rig'de yapildi: ofsetli hedef YOK, soguk maskeleme YOK, rejim uzmani YOK,
ek koken YOK (1,04M satir). xgb icin HIC taranmadi.

Kodun kendisi kopyalamayi itiraf ediyor -- ``deney.py:432-440``: xgb ayarlari
"bilerek BENZER" tutulmus, amac "hangi ailenin daha iyi AYARLANABILDIGI degil
aralarindaki CESITLILIK".

NEDEN TAM XGB'DE ONEMLI
-----------------------
Ek koken aileleri ESIT OLMAYAN olcude gucluyor
(``experiments/aile_koken.jsonl``): cat +0,0061, lgbm +0,0090, **xgb +0,0329**.
Ve xgb'nin SIRALAMASI doniyor: 1,04M satirda ucun EN KOTUSU (0,85126),
2,86M satirda EN IYISI (0,78953). Yani kapasitesi, en kotu oldugu ve hic
ayarlanmadigi rig'de donmus durumda.

Ayrica "duz egri" ozelliginin AILEYE OZGU oldugunun kaniti ayni repoda:
CatBoost taranınca egri duz DEGIL, tekduze kotulesiyordu
(``deney.py:806-812``, d6: 200 -> 1,11283, 400 -> 1,12004, 2000 -> 1,18758).
Bir ailenin duz egrisi digerine tasinamaz.

UCUZ TARAMA
-----------
Tek fit ``n_estimators=1200``, sonra ``predict(iteration_range=(0, n))`` ile
400/600/800/1000/1200 egrisi BEDAVA okunur. Sabit tohumda 1200 agacli bir
modelin ilk n agaci, n agacli modelle BIREBIR aynidir (gradyan artirma
toplamsaldir). ``deney.py:850`` CatBoost'ta ``ntree_end`` ile ayni numarayi
kullaniyor.

Beklenti olculu: xgb sicak harmanin 1/6,4 = %15,6'si ve soguk harmanin
SIFIRI. Cat'in kendi kapasite yayilimi ~0,006 idi ve genele 0,0005 ediyordu;
xgb icin tavan benzer buyuklukte.

    python scripts/deney_xgb_agac.py            # 1 tohum on-eleme
    python scripts/deney_xgb_agac.py --tohum 3  # eslenik hukum
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
KAYIT = KOK / "experiments" / "xgb_agac.jsonl"
GBDT_AGIRLIK = (3.0, 1.0, 1.0)
AG_AGIRLIK = 1.4
MASKE = 0.15
TAVAN_AGAC = 1200
AGACLAR = (400, 600, 800, 1000, 1200)
URETIM_AGAC = 400
SICAK_KATSAYI = 0.7784 * 0.74263 / 1.07907


def main() -> int:
    satir_tamponlu_cikti()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tohum", type=int, default=1)
    ar = ap.parse_args()
    tohumlar = tuple(1000 + i for i in range(ar.tohum))

    t0 = time.time()
    print("=" * 100)
    print("xgb AGAC SAYISI -- tek fit 1200, egri iteration_range ile bedava")
    print("=" * 100)

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    taban_kol = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    ek10 = list(tm.REJIM_AYARLARI["sicak"]["ek_kolon"])  # type: ignore[index]
    kol = taban_kol + ek10
    tm.kategorik_kodla(egitim, test)
    guc_kenar = ol.guc_kenarlari(test)
    te_s = test[test["soguk_mu"] != 1]
    print(f"  URETIM rig: {len(kol)} kolon (pg10), ek kokenli, maske {MASKE}")

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
        for t in tohumlar:
            for a in ("lgbm", "sinir_agi"):
                blok[(t, a)] = np.load(DIZIN / f"{b.ad}_{t}_{a}_uretim.npy").astype("float64")
            p = DIZIN / f"{b.ad}_{t}_cat_pg10.npy"
            blok[(t, "cat")] = np.load(p).astype("float64")
        print(f"\n  {b.ad}  egitim {len(parca):,}  sicak {int(sicak.sum()):,}")
        for t in tohumlar:
            eksik = [n for n in AGACLAR if not (DIZIN / f"{b.ad}_{t}_xgb_a{n}.npy").exists()]
            if eksik:
                import xgboost as xgb

                t1 = time.time()
                maskeli = d.soguk_maskele(parca, kol, MASKE, t)
                y = np.log1p(maskeli[tm.HEDEF].clip(lower=0.0)) - np.log1p(maskeli["guc"])
                model = xgb.XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=TAVAN_AGAC,
                    learning_rate=0.05,
                    max_depth=8,
                    min_child_weight=20,
                    subsample=0.85,
                    colsample_bytree=0.75,
                    reg_lambda=2.0,
                    random_state=t,
                    n_jobs=-1,
                    tree_method="hist",
                    enable_categorical=True,
                    verbosity=0,
                )
                model.fit(maskeli[kol], y)
                geri = np.log1p(dogrulama["guc"]).to_numpy()
                for n in AGACLAR:
                    ham = model.predict(dogrulama[kol], iteration_range=(0, n)) + geri
                    v = ham[sicak] if ham.shape[0] == soguk.size else ham
                    np.save(DIZIN / f"{b.ad}_{t}_xgb_a{n}.npy", v.astype("float32"))
                print(f"    tohum {t}  tek fit {TAVAN_AGAC} agac ({time.time() - t1:.0f} sn)")
            for n in AGACLAR:
                blok[(t, "xgb", n)] = np.load(DIZIN / f"{b.ad}_{t}_xgb_a{n}.npy").astype("float64")
        veri[b.ad] = blok

    def skor(bad: str, tohum: int, n: int) -> float:
        v = veri[bad]
        top = sum(GBDT_AGIRLIK) + AG_AGIRLIK
        pay = (
            GBDT_AGIRLIK[0] * v[(tohum, "cat")]
            + GBDT_AGIRLIK[1] * v[(tohum, "xgb", n)]
            + GBDT_AGIRLIK[2] * v[(tohum, "lgbm")]
            + AG_AGIRLIK * v[(tohum, "sinir_agi")]
        )
        return ol.agirlikli_rmsle(v["gercek"], np.clip(np.expm1(pay / top), 0.0, None), v["w"])

    ciftler = [(b.ad, t) for b in tm.BLOKLAR for t in tohumlar]
    print("\n" + "-" * 100)
    print("AGAC EGRISI (uretim harmani icinde, teste agirliklandirilmis)")
    print("-" * 100)
    print(f"  {'agac':>6}{'ortalama':>11}", end="")
    for b in tm.BLOKLAR:
        print(f"{b.ad:>11}", end="")
    print(f"{'400e gore':>12}")
    skorlar = {n: {c: skor(*c, n) for c in ciftler} for n in AGACLAR}
    taban = skorlar[URETIM_AGAC]
    kayitlar = []
    for n in AGACLAR:
        s = skorlar[n]
        blok_ort = [np.mean([s[(b.ad, t)] for t in tohumlar]) for b in tm.BLOKLAR]
        ort = float(np.mean(list(s.values())))
        f = np.array([taban[c] - s[c] for c in ciftler])
        bayrak = "  <- URETIM" if n == URETIM_AGAC else ""
        print(
            f"  {n:6d}{ort:11.5f}"
            + "".join(f"{x:11.5f}" for x in blok_ort)
            + f"{f.mean():+12.5f}{bayrak}"
        )
        kayitlar.append({"agac": n, "ortalama": ort, "blok": [float(x) for x in blok_ort]})

    print("\n  ESLENIK FARK (400 - aday; POZITIF = aday IYI)")
    for n in AGACLAR:
        if n == URETIM_AGAC:
            continue
        f = np.array([taban[c] - skorlar[n][c] for c in ciftler])
        sh = float(f.std(ddof=1) / np.sqrt(len(f))) if len(f) > 1 else 0.0
        t_d = f.mean() / sh if sh > 0 else 0.0
        blok_ort = {
            b.ad: float(np.mean([taban[(b.ad, t)] - skorlar[n][(b.ad, t)] for t in tohumlar]))
            for b in tm.BLOKLAR
        }
        uc = all(v > 0 for v in blok_ort.values())
        hukum = "AL" if (t_d >= 2 and uc) else ("REDDET" if t_d <= -2 else "esik alti")
        print(
            f"    {n:5d}  fark {f.mean():+.5f}  SH {sh:.5f}  t {t_d:+.2f}"
            f"  {(f > 0).sum()}/{len(f)}  uc blok {'EVET' if uc else 'HAYIR'}"
            f"  genel {-f.mean() * SICAK_KATSAYI:+.5f}  {hukum}"
        )

    if len(tohumlar) == 1:
        print("\n  NOT: tek tohum ON-ELEME. SH tohum yayilmasini ICERMEZ; egri duzse")
        print("       orada birakilir, iniyorsa --tohum 3 ile eslenik hukum alinir.")

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        for k in kayitlar:
            fh.write(json.dumps({"tohum_sayisi": len(tohumlar), **k}, ensure_ascii=False) + "\n")
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
