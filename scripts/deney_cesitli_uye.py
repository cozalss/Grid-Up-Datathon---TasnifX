"""GERCEKTEN FARKLI 5. UYE: maske=1,00 uzmani SICAK satirlara uygulanmis.

BUGUNUN DERSI
-------------
``deney_ag_karsilastir.py`` §3 olctu: sinir agi tek basina GBDT harmanindan
COK daha kotu (kis26 0,873 vs 0,739) ama AYRISMAYI IKIYE KATLIYOR
(0,02856 -> 0,06363) ve harmani 0,90065 -> 0,89718 cekiyor. Yani bu
toplulukta CESITLILIK doğruluktan degerli.

``deney_ofsetsiz_uye.py`` ise bunun tersini gosterdi: ``ofset=False`` uyesi
ayrismayi BUYUTMEDI, kucultttu (0,06363 -> 0,06170) ve blok tutarliligi 1/3
cikti. Sebebi mekanik: kapasite ofseti ``log1p(guc)`` satir basina sabit ve
``guc`` modelin zaten gordugu bir kolon, yani iki hedef agac icin denk.

Cikan kural: **bir "cesitlilik uyesi" adayi ancak AYRISMAYI BUYUTUYORSA
umut vaat eder**, ve bu skordan ONCE, ucuz bir onbellekle olculebilir.

BU ADAY
-------
Maske orani 1,00 -- yani model hicbir trafonun gecmis ozetini gormez, butun
``t_*`` kolonlari NaN. Uretimde bu SOGUK uzmanidir; burada ayni modeli SICAK
satirlara uyguluyoruz. Bilgi tabani uretim uyelerininkinden YAPISAL olarak
farkli: onlar gecmise dayaniyor, bu dayanmiyor. Tek basina cok daha kotu
olacak -- soru o degil, ayrismayi buyutup buyutmedigi.

Beklenti duruslugu icin: bu aday BASARISIZ olursa da bilgi degerlidir,
cunku "cesitlilik uyesi" ailesinin bu veri setinde tukendigini gosterir.

    python scripts/deney_cesitli_uye.py
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
KAYIT = KOK / "experiments" / "cesitli_uye.jsonl"
TOHUMLAR = (1000, 1001, 1002)
GBDT = ("cat", "xgb", "lgbm")
GBDT_AGIRLIK = (3.0, 1.0, 1.0)
AG_AGIRLIK = 1.4
#: Uretimdeki SOGUK uzmani: maske 1,00, cat depth 7 (REJIM_AYARLARI).
KOR_MASKE = 1.00
KOR_CAT: dict[str, object] = {"depth": 7}
YENI_AGIRLIKLAR = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.5)
SICAK_KATSAYI = 0.7784 * 0.74263 / 1.07907


def main() -> int:
    satir_tamponlu_cikti()
    t0 = time.time()
    print("=" * 100)
    print("GERCEKTEN FARKLI 5. UYE: maske=1,00 (gecmissiz) uzman, sicak satirlarda")
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
            yol = DIZIN / f"{b.ad}_{t}_cat_kor.npy"
            if yol.exists():
                blok[(t, "kor")] = np.load(yol).astype("float64")
                continue
            t1 = time.time()
            maskeli = d.soguk_maskele(parca, kol, KOR_MASKE, t)
            log_t = di.egit_tahmin("cat", maskeli, dogrulama, kol, t, **KOR_CAT)
            v = log_t[sicak] if log_t.shape[0] == soguk.size else log_t
            np.save(yol, v.astype("float32"))
            blok[(t, "kor")] = v.astype("float64")
            print(f"    tohum {t}  cat/kor (maske 1,00)  ({time.time() - t1:.0f} sn)")
        veri[b.ad] = blok

    # ------------------------------------- ONCE AYRISMA: aday cesitli mi?
    print("\n" + "-" * 100)
    print("1) AYRISMA -- aday cesitliligi BUYUTUYOR mu? (skordan ONCE sorulur)")
    print("-" * 100)
    print(f"  {'blok':>8}{'kor tek':>10}{'AYR(uretim)':>14}{'AYR(+kor)':>12}{'buyume':>10}")
    for b in tm.BLOKLAR:
        v = veri[b.ad]
        glog = np.log1p(np.clip(v["gercek"], 0.0, None))
        uye = [np.mean([v[(t, a)] for t in TOHUMLAR], axis=0) for a in (*GBDT, "sinir_agi")]
        kor = np.mean([v[(t, "kor")] for t in TOHUMLAR], axis=0)
        ayr = {}
        for etiket, uyeler, agr in (
            ("u", uye, [*GBDT_AGIRLIK, AG_AGIRLIK]),
            ("y", [*uye, kor], [*GBDT_AGIRLIK, AG_AGIRLIK, 1.0]),
        ):
            a = np.array(agr, dtype="float64")
            a = a / a.sum()
            har = sum(wi * m for wi, m in zip(a, uyeler, strict=True))
            ayr[etiket] = float(
                sum(wi * np.mean((m - har) ** 2) for wi, m in zip(a, uyeler, strict=True))
            )
        tek = float(np.sqrt(np.mean((glog - kor) ** 2)))
        print(
            f"  {b.ad:>8}{tek:10.5f}{ayr['u']:14.5f}{ayr['y']:12.5f}"
            f"{100 * (ayr['y'] / ayr['u'] - 1):9.0f}%"
        )

    # --------------------------------------------------- 2) agirlik taramasi
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
                    pay = pay + yeni_w * v[(t, "kor")]
                    top = top + yeni_w
                yig.append(pay / top)
            tahmin = np.clip(np.expm1(np.mean(yig, axis=0)), 0.0, None)
            y, w = v["gercek"], v["w"]
            k_ag += ol.agirlikli_rmsle(y, tahmin, w) ** 2 * w.sum()
            w_top += w.sum()
        return float(np.sqrt(k_ag / w_top))

    print("\n" + "-" * 100)
    print("2) AGIRLIK TARAMASI (uretim harmani sabit)")
    print("-" * 100)
    print(f"  {'kor w':>7}{'agirlikli':>12}", end="")
    for b in tm.BLOKLAR:
        print(f"{b.ad:>11}", end="")
    print()
    kayitlar = []
    for w in YENI_AGIRLIKLAR:
        g = skorla(w)
        blok_skor = [skorla(w, b.ad) for b in tm.BLOKLAR]
        print(f"  {w:7.2f}{g:12.5f}" + "".join(f"{x:11.5f}" for x in blok_skor), end="")
        print("  <- URETIM (kor uye YOK)" if w == 0.0 else "")
        kayitlar.append({"w": w, "agirlikli": g, "blok": blok_skor})

    taban, en = kayitlar[0], min(kayitlar, key=lambda k: k["agirlikli"])
    fark = taban["agirlikli"] - en["agirlikli"]
    kazanan = sum(1 for i in range(len(tm.BLOKLAR)) if en["blok"][i] < taban["blok"][i])
    print(f"\n  en iyi w={en['w']:.2f}  {en['agirlikli']:.5f}   uretim {taban['agirlikli']:.5f}")
    print(f"  fark {fark:+.5f}   genel skora tahmini etki {-fark * SICAK_KATSAYI:+.5f}")
    print(f"  BLOK TUTARLILIGI: {kazanan}/{len(tm.BLOKLAR)}")
    if kazanan < len(tm.BLOKLAR):
        print("  UYARI: uc blokta birden kazanmiyor -- hipotez SAYILMAZ.")

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        for k in kayitlar:
            fh.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
