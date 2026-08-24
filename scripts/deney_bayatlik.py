"""BAYATLIK EKSENI: model bayat satirlarda YANLI mi, yoksa yalnizca gurultulu mu?

NEDEN ONEMLI
------------
``deney_olcut_kayma.py`` sunu olctu:

    kova    TEST%   CV%          guz25 RMSLE
    0 gun   84,47   98,3         0,79052
    1-6      7,76    0,14-0,50   2,21857   <- SOGUKTAN (1,82) BETER
    90+      6,63    0,00-0,41   0,95-1,58

Testin sicak tarafinin %14,4'u, CV'de %0,6 olan iki kovada. Ve o kovalarda
hata iki-uc kat. Sicak skorun teste agirliklandirilmis hali 0,75865 -> 0,85865.

Ama BUYUK HATA tek basina duzeltilebilir demek DEGILDIR. Iki ayri dunya:

    YANLILIK  model sistematik olarak yanlis tarafa tahmin ediyor
              -> kaydirma/buzme ile duzeltilebilir
    VARYANS   tahmin dogru merkezde ama satirlar oz olarak ongorulemez
              -> duzeltilemez, yalnizca kabul edilir

Bu betik ikisini AYIRIR ve duzeltmenin TASIYIP tasimadigini sinar.

PROTOKOL: BLOK-DISI (bu gecenin dersi)
--------------------------------------
Gecen gece tek blokta (kis26) uydurulmus bir son islem revizyonu LB'de
+0,00414 ile curudu (docs/39 §3). Bu yuzden burada duzeltme ASLA olculdugu
blokta uydurulmaz:

    duzeltme yaz25+guz25'te uydurulur  ->  kis26'da OLCULUR
    duzeltme yaz25+kis26'da uydurulur  ->  guz25'te OLCULUR
    duzeltme guz25+kis26'da uydurulur  ->  yaz25'te OLCULUR

Uc kattan ucunde de kazandiriyorsa transfer kaniti var; biri bile ters
donuyorsa yoktur. SICAK satirlar icin uc blok da mesru -- ezber kanali
(docs/35) yalnizca SOGUK satirlari kirletir, cunku sicak trafolar testte de
egitimde de mevcuttur, kanal simetriktir.

    python scripts/deney_bayatlik.py
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
KAYIT = KOK / "experiments" / "bayatlik.jsonl"
TOHUMLAR = (1000, 1001, 1002)
AILELER = ("cat", "xgb", "lgbm")
URETIM_HARMAN = (3.0, 1.0, 1.0)
ETIKET = ("0 gun", "1-6", "7-29", "30-89", "90+")

#: ``d(genel)/d(sicak)`` -- sicak taraftaki kazancin genel skora cevrimi.
#: genel = sqrt(0,7784*sicak^2 + 0,2216*soguk^2); turev = 0,7784*sicak/genel.
SICAK_KATSAYI = 0.7784 * 0.74263 / 1.07907


def _blend(z, blok: str) -> np.ndarray:  # noqa: ANN001
    toplam = sum(URETIM_HARMAN)
    yig = [
        sum(w * z[f"{blok}_{t}_{a}"] for a, w in zip(AILELER, URETIM_HARMAN, strict=True)) / toplam
        for t in TOHUMLAR
    ]
    return np.mean(yig, axis=0)


def _blok_verisi(egitim, test, z, guc_kenar):  # noqa: ANN001, ANN202
    """Blok basina: gercek log, tahmin log, bayatlik kovasi, test agirligi."""
    veri = {}
    te_s = test[test["soguk_mu"] != 1]
    for b in tm.BLOKLAR:
        _, dogrulama, gercek, soguk = di.blok_parcalari(egitim, b.ad)
        sicak = ~soguk
        dsicak = dogrulama[~soguk]
        p = _blend(z, b.ad)
        p_sicak = p[sicak] if p.shape[0] == soguk.size else p
        yas = dsicak["t_son_kayit_yasi"].to_numpy(dtype="float64")
        w, _ = ol.test_agirliklari(dsicak, te_s, guc_kenar, eksenler=("bayatlik",))
        veri[b.ad] = {
            "g": np.log1p(np.clip(gercek[sicak], 0.0, None)),
            "p": p_sicak,
            "kova": ol._kova(yas, ol.BAYATLIK_KENARLARI),
            "w": w,
        }
    return veri


def main() -> int:
    satir_tamponlu_cikti()
    t0 = time.time()
    print("=" * 100)
    print("BAYATLIK EKSENI -- yanlilik mi varyans mi, ve duzeltme TASIR mi?")
    print("=" * 100)

    egitim, test = d.cerceveleri_kur()
    tm.kategorik_kodla(egitim, test)
    guc_kenar = ol.guc_kenarlari(test)
    z = np.load(ONBELLEK)
    veri = _blok_verisi(egitim, test, z, guc_kenar)
    adlar = [b.ad for b in tm.BLOKLAR]

    # ------------------------------------------------ 1) yanlilik ayrismasi
    print("\n" + "-" * 100)
    print("1) KOVA BASINA YANLILIK  (artik = gercek_log - tahmin_log;  + = model AZ tahmin ediyor)")
    print("-" * 100)
    print(f"  {'kova':>8}", end="")
    for ad in adlar:
        print(f"{ad + ' yanli':>15}{'n':>8}", end="")
    print(f"{'YANLI^2 payi':>14}")

    kayitlar = []
    for i, ad in enumerate(ETIKET):
        print(f"  {ad:>8}", end="")
        yanli_kare, toplam_kare, n_top = 0.0, 0.0, 0
        for bad in adlar:
            v = veri[bad]
            m = v["kova"] == i
            if m.sum() < 30:
                print(f"{'--':>15}{int(m.sum()):8,}", end="")
                continue
            art = v["g"][m] - v["p"][m]
            sh = art.std(ddof=1) / np.sqrt(m.sum())
            print(
                f"{art.mean():+11.4f}{'(t' + f'{art.mean() / sh:+.0f}' + ')':>4}{int(m.sum()):8,}",
                end="",
            )
            yanli_kare += art.mean() ** 2 * m.sum()
            toplam_kare += np.dot(art, art)
            n_top += int(m.sum())
            kayitlar.append(
                {
                    "tur": "yanlilik",
                    "blok": bad,
                    "kova": ad,
                    "n": int(m.sum()),
                    "yanli": float(art.mean()),
                    "sh": float(sh),
                }
            )
        pay = 100 * yanli_kare / toplam_kare if toplam_kare > 0 else 0.0
        print(f"{pay:13.1f}%")

    print("\n  YANLI^2 payi = hatanin yuzde kaci SISTEMATIK kaymadan geliyor.")
    print("  Yuksekse duzeltilebilir; dusukse hata oz varyanstir ve kaydirma kar etmez.")

    # -------------------------------------------- 2) blok-disi duzeltme
    print("\n" + "-" * 100)
    print("2) BLOK-DISI KAYDIRMA  (duzeltme DIGER iki blokta uydurulur, burada olculur)")
    print("-" * 100)
    print(
        f"  {'olculen blok':>14}{'duz once':>10}{'duz sonra':>11}"
        f"{'agir once':>11}{'agir sonra':>12}{'agir fark':>11}"
    )

    ozet = []
    for hedef in adlar:
        kaynak = [a for a in adlar if a != hedef]
        # kova basina kaydirma, KAYNAK bloklardan (hedef hic gorulmez)
        kaydirma = np.zeros(len(ETIKET))
        for i in range(len(ETIKET)):
            art_yig = [
                veri[k]["g"][veri[k]["kova"] == i] - veri[k]["p"][veri[k]["kova"] == i]
                for k in kaynak
            ]
            art_yig = [a for a in art_yig if len(a) >= 30]
            if art_yig:
                birlesik = np.concatenate(art_yig)
                kaydirma[i] = float(birlesik.mean())
        v = veri[hedef]
        p_yeni = v["p"] + kaydirma[v["kova"]]
        y = np.expm1(v["g"])
        once = np.clip(np.expm1(v["p"]), 0.0, None)
        sonra = np.clip(np.expm1(p_yeni), 0.0, None)
        d_once, d_sonra = ol.agirlikli_rmsle(y, once), ol.agirlikli_rmsle(y, sonra)
        a_once = ol.agirlikli_rmsle(y, once, v["w"])
        a_sonra = ol.agirlikli_rmsle(y, sonra, v["w"])
        print(
            f"  {hedef:>14}{d_once:10.5f}{d_sonra:11.5f}{a_once:11.5f}{a_sonra:12.5f}"
            f"{a_once - a_sonra:+11.5f}"
        )
        ozet.append(
            {
                "blok": hedef,
                "duz_fark": d_once - d_sonra,
                "agir_fark": a_once - a_sonra,
                "kaydirma": kaydirma.tolist(),
            }
        )

    kazanan = sum(1 for o in ozet if o["agir_fark"] > 0)
    ort = float(np.mean([o["agir_fark"] for o in ozet]))
    print(f"\n  {kazanan}/3 blokta kazandirdi.  ortalama agirlikli kazanc {ort:+.5f}")
    print(
        f"  genel skora tahmini etki {-ort * SICAK_KATSAYI:+.5f}"
        f"  (d(genel)/d(sicak)={SICAK_KATSAYI:.3f})"
    )
    if kazanan < 3:
        print("  UYARI: uc blokta birden kazanmiyor -- TRANSFER KANITI YOK, uygulanmaz.")

    print("\n  Uydurulan kaydirmalar (blok-disi, kova basina):")
    print(f"  {'olculen':>14}", end="")
    for ad in ETIKET:
        print(f"{ad:>10}", end="")
    print()
    for o in ozet:
        print(f"  {o['blok']:>14}", end="")
        for k in o["kaydirma"]:
            print(f"{k:+10.4f}", end="")
        print()

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        for k in kayitlar:
            fh.write(json.dumps(k, ensure_ascii=False) + "\n")
        for o in ozet:
            fh.write(json.dumps({"tur": "blok_disi", **o}, ensure_ascii=False) + "\n")
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
