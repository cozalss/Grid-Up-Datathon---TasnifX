"""UCUNCU HAK: iki YAPILANDIRMANIN log uzayinda karisimi.

FIKIR
-----
Elde iki uretim dosyasi olacak:

    v50   105 kolon (eski yapilandirma), 30 tohum
    v51   115 kolon (pg10),              ~13 tohum

Ucuncu gonderim hakki icin bunlari log uzayinda karistirmak BEDAVA bir aday:
yeniden egitim yok, yalnizca aritmetik. Ve bugun olculen Krogh-Vedelsby
ayrismasi bu toplulukta CESITLILIGIN doğruluktan degerli oldugunu gosterdi --
farkli kolon setleriyle egitilmis iki model kismen dekoreledir.

    karisim = expm1( (1-a) * log1p(v50) + a * log1p(v51) )

AMA AGIRLIK UYDURULMAZ. Her iki yapilandirmanin CV tahminleri onbellekte
(``cat_kap250`` ve ``cat_pg10``), yani a dogrudan olculur.

IKI TUZAK VE NASIL ELE ALINDIGI
-------------------------------
1. TOHUM SAYISI FARKI. CV'de iki kol da 3 tohum; uretimde v50=30, v51=13.
   Tohum ortalamasi YANLILIGI degistirmez, yalnizca gurultuyu ~1/k duser.
   Yani uretimdeki iki tahmin CV'dekinden DAHA TEMIZDIR ve optimum a
   yanliliklarin oranina gore belirlenir. Bu betik a egrisini hem k=3
   olcumunde hem de gurultu terimi CIKARILMIS halde raporlar.

2. BLOK TUTARLILIGI. Havuzlanmis optimum bugun uc kez kandirdi. a her blokta
   ayri raporlanir; bloklar arasi kararsizsa karisim gonderilmez.

    python scripts/deney_karisim.py
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
KAYIT = KOK / "experiments" / "karisim.jsonl"
TOHUMLAR = (1000, 1001, 1002)
GBDT_AGIRLIK = (3.0, 1.0, 1.0)
AG_AGIRLIK = 1.4
ALFALAR = (0.0, 0.15, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0)
SICAK_KATSAYI = 0.7784 * 0.74263 / 1.07907


def main() -> int:
    satir_tamponlu_cikti()
    t0 = time.time()
    print("=" * 100)
    print("UCUNCU HAK: v50 (105 kolon) x v51 (115 kolon) log uzayinda karisim")
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
        top = sum(GBDT_AGIRLIK) + AG_AGIRLIK
        for etiket, cat_sonek, gbdt_sonek in (
            ("A", "cat_kap250", "uretim"),
            ("B", "cat_pg10", "pg10"),
        ):
            yig = []
            for t in TOHUMLAR:
                cat = np.load(DIZIN / f"{b.ad}_{t}_{cat_sonek}.npy").astype("float64")
                # xgb/lgbm 115 kolonlu surumu varsa onu kullan (uretim yolu),
                # yoksa 105 kolonluk uretim onbellegine dus ve bunu BILDIR.
                aile = {}
                for a in ("xgb", "lgbm"):
                    p115 = DIZIN / f"{b.ad}_{t}_{a}_pg10.npy"
                    p105 = DIZIN / f"{b.ad}_{t}_{a}_uretim.npy"
                    yol = p115 if (gbdt_sonek == "pg10" and p115.exists()) else p105
                    aile[a] = np.load(yol).astype("float64")
                ag = np.load(DIZIN / f"{b.ad}_{t}_sinir_agi_uretim.npy").astype("float64")
                yig.append(
                    (
                        GBDT_AGIRLIK[0] * cat
                        + GBDT_AGIRLIK[1] * aile["xgb"]
                        + GBDT_AGIRLIK[2] * aile["lgbm"]
                        + AG_AGIRLIK * ag
                    )
                    / top
                )
            blok[etiket] = np.mean(yig, axis=0)
            blok[etiket + "_tohumlar"] = yig
        veri[b.ad] = blok

    b115 = (DIZIN / f"{tm.BLOKLAR[0].ad}_1000_xgb_pg10.npy").exists()
    print(f"  B kolunda xgb/lgbm 115 kolonlu mu: {'EVET' if b115 else 'HAYIR (105 kolonluk)'}")

    def skorla(a: float, blok: str | None = None) -> float:
        bloklar = [b.ad for b in tm.BLOKLAR] if blok is None else [blok]
        kare, agir = 0.0, 0.0
        for bad in bloklar:
            v = veri[bad]
            log_k = (1.0 - a) * v["A"] + a * v["B"]
            tahmin = np.clip(np.expm1(log_k), 0.0, None)
            kare += ol.agirlikli_rmsle(v["gercek"], tahmin, v["w"]) ** 2 * v["w"].sum()
            agir += v["w"].sum()
        return float(np.sqrt(kare / agir))

    print("\n" + "-" * 100)
    print("ALFA EGRISI (a=0 -> saf v50 yapilandirmasi, a=1 -> saf v51)")
    print("-" * 100)
    print(f"  {'a':>6}{'genel':>11}", end="")
    for b in tm.BLOKLAR:
        print(f"{b.ad:>11}", end="")
    print(f"{'a=1e gore':>12}")
    kayitlar = []
    egri = [(a, skorla(a), [skorla(a, b.ad) for b in tm.BLOKLAR]) for a in ALFALAR]
    saf_b = next(s for a, s, _ in egri if a == 1.0)
    for a, s, bs in egri:
        print(f"  {a:6.2f}{s:11.5f}" + "".join(f"{x:11.5f}" for x in bs) + f"{saf_b - s:+12.5f}")
        kayitlar.append({"alfa": a, "genel": s, "blok": bs})

    en = min(egri, key=lambda e: e[1])
    print(f"\n  HAVUZLANMIS optimum a={en[0]:.2f}  {en[1]:.5f}")
    print(f"  saf v51 (a=1)              {saf_b:.5f}   kazanc {saf_b - en[1]:+.5f}")
    print(f"  genel skora tahmini etki   {-(saf_b - en[1]) * SICAK_KATSAYI:+.5f}")

    print("\n  BLOK BASINA optimum a:")
    kararli = True
    for i, b in enumerate(tm.BLOKLAR):
        yerel = min(egri, key=lambda e: e[2][i])
        print(f"    {b.ad:8} a={yerel[0]:.2f}  {yerel[2][i]:.5f}")
        if abs(yerel[0] - en[0]) > 0.3:
            kararli = False
    print(f"\n  BLOKLAR ARASI KARARLI MI: {'EVET' if kararli else 'HAYIR'}")
    if not kararli:
        print("  -> optimum bloklar arasi savruluyor; karisim gonderilmez, saf v51 tercih edilir.")
    elif saf_b - en[1] < 0.0005:
        print("  -> kazanc esik alti; ucuncu hak baska bir soruya harcanmali.")
    else:
        print("  -> karisim ucuncu hak icin ADAY. Uretimde:")
        print(f"     birlestir_tohum.py ile log uzayinda a={en[0]:.2f} agirlikli birlestir.")

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        for k in kayitlar:
            fh.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
