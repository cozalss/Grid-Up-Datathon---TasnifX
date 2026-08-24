"""SOGUK KARARLARI DUZELTILMIS kVA KARISIMINDA YENIDEN PUANLA.

NEDEN
-----
Soguk tarafta sorun EGITIMDE degil OLCUTTE. Olculdu (dagilim sayimi):

    kova (test soguk kuantilleri)        0     1     2     3     4     5     6
    TEST soguk %                      6,77  9,19 13,99 17,46 13,23 22,35 17,01
    kis26 soguk % (dogrulama)         9,78 12,10 19,27 28,68 11,57 13,70  4,91
    EGITIM tumu % (kaynak)           11,61 11,38 16,24 21,43 10,33 16,44 12,57

Egitim dagilimi teste yakin (en buyuk kova 12,57 vs 17,01 = 1,35x), ama
DOGRULAMA bloklarinin soguk satirlari degil: en buyuk kovada 4,91 vs 17,01,
yani **3,5 kat** eksik. Yani kis26 uzerinde alinan her soguk karar, testin
kVA karisiminin ucte birini temsil eden bir kovada olculmus oluyor.

docs/39 §3 bunu zaten teshis etmisti ("kis26 soguk medyan 400, TEST 630;
kova 12 payi %4,4 vs %16,6") ve gecenin son islem revizyonu bu yuzden
LB'de curudu. Ama ILAC UYGULANMADI: uretimdeki soguk kararlar hala HAM
kis26 uzerinde secilmis sayilarla duruyor.

Bu betik iki uretim kararini duzeltilmis karisim altinda yeniden puanlar:

    1) ``son_islem.py`` BETA'si (uretim 0,60; ham kis26 dibi 0,40)
    2) soguk harman (uretim yalniz cat)

HUKUM DEGIL, ADAY URETIR
------------------------
docs/39 §9 kalici kurali acik: soguk son isleme kis26'ya bakarak
DOKUNULMAZ; degisiklik ancak IZOLE bir LB gonderimiyle sinanir. Bu betik o
kurali kaldirmiyor -- kurali doguran olcut hatasini duzeltip aday
uretiyor. Ciktisi "uygula" degil, "LB'de sinanmaya deger".

    python scripts/deney_soguk_kva_olcut.py
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

BLOK = "kis26"
ONBELLEK = KOK / "data" / "interim" / "deney" / f"soguk_tahmin_{BLOK}.npz"
KAYIT = KOK / "experiments" / "soguk_kva_olcut.jsonl"
TOHUMLAR = (1000, 1001, 1002)
AILELER = ("cat", "xgb", "lgbm")
BETALAR = (1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20)
HARMANLAR = ((1.0, 0.0, 0.0), (3.0, 1.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0))

#: ``d(genel)/d(soguk)`` -- genel = sqrt(0,7784*sicak^2 + 0,2216*soguk^2).
SOGUK_KATSAYI = 0.2216 * 1.82133 / 1.07907


def main() -> int:
    satir_tamponlu_cikti()
    t0 = time.time()
    print("=" * 96)
    print(f"SOGUK KARARLARI -- duzeltilmis kVA karisimi ({BLOK})")
    print("=" * 96)

    egitim, test = d.cerceveleri_kur()
    tm.kategorik_kodla(egitim, test)
    _, dogrulama, gercek, soguk = di.blok_parcalari(egitim, BLOK)
    dg = dogrulama[soguk]
    y = gercek[soguk]
    log_guc = np.log1p(dg["guc"].to_numpy(dtype="float64"))

    te_c = test[test["soguk_mu"] == 1]
    guc_kenar = ol.guc_kenarlari(te_c)
    w, tani = ol.test_agirliklari(dg, te_c, guc_kenar, eksenler=("guc",))
    print(f"\n  {BLOK} soguk {len(y):,} satir")
    print(f"  kVA agirliklandirmasi: ESS %{100 * tani['ess_orani']:.1f}  tabaka {tani['tabaka']}")
    print(f"  medyan guc  dogrulama {dg['guc'].median():.0f}  TEST {te_c['guc'].median():.0f}")

    z = np.load(ONBELLEK)

    def harman_log(agr) -> np.ndarray:  # noqa: ANN001
        top = sum(agr)
        return np.mean(
            [
                sum(wi * z[f"{t}_{a}"] for a, wi in zip(AILELER, agr, strict=True)) / top
                for t in TOHUMLAR
            ],
            axis=0,
        )

    def buz(log_t, beta) -> np.ndarray:  # noqa: ANN001
        r = log_t - log_guc
        return np.clip(np.expm1(r.mean() + beta * (r - r.mean()) + log_guc), 0.0, None)

    # --------------------------------------------------------- 1) beta
    print("\n" + "-" * 96)
    print("1) SON ISLEM BETA'SI  (uretim 0,60)")
    print("-" * 96)
    print(f"  {'beta':>6}{'HAM kis26':>12}{'kVA duzeltilmis':>18}{'ham fark':>11}{'duz fark':>11}")
    log_uretim = harman_log((1.0, 0.0, 0.0))
    kayitlar, egri = [], []
    for beta in BETALAR:
        t = buz(log_uretim, beta)
        ham = ol.agirlikli_rmsle(y, t)
        duz = ol.agirlikli_rmsle(y, t, w)
        egri.append((beta, ham, duz))
    ham60 = next(h for b, h, _ in egri if abs(b - 0.60) < 1e-9)
    duz60 = next(dd for b, _, dd in egri if abs(b - 0.60) < 1e-9)
    for beta, ham, duz in egri:
        yildiz = "  <- URETIM" if abs(beta - 0.60) < 1e-9 else ""
        print(
            f"  {beta:6.2f}{ham:12.5f}{duz:18.5f}{ham - ham60:+11.5f}{duz - duz60:+11.5f}{yildiz}"
        )
        kayitlar.append({"tur": "beta", "beta": beta, "ham": ham, "duzeltilmis": duz})

    en_ham = min(egri, key=lambda e: e[1])
    en_duz = min(egri, key=lambda e: e[2])
    print(f"\n  HAM kis26 dibi        beta {en_ham[0]:.2f}  {en_ham[1]:.5f}")
    print(f"  kVA DUZELTILMIS dibi  beta {en_duz[0]:.2f}  {en_duz[2]:.5f}")
    kazanc = duz60 - en_duz[2]
    print(f"  uretim 0,60'a gore duzeltilmis kazanc {kazanc:+.5f}")
    print(f"  genel skora tahmini etki {-kazanc * SOGUK_KATSAYI:+.5f}")
    if abs(en_duz[0] - 0.60) < 1e-9:
        print("  -> Optimum URETIMDE. Degisiklik onerilmiyor.")

    # ------------------------------------------------------- 2) harman
    print("\n" + "-" * 96)
    print("2) SOGUK HARMAN  (uretim yalniz cat), her biri kendi en iyi beta'siyla")
    print("-" * 96)
    print(
        f"  {'harman':>12}{'beta060 ham':>14}{'beta060 duz':>14}"
        f"{'en iyi beta':>13}{'o beta duz':>12}"
    )
    for agr in HARMANLAR:
        lg = harman_log(agr)
        ham = ol.agirlikli_rmsle(y, buz(lg, 0.60))
        duz = ol.agirlikli_rmsle(y, buz(lg, 0.60), w)
        en = min(((b, ol.agirlikli_rmsle(y, buz(lg, b), w)) for b in BETALAR), key=lambda e: e[1])
        etik = "/".join(str(int(x)) for x in agr)
        bayrak = "  <- URETIM" if agr == (1.0, 0.0, 0.0) else ""
        print(f"  {etik:>12}{ham:15.5f}{duz:15.5f}{en[0]:13.2f}{en[1]:12.5f}{bayrak}")
        kayitlar.append(
            {
                "tur": "harman",
                "harman": list(agr),
                "ham60": ham,
                "duz60": duz,
                "en_beta": en[0],
                "en_duz": en[1],
            }
        )

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        for k in kayitlar:
            fh.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
