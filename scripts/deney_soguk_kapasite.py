"""SOGUK UZMAN KAPASITESI -- URETIM GENISLIGINDE (105 kolon) hic ayarlanmadi.

BULGU
-----
Soguk uzman ``depth=7`` (``REJIM_AYARLARI['soguk']``). Bu deger
``deney_soguk_uzman.py`` ve ``deney_ayar2.py`` ile secildi, ama her iki betik
de kolonlari soyle kuruyor::

    kolonlar = [k for k in tm.oznitelikler(egitim) if k in test.columns]

-- yani ``YALIN_CIKARILAN`` filtresi YOK. Uretim ise 105 kolonla calisiyor
(``tuketim_model.py:1428``). Yani derinlik **151 kolonluk** bir problemde
tarandi, **105 kolonluk** bir probleme uygulaniyor.

Derinlik, oznitelik SAYISINA dogrudan duyarlidir: CatBoost oblivious agac
kullanir, her derinlik seviyesinde TEK bolme secilir; kolon sayisi dustukce
ayni derinligin tasidigi etkilesim kapasitesi degisir. 46 kolon cikarmak
(%30) bu ekseni kaydirir.

Soguk taraf hatanin ~%47'sini tasiyor ve bugun ona yalnizca HARMAN ve BETA
acisindan bakildi -- modelin kapasitesine hic bakilmadi.

NEDEN YALNIZCA kis26
--------------------
docs/35: ``yaz25``/``guz25`` bloklarinin SOGUK satirlarinin %94-95'i baska
katlarda mevcut ve ``tanim_num`` uzerinden ezberlenebilir; ``kis26``in ezber
orani %0. Soguk kararlar icin gecerli TEK kat kis26'dir. Diger iki blok
BILGI AMACLI basilir ve hukme KATILMAZ.

Uc koruma (docs/39 §9 kuralini kaldirmadan, onu doguran olcut hatasini
duzelterek):
  1. Skor ``olcut.py`` ile TESTIN kVA karisimina agirliklandirilir -- kis26
     soguk medyani 400, testinki 630; en buyuk kovada 3,5 kat eksik.
  2. Skor ``son_islem.py`` buzmesinden (beta=0,60) SONRA olculur -- uretimde
     soguk tahminler oyle kullaniliyor.
  3. Nokta optimumu degil DUZ/MONOTON bolge aranir; degisiklik ancak IZOLE
     bir LB gonderimiyle sinanir.

    python scripts/deney_soguk_kapasite.py
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

DIZIN = KOK / "data" / "interim" / "soguk_kapasite"
KAYIT = KOK / "experiments" / "soguk_kapasite.jsonl"
BLOK = "kis26"
TOHUMLAR = (1000, 1001, 1002)
SOGUK_MASKE = 1.00
BETA = 0.60

#: Kol -> CatBoost ustyazimi. "d7i250" URETIM.
KOLLAR: dict[str, dict[str, object]] = {
    "d5i250": {"depth": 5, "iterations": 250},
    "d6i250": {"depth": 6, "iterations": 250},
    "d7i250": {"depth": 7, "iterations": 250},
    "d8i250": {"depth": 8, "iterations": 250},
    "d6i500": {"depth": 6, "iterations": 500},
    "d7i500": {"depth": 7, "iterations": 500},
}
URETIM = "d7i250"

#: ``d(genel)/d(soguk)`` -- genel = sqrt(0,7784*sicak^2 + 0,2216*soguk^2).
SOGUK_KATSAYI = 0.2216 * 1.82133 / 1.07907


def main() -> int:
    satir_tamponlu_cikti()
    t0 = time.time()
    print("=" * 100)
    print(f"SOGUK UZMAN KAPASITESI -- uretim genisliginde (105 kolon), {BLOK}")
    print("=" * 100)

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    kol = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)
    print(f"  tam set {len(tum)} kolon -> URETIM {len(kol)} kolon")

    parca, dogrulama, gercek, soguk = di.blok_parcalari(egitim, BLOK)
    dg = dogrulama[soguk]
    y = gercek[soguk]
    log_guc = np.log1p(dg["guc"].to_numpy(dtype="float64"))

    te_c = test[test["soguk_mu"] == 1]
    guc_kenar = ol.guc_kenarlari(te_c)
    w, tani = ol.test_agirliklari(dg, te_c, guc_kenar, eksenler=("guc",))
    print(f"  {BLOK} soguk {len(y):,} satir   egitim {len(parca):,} (ek koken YOK)")
    print(f"  kVA agirliklandirmasi ESS %{100 * tani['ess_orani']:.1f}")

    def buz(log_t: np.ndarray, beta: float = BETA) -> np.ndarray:
        r = log_t - log_guc
        return np.clip(np.expm1(r.mean() + beta * (r - r.mean()) + log_guc), 0.0, None)

    DIZIN.mkdir(parents=True, exist_ok=True)
    tahminler: dict[str, list[np.ndarray]] = {}
    for ad, ust in KOLLAR.items():
        tahminler[ad] = []
        for t in TOHUMLAR:
            yol = DIZIN / f"{BLOK}_{t}_{ad}.npy"
            if yol.exists():
                tahminler[ad].append(np.load(yol).astype("float64"))
                continue
            t1 = time.time()
            maskeli = d.soguk_maskele(parca, kol, SOGUK_MASKE, t)
            log_t = di.egit_tahmin("cat", maskeli, dogrulama, kol, t, **ust)
            v = log_t[soguk] if log_t.shape[0] == soguk.size else log_t
            np.save(yol, v.astype("float32"))
            tahminler[ad].append(v.astype("float64"))
            print(f"    {ad:8} tohum {t}  ({time.time() - t1:.0f} sn)")

    print("\n" + "-" * 100)
    print("HUKUM (son islem SONRASI; kVA duzeltilmis = testin karisimi)")
    print("-" * 100)
    print(f"  {'kol':>8}{'HAM kis26':>12}{'kVA duzeltilmis':>18}{'uretime gore':>14}")
    kayitlar = []
    for ad in KOLLAR:
        log_ort = np.mean(tahminler[ad], axis=0)
        t_buz = buz(log_ort)
        ham = ol.agirlikli_rmsle(y, t_buz)
        duz = ol.agirlikli_rmsle(y, t_buz, w)
        kayitlar.append({"kol": ad, "ham": ham, "duzeltilmis": duz})
    u = next(k for k in kayitlar if k["kol"] == URETIM)
    for k in kayitlar:
        bayrak = "  <- URETIM" if k["kol"] == URETIM else ""
        print(
            f"  {k['kol']:>8}{k['ham']:12.5f}{k['duzeltilmis']:18.5f}"
            f"{u['duzeltilmis'] - k['duzeltilmis']:+14.5f}{bayrak}"
        )

    en = min(kayitlar, key=lambda k: k["duzeltilmis"])
    fark = u["duzeltilmis"] - en["duzeltilmis"]
    print(f"\n  en iyi {en['kol']}  {en['duzeltilmis']:.5f}   uretim {u['duzeltilmis']:.5f}")
    print(f"  fark {fark:+.5f}   genel skora tahmini etki {-fark * SOGUK_KATSAYI:+.5f}")
    if en["kol"] == URETIM:
        print("  -> Optimum URETIMDE. Degisiklik onerilmiyor.")
    else:
        print("  -> kis26 TEK kat; nokta optimumuna atlanmaz. Komsu kollara bak:")
        print("     egri duz/monoton ise aday, degilse gurultu. IZOLE LB gonderimiyle sinanir.")

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        for k in kayitlar:
            fh.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
