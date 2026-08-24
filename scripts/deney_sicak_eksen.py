"""SICAK TAHMINLER: GUN ekseni ile TRAFO ekseni ayri ayri kalibre edilmeli mi?

NEDEN
-----
Sicakta GLOBAL afin egim olculdu ve ~1 cikti (yaz25 +1,0365 / guz25 +1,0628 /
kis26 +0,9671) -- "yapacak bir sey yok" diye kapatildi. Ama o egim IKI EKSENI
BIRDEN olcuyor:

    r = gun_ort(r) + (r - gun_ort(r))
        [zaman ekseni]      [trafo ekseni]

Sogukta tam bu ayrimin yapilmamis olmasi 0,0017'lik bir kusura yol acmisti
(docs/37): zaman ekseni duz oldugu icin kis26'da bedava gorunen bir islem,
test penceresinde mevsim rampasini eziyordu. Sicakta ayni ayrim HIC
yapilmadi. Global egimin 1 olmasi, iki eksenin AYRI AYRI 1 oldugu anlamina
gelmez -- biri 1'in ustunde biri altinda olup ortalamada 1 verebilir.

Iki egim capraz blokta kestirilir:
    b_gun   gun ortalamalari uzerinde   gercek_gun ~ tahmin_gun
    b_ici   gun ICINDE, merkezlenmis    gercek_ici ~ tahmin_ici

Kesme YOK -- kesme bloga ozgu bir seviyedir ve bloklar arasi tasinmadigi
capraz blokta iki kez olculdu (docs/37: ufuk kalibrasyonu 0/3).

Uygulama:  r' = gun_ort + b_gun_duzeltmesi + b_ici * (r - gun_ort)

Uc blogun UCUNDE de kazanmiyorsa REDDEDILIR.

Onbelleklenmis sicak tahminler (``deney_sicak_agirlik.py`` uretti,
3 tohum torbalanmis -- gonderim gibi). Fit YOK, saniyeler.

    python scripts/deney_sicak_eksen.py
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
import tuketim_model as tm  # noqa: E402

from gridup.reporting import satir_tamponlu_cikti  # noqa: E402

AILELER = ("cat", "xgb", "lgbm")
AGIRLIK = (3.0, 1.0, 1.0)
ONBELLEK = KOK / "data" / "interim" / "deney" / "sicak_tahmin.npz"
KAYIT = KOK / "experiments" / "sicak_eksen.jsonl"


def main() -> int:
    satir_tamponlu_cikti()
    t0 = time.time()
    print("=" * 92)
    print("SICAK: gun ekseni ile trafo ekseni ayri kalibrasyon  --  capraz blok")
    print("=" * 92)

    if not ONBELLEK.exists():
        raise RuntimeError(f"onbellek yok: {ONBELLEK}")
    egitim, test = d.cerceveleri_kur()
    tm.kategorik_kodla(egitim, test)
    z = np.load(ONBELLEK)
    parcalar = {b.ad: di.blok_parcalari(egitim, b.ad) for b in tm.BLOKLAR}

    veri: dict[str, dict[str, np.ndarray]] = {}
    for b in tm.BLOKLAR:
        _, dogrulama, gercek, soguk = parcalar[b.ad]
        pay = sum(AGIRLIK)
        loglar = [
            sum(AGIRLIK[i] * z[f"{b.ad}_{t}_{a}"] for i, a in enumerate(AILELER)) / pay
            for t in di.TOHUMLAR
        ]
        log_t = np.mean(loglar, axis=0)
        dg = dogrulama[~soguk]
        lg = np.log1p(dg["guc"].to_numpy(dtype="float64"))
        gun = pd.to_datetime(dg["tarih"]).to_numpy()
        r = log_t - lg
        g = np.log1p(gercek[~soguk]) - lg
        gr = pd.Series(r).groupby(gun).transform("mean").to_numpy()
        gg = pd.Series(g).groupby(gun).transform("mean").to_numpy()
        veri[b.ad] = {
            "r": r,
            "g": g,
            "lg": lg,
            "y": gercek[~soguk],
            "gun": gun,
            "gr": gr,
            "gg": gg,
            "ri": r - gr,
            "gi": g - gg,
        }

    print(f"\n  {'blok':7}{'b_gun':>9}{'b_ici':>9}{'global':>9}   (kendi bloklarinda)")
    for b in tm.BLOKLAR:
        v = veri[b.ad]
        bg = float(np.polyfit(v["gr"] - v["gr"].mean(), v["gg"] - v["gg"].mean(), 1)[0])
        bi = float(np.polyfit(v["ri"], v["gi"], 1)[0])
        gl = float(np.polyfit(v["r"] - v["r"].mean(), v["g"] - v["g"].mean(), 1)[0])
        print(f"  {b.ad:7}{bg:9.4f}{bi:9.4f}{gl:9.4f}")

    print("\n  CAPRAZ BLOK UYGULAMA (katsayilar diger iki bloktan)")
    print(f"  {'blok':7}{'b_ici(kaynak)':>15}{'once':>10}{'sonra':>10}{'fark':>10}")
    kayitlar = []
    for b in tm.BLOKLAR:
        kaynak = [o.ad for o in tm.BLOKLAR if o.ad != b.ad]
        ri_k = np.concatenate([veri[k]["ri"] for k in kaynak])
        gi_k = np.concatenate([veri[k]["gi"] for k in kaynak])
        b_ici = float(np.polyfit(ri_k, gi_k, 1)[0])
        v = veri[b.ad]
        yeni = v["gr"] + b_ici * v["ri"]
        onceki = tm.rmsle(v["y"], np.clip(np.expm1(v["r"] + v["lg"]), 0.0, None))
        sonraki = tm.rmsle(v["y"], np.clip(np.expm1(yeni + v["lg"]), 0.0, None))
        print(f"  {b.ad:7}{b_ici:15.4f}{onceki:10.5f}{sonraki:10.5f}{sonraki - onceki:+10.5f}")
        kayitlar.append(
            {
                "blok": b.ad,
                "b_ici": b_ici,
                "once": onceki,
                "sonra": sonraki,
                "fark": sonraki - onceki,
            }
        )

    kazanan = sum(1 for k in kayitlar if k["fark"] < 0)
    ort = float(np.mean([k["fark"] for k in kayitlar]))
    hukum = "AL" if kazanan == 3 and ort < -0.001 else "REDDET"
    print(f"\n  {kazanan}/3 blokta kazanc, ortalama {ort:+.5f}   HUKUM: {hukum}")
    print(f"  genel skora tahmini etki {ort * 0.528:+.5f}")

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        for k in kayitlar:
            fh.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
