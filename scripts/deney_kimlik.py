"""tanim_* kolonlari SOGUK uzmandan cikarilmali mi? Karar mercii kis26.

NEDEN
-----
``tanim_num`` %100 bire-bir trafo kimligi ve ``t_`` onekiyle baslamadigi icin
maskelemeden SAG CIKIYOR (docs/35). CV'de soguk sayilan trafolarin %94-95'i
baska egitim katlarinda mevcut -> model onlarin seviyesini EZBERLEYEBILIYOR.
Testte 2.024 soguk trafonun 2.024'u gorulmemis -> kanal KAPALI.

Olculdu (2026-08-23, uretim tohum 1):

    blok      kova     ilcexkova    URETIM     fark
    yaz25   1,7663      1,4126     1,47665   -0,2896   model kovadan IYI
    guz25   1,7885      1,5841     1,63689   -0,1516   model kovadan IYI
    kis26   1,8162      1,6830     1,86509   +0,0489   model kovadan KOTU

kis26 ezber orani %0,0 olan TEK kattir ve orada model onemsiz bir tabandan
geride. Yani gorunen soguk kalitemiz buyuk olcude EZBER.

Bu deney tanim_* kolonlarini soguk uzmandan cikarip UC BLOKTA olcer. Beklenti:
kis26'da KAZANC, yaz25/guz25'te KAYIP (cunku oradaki ezber gerceklen kar
getiriyor -- ama testte getirmeyecek). Karar YALNIZCA kis26'ya gore verilir.

    python scripts/deney_kimlik.py
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
import tuketim_model as tm  # noqa: E402

from gridup.reporting import satir_tamponlu_cikti  # noqa: E402

KIMLIK = ("tanim_num", "tanim_uzunluk", "tanim_on2", "tanim_on3", "tanim_on4", "tanim_on5")
USTYAZIM: dict[str, object] = {"depth": 7}
AGIRLIK = {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0}
TOHUMLAR = (1000, 1001, 1002)
KAYIT = KOK / "experiments" / "kimlik.jsonl"


def main() -> int:
    satir_tamponlu_cikti()
    t0 = time.time()
    print("=" * 96)
    print("tanim_* SOGUK uzmandan cikarilsin mi?  --  karar mercii kis26")
    print("=" * 96)

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    uretim = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    var = [k for k in KIMLIK if k in uretim]
    tm.kategorik_kodla(egitim, test)
    print(f"  uretim {len(uretim)} kolon | cikarilacak {len(var)}: {var}")

    tekil: dict[str, dict[tuple[str, int], float]] = {"TABAN": {}, "-KIMLIK": {}}
    for b in tm.BLOKLAR:
        parca, dogrulama, gercek, soguk = di.blok_parcalari(egitim, b.ad)
        satir = []
        for ad, kol in (("TABAN", uretim), ("-KIMLIK", [k for k in uretim if k not in var])):
            blok_skor = []
            for tohum in TOHUMLAR:
                maskeli = d.soguk_maskele(parca, kol, 1.00, tohum)
                loglar = {}
                for aile in AGIRLIK:
                    ust = USTYAZIM if aile == "cat" else {}
                    loglar[aile] = di.egit_tahmin(aile, maskeli, dogrulama, kol, tohum, **ust)
                pay = sum(AGIRLIK.values())
                harman = sum(AGIRLIK[a] * loglar[a] for a in AGIRLIK) / pay
                tah = np.clip(np.expm1(harman), 0.0, None)
                s = tm.rmsle(gercek[soguk], tah[soguk])
                tekil[ad][(b.ad, tohum)] = s
                blok_skor.append(s)
            satir.append(f"{ad} {np.mean(blok_skor):.5f}")
        f = np.array([tekil["TABAN"][(b.ad, t)] - tekil["-KIMLIK"][(b.ad, t)] for t in TOHUMLAR])
        isaret = "KAZANC" if f.mean() > 0 else "KAYIP"
        print(
            f"  {b.ad:6} {satir[0]:20} {satir[1]:22} "
            f"fark {f.mean():+.5f}  ({(f > 0).sum()}/3)  {isaret}"
        )

    print("\n--- KARAR: yalnizca kis26 ---")
    fk = np.array([tekil["TABAN"][("kis26", t)] - tekil["-KIMLIK"][("kis26", t)] for t in TOHUMLAR])
    o, sh = float(fk.mean()), float(fk.std(ddof=1) / np.sqrt(len(fk)))
    t_d = o / sh if sh > 0 else 0.0
    print(f"  kis26 soguk farki {o:+.5f}  SH {sh:.5f}  t {t_d:+.2f}")
    print(f"  genel skora etkisi {o * 0.350:+.5f}   (d(genel)/d(soguk) = 0,350)")
    hukum = "CIKAR" if t_d >= 2 else ("TUT" if t_d <= -2 else "esik alti -- KARARSIZ")
    print(f"  HUKUM: {hukum}")

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        kayit = {"kis26_fark": o, "sh": sh, "t": t_d, "hukum": hukum}
        fh.write(json.dumps(kayit, ensure_ascii=False) + "\n")
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
