"""HUCRE TABLOSU PENCERESI -- MODELDEN BAGIMSIZ, UC BLOKTA.

BULGU (butunluk sinamasi sirasinda ortaya cikti)
------------------------------------------------
``son_islem_gun.py`` hucre tablosunu ``train.csv``in TAMAMINDAN kuruyor.
kis26 uzerinde olculdu: bu EN KOTU secim.

    tablo kaynagi (kis26 basina kadar)   soguk RMSLE
    11 ay (2025-01'den)                    1,82233   <- uretim
    10 ay                                  1,82188
     9 ay (2025-03'ten)                    1,82127   <- en iyi
     8 ay                                  1,82139
     6 ay                                  1,82209
     3 ay                                  1,82350

Ic optimum var: eski veri satir sayisini artirir (varyans duser) ama
farkli bir rejimden gelir (yanlilik artar). Trafo nufusu donem boyunca
neredeyse IKI KATINA cikiyor (gunluk 2.065 -> 3.896 satir), yani en eski
aylar hem seyrek hem farkli bir populasyon.

AMA tek blokta "son 9 ay" ile "veri setinin ilk 2 ayini at" ayni seye
denk geliyor -- ayrilamiyorlar. Bu betik ikisini AYIRIR: pencere UZUNLUGU
uc blokta birden taranir.

MODELDEN BAGIMSIZ OLCUT
-----------------------
Tablonun kalitesi, model karistirilmadan olculur: gun ICINDE merkezlenmis
tablo etkisi ile gun ICINDE merkezlenmis GERCEK ofset sapmasi arasindaki
R^2. Model yok -> docs/35'teki ezber kanali yok -> UC BLOK DA GECERLI.

Ayrica optimal agirlik da dogrudan cikar: a* = Cov(etki, gercek)/Var(etki).

    python scripts/deney_tablo_pencere2.py
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
import son_islem_gun as si  # noqa: E402
import tuketim_model as tm  # noqa: E402

from gridup.reporting import satir_tamponlu_cikti  # noqa: E402

KAYIT = KOK / "experiments" / "tablo_pencere2.jsonl"
#: Pencere uzunluklari (ay). Hepsi hedef blogun BASLANGICINDA biter.
UZUNLUKLAR = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14)


def main() -> int:
    satir_tamponlu_cikti()
    t0 = time.time()
    print("=" * 92)
    print("HUCRE TABLOSU PENCERESI -- modelden bagimsiz, uc blok")
    print("=" * 92)

    egitim, test = d.cerceveleri_kur()
    tm.kategorik_kodla(egitim, test)
    ham = pd.read_csv(KOK / "data/raw/train.csv", encoding="utf-8", dtype={"tanim": str})
    ham["t"] = pd.to_datetime(ham["tarih"])
    veri_bas = ham["t"].min()
    print(f"  ham veri {veri_bas.date()} .. {ham['t'].max().date()}")

    kayitlar = []
    for b in tm.BLOKLAR:
        _, dogrulama, gercek, soguk = di.blok_parcalari(egitim, b.ad)
        dg = dogrulama[soguk]
        lg = np.log1p(dg["guc"].to_numpy(dtype="float64"))
        gun = pd.to_datetime(dg["tarih"]).to_numpy()
        gercek_ofs = np.log1p(gercek[soguk]) - lg
        # gun ICINDE merkezlenmis gercek sapma -- tablonun aciklamasi gereken sey
        g_ici = gercek_ofs - pd.Series(gercek_ofs).groupby(gun).transform("mean").to_numpy()
        blok_bas = pd.Timestamp(gun.min())
        hedef = pd.DataFrame({"guc": dg["guc"].to_numpy(), "lokasyon": dg["lokasyon"].to_numpy()})
        print(f"\n  {b.ad}  blok basi {blok_bas.date()}  |  {len(g_ici):,} soguk satir"
              f"  |  gun ici gercek std {g_ici.std():.4f}")
        print(f"  {'uzunluk':>8}{'baslangic':>12}{'satir':>10}{'a*':>8}{'R^2 %':>8}")
        for u in UZUNLUKLAR:
            bas = blok_bas - pd.DateOffset(months=u)
            if bas < veri_bas - pd.Timedelta(days=1):
                continue
            kaynak = ham[(ham["t"] >= bas) & (ham["t"] < blok_bas)]
            if len(kaynak) < 50_000:
                continue
            h = si.hucre_etkisi(kaynak, hedef)
            e = h - pd.Series(h).groupby(gun).transform("mean").to_numpy()
            var = float(np.dot(e, e) / len(e))
            if var <= 0:
                continue
            a_yildiz = float(np.dot(e, g_ici) / np.dot(e, e))
            r2 = 100.0 * a_yildiz**2 * var / float(np.dot(g_ici, g_ici) / len(g_ici))
            print(f"  {u:8d}{str(bas.date()):>12}{len(kaynak):10,}{a_yildiz:8.3f}{r2:8.3f}")
            kayitlar.append({"blok": b.ad, "uzunluk": u, "n": int(len(kaynak)),
                             "a_yildiz": a_yildiz, "r2": r2})

    print("\n  UZUNLUGA GORE ORTALAMA R^2 (yuksek = iyi)")
    print(f"  {'uzunluk':>8}{'blok sayisi':>13}{'ort R^2 %':>11}")
    for u in UZUNLUKLAR:
        alt = [k["r2"] for k in kayitlar if k["uzunluk"] == u]
        if len(alt) >= 2:
            print(f"  {u:8d}{len(alt):13d}{np.mean(alt):11.3f}")

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        for k in kayitlar:
            fh.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
