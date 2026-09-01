"""p32-E: H-capa kazanci GECMIS UZUNLUGUNA mi bagli?

SORU
----
Hcift kazanci yaz25'te SIFIR, guz25'te +0.008, kis26'da +0.025.
Iki rakip aciklama:
 (A) MEVSIM: kazanc kista var, yazda yok -> TEST (Nis-Tem) yaz25 gibi
     davranir, katman ISE YARAMAZ.
 (B) GECMIS UZUNLUGU: capanin (mu_i, sd_i) kalitesi gecmis gun sayisina
     bagli. yaz25'in gecmisi 90 gun (yalniz Oca-Mar, tek mevsim!), guz25
     212, kis26 334, TEST 455 (tam bir yil + fazlasi).
     -> TEST kis26'dan bile IYI capaya sahip, katman ISE YARAR.

AYIRICI OLCUM
-------------
Her blogun kendi icinde trafolari GECMIS GUN SAYISINA gore kovalara ayir
ve kova basina dMSE olc. (A) dogruysa kazanc blok icinde gecmis uzunluguna
DUYARSIZ olur; (B) dogruysa blok icinde de uzun gecmisli trafolarda buyur.

Ek olarak: mevsim KAPSAMI (capanin kac farkli takvim ayini gordugu) da
ayrilir -- (B)'nin gercek mekanizmasi bu olabilir.

Cikti: p_kalici/p32_katmanlar.json ["K3c_gecmis_uzunlugu"]
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import p24_b_olc as PB
from p32_d_hcapa import wins_h
from p32_ortak import BLOKLAR, KESIM, PK, _ham

K = 2.0
KOVA = [90, 150, 210, 270, 330, 400]


def capa_ve_kapsam(kesim_ad: str, tanimlar: np.ndarray):
    tr = _ham()
    g = tr[tr["tarih"] < pd.Timestamp(KESIM[kesim_ad])]
    lg = np.log1p(np.clip(g["tuketim"].to_numpy("float64"), 0, None))
    df = pd.DataFrame(
        {"tanim": g["tanim"].to_numpy(), "l": lg, "ay": g["tarih"].dt.month.to_numpy()}
    )
    gb = df.groupby("tanim")
    mu, sd, n = gb["l"].mean(), gb["l"].std(), gb["l"].size()
    ay = gb["ay"].nunique()
    idx = pd.Index(tanimlar)
    return (
        mu.reindex(idx).to_numpy("float64"),
        sd.reindex(idx).to_numpy("float64"),
        n.reindex(idx).to_numpy("float64"),
        ay.reindex(idx).to_numpy("float64"),
    )


def main() -> None:
    B = PB.veri_kur()
    R: dict = {"00_k": K, "01_soru": "kazanc MEVSIM'e mi GECMIS UZUNLUGUNA mi bagli?"}

    kov: dict = {}
    ayk: dict = {}
    for b in BLOKLAR:
        bb = B[b]
        mu, sd, n, ay = capa_ve_kapsam(b, bb["tanim"])
        p0 = PB.harman(bb, PB.ADAYLAR["URETIM"])
        p1 = wins_h(p0, K, "cift", (mu, sd))
        e0, e1 = bb["y"] - p0, bb["y"] - p1
        d2 = (e0 * e0 - e1 * e1) * bb["w"]
        kb = np.digitize(np.nan_to_num(n, nan=-1), KOVA)
        kov[b] = {
            f"kova{int(i)}": {
                "gecmis_gun_araligi": ([0] + KOVA + [9999])[int(i)],
                "n_satir": int((kb == i).sum()),
                "dMSE_agr": round(float(d2[kb == i].mean()), 6),
            }
            for i in np.unique(kb)
        }
        ab = np.digitize(np.nan_to_num(ay, nan=-1), [3, 6, 9, 12])
        ayk[b] = {
            f"ay{int(i)}": {
                "n_satir": int((ab == i).sum()),
                "dMSE_agr": round(float(d2[ab == i].mean()), 6),
            }
            for i in np.unique(ab)
        }
    R["02_gecmis_gun_kovalari"] = kov
    R["03_mevsim_kapsami_kovalari"] = ayk

    # blok basina gecmis/kapsam ozeti + TEST
    ozet: dict = {}
    T = pd.read_parquet(os.path.join(PB.DN, "test.parquet"))
    ts = T[T["soguk_mu"] == 0]
    mu, sd, n, ay = capa_ve_kapsam("TEST", ts["tanim"].astype(str).to_numpy())
    for ad, nn, aa in [(b, capa_ve_kapsam(b, B[b]["tanim"])[2], capa_ve_kapsam(b, B[b]["tanim"])[3])
                       for b in BLOKLAR] + [("TEST", n, ay)]:
        ozet[ad] = {
            "gecmis_gun_medyan": float(np.nanmedian(nn)),
            "gecmis_gun_ort": round(float(np.nanmean(nn)), 1),
            "farkli_ay_medyan": float(np.nanmedian(aa)),
            "farkli_ay_ort": round(float(np.nanmean(aa)), 2),
            "capasi_olan_pay": round(float(np.isfinite(nn).mean()), 4),
        }
    R["04_blok_ozeti"] = ozet

    yol = os.path.join(PK, "p32_katmanlar.json")
    mevcut = {}
    if os.path.exists(yol):
        with open(yol, encoding="utf-8") as fh:
            mevcut = json.load(fh)
    mevcut["K3c_gecmis_uzunlugu"] = R
    with open(yol, "w", encoding="utf-8") as fh:
        json.dump(mevcut, fh, ensure_ascii=False, indent=1)
    print(json.dumps(R, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
