"""SON ISLEM -- GUN ORTALAMASI KORUMALI, BASKA HICBIR SEY DEGISMEDEN.

NEDEN AYRI BIR DOSYA
--------------------
``son_islem_gun.py`` LB'de CURUDU (v44 = 1,03053, v30'a gore +0,00414) ama
BES degisikligi birden tasiyordu: gun ekseni korumasi, ilce x kova hucre
etkisi, hucre/model agirlik ayrimi, seyrek gun kapisi, tablo penceresi.
docs/39 §3 curumenin muhtemel sebebini de yaziyor ve o sebep GUN EKSENI
DEGIL -- isareti mevsime gore donen sey HUCRE ETKISI:

    ayni gun x ilce x kVA hucresinde (soguk - sicak) ofset
    yaz25 -0,1690    guz25 +0,3316    kis26 +0,1844

Bu betik o besliden YALNIZCA birincisini alir. Hucre yok, kapi yok, pencere
yok, uydurulan parametre yok. Tek satirlik fark:

    URETIM     r' = ort_TUM(r)  + beta * (r - ort_TUM(r))
    BURADA     r' = ort_GUN(r)  + beta * (r - ort_GUN(r))

Yani buzme trafolar arasi yayilmaya uygulanir, GUNLER arasi rampaya degil.

NEDEN GUN EKSENI KORUNMALI
--------------------------
Buzme gun ortalamalarinin yayilmasini TAM beta katiyla ezer (olculdu: gun
ortalamalarinin std'si 0,3003 -> 0,1802 = x0,60). Test penceresi Nis-Tem ve o
pencerede mevsim rampasi gercek. docs/37'deki tablo (gonderim dosyalari
uzerinde, 158.369 soguk satir):

    ay    v27 (buzmesiz)   v30 (beta=0,60)   2025 ayni ay GERCEK
    04       +0,1219          +0,2611             +0,0408
    05       +0,1169          +0,2581             +0,0056
    06       +0,4138          +0,4363             +0,4706
    07       +0,7642          +0,6465             +0,9517
    rampa     0,642            0,386               0,911

Gercegin rampasi (0,911) buzmesiz modelinkinden (0,642) DIK, buzulmusunden
(0,386) iki kattan fazla dik. Yani buzme gun eksenini YANLIS yone, hem de
buyuk bir katsayiyla itiyor. Gun korumasi bu hatanin YARISINI kaldirir; modelin
kendi rampasinin fazla duz olmasini duzeltmez (o ayri ve daha riskli bir soru).

NEDEN DOGRULAMA BUNU GOREMEZ
----------------------------
kis26 Ara-Mar'dir ve kis ofseti duzdur: ay ekseni varyansi 0,00113. Test
penceresinin mevsimsel ikizinde 0,15298 -- **136 kat**. Yani gun ekseni
korumasi kis26'da neredeyse BEDAVA gorunur ve kazanci yalnizca testte ortaya
cikar. Dogru yapisal duzeltmenin imzasi budur, ama ayni imza sahte bir
duzeltmede de gorunur. Bu yuzden burada hukum LB'ye birakilir ve betik
uretim varsayilanini DEGISTIRMEZ.

GUVENLIK KAPILARI
-----------------
  * Gun ortalamasi TAM olarak korunur -- dogrulanir, yazdirilir.
  * Seyrek gun YOK: test penceresinde her gunun soguk satiri >1.000.
  * Sicak satirlara DOKUNULMAZ -- sapma 0 oldugu dogrulanir.
  * beta ayni (0,60); tek degisen buzmenin HEDEFI.

    python scripts/son_islem_gunsade.py --giris submissions/X_ham.csv \
        --cikis submissions/Y.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
BETA = 0.60
ASGARI_GUN_SATIRI = 200


def main() -> int:
    a = argparse.ArgumentParser(description="soguk buzme -- gun ortalamasi korumali")
    a.add_argument("--giris", required=True)
    a.add_argument("--cikis", required=True)
    a.add_argument("--beta", type=float, default=BETA)
    ar = a.parse_args()

    ornek = pd.read_csv(KOK / "data/raw/sample_submission.csv", encoding="utf-8")
    te = pd.read_csv(
        KOK / "data/raw/test.csv",
        usecols=["id", "tanim", "guc", "tarih"],
        encoding="utf-8",
        dtype={"tanim": str},
    )
    tr = pd.read_csv(
        KOK / "data/raw/train.csv", usecols=["tanim"], encoding="utf-8", dtype={"tanim": str}
    )
    yol = Path(ar.giris)
    if not yol.is_absolute() and not yol.exists():
        yol = KOK / "submissions" / yol.name
    sub = pd.read_csv(yol, encoding="utf-8")
    if not sub["id"].equals(ornek["id"]):
        raise RuntimeError("id sirasi sample_submission ile ayni degil")

    m = sub.merge(te, on="id", how="left", validate="one_to_one")
    if len(m) != len(sub):
        raise RuntimeError("birlestirme satir sayisini bozdu")
    soguk = ~m["tanim"].isin(set(tr["tanim"])).to_numpy()
    gun = m["tarih"].to_numpy()

    log_guc = np.log1p(m["guc"].to_numpy(dtype="float64"))
    r = np.log1p(m["tuketim"].to_numpy(dtype="float64")) - log_guc

    # Gun basina soguk ortalama. Seyrek gun kapisi YOK cunku gerek yok --
    # dogrulanir: her gunun soguk satiri esigin uzerinde.
    d = pd.DataFrame({"g": gun[soguk], "r": r[soguk]})
    sayim = d.groupby("g")["r"].size()
    if int(sayim.min()) < ASGARI_GUN_SATIRI:
        raise RuntimeError(f"seyrek gun var: en az {int(sayim.min())} satir -- kurgu gecersiz")
    gun_ort = d.groupby("g")["r"].mean()
    taban = m.loc[soguk, "tarih"].map(gun_ort).to_numpy(dtype="float64")

    r_yeni = r.copy()
    r_yeni[soguk] = taban + ar.beta * (r[soguk] - taban)
    yeni = np.clip(np.expm1(r_yeni + log_guc), 0.0, None)

    # ---- KAPILAR ----
    sicak_sapma = float(np.abs(yeni[~soguk] - m.loc[~soguk, "tuketim"].to_numpy()).max())
    if sicak_sapma > 0:
        raise RuntimeError(f"sicak satirlar degisti: sapma {sicak_sapma}")
    yeni_gun_ort = pd.DataFrame({"g": gun[soguk], "r": r_yeni[soguk]}).groupby("g")["r"].mean()
    gun_sapma = float((yeni_gun_ort - gun_ort).abs().max())
    if gun_sapma > 1e-10:
        raise RuntimeError(f"gun ortalamasi korunmadi: sapma {gun_sapma:.3e}")
    if np.isnan(yeni).any() or (yeni < 0).any():
        raise RuntimeError("NaN veya negatif tahmin")

    print(f"  soguk satir {int(soguk.sum()):,} / {len(m):,}  gun {gun_ort.size}")
    print(f"  gun basina en az soguk satir {int(sayim.min()):,}  (esik {ASGARI_GUN_SATIRI})")
    print(f"  GUN ORTALAMASI korundu, azami sapma {gun_sapma:.2e}")
    print(f"  sicak satir sapmasi {sicak_sapma:.1e}  (0 olmali)")
    print(f"  gun-ortalamalarinin std'si {float(gun_ort.std()):.5f} (URETIM x{ar.beta:.2f} ederdi)")
    ay = (
        pd.Series(gun_ort.values, index=pd.to_datetime(gun_ort.index))
        .groupby(lambda t: t.month)
        .mean()
    )
    print("  ay bazinda korunan ofset: " + "  ".join(f"{k:02d} {v:+.4f}" for k, v in ay.items()))
    print(f"  min {yeni.min():.1f}  medyan {float(np.median(yeni)):.1f}  maks {yeni.max():.1f}")

    cik = Path(ar.cikis)
    if not cik.is_absolute():
        cik = KOK / "submissions" / cik.name
    pd.DataFrame({"id": sub["id"], "tuketim": yeni}).to_csv(cik, index=False)
    print(f"  yazildi: {cik}  ({len(sub):,} satir)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
