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

SEYREK GUN SORUNU VE AMPIRIK-BAYES HEDEFI
-----------------------------------------
Soguk trafolar test penceresinde SIRAYLA devreye giriyor. Gun basina soguk
satir 1 Nisan'da **1**, 31 Temmuz'da **1.962**. Yani ham "gun ortalamasi"
Nisan basinda tek bir satirin kendisidir ve o satira buzme UYGULANMAZ olur.

Cozum uydurma bir esik degil, dogru kestirici: gun hedefi ampirik-Bayes ile
genel ortalamaya buzulur.

    hedef_d = (n_d * ort_gun_d + M * ort_genel) / (n_d + M)
    M       = sigma^2_gun_ici / sigma^2_gunler_arasi

``M`` ETIKETSIZ turetilir -- yalnizca tahminlerin kendi varyans ayrismasindan.
docs/39 §3'un dersi tam buydu: model-disi bir nicelikten turetilen kestirim
LB'ye TASINDI, tek bir dogrulama blogundan turetilen tasinmadi.

Sonuc kendiliginden dogru davranir: n=1.834 olan bir gun kendi ortalamasinin
~%99'unu korur, n=1 olan bir gun neredeyse tamamen URETIM davranisina duser.
Satirlarin %98,7'si n>=300 olan gunlerde, yani duzeltme kutlenin tamamina
uygulanirken seyrek kuyruk risksizce eski haline birakilir.

GUVENLIK KAPILARI
-----------------
  * Gun ortalamasi (EB hedefine gore) TAM korunur -- dogrulanir, yazdirilir.
  * Sicak satirlara DOKUNULMAZ -- sapma 0 oldugu dogrulanir.
  * beta ayni (0,60); tek degisen buzmenin HEDEFI.
  * ``--m`` verilmezse M olculur; elle verilmesi yalnizca duyarlilik icin.

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


def main() -> int:
    a = argparse.ArgumentParser(description="soguk buzme -- gun ortalamasi korumali")
    a.add_argument("--giris", required=True)
    a.add_argument("--cikis", required=True)
    a.add_argument("--beta", type=float, default=BETA)
    a.add_argument("--m", type=float, default=None, help="EB M; verilmezse olculur")
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

    # ---- AMPIRIK-BAYES GUN HEDEFI (etiketsiz) ----
    d = pd.DataFrame({"g": gun[soguk], "r": r[soguk]})
    sayim = d.groupby("g")["r"].size()
    gun_ort = d.groupby("g")["r"].mean()
    genel = float(r[soguk].mean())

    # Varyans ayrismasi: gunler arasi (satir agirlikli) ve gun ici.
    agir = sayim.to_numpy(dtype="float64")
    s2_arasi = float(np.average((gun_ort.to_numpy() - genel) ** 2, weights=agir))
    s2_ici = float(d.groupby("g")["r"].transform("mean").rsub(d["r"]).pow(2).mean())
    if s2_arasi <= 0:
        raise RuntimeError("gunler arasi varyans sifir -- gun ekseni yok, duzeltme anlamsiz")
    M = float(ar.m) if ar.m is not None else s2_ici / s2_arasi

    agirlik = sayim / (sayim + M)
    hedef_gun = agirlik * gun_ort + (1.0 - agirlik) * genel
    taban = m.loc[soguk, "tarih"].map(hedef_gun).to_numpy(dtype="float64")

    r_yeni = r.copy()
    r_yeni[soguk] = taban + ar.beta * (r[soguk] - taban)
    yeni = np.clip(np.expm1(r_yeni + log_guc), 0.0, None)

    # ---- KAPILAR ----
    # Sicak satirlar HIC dokunulmadan gecer; tek fark expm1(log1p(x)) gidis
    # donusunun kayan nokta artigidir (olculdu: 1,5e-11 mutlak, ~1e-15 goreli).
    # Kapi bu yuzden GORELI: gercek bir degisiklik 1e-12'yi kat kat asar.
    eski_s = m.loc[~soguk, "tuketim"].to_numpy(dtype="float64")
    sicak_sapma = float((np.abs(yeni[~soguk] - eski_s) / np.maximum(np.abs(eski_s), 1.0)).max())
    if sicak_sapma > 1e-12:
        raise RuntimeError(f"sicak satirlar degisti: goreli sapma {sicak_sapma:.3e}")
    yeni_gun_ort = pd.DataFrame({"g": gun[soguk], "r": r_yeni[soguk]}).groupby("g")["r"].mean()
    beklenen = hedef_gun + ar.beta * (gun_ort - hedef_gun)
    gun_sapma = float((yeni_gun_ort - beklenen).abs().max())
    if gun_sapma > 1e-10:
        raise RuntimeError(f"gun ekseni beklendigi gibi degil: sapma {gun_sapma:.3e}")
    if np.isnan(yeni).any() or (yeni < 0).any():
        raise RuntimeError("NaN veya negatif tahmin")

    yeni_std = float(np.sqrt(np.average((yeni_gun_ort.to_numpy() - genel) ** 2, weights=agir)))
    print(f"  soguk satir {int(soguk.sum()):,} / {len(m):,}  gun {gun_ort.size}")
    print(
        f"  gun basina soguk satir: min {int(sayim.min())} medyan {int(sayim.median())} "
        f"maks {int(sayim.max())}"
    )
    print(f"  varyans: gunler arasi {s2_arasi:.5f}  gun ici {s2_ici:.5f}  ->  M = {M:.1f}")
    print(
        f"  EB agirligi: min %{100 * agirlik.min():.1f}  medyan %{100 * agirlik.median():.1f}"
        f"  maks %{100 * agirlik.max():.1f}"
    )
    print(
        f"  gun ekseni std: {np.sqrt(s2_arasi):.5f} -> {yeni_std:.5f}"
        f"   (URETIM {ar.beta * np.sqrt(s2_arasi):.5f} ederdi)"
    )
    print(f"  gun ekseni kapisi TAMAM, azami sapma {gun_sapma:.2e}")
    print(f"  sicak satir GORELI sapmasi {sicak_sapma:.1e}  (kayan nokta artigi)")
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
