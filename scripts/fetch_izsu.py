"""IZSU su tuketiminden ILCE x AY mevsimsel profili uretir.

NE VE NEDEN
-----------
Izmir Buyuksehir Acik Veri portali, IZSU'nun ilce+mahalle bazli AYLIK su
abone sayisi ve ortalama tuketimini yayimliyor (2014-01 .. 2024-09,
1.143.347 satir, CC BY 4.0, kayit gerekmez).

Bu, yaz nufus hareketinin en DOGRUDAN vekilidir. Turizm istatistigi
"kac gece konaklandi" der; su verisi "ilcede kac kisi YASIYOR" der --
yazlikci, gunubirlikci ve tarim iscisi dahil.

OLCULDU (2014-2024, konut abonesi, abone basina):

    CESME        2.18x   yaz/kis orani
    KARABURUN    1.68x
    SEFERIHISAR  1.57x
    ...
    KONAK        1.04x
    KARSIYAKA    1.03x
    BORNOVA      0.94x

Kiyi ilceleri yazin iki kat su kullaniyor, kent merkezleri hic artmiyor.
ODEMIS (1.70x) ve TIRE (1.57x) gibi ic ilceler de yuksek -- onlarinki
turizm degil TARIMSAL SULAMA. Yani tek gosterge iki olguyu birden yakalar.

TURIZM VERISIYLE ORTUSMUYOR (olculdu): 24 ortak ilcede Spearman = 0.283.
Yani bu tabloyu turizm gecelemesinden TURETEMEZSINIZ; bagimsiz bilgi.

NEDEN ZAMAN SERISI DEGIL, PROFIL
--------------------------------
Veri 2024-09'da bitiyor. Zaman serisi olarak kullanilsaydi 2025-2026
panelinde NaN olurdu -- egitimde dolu, testte bos: yangin verisinde
kapatilan hatanin aynisi.

Bunun yerine 10 yillik gecmisten ILCE x AY MEVSIMSEL PROFILI cikariyoruz.
Profil bir iklim normali gibidir: hangi yila uygulanirsa uygulansin gecerli,
kapsam sinirindan etkilenmez ve HEDEFTEN turetilmedigi icin fold gerektirmez.

KAPSAM SINIRI -- DURUSTCE
-------------------------
Yalnizca **30 Izmir ilcesi**. Manisa, Aydin, Denizli ve MUGLA yok --
yani Bodrum, Marmaris, Fethiye kapsam disi. Panelin geri kalani NaN kalir.
Bu bir ILCE sinirIdir, ZAMAN sinirI degil: egitim ile test arasinda
degismez, dolayisiyla dagilim kaymasi yaratmaz.

KULLANIM
--------
::

    python scripts/fetch_izsu.py
    python scripts/fetch_izsu.py --max-yil 2023   # profili 2023'te kes

Cikti: ``data/external/izsu_su_profili.parquet`` (ilce_key x ay, 360 satir).
Kaynak: Izmir Buyuksehir Belediyesi Acik Veri, CC BY 4.0.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gridup import read_any  # noqa: E402
from gridup.io_utils import atomic_write_dataframe  # noqa: E402
from gridup.turkish import join_key  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CIKTI = ROOT / "data" / "external" / "izsu_su_profili.parquet"

KAYNAK_URL = "https://openfiles.izmir.bel.tr/309436/docs/izsu-yillik-ilce-mahalle-su-tuketimi.csv"

#: Yaz ve kis aylari. Temmuz-Agustos Ege'de doruk; Ocak-Subat taban.
YAZ_AYLARI = (7, 8)
KIS_AYLARI = (1, 2)

#: Bir ilcenin profile girmesi icin gereken en az ay sayisi. Eksik aylari
#: olan ilce mevsimsellik uretemez; sessizce yanlis oran vermek yerine
#: DISARIDA birakilir.
ASGARI_AY = 12


def indir(hedef: Path, *, timeout: int = 600) -> Path:
    """CSV'yi indirir. Kayit/anahtar gerektirmez."""
    print(f"Indiriliyor: {KAYNAK_URL}")
    yanit = requests.get(KAYNAK_URL, timeout=timeout)
    yanit.raise_for_status()
    hedef.write_bytes(yanit.content)
    print(f"  {len(yanit.content):,} bayt")
    return hedef


def profil_uret(ham: pd.DataFrame, *, max_yil: int | None = None) -> pd.DataFrame:
    """Ham IZSU tablosundan ilce x ay mevsimsel profili cikarir.

    Yalnizca KONUT abonelikleri kullanilir: sanayi/ticari abone sayisi yaz
    nufusuyla degil ekonomik faaliyetle degisir ve sinyali bulandirir.

    Tuketim ABONE BASINA normalize edilir -- mutlak tuketim buyuk ilcede
    her zaman yuksektir, oysa aradigimiz sey nufusa GORE mevsimsel fazlalik.

    Raises:
        ValueError: Beklenen kolonlar yoksa veya hicbir ilce esige ulasmazsa.
    """
    gerekli = {"yil", "ay", "ilce", "abonelik_grubu", "abone_adedi", "ortalama_tuketim"}
    eksik = gerekli - set(ham.columns)
    if eksik:
        raise ValueError(f"IZSU tablosunda beklenen kolonlar yok: {sorted(eksik)}")

    konut = ham[ham["abonelik_grubu"].astype(str).str.strip().str.casefold().eq("konut")]
    if max_yil is not None:
        konut = konut[konut["yil"].astype(int) <= int(max_yil)]
    if konut.empty:
        raise ValueError("Konut abonesi satiri bulunamadi; kaynak semasi degismis olabilir.")

    toplam = konut.groupby(["ilce", "ay"], as_index=False).agg(
        tuketim=("ortalama_tuketim", "sum"), abone=("abone_adedi", "sum")
    )
    # Abone 0 ise oran tanimsizdir; sifira bolup sonsuz uretmek yerine
    # satiri dusuruyoruz -- sonsuz deger modele girmemeli.
    toplam = toplam[toplam["abone"] > 0].copy()
    toplam["abone_basi"] = toplam["tuketim"] / toplam["abone"]

    tam = toplam.groupby("ilce")["ay"].nunique().eq(ASGARI_AY)
    tam_ilceler = set(tam[tam].index)
    atilan = sorted(set(toplam["ilce"]) - tam_ilceler)
    if atilan:
        print(f"  {len(atilan)} ilce 12 aya ulasmadigi icin disarida: {atilan[:5]}")
    toplam = toplam[toplam["ilce"].isin(tam_ilceler)]
    if toplam.empty:
        raise ValueError("Hicbir ilce 12 aylik kapsama ulasmadi.")

    # Ay endeksi: o ayin degeri / ilcenin kendi 12 ay ortalamasi.
    # Ilceler arasi olcek farkini yok eder, yalnizca MEVSIMSELLIK SEKLI kalir.
    ortalama = toplam.groupby("ilce")["abone_basi"].transform("mean")
    toplam["su_ay_endeksi"] = toplam["abone_basi"] / ortalama

    yaz = toplam[toplam["ay"].isin(YAZ_AYLARI)].groupby("ilce")["abone_basi"].mean()
    kis = toplam[toplam["ay"].isin(KIS_AYLARI)].groupby("ilce")["abone_basi"].mean()
    oran = (yaz / kis.where(kis > 0)).replace([np.inf, -np.inf], np.nan).rename("su_yaz_kis")

    profil = toplam.merge(oran, left_on="ilce", right_index=True, how="left")
    profil["ilce_key"] = [join_key(x) for x in profil["ilce"].astype(str)]
    profil = profil[["ilce_key", "ay", "su_ay_endeksi", "su_yaz_kis"]]
    profil["ay"] = profil["ay"].astype("int8")
    return profil.sort_values(["ilce_key", "ay"]).reset_index(drop=True)


def kalite_kapisi(profil: pd.DataFrame) -> None:
    """Kabul edilemez profili YAZMADAN ONCE reddeder."""
    n_ilce = profil["ilce_key"].nunique()
    print(f"  {n_ilce} ilce x {profil['ay'].nunique()} ay = {len(profil)} satir")
    if n_ilce < 20:
        raise ValueError(f"Kalite kapisi: yalnizca {n_ilce} ilce; kaynak eksik gelmis olabilir.")
    if profil.duplicated(subset=["ilce_key", "ay"]).any():
        raise ValueError("Kalite kapisi: tekrarlanan (ilce_key, ay) satiri var.")
    for kolon in ("su_ay_endeksi", "su_yaz_kis"):
        if not np.isfinite(profil[kolon].dropna()).all():
            raise ValueError(f"Kalite kapisi: '{kolon}' sonsuz deger iceriyor.")
    endeks = profil["su_ay_endeksi"]
    if not (endeks.min() > 0.2 and endeks.max() < 5.0):
        raise ValueError(
            f"Kalite kapisi: su_ay_endeksi makul araligin disinda "
            f"({endeks.min():.2f}..{endeks.max():.2f}); normalizasyon bozulmus olabilir."
        )
    oran = profil.drop_duplicates("ilce_key")["su_yaz_kis"].dropna()
    print(f"  yaz/kis orani: {oran.min():.2f}x .. {oran.max():.2f}x (medyan {oran.median():.2f}x)")


def main() -> int:
    ayristirici = argparse.ArgumentParser(description=__doc__)
    ayristirici.add_argument("--max-yil", type=int, default=None, help="Profili bu yilda kes")
    ayristirici.add_argument("--csv", default=None, help="Indirilmis CSV yolu (tekrar indirme)")
    ayristirici.add_argument("--out", default=str(CIKTI))
    args = ayristirici.parse_args()

    if args.csv:
        yol = Path(args.csv)
        if not yol.is_file():
            print(f"HATA: CSV yok: {yol}")
            return 1
    else:
        gecici = Path(tempfile.gettempdir()) / "izsu-su-tuketimi.csv"
        yol = indir(gecici)

    ham = read_any(yol)
    print(f"\nHam tablo: {len(ham):,} satir, {ham['yil'].min()}-{ham['yil'].max()}")

    profil = profil_uret(ham, max_yil=args.max_yil)
    print("\nProfil:")
    kalite_kapisi(profil)

    atomic_write_dataframe(profil, Path(args.out))
    print(f"\nYazildi: {args.out}")
    print("\n  Kaynak: Izmir Buyuksehir Belediyesi Acik Veri (IZSU), CC BY 4.0.")
    print("  KAPSAM: yalnizca 30 Izmir ilcesi. Manisa/Aydin/Denizli/Mugla YOK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
