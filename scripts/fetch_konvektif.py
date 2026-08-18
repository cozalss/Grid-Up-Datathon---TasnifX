"""KONVEKTIF INSTABILITE: CAPE, lifted index, konvektif inhibisyon.

NEDEN AYRI BIR UC NOKTA
-----------------------
Yildirim, dagitim sebekesinde hava kaynakli arizanin en buyuk tek
sebeplerindendir. Elimizdeki hicbir degisken bunu yakalamiyordu: yagis
"yagmur yagdi" der, ruzgar "esti" der, ama ikisi de KONVEKTIF firtinayi
sakin bir yagmurdan ayirmaz.

Iki vekil denendi ve IKISI DE OLCUMLE ELENDI (2026-08-17/18):

  weather_code   Izmir'de 3 yil / 26.304 saat cekildi -> gok gurultulu
                 firtina kodu (WMO 95/96/99) SIFIR kez gecti. ERA5 turevi
                 kodlar konvektif kategori uretmiyor.
  cape (arsivde) archive-api'de kolon YANITTA GELIYOR ama %100 NaN.
                 Yani ERA5 arsivinde CAPE yok.

Ucuncu deneme tuttu: ``historical-forecast-api`` FARKLI bir veri setidir --
ERA5 degil, arsivlenmis TAHMIN MODELI kosularindan dikilir ve CAPE o
modellerde yerel bir alandir.

    OLCULDU 2026-08-18, Izmir, 2024-07-01..10:
      archive-api          cape %100 NaN, lifted_index %100 NaN
      historical-forecast  cape %0 NaN (max 1530 J/kg), lifted_index %0 NaN

KAPSAM SINIRI -- ONEMLI VE OLCULDU
----------------------------------
Bu uc nokta arsivlenmis tahminlerden kuruldugu icin gecmisi KISADIR.
Ikili aramayla sinir bulundu:

    2020-06-01  BOS      2021-03-01  BOS
    2020-09-01  BOS      2021-06-01  DOLU (%0 NaN)
    2020-12-01  BOS

Yani veri ~2021 Nisan-Mayis'ta basliyor. Panelimiz 2020-01-01'de basladigi
icin ILK ~15 AY NaN kalir.

BU KABUL EDILEBILIR, cunku bosluk panelin BASINDA -- SONUNDA degil.
Tehlikeli desen "egitimde dolu / testte bos"tur: model bir sinyale guvenmeyi
ogrenir, sonra tam test aninda kaybeder. Burada tam TERSI: test penceresi
(2026) DOLU, yalnizca en eski egitim yili bos. GBDT bunu dogal isler.

KULLANIM
--------
::

    python scripts/fetch_konvektif.py
    python scripts/fetch_konvektif.py --start 2021-05-01 --end 2026-08-15

Cikti: ``data/external/konvektif_gunluk.parquet`` (ilce_key x tarih).
Kaynak: Open-Meteo Historical Forecast API, CC-BY-4.0. Anahtar GEREKMEZ.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_weather import RATE_LIMIT_BACKOFF, cap_end_date  # noqa: E402

from gridup.io_utils import atomic_write_dataframe  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REFERANS = ROOT / "data" / "reference" / "ilceler_gdz_adm.parquet"
CIKTI = ROOT / "data" / "external" / "konvektif_gunluk.parquet"
CKPT_DIR = ROOT / "data" / "external" / ".konvektif_ckpt"

FORECAST_ARCHIVE = "https://historical-forecast-api.open-meteo.com/v1/forecast"

HOURLY_VARIABLES = ["cape", "lifted_index", "convective_inhibition"]

#: Olculen kapsam basi (bkz. modul docstring). Oncesini istemek kotayi bos
#: yere yakar: yanit gelir ama tamami NaN'dir.
KAPSAM_BASI = "2021-05-01"

#: CAPE esikleri (J/kg). Meteoroloji pratigi: ~1000 orta siddetli
#: instabilite, ~2500 kuvvetli. Yuvarlak secilmedi, standart esikler.
CAPE_ORTA = 1000.0
CAPE_KUVVETLI = 2500.0

#: Lifted index NEGATIF oldukca instabildir. -2 altI instabil, -6 altI siddetli.
LI_INSTABIL = -2.0
LI_SIDDETLI = -6.0


def _cek(ad: str, lat: float, lon: float, bas: str, son: str, *, retries: int = 3) -> pd.DataFrame:
    """Tek ilcenin tum araligini ceker. Sessiz bos DataFrame DONMEZ."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": bas,
        "end_date": son,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "Europe/Istanbul",
    }
    son_hata: Exception | None = None
    for deneme in range(1, retries + 1):
        try:
            yanit = requests.get(FORECAST_ARCHIVE, params=params, timeout=120)
            if yanit.status_code == 429:
                bekle = RATE_LIMIT_BACKOFF[min(deneme - 1, len(RATE_LIMIT_BACKOFF) - 1)]
                print(f"  {ad}: hiz/kota siniri; {bekle} sn bekleniyor [{deneme}/{retries}]")
                time.sleep(bekle)
                son_hata = requests.HTTPError("429")
                continue
            yanit.raise_for_status()
            payload = yanit.json()
            break
        except (requests.RequestException, ValueError) as hata:
            son_hata = hata
            if deneme < retries:
                time.sleep(2**deneme)
    else:
        raise RuntimeError(f"{ad}: {retries} denemede alinamadi. Son hata: {son_hata}")

    if "hourly" not in payload:
        raise RuntimeError(f"{ad}: yanitta 'hourly' yok. Yanit: {str(payload)[:250]}")
    birimler = payload.get("hourly_units", {})
    if birimler.get("cape") != "J/kg":
        raise RuntimeError(f"{ad}: cape birimi J/kg degil: {birimler.get('cape')!r}")

    frame = pd.DataFrame(payload["hourly"]).rename(columns={"time": "zaman"})
    frame["zaman"] = pd.to_datetime(frame["zaman"])
    return frame


def gunluge_indir(saatlik: pd.DataFrame, ilce_key: str) -> pd.DataFrame:
    """Saatlik konvektif seriyi gunluk turevlere indirir.

    Gunluk MAKSIMUM ortalamadan cok daha anlamlidir: firtina gunun birkac
    saatinde olur ve ortalama onu seyreltir. Esik-ustu SAAT SAYISI ise
    olayin ne kadar surdugunu tasir.
    """
    f = saatlik.copy()
    f["tarih"] = f["zaman"].dt.normalize()
    cape = pd.to_numeric(f["cape"], errors="coerce")
    li = pd.to_numeric(f["lifted_index"], errors="coerce")

    f["_cape_orta"] = cape >= CAPE_ORTA
    f["_cape_kuvvetli"] = cape >= CAPE_KUVVETLI
    f["_li_instabil"] = li <= LI_INSTABIL
    f["_li_siddetli"] = li <= LI_SIDDETLI

    gun = f.groupby("tarih").agg(
        cape_max=("cape", "max"),
        cape_ort=("cape", "mean"),
        cape_orta_saat=("_cape_orta", "sum"),
        cape_kuvvetli_saat=("_cape_kuvvetli", "sum"),
        li_min=("lifted_index", "min"),
        li_instabil_saat=("_li_instabil", "sum"),
        li_siddetli_saat=("_li_siddetli", "sum"),
        cin_min=("convective_inhibition", "min"),
    )
    for kolon in ("cape_orta_saat", "cape_kuvvetli_saat", "li_instabil_saat", "li_siddetli_saat"):
        gun[kolon] = gun[kolon].astype("int16")
    gun = gun.reset_index()
    gun.insert(0, "ilce_key", ilce_key)
    return gun


def kalite_kapisi(birlesik: pd.DataFrame, n_ilce: int) -> None:
    """Kabul edilemez veriyi YAZMADAN ONCE reddeder."""
    eksik = n_ilce - birlesik["ilce_key"].nunique()
    if eksik > 0:
        raise ValueError(f"Kalite kapisi: {eksik} ilce icin veri yok; panel delik kalir.")
    if birlesik.duplicated(subset=["ilce_key", "tarih"]).any():
        raise ValueError("Kalite kapisi: tekrarlanan (ilce_key, tarih) satiri var.")

    print("  NaN oranlari:")
    red: list[str] = []
    for kolon in birlesik.columns:
        if kolon in ("ilce_key", "tarih"):
            continue
        oran = float(birlesik[kolon].isna().mean())
        isaret = ""
        if oran >= 0.10:
            isaret = "  <-- RED"
            red.append(f"{kolon} %{100 * oran:.1f}")
        print(f"    {kolon:22s} %{100 * oran:.3f}{isaret}")
    if red:
        raise ValueError(f"Kalite kapisi: NaN esigi asildi -> {', '.join(red)}. YAZILMADI.")

    # FIZIK: konvektif aktivite Ege'de YAZIN zirve yapmali. Bu kontrol,
    # sema dogru ama degerler bozuk oldugunda tek uyaran seydir.
    ay = birlesik["tarih"].dt.month
    yaz = float(birlesik.loc[ay.isin([6, 7, 8]), "cape_max"].mean())
    kis = float(birlesik.loc[ay.isin([12, 1, 2]), "cape_max"].mean())
    print(f"  CAPE max: yaz {yaz:.0f} vs kis {kis:.0f} J/kg  (oran {yaz / max(kis, 1e-9):.1f}x)")
    if yaz <= kis:
        raise ValueError("Kalite kapisi: konvektif aktivite yazin yuksek degil -- veri supheli.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default=KAPSAM_BASI, help=f"Baslangic (varsayilan {KAPSAM_BASI})")
    ap.add_argument("--end", default="2026-08-15")
    ap.add_argument("--pause", type=float, default=0.6)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    son, uyari = cap_end_date(args.end)
    if uyari:
        print(f"UYARI: {uyari}")
    if not REFERANS.is_file():
        print(f"HATA: ilce referansi yok: {REFERANS}")
        return 1

    ilceler = pd.read_parquet(REFERANS)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Konvektif cekim: {len(ilceler)} ilce, {args.start} .. {son}")
    print(f"Degiskenler: {', '.join(HOURLY_VARIABLES)}\n")

    basarisiz: list[str] = []
    for sira, satir in enumerate(ilceler.itertuples(index=False), start=1):
        k = str(satir.ilce_key)
        ckpt = CKPT_DIR / f"{k}.parquet"
        if ckpt.is_file() and not args.fresh:
            print(f"[{sira:3d}/{len(ilceler)}] {k:16s} kontrol noktasindan")
            continue
        print(f"[{sira:3d}/{len(ilceler)}] {k:16s} ", end="", flush=True)
        try:
            saatlik = _cek(k, float(satir.lat), float(satir.lon), args.start, son)
            gunluk = gunluge_indir(saatlik, k)
            atomic_write_dataframe(gunluk, ckpt)
            print(f"{len(saatlik):,} saat -> {len(gunluk):,} gun")
        except (RuntimeError, requests.RequestException) as hata:
            print(f"HATA: {hata}")
            basarisiz.append(k)
        time.sleep(args.pause)

    parcalar = [
        pd.read_parquet(CKPT_DIR / f"{k}.parquet")
        for k in ilceler["ilce_key"].astype(str)
        if (CKPT_DIR / f"{k}.parquet").is_file()
    ]
    if not parcalar:
        print("\nHicbir ilce icin veri yok.")
        return 1

    birlesik = pd.concat(parcalar, ignore_index=True)
    birlesik = birlesik.drop_duplicates(subset=["ilce_key", "tarih"])
    birlesik = birlesik.sort_values(["ilce_key", "tarih"]).reset_index(drop=True)
    print(f"\n{len(birlesik):,} satir x {birlesik.shape[1]} kolon")
    kalite_kapisi(birlesik, len(ilceler))

    atomic_write_dataframe(birlesik, CIKTI)
    print(f"\nYazildi: {CIKTI}")
    print(f"  {birlesik['tarih'].min().date()} .. {birlesik['tarih'].max().date()}")
    if basarisiz:
        print(f"\n  UYARI: {len(basarisiz)} ilce alinamadi: {basarisiz[:5]}")
        return 1
    print("\n  Kaynak: Open-Meteo (CC-BY-4.0). Notebook'ta atif ver.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
