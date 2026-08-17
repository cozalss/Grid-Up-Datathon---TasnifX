"""Yangin sicak-nokta tespitlerini indirir (NASA FIRMS, Ege sinir kutusu).

NEDEN BU BETIK
--------------
docs/10 bolum 5: 2021 Marmaris yangini 70k hektar yakti ve Mugla sebekesinde
somut kesinti dalgasi uretti. Yangin gunleri/yakinligi, yaz kesinti riskinin
dogrudan fiziksel sinyalidir.

KAYNAK SECIMI -- EFFIS DENENDI, ERISILEMEDI (2026-08-16 olculdu)
----------------------------------------------------------------
Birincil hedef EFFIS yanmis-alan POLIGONLARIYDI. Gercek denemeler:
  * maps.effis.emergency.copernicus.eu/effis (WFS) -> HTTP 503 (3 deneme,
    90 sn zaman asimi dahil)
  * maps.wild-fire.eu/gwis (WFS)                   -> HTTP 503
  * api2.effis.emergency.copernicus.eu             -> openapi.json yalnizca
    /geocoder, /healtz, /rda-stats, /status yollarini listeler; yanmis-alan
    ucu YOK
  * ies-ows.jrc.ec.europa.eu/effis (WFS)           -> ayakta ama katman
    listesinde yanmis-alan yok; klasik "modis.ba.poly" adi ExceptionReport,
    "ercc.ba" sunucu tarafinda "Failed opening layer" veriyor

YEDEK: NASA FIRMS ulke-yillik CSV arsivi (anahtar GEREKTIRMEZ):
  firms.modaps.eosdis.nasa.gov/data/country/{sensor}/{yil}/{sensor}_{yil}_Turkey.csv

ANLAMSAL FARK -- DIKKAT
-----------------------
FIRMS verisi SICAK NOKTA TESPITIDIR (uydu piksel algisi), yanmis alan
poligonu DEGILDIR. Yani:
  * alan_ha kolonu YOKTUR ve uydurulmamistir -- FRP (fire radiative power)
    siddet vekilidir.
  * Ayni yangin gunde birden cok tespit uretir (piksel basina bir satir);
    "kac yangin" degil "ne kadar yangin aktivitesi" diye okunmali.
  * MODIS (1 km piksel) ve VIIRS-SNPP (375 m piksel) AYNI yangini ikisi de
    gorur; sensorler arasi toplama cift sayimdir. 'aygit' kolonuyla ayrilir.

KAPSAM: yillik dosyalar su an 2020-2024 icin yayimlanmis durumda (2025-2026
dosyalari NASA tarafindan henuz uretilmedi; betik her yili dener, 404'u
atlar -- ileride tekrar calistirilinca yeni yillar kendiliginden eklenir).

Kullanim::

    python scripts/fetch_yangin.py
    python scripts/fetch_yangin.py --start-yil 2020 --end-yil 2026
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gridup.io_utils import atomic_write_dataframe  # noqa: E402

FIRMS_URL = (
    "https://firms.modaps.eosdis.nasa.gov/data/country/{sensor}/{yil}/{sensor}_{yil}_Turkey.csv"
)

#: Ege sinir kutusu -- fetch_deprem.py ile ayni.
LAT_MIN, LAT_MAX = 36.0, 39.5
LON_MIN, LON_MAX = 26.0, 30.5

SENSORLER = ("modis", "viirs-snpp")
REQUEST_PAUSE_S = 1.0
TIMEOUT_S = 120
RETRIES = 3

CIKTI_YOLU = Path("data/external/yanginlar.parquet")


def fetch_firms_yil(sensor: str, yil: int) -> pd.DataFrame | None:
    """Tek sensor-yil CSV'sini ceker. Dosya yayimlanmamissa (404) None doner."""
    url = FIRMS_URL.format(sensor=sensor, yil=yil)
    son_hata: Exception | None = None
    for deneme in range(1, RETRIES + 1):
        try:
            yanit = requests.get(
                url,
                timeout=TIMEOUT_S,
                headers={"User-Agent": "Mozilla/5.0 (datathon veri toplayici)"},
            )
            if yanit.status_code == 404:
                return None
            yanit.raise_for_status()
            return pd.read_csv(io.StringIO(yanit.text))
        except (requests.RequestException, pd.errors.ParserError) as hata:
            son_hata = hata
            if deneme < RETRIES:
                time.sleep(2**deneme)
    raise RuntimeError(f"{url}: {RETRIES} denemede alinamadi. Son hata: {son_hata}")


def _sadelestir(ham: pd.DataFrame, sensor: str) -> pd.DataFrame:
    """Ham FIRMS kolonlarini ortak semaya indirger ve kutuya kirpar.

    'confidence' MODIS'te 0-100 sayi, VIIRS'te l/n/h harfidir; oldugu gibi
    metin olarak saklanir -- iki olcegi tek sayiya zorlamak bilgi uydurmak
    olurdu.
    """
    kutu = ham[
        ham["latitude"].between(LAT_MIN, LAT_MAX) & ham["longitude"].between(LON_MIN, LON_MAX)
    ]
    return pd.DataFrame(
        {
            "tarih": pd.to_datetime(kutu["acq_date"]).dt.date,
            "lat": kutu["latitude"].astype("float64"),
            "lon": kutu["longitude"].astype("float64"),
            "frp": pd.to_numeric(kutu["frp"], errors="coerce"),
            "guven": kutu["confidence"].astype(str),
            "aygit": "MODIS" if sensor == "modis" else "VIIRS-SNPP",
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-yil", type=int, default=2020)
    parser.add_argument("--end-yil", type=int, default=2026)
    parser.add_argument("--out", default=str(CIKTI_YOLU), help="Cikti parquet yolu")
    args = parser.parse_args()

    parcalar: list[pd.DataFrame] = []
    eksik_yillar: list[str] = []
    for sensor in SENSORLER:
        for yil in range(args.start_yil, args.end_yil + 1):
            ham = fetch_firms_yil(sensor, yil)
            if ham is None:
                eksik_yillar.append(f"{sensor}/{yil}")
                print(f"  {sensor} {yil}: dosya yayimlanmamis (404), atlandi")
                continue
            tablo = _sadelestir(ham, sensor)
            print(f"  {sensor} {yil}: {len(ham)} tespit (TR), {len(tablo)} kutu ici")
            parcalar.append(tablo)
            time.sleep(REQUEST_PAUSE_S)

    if not parcalar:
        print("Hicbir sensor-yil dosyasi alinamadi. Internet baglantisini kontrol et.")
        return 1

    birlesik = pd.concat(parcalar, ignore_index=True)
    birlesik = birlesik.drop_duplicates().sort_values(["tarih", "aygit"])
    birlesik = birlesik.reset_index(drop=True)

    cikti = Path(args.out)
    atomic_write_dataframe(birlesik, cikti)

    print(f"Yazildi: {cikti}")
    print(f"  {len(birlesik):,} tespit, {birlesik['tarih'].min()} - {birlesik['tarih'].max()}")
    print("  UYARI: sicak nokta tespiti, yanmis alan poligonu DEGIL (bkz. docstring).")
    if eksik_yillar:
        print(f"  Yayimlanmamis dosyalar: {eksik_yillar}")
    print("  Kaynak: NASA FIRMS (MODIS/VIIRS). Notebook'ta atif ver.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
