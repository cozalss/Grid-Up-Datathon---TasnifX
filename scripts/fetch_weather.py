"""GDZ/ADM hizmet bolgesi icin gecmis hava durumu verisi indirir (Open-Meteo).

NEDEN BU BETIK
--------------
Elektrik dagitim problemlerinde -- ariza tahmini, yuk tahmini, kesinti suresi --
en guclu HARICI sinyal hava durumudur:

  * Sicaklik -> klima/isitma yuku (yaz ve kis cift tepeli tuketim egrisi)
  * Ruzgar   -> agac temasi, iletken salinimi, direk hasari
  * Yagis    -> yalitim bozulmasi, toprak kacagi, sel
  * Yildirim -> ani ariza (en guclu tekil ariza sebebi)

Open-Meteo'nun arsiv API'si SECILDI cunku:
  * API anahtari GEREKTIRMEZ (kayit yok, kota takibi yok)
  * 1940'a kadar gecmis, saatlik cozunurluk
  * Ticari olmayan kullanim ucretsiz (CC-BY-4.0 atif ile)
  * ERA5 yeniden analiz verisine dayanir -- istasyon boslugu yok

KAGGLE UYARISI
--------------
Kaggle notebook'unda internet KAPALI olabilir. Akis su olmali:
  1. Bu betigi YERELDE calistir -> data/external/hava_gunluk.parquet
  2. Dosyayi Kaggle'a "Dataset" olarak yukle
  3. Notebook'ta o dataset'i input olarak ekle ve oku
Notebook icinde canli API cagirmak, internet kapaliysa yarismanin son gunu
sessizce cokmenin en hizli yoludur.

Kullanim::

    python scripts/fetch_weather.py --start 2022-01-01 --end 2026-09-01
    python scripts/fetch_weather.py --districts        # ilce bazinda (daha ince)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gridup.turkish import join_key  # noqa: E402

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Il merkezleri. GDZ = Izmir, Manisa | ADM = Aydin, Denizli, Mugla.
# Koordinatlar il merkezidir; buyuk illerde ilce bazi daha dogrudur (--districts).
PROVINCE_COORDINATES: dict[str, tuple[float, float]] = {
    "İzmir": (38.4237, 27.1428),
    "Manisa": (38.6191, 27.4289),
    "Aydın": (37.8560, 27.8416),
    "Denizli": (37.7765, 29.0864),
    "Muğla": (37.2153, 28.3636),
}

# Sahil/turizm ve tarim ilceleri: il merkezinden iklimsel olarak belirgin farkli.
DISTRICT_COORDINATES: dict[str, tuple[float, float]] = {
    "Çeşme": (38.3235, 26.3060),
    "Bergama": (39.1204, 27.1804),
    "Ödemiş": (38.2294, 27.9714),
    "Akhisar": (38.9186, 27.8404),
    "Salihli": (38.4823, 28.1394),
    "Kuşadası": (37.8600, 27.2597),
    "Didim": (37.3775, 27.2661),
    "Nazilli": (37.9134, 28.3200),
    "Söke": (37.7500, 27.4100),
    "Çivril": (38.2986, 29.7386),
    "Bodrum": (37.0344, 27.4305),
    "Fethiye": (36.6213, 29.1164),
    "Marmaris": (36.8550, 28.2740),
    "Milas": (37.3164, 27.7839),
    "Datça": (36.7256, 27.6889),
}

# Gunluk degiskenler: dagitim sebekesi acisindan anlamli olanlar.
DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "precipitation_hours",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "shortwave_radiation_sum",
]

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_gusts_10m",
    "cloud_cover",
]

# Turkce kolon adlari -- veri setinin geri kalaniyla tutarli olsun.
DAILY_RENAME = {
    "time": "tarih",
    "temperature_2m_max": "sicaklik_max",
    "temperature_2m_min": "sicaklik_min",
    "temperature_2m_mean": "sicaklik_ort",
    "apparent_temperature_max": "hissedilen_max",
    "precipitation_sum": "yagis_toplam",
    "rain_sum": "yagmur_toplam",
    "snowfall_sum": "kar_toplam",
    "precipitation_hours": "yagis_saati",
    "wind_speed_10m_max": "ruzgar_max",
    "wind_gusts_10m_max": "firtina_max",
    "shortwave_radiation_sum": "gunes_radyasyon",
}


def fetch_location(
    name: str,
    latitude: float,
    longitude: float,
    start: str,
    end: str,
    *,
    hourly: bool = False,
    timeout: int = 60,
    retries: int = 3,
) -> pd.DataFrame:
    """Tek bir konum icin hava verisi ceker.

    Raises:
        RuntimeError: Tum denemeler basarisiz olursa. SESSIZ bos DataFrame
            dondurmek yerine acikca patlar -- eksik hava verisi, farkedilmeden
            modeli bozar.
    """
    resolution = "hourly" if hourly else "daily"
    variables = HOURLY_VARIABLES if hourly else DAILY_VARIABLES

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start,
        "end_date": end,
        resolution: ",".join(variables),
        "timezone": "Europe/Istanbul",
    }

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(ARCHIVE_URL, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            break
        except (requests.RequestException, ValueError) as error:
            last_error = error
            if attempt < retries:
                wait = 2**attempt
                print(f"  {name}: deneme {attempt} basarisiz ({error}); {wait} sn bekleniyor")
                time.sleep(wait)
    else:
        raise RuntimeError(f"{name}: {retries} denemede veri alinamadi. Son hata: {last_error}")

    if resolution not in payload:
        raise RuntimeError(
            f"{name}: yanitta '{resolution}' bolumu yok. Yanit: {str(payload)[:300]}"
        )

    frame = pd.DataFrame(payload[resolution])
    frame = frame.rename(columns=DAILY_RENAME if not hourly else {"time": "zaman"})

    time_column = "tarih" if not hourly else "zaman"
    frame[time_column] = pd.to_datetime(frame[time_column])

    # join_key: veri setindeki il adiyla eslestirmek icin AYRI kolon.
    # Ham ad gosterim, join_key eslestirme icindir -- karistirma.
    frame.insert(0, "konum", name)
    frame.insert(1, "konum_key", join_key(name))
    return frame


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Sebeke acisindan anlamli turetilmis kolonlar ekler. YENI frame dondurur.

    Bunlar ham hava degerlerinden daha guclu sinyaldir cunku FIZIKSEL bir
    mekanizmayi temsil ederler.
    """
    if "sicaklik_ort" not in frame.columns:
        return frame

    temperature = frame["sicaklik_ort"]

    new_columns = {
        # Derece-gun: enerji sektorunun standart yuk gostergesi.
        # 18 C nötr kabul edilir; altinda isitma, ustunde sogutma yuku baslar.
        "isitma_derece_gun": (18.0 - temperature).clip(lower=0).astype("float32"),
        "sogutma_derece_gun": (temperature - 18.0).clip(lower=0).astype("float32"),
        # Asiri sicak: klima yukunun dogrusal olmayan sicradigi esik.
        "asiri_sicak": (frame.get("sicaklik_max", temperature) > 35).astype("int8"),
        "asiri_soguk": (frame.get("sicaklik_min", temperature) < 0).astype("int8"),
    }

    if "firtina_max" in frame.columns:
        # Firtina esigi: 60 km/s ustu ruzgar agac/direk hasarinin belirgin arttigi bolge.
        new_columns["firtina_gunu"] = (frame["firtina_max"] > 60).astype("int8")

    if "yagis_toplam" in frame.columns and "firtina_max" in frame.columns:
        # Firtina + yagis birlikte: en yuksek ariza riski.
        new_columns["siddetli_hava"] = (
            (frame["yagis_toplam"] > 20) & (frame["firtina_max"] > 50)
        ).astype("int8")

    return frame.assign(**new_columns)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2022-01-01", help="Baslangic tarihi (YYYY-AA-GG)")
    parser.add_argument("--end", default="2026-09-01", help="Bitis tarihi (YYYY-AA-GG)")
    parser.add_argument("--hourly", action="store_true", help="Saatlik cozunurluk (buyuk dosya)")
    parser.add_argument("--districts", action="store_true", help="Ilce merkezlerini de cek")
    parser.add_argument(
        "--out", default="data/external", help="Cikti dizini (varsayilan: data/external)"
    )
    args = parser.parse_args()

    locations = dict(PROVINCE_COORDINATES)
    if args.districts:
        locations.update(DISTRICT_COORDINATES)

    print(f"{len(locations)} konum, {args.start} - {args.end}, ", end="")
    print("saatlik" if args.hourly else "gunluk")

    frames = []
    failures = []

    for index, (name, (latitude, longitude)) in enumerate(locations.items(), start=1):
        print(f"[{index}/{len(locations)}] {name} ...", end=" ", flush=True)
        try:
            frame = fetch_location(
                name, latitude, longitude, args.start, args.end, hourly=args.hourly
            )
        except RuntimeError as error:
            print(f"BASARISIZ -- {error}")
            failures.append(name)
            continue

        frames.append(frame)
        print(f"{len(frame)} satir")
        time.sleep(0.4)  # API'ye nazik ol

    if not frames:
        print("\nHicbir konum icin veri alinamadi. Internet baglantisini kontrol et.")
        return 1

    combined = pd.concat(frames, ignore_index=True)
    if not args.hourly:
        combined = add_derived_features(combined)

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "saatlik" if args.hourly else "gunluk"
    output_path = output_dir / f"hava_{suffix}.parquet"
    combined.to_parquet(output_path, index=False)

    print(f"\nYazildi: {output_path}")
    print(f"  {len(combined):,} satir x {combined.shape[1]} kolon")
    print(f"  Konumlar: {combined['konum'].nunique()}")
    time_column = "tarih" if not args.hourly else "zaman"
    print(f"  Tarih araligi: {combined[time_column].min()} - {combined[time_column].max()}")
    print("\n  Join icin: veri setindeki il adini join_key() ile normalize et,")
    print("  sonra 'konum_key' kolonuna merge et. ASLA ham .lower() kullanma.")

    if failures:
        print(f"\n  UYARI: {len(failures)} konum alinamadi: {failures}")
        return 1

    print("\n  Kaynak: Open-Meteo (CC-BY-4.0). Notebook'ta atif ver.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
