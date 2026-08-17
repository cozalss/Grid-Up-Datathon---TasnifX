"""GDZ + ADM bolgesinin TUM ilcelerini koordinatlariyla indirir.

NEDEN GEREKLI
-------------
``features/spatial.py`` komsu ilce sinyalini uretiyor -- 2024 kazanan
cozumunde bulunan, "ucuz ve yuksek getirili" bir feature ailesi. Ama
calismasi icin her ilcenin koordinati lazim.

Su an ``fetch_weather.py`` icinde ELLE girilmis 20 konum var. Bolgede
**96 ilce** var (GDZ 47 + ADM 49). Yani komsuluk grafigi eksik kuruluyor:
bir ilcenin gercek en yakin komsusu listede yoksa, model yanlis komsudan
sinyal aliyor.

KAYNAK SECIMI
-------------
Turkiye ilce verisi icin birden fazla acik kaynak var. Sirayla denenir ve
ILKI CALISAN kullanilir; hepsi basarisiz olursa acik hata verilir --
sessizce eksik bir tabloyla devam etmek, komsuluk grafigini fark ettirmeden
bozar.

Calistirma::

    python scripts/fetch_districts.py
    python scripts/fetch_districts.py --all-turkey   # 81 ilin tamami
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gridup.io_utils import publish_dataframe  # noqa: E402
from gridup.turkish import join_key, tr_sorted  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "reference"

# GDZ = Izmir, Manisa | ADM = Aydin, Denizli, Mugla
TARGET_PROVINCES = {"izmir", "manisa", "aydin", "denizli", "mugla"}
COMPANY_OF = {
    "izmir": "GDZ",
    "manisa": "GDZ",
    "aydin": "ADM",
    "denizli": "ADM",
    "mugla": "ADM",
}

# Aday kaynaklar. Her biri (ad, url, ayristirici) uclusu.
SOURCES: list[tuple[str, str]] = [
    ("turkiye-api", "https://turkiyeapi.dev/api/v1/districts?limit=1000"),
    (
        "ubeydeozdmr-raw",
        "https://raw.githubusercontent.com/ubeydeozdmr/turkiye-api/master/data/districts.json",
    ),
]


def _rows_from_turkiye_api(payload: object) -> list[dict]:
    """turkiyeapi.dev bicimi: {"status":..., "data":[{name, province, ...}]}."""
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            return data
    if isinstance(payload, list):
        return payload
    return []


def fetch_districts(timeout: int = 60) -> tuple[pd.DataFrame, str]:
    """Ilce tablosunu ceker. ``(frame, kaynak_adi)`` dondurur.

    Raises:
        RuntimeError: Hicbir kaynak calismazsa.
    """
    errors: list[str] = []

    for name, url in SOURCES:
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, json.JSONDecodeError, ValueError) as error:
            errors.append(f"{name}: {type(error).__name__} -- {str(error)[:80]}")
            continue

        rows = _rows_from_turkiye_api(payload)
        if not rows:
            errors.append(f"{name}: yanit taninmadi (tip={type(payload).__name__})")
            continue

        frame = pd.json_normalize(rows)
        if "name" not in frame.columns:
            errors.append(f"{name}: 'name' kolonu yok. Kolonlar: {list(frame.columns)[:8]}")
            continue

        return frame, name

    raise RuntimeError(
        "Ilce verisi hicbir kaynaktan alinamadi:\n  "
        + "\n  ".join(errors)
        + "\n\nEksik bir tabloyla devam etmek komsuluk grafigini sessizce bozar."
    )


def _pick_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"

# Il sinir kutulari (kabaca, bol paylı). Geocoding sonucunu DOGRULAMAK icin.
#
# NEDEN GEREKLI: ilce adi Turkiye'de benzersiz DEGIL. "Efeler" sorgusu
# Artvin'deki bir yeri dondurdu (41.44, 41.92) -- Aydin'daki Efeler ilcesi
# yerine. Bu, hicbir hata vermeden komsuluk grafigini bozar: model yanlis
# komsudan sinyal alir ve neden calismadigi anlasilmaz.
#
# (lat_min, lat_max, lon_min, lon_max)
PROVINCE_BOUNDS: dict[str, tuple[float, float, float, float]] = {
    "izmir": (37.5, 39.4, 26.0, 28.5),
    "manisa": (38.0, 39.4, 27.0, 29.4),
    "aydin": (37.1, 38.3, 26.9, 29.0),
    "denizli": (36.9, 38.5, 28.2, 30.1),
    "mugla": (36.1, 37.7, 27.0, 29.7),
}


# ELLE GIRILEN KOORDINATLAR.
#
# Bu dort ilce tesadufen eksik degil: hepsi 2012'de 6360 sayili Buyuksehir
# Yasasi ile KURULMUS yeni merkez ilcelerdir. Geocoding servisleri eski
# "Merkez" adini tasiyor ve yeni adlari tanimiyor.
#
# Ayni sebep yarisma verisinde de sorun cikarabilir: 2012 oncesi kayitlar
# eski ilce adlarini tasiyabilir. Veri geldiginde ilce adi kumesini bu
# tabloyla karsilastir.
MANUAL_COORDINATES: dict[str, tuple[float, float]] = {
    "aydin|efeler": (37.8560, 27.8416),  # Aydin merkez
    "manisa|sehzadeler": (38.6191, 27.4289),  # Manisa merkez (dogu)
    "manisa|yunusemre": (38.6350, 27.3650),  # Manisa merkez (bati)
    "mugla|seydikemer": (36.6333, 29.3167),  # Fethiye dogusu
}


def _within_province(latitude: float, longitude: float, province_key: str) -> bool:
    """Koordinat ilin makul sinirlari icinde mi?"""
    bounds = PROVINCE_BOUNDS.get(province_key)
    if bounds is None or pd.isna(latitude) or pd.isna(longitude):
        return False
    lat_min, lat_max, lon_min, lon_max = bounds
    return lat_min <= latitude <= lat_max and lon_min <= longitude <= lon_max


def geocode_districts(
    frame: pd.DataFrame, *, pause: float = 0.4, timeout: int = 30
) -> pd.DataFrame:
    """Ilce merkezlerinin koordinatlarini Open-Meteo geocoding ile doldurur.

    NEDEN BU KAYNAK: API anahtari gerektirmiyor, ucretsiz ve zaten hava verisi
    icin ayni saglayiciyi kullaniyoruz -- yani koordinat ile hava verisi ayni
    referans sisteminde kaliyor.

    ESLESME STRATEJISI: Ilce adi tek basina belirsizdir (Turkiye'de ayni adli
    birden fazla ilce var). Bu yuzden donen adaylar ADMIN1 (il) alanina gore
    filtrelenir ve il eslesmesi ``join_key`` ile yapilir -- ham karsilastirma
    'İzmir' vs 'Izmir' farkinda sessizce basarisiz olurdu.

    Returns:
        ``lat`` ve ``lon`` kolonlari doldurulmus YENI frame.
    """
    latitudes: list[float] = []
    longitudes: list[float] = []
    unmatched: list[str] = []

    def _query(name: str) -> list[dict]:
        try:
            response = requests.get(
                GEOCODE_URL,
                params={"name": name, "count": 20, "language": "tr", "country": "TR"},
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json().get("results", []) or []
        except (requests.RequestException, ValueError):
            return []

    for index, row in enumerate(frame.itertuples(index=False), start=1):
        district, province_key = row.ilce, row.il_key
        latitude = longitude = float("nan")

        # Iki sorgu bicimi: yalin ilce adi, sonra "ilce il". Ikincisi
        # belirsiz adlarda (Merkez, Efeler) dogru ili bulmayi kolaylastirir.
        for query in (district, f"{district} {row.il}"):
            for candidate in _query(query):
                candidate_lat = float(candidate["latitude"])
                candidate_lon = float(candidate["longitude"])

                # Sinir kutusu SART: il adi eslesse bile kutu disindaysa
                # reddedilir. "Efeler" sorgusu Artvin'de bir yer dondurmustu.
                if _within_province(candidate_lat, candidate_lon, province_key):
                    latitude, longitude = candidate_lat, candidate_lon
                    break
            if not pd.isna(latitude):
                break
            time.sleep(pause)

        if pd.isna(latitude):
            unmatched.append(f"{row.il}/{district}")

        latitudes.append(latitude)
        longitudes.append(longitude)

        if index % 20 == 0:
            print(f"    {index}/{len(frame)} ilce geocode edildi")
        time.sleep(pause)

    result = frame.assign(lat=latitudes, lon=longitudes)

    # Elle girilen koordinatlarla tamamla (6360 sayili yasa ile kurulan
    # yeni merkez ilceler -- geocoding tanimıyor).
    filled: list[str] = []
    for key, (manual_lat, manual_lon) in MANUAL_COORDINATES.items():
        mask = (result["anahtar"] == key) & result["lat"].isna()
        if mask.any():
            result.loc[mask, ["lat", "lon"]] = [manual_lat, manual_lon]
            filled.append(key)

    if filled:
        print(f"\n  {len(filled)} ilce elle tanimli koordinatla dolduruldu: {filled}")

    still_missing = result["lat"].isna().sum()
    if unmatched and still_missing:
        print(f"\n  {still_missing} ilce icin il sinirlari icinde koordinat BULUNAMADI:")
        for item in unmatched[:15]:
            print(f"    {item}")
        print("  NaN birakildi -- yanlis koordinat, eksik koordinattan kotudur.")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all-turkey", action="store_true", help="5 il yerine 81 ilin tamamini kaydet"
    )
    parser.add_argument(
        "--no-geocode",
        action="store_true",
        help="Koordinat aramayi atla (hizli, ama komsuluk kurulamaz)",
    )
    parser.add_argument(
        "--pause", type=float, default=0.4, help="Geocoding istekleri arasi bekleme (sn)"
    )
    args = parser.parse_args()

    print("Ilce verisi indiriliyor...")
    raw, source = fetch_districts()
    print(f"  Kaynak: {source}  ({len(raw):,} kayit)")
    print(f"  Kolonlar: {list(raw.columns)[:12]}")

    name_column = _pick_column(raw, ("name",))
    province_column = _pick_column(raw, ("province", "provinceName", "il"))
    lat_column = _pick_column(raw, ("latitude", "lat", "coordinates.latitude"))
    lon_column = _pick_column(raw, ("longitude", "lon", "lng", "coordinates.longitude"))
    population_column = _pick_column(raw, ("population",))
    area_column = _pick_column(raw, ("area",))

    if not (name_column and province_column):
        print(f"HATA: ad/il kolonu bulunamadi. Kolonlar: {list(raw.columns)}")
        return 1

    frame = pd.DataFrame(
        {
            "ilce": raw[name_column].astype(str),
            "il": raw[province_column].astype(str),
        }
    )
    for target, source_column in (
        ("lat", lat_column),
        ("lon", lon_column),
        ("nufus", population_column),
        ("alan_km2", area_column),
    ):
        if source_column:
            frame[target] = pd.to_numeric(raw[source_column], errors="coerce")

    # join_key: yarisma verisiyle eslestirmenin TEK guvenli yolu.
    frame["il_key"] = frame["il"].map(join_key)
    frame["ilce_key"] = frame["ilce"].map(join_key)
    # Ilce adi tek basina benzersiz DEGIL (5 ilde ayni adli ilceler var).
    frame["anahtar"] = frame["il_key"] + "|" + frame["ilce_key"]

    if not args.all_turkey:
        frame = frame[frame["il_key"].isin(TARGET_PROVINCES)].reset_index(drop=True)
        frame["sirket"] = frame["il_key"].map(COMPANY_OF)

    frame = frame.sort_values(["il_key", "ilce_key"]).reset_index(drop=True)

    # Koordinat yoksa geocode et -- komsuluk grafigi buna bagli.
    needs_coords = "lat" not in frame.columns or frame["lat"].isna().all()
    if needs_coords and not args.no_geocode:
        print(f"\nKoordinatlar geocode ediliyor ({len(frame)} ilce)...")
        frame = geocode_districts(frame, pause=args.pause)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "turkiye" if args.all_turkey else "gdz_adm"
    path = OUTPUT_DIR / f"ilceler_{suffix}.parquet"
    csv_path = path.with_suffix(".csv")
    required_columns = ("ilce", "il", "il_key", "ilce_key", "anahtar")
    minimum_rows = 81 if args.all_turkey else len(TARGET_PROVINCES)
    province_count = frame["il_key"].nunique()
    expected_provinces = 81 if args.all_turkey else len(TARGET_PROVINCES)
    if province_count < expected_provinces:
        raise ValueError(
            f"Ilce kaynagi {province_count} il kapsiyor; en az {expected_provinces} gerekli."
        )
    source_url = dict(SOURCES)[source]
    publish_dataframe(
        frame,
        path,
        required_columns=required_columns,
        min_rows=minimum_rows,
        source=source_url,
    )
    publish_dataframe(
        frame,
        csv_path,
        required_columns=required_columns,
        min_rows=minimum_rows,
        source=source_url,
        csv_encoding="utf-8",
    )

    print(f"\nYazildi: {path}")
    print(f"         {csv_path}")
    print(f"  {len(frame)} ilce x {frame.shape[1]} kolon")

    if not args.all_turkey:
        print("\n  Il bazinda ilce sayisi:")
        counts = frame.groupby(["il", "sirket"], observed=True).size()
        for (province, company), count in counts.items():
            print(f"    {province:<10} ({company})  {count:>2} ilce")
        print(f"    {'TOPLAM':<10}        {len(frame):>2} ilce")

    has_coords = "lat" in frame.columns
    missing_coords = (
        int(frame[["lat", "lon"]].isna().any(axis=1).sum()) if has_coords else len(frame)
    )
    if missing_coords:
        print(f"\n  UYARI: {missing_coords} ilcede koordinat eksik.")
        print("  Komsuluk grafigi bu ilceler icin kurulamaz.")
    else:
        print("\n  Tum ilcelerde koordinat var -- komsuluk grafigi kurulabilir.")

    print("\n  Ornek satirlar:")
    show = [c for c in ("il", "ilce", "lat", "lon", "nufus", "anahtar") if c in frame.columns]
    print(frame[show].head(8).to_string(index=False))

    print("\n  Ilce adlari (join_key):")
    print("  " + ", ".join(tr_sorted(frame["ilce_key"].tolist())[:25]) + " ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
