"""ESA WorldCover 10m -> ILCE BASINA arazi ortusu oranlari (docs/18 bolum A2).

NEDEN BU BETIK
--------------
Agac temasi dagitim sebekesinde kesintinin en buyuk tek sebep sinifidir;
literaturun vegetation risk model'leri canopy cover/height ile AUC 0,832'ye
cikiyor. Bizde bitki ortusu feature'i SIFIRDI. Model su an "bu ilcede ruzgar
neden kesinti yapar, obruunde yapmaz" sorusuna yalnizca nufus ve ilce kimligi
uzerinden cevap veriyor -- fiziksel yarisi eksik.

Ruzgar x agac ortusu etkilesimi literaturun merkezindeki terimdir. Bu betik o
etkilesimin eksik yarisini getirir.

KAYNAK VE LISANS
----------------
ESA WorldCover 10m v200 (2021), Sentinel-1 + Sentinel-2 turevli, 11 sinif.
Lisans **CC-BY-4.0** -- yeniden dagitima IZINLI, atif ZORUNLU. AWS Open Data
uzerinde acik S3'te barinir; COG oldugu icin TUM tile indirilmez, yalnizca
gereken pencere okunur.

SIZINTI RISKI: YOK. Uretilen sey ilce basina TEK bir statik satirdir; zaman
boyutu yoktur, dolayisiyla ufuk/ambargo hesabina girmez.

GEOMETRI YAKLASIKLIGI -- ACIKCA SOYLENIYOR
------------------------------------------
Elimizde ilce POLIGONU yok; referans tabloda merkez (lat, lon) ve alan_km2
var. Bu yuzden her ilce, **alanina esit bir daireyle** temsil edilir
(r = sqrt(alan/pi), merkez = ilce merkezi). Yani sayilar "ilce sinirlarinin
tam icindeki ortu" DEGIL, "ilce merkezinin cevresindeki, ilce buyuklugunde bir
dairede olculen ortu"dur. Uzun/kiyi ilcelerinde sapar.

Bu bilincli bir tercihtir: gercek poligon icin GADM (ticari kullanimda
kisitli) ya da OSM sinirlari (ek bagimlilik) gerekirdi. Ozellik zaten
ILCELER ARASI GORECE fark icin kullanilacagi icin daire yaklasimi yeterlidir;
poligona gecilirse sayilar degisir ama siralamanin buyuk olcude korunmasi
beklenir. Yarisma gunu ilce poligonu verilirse ``--poligon`` yolu eklenmelidir.

BAGIMLILIK
----------
``rasterio`` bu projenin bagimliligi DEGILDIR (yalnizca bu tek seferlik veri
hazirligi icin gerekir; uretilen parquet kucuktur ve Kaggle tarafina agir
bagimlilik tasimaz). Ayri bir arac ortamindan kosun:

    uv venv geo && VIRTUAL_ENV=geo uv pip install rasterio pandas pyarrow
    ./geo/Scripts/python.exe scripts/fetch_arazi_ortusu.py

KULLANIM
    python scripts/fetch_arazi_ortusu.py                 # 96 ilce
    python scripts/fetch_arazi_ortusu.py --cozunurluk 100
Cikti: data/external/arazi_ortusu_ilce.parquet
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))

REFERANS = KOK / "data" / "reference" / "ilceler_gdz_adm.parquet"
CIKTI = KOK / "data" / "external" / "arazi_ortusu_ilce.parquet"

TABAN_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"

#: WorldCover sinif kodu -> kolon adi. Kodlar urun dokumanindan birebir alindi;
#: 95 (mangrov) Turkiye'de beklenmez ama sema tam tutulur ki "eksik sinif" ile
#: "sifir oranli sinif" karistirilmasin.
SINIFLAR: dict[int, str] = {
    10: "agac_orani",
    20: "calilik_orani",
    30: "otlak_orani",
    40: "tarim_orani",
    50: "yerlesim_orani",
    60: "ciplak_orani",
    70: "kar_buz_orani",
    80: "su_orani",
    90: "sulakalan_orani",
    95: "mangrov_orani",
    100: "yosun_orani",
}

#: WorldCover tile'lari 3x3 derecedir; ad guneybati kosesinden turetilir.
TILE_DERECE = 3
#: Native cozunurluk (m). Decimation faktoru bundan hesaplanir.
NATIVE_M = 10.0
#: Daire yaricapinin makul sinirlari (km). Cok kucuk ilcede orneklem yetmez,
#: cok buyukte pencere gereksiz sisip ag maliyeti buyur.
MIN_YARICAP_KM = 3.0
MAX_YARICAP_KM = 25.0
#: Bir derece enlem kac km (kure yaklasimi; bolgesel farki ihmal edilebilir).
KM_PER_DEG = 111.32


def _tile_adi(lat: float, lon: float) -> str:
    """Verilen noktayi iceren WorldCover tile adini uretir (ornek N36E027)."""
    lat_taban = math.floor(lat / TILE_DERECE) * TILE_DERECE
    lon_taban = math.floor(lon / TILE_DERECE) * TILE_DERECE
    ky = "N" if lat_taban >= 0 else "S"
    dy = "E" if lon_taban >= 0 else "W"
    return f"{ky}{abs(lat_taban):02d}{dy}{abs(lon_taban):03d}"


def _tile_url(ad: str) -> str:
    return f"/vsicurl/{TABAN_URL}/ESA_WorldCover_10m_2021_v200_{ad}_Map.tif"


def _kapsayan_tiles(guney: float, kuzey: float, bati: float, dogu: float) -> list[str]:
    """Kutuyu ortten kesen TUM tile adlari (ilce tile sinirini asabilir)."""
    adlar: list[str] = []
    lat = math.floor(guney / TILE_DERECE) * TILE_DERECE
    while lat <= kuzey:
        lon = math.floor(bati / TILE_DERECE) * TILE_DERECE
        while lon <= dogu:
            adlar.append(_tile_adi(lat + 0.5, lon + 0.5))
            lon += TILE_DERECE
        lat += TILE_DERECE
    return adlar


def ilce_sayimlari(
    rasterio_mod: Any,
    resampling: Any,
    *,
    lat: float,
    lon: float,
    yaricap_km: float,
    cozunurluk_m: float,
    tile_onbellek: dict[str, Any],
) -> dict[int, int]:
    """Bir ilcenin dairesi icindeki WorldCover sinif piksel sayilari.

    Pencere, tile sinirini asarsa her tile'dan ayri okunup sayimlar TOPLANIR
    (kirpip gecmek daireyi sessizce kucultur ve oranlari saptirirdi).
    """
    dlat = yaricap_km / KM_PER_DEG
    dlon = yaricap_km / (KM_PER_DEG * math.cos(math.radians(lat)))
    guney, kuzey = lat - dlat, lat + dlat
    bati, dogu = lon - dlon, lon + dlon

    adim = max(1, int(round(cozunurluk_m / NATIVE_M)))
    sayimlar: dict[int, int] = {}

    for tile_ad in _kapsayan_tiles(guney, kuzey, bati, dogu):
        if tile_ad not in tile_onbellek:
            tile_onbellek[tile_ad] = rasterio_mod.open(_tile_url(tile_ad))
        src = tile_onbellek[tile_ad]
        sinir = src.bounds
        # Kutuyu tile'a kirp; kesisim yoksa atla.
        k_bati, k_dogu = max(bati, sinir.left), min(dogu, sinir.right)
        k_guney, k_kuzey = max(guney, sinir.bottom), min(kuzey, sinir.top)
        if k_bati >= k_dogu or k_guney >= k_kuzey:
            continue

        pencere = rasterio_mod.windows.from_bounds(
            k_bati, k_guney, k_dogu, k_kuzey, transform=src.transform
        )
        genislik = max(1, int(round(pencere.width / adim)))
        yukseklik = max(1, int(round(pencere.height / adim)))
        veri = src.read(
            1,
            window=pencere,
            out_shape=(yukseklik, genislik),
            resampling=resampling.nearest,
            boundless=False,
        )

        # Decimated piksel merkezlerinin koordinatlari (kirpilmis kutu icinde
        # duzgun yayilmis kabul edilir -- decimation duzgun oldugu icin gecerli).
        lon_eksen = np.linspace(
            k_bati + (k_dogu - k_bati) / (2 * genislik),
            k_dogu - (k_dogu - k_bati) / (2 * genislik),
            genislik,
        )
        lat_eksen = np.linspace(
            k_kuzey - (k_kuzey - k_guney) / (2 * yukseklik),
            k_guney + (k_kuzey - k_guney) / (2 * yukseklik),
            yukseklik,
        )
        dx = (lon_eksen[None, :] - lon) * KM_PER_DEG * math.cos(math.radians(lat))
        dy = (lat_eksen[:, None] - lat) * KM_PER_DEG
        daire = (dx**2 + dy**2) <= yaricap_km**2

        secili = veri[daire]
        if secili.size == 0:
            continue
        kod, adet = np.unique(secili, return_counts=True)
        for k, a in zip(kod.tolist(), adet.tolist(), strict=True):
            sayimlar[int(k)] = sayimlar.get(int(k), 0) + int(a)
    return sayimlar


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cozunurluk",
        type=float,
        default=100.0,
        help="Ornekleme cozunurlugu (m). 100 m: ~10x decimation, oran tahmini icin yeterli",
    )
    parser.add_argument("--out", default=str(CIKTI))
    args = parser.parse_args()

    try:
        import rasterio
        import rasterio.windows  # noqa: F401
        from rasterio.enums import Resampling
    except ImportError:
        print(
            "HATA: rasterio yok. Bu betik tek seferlik veri hazirligidir ve "
            "rasterio projenin bagimliligi DEGILDIR. Ayri arac ortami kurun:\n"
            "  uv venv geo && VIRTUAL_ENV=geo uv pip install rasterio pandas pyarrow\n"
            "  ./geo/Scripts/python.exe scripts/fetch_arazi_ortusu.py"
        )
        return 1

    if not REFERANS.exists():
        print(f"HATA: {REFERANS} yok.")
        return 1

    referans = pd.read_parquet(REFERANS)
    gerekli = {"il_key", "ilce_key", "lat", "lon", "alan_km2"}
    eksik = gerekli - set(referans.columns)
    if eksik:
        print(f"HATA: referans tabloda eksik kolon: {sorted(eksik)}")
        return 1

    print(f"1/2  {len(referans)} ilce, WorldCover v200, ornekleme {args.cozunurluk:.0f} m")
    tile_onbellek: dict[str, Any] = {}
    satirlar: list[dict[str, Any]] = []
    try:
        for sira, kayit in enumerate(referans.itertuples(index=False), start=1):
            alan = float(kayit.alan_km2)
            yaricap = min(MAX_YARICAP_KM, max(MIN_YARICAP_KM, math.sqrt(alan / math.pi)))
            sayimlar = ilce_sayimlari(
                rasterio,
                Resampling,
                lat=float(kayit.lat),
                lon=float(kayit.lon),
                yaricap_km=yaricap,
                cozunurluk_m=args.cozunurluk,
                tile_onbellek=tile_onbellek,
            )
            toplam = sum(sayimlar.values())
            satir: dict[str, Any] = {
                "il_key": kayit.il_key,
                "ilce_key": kayit.ilce_key,
                "ortu_yaricap_km": round(yaricap, 2),
                "ortu_piksel": int(toplam),
            }
            for kod, kolon in SINIFLAR.items():
                satir[kolon] = (sayimlar.get(kod, 0) / toplam) if toplam else float("nan")
            satirlar.append(satir)
            if sira % 20 == 0 or sira == len(referans):
                print(f"  {sira}/{len(referans)}")
    finally:
        for src in tile_onbellek.values():
            src.close()

    ortu = pd.DataFrame(satirlar)

    # TURETILMIS RISK VEKILLERI. Ham oranlar zaten modele girer; bunlar
    # literaturun kullandigi birlesik terimleri acik hale getirir.
    ortu["bitki_ortusu_orani"] = ortu["agac_orani"] + ortu["calilik_orani"] + ortu["otlak_orani"]
    #: Agac/yerlesim dengesi: kirsal-agaclik ilce mi, kentsel ilce mi. Yerlesim
    #: sifira giderken bolme patlamasin diye kucuk bir taban eklenir.
    ortu["agac_yerlesim_orani"] = ortu["agac_orani"] / (ortu["yerlesim_orani"] + 1e-3)

    bos = ortu["ortu_piksel"] == 0
    print("\n2/2  ozet")
    print(f"  piksel okunamayan ilce : {int(bos.sum())}")
    print(
        f"  agac orani  ort {ortu['agac_orani'].mean():.3f}  "
        f"min {ortu['agac_orani'].min():.3f}  max {ortu['agac_orani'].max():.3f}"
    )
    print(
        f"  yerlesim    ort {ortu['yerlesim_orani'].mean():.3f}  "
        f"max {ortu['yerlesim_orani'].max():.3f}"
    )
    for etiket, kolon in (("en agacli ", "agac_orani"), ("en kentsel", "yerlesim_orani")):
        ilk = ortu.nlargest(3, kolon)
        metin = ", ".join(f"{r.ilce_key} {getattr(r, kolon):.2f}" for r in ilk.itertuples())
        print(f"  {etiket}  : {metin}")

    from gridup.io_utils import publish_dataframe  # noqa: PLC0415

    yol = Path(args.out)
    yol.parent.mkdir(parents=True, exist_ok=True)
    publish_dataframe(
        ortu,
        yol,
        # Sema sozlesmesi: anahtarlar + iki temel sinif. Hepsini listelemek
        # yeni sinif eklenmesini kirilgan hale getirirdi; bunlar olmadan
        # tablo zaten ise yaramaz.
        required_columns=("il_key", "ilce_key", "agac_orani", "yerlesim_orani"),
        min_rows=len(referans),
        source=f"{TABAN_URL} (ESA WorldCover 10m 2021 v200, CC-BY-4.0)",
    )
    print(f"\nYazildi: {yol}  ({len(ortu)} satir, {len(ortu.columns)} kolon)")
    print(
        "ATIF (CC-BY-4.0, zorunlu): ESA WorldCover 10m 2021 v200, "
        "Zanaga et al., doi:10.5281/zenodo.7254221"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
