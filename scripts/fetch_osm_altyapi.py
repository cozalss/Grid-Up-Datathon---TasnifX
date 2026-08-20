"""OpenStreetMap ``power=*`` -> ILCE BASINA altyapi maruziyeti (docs/18 bolum A1).

NEDEN BU BETIK
--------------
Kesinti tahmini literaturunun en net ablasyonu (arXiv 2404.03115, ayni model
ayni hedef): yalniz hava MAE 0,03547 -> +mesafeler 0,01916 -> +sosyo-ekonomik
ve ALTYAPI SAYIMLARI 0,01346. Kullandiklari altyapi feature'lari tam olarak
OSM'den cekilmis direk/hat/trafo/salt sayilariydi.

Bizde bu aile SIFIRDI. Referans tabloda yalnizca nufus, alan, lat/lon var --
yani model "bu ilcede kac km hat asili, kac direk var" bilgisine hic sahip
degildi. Kesinti fiziksel olarak ALTYAPI uzerinde olur; maruziyet olmadan
hava tek basina yarim bir aciklamadir.

DIKKAT -- KAPSAMA VARSAYILMIYOR, OLCULUYOR
------------------------------------------
OSM gonullu kaynaklidir. Turkiye'de iletim hatlari (power=line/tower) iyi
haritalanmisken DAGITIM sebekesi (power=pole/minor_line) seyrek olabilir.
Bu bir problem degil, bir OLCUM konusudur: betik her ilce icin sayimlari ve
sifir cikan ilce sayisini raporlar. Cogu ilcede sifir cikarsa feature ISE
YARAMAZ ve bunu acikca soyleriz -- LOGO ablasyonu son sozu soyler.

LISANS
------
OpenStreetMap, **ODbL 1.0**. Yeniden dagitima izinli; ATIF ZORUNLU ve
turetilmis veritabani ayni lisansla paylasilir. Notebook'a atif hucresi
girmeli: "(c) OpenStreetMap katkicilari, ODbL".

GEOMETRI: WorldCover ile AYNI daire yaklasimi (r = sqrt(alan/pi), merkez =
ilce merkezi). Iki aile ayni cografi tanimi paylassin diye bilerek boyle --
yoksa "agac orani" ile "direk yogunlugu" farkli alanlari olcerdi.

SIZINTI RISKI: YOK. Ilce basina tek statik satir; zaman boyutu yok.

KULLANIM
    python scripts/fetch_osm_altyapi.py              # 96 ilce, ~10 dk
    python scripts/fetch_osm_altyapi.py --bekleme 3  # Overpass'a daha nazik
Cikti: data/external/osm_altyapi_ilce.parquet
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))

from gridup.io_utils import publish_dataframe  # noqa: E402

REFERANS = KOK / "data" / "reference" / "ilceler_gdz_adm.parquet"
CIKTI = KOK / "data" / "external" / "osm_altyapi_ilce.parquet"

OVERPASS = "https://overpass-api.de/api/interpreter"
#: Overpass sunuculari User-Agent'siz istegi 406 ile reddediyor (olculdu).
BASLIKLAR = {"User-Agent": "GridUpDatathon/1.0 (+veri hazirligi; iletisim: repo sahibi)"}

#: WorldCover ile AYNI yaricap kurali -- iki aile ayni alani olcsun.
MIN_YARICAP_KM = 3.0
MAX_YARICAP_KM = 25.0
KM_PER_DEG = 111.32

#: Sayilacak nokta tipleri: power=<deger> -> kolon adi.
NOKTA_TIPLERI = {
    "pole": "osm_direk",
    "tower": "osm_kule",
    "transformer": "osm_trafo",
    "substation": "osm_salt",
    "portal": "osm_portal",
    "generator": "osm_uretim",
}
#: Uzunlugu olculecek hat tipleri: power=<deger> -> kolon adi (km).
HAT_TIPLERI = {
    "line": "osm_iletim_hat_km",
    "minor_line": "osm_dagitim_hat_km",
    "cable": "osm_kablo_km",
}

SORGU = """
[out:json][timeout:180];
(
  node(around:{yaricap_m},{lat},{lon})["power"];
  way(around:{yaricap_m},{lat},{lon})["power"];
);
out tags geom;
"""


def _mesafe_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Kucuk mesafeler icin duzlem yaklasimi (bolgesel olcekte hatasi ihmal edilir)."""
    dy = (lat2 - lat1) * KM_PER_DEG
    dx = (lon2 - lon1) * KM_PER_DEG * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot(dx, dy)


def ilce_altyapisi(
    oturum: requests.Session,
    *,
    lat: float,
    lon: float,
    yaricap_km: float,
    deneme: int = 3,
) -> dict[str, float]:
    """Bir ilce dairesindeki OSM guc altyapisi: nokta sayilari + hat uzunluklari.

    Hat uzunlugu, ``around`` ile secilen yolun TUM geometrisi uzerinden degil,
    **orta noktasi daire icinde kalan segmentler** uzerinden toplanir. Overpass
    daireye DEGEN yolun tamamini dondurur; tamamini saymak, sinirdaki uzun bir
    iletim hattini tumuyle bu ilceye yazardi.
    """
    sorgu = SORGU.format(yaricap_m=int(yaricap_km * 1000), lat=lat, lon=lon)
    son_hata: Exception | None = None
    for tur in range(deneme):
        try:
            yanit = oturum.post(OVERPASS, data={"data": sorgu}, timeout=240)
            if yanit.status_code in (429, 502, 503, 504):
                raise RuntimeError(f"Overpass mesgul: HTTP {yanit.status_code}")
            yanit.raise_for_status()
            ogeler = yanit.json()["elements"]
            break
        except (requests.RequestException, RuntimeError, ValueError) as hata:
            son_hata = hata
            # Ustel geri cekilme: sunucu yuku gecici, hemen tekrar denemek
            # kotayi tuketir ve kalici 429'a yol acar.
            time.sleep(5 * (tur + 1))
    else:
        raise RuntimeError(f"Overpass {deneme} denemede yanit vermedi: {son_hata}")

    sonuc: dict[str, float] = dict.fromkeys(NOKTA_TIPLERI.values(), 0.0)
    sonuc.update(dict.fromkeys(HAT_TIPLERI.values(), 0.0))

    for oge in ogeler:
        tur_deger = (oge.get("tags") or {}).get("power")
        if tur_deger is None:
            continue
        if oge.get("type") == "node" and tur_deger in NOKTA_TIPLERI:
            if _mesafe_km(lat, lon, oge["lat"], oge["lon"]) <= yaricap_km:
                sonuc[NOKTA_TIPLERI[tur_deger]] += 1.0
        elif oge.get("type") == "way" and tur_deger in HAT_TIPLERI:
            nokta = oge.get("geometry") or []
            for onceki, simdiki in zip(nokta, nokta[1:], strict=False):
                orta_lat = (onceki["lat"] + simdiki["lat"]) / 2
                orta_lon = (onceki["lon"] + simdiki["lon"]) / 2
                if _mesafe_km(lat, lon, orta_lat, orta_lon) > yaricap_km:
                    continue
                sonuc[HAT_TIPLERI[tur_deger]] += _mesafe_km(
                    onceki["lat"], onceki["lon"], simdiki["lat"], simdiki["lon"]
                )
    return sonuc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bekleme", type=float, default=2.0, help="Istekler arasi saniye (Overpass nezaketi)"
    )
    parser.add_argument("--out", default=str(CIKTI))
    args = parser.parse_args()

    if not REFERANS.exists():
        print(f"HATA: {REFERANS} yok.")
        return 1
    referans = pd.read_parquet(REFERANS)

    print(f"1/2  {len(referans)} ilce, Overpass ({args.bekleme:.0f} sn bekleme)")
    oturum = requests.Session()
    oturum.headers.update(BASLIKLAR)
    satirlar: list[dict[str, Any]] = []
    for sira, kayit in enumerate(referans.itertuples(index=False), start=1):
        alan = float(kayit.alan_km2)
        yaricap = min(MAX_YARICAP_KM, max(MIN_YARICAP_KM, math.sqrt(alan / math.pi)))
        olcum = ilce_altyapisi(
            oturum, lat=float(kayit.lat), lon=float(kayit.lon), yaricap_km=yaricap
        )
        daire_km2 = math.pi * yaricap**2
        satir: dict[str, Any] = {
            "il_key": kayit.il_key,
            "ilce_key": kayit.ilce_key,
            "osm_yaricap_km": round(yaricap, 2),
            **{ad: round(deger, 3) for ad, deger in olcum.items()},
        }
        satir["osm_toplam_hat_km"] = round(
            sum(olcum[ad] for ad in HAT_TIPLERI.values()),
            3,
        )
        satir["osm_direk_yogunlugu"] = round(olcum["osm_direk"] / daire_km2, 4)
        satir["osm_hat_yogunlugu"] = round(satir["osm_toplam_hat_km"] / daire_km2, 4)
        satir["osm_kule_yogunlugu"] = round(olcum["osm_kule"] / daire_km2, 4)
        satirlar.append(satir)
        if sira % 10 == 0 or sira == len(referans):
            print(f"  {sira}/{len(referans)}")
        time.sleep(args.bekleme)

    altyapi = pd.DataFrame(satirlar)

    # KAPSAMA RAPORU -- feature'in ise yarayip yaramayacagini BURASI soyler.
    print("\n2/2  kapsama (OSM gonullu kaynaklidir; sifir = 'haritalanmamis')")
    for kolon in ("osm_direk", "osm_kule", "osm_trafo", "osm_salt"):
        sifir = int((altyapi[kolon] == 0).sum())
        print(
            f"  {kolon:12s} sifir olan ilce {sifir:3d}/{len(altyapi)}  "
            f"medyan {altyapi[kolon].median():8.1f}  max {altyapi[kolon].max():8.0f}"
        )
    for kolon in ("osm_iletim_hat_km", "osm_dagitim_hat_km", "osm_toplam_hat_km"):
        sifir = int((altyapi[kolon] == 0).sum())
        print(
            f"  {kolon:20s} sifir {sifir:3d}/{len(altyapi)}  "
            f"medyan {altyapi[kolon].median():8.1f} km"
        )
    tamamen_bos = int((altyapi["osm_toplam_hat_km"] + altyapi["osm_direk"] == 0).sum())
    print(f"  HIC altyapi bulunamayan ilce: {tamamen_bos}/{len(altyapi)}")
    if tamamen_bos > len(altyapi) // 2:
        print(
            "  UYARI: ilcelerin yarisindan cogunda kayit yok. Bu aile muhtemelen"
            " ISE YARAMAZ; LOGO ablasyonuna sokmadan gonderime alma."
        )

    yol = Path(args.out)
    yol.parent.mkdir(parents=True, exist_ok=True)
    publish_dataframe(
        altyapi,
        yol,
        required_columns=("il_key", "ilce_key", "osm_direk", "osm_toplam_hat_km"),
        min_rows=len(referans),
        source=f"{OVERPASS} (OpenStreetMap power=*, ODbL 1.0)",
    )
    print(f"\nYazildi: {yol}  ({len(altyapi)} satir, {len(altyapi.columns)} kolon)")
    print("ATIF (ODbL, zorunlu): (c) OpenStreetMap katkicilari")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
