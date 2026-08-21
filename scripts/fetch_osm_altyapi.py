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
import json
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

#: Overpass ORNEKLERI. AYNALAR ESIT DEGIL -- olculdu (2026-08-21, ayni sorgu):
#:   overpass-api.de          HTTP 200   2,4 sn   252 oge
#:   overpass.kumi.systems    HTTP 502   6,8 sn
#:   overpass.private.coffee  HTTP 500  34,9 sn
#: Bu yuzden duz rotasyon YANLISTI: alti denemenin dordu bozuk sunucuya
#: gidiyor, sonra "Overpass yanit vermedi" deniyordu. Asil sunucu saglikli
#: ve hizli; aynalar yalnizca SON CARE.
OVERPASS_BIRINCIL = "https://overpass-api.de/api/interpreter"
OVERPASS_AYNALAR = (
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)
#: Deneme sirasi: once birincilde israr (gecici 504 bekleyerek gecer), sonra
#: aynalar. Aynalarin arasina yine birincil serpistirilir.
DENEME_SIRASI = (
    OVERPASS_BIRINCIL,
    OVERPASS_BIRINCIL,
    OVERPASS_BIRINCIL,
    OVERPASS_AYNALAR[0],
    OVERPASS_BIRINCIL,
    OVERPASS_AYNALAR[1],
    OVERPASS_BIRINCIL,
)
OVERPASS = OVERPASS_BIRINCIL
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

#: Sorgu YALNIZCA kullandigimiz etiketleri ister. ``["power"]`` genel filtresi
#: power=plant/portal/catenary_mast gibi ise yaramayan her seyi de cekiyordu ve
#: ``out geom`` bunlarin poligonlarini da dolduruyordu -- 504'lerin asil sebebi
#: bu agirliktir.
#:
#: NOKTALAR ICIN ``out body`` SART, ``out tags`` DEGIL: ``out tags`` koordinat
#: dondurmez ve yaricap filtresi lat/lon ister (olculdu: KeyError 'lat').
SORGU = """
[out:json][timeout:180];
node(around:{yaricap_m},{lat},{lon})["power"~"^({nokta})$"];
out body;
way(around:{yaricap_m},{lat},{lon})["power"~"^({hat})$"];
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
    deneme: int = len(DENEME_SIRASI),
) -> dict[str, float]:
    """Bir ilce dairesindeki OSM guc altyapisi: nokta sayilari + hat uzunluklari.

    Hat uzunlugu, ``around`` ile secilen yolun TUM geometrisi uzerinden degil,
    **orta noktasi daire icinde kalan segmentler** uzerinden toplanir. Overpass
    daireye DEGEN yolun tamamini dondurur; tamamini saymak, sinirdaki uzun bir
    iletim hattini tumuyle bu ilceye yazardi.

    DAYANIKLILIK: deneme sirasi ``DENEME_SIRASI`` -- once saglikli birincilde
    israr, aynalar son care. Duz rotasyon OLCULDU ve zararliydi: aynalar 502
    ve 500 donduyordu, yani alti denemenin dordu bosa gidiyordu.
    """
    sorgu = SORGU.format(
        yaricap_m=int(yaricap_km * 1000),
        lat=lat,
        lon=lon,
        nokta="|".join(NOKTA_TIPLERI),
        hat="|".join(HAT_TIPLERI),
    )
    son_hata: Exception | None = None
    for tur in range(deneme):
        adres = DENEME_SIRASI[tur % len(DENEME_SIRASI)]
        try:
            yanit = oturum.post(adres, data={"data": sorgu}, timeout=240)
            if yanit.status_code in (429, 502, 503, 504):
                raise RuntimeError(f"HTTP {yanit.status_code} ({adres.split('/')[2]})")
            yanit.raise_for_status()
            ogeler = yanit.json()["elements"]
            break
        except (requests.RequestException, RuntimeError, ValueError) as hata:
            son_hata = hata
            # Ustel geri cekilme: sunucu yuku gecici, hemen tekrar denemek
            # kotayi tuketir ve kalici 429'a yol acar.
            time.sleep(min(60, 5 * 2**tur))
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

    # CHECKPOINT: 96 ardisik ag cagrisinda bir hata TUM kosuyu cope atmasin.
    # Tamamlanan ilceler diske yazilir; betik yeniden kosunca kalan yerden
    # devam eder. Overpass'a gereksiz yuk bindirmemek de bir nezaket kurali.
    ara_dosya = Path(args.out).with_suffix(".kismi.json")
    tamamlanan: dict[str, dict[str, Any]] = {}
    if ara_dosya.exists():
        tamamlanan = json.loads(ara_dosya.read_text(encoding="utf-8"))
        print(f"  checkpoint bulundu: {len(tamamlanan)} ilce zaten cekilmis, atlaniyor")

    print(f"1/2  {len(referans)} ilce, Overpass ({args.bekleme:.0f} sn bekleme)")
    oturum = requests.Session()
    oturum.headers.update(BASLIKLAR)
    satirlar: list[dict[str, Any]] = []
    for sira, kayit in enumerate(referans.itertuples(index=False), start=1):
        anahtar = f"{kayit.il_key}|{kayit.ilce_key}"
        if anahtar in tamamlanan:
            satirlar.append(tamamlanan[anahtar])
            continue
        alan = float(kayit.alan_km2)
        yaricap = min(MAX_YARICAP_KM, max(MIN_YARICAP_KM, math.sqrt(alan / math.pi)))
        try:
            olcum = ilce_altyapisi(
                oturum, lat=float(kayit.lat), lon=float(kayit.lon), yaricap_km=yaricap
            )
        except Exception as hata:  # noqa: BLE001 -- ilerlemeyi kaybetmemek her hatadan onemli
            # Ilerlemeyi KAYBETME. Genis yakalama BILEREK: bu dongu 96 ag
            # cagrisidir ve beklenmedik bir hata (semada degisiklik, bozuk
            # yanit) saatlerce suren ilerlemeyi cope atardi. Olculdu: ikinci
            # kosuda KeyError geldi, dar ``RuntimeError`` yakalamasi kacirdi
            # ve checkpoint yazilmadi. Hata YUTULMAZ -- basilir ve exit 1.
            ara_dosya.write_text(json.dumps(tamamlanan, ensure_ascii=False), encoding="utf-8")
            print(f"\nHATA ({sira}/{len(referans)}, {anahtar}): {hata}")
            print(f"Ilerleme kaydedildi ({len(tamamlanan)} ilce): {ara_dosya}")
            print("Betigi tekrar kosun; kalan yerden devam eder.")
            return 1
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
        tamamlanan[anahtar] = satir
        ara_dosya.write_text(json.dumps(tamamlanan, ensure_ascii=False), encoding="utf-8")
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
    # SABIT KOLONLARI DUS. Butun ilcelerde ayni degeri tasiyan bir kolon
    # (pratikte: OSM'de hic haritalanmamis bir tip, hepsi 0) SIFIR bilgi
    # tasir; modele girerse yalnizca feature sayisini sisirir ve "bu aile
    # 9 kolon getirdi" gibi yaniltici bir izlenim yaratir. Dusulenler
    # RAPORLANIR -- sessizce yok olmaz, cunku "sifir" burada bir OLCUM
    # sonucudur: o altyapi tipi Turkiye OSM'inde yok demektir.
    olcumler = [k for k in altyapi.columns if k.startswith("osm_")]
    olcum_kolonlari = [k for k in olcumler if k != "osm_yaricap_km"]
    sabitler = [k for k in olcum_kolonlari if altyapi[k].nunique(dropna=False) <= 1]
    if len(sabitler) == len(olcum_kolonlari):
        print("\nHATA: TUM olcum kolonlari sabit -- OSM bu bolgede hicbir sey tasimiyor.")
        print("Aile yayinlanmadi; feature olarak eklemenin anlami yok.")
        return 1
    if sabitler:
        print(f"\n  DUSULEN sabit kolon ({len(sabitler)}): {', '.join(sabitler)}")
        print("    -> bu altyapi tipleri Turkiye OSM'inde haritalanmamis (hepsi ayni deger).")
        altyapi = altyapi.drop(columns=sabitler)

    tamamen_bos = int((altyapi.get("osm_toplam_hat_km", 0) == 0).sum())
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
        # Yalnizca ANAHTARLAR zorunlu: hangi olcum kolonunun hayatta kalacagi
        # OSM kapsamasina baglidir (sabit olanlar dusulur) ve semayi ona
        # baglamak, kapsamanin degistigi gun yayini sebepsiz kirardi.
        required_columns=("il_key", "ilce_key"),
        min_rows=len(referans),
        source=f"{OVERPASS} (OpenStreetMap power=*, ODbL 1.0)",
    )
    # Checkpoint yalnizca YAYIN BASARILI olduktan sonra silinir; publish
    # dogrulamadan gecemezse ilerleme yerinde kalir.
    ara_dosya.unlink(missing_ok=True)
    print(f"\nYazildi: {yol}  ({len(altyapi)} satir, {len(altyapi.columns)} kolon)")
    print("ATIF (ODbL, zorunlu): (c) OpenStreetMap katkicilari")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
