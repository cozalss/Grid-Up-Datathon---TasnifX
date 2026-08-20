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
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gridup.io_utils import publish_dataframe, validate_published_dataframe  # noqa: E402
from gridup.turkish import join_key  # noqa: E402

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# 429 (hiz siniri) sonrasi bekleme merdiveni, saniye. Olculen: uzun tarih
# araligi + cok konum istegi dakikalik kotayi hizla tuketiyor ve 2-4 sn'lik
# geri cekilme yetmiyor. Kota dakika bazli oldugu icin bir dakikayi asmali.
RATE_LIMIT_BACKOFF = (65, 130, 300)

#: 429 gövdesinde limitin HANGI pencereye ait oldugu yazar. Open-Meteo uc
#: ayri pencere isletir ve mesajlari birbirinden farklidir::
#:
#:     "Minutely API request limit exceeded..."
#:     "Hourly API request limit exceeded. Please try again in the next hour."
#:     "Daily API request limit exceeded..."
#:
#: OLCULDU 2026-08-20 21:41: 96 ilcelik saatlik cekimde SAATLIK limit doldu.
#: Merdiven (65/130/300 sn) dakikalik limit icin kalibre edilmisti; uc deneme
#: toplam ~8 dakika eder. Sunucu ise "bir sonraki saat" diyordu -- 19 dakika.
#: Sonuc: kalan 23 ilcenin her biri sekiz dakika bosuna deneyip DUSUYORDU,
#: ve bunu yapmasi ~3 saat surecekti.
#:
#: Yarisma gunu bu somut bir kayiptir: veriyi tazelemek isteyen ekip,
#: "yeniden dene" demesi yeterliyken cekimin basarisiz oldugunu gorur.
#: Asagidaki yardimci, mesajdaki pencereyi okuyup SAAT SINIRINA kadar bekler.
SAATLIK_LIMIT_TAVANI_SN = 3900  # 65 dk: saat siniri + emniyet payi


def rate_limit_beklemesi(
    yanit: object,
    deneme: int,
    *,
    simdi: datetime | None = None,
    merdiven: Sequence[int] = RATE_LIMIT_BACKOFF,
) -> tuple[int, str]:
    """429 icin ne kadar beklenecegini ve NEDENINI doner.

    Oncelik sirasi:
      1. ``Retry-After`` basligi -- sunucunun kendi soyledigi sure.
      2. Govde metni SAATLIK/GUNLUK limit diyorsa: bir sonraki saat basina
         kadar bekle. Kisa merdiven burada ise yaramaz.
      3. Aksi halde dakikalik merdiven.

    Returns:
        ``(saniye, gerekce)``. Gerekce dogrudan kullaniciya yazdirilir --
        "neden 19 dakika bekliyoruz" sorusu cevapsiz kalmasin.
    """
    baslik = getattr(yanit, "headers", {}).get("Retry-After") if yanit is not None else None
    if baslik and str(baslik).isdigit():
        return int(baslik), "sunucunun Retry-After basligi"

    govde = ""
    try:
        govde = str(getattr(yanit, "text", "") or "").lower()
    except Exception:  # noqa: BLE001 - govde okunamazsa merdivene dus
        govde = ""

    if "hour" in govde or "dai" in govde:
        an = simdi or datetime.now()
        sonraki_saat = (an + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        saniye = int((sonraki_saat - an).total_seconds()) + 30  # emniyet payi
        saniye = max(60, min(saniye, SAATLIK_LIMIT_TAVANI_SN))
        return saniye, "SAATLIK/GUNLUK limit -- saat basina kadar"

    # ``merdiven`` parametrik: EPIAS kendi kotasi icin ayri kalibre edilmis
    # bir merdiven tutar (30/90/240). Saatlik-pencere tespiti ise ORTAKTIR --
    # her saglayicida ayni sessiz kayip bicimini uretir.
    return int(merdiven[min(deneme - 1, len(merdiven) - 1)]), "dakikalik merdiven"


# Istekler arasi varsayilan bekleme. Uzun aralik cekerken artir.
DEFAULT_PAUSE = 1.5

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


#: 96 ilcenin dogrulanmis koordinatlari burada durur (scripts/fetch_districts.py
#: uretir, sinir kutusu kontrolunden gecmistir).
REFERENCE_PATH = "data/reference/ilceler_gdz_adm.parquet"


def load_reference_locations(path: str = REFERENCE_PATH) -> dict[str, tuple[float, float]]:
    """96 ilcenin tamamini referans tablosundan okur.

    NEDEN ELLE LISTE YETMIYOR
    -------------------------
    Elle yazilmis 20 konum (5 il + 15 ilce), 96 ilcelik bir panel icin cok
    kabadir: Izmir'in 30 ilcesi 4 noktaya dusuyordu. Cesme'nin denizel
    iklimi ile Kiraz'in karasal iklimi ayni satira yaziliyordu -- oysa yaz
    tepe yuku ve firtina maruziyeti arasindaki fark tam da orada.

    Referanstan okumak ayrica listenin PANELDEN SAPMASINI da engeller:
    yeni bir ilce eklenirse hava verisi de otomatik olarak onu kapsar.
    """
    frame = validate_published_dataframe(
        path,
        required_columns=("il", "ilce", "lat", "lon"),
        min_rows=len(PROVINCE_COORDINATES),
    )
    eksik = frame[["lat", "lon"]].isna().any(axis=1)
    if eksik.any():
        raise ValueError(
            f"{int(eksik.sum())} ilcenin koordinati eksik. "
            "Once scripts/fetch_districts.py calistir."
        )
    # Anahtar olarak "Il-Ilce" kullaniyoruz: ayni ilce adi iki ilde olabilir
    # (or. Merkez). Duz ilce adi kullanmak sessizce satir kaybettirirdi.
    return {
        f"{satir.il}-{satir.ilce}": (float(satir.lat), float(satir.lon))
        for satir in frame.itertuples()
    }


#: Arsiv API'sinin gercek ucu ile bugun arasindaki GUVENLIK PAYI (gun).
#:
#: OLCULDU 2026-08-20, iki konumda (Izmir 38.42/27.14, Mugla 37.21/28.36):
#:
#:     bugun+0 (2026-08-20) -> HTTP 200, son dolu gun 2026-08-20, 0 bos
#:     bugun+1 (2026-08-21) -> HTTP 400
#:     bugun+2, +3, +7      -> HTTP 400
#:
#: Yani sinir BUGUNDUR; API bugune kadar bosluksuz veri sunar ve yalnizca
#: GELECEK bir tarih istenince 400 doner.
#:
#: Bu sabit onceden 6 idi ve OLCULMEMISTI -- "birkac gun geriden gelir"
#: varsayimiydi. Bedeli somuttu: her cekici bitisi 6 gun geriye kirpiyor,
#: tablolar birbirinden farkli gunlerde bitiyor ve panelin SON gunlerinde
#: -- yani bir yarismada test blogunun oturdugu yerde -- feature'lar NaN
#: kaliyordu. Egitimde dolu, testte bos bir feature EKSIK feature'dan daha
#: kotudur: model ona guvenmeyi ogrenir, sonra kaybeder.
#:
#: 1 gun pay BILEREK birakildi: kosu gece yarisina denk gelirse ya da
#: sunucunun gunu bizimkinden geride kalirsa 400 yemeyelim. 0 yapmak bu
#: yarim gunluk pencerede tum cekimi kirardi.
ARCHIVE_LAG_DAYS = 1


def checkpoint_covers(path: Path, start: str, end: str, *, column: str = "tarih") -> bool:
    """Kontrol noktasi istenen araligi KAPSIYOR mu?

    NEDEN BU FONKSIYON ORTAK: uc cekicide ayni hata vardi (olculdu
    2026-08-20) -- kontrol noktasi yalnizca VAR MI diye soruluyordu::

        if ckpt.is_file() and not args.fresh:
            continue          # <-- araligi kapsiyor mu? SORULMUYOR

    Sonucu sessiz bayatliktir: ``--end`` ileri tasinir, betik "kontrol
    noktasindan" yazip 96 ilcenin hepsini atlar, exit 0 doner ve tablo eski
    tarihte kalir. Hicbir hata mesaji yoktur; yalnizca panelin son gunleri
    bostur. Hava kalitesi tablosu tam olarak boyle 08-12'de takili kaldi.

    Bozuk/yarim inmis parquet okunamazsa "kapsamiyor" sayilir: sessizce
    eksik veri tasimaktansa yeniden indirmek ucuzdur.

    Args:
        column: Kapsam olculecek tarih kolonu -- gunluk tabloda ``tarih``,
            ham saatlik tabloda ``zaman``.
    """
    if not path.exists():
        return False
    try:
        span = pd.read_parquet(path, columns=[column])[column]
    except (OSError, ValueError, KeyError):
        return False
    if span.empty:
        return False
    return bool(
        pd.Timestamp(span.min()) <= pd.Timestamp(start)
        and pd.Timestamp(span.max()) >= pd.Timestamp(end)
    )


def cap_end_date(end: str, *, today: date | None = None) -> tuple[str, str | None]:
    """Bitis tarihini arsivin gercekten sundugu en son gune kirpar.

    NEDEN: arsiv API'si GELECEK bir ``end_date`` icin **HTTP 400** doner --
    hem de neyin yanlis oldugunu soylemeden. Olculdu: ``end=2026-09-01``
    (bugun 2026-08-15) 96 konumun 96'sinda da 400 verdi ve indirme tamamen
    basarisiz oldu.

    Yarisma gunu "bitisi yarismanin sonuna ayarlayayim" demek son derece
    dogal bir reflekstir -- ve tam orada patlardi. Sessizce kirpip
    NEDENINI SOYLUYORUZ.

    Returns:
        ``(kirpilmis_tarih, uyari_metni)``. Kirpma olmadiysa uyari ``None``.
    """
    bugun = today or date.today()
    en_son = bugun - timedelta(days=ARCHIVE_LAG_DAYS)
    istenen = date.fromisoformat(end)
    if istenen <= en_son:
        return end, None
    return en_son.isoformat(), (
        f"Bitis {end} arsivin otesinde (bugun {bugun}, arsiv ~{ARCHIVE_LAG_DAYS} "
        f"gun geriden gelir). {en_son} olarak kirpildi -- aksi halde API "
        "her konum icin HTTP 400 doner."
    )


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

            # 429 = hiz siniri. Bu, gecici bir ag hatasi DEGIL -- kisa bir geri
            # cekilme ise yaramaz. Olculen davranis: 2 sn ve 4 sn beklemek
            # yetmedi, ardisik 7 konum dustu. Sunucu Retry-After verirse ona
            # uy, vermezse dakika mertebesinde bekle.
            if response.status_code == 429:
                wait, gerekce = rate_limit_beklemesi(response, attempt)
                print(
                    f"  {name}: hiz siniri (429); {wait} sn bekleniyor "
                    f"({gerekce}) [deneme {attempt}/{retries}]"
                )
                time.sleep(wait)
                last_error = requests.HTTPError("429 Too Many Requests")
                continue

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
    # IL ve ILCE anahtarlarini AYRI kolon olarak da yaziyoruz.
    #
    # Konum adi artik "Il-Ilce" bicimindedir (96 ilcelik panelle birebir
    # eslesmek icin). Ama yarisma verisinin hangi granulariteyle gelecegini
    # BILMIYORUZ: il bazinda gelirse ilce anahtariyla join %0 eslesir.
    # OLCULDU: bu degisiklikten sonra full_pipeline'in hava join'i
    # "eslesme orani %0.0" verdi.
    #
    # Iki kolon birden yazmak, veri gunu hangi seviye gelirse gelsin
    # calisan tek cozumdur.
    _il, _, _ilce = name.partition("-")
    frame.insert(2, "il_key", join_key(_il))
    frame.insert(3, "ilce_key", join_key(_ilce) if _ilce else "")
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
    parser.add_argument(
        "--districts",
        action="store_true",
        help="Elle yazilmis 15 ilce merkezini de cek (eski, kaba)",
    )
    parser.add_argument(
        "--all-districts",
        action="store_true",
        help="96 ilcenin TAMAMINI referans tablosundan cek (ONERILEN)",
    )
    parser.add_argument(
        "--out", default="data/external", help="Cikti dizini (varsayilan: data/external)"
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=DEFAULT_PAUSE,
        help=f"Istekler arasi bekleme, sn (varsayilan: {DEFAULT_PAUSE})",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Mevcut dosyayi yok say ve her konumu bastan indir",
    )
    args = parser.parse_args()

    kirpilmis, uyari = cap_end_date(args.end)
    if uyari:
        print(f"UYARI: {uyari}")
        args.end = kirpilmis

    if args.all_districts:
        locations = load_reference_locations()
        print(f"Referans tablosundan {len(locations)} ilce yuklendi (panelle birebir).")
    else:
        locations = dict(PROVINCE_COORDINATES)
        if args.districts:
            locations.update(DISTRICT_COORDINATES)

    output_dir = Path(args.out)
    suffix = "saatlik" if args.hourly else "gunluk"
    output_path = output_dir / f"hava_{suffix}.parquet"

    # DEVAM ETME: onceki kosuda inen konumlari tekrar indirme. Hiz siniri
    # nedeniyle kismi kalan bir indirmeyi tamamlamak icin -- 20 konumun 13'u
    # inmisse yalnizca 7'sini istemek hem hizli hem kotaya nazik.
    frames: list[pd.DataFrame] = []
    already: set[str] = set()

    time_column = "tarih" if not args.hourly else "zaman"
    if output_path.exists() and not args.fresh:
        try:
            existing = validate_published_dataframe(
                output_path,
                required_columns=("konum", time_column),
                min_rows=1,
                source=ARCHIVE_URL,
            )
        except (OSError, ValueError) as error:
            print(f"Mevcut hava cache'i dogrulanamadi; kullanilmayacak: {error}")
            existing = None
        covered = set() if existing is None else set(existing["konum"].unique())
        # Yalnizca istenen tarih araligini KAPSAYAN konumlari kabul et.
        wanted_start, wanted_end = pd.Timestamp(args.start), pd.Timestamp(args.end)
        for name in covered:
            span = existing.loc[existing["konum"] == name, time_column]
            if span.min() <= wanted_start and span.max() >= wanted_end:
                already.add(name)
        if already:
            frames.append(existing[existing["konum"].isin(already)])
            print(f"Mevcut dosyada {len(already)} konum tam kapsamli -- atlaniyor.")

    pending = {name: coords for name, coords in locations.items() if name not in already}

    print(f"{len(pending)} konum indirilecek, {args.start} - {args.end}, ", end="")
    print("saatlik" if args.hourly else "gunluk")

    failures = []

    for index, (name, (latitude, longitude)) in enumerate(pending.items(), start=1):
        print(f"[{index}/{len(pending)}] {name} ...", end=" ", flush=True)
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
        time.sleep(args.pause)  # API'ye nazik ol

    if not frames:
        print("\nHicbir konum icin veri alinamadi. Internet baglantisini kontrol et.")
        return 1

    combined = pd.concat(frames, ignore_index=True)
    if not args.hourly:
        combined = add_derived_features(combined)

    output_dir.mkdir(parents=True, exist_ok=True)
    combined = combined.drop_duplicates(subset=["konum", "tarih" if not args.hourly else "zaman"])
    publish_dataframe(
        combined,
        output_path,
        required_columns=("konum", time_column),
        min_rows=max(1, combined["konum"].nunique()),
        source=ARCHIVE_URL,
    )

    print(f"\nYazildi: {output_path}")
    print(f"  {len(combined):,} satir x {combined.shape[1]} kolon")
    print(f"  Konumlar: {combined['konum'].nunique()}")
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
