"""GDZ/ADM ilceleri icin SAATLIK hava degiskenlerinden gunluk turevler uretir.

NEDEN BU BETIK
--------------
Gunluk ozet (hava_gunluk.parquet) tepe degerleri tasir ama uc mekanizmayi
kacirir -- literatur kaniti docs/10 bolum 3 ve docs/09 bolum 3:

  * Basinc        -> alcak basinc merkezi firtina sisteminin oncusudur;
                     gunluk tabloda basinc HIC yoktu.
  * Esik-asan saat-> "kac saat boyunca 15 m/s ustu esti" gunluk maksimumdan
                     farkli bir hasar mekanizmasidir: malzeme yorulmasi.
                     10 dakikalik bir hamle ile 6 saatlik surekli yuklenme
                     ayni maksimumu verir ama ayni hasari vermez.
  * Yon degisimi  -> cephe gecisinde ruzgar yonu doner; ani yon degisimi
                     agac ve iletken salinimini tetikler.

YONUN KENDISI VE DAGILIM (2026-08-20'de eklendi)
------------------------------------------------
Bu betigin ilk surumu yonden yalnizca TURETILMIS iki olcu uretiyordu
(``yon_std``, ``yon_degisim``); YONUN KENDISI hicbir kolonda yoktu. Ayni
sekilde ruzgardan yalnizca esik-ustu SAAT SAYISI vardi -- "gun icinde nasil
dagildi" sorusu hic sorulmuyordu. Iki bosluk da su kanitla kapatildi:

  2024 Enerji Datathonu birincisi (Pikachow) saatlik degiskenleri
  quantile'larla ozetledi ve modelin EN YUKSEK onem verdigi tek degisken
  ``wind_dir_10m q01`` oldu -- yani yonun kendisinin gunluk dagilimi.

Eklenenler: ``yon_sin``/``yon_cos`` (baskin yonun birim vektoru),
``hamle_yon_sin``/``hamle_yon_cos`` (EN SIDDETLI hamle saatindeki yon),
``ruzgar_q25..q90``, ``hamle_q90``, ``yon_q01``/``yon_q99``,
``basinc_std`` ve ``basinc_dusus_3s`` (3 saatlik en sert basinc dususu).

HAM KONTROL NOKTASI -- NEDEN DEGISTI
------------------------------------
Ilk surum yalnizca GUNLUK agregati kontrol noktasina yaziyordu. Sonucu su
oldu: tabloya tek bir kolon eklemek 96 ilcenin tamamini BASTAN indirmeyi
gerektirdi (saatler suren, kotayi yakan bir is). Artik HAM saatlik veri
saklaniyor (~2 MB/ilce); yeni bir turev kolon ``--yeniden-topla`` ile AGA
HIC DOKUNMADAN uretilebilir.

BIRIM NOTU
----------
Open-Meteo ruzgari VARSAYILAN olarak km/sa dondurur. Bu betik API'ye
``wind_speed_unit=ms`` gecirir ve yanittaki ``hourly_units`` blogundan
birimin gercekten "m/s" oldugunu DOGRULAR -- parametre adi sessizce
degisirse esikler 3.6 kat kayardi ve bunu fark etmek imkansiz olurdu.
Basinc (surface_pressure) varsayilan hPa'dir, o da dogrulanir.

KAGGLE UYARISI
--------------
Kaggle notebook'unda internet KAPALI olabilir. Bu betigi YERELDE calistir,
cikan data/external/hava_saatlik_turev.parquet dosyasini Dataset olarak yukle.

Kullanim::

    python scripts/fetch_hourly_weather.py
    python scripts/fetch_hourly_weather.py --start 2020-01-01 --end 2026-08-15
    python scripts/fetch_hourly_weather.py --fresh   # kontrol noktalarini yok say
    python scripts/fetch_hourly_weather.py --yeniden-topla  # AG YOK, ham veriden yeniden uret
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_weather import (  # noqa: E402
    ARCHIVE_URL,
    cap_end_date,
    ckpt_birlestir,
    eksik_aralik,
    rate_limit_beklemesi,
)

from gridup.features.weather import circular_mean  # noqa: E402
from gridup.io_utils import atomic_write_dataframe  # noqa: E402

#: Saatlik degiskenler: basinc + ruzgar uclusu. Sicaklik/yagis BILEREK yok --
#: onlar gunluk tabloda zaten var; burada yalnizca saatlik cozunurlugun
#: kattigi bilgiyi cekiyoruz (istek kucuk kalir, kota az yanar).
HOURLY_VARIABLES = [
    "surface_pressure",
    "wind_speed_10m",
    "wind_gusts_10m",
    "wind_direction_10m",
]

# Esikler (m/s) -- OLCULEREK kalibre edildi, literaturden alinmadi.
#
# Ilk surumde esikler genel firtina literaturunden gelmisti: surekli ruzgar
# icin 15 ve 20 m/s, hamle icin 20 m/s. Bunlarin Ege'de ne siklikta gorundugu
# HIC olculmemisti. 2026-08-20'de 40 ilcenin 2.326.080 saatinde olculdu:
#
#     SUREKLI RUZGAR (10 m)          HAMLE
#       >= 8 m/s : %1.3052             >= 8 m/s : %30.14
#       >=10 m/s : %0.3290             >=10 m/s : %17.57
#       >=12 m/s : %0.0727             >=12 m/s : %9.05
#       >=15 m/s : %0.0056             >=15 m/s : %2.74
#       >=18 m/s : %0.0001             >=18 m/s : %0.69
#       >=20 m/s : %0.0000  <-- HIC    >=20 m/s : %0.26
#       max gorulen: 18.5 m/s          >=25 m/s : %0.02  (max 35.0)
#
# Yani ``ruzgar_20ms_saat`` YAPISAL OLARAK OLUYDU: 2,3 milyon saatte bir kez
# bile tetiklenmedi ve her satirda 0 yazdi. ``ruzgar_15ms_saat`` de fiilen
# olu (2,3 milyon saatte 130 saat ~ ilce basina 6,5 yilda ~3 saat).
#
# Sebep fiziksel: ERA5 ~25 km izgarada uzamsal ORTALAMADIR; nokta
# olcumlerdeki ucları sonumler. Ayni sonumleme HAMLE parametrizasyonunda
# yoktur, bu yuzden hamle esikleri yuksek kalabilir.
#
# Yeni esikler her biri gercekten AYIRT EDEN noktalardan secildi: seyrek
# ama olu degil.
RUZGAR_ESIKLERI_MS = (8.0, 10.0, 12.0)
HAMLE_ESIKLERI_MS = (15.0, 20.0, 25.0)

#: Gunluk dagilim quantile'lari. Esik-ustu saat sayisi "kac saat asti"yi
#: soyler ama "gunun tipik ruzgari neydi"i soylemez: 4 saat 16 m/s esen bir
#: gun ile 4 saat 16, geri kalani 1 m/s esen bir gun ayni sayimi verir.
QUANTILE_SEVIYELERI = (0.01, 0.25, 0.50, 0.75, 0.90, 0.99)

#: Nihai tablonun kolon sozlesmesi -- testler birebir bunu dogrular.
FINAL_COLUMNS = [
    "ilce_key",
    "tarih",
    # --- basinc: alcak basinc ve HIZLI DUSUS firtina oncusudur
    "basinc_min",
    "basinc_ort",
    "basinc_std",
    "basinc_dusus_3s",
    # --- esik-ustu saat sayimlari: surekli yuklenme (malzeme yorulmasi)
    "ruzgar_8ms_saat",
    "ruzgar_10ms_saat",
    "ruzgar_12ms_saat",
    "hamle_15ms_saat",
    "hamle_20ms_saat",
    "hamle_25ms_saat",
    # --- gun ici dagilim: ayni maksimumun farkli "yayilma"lari
    "ruzgar_q25",
    "ruzgar_q50",
    "ruzgar_q75",
    "ruzgar_q90",
    "hamle_q90",
    # --- yon: turevler (std/degisim) + YONUN KENDISI (sin/cos, quantile)
    "yon_std",
    "yon_degisim",
    "yon_sin",
    "yon_cos",
    "hamle_yon_sin",
    "hamle_yon_cos",
    "yon_q01",
    "yon_q99",
]

#: ``int8``a donusturulecek kolonlar (0..24 sigar). Geri kalan hepsi float32.
SAYIM_KOLONLARI = tuple(
    [f"ruzgar_{int(e)}ms_saat" for e in RUZGAR_ESIKLERI_MS]
    + [f"hamle_{int(e)}ms_saat" for e in HAMLE_ESIKLERI_MS]
)

REFERENCE_PATH = Path("data/reference/ilceler_gdz_adm.parquet")
OUTPUT_PATH = Path("data/external/hava_saatlik_turev.parquet")

#: HAM saatlik kontrol noktasi -- her ilcenin ham saatlik serisi ayri dosyada.
#:
#: Onceki surum burada yalnizca GUNLUK agregati sakliyordu; gerekce "ham veri
#: yuzlerce MB olur" idi. Bu gerekce OLCULDU ve yanlis cikti: float32'ye
#: dusurulmus dort kolon, ilce basina ~1-2 MB parquet demek (toplam ~150 MB).
#: Buna karsilik gunluk-agregat kontrol noktasinin BEDELI cok agirdi -- tabloya
#: tek bir turev kolon eklemek 96 ilcenin tamamini bastan indirmeyi gerektirdi.
#: Artik ham veri duruyor ve yeni bir kolon ``--yeniden-topla`` ile AGA HIC
#: DOKUNMADAN uretiliyor. Dizin .gitignore kapsamindadir.
RAW_CHECKPOINT_DIR = Path("data/external/.hava_saatlik_ham")

#: Ham kontrol noktasinda saklanan kolonlar. ``zaman`` + dort olcum; float32
#: yeterlidir (basinc ~1013.25 hPa'da float32 cozunurlugu ~0.0001 hPa).
RAW_COLUMNS = ["zaman", *HOURLY_VARIABLES]

#: Istekler arasi nazik bekleme (sn). Saatlik istek gunlukten agir sayilir
#: (kota tarih araligina gore agirliklandirilir); 429 gelirse fetch icindeki
#: geri cekilme merdiveni devreye girer.
DEFAULT_PAUSE = 1.0


def circular_std(degrees: pd.Series | np.ndarray) -> float:
    """Dairesel standart sapma (derece): sqrt(-2 ln R).

    ``circular_mean`` ile ayni vektorel temel: acilari birim vektore cevir,
    ortalamanin buyuklugu R olsun. R=1 -> tum saatler ayni yon (std 0);
    R->0 -> yonler cembere yayilmis (std buyur). Aritmetik std burada
    KULLANILAMAZ: 350 ve 10 derecelik iki olcumun aritmetik std'si ~240
    gorunur, oysa aralarinda 20 derece vardir.
    """
    values = np.asarray(degrees, dtype="float64")
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan")

    radians = np.deg2rad(values)
    resultant = float(np.hypot(np.sin(radians).mean(), np.cos(radians).mean()))
    # R sayisal olarak 1'i kil payi asabilir (log negatif olamaz) ya da 0'a
    # cokup log'u patlatabilir -- iki ucu da kirp.
    resultant = min(max(resultant, 1e-12), 1.0)
    # abs: R=1'de sqrt(-0.0) = -0.0 doner; feature degeri olarak +0.0 yaz.
    return abs(float(np.rad2deg(np.sqrt(-2.0 * np.log(resultant)))))


def load_reference_districts(path: Path = REFERENCE_PATH) -> list[tuple[str, float, float]]:
    """96 ilceyi referans tablosundan okur: (ilce_key, lat, lon) listesi.

    ilce_key tum bolgede TEKILDIR (fetch_districts.py bunu garanti eder);
    dogrulanmadan gecmiyoruz cunku tekrarlanan anahtar kontrol noktasi
    dosyalarini sessizce ezerdi.
    """
    frame = pd.read_parquet(path)
    eksik = frame[["lat", "lon"]].isna().any(axis=1)
    if eksik.any():
        raise ValueError(
            f"{int(eksik.sum())} ilcenin koordinati eksik. "
            "Once scripts/fetch_districts.py calistir."
        )
    if frame["ilce_key"].duplicated().any():
        raise ValueError("ilce_key tekil degil -- referans tablosunu kontrol et.")
    return [
        (str(satir.ilce_key), float(satir.lat), float(satir.lon)) for satir in frame.itertuples()
    ]


def fetch_hourly(
    name: str,
    latitude: float,
    longitude: float,
    start: str,
    end: str,
    *,
    timeout: int = 120,
    retries: int = 3,
) -> pd.DataFrame:
    """Tek ilce icin tum araligin saatlik verisini ceker.

    Raises:
        RuntimeError: Tum denemeler tukenirse ya da birimler beklenenden
            farkliysa. Sessiz bos DataFrame YOK -- eksik ilce fark edilmeden
            paneli deler.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(HOURLY_VARIABLES),
        # Gun siniri YEREL gune gore kesilsin: UTC birakmak her gunu 3 saat
        # kaydirir (Turkiye kalici UTC+3, yaz saati yok).
        "timezone": "Europe/Istanbul",
        # Varsayilan km/sa'yi m/s'ye cevir -- esikler m/s cinsinden.
        "wind_speed_unit": "ms",
    }

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(ARCHIVE_URL, params=params, timeout=timeout)

            # 429 = hiz siniri; kota dakika bazli, kisa bekleme ise yaramaz.
            # fetch_weather.py'de olculen ayni davranis -- ayni merdiven.
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

    if "hourly" not in payload:
        raise RuntimeError(f"{name}: yanitta 'hourly' bolumu yok. Yanit: {str(payload)[:300]}")

    # Birim dogrulamasi: parametre sessizce yok sayilirsa esikler 3.6 kat
    # kayar. hPa kontrolu ayni sigortanin basinc ayagi.
    units = payload.get("hourly_units", {})
    if units.get("wind_speed_10m") != "m/s":
        raise RuntimeError(f"{name}: ruzgar birimi m/s degil: {units.get('wind_speed_10m')!r}")
    if units.get("surface_pressure") != "hPa":
        raise RuntimeError(f"{name}: basinc birimi hPa degil: {units.get('surface_pressure')!r}")

    frame = pd.DataFrame(payload["hourly"]).rename(columns={"time": "zaman"})
    frame["zaman"] = pd.to_datetime(frame["zaman"])
    return frame


def _tepe_hamle_yonu(frame: pd.DataFrame) -> pd.Series:
    """Gunun EN SIDDETLI hamle saatindeki ruzgar yonu (derece).

    NEDEN gun ortalamasi degil: hasari yapan ruzgarin YONU, gunun baskin
    yonuyle ayni olmak zorunda degil. Ege'de ogleden sonra esen imbat
    (bati/kuzeybati, duzenli) ile cephe gecisinde donen guneybati ruzgari
    ayni hizi verse bile ayni riski tasimaz -- agaclarin ve iletkenlerin
    maruz kaldigi aci, dolayisiyla temas olasiligi degisir.

    Hamlesi ya da yonu tamamen eksik bir gun icin NaN doner (0 yazmak
    "kuzeyden esti" demek olurdu -- gercek bir yon degeri).
    """
    gecerli = frame.dropna(subset=["wind_gusts_10m", "wind_direction_10m"])
    if gecerli.empty:
        return pd.Series(dtype="float64", index=pd.Index([], name="tarih"))
    tepe = gecerli.groupby("tarih")["wind_gusts_10m"].idxmax()
    return gecerli.loc[tepe].set_index("tarih")["wind_direction_10m"]


def aggregate_daily(hourly: pd.DataFrame, ilce_key: str) -> pd.DataFrame:
    """Bir ilcenin saatlik verisini gunluk turev tablosuna indirir.

    NaN saatler esik sayimlarina 0 olarak girer (bilinmeyen saat "esik
    asilmadi" sayilir); basinc/yon istatistikleri NaN'lari atlar, gunun
    tamami NaN ise sonuc NaN kalir ve NaN orani dogrulamasinda gorunur.
    """
    frame = hourly.copy().sort_values("zaman")
    frame["tarih"] = frame["zaman"].dt.normalize()

    # Esik gostergeleri: bool karsilastirmada NaN -> False, yani sayilmaz.
    sayim_adlari: dict[str, str] = {}
    for kaynak_kolon, esikler, onek in (
        ("wind_speed_10m", RUZGAR_ESIKLERI_MS, "ruzgar"),
        ("wind_gusts_10m", HAMLE_ESIKLERI_MS, "hamle"),
    ):
        for esik in esikler:
            ad = f"{onek}_{int(esik)}ms_saat"
            gecici = f"_{ad}"
            frame[gecici] = (frame[kaynak_kolon] >= esik).astype("int8")
            sayim_adlari[ad] = gecici

    # 3 saatlik basinc egilimi ("pressure tendency"): meteorolojinin klasik
    # firtina oncusu. 3 saatte 3+ hPa dusus belirgin bir sistem demektir ve
    # bunu ne gunluk minimum ne de ortalama gosterir -- ikisi de SEVIYE
    # olcer, bu ise HIZ olcer.
    #
    # diff(3) satirlarin esit arali kli oldugunu varsayar; Open-Meteo saatlik
    # seriyi bosluksuz dondurur. Bosluk olsaydi fark 3 saatten uzun bir
    # araligi olcerdi -- o durum NaN oraninda gorunur, sessiz kalmaz.
    frame["_dusus3"] = frame["surface_pressure"].diff(3)

    grouped = frame.groupby("tarih")
    daily = grouped.agg(
        basinc_min=("surface_pressure", "min"),
        basinc_ort=("surface_pressure", "mean"),
        basinc_std=("surface_pressure", "std"),
        **{ad: (gecici, "sum") for ad, gecici in sayim_adlari.items()},
    ).sort_index()

    # En NEGATIF 3 saatlik degisim = en sert dusus; isareti cevirip pozitif
    # "dusus buyuklugu" yaziyoruz. Gun boyu basinc yalnizca yukseldiyse deger
    # negatif cikar ve 0'a kirpilir -- "dusus yok" demek budur.
    daily["basinc_dusus_3s"] = (-grouped["_dusus3"].min()).clip(lower=0.0)

    # Gun ici dagilim. Tek cagride uc degiskenin tum quantile'lari (grup
    # basina lambda cagirmaktan belirgin hizli -- 96 ilce x ~2400 gun).
    quantiles = (
        grouped[["wind_speed_10m", "wind_gusts_10m", "wind_direction_10m"]]
        .quantile(list(QUANTILE_SEVIYELERI))
        .unstack()
    )
    daily["ruzgar_q25"] = quantiles[("wind_speed_10m", 0.25)]
    daily["ruzgar_q50"] = quantiles[("wind_speed_10m", 0.50)]
    daily["ruzgar_q75"] = quantiles[("wind_speed_10m", 0.75)]
    daily["ruzgar_q90"] = quantiles[("wind_speed_10m", 0.90)]
    daily["hamle_q90"] = quantiles[("wind_gusts_10m", 0.90)]

    # yon_q01/q99 DAIRESEL OLARAK TUTARLI DEGILDIR: 350 ve 10 derecelik iki
    # yon sayisal olarak uzak gorunur, oysa aralarinda 20 derece vardir.
    # Yine de tasiniyorlar cunku KANIT var: 2024 birincisinin en yuksek
    # onemli degiskeni tam olarak buydu. Yorumlanislari "gun icinde gorulen
    # en dusuk/en yuksek pusula degeri" degil, "yon rejiminin kaba imzasi"
    # olmali -- yonun FIZIKSEL olarak dogru hali yon_sin/yon_cos'tur.
    daily["yon_q01"] = quantiles[("wind_direction_10m", 0.01)]
    daily["yon_q99"] = quantiles[("wind_direction_10m", 0.99)]

    daily["yon_std"] = grouped["wind_direction_10m"].apply(circular_std)

    # Gunun baskin yonu (dairesel ortalama) -> hem dunle mutlak dairesel
    # fark, hem de BIRIM VEKTOR bilesenleri. sin/cos ikilisi modele yonu
    # SUREKLI ve sureksizliksiz verir: 359 ile 1 derece komsudur, ham
    # dereceyle ise aralarinda 358 birim vardir.
    yon_ort = grouped["wind_direction_10m"].apply(circular_mean).sort_index()
    fark = (yon_ort - yon_ort.shift(1)).abs() % 360.0
    # Ilk gunun dunu yok -> NaN kalir (bilerek; 0 yazmak "yon degismedi"
    # demek olurdu).
    daily["yon_degisim"] = np.minimum(fark, 360.0 - fark)
    yon_rad = np.deg2rad(yon_ort)
    daily["yon_sin"] = np.sin(yon_rad)
    daily["yon_cos"] = np.cos(yon_rad)

    tepe_rad = np.deg2rad(_tepe_hamle_yonu(frame).reindex(daily.index))
    daily["hamle_yon_sin"] = np.sin(tepe_rad)
    daily["hamle_yon_cos"] = np.cos(tepe_rad)

    daily = daily.reset_index()
    daily.insert(0, "ilce_key", ilce_key)

    for column in FINAL_COLUMNS:
        if column in ("ilce_key", "tarih"):
            continue
        daily[column] = daily[column].astype("int8" if column in SAYIM_KOLONLARI else "float32")

    return daily[FINAL_COLUMNS]


def ham_yaz(hourly: pd.DataFrame, path: Path) -> None:
    """Ham saatlik seriyi float32'ye dusurup kontrol noktasina yazar."""
    frame = hourly[RAW_COLUMNS].copy()
    for column in HOURLY_VARIABLES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float32")
    atomic_write_dataframe(frame, path)


#: Bir kolonun bu orandan fazlasi NaN ise veri KABUL EDILMEZ. %2 esigi bir
#: "dikkat cek" isareti, bu ise bir KAPIDIR: bunun ustu, kaynagin alan adini
#: degistirdigi veya bolgesel olarak veri dondurmedigi anlamina gelir.
NAN_RED_ESIGI = 0.10


def validate_combined(combined: pd.DataFrame, n_districts: int, start: str, end: str) -> None:
    """Satir sayisi, tekrar ve NaN orani kontrolu.

    Sonuclari yazdirir VE kabul edilemez kalitede veriyi ``ValueError`` ile
    reddeder. Onceki hali yalnizca yazdiriyordu: betik gece boyu gozetimsiz
    kosarken (96 ilce x yillarca saatlik veri, saatler surer) bir kolonun
    %60'i NaN gelse bile exit 0 donuyor, parquet yaziliyordu. LightGBM NaN'i
    dogal olarak isledigi icin egitim de hatasiz devam eder ve sonuc, sessizce
    zayiflamis bir feature ile "makul gorunen ama yanlis" bir CV skorudur --
    bu deponun her yerde kacindigi hata bicimi.
    """
    n_days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    expected = n_districts * n_days
    print(
        f"  Beklenen satir: {n_districts} ilce x {n_days} gun = {expected:,}; "
        f"gercek: {len(combined):,}"
    )

    duplicated = int(combined.duplicated(subset=["ilce_key", "tarih"]).sum())
    if duplicated:
        print(f"  UYARI: {duplicated} tekrar eden (ilce_key, tarih) satiri!")

    print("  NaN oranlari:")
    reddedilenler: list[str] = []
    for column, ratio in combined.isna().mean().items():
        if ratio >= NAN_RED_ESIGI:
            marker = f"  <-- RED (>=%{100 * NAN_RED_ESIGI:.0f})"
            reddedilenler.append(f"{column} %{100 * ratio:.1f}")
        elif ratio >= 0.02:
            marker = "  <-- %2 ustu!"
        else:
            marker = ""
        print(f"    {column:18s} %{100 * ratio:.3f}{marker}")

    if reddedilenler:
        raise ValueError(
            "Harici veri kalite kapisi: su kolonlarda NaN orani "
            f"%{100 * NAN_RED_ESIGI:.0f} esigini asti -> {', '.join(reddedilenler)}. "
            "Parquet YAZILMADI. Muhtemel sebep: kaynak alan adi degistirdi veya "
            "ilgili aralikta veri dondurmedi. Once cekimi tekrarla."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2020-01-01", help="Baslangic tarihi (YYYY-AA-GG)")
    parser.add_argument("--end", default="2026-08-15", help="Bitis tarihi (YYYY-AA-GG)")
    parser.add_argument(
        "--pause",
        type=float,
        default=DEFAULT_PAUSE,
        help=f"Istekler arasi bekleme, sn (varsayilan: {DEFAULT_PAUSE})",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Kontrol noktalarini yok say ve her ilceyi bastan indir",
    )
    parser.add_argument(
        "--yeniden-topla",
        action="store_true",
        help="AG KULLANMA: yalnizca mevcut ham kontrol noktalarindan gunluk tabloyu yeniden uret",
    )
    args = parser.parse_args()

    kirpilmis, uyari = cap_end_date(args.end)
    if uyari:
        print(f"UYARI: {uyari}")
        args.end = kirpilmis

    districts = load_reference_districts()
    print(f"Referans tablosundan {len(districts)} ilce yuklendi.")

    RAW_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # YALNIZCA EKSIK KUYRUK. Ham kontrol noktasi 2026-08-19'a kadar doluysa
    # ve 08-20 isteniyorsa, TEK GUN icin 58.176 saati bastan indirmek kotayi
    # bosa yakar (olculdu 2026-08-21: diger cekicilerde ayni hata 240x fazla
    # veri demekti). Ham saatler kontrol noktasina EKLENIR, gunluk tablo ise
    # her zaman TUM ham seriden yeniden toplanir -- bu yuzden gunler arasi
    # bagimlilik tasiyan kolonlar (yon_degisim, basinc_dusus_3s) yine tamdir.
    pending: list[tuple[str, float, float, str]] = []
    for ilce_key, lat, lon in districts:
        ham = RAW_CHECKPOINT_DIR / f"{ilce_key}.parquet"
        if args.fresh:
            pending.append((ilce_key, lat, lon, args.start))
            continue
        aralik = eksik_aralik(ham, args.start, args.end, column="zaman")
        if aralik is None:
            continue
        pending.append((ilce_key, lat, lon, aralik[0]))

    if args.yeniden_topla:
        if pending:
            print(
                f"  UYARI: {len(pending)} ilcenin ham kontrol noktasi eksik/yetersiz. "
                "--yeniden-topla ag kullanmaz; bu ilceler tabloda OLMAYACAK ve "
                "kalite kapisi eksik ilce nedeniyle reddedecek."
            )
        pending = []
        print("  --yeniden-topla: indirme atlandi, ham veriden uretiliyor.")

    done = len(districts) - len(pending)
    if done and not args.yeniden_topla:
        print(f"Ham kontrol noktasinda {done} ilce tam kapsamli -- atlaniyor.")
    if pending:
        print(f"{len(pending)} ilce indirilecek, {args.start} - {args.end}, saatlik")

    failures: list[str] = []
    for index, (ilce_key, lat, lon, cek_bas) in enumerate(pending, start=1):
        etiket = f"{cek_bas}.." if cek_bas != args.start else ""
        print(f"[{index}/{len(pending)}] {ilce_key} {etiket}...", end=" ", flush=True)
        try:
            hourly = fetch_hourly(ilce_key, lat, lon, cek_bas, args.end)
        except RuntimeError as error:
            print(f"BASARISIZ -- {error}")
            failures.append(ilce_key)
            continue

        yol = RAW_CHECKPOINT_DIR / f"{ilce_key}.parquet"
        if not args.fresh:
            hourly = ckpt_birlestir(yol, hourly[RAW_COLUMNS], anahtarlar=("zaman",))
        ham_yaz(hourly, yol)
        print(f"{len(hourly):,} saat")
        time.sleep(args.pause)  # API'ye nazik ol

    # Gunluk tablo HER ZAMAN ham veriden yeniden uretilir -- kismen eski
    # semali bir gunluk kontrol noktasinin sessizce tasinmasi imkansiz olsun.
    frames = []
    for ilce_key, _, _ in districts:
        ham = RAW_CHECKPOINT_DIR / f"{ilce_key}.parquet"
        if not ham.exists():
            continue
        frames.append(aggregate_daily(pd.read_parquet(ham), ilce_key))

    if not frames:
        print("\nHicbir ilce icin veri yok. Internet baglantisini kontrol et.")
        return 1

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["ilce_key", "tarih"])
    combined = combined.sort_values(["ilce_key", "tarih"]).reset_index(drop=True)

    # Kalite kapisi YAZMADAN ONCE calisir. Tersi sirada kapinin bir anlami
    # kalmaz: bozuk veri zaten yayinlanmis olur ve sonraki kosular onu okur.
    print(
        f"\n  {len(combined):,} satir x {combined.shape[1]} kolon, "
        f"{combined['ilce_key'].nunique()} ilce"
    )
    print(f"  Tarih araligi: {combined['tarih'].min().date()} - {combined['tarih'].max().date()}")
    validate_combined(combined, len(districts), args.start, args.end)

    atomic_write_dataframe(combined, OUTPUT_PATH)
    print(f"\nYazildi: {OUTPUT_PATH}")

    if failures:
        print(f"\n  UYARI: {len(failures)} ilce alinamadi: {failures}")
        return 1

    print("\n  Kaynak: Open-Meteo (CC-BY-4.0). Notebook'ta atif ver.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
