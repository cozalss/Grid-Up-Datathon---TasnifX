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

from fetch_weather import ARCHIVE_URL, RATE_LIMIT_BACKOFF, cap_end_date  # noqa: E402

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

# Esikler (m/s). Literaturde surekli ruzgar hasari ~15 m/s uzerinde belirgin
# artar; 20 m/s "firtina" siniridir (Beaufort 8-9). Hamle icin 20 m/s,
# gunluk tablodaki firtina_gunu esigiyle (60 km/sa ~ 16.7 m/s) uyumlu ama
# saatlik cozunurlukte daha secici.
RUZGAR_ESIK_15_MS = 15.0
RUZGAR_ESIK_20_MS = 20.0
HAMLE_ESIK_20_MS = 20.0

#: Nihai tablonun kolon sozlesmesi -- testler birebir bunu dogrular.
FINAL_COLUMNS = [
    "ilce_key",
    "tarih",
    "basinc_min",
    "basinc_ort",
    "ruzgar_15ms_saat",
    "ruzgar_20ms_saat",
    "hamle_20ms_saat",
    "yon_std",
    "yon_degisim",
]

REFERENCE_PATH = Path("data/reference/ilceler_gdz_adm.parquet")
OUTPUT_PATH = Path("data/external/hava_saatlik_turev.parquet")

#: Kontrol noktasi dizini: her ilcenin GUNLUK agregati ayri dosyada durur.
#: Ham saatlik veriyi saklamiyoruz (96 x ~58k saat x 4 kolon parquet'te bile
#: yuzlerce MB olurdu); agregat ilce basina ~2400 satirdir. Yarim kalan bir
#: kosu, inen ilceleri tekrar indirmeden devam eder.
CHECKPOINT_DIR = Path("data/external/.hava_saatlik_ckpt")

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
                header = response.headers.get("Retry-After")
                wait = (
                    int(header)
                    if header and header.isdigit()
                    else RATE_LIMIT_BACKOFF[min(attempt - 1, len(RATE_LIMIT_BACKOFF) - 1)]
                )
                print(
                    f"  {name}: hiz siniri (429); {wait} sn bekleniyor [deneme {attempt}/{retries}]"
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


def aggregate_daily(hourly: pd.DataFrame, ilce_key: str) -> pd.DataFrame:
    """Bir ilcenin saatlik verisini gunluk turev tablosuna indirir.

    NaN saatler esik sayimlarina 0 olarak girer (bilinmeyen saat "esik
    asilmadi" sayilir); basinc/yon istatistikleri NaN'lari atlar, gunun
    tamami NaN ise sonuc NaN kalir ve NaN orani dogrulamasinda gorunur.
    """
    frame = hourly.copy()
    frame["tarih"] = frame["zaman"].dt.normalize()

    # Esik gostergeleri: bool karsilastirmada NaN -> False, yani sayilmaz.
    frame["r15"] = (frame["wind_speed_10m"] >= RUZGAR_ESIK_15_MS).astype("int8")
    frame["r20"] = (frame["wind_speed_10m"] >= RUZGAR_ESIK_20_MS).astype("int8")
    frame["h20"] = (frame["wind_gusts_10m"] >= HAMLE_ESIK_20_MS).astype("int8")

    grouped = frame.groupby("tarih")
    daily = grouped.agg(
        basinc_min=("surface_pressure", "min"),
        basinc_ort=("surface_pressure", "mean"),
        ruzgar_15ms_saat=("r15", "sum"),
        ruzgar_20ms_saat=("r20", "sum"),
        hamle_20ms_saat=("h20", "sum"),
    )
    daily["yon_std"] = grouped["wind_direction_10m"].apply(circular_std)

    # Gunun baskin yonu (dairesel ortalama) -> dunle mutlak dairesel fark.
    # 350 -> 10 derece donusu 20'dir, 340 DEGIL; fark [0, 180] araligindadir.
    # Ilk gunun dunu yok -> NaN kalir (bilerek; 0 yazmak "yon degismedi"
    # demek olurdu).
    yon_ort = grouped["wind_direction_10m"].apply(circular_mean)
    daily = daily.sort_index()
    yon_ort = yon_ort.sort_index()
    fark = (yon_ort - yon_ort.shift(1)).abs() % 360.0
    daily["yon_degisim"] = np.minimum(fark, 360.0 - fark)

    daily = daily.reset_index()
    daily.insert(0, "ilce_key", ilce_key)

    for column in ["basinc_min", "basinc_ort", "yon_std", "yon_degisim"]:
        daily[column] = daily[column].astype("float32")
    for column in ["ruzgar_15ms_saat", "ruzgar_20ms_saat", "hamle_20ms_saat"]:
        daily[column] = daily[column].astype("int8")  # 0..24 sigar

    return daily[FINAL_COLUMNS]


def checkpoint_covers(path: Path, start: str, end: str) -> bool:
    """Kontrol noktasi istenen araligi kapsiyor mu? Bozuk dosya = kapsamiyor."""
    if not path.exists():
        return False
    try:
        span = pd.read_parquet(path, columns=["tarih"])["tarih"]
    except (OSError, ValueError):
        return False
    if span.empty:
        return False
    return bool(span.min() <= pd.Timestamp(start) and span.max() >= pd.Timestamp(end))


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
    args = parser.parse_args()

    kirpilmis, uyari = cap_end_date(args.end)
    if uyari:
        print(f"UYARI: {uyari}")
        args.end = kirpilmis

    districts = load_reference_districts()
    print(f"Referans tablosundan {len(districts)} ilce yuklendi.")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    pending = []
    for ilce_key, lat, lon in districts:
        ckpt = CHECKPOINT_DIR / f"{ilce_key}.parquet"
        if not args.fresh and checkpoint_covers(ckpt, args.start, args.end):
            continue
        pending.append((ilce_key, lat, lon))

    done = len(districts) - len(pending)
    if done:
        print(f"Kontrol noktasinda {done} ilce tam kapsamli -- atlaniyor.")
    print(f"{len(pending)} ilce indirilecek, {args.start} - {args.end}, saatlik")

    failures: list[str] = []
    for index, (ilce_key, lat, lon) in enumerate(pending, start=1):
        print(f"[{index}/{len(pending)}] {ilce_key} ...", end=" ", flush=True)
        try:
            hourly = fetch_hourly(ilce_key, lat, lon, args.start, args.end)
        except RuntimeError as error:
            print(f"BASARISIZ -- {error}")
            failures.append(ilce_key)
            continue

        daily = aggregate_daily(hourly, ilce_key)
        atomic_write_dataframe(daily, CHECKPOINT_DIR / f"{ilce_key}.parquet")
        print(f"{len(hourly):,} saat -> {len(daily):,} gun")
        time.sleep(args.pause)  # API'ye nazik ol

    frames = []
    for ilce_key, _, _ in districts:
        ckpt = CHECKPOINT_DIR / f"{ilce_key}.parquet"
        if ckpt.exists():
            frames.append(pd.read_parquet(ckpt))

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
