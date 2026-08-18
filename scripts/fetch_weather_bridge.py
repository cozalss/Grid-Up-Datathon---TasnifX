"""Hava arsivinin bittigi yerden BUGUN+16'ya kadar kopru kurar (Open-Meteo forecast).

NEDEN BU BETIK
--------------
``fetch_weather.py`` ERA5 ARSIVINI ceker; arsiv bugunden ~6 gun geride biter
(olculdu: 2026-08-18'de son gun 2026-08-09). Yarisma test blogu bu tarihi
asarsa 17 hava kolonu YALNIZCA testte NaN olur -- CV'de gorunmeyen, sessiz
bir bozulma (2026-08-18 denetimi, P0-10). Open-Meteo'nun forecast API'si ayni
gunluk degiskenleri ``past_days`` (<=92) ile geriye ve ``forecast_days``
(<=16) ile ileriye verir. Bu betik:

  1. ``hava_gunluk.parquet``i okur, arsivin son gununu bulur;
  2. 96 ilce icin forecast API'den (arsiv_son - 2 gun) .. (bugun + 16 gun)
     araligini ceker (arsivle 2 gunluk ortusme = dikis kontrolu);
  3. yalnizca arsivde OLMAYAN gunleri ``hava_tahmin=1`` bayragiyla ekler,
     arsiv satirlari ``hava_tahmin=0``; ayni semayla yeniden yayimlar.

``hava_tahmin`` SAYISAL (int8) tutulur: mevcut feature kurucular tum sayisal
kolonlari aldigi icin metin bayragi LightGBM'i dusururdu; sayisal bayrak hem
zararsizdir hem de modele "bu satir tahmin verisi" bilgisini verir.

DIKIS KONTROLU: ortusen 2 gunde arsiv ve forecast sicaklik_ort farki ilce
basina ortalama >3 C ise betik REDDEDER (yanlis koordinat / birim degisikligi
belirtisi). Turetilmis kolonlar (derece-gun, esikler) fetch_weather ile AYNI
fonksiyondan uretilir.

Kullanim::

    python scripts/fetch_weather_bridge.py            # varsayilan: +16 gun
    python scripts/fetch_weather_bridge.py --dry-run  # yazmadan raporla
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

import fetch_weather as fw  # noqa: E402  (ayni degisken listesi ve turevler)

from gridup.io_utils import publish_dataframe  # noqa: E402
from gridup.turkish import join_key  # noqa: E402

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HAVA_YOLU = KOK / "data" / "external" / "hava_gunluk.parquet"
KAYNAK_KOLONU = "hava_tahmin"
MAX_PAST_DAYS = 92
MAX_FORECAST_DAYS = 16
ORTUSME_GUN = 2
DIKIS_ESIGI_C = 3.0
PAUSE_S = 0.4


def forecast_frame(
    name: str, latitude: float, longitude: float, *, past_days: int, forecast_days: int
) -> pd.DataFrame:
    """Tek ilce icin forecast API gunluk tablosu, fetch_weather semasiyla."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "past_days": past_days,
        "forecast_days": forecast_days,
        "daily": ",".join(fw.DAILY_VARIABLES),
        "timezone": "Europe/Istanbul",
    }
    son_hata: Exception | None = None
    for deneme in range(1, 4):
        try:
            yanit = requests.get(FORECAST_URL, params=params, timeout=60)
            if yanit.status_code == 429:
                bekle = fw.RATE_LIMIT_BACKOFF[min(deneme - 1, len(fw.RATE_LIMIT_BACKOFF) - 1)]
                print(f"  {name}: hiz siniri; {bekle} sn")
                time.sleep(bekle)
                son_hata = requests.HTTPError("429")
                continue
            yanit.raise_for_status()
            veri = yanit.json()
            break
        except (requests.RequestException, ValueError) as hata:
            son_hata = hata
            time.sleep(2**deneme)
    else:
        raise RuntimeError(f"{name}: forecast alinamadi. Son hata: {son_hata}")
    if "daily" not in veri:
        raise RuntimeError(f"{name}: yanitta 'daily' yok: {str(veri)[:200]}")

    frame = pd.DataFrame(veri["daily"]).rename(columns=fw.DAILY_RENAME)
    frame["tarih"] = pd.to_datetime(frame["tarih"])
    frame.insert(0, "konum", name)
    frame.insert(1, "konum_key", join_key(name))
    il, _, ilce = name.partition("-")
    frame.insert(2, "il_key", join_key(il))
    frame.insert(3, "ilce_key", join_key(ilce) if ilce else "")
    return frame


def dikis_kontrolu(arsiv: pd.DataFrame, tahmin: pd.DataFrame) -> float:
    """Ortusen gunlerde arsiv-forecast sicaklik farkinin ortalamasini (C) dondurur.

    Raises:
        ValueError: Ortusme yoksa veya fark ``DIKIS_ESIGI_C``yi asarsa.
    """
    ortak = arsiv.merge(tahmin, on=["konum_key", "tarih"], suffixes=("_arsiv", "_tahmin"))
    if ortak.empty:
        raise ValueError("Arsiv ile forecast arasinda ortusen gun yok; dikis dogrulanamadi.")
    fark = (ortak["sicaklik_ort_arsiv"] - ortak["sicaklik_ort_tahmin"]).abs().mean()
    if fark > DIKIS_ESIGI_C:
        raise ValueError(
            f"Dikis basarisiz: ortusen gunlerde ortalama sicaklik farki {fark:.2f} C > "
            f"{DIKIS_ESIGI_C} C. Koordinat/birim uyumsuzlugu olabilir; yazilmadi."
        )
    return float(fark)


def kopru_kur(arsiv: pd.DataFrame, tahmin: pd.DataFrame) -> pd.DataFrame:
    """Arsiv + (arsivde olmayan) forecast gunleri; ``hava_tahmin`` bayragiyla."""
    arsiv_son = arsiv["tarih"].max()
    yeni = tahmin[tahmin["tarih"] > arsiv_son].copy()
    # Forecast'in son gunu (16.) cogu zaman eksik gelir (olculdu: 96 satirda
    # sicaklik_ort NaN). Cekirdek degiskeni eksik gunler KIRPILIR, sessizce
    # NaN olarak yayimlanmaz; kalan eksik main()'de reddedilir.
    cekirdek = fw.DAILY_RENAME["temperature_2m_mean"]
    kirpilan = int(yeni[cekirdek].isna().sum())
    if kirpilan:
        print(f"  Forecast'in {kirpilan} satiri (son gun) eksik; kirpildi.")
        yeni = yeni[yeni[cekirdek].notna()]
    yeni = fw.add_derived_features(yeni)
    yeni[KAYNAK_KOLONU] = 1
    eski = arsiv.copy()
    if KAYNAK_KOLONU not in eski.columns:
        eski[KAYNAK_KOLONU] = 0
    # Onceki bir kopru kosusundan kalan tahmin satirlari, arsiv o gunleri
    # kapsadiysa artik arsivdir; kapsamadiysa yeni tahminle DEGISTIRILIR.
    eski = eski[(eski[KAYNAK_KOLONU] == 0) | (eski["tarih"] <= arsiv_son)]
    birlesik = pd.concat([eski, yeni[eski.columns]], ignore_index=True)
    birlesik[KAYNAK_KOLONU] = birlesik[KAYNAK_KOLONU].astype("int8")
    birlesik = birlesik.drop_duplicates(subset=["konum", "tarih"], keep="first")
    return birlesik.sort_values(["konum", "tarih"]).reset_index(drop=True)


def referans_konumlar(yol: Path) -> dict[str, tuple[float, float]]:
    """96 ilcenin "Il-Ilce" -> (lat, lon) sozlugu (fetch_weather ile ayni anahtar).

    ``fw.load_reference_locations`` yayin metadata'si ister; referans parquet
    metadata sisteminden ONCE uretildigi icin sidecar'i yok (olculdu). Burada
    dogrudan okunur; koordinat eksigi yine reddedilir.
    """
    ref = pd.read_parquet(yol)
    eksik = ref[["lat", "lon"]].isna().any(axis=1)
    if eksik.any():
        raise ValueError(f"{int(eksik.sum())} ilcenin koordinati eksik: scripts/fetch_districts.py")
    return {f"{r.il}-{r.ilce}": (float(r.lat), float(r.lon)) for r in ref.itertuples()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-days", type=int, default=MAX_FORECAST_DAYS)
    parser.add_argument("--dry-run", action="store_true", help="Yazmadan raporla")
    parser.add_argument("--pause", type=float, default=PAUSE_S)
    args = parser.parse_args()

    arsiv = pd.read_parquet(HAVA_YOLU)
    arsiv_gercek = arsiv[arsiv[KAYNAK_KOLONU] == 0] if KAYNAK_KOLONU in arsiv.columns else arsiv
    arsiv_son = pd.Timestamp(arsiv_gercek["tarih"].max()).date()
    bugun = date.today()
    past_days = min(MAX_PAST_DAYS, max(ORTUSME_GUN, (bugun - arsiv_son).days + ORTUSME_GUN))
    print(f"Arsiv son gunu {arsiv_son}, bugun {bugun}; past_days={past_days}, "
          f"forecast_days={args.forecast_days}")  # fmt: skip

    konumlar = referans_konumlar(KOK / fw.REFERENCE_PATH)
    parcalar: list[pd.DataFrame] = []
    for sira, (ad, (lat, lon)) in enumerate(sorted(konumlar.items()), start=1):
        parcalar.append(
            forecast_frame(ad, lat, lon, past_days=past_days, forecast_days=args.forecast_days)
        )
        if sira % 16 == 0:
            print(f"  {sira}/{len(konumlar)} konum")
        time.sleep(args.pause)
    tahmin = pd.concat(parcalar, ignore_index=True)

    fark = dikis_kontrolu(arsiv_gercek, tahmin)
    birlesik = kopru_kur(arsiv_gercek, tahmin)
    yeni_gun = birlesik.loc[birlesik[KAYNAK_KOLONU] == 1, "tarih"]
    print(f"Dikis: ortusen gunlerde ort. sicaklik farki {fark:.2f} C (esik {DIKIS_ESIGI_C})")
    print(f"Kopru: {int((birlesik[KAYNAK_KOLONU] == 1).sum())} tahmin satiri, "
          f"{yeni_gun.min().date()} .. {yeni_gun.max().date()}")  # fmt: skip
    eksik = int(birlesik[fw.DAILY_RENAME["temperature_2m_mean"]].isna().sum())
    if eksik:
        raise ValueError(f"Kopru sonrasi {eksik} satirda sicaklik_ort NaN; yazilmadi.")
    if args.dry_run:
        print("--dry-run: yazilmadi.")
        return 0
    publish_dataframe(
        birlesik,
        HAVA_YOLU,
        required_columns=("konum", "tarih", "sicaklik_ort", KAYNAK_KOLONU),
        min_rows=len(arsiv_gercek),
        source=f"{fw.ARCHIVE_URL} + {FORECAST_URL}",
    )
    print(
        f"Yazildi: {HAVA_YOLU}  {len(birlesik):,} satir, son gun {birlesik['tarih'].max().date()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
