"""HAVA KALITESI: PM10, PM2.5 ve toz -- izolator kirlenmesinin EKSIK YARISI.

NEDEN BU BETIK
--------------
Izolator ATLAMASI (flashover) tek basina nemle olmaz; **kirlilik birikimi +
nem** ciftiyle olur. Kuru kirli izolator atlamaz, ISLANAN atlar. Dagitim
sebekesinde "sebepsiz gorunen" kesintilerin bilinen bir kismi budur.

Bu ciftin bir yarisi (nem, sis, ciy noktasi) ``nem_toprak_gunluk`` ile
elimize gecti. Diger yarisi -- KIRLILIK -- hicbir kaynagimizda yoktu.

Ege icin ayrica SAHRA TOZU tasinimlari onemli: yilda birkac kez PM10 ve dust
degerleri katlanir, ve bu olaylar yagmurla birlikte gelmedigi icin klasik
hava degiskenleriyle yakalanmaz.

NEDEN Open-Meteo Air Quality (baska bir listeden degil)
------------------------------------------------------
2026-08-18'de public-apis listesi tarandi. Hava/cevre bolumundeki kaynaklarin
neredeyse tamami TICARI ve anahtar istiyor (Weatherstack, IQAir, BreezoMeter,
Climatiq); Turkiye kapsami yok. OpenAQ ucretsiz ama anahtar istiyor ve
Turkiye istasyonlari seyrek. "National Grid ESO" gercek acik veri ama BUYUK
BRITANYA sebekesi.

Open-Meteo Air Quality ise: anahtarsiz, CC-BY-4.0, ve zaten kullandigimiz
saglayici. Ayrica AYRI KOTASI var -- arsiv API'si tukendiginde bile calisir
(olculdu 2026-08-18).

KAPSAM -- OLCULDU
-----------------
    2020-01-15  DOLU (%0 NaN)      2023-01-15  DOLU
    2021-01-15  DOLU               2026-08-10  DOLU
    2022-01-15  DOLU

Panelin tamamini kapsiyor. CAPE'ten (2021-05 baslangicli) daha iyi.

KULLANIM
--------
::

    python scripts/fetch_hava_kalitesi.py
    python scripts/fetch_hava_kalitesi.py --start 2020-01-01 --end 2026-08-15

Cikti: ``data/external/hava_kalitesi_gunluk.parquet`` (ilce_key x tarih).
Kaynak: Open-Meteo Air Quality API, CC-BY-4.0. Anahtar GEREKMEZ.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_weather import (  # noqa: E402
    cap_end_date,
    ckpt_birlestir,
    eksik_aralik,
    rate_limit_beklemesi,
)

from gridup.io_utils import atomic_write_dataframe  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REFERANS = ROOT / "data" / "reference" / "ilceler_gdz_adm.parquet"
CIKTI = ROOT / "data" / "external" / "hava_kalitesi_gunluk.parquet"
CKPT_DIR = ROOT / "data" / "external" / ".hava_kalitesi_ckpt"

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

#: ``aerosol_optical_depth`` BILEREK YOK. Cekildi ve olculdu (2026-08-18):
#:   2020 %100 NaN · 2021 %100 · 2022 %58.9 · 2023+ %0
#: Yani urun ~2022 ortasinda basliyor ve panelin ilk UC yilini bos birakiyor.
#: Diger uc degisken (pm10, pm2_5, dust) tum aralikta %0 NaN. AOD ayrica
#: PM10/toz ile buyuk olcude ayni seyi olcer (aerosol yuku), yani kapsami
#: bozuk bir kolonu tasimanin karsiligi yok. Kalite kapisi bunu %39 NaN ile
#: zaten reddetmisti -- esigi yukseltip gecirmek yerine kolon dusuruldu.
HOURLY_VARIABLES = ["pm10", "pm2_5", "dust"]

#: Beklenen birimler. Parametre sessizce yok sayilirsa esik sayimlari
#: anlamsizlasir -- fail-closed.
BEKLENEN_BIRIM = {"pm10": "μg/m³", "pm2_5": "μg/m³", "dust": "μg/m³"}

#: PM10 esikleri. DSO/AB gunluk sinir degeri 50 ug/m3; 100 belirgin kirli gun.
#: Izolator kirlenmesi BIRIKIMLIDIR, bu yuzden esik-ustu SAAT sayisi mutlak
#: ortalamadan daha bilgilendiricidir.
PM10_SINIR = 50.0
PM10_YUKSEK = 100.0

#: Toz esigi. Ege'de Sahra tasiniminda dust degerleri onlarca kat artar;
#: 20 ug/m3 "belirgin tasinim" isaretidir.
TOZ_TASINIM = 20.0


def _cek(ad: str, lat: float, lon: float, bas: str, son: str, *, retries: int = 3) -> pd.DataFrame:
    """Tek ilcenin tum araligini ceker. Sessiz bos DataFrame DONMEZ."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": bas,
        "end_date": son,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "Europe/Istanbul",
    }
    son_hata: Exception | None = None
    for deneme in range(1, retries + 1):
        try:
            yanit = requests.get(AIR_QUALITY_URL, params=params, timeout=120)
            if yanit.status_code == 429:
                bekle, gerekce = rate_limit_beklemesi(yanit, deneme)
                print(
                    f"  {ad}: hiz/kota siniri; {bekle} sn bekleniyor "
                    f"({gerekce}) [{deneme}/{retries}]"
                )
                time.sleep(bekle)
                son_hata = requests.HTTPError("429")
                continue
            yanit.raise_for_status()
            payload = yanit.json()
            break
        except (requests.RequestException, ValueError) as hata:
            son_hata = hata
            if deneme < retries:
                time.sleep(2**deneme)
    else:
        raise RuntimeError(f"{ad}: {retries} denemede alinamadi. Son hata: {son_hata}")

    if "hourly" not in payload:
        raise RuntimeError(f"{ad}: yanitta 'hourly' yok. Yanit: {str(payload)[:250]}")
    birimler = payload.get("hourly_units", {})
    for degisken, beklenen in BEKLENEN_BIRIM.items():
        gelen = birimler.get(degisken)
        if gelen != beklenen:
            raise RuntimeError(
                f"{ad}: {degisken} birimi {gelen!r} != beklenen {beklenen!r}. "
                "Esik sayimlari anlamsizlasacagi icin durduruldu."
            )

    frame = pd.DataFrame(payload["hourly"]).rename(columns={"time": "zaman"})
    frame["zaman"] = pd.to_datetime(frame["zaman"])
    return frame


def gunluge_indir(saatlik: pd.DataFrame, ilce_key: str) -> pd.DataFrame:
    """Saatlik hava kalitesini gunluk turevlere indirir.

    Esik-ustu SAAT SAYISI, ortalamadan daha bilgilendiricidir: izolator
    kirlenmesi birikimli bir surectir ve "kac saat kirli kaldi" sorusu
    "ortalama ne kadar kirliydi"den daha yakindir.
    """
    f = saatlik.copy()
    f["tarih"] = f["zaman"].dt.normalize()
    pm10 = pd.to_numeric(f["pm10"], errors="coerce")
    toz = pd.to_numeric(f["dust"], errors="coerce")

    f["_pm10_sinir"] = pm10 >= PM10_SINIR
    f["_pm10_yuksek"] = pm10 >= PM10_YUKSEK
    f["_toz_tasinim"] = toz >= TOZ_TASINIM

    gun = f.groupby("tarih").agg(
        pm10_ort=("pm10", "mean"),
        pm10_max=("pm10", "max"),
        pm10_sinir_saat=("_pm10_sinir", "sum"),
        pm10_yuksek_saat=("_pm10_yuksek", "sum"),
        pm25_ort=("pm2_5", "mean"),
        toz_ort=("dust", "mean"),
        toz_max=("dust", "max"),
        toz_tasinim_saat=("_toz_tasinim", "sum"),
    )
    for kolon in ("pm10_sinir_saat", "pm10_yuksek_saat", "toz_tasinim_saat"):
        gun[kolon] = gun[kolon].astype("int16")
    gun = gun.reset_index()
    gun.insert(0, "ilce_key", ilce_key)
    return gun


def kalite_kapisi(birlesik: pd.DataFrame, n_ilce: int) -> None:
    """Kabul edilemez veriyi YAZMADAN ONCE reddeder."""
    eksik = n_ilce - birlesik["ilce_key"].nunique()
    if eksik > 0:
        raise ValueError(f"Kalite kapisi: {eksik} ilce icin veri yok; panel delik kalir.")
    if birlesik.duplicated(subset=["ilce_key", "tarih"]).any():
        raise ValueError("Kalite kapisi: tekrarlanan (ilce_key, tarih) satiri var.")

    print("  NaN oranlari:")
    red: list[str] = []
    for kolon in birlesik.columns:
        if kolon in ("ilce_key", "tarih"):
            continue
        oran = float(birlesik[kolon].isna().mean())
        isaret = ""
        if oran >= 0.10:
            isaret = "  <-- RED"
            red.append(f"{kolon} %{100 * oran:.1f}")
        print(f"    {kolon:20s} %{100 * oran:.3f}{isaret}")
    if red:
        raise ValueError(f"Kalite kapisi: NaN esigi asildi -> {', '.join(red)}. YAZILMADI.")

    # FIZIK: PM10 negatif olamaz ve makul araligin disina cikmamali.
    if float(birlesik["pm10_ort"].min()) < 0:
        raise ValueError("Kalite kapisi: negatif PM10 -- veri bozuk.")
    ust = float(birlesik["pm10_max"].max())
    if ust > 5000:
        raise ValueError(f"Kalite kapisi: PM10 max {ust:.0f} ug/m3 gercekci degil.")
    print(f"  PM10 ort {birlesik['pm10_ort'].mean():.1f}  max {ust:.0f} ug/m3")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-08-15")
    ap.add_argument("--pause", type=float, default=0.5)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    son, uyari = cap_end_date(args.end)
    if uyari:
        print(f"UYARI: {uyari}")
    if not REFERANS.is_file():
        print(f"HATA: ilce referansi yok: {REFERANS}")
        return 1

    ilceler = pd.read_parquet(REFERANS)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Hava kalitesi cekimi: {len(ilceler)} ilce, {args.start} .. {son}")
    print(f"Degiskenler: {', '.join(HOURLY_VARIABLES)}\n")

    basarisiz: list[str] = []
    for sira, satir in enumerate(ilceler.itertuples(index=False), start=1):
        k = str(satir.ilce_key)
        ckpt = CKPT_DIR / f"{k}.parquet"
        # Yalnizca eksik kuyruk -- bkz. fetch_weather.eksik_aralik gerekcesi.
        aralik = None if args.fresh else eksik_aralik(ckpt, args.start, son)
        if not args.fresh and aralik is None:
            print(f"[{sira:3d}/{len(ilceler)}] {k:16s} kontrol noktasindan")
            continue
        cek_bas, cek_son = (args.start, son) if args.fresh else aralik
        print(f"[{sira:3d}/{len(ilceler)}] {k:16s} {cek_bas}..{cek_son} ", end="", flush=True)
        try:
            saatlik = _cek(k, float(satir.lat), float(satir.lon), cek_bas, cek_son)
            gunluk = gunluge_indir(saatlik, k)
            if not args.fresh:
                gunluk = ckpt_birlestir(ckpt, gunluk, anahtarlar=("ilce_key", "tarih"))
            atomic_write_dataframe(gunluk, ckpt)
            print(f"{len(saatlik):,} saat -> {len(gunluk):,} gun")
        except (RuntimeError, requests.RequestException) as hata:
            print(f"HATA: {hata}")
            basarisiz.append(k)
        time.sleep(args.pause)

    parcalar = [
        pd.read_parquet(CKPT_DIR / f"{k}.parquet")
        for k in ilceler["ilce_key"].astype(str)
        if (CKPT_DIR / f"{k}.parquet").is_file()
    ]
    if not parcalar:
        print("\nHicbir ilce icin veri yok.")
        return 1

    birlesik = pd.concat(parcalar, ignore_index=True)
    # Eski kontrol noktalari aod_ort tasiyor olabilir (bkz. HOURLY_VARIABLES
    # notu). Yeniden indirmeye gerek yok; kolon burada dusuruluyor.
    birlesik = birlesik.drop(columns=["aod_ort"], errors="ignore")
    birlesik = birlesik.drop_duplicates(subset=["ilce_key", "tarih"])
    birlesik = birlesik.sort_values(["ilce_key", "tarih"]).reset_index(drop=True)
    print(f"\n{len(birlesik):,} satir x {birlesik.shape[1]} kolon")
    kalite_kapisi(birlesik, len(ilceler))

    atomic_write_dataframe(birlesik, CIKTI)
    print(f"\nYazildi: {CIKTI}")
    print(f"  {birlesik['tarih'].min().date()} .. {birlesik['tarih'].max().date()}")
    if basarisiz:
        print(f"\n  UYARI: {len(basarisiz)} ilce alinamadi: {basarisiz[:5]}")
        return 1
    print("\n  Kaynak: Open-Meteo Air Quality (CC-BY-4.0). Notebook'ta atif ver.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
