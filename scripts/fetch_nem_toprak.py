"""NEM, CIY, TOPRAK NEMI, VPD, BULUT ve ET0 -- gunluk turevler.

NEDEN BU BETIK
--------------
``fetch_weather.py`` gunluk tabloyu, ``fetch_hourly_weather.py`` basinc/ruzgar
turevlerini cekiyor. Ikisi de Open-Meteo'nun ayni ucretsiz arsiv ucundan
besleniyor ama su alti degiskeni HIC istenmiyordu -- oysa ayni istekte,
ayni lisansla (CC-BY-4.0), ek maliyet olmadan geliyorlar:

  relative_humidity_2m        nem
  dew_point_2m                ciy noktasi
  soil_moisture_0_to_7cm      yuzey toprak nemi
  vapour_pressure_deficit     buhar basinci acigi (VPD)
  cloud_cover_low             alcak bulut
  et0_fao_evapotranspiration  referans buharlasma-terleme

Her birinin dagitim sebekesine giden AYRI bir zinciri var:

* **Nem yuksek (>=%90)** -> izolatorde kirlilik + nem birleserek yuzey
  kacak akimi ve ATLAMA (flashover) uretir. Kuru kirli izolator atlamaz;
  islanan atlar. Bu, sicaklik veya yagisla yakalanamayan ayri bir yoldur:
  yagmur yagmadan da nem %95 olabilir.
* **Nem dusuk (<%30) + VPD yuksek** -> bitki ortusu kurur, yangin riski
  artar; ayni kosul iletken sarkmasini da degistirir.
* **Sis (nem >=%97)** -> hem izolator hem gorus; saha ekibinin ariza
  giderme suresi uzar.
* **Toprak nemi yuksek** -> zemin gevser, DIREK DEVRILMESI ve kazi kaynakli
  ariza olasiligi artar. Ruzgar hasarinin ayni ruzgar hizinda islak zeminde
  daha buyuk olmasinin sebebi budur -- yani ruzgarla ETKILESIMLI bir
  degiskendir, tek basina degil.
* **ET0** -> tarimsal SULAMA TALEBININ fiziksel vekili. "Mayis-Eylul sulama
  sezonu" bayragindan cok daha iyidir: sulama, takvime degil bitkinin su
  kaybina gore yapilir. Manisa bagciligi, Aydin incir/zeytini ve Denizli
  tariminda sulama pompalari kirsal trafolarda yaz yukunu belirler.
* **Alcak bulut** -> hem konvektif aktivite gostergesi hem cati GES
  uretiminin dususu (dagitik uretim, ADM/GDZ'nin DERMS gundemi).

NEDEN ``models=era5_land`` KULLANILMIYOR -- OLCULDU (2026-08-18)
--------------------------------------------------------------
``models=era5_land`` cozunurlugu 0.25 dereceden (~25 km) 0.1 dereceye
(~11 km) indirir ve toprak neminde GERCEKTEN farkli deger dondurur
(olculdu: 0.1043 vs 0.0940, ayni gun ayni koordinat).

Ama bir YUKSELTME degil, TAKAS oldugu olculdu -- ayni istekte alti
degiskenin ikisi tamamen bos geliyor:

    relative_humidity_2m        ERA5 %0 NaN   era5_land %0 NaN
    dew_point_2m                     %0            %0
    soil_moisture_0_to_7cm           %0            %0
    vapour_pressure_deficit          %0            %0
    cloud_cover_low                  %0        %100 NaN
    et0_fao_evapotranspiration       %0        %100 NaN

Kaybedilenlerden ET0, bu tablonun EN DEGERLI degiskeni: tarimsal sulama
talebinin fiziksel vekili ve "Mayis-Eylul sulama sezonu" bayragindan cok
daha iyi. Daha ince toprak nemi icin onu vermek kotu bir takas.

Ayrica ilk 36 ilce zaten varsayilan ERA5 ile cekilmisti; yarisindan sonra
model degistirmek ILCELER ARASI sessiz bir tutarsizlik yaratirdi.

KULLANIM
--------
::

    python scripts/fetch_nem_toprak.py
    python scripts/fetch_nem_toprak.py --start 2020-01-01 --end 2026-08-15
    python scripts/fetch_nem_toprak.py --fresh    # kontrol noktalarini yok say

Cikti: ``data/external/nem_toprak_gunluk.parquet`` (ilce_key x tarih).
Kaynak: Open-Meteo Archive API, CC-BY-4.0. Anahtar GEREKMEZ.
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
    ARCHIVE_URL,
    cap_end_date,
    checkpoint_covers,
    rate_limit_beklemesi,
)

from gridup.io_utils import atomic_write_dataframe  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REFERANS = ROOT / "data" / "reference" / "ilceler_gdz_adm.parquet"
CIKTI = ROOT / "data" / "external" / "nem_toprak_gunluk.parquet"
CKPT_DIR = ROOT / "data" / "external" / ".nem_toprak_ckpt"

#: Saatlik degiskenler. Sicaklik/yagis/ruzgar BILEREK yok -- onlar diger iki
#: tabloda var; burada yalnizca eksik olan alti degisken isteniyor.
HOURLY_VARIABLES = [
    "relative_humidity_2m",
    "dew_point_2m",
    "soil_moisture_0_to_7cm",
    "vapour_pressure_deficit",
    "cloud_cover_low",
    "et0_fao_evapotranspiration",
]

#: API'nin dondurmesi GEREKEN birimler. Parametre sessizce yok sayilir veya
#: birim degisirse esik sayimlari anlamsizlasir; bu yuzden fail-closed.
BEKLENEN_BIRIM = {
    "relative_humidity_2m": "%",
    "soil_moisture_0_to_7cm": "m³/m³",
    "vapour_pressure_deficit": "kPa",
    "cloud_cover_low": "%",
    "et0_fao_evapotranspiration": "mm",
}

#: Esikler. Her biri fiziksel bir olguya baglidir, yuvarlak sayi secilmedi:
#: %90 izolator yuzey iletkenliginin belirgin arttigi bolge, %97 sis
#: pratigi, %30 kuruluk/yangin esigi (meteoroloji standardi).
NEM_YUKSEK = 90.0
NEM_SIS = 97.0
NEM_DUSUK = 30.0
#: m3/m3. Cogu toprak icin doygunluk ~0.40-0.45; 0.35 "islak zemin" sinirini
#: temsil eder. TAHMINIDIR: toprak tipine gore kayar, bu yuzden surekli
#: degerler (ort/max) de tabloda tutuluyor ve model kendi esigini bulabilir.
TOPRAK_ISLAK = 0.35


def _kos(
    name: str,
    latitude: float,
    longitude: float,
    start: str,
    end: str,
    *,
    timeout: int = 120,
    retries: int = 3,
) -> pd.DataFrame:
    """Tek ilcenin tum araligini ceker. Sessiz bos DataFrame DONMEZ."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "Europe/Istanbul",
    }

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(ARCHIVE_URL, params=params, timeout=timeout)
            if response.status_code == 429:
                wait, gerekce = rate_limit_beklemesi(response, attempt)
                print(
                    f"  {name}: hiz siniri (429); {wait} sn bekleniyor "
                    f"({gerekce}) [{attempt}/{retries}]"
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
                print(f"  {name}: deneme {attempt} basarisiz ({error}); {wait} sn")
                time.sleep(wait)
    else:
        raise RuntimeError(f"{name}: {retries} denemede veri alinamadi. Son hata: {last_error}")

    if "hourly" not in payload:
        raise RuntimeError(f"{name}: yanitta 'hourly' bolumu yok. Yanit: {str(payload)[:300]}")

    birimler = payload.get("hourly_units", {})
    for degisken, beklenen in BEKLENEN_BIRIM.items():
        gelen = birimler.get(degisken)
        if gelen != beklenen:
            raise RuntimeError(
                f"{name}: {degisken} birimi beklenenden farkli: {gelen!r} != {beklenen!r}. "
                "Esik sayimlari anlamsizlasacagi icin durduruldu."
            )

    frame = pd.DataFrame(payload["hourly"]).rename(columns={"time": "zaman"})
    frame["zaman"] = pd.to_datetime(frame["zaman"])
    return frame


def gunluge_indir(saatlik: pd.DataFrame, ilce_key: str) -> pd.DataFrame:
    """Saatlik seriyi gunluk turev tablosuna indirir.

    Esik sayimlarinda NaN saat "esik asilmadi" sayilir (bool karsilastirmada
    NaN -> False). Surekli istatistikler NaN atlar; gunun tamami NaN ise
    sonuc NaN kalir ve kalite kapisinda gorunur.
    """
    f = saatlik.copy()
    f["tarih"] = f["zaman"].dt.normalize()
    nem = f["relative_humidity_2m"]
    toprak = f["soil_moisture_0_to_7cm"]

    f["_nem_yuksek"] = nem >= NEM_YUKSEK
    f["_nem_sis"] = nem >= NEM_SIS
    f["_nem_dusuk"] = nem < NEM_DUSUK
    f["_toprak_islak"] = toprak >= TOPRAK_ISLAK

    gun = f.groupby("tarih").agg(
        nem_ort=("relative_humidity_2m", "mean"),
        nem_min=("relative_humidity_2m", "min"),
        nem_max=("relative_humidity_2m", "max"),
        nem_yuksek_saat=("_nem_yuksek", "sum"),
        sis_saat=("_nem_sis", "sum"),
        nem_dusuk_saat=("_nem_dusuk", "sum"),
        ciy_ort=("dew_point_2m", "mean"),
        ciy_max=("dew_point_2m", "max"),
        toprak_nem_ort=("soil_moisture_0_to_7cm", "mean"),
        toprak_nem_max=("soil_moisture_0_to_7cm", "max"),
        toprak_islak_saat=("_toprak_islak", "sum"),
        vpd_ort=("vapour_pressure_deficit", "mean"),
        vpd_max=("vapour_pressure_deficit", "max"),
        bulut_dusuk_ort=("cloud_cover_low", "mean"),
        et0_toplam=("et0_fao_evapotranspiration", "sum"),
    )
    for kolon in ("nem_yuksek_saat", "sis_saat", "nem_dusuk_saat", "toprak_islak_saat"):
        gun[kolon] = gun[kolon].astype("int16")
    gun = gun.reset_index()
    gun.insert(0, "ilce_key", ilce_key)
    return gun


def kalite_kapisi(birlesik: pd.DataFrame, n_ilce: int, start: str, end: str) -> None:
    """Kabul edilemez veriyi YAZMADAN ONCE reddeder.

    ``fetch_hourly_weather``te ogrenilen ders: yalnizca yazdiran bir
    dogrulama, gozetimsiz gece kosusunda bozuk veriyi sessizce yayinlar.
    """
    n_gun = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    beklenen = n_ilce * n_gun
    print(f"  Beklenen ~{beklenen:,} satir ({n_ilce} ilce x {n_gun} gun); gercek {len(birlesik):,}")

    tekrar = int(birlesik.duplicated(subset=["ilce_key", "tarih"]).sum())
    if tekrar:
        raise ValueError(f"Kalite kapisi: {tekrar} tekrar eden (ilce_key, tarih) satiri.")

    eksik_ilce = n_ilce - birlesik["ilce_key"].nunique()
    if eksik_ilce > 0:
        raise ValueError(
            f"Kalite kapisi: {eksik_ilce} ilce icin hic veri yok. Panel delik kalir; "
            "cekimi tekrarla (kontrol noktalari korunuyor, bastan indirmez)."
        )

    print("  NaN oranlari:")
    red: list[str] = []
    for kolon in birlesik.columns:
        if kolon in ("ilce_key", "tarih"):
            continue
        oran = float(birlesik[kolon].isna().mean())
        isaret = ""
        if oran >= 0.10:
            isaret = "  <-- RED (>=%10)"
            red.append(f"{kolon} %{100 * oran:.1f}")
        elif oran >= 0.02:
            isaret = "  <-- %2 ustu"
        print(f"    {kolon:20s} %{100 * oran:.3f}{isaret}")

    if red:
        raise ValueError(
            "Kalite kapisi: su kolonlarda NaN orani %10 esigini asti -> "
            f"{', '.join(red)}. Parquet YAZILMADI."
        )


def main() -> int:
    ayristirici = argparse.ArgumentParser(description=__doc__)
    ayristirici.add_argument("--start", default="2020-01-01", help="Baslangic (YYYY-AA-GG)")
    ayristirici.add_argument("--end", default="2026-08-15", help="Bitis (YYYY-AA-GG)")
    ayristirici.add_argument("--pause", type=float, default=0.6, help="Istekler arasi bekleme (sn)")
    ayristirici.add_argument("--fresh", action="store_true", help="Kontrol noktalarini yok say")
    args = ayristirici.parse_args()

    # cap_end_date CIFT doner: (kirpilmis_tarih, uyari_metni | None). Donusu
    # dogrudan kullanmak, requests'in end_date'i IKI KEZ gondermesine ve
    # API'nin her konum icin HTTP 400 vermesine yol acar (olculdu).
    end, uyari = cap_end_date(args.end)
    if uyari:
        print(f"UYARI: {uyari}")

    if not REFERANS.is_file():
        print(f"HATA: ilce referansi yok: {REFERANS}")
        print("      Once: python scripts/fetch_districts.py")
        return 1

    ilceler = pd.read_parquet(REFERANS)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Nem/toprak/VPD cekimi: {len(ilceler)} ilce, {args.start} .. {end}")
    print(f"Degiskenler: {', '.join(HOURLY_VARIABLES)}\n")

    basarisiz: list[str] = []
    for sira, satir in enumerate(ilceler.itertuples(index=False), start=1):
        ilce_key = str(satir.ilce_key)
        ckpt = CKPT_DIR / f"{ilce_key}.parquet"
        if not args.fresh and checkpoint_covers(ckpt, args.start, end):
            print(f"[{sira:3d}/{len(ilceler)}] {ilce_key:16s} kontrol noktasindan")
            continue
        print(f"[{sira:3d}/{len(ilceler)}] {ilce_key:16s} ", end="", flush=True)
        try:
            saatlik = _kos(ilce_key, float(satir.lat), float(satir.lon), args.start, end)
            gunluk = gunluge_indir(saatlik, ilce_key)
            atomic_write_dataframe(gunluk, ckpt)
            print(f"{len(saatlik):,} saat -> {len(gunluk):,} gun")
        except (RuntimeError, requests.RequestException) as hata:
            print(f"HATA: {hata}")
            basarisiz.append(ilce_key)
        time.sleep(args.pause)

    parcalar = [
        pd.read_parquet(CKPT_DIR / f"{k}.parquet")
        for k in ilceler["ilce_key"].astype(str)
        if (CKPT_DIR / f"{k}.parquet").is_file()
    ]
    if not parcalar:
        print("\nHicbir ilce icin veri yok. Internet baglantisini kontrol et.")
        return 1

    birlesik = pd.concat(parcalar, ignore_index=True)
    birlesik = birlesik.drop_duplicates(subset=["ilce_key", "tarih"])
    birlesik = birlesik.sort_values(["ilce_key", "tarih"]).reset_index(drop=True)

    print(f"\n{len(birlesik):,} satir x {birlesik.shape[1]} kolon")
    kalite_kapisi(birlesik, len(ilceler), args.start, end)

    atomic_write_dataframe(birlesik, CIKTI)
    print(f"\nYazildi: {CIKTI}")
    print(f"  Tarih araligi: {birlesik['tarih'].min().date()} - {birlesik['tarih'].max().date()}")

    if basarisiz:
        print(f"\n  UYARI: {len(basarisiz)} ilce alinamadi: {basarisiz}")
        return 1

    print("\n  Kaynak: Open-Meteo (CC-BY-4.0). Notebook'ta atif ver.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
