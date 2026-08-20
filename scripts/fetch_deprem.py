"""AFAD deprem katalogunu indirir (Ege bolgesi, M>=4.0).

NEDEN BU BETIK
--------------
docs/10 bolum 5: Aydin-Denizli graben hatti aktif deprem bolgesidir ve buyuk
sarsintilar sebeke arizasi/kesinti sicramasi uretir (2020 Samos depremi Izmir
dagitim sebekesinde somut ornek). Deprem gunleri ve ilceye uzaklik, kesinti
tahmini icin ucuz ve tarihi kesin bir feature kaynagidir.

KAYNAK SECIMI
-------------
BIRINCIL: AFAD apiv2 (deprem.afad.gov.tr/apiv2/event/filter). Dogrulandi:
  anahtar gerektirmez, JSON doner, il/ilce alanlarini da verir.
  Parametre adlari: start, end, minmag, minlat/maxlat/minlon/maxlon.
YEDEK: USGS FDSN (earthquake.usgs.gov/fdsnws/event/1/query, format=csv).
  AFAD erisilemezse ayni sinir kutusuyla otomatik devreye girer; il/ilce
  kolonlari o durumda bos kalir (USGS Turk idari birimini bilmez).
Bu kosuda kazanan: AFAD (betigi calistiran cikti satiri hangisinin
kullanildigini ayrica yazar).

Kullanim::

    python scripts/fetch_deprem.py
    python scripts/fetch_deprem.py --start 2020-01-01 --end 2026-08-15 --minmag 4.0
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gridup.io_utils import atomic_write_dataframe  # noqa: E402

AFAD_URL = "https://deprem.afad.gov.tr/apiv2/event/filter"
USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

#: Ege sinir kutusu -- GDZ (Izmir, Manisa) + ADM (Aydin, Denizli, Mugla)
#: bolgesini ve kiyi otesindeki fay hatlarini (Samos, Gokova) kapsar.
LAT_MIN, LAT_MAX = 36.0, 39.5
LON_MIN, LON_MAX = 26.0, 30.5

#: Tek istekte cok yil istemek sessiz sonuc kirpilmasina acik; yil yil cekip
#: birlestiriyoruz. Istekler arasi bekleme API'ye nezakettir.
REQUEST_PAUSE_S = 1.0
TIMEOUT_S = 60
RETRIES = 3

CIKTI_YOLU = Path("data/external/depremler.parquet")

#: Varsayilan alt buyukluk esigi. 4.0'DAN 3.0'A INDIRILDI (2026-08-20).
#:
#: OLCULDU: M>=4.0 esigiyle 6,5 yilda toplam 373 olay vardi -- 96 ilce x
#: ~2400 gun = 230.400 panel satirinda 217 farkli gun. Yani deprem yogunlugu
#: feature'i satirlarin %99.9'unda SIFIRDI ve modele fiilen hicbir sey
#: soylemiyordu. Gutenberg-Richter yasasi geregi bir birim dusuk esik ~10 kat
#: cok olay demektir; M>=3.0 sinyale gercek bir cozunurluk kazandirir.
#:
#: Bunun GUVENLI olmasi ENERJI AGIRLIGINA baglidir (asagi bkz.): M3 bir olay
#: toplama M4'un ~1/32'si kadar katkida bulunur, dolayisiyla esigi dusurmek
#: buyuk depremlerin isaretini SULANDIRMAZ, yalnizca "sismik olarak hareketli
#: gun" kavramini olculebilir kilar.
VARSAYILAN_MINMAG = 3.0

#: Buyuklugun kendisi AGIRLIK OLARAK KULLANILAMAZ: Richter olcegi
#: LOGARITMIKTIR. M5 bir deprem M3'un yaklasik 1000 KATI enerji birakir, ama
#: buyukluk degeri olarak yalnizca 1,67 katidir. ``buyukluk`` kolonunu
#: toplayan bir yogunluk feature'i bu yuzden otuz kucuk sarsintiyi bir buyuk
#: depremden DAHA ONEMLI gosterir -- tam tersi dogrudur.
#:
#: ``enerji`` kolonu bunu duzeltir: log10(E) ~ 1.5*M (Gutenberg-Richter
#: enerji bagintisi). M4'e gore GORECELI yaziyoruz ki sayilar 1 civarinda
#: kalsin ve float32'de tasmasin:
#:
#:     M3.0 -> 0.032    M4.0 -> 1.0    M5.0 -> 31.6    M6.0 -> 1000
ENERJI_REFERANS_MAG = 4.0


def enerji_agirligi(buyukluk: pd.Series) -> pd.Series:
    """Buyuklugu, M4'e gore goreceli sismik enerjiye cevirir."""
    return 10.0 ** (1.5 * (pd.to_numeric(buyukluk, errors="coerce") - ENERJI_REFERANS_MAG))


def _yil_dilimleri(start: str, end: str) -> list[tuple[str, str]]:
    """[start, end] araligini yil sinirlarinda dilimlere boler."""
    bas = pd.Timestamp(start)
    son = pd.Timestamp(end)
    dilimler: list[tuple[str, str]] = []
    while bas <= son:
        yil_sonu = min(pd.Timestamp(year=bas.year, month=12, day=31), son)
        dilimler.append((bas.date().isoformat(), yil_sonu.date().isoformat()))
        bas = yil_sonu + pd.Timedelta(days=1)
    return dilimler


def _getir(url: str, params: dict[str, str]) -> requests.Response:
    """GET + yeniden deneme. Tum denemeler duserse son hatayi firlatir."""
    son_hata: Exception | None = None
    for deneme in range(1, RETRIES + 1):
        try:
            yanit = requests.get(
                url,
                params=params,
                timeout=TIMEOUT_S,
                headers={"User-Agent": "Mozilla/5.0 (datathon veri toplayici)"},
            )
            yanit.raise_for_status()
            return yanit
        except requests.RequestException as hata:
            son_hata = hata
            if deneme < RETRIES:
                time.sleep(2**deneme)
    raise RuntimeError(f"{url}: {RETRIES} denemede yanit alinamadi. Son hata: {son_hata}")


def fetch_afad(start: str, end: str, minmag: float) -> pd.DataFrame:
    """AFAD apiv2'den olay listesi ceker; yil dilimleriyle birlestirir."""
    parcalar: list[pd.DataFrame] = []
    for dilim_bas, dilim_son in _yil_dilimleri(start, end):
        params = {
            "start": f"{dilim_bas} 00:00:00",
            "end": f"{dilim_son} 23:59:59",
            "minmag": str(minmag),
            "minlat": str(LAT_MIN),
            "maxlat": str(LAT_MAX),
            "minlon": str(LON_MIN),
            "maxlon": str(LON_MAX),
            "orderby": "timedesc",
        }
        yanit = _getir(AFAD_URL, params)
        kayitlar = yanit.json()
        print(f"  AFAD {dilim_bas}..{dilim_son}: {len(kayitlar)} olay")
        if kayitlar:
            parcalar.append(pd.DataFrame(kayitlar))
        time.sleep(REQUEST_PAUSE_S)

    if not parcalar:
        raise RuntimeError("AFAD hicbir dilim icin olay dondurmedi.")

    ham = pd.concat(parcalar, ignore_index=True)
    ham = ham.drop_duplicates(subset=["eventID"])
    return pd.DataFrame(
        {
            # AFAD tarihleri karisik ISO bicimli: kimi kayitta saniye kesiri
            # var, kiminde yok. format="ISO8601" ikisini de kabul eder.
            "tarih": pd.to_datetime(ham["date"], format="ISO8601").dt.date,
            "lat": pd.to_numeric(ham["latitude"], errors="coerce"),
            "lon": pd.to_numeric(ham["longitude"], errors="coerce"),
            "buyukluk": pd.to_numeric(ham["magnitude"], errors="coerce"),
            "derinlik_km": pd.to_numeric(ham["depth"], errors="coerce"),
            "il": ham["province"],
            "ilce": ham["district"],
            "kaynak": "AFAD",
        }
    )


def fetch_usgs(start: str, end: str, minmag: float) -> pd.DataFrame:
    """USGS FDSN CSV yedegi -- AFAD erisilemezse ayni kutu, ayni esik."""
    params = {
        "format": "csv",
        "starttime": start,
        "endtime": f"{end}T23:59:59",
        "minmagnitude": str(minmag),
        "minlatitude": str(LAT_MIN),
        "maxlatitude": str(LAT_MAX),
        "minlongitude": str(LON_MIN),
        "maxlongitude": str(LON_MAX),
        "orderby": "time",
    }
    yanit = _getir(USGS_URL, params)
    ham = pd.read_csv(io.StringIO(yanit.text))
    return pd.DataFrame(
        {
            "tarih": pd.to_datetime(ham["time"]).dt.date,
            "lat": ham["latitude"],
            "lon": ham["longitude"],
            "buyukluk": ham["mag"],
            "derinlik_km": ham["depth"],
            "il": pd.NA,
            "ilce": pd.NA,
            "kaynak": "USGS",
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2020-01-01", help="Baslangic (YYYY-AA-GG)")
    parser.add_argument("--end", default="2026-08-15", help="Bitis (YYYY-AA-GG)")
    parser.add_argument(
        "--minmag",
        type=float,
        default=VARSAYILAN_MINMAG,
        help=f"Alt buyukluk esigi (varsayilan {VARSAYILAN_MINMAG}; gerekce icin modul basligi)",
    )
    parser.add_argument("--out", default=str(CIKTI_YOLU), help="Cikti parquet yolu")
    args = parser.parse_args()

    print(
        f"Deprem katalogu: {args.start} - {args.end}, M>={args.minmag}, "
        f"kutu lat {LAT_MIN}-{LAT_MAX} lon {LON_MIN}-{LON_MAX}"
    )

    try:
        tablo = fetch_afad(args.start, args.end, args.minmag)
        print("Kaynak: AFAD (birincil)")
    except (RuntimeError, requests.RequestException, ValueError, KeyError) as hata:
        print(f"AFAD basarisiz ({hata}); USGS FDSN yedegine geciliyor.")
        tablo = fetch_usgs(args.start, args.end, args.minmag)
        print("Kaynak: USGS (yedek)")

    tablo = tablo.dropna(subset=["lat", "lon", "buyukluk"]).sort_values("tarih")
    tablo = tablo.reset_index(drop=True)
    tablo["enerji"] = enerji_agirligi(tablo["buyukluk"]).astype("float64")

    cikti = Path(args.out)
    atomic_write_dataframe(tablo, cikti)

    print(f"Yazildi: {cikti}")
    print(f"  {len(tablo)} olay, {tablo['tarih'].min()} - {tablo['tarih'].max()}")
    print(f"  Buyukluk: {tablo['buyukluk'].min():.1f} - {tablo['buyukluk'].max():.1f}")
    print(f"  Farkli gun: {tablo['tarih'].nunique()}")
    # Enerji toplaminin ne kadarini buyuk depremler tasiyor -- esigi
    # dusurmenin sinyali sulandirmadiginin dogrudan kaniti.
    buyuk = tablo.loc[tablo["buyukluk"] >= 4.0, "enerji"].sum()
    toplam = float(tablo["enerji"].sum())
    if toplam > 0:
        print(f"  Toplam enerjinin %{100 * buyuk / toplam:.1f}'i M>=4.0 olaylarindan")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
