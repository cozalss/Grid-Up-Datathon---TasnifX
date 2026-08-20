"""SAATLIK TUREVLI PANEL TABLOLARINI GELECEGE KOPRULER (Open-Meteo forecast).

NEDEN BU BETIK
--------------
``fetch_weather_bridge.py`` GUNLUK hava tablosunu bugun+16'ya kadar kopruler
ve gerekcesi acikti: "yarisma test blogu arsivin bittigi tarihi asarsa 17 hava
kolonu YALNIZCA testte NaN olur -- CV'de gorunmeyen, sessiz bir bozulma."

Ama o koprü tek bir tabloyu kurtardi. 2026-08-20 denetiminde olculdu: ayni
tehlike, saatlikten turetilen UC tablonun tamaminda acikta duruyordu --

    hava_saatlik_turev · konvektif_gunluk · nem_toprak_gunluk

Bunlarin hicbiri arsivin otesine gecmiyordu. Sonuc, panelin gelecege bakan
her satirinda gunluk hava ailesinin DOLU, digerlerinin tamamen BOS olmasidir.
Bu, tek bir kaynagin eksik olmasindan daha kotudur: model egitimde bu
kolonlara guvenmeyi ogrenir, tam teslim aninda hicbiri yoktur.

NEDEN TEK ISTEK
---------------
Olculdu (2026-08-20): forecast API'si uc tablonun ihtiyac duydugu ON UC
degiskenin tamamini AYNI ADLA ve AYNI BIRIMLE tek yanitta veriyor, NaN orani
<=%1.4. Yani ilce basina UC degil BIR istek yetiyor -- kota ucte bire iner.

Ayrica forecast API'si arsivden AYRI kotada isliyor (olculdu: arsiv 429
verirken forecast 200 dondu), yani arsiv cekimi tikandiginda bile bu betik
calisabilir.

DIKIS KONTROLU
--------------
Arsiv ve tahmin, ortusen gunlerde AYNI degeri vermelidir. Vermiyorsa ya
koordinat kaymistir ya birim degismistir ya da model tamamen farkli bir
degiskeni dondurmustur. Her koprü icin bir referans kolon ve tolerans
tanimli; asilirsa betik REDDEDER ve hicbir sey yazmaz.

TURETME AYNI FONKSIYONDAN
-------------------------
Gunluk kolonlar, arsiv cekicilerinin KENDI ``gunluge_indir`` /
``aggregate_daily`` fonksiyonlariyla uretilir. Kopyalanmis bir toplama
mantigi, esikler degistiginde sessizce ayrisirdi -- bu depoda bugun tam
olarak o hata bulundu (bkz. docs/17, madde 3.4).

KULLANIM
--------
::

    python scripts/kopru_saatlik.py
    python scripts/kopru_saatlik.py --gun 16 --dry-run

Cikis kodu: 0 = koprü kuruldu, 1 = en az bir tablo reddedildi.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fetch_weather import rate_limit_beklemesi  # noqa: E402

from gridup.io_utils import atomic_write_dataframe  # noqa: E402

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
REFERANS = ROOT / "data" / "reference" / "ilceler_gdz_adm.parquet"

#: Forecast API'sinin geriye donuk verebilecegi en fazla gun.
MAX_PAST_DAYS = 92
#: Ileriye verebilecegi en fazla gun.
MAX_FORECAST_DAYS = 16

#: Arsivle kac gun ORTUSSUN. Dikis kontrolu bu gunlerde yapilir; iki gun,
#: tek gunluk bir tesadufi uyumun kontrolu gecmesini engelleyecek kadar.
ORTUSME_GUN = 3

#: Tahmin satirlarini isaretleyen kolon. Feature DEGILDIR -- external.py
#: bunu panele tasimaz (bkz. hava_tahmin icin ayni gerekce).
TAHMIN_KOLONU = "tahmin"

DEFAULT_PAUSE = 0.4


def _modul(ad: str):
    """``scripts/<ad>.py``yi modul olarak yukler (betikler paket degil).

    Zaten yuklenmisse YENIDEN CALISTIRMAZ. Her cagride exec etmek, ayni
    dosyadan iki AYRI fonksiyon nesnesi uretir; o zaman "koprü arsiv
    cekicisinin KENDI fonksiyonunu mu kullaniyor" sorusu kimlikle
    dogrulanamaz hale gelir (ve modul duzeyi is iki kez yapilir).
    """
    var_olan = sys.modules.get(ad)
    if var_olan is not None and getattr(var_olan, "__file__", None):
        return var_olan
    spec = importlib.util.spec_from_file_location(ad, ROOT / "scripts" / f"{ad}.py")
    assert spec is not None and spec.loader is not None
    modul = importlib.util.module_from_spec(spec)
    sys.modules[ad] = modul
    spec.loader.exec_module(modul)
    return modul


@dataclass(frozen=True)
class Kopru:
    """Bir panel tablosunun koprü sozlesmesi."""

    ad: str
    yol: str
    #: Bu tablonun toplama fonksiyonunun bekledigi saatlik degiskenler.
    degiskenler: tuple[str, ...]
    #: ``(saatlik_frame, ilce_key) -> gunluk_frame``
    topla: Callable[[pd.DataFrame, str], pd.DataFrame]
    #: Dikis kontrolu icin referans kolon ve izin verilen mutlak fark.
    dikis_kolonu: str
    dikis_toleransi: float


def _kopruleri_kur() -> tuple[Kopru, ...]:
    """Koprüleri, arsiv cekicilerinin KENDI fonksiyonlarina baglayarak kurar."""
    saatlik = _modul("fetch_hourly_weather")
    konvektif = _modul("fetch_konvektif")
    nem = _modul("fetch_nem_toprak")
    return (
        Kopru(
            ad="hava_saatlik_turev",
            yol="data/external/hava_saatlik_turev.parquet",
            degiskenler=tuple(saatlik.HOURLY_VARIABLES),
            topla=saatlik.aggregate_daily,
            # Basinc gunler arasi yavas degisir ve modeller uzerinde iyi
            # uyusur; 2 hPa gercek bir kaymayi yakalar, gurultu yaratmaz.
            dikis_kolonu="basinc_ort",
            dikis_toleransi=2.0,
        ),
        Kopru(
            ad="konvektif_gunluk",
            yol="data/external/konvektif_gunluk.parquet",
            degiskenler=tuple(konvektif.HOURLY_VARIABLES),
            topla=konvektif.gunluge_indir,
            # CAPE dogasi geregi oynak; mutlak fark yerine genis bir tolerans
            # kullaniyoruz. Amac "ayni buyukluk mertebesi mi" sorusudur --
            # birim/koordinat hatasi mertebeyi degistirir, gurultu degistirmez.
            dikis_kolonu="cape_ort",
            dikis_toleransi=150.0,
        ),
        Kopru(
            ad="nem_toprak_gunluk",
            yol="data/external/nem_toprak_gunluk.parquet",
            degiskenler=tuple(nem.HOURLY_VARIABLES),
            topla=nem.gunluge_indir,
            # Bagil nem yuzde; 8 puan fark model farkiyla aciklanabilir,
            # fazlasi baska bir degiskene bakildigi anlamina gelir.
            dikis_kolonu="nem_ort",
            dikis_toleransi=8.0,
        ),
    )


def tum_degiskenler(kopruler: Sequence[Kopru]) -> list[str]:
    """Koprülerin ihtiyac duydugu degiskenlerin BIRLESIMI, sirali ve tekil."""
    gorulen: dict[str, None] = {}
    for kopru in kopruler:
        for degisken in kopru.degiskenler:
            gorulen[degisken] = None
    return list(gorulen)


def forecast_cek(
    ad: str,
    lat: float,
    lon: float,
    degiskenler: Sequence[str],
    *,
    past_days: int,
    forecast_days: int,
    retries: int = 3,
) -> pd.DataFrame:
    """Tek ilcenin koprü penceresini ceker. Sessiz bos frame DONMEZ."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(degiskenler),
        "past_days": min(past_days, MAX_PAST_DAYS),
        "forecast_days": min(forecast_days, MAX_FORECAST_DAYS),
        "timezone": "Europe/Istanbul",
        "wind_speed_unit": "ms",
    }
    son_hata: Exception | None = None
    for deneme in range(1, retries + 1):
        try:
            yanit = requests.get(FORECAST_URL, params=params, timeout=120)
            if yanit.status_code == 429:
                bekle, gerekce = rate_limit_beklemesi(yanit, deneme)
                print(f"  {ad}: hiz siniri; {bekle} sn ({gerekce}) [{deneme}/{retries}]")
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
    eksik = [d for d in degiskenler if d not in payload["hourly"]]
    if eksik:
        raise RuntimeError(
            f"{ad}: forecast API su degiskenleri dondurmedi: {eksik}. "
            "Arsivle AYNI degiskeni uretemezsek koprü dikisi anlamsizdir."
        )
    frame = pd.DataFrame(payload["hourly"]).rename(columns={"time": "zaman"})
    frame["zaman"] = pd.to_datetime(frame["zaman"])
    return frame


def dikis_farki(arsiv: pd.DataFrame, tahmin: pd.DataFrame, kopru: Kopru) -> float:
    """Ortusen gunlerde arsiv ve tahmin arasindaki ortalama mutlak fark.

    Raises:
        ValueError: Hic ortusme yoksa -- dikis kontrolsuz birlestirme,
            kontrolun kendisini anlamsiz kilar.
    """
    a = arsiv.set_index(["ilce_key", "tarih"])[kopru.dikis_kolonu]
    t = tahmin.set_index(["ilce_key", "tarih"])[kopru.dikis_kolonu]
    ortak = a.index.intersection(t.index)
    if len(ortak) == 0:
        raise ValueError(
            f"{kopru.ad}: arsiv ve tahmin hic ortusmuyor; dikis kontrolu yapilamaz. "
            "past_days degerini artir."
        )
    return float((a.loc[ortak] - t.loc[ortak]).abs().mean())


def kopruyu_birlestir(arsiv: pd.DataFrame, tahmin: pd.DataFrame) -> pd.DataFrame:
    """Yalnizca arsivde OLMAYAN gunleri ekler; arsiv her zaman kazanir.

    Arsiv (ERA5) tahminden dogrudur; ortusen bir gunde tahmini tercih etmek
    veriyi bilerek kotulestirmek olurdu.
    """
    eski = arsiv.copy()
    if TAHMIN_KOLONU not in eski.columns:
        eski[TAHMIN_KOLONU] = 0
    yeni = tahmin.copy()
    yeni[TAHMIN_KOLONU] = 1

    mevcut = set(map(tuple, eski[["ilce_key", "tarih"]].to_numpy()))
    maske = [tuple(x) not in mevcut for x in yeni[["ilce_key", "tarih"]].to_numpy()]
    yeni = yeni.loc[maske, eski.columns]

    birlesik = pd.concat([eski, yeni], ignore_index=True)
    birlesik[TAHMIN_KOLONU] = birlesik[TAHMIN_KOLONU].astype("int8")
    return birlesik.sort_values(["ilce_key", "tarih"]).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gun", type=int, default=MAX_FORECAST_DAYS, help="Ileri kac gun")
    ap.add_argument("--pause", type=float, default=DEFAULT_PAUSE)
    ap.add_argument("--dry-run", action="store_true", help="Yazmadan raporla")
    args = ap.parse_args()

    if not REFERANS.is_file():
        print(f"HATA: ilce referansi yok: {REFERANS}")
        return 1
    ilceler = pd.read_parquet(REFERANS)
    kopruler = _kopruleri_kur()

    mevcut: dict[str, pd.DataFrame] = {}
    for kopru in kopruler:
        yol = ROOT / kopru.yol
        if not yol.is_file():
            print(f"HATA: {kopru.ad} tablosu yok: {yol}")
            return 1
        frame = pd.read_parquet(yol)
        frame["tarih"] = pd.to_datetime(frame["tarih"]).dt.normalize()
        mevcut[kopru.ad] = frame

    bugun = pd.Timestamp.today().normalize()
    # past_days: EN ESKI arsiv ucunu bile ortusme payiyla yakalayacak kadar.
    en_eski_uc = min(f["tarih"].max() for f in mevcut.values())
    past_days = int((bugun - en_eski_uc).days) + ORTUSME_GUN
    if past_days > MAX_PAST_DAYS:
        print(
            f"HATA: en eski tablo ucu {en_eski_uc.date()}, bugunden {past_days} gun geride; "
            f"forecast API en fazla {MAX_PAST_DAYS} gun geriye veriyor. "
            "Once arsiv cekicilerini calistir."
        )
        return 1

    degiskenler = tum_degiskenler(kopruler)
    print(f"SAATLIK KOPRU  ({bugun.date()})")
    print("=" * 74)
    for kopru in kopruler:
        print(f"  {kopru.ad:22s} arsiv ucu {mevcut[kopru.ad]['tarih'].max().date()}")
    print(f"  pencere: bugun-{past_days} .. bugun+{args.gun} · {len(degiskenler)} degisken/istek")

    parcalar: dict[str, list[pd.DataFrame]] = {k.ad: [] for k in kopruler}
    basarisiz: list[str] = []
    for sira, satir in enumerate(ilceler.itertuples(index=False), start=1):
        anahtar = str(satir.ilce_key)
        print(f"[{sira:3d}/{len(ilceler)}] {anahtar:16s} ", end="", flush=True)
        try:
            saatlik = forecast_cek(
                anahtar,
                float(satir.lat),
                float(satir.lon),
                degiskenler,
                past_days=past_days,
                forecast_days=args.gun,
            )
        except RuntimeError as hata:
            print(f"BASARISIZ -- {hata}")
            basarisiz.append(anahtar)
            continue
        for kopru in kopruler:
            alt = saatlik[["zaman", *kopru.degiskenler]]
            parcalar[kopru.ad].append(kopru.topla(alt, anahtar))
        print(f"{len(saatlik):,} saat")
        time.sleep(args.pause)

    if basarisiz:
        print(f"\nHATA: {len(basarisiz)} ilce alinamadi: {basarisiz[:8]}")
        print("Eksik ilceli bir koprü paneli DELIK birakir; hicbir sey yazilmadi.")
        return 1

    hata_var = False
    for kopru in kopruler:
        tahmin = pd.concat(parcalar[kopru.ad], ignore_index=True)
        tahmin["tarih"] = pd.to_datetime(tahmin["tarih"]).dt.normalize()
        arsiv = mevcut[kopru.ad]

        fark = dikis_farki(arsiv, tahmin, kopru)
        durum = "GECTI" if fark <= kopru.dikis_toleransi else "RED  "
        print(
            f"\n  [{durum}] {kopru.ad}: dikis farki {kopru.dikis_kolonu}="
            f"{fark:.3f} (tolerans {kopru.dikis_toleransi})"
        )
        if fark > kopru.dikis_toleransi:
            print(
                "          Arsiv ve tahmin ortusen gunlerde AYRISIYOR. Koordinat "
                "kaymasi, birim degisikligi ya da baska bir degisken donmus olabilir. "
                "Bu tablo YAZILMADI."
            )
            hata_var = True
            continue

        birlesik = kopruyu_birlestir(arsiv, tahmin)
        eklenen = len(birlesik) - len(arsiv)
        print(
            f"          {eklenen:,} satir eklendi -> "
            f"{birlesik['tarih'].min().date()} .. {birlesik['tarih'].max().date()}"
        )
        if args.dry_run:
            continue
        atomic_write_dataframe(birlesik, ROOT / kopru.yol)

    print("\n" + "=" * 74)
    if hata_var:
        print("KOPRU EKSIK. Reddedilen tablolar icin dikis farkini incele.")
        return 1
    print("Tum koprüler kuruldu." + ("  (--dry-run: yazilmadi)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
