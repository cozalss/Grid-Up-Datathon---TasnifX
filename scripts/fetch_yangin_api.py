"""FIRMS Area API ile 2025-2026 yangin boslugunu kapatir.

NEDEN AYRI BIR BETIK
--------------------
``fetch_yangin.py`` FIRMS'in YILLIK ULKE ARSIVINI kullanir
(``{sensor}_{yil}_Turkey.csv``). O dosyalar 2020-2024 icin yayimlanmis,
**2025 ve 2026 icin YAYIMLANMAMIS** -- ikisi de HTTP 404 doner (olculdu).
Yani ``data/external/yanginlar.parquet`` 2024-12-31'de biter.

Bu, veri eksikliginden daha kotudur: egitim verisi 2026'ya uzaniyorsa
yangin feature'i egitimde DOLU, testte TAMAMEN NaN olur. Modelin egitimde
guvendigi bir sinyal test aninda kaybolur -- klasik dagilim kaymasi.

Bosluk yalnizca **Area API** ile kapatilir ve o da MAP_KEY ister.

ISLEME TURLERI -- OLCULEREK SECILDI
-----------------------------------
FIRMS ayni sensoru iki islemede sunar:

  _SP  (Standard Processing)  arsiv urunu, ~2-3 ay GECIKMELI
  _NRT (Near Real-Time)       yakin donem, gecmise gitmez

Olculdu (2026-08-17, Ege kutusu):

  VIIRS_SNPP_SP   2025-07-15 -> 102 satir    2026-04-10 -> 31 satir
                  2026-06-10 -> 0 satir      (arsiv sinirI ~Nisan 2026)
  VIIRS_SNPP_NRT  2025-07-15 -> 0 satir      (gecmise gitmiyor)
                  2026-05-01 -> 17 satir     2026-08-15 -> 54 satir

Yani ikisi BIRLIKTE boslugu kapatir: _SP eski tarafi, _NRT yakin tarafi.
Betik sinirI kendisi bulur -- sabit tarih varsaymaz, cunku sinir her ay
ilerler.

KOTA
----
5000 islem / 10 dakika. Istek basina en fazla 5 gun. 2025-01-01..bugun
araligi icin ~120 pencere x 2 sensor ~= 240 istek -- rahat sigar.

KULLANIM
--------
::

    python scripts/fetch_yangin_api.py
    python scripts/fetch_yangin_api.py --start 2025-01-01 --end 2026-08-15
    python scripts/fetch_yangin_api.py --birlestir   # eski parquet ile birlestir

Anahtar ``.env`` icindeki ``FIRMS_MAP_KEY``ten okunur ve hicbir ciktiya
yazdirilmaz.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gridup.io_utils import atomic_write_dataframe  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MEVCUT = ROOT / "data" / "external" / "yanginlar.parquet"
CIKTI = ROOT / "data" / "external" / "yanginlar.parquet"

BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

#: Ege kutusu: bes ilimizi kapsar. Turkiye'nin tamamini indirmek kotayi
#: bosuna yakar ve panele girmeyecek satir uretir.
LON_MIN, LAT_MIN, LON_MAX, LAT_MAX = 26.0, 36.0, 30.5, 39.5
KUTU = f"{LON_MIN},{LAT_MIN},{LON_MAX},{LAT_MAX}"

#: API'nin izin verdigi en genis pencere.
PENCERE_GUN = 5

#: (API kaynak adi, cikti aygit etiketi). Etiketler MEVCUT parquet ile
#: birebir ayni olmali, aksi halde birlestirmede iki farkli "VIIRS" olusur.
KAYNAKLAR: tuple[tuple[str, str, str], ...] = (
    ("MODIS_SP", "MODIS_NRT", "MODIS"),
    ("VIIRS_SNPP_SP", "VIIRS_SNPP_NRT", "VIIRS-SNPP"),
)

RETRIES = 3


def anahtari_oku() -> str:
    """``.env`` veya ortamdan MAP_KEY okur. Degeri ASLA yazdirmaz."""
    anahtar = os.environ.get("FIRMS_MAP_KEY", "").strip()
    if not anahtar:
        env = ROOT / ".env"
        if env.is_file():
            eslesme = re.search(
                r"^FIRMS_MAP_KEY=(.*)$", env.read_text(encoding="utf-8", errors="replace"), re.M
            )
            if eslesme:
                anahtar = eslesme.group(1).strip()
    if not anahtar:
        raise SystemExit(
            "FIRMS_MAP_KEY bulunamadi.\n"
            "  1. https://firms.modaps.eosdis.nasa.gov/api/map_key/ adresinden al\n"
            "  2. .env icindeki FIRMS_MAP_KEY= satirina yapistir"
        )
    return anahtar


def _cek(anahtar: str, kaynak: str, baslangic: date) -> pd.DataFrame | None:
    """Tek pencere ceker. Bos yanit ``None`` degil BOS FRAME'dir -- ikisi farkli.

    ``None`` yalnizca "bu kaynak bu tarihte HIC veri sunmuyor" demektir ve
    sinir tespitinde kullanilir; bos frame "sundu ama yangin yoktu" demektir.
    """
    url = f"{BASE}/{anahtar}/{kaynak}/{KUTU}/{PENCERE_GUN}/{baslangic:%Y-%m-%d}"
    son_hata: Exception | None = None
    for deneme in range(1, RETRIES + 1):
        try:
            yanit = requests.get(url, timeout=120)
            govde = yanit.text.strip()
            if yanit.status_code == 429:
                bekle = 20 * deneme
                print(f"    kota/hiz siniri; {bekle} sn bekleniyor")
                time.sleep(bekle)
                son_hata = requests.HTTPError("429")
                continue
            yanit.raise_for_status()
            if govde.lower().startswith(("invalid", "error")):
                # Anahtar/parametre hatasi -- yeniden denemek anlamsiz.
                raise RuntimeError(f"API reddetti: {govde[:160]}")
            if not govde or "\n" not in govde:
                return pd.DataFrame()
            return pd.read_csv(io.StringIO(govde))
        except (requests.RequestException, ValueError) as hata:
            son_hata = hata
            if deneme < RETRIES:
                time.sleep(2**deneme)
    raise RuntimeError(f"{kaynak} {baslangic}: {RETRIES} denemede alinamadi ({son_hata})")


def _sadelestir(ham: pd.DataFrame, aygit: str) -> pd.DataFrame:
    """Ham FIRMS kolonlarini MEVCUT parquet semasina indirger.

    ``confidence`` MODIS'te 0-100 sayi, VIIRS'te l/n/h harfidir; ``fetch_yangin``
    ile ayni sekilde METIN olarak saklanir -- iki olcegi tek sayiya zorlamak
    bilgi uydurmak olurdu.
    """
    if ham.empty:
        return pd.DataFrame(columns=["tarih", "lat", "lon", "frp", "guven", "aygit"])
    eksik = {"latitude", "longitude", "acq_date"} - set(ham.columns)
    if eksik:
        raise RuntimeError(f"FIRMS yaniti beklenen kolonlari tasimiyor, eksik: {sorted(eksik)}")
    kutu = ham[
        ham["latitude"].between(LAT_MIN, LAT_MAX) & ham["longitude"].between(LON_MIN, LON_MAX)
    ]
    return pd.DataFrame(
        {
            "tarih": pd.to_datetime(kutu["acq_date"]).dt.date,
            "lat": kutu["latitude"].astype("float64"),
            "lon": kutu["longitude"].astype("float64"),
            "frp": pd.to_numeric(kutu.get("frp"), errors="coerce"),
            "guven": kutu.get("confidence", pd.Series("", index=kutu.index)).astype(str),
            "aygit": aygit,
        }
    )


def kaynagi_tara(anahtar: str, sp: str, nrt: str, aygit: str, bas: date, son: date) -> pd.DataFrame:
    """Bir sensorun tum araligini _SP + _NRT ile tarar.

    Once _SP denenir; bos donmeye BASLADIGI noktadan sonra _NRT'ye gecilir.
    Sinir sabit varsayilmaz cunku Standard Processing her ay ilerler.
    """
    parcalar: list[pd.DataFrame] = []
    sp_bitti = False
    pencere = bas
    toplam = 0
    while pencere <= son:
        kaynak = nrt if sp_bitti else sp
        ham = _cek(anahtar, kaynak, pencere)
        satir = 0 if ham is None or ham.empty else len(ham)

        # _SP arsiv sinirina ulasildi mi? Ard arda bos pencere + ileri tarih
        # sinyali: NRT'ye gec ve AYNI pencereyi tekrar dene.
        if not sp_bitti and satir == 0 and pencere > date(2026, 1, 1):
            deneme = _cek(anahtar, nrt, pencere)
            if deneme is not None and len(deneme) > 0:
                print(f"    {aygit}: _SP arsiv siniri {pencere} -- _NRT'ye gecildi")
                sp_bitti = True
                ham, satir = deneme, len(deneme)

        if satir:
            parcalar.append(_sadelestir(ham, aygit))
            toplam += satir
        pencere += timedelta(days=PENCERE_GUN)
        time.sleep(0.15)

    print(f"  {aygit:12s} {toplam:6,d} ham tespit")
    if not parcalar:
        return pd.DataFrame(columns=["tarih", "lat", "lon", "frp", "guven", "aygit"])
    return pd.concat(parcalar, ignore_index=True)


def kalite_kapisi(yeni: pd.DataFrame, bas: date, son: date) -> None:
    """Kabul edilemez sonucu YAZMADAN ONCE reddeder."""
    if yeni.empty:
        raise ValueError("Kalite kapisi: hicbir tespit alinamadi. Parquet YAZILMADI.")
    kapsam = pd.to_datetime(yeni["tarih"])
    ilk, sonuncu = kapsam.min().date(), kapsam.max().date()
    print(f"  kapsam: {ilk} .. {sonuncu} ({len(yeni):,} tespit)")

    # Istenen araligin sonuna makul olcude yaklasilmali; aksi halde bosluk
    # kapanmamis demektir ve bunu sessizce kabul etmek en bastaki hatayi
    # tekrar etmek olur.
    acik = (son - sonuncu).days
    if acik > 30:
        raise ValueError(
            f"Kalite kapisi: veri {sonuncu}'de bitiyor, istenen bitis {son} "
            f"({acik} gun acik). Bosluk kapanmadi; Parquet YAZILMADI."
        )
    if (ilk - bas).days > 30:
        raise ValueError(
            f"Kalite kapisi: veri {ilk}'de basliyor, istenen baslangic {bas}. Parquet YAZILMADI."
        )
    for kolon in ("lat", "lon"):
        if yeni[kolon].isna().any():
            raise ValueError(f"Kalite kapisi: '{kolon}' icinde NaN var.")


def main() -> int:
    ayristirici = argparse.ArgumentParser(description=__doc__)
    ayristirici.add_argument("--start", default="2025-01-01", help="Baslangic (YYYY-AA-GG)")
    ayristirici.add_argument("--end", default=None, help="Bitis; varsayilan bugun")
    ayristirici.add_argument(
        "--birlestir",
        action="store_true",
        help="Mevcut yanginlar.parquet ile birlestir (varsayilan: yalnizca yeni araligi yaz)",
    )
    ayristirici.add_argument("--out", default=str(CIKTI))
    args = ayristirici.parse_args()

    anahtar = anahtari_oku()
    bas = datetime.strptime(args.start, "%Y-%m-%d").date()
    son = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else date.today()
    pencere_sayisi = ((son - bas).days // PENCERE_GUN + 1) * len(KAYNAKLAR)
    print(f"FIRMS Area API: {bas} .. {son} | ~{pencere_sayisi} istek | kutu {KUTU}\n")

    parcalar = [kaynagi_tara(anahtar, sp, nrt, aygit, bas, son) for sp, nrt, aygit in KAYNAKLAR]
    yeni = pd.concat(parcalar, ignore_index=True)
    yeni = yeni.drop_duplicates(subset=["tarih", "lat", "lon", "aygit"])

    print("\nYeni aralik:")
    kalite_kapisi(yeni, bas, son)

    cikti = yeni
    if args.birlestir and MEVCUT.is_file():
        eski = pd.read_parquet(MEVCUT)
        onceki = len(eski)
        cikti = pd.concat([eski, yeni], ignore_index=True)
        cikti = cikti.drop_duplicates(subset=["tarih", "lat", "lon", "aygit"])
        cikti = cikti.sort_values("tarih").reset_index(drop=True)
        print(
            f"\nBirlestirildi: {onceki:,} (eski) + {len(yeni):,} (yeni) -> {len(cikti):,} benzersiz"
        )

    atomic_write_dataframe(cikti, Path(args.out))
    print(f"\nYazildi: {args.out}")
    kap = pd.to_datetime(cikti["tarih"])
    print(f"  {len(cikti):,} tespit | {kap.min().date()} .. {kap.max().date()}")
    print("\n  Kaynak: NASA FIRMS (MODIS/VIIRS). Notebook'ta atif ver.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
