"""TUM HARICI VERIYI DOGRU SIRAYLA TAZELER -- tek komut, tek dogru sira.

NEDEN BU BETIK
--------------
Harici veri tazelemenin bir SIRASI vardir ve sirayi bozmak paneli sessizce
bozar:

    arsiv cekicileri  ->  koprüler  ->  kapilar

Cekiciler tabloyu kontrol noktalarindan YENIDEN URETIR; bu, onceki koprü
kosusunun ekledigi tahmin satirlarini siler. Yani "nem tablosunu tazeleyeyim"
demek, farkinda olmadan panelin ileri ucunu delmek demektir -- ve o delik
yalnizca ``veri_sagligi.py`` calistirilirsa gorunur.

Yarisma gunu bu tam olarak yapilacak hatadir: veri tazelenir, kimse koprüyu
yeniden kurmaz, panelin son gunlerinde bazi aileler bos kalir ve CV bunu
GORMEZ (cunku CV de ayni bos veriyle kosar).

Bu betik siranin dogru olmasini garanti eder ve sonunda kapilari kosar.

KOTA GERCEGI (olculdu 2026-08-20)
---------------------------------
Open-Meteo uc ayri kota penceresi isletir ve arsiv istekleri agirdir::

    archive + historical-forecast  -> AYNI kota (saatlik limit ~20 ilce)
    forecast                       -> ayri
    air-quality                    -> ayri

Yani arsiv adimlari SAATLER surer (96 ilce x 6,5 yil saatlik veri, saatlik
limit dolunca bir sonraki saat basina kadar bekleme). Koprü adimlari ise
dakikalar surer. ``--yalniz-kopru`` bu yuzden vardir: veri zaten tazeyse
saatlerce beklemeden yalnizca ileri ucu yenilemek icin.

KULLANIM
--------
::

    python scripts/veri_tazele.py                 # tam tazeleme (SAATLER)
    python scripts/veri_tazele.py --yalniz-kopru  # yalnizca ileri uc (dakikalar)
    python scripts/veri_tazele.py --kuru          # ne calisacagini yazdir, calistirma

Cikis kodu: 0 = tum adimlar ve kapilar gecti, 1 = en az biri basarisiz.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


@dataclass(frozen=True)
class Adim:
    """Tazeleme hattinin bir adimi."""

    ad: str
    komut: list[str]
    #: ``True`` ise basarisizlik hatti DURDURUR. Kapilar ve koprüler boyledir.
    #: ``False`` ise uyarilir ve devam edilir -- tek bir kaynagin gecici
    #: erisilemezligi (or. EPIAS 503) tum tazelemeyi iptal etmemeli.
    zorunlu: bool
    #: Yalnizca ileri ucu yenileyen hizli modda da kosulsun mu?
    kopru_modunda: bool = False
    aciklama: str = ""


def _dun() -> str:
    """Arsivin guvenle isteyebilecegi en son gun (bkz. ARCHIVE_LAG_DAYS)."""
    return (date.today() - timedelta(days=1)).isoformat()


def hat(bugune_kadar: str) -> tuple[Adim, ...]:
    """Tazeleme hatti -- SIRA ONEMLIDIR, degistirme."""
    return (
        # --- 1. ARSIV. Agir ve yavas; tablolari kontrol noktalarindan
        #     yeniden uretir ve onceki koprü satirlarini siler.
        Adim(
            "hava_gunluk (arsiv)",
            [
                PY,
                "scripts/fetch_weather.py",
                "--all-districts",
                "--start",
                "2020-01-01",
                "--end",
                bugune_kadar,
            ],
            zorunlu=False,
            aciklama="ERA5 gunluk; saatlik kotaya takilir, saatler surebilir",
        ),
        Adim(
            "hava_saatlik_turev (arsiv)",
            [PY, "scripts/fetch_hourly_weather.py", "--start", "2020-01-01", "--end", bugune_kadar],
            zorunlu=False,
            aciklama="ham saatlik kontrol noktalarindan; eksikse indirir",
        ),
        Adim(
            "nem_toprak (arsiv)",
            [PY, "scripts/fetch_nem_toprak.py", "--start", "2020-01-01", "--end", bugune_kadar],
            zorunlu=False,
        ),
        Adim(
            "konvektif (arsiv)",
            [PY, "scripts/fetch_konvektif.py", "--start", "2020-01-01", "--end", bugune_kadar],
            zorunlu=False,
        ),
        Adim(
            "hava_kalitesi (arsiv)",
            [PY, "scripts/fetch_hava_kalitesi.py", "--start", "2020-01-01", "--end", bugune_kadar],
            zorunlu=False,
            aciklama="ayri kotada -- arsiv tikansa bile calisir",
        ),
        Adim(
            "epias (ulusal yuk)",
            [PY, "scripts/fetch_epias_load.py", "--start", "2020-01-01", "--end", bugune_kadar],
            zorunlu=False,
            aciklama="kimlik dogrulama gerekir; panel ufkunu BU kaynak sinirlar",
        ),
        # --- 2. KOPRULER. Arsivden SONRA kosmak ZORUNDA: cekiciler tabloyu
        #     yeniden uretip onceki tahmin kuyrugunu siler.
        Adim(
            "koprü: gunluk hava",
            [PY, "scripts/fetch_weather_bridge.py", "--forecast-days", "7"],
            zorunlu=True,
            kopru_modunda=True,
            aciklama="+7: hava kalitesinin tavani (panel en zayif kaynagi kadar uzar)",
        ),
        Adim(
            "koprü: saatlik/konvektif/nem/hava kalitesi",
            [PY, "scripts/kopru_saatlik.py", "--gun", "7"],
            zorunlu=True,
            kopru_modunda=True,
        ),
        # --- 3. KAPILAR. Tazelemenin gecerli sayilmasi bunlara bagli.
        Adim(
            "kapi: veri sagligi",
            [PY, "scripts/veri_sagligi.py"],
            zorunlu=True,
            kopru_modunda=True,
            aciklama="kapsam, butunluk, fizik, panel hizalamasi ve tazeligi",
        ),
        Adim(
            "kapi: kapsam deseni",
            [PY, "scripts/kapsam_deseni.py"],
            zorunlu=True,
            kopru_modunda=True,
            aciklama="feature egitimde dolu / testte bos olamaz",
        ),
        Adim(
            "kapi: manifest tazeleme",
            [PY, "scripts/manifest_tazele.py"],
            zorunlu=True,
            kopru_modunda=True,
            aciklama="olculen alanlar (sha256, min_rows) yenilenir",
        ),
        Adim(
            "kapi: koken dogrulama",
            [PY, "security/verify_sources.py", "--manifest", "data/sources.yml"],
            zorunlu=True,
            kopru_modunda=True,
        ),
    )


def calistir(adim: Adim) -> int:
    """Adimi kosar, ciktisini AKTARIR ve cikis kodunu doner."""
    print(f"\n{'=' * 74}\n>>> {adim.ad}")
    if adim.aciklama:
        print(f"    ({adim.aciklama})")
    print(f"    $ {' '.join(adim.komut)}\n")
    basladi = time.monotonic()
    sonuc = subprocess.run(adim.komut, cwd=ROOT, check=False)  # noqa: S603
    sure = time.monotonic() - basladi
    durum = "TAMAM" if sonuc.returncode == 0 else f"BASARISIZ (exit {sonuc.returncode})"
    print(f"\n<<< {adim.ad}: {durum}  [{sure / 60:.1f} dk]")
    return int(sonuc.returncode)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--yalniz-kopru",
        action="store_true",
        help="Arsiv cekicilerini ATLA; yalnizca koprüleri ve kapilari kos (dakikalar)",
    )
    ap.add_argument("--kuru", action="store_true", help="Calistirma, yalnizca sirayi yazdir")
    ap.add_argument("--end", default=None, help="Arsiv bitisi (varsayilan: dun)")
    args = ap.parse_args()

    adimlar = hat(args.end or _dun())
    if args.yalniz_kopru:
        adimlar = tuple(a for a in adimlar if a.kopru_modunda)

    print("HARICI VERI TAZELEME")
    print("=" * 74)
    print(f"  {len(adimlar)} adim" + ("  (yalniz koprü modu)" if args.yalniz_kopru else ""))
    for sira, adim in enumerate(adimlar, start=1):
        isaret = "!" if adim.zorunlu else " "
        print(f"  {sira:2d}.{isaret} {adim.ad}")
    print("\n  ! = basarisizligi hatti DURDURUR")

    if args.kuru:
        print("\n--kuru: hicbir sey calistirilmadi.")
        return 0

    basarisizlar: list[str] = []
    for adim in adimlar:
        kod = calistir(adim)
        if kod == 0:
            continue
        basarisizlar.append(adim.ad)
        if adim.zorunlu:
            print(
                f"\nZORUNLU adim basarisiz: {adim.ad}. Hat DURDURULDU.\n"
                "Sonraki adimlar bu adimin ciktisina dayaniyor; devam etmek "
                "yarim bir panel uretirdi."
            )
            return 1
        print(f"\n  UYARI: {adim.ad} basarisiz ama zorunlu degil -- devam ediliyor.")

    print("\n" + "=" * 74)
    if basarisizlar:
        print(f"Tazeleme bitti, {len(basarisizlar)} zorunlu-olmayan adim basarisiz:")
        for ad in basarisizlar:
            print(f"  - {ad}")
        print("Kapilar gecti, ama bu kaynaklar TAZELENMEDI. Sebebini incele.")
        return 1
    print("Tum adimlar ve kapilar gecti.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
