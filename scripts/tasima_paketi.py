"""VERIYI BASKA MAKINEYE TASIR ve karsi tarafta DOGRULAR.

NEDEN BU BETIK
--------------
Kod GitHub'da; ``git clone`` + ``uv sync`` ile bir dakikada gelir. Ama
``data/`` .gitignore kapsamindadir ve depoda YOKTUR. Yani yeni makinede
kod calisir, veri yoktur ve her sey sessizce bos doner.

Tasinmasi gereken 341 MB'nin buyuk kismi da yeniden indirilemez degil ama
PAHALIDIR: ``data/external/.*_ckpt`` ve ``.hava_saatlik_ham`` dizinleri
Open-Meteo kotasiyla alinmis ham saatlik veridir. Kotayi yeniden yakmak
saatler surer ve gunluk limit dolarsa ERTESI GUNE kalir -- yarisma gunu
kabul edilemez. Bu yuzden paket kontrol noktalarini DA tasir.

ASIL MESELE: DOGRULAMA
----------------------
Yarim kopyalanmis bir parquet, okunurken degil MODEL EGITILIRKEN patlar --
ya da hic patlamaz, sessizce eksik satirla devam eder. Bu yuzden paketin
icine her dosyanin SHA-256'si yazilir ve karsi tarafta TEK TEK dogrulanir.
"Kopyalandi" ile "dogru kopyalandi" ayni sey degildir.

GIZLI BILGI PAKETE GIRMEZ
-------------------------
``.env`` (FIRMS anahtari, EPIAS kimlik bilgileri) BILEREK pakete konmaz.
Bir arsiv kazara buluta yuklenebilir, paylasilabilir, yedeklenebilir; icinde
kimlik bilgisi varsa sizinti kalicidir. Betik ``.env``i ayrica kopyalamani
soyler -- tek dosya, elle tasinir.

KULLANIM
--------
Bu makinede::

    python scripts/tasima_paketi.py paketle
    python scripts/tasima_paketi.py paketle --cikti D:/datahon_veri.zip

Yeni makinede (once git clone + uv sync)::

    python scripts/tasima_paketi.py ac datahon_veri.zip
    python scripts/tasima_paketi.py dogrula      # sonradan tekrar kontrol

Cikis kodu: 0 = tamam, 1 = eksik/bozuk dosya var.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Pakete girecek dizinler. Ikisi de .gitignore kapsamindadir, yani
#: ``git clone`` onlari GETIRMEZ:
#:   data/        -- yarisma verisi, harici veri, kontrol noktalari
#:   submissions/ -- uretilmis gonderim dosyalari (herbiri ~27 MB)
KAYNAK_DIZINLER = ("data", "submissions")

#: ``--yalin`` modda dislanan, YENIDEN URETILEBILIR yollar.
#: Dislanan her sey ciktida ACIKCA listelenir -- sessiz kirpma yok.
YALIN_DISLANAN = ("data/interim",)

#: ``--yalin`` modda saklanacak en yeni gonderim sayisi. Eskiler yeniden
#: uretilebilir; yenisi LB karsilastirmasi icin gerekir.
YALIN_GONDERIM_SAYISI = 4

#: Pakete ASLA girmeyecekler. ``.env`` gizli bilgi tasir; digerleri ya
#: yeniden uretilir ya da devasa.
DISLANAN_PARCALAR = (".venv", ".git", "__pycache__", ".pytest_cache", ".mypy_cache")
DISLANAN_ADLAR = (".env",)

#: Paketin icindeki dogrulama dosyasi.
MANIFEST_ADI = "TASIMA.json"

VARSAYILAN_CIKTI = "datahon_veri.zip"


def _dosya_hash(yol: Path) -> str:
    ozet = hashlib.sha256()
    with yol.open("rb") as akis:
        for parca in iter(lambda: akis.read(1024 * 1024), b""):
            ozet.update(parca)
    return ozet.hexdigest()


def _tasinacaklar(yalin: bool = False) -> tuple[list[Path], list[str]]:
    """Pakete girecek dosyalar ve DISLANANLARIN gerekceleri.

    Dislananlar ayrica dondurulur cunku sessiz kirpma, "her sey tasindi"
    izlenimi verir ve karsi tarafta eksik fark edilmez.
    """
    secilen: list[Path] = []
    notlar: list[str] = []
    for dizin in KAYNAK_DIZINLER:
        kok = ROOT / dizin
        if not kok.is_dir():
            continue
        for yol in sorted(kok.rglob("*")):
            if not yol.is_file():
                continue
            if yol.name in DISLANAN_ADLAR:
                continue
            if any(parca in yol.parts for parca in DISLANAN_PARCALAR):
                continue
            secilen.append(yol.relative_to(ROOT))

    if not yalin:
        return secilen, notlar

    onceki = len(secilen)
    boyut = lambda ys: sum((ROOT / y).stat().st_size for y in ys) / 1024 / 1024  # noqa: E731
    atilan = [y for y in secilen if any(y.as_posix().startswith(d) for d in YALIN_DISLANAN)]
    if atilan:
        notlar.append(
            f"{len(atilan)} dosya ({boyut(atilan):.0f} MB) DISLANDI: {', '.join(YALIN_DISLANAN)}"
            " -- deney tezgahi bunlari raw+external'dan yeniden uretir."
        )
    secilen = [y for y in secilen if y not in set(atilan)]

    gonderimler = sorted(
        (y for y in secilen if y.as_posix().startswith("submissions/")),
        key=lambda y: (ROOT / y).stat().st_mtime,
        reverse=True,
    )
    eski_gonderim = gonderimler[YALIN_GONDERIM_SAYISI:]
    if eski_gonderim:
        notlar.append(
            f"{len(eski_gonderim)} eski gonderim ({boyut(eski_gonderim):.0f} MB) DISLANDI; "
            f"en yeni {YALIN_GONDERIM_SAYISI} tanesi tasiniyor: "
            + ", ".join(y.name for y in gonderimler[:YALIN_GONDERIM_SAYISI])
        )
    secilen = [y for y in secilen if y not in set(eski_gonderim)]
    notlar.append(f"toplam {onceki - len(secilen)} dosya dislandi.")
    return secilen, notlar


def paketle(cikti: Path, yalin: bool = False) -> int:
    dosyalar, notlar = _tasinacaklar(yalin)
    for not_ in notlar:
        print(f"  NOT: {not_}")
    if not dosyalar:
        print("HATA: tasinacak dosya bulunamadi. data/ dizini bos mu?")
        return 1

    print(f"TASIMA PAKETI  ->  {cikti}")
    print("=" * 70)
    toplam_bayt = sum((ROOT / y).stat().st_size for y in dosyalar)
    print(f"  {len(dosyalar):,} dosya · {toplam_bayt / 1024 / 1024:.0f} MB")

    kayit: dict[str, dict[str, object]] = {}
    cikti.parent.mkdir(parents=True, exist_ok=True)
    # Parquet zaten sikistirilmis; seviye 1 zamandan kazandirir, boyuttan
    # kaybettirmez.
    with zipfile.ZipFile(cikti, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as arsiv:
        for sira, gorece in enumerate(dosyalar, start=1):
            tam = ROOT / gorece
            anahtar = gorece.as_posix()
            kayit[anahtar] = {"sha256": _dosya_hash(tam), "bayt": tam.stat().st_size}
            arsiv.write(tam, anahtar)
            if sira % 50 == 0 or sira == len(dosyalar):
                print(f"    {sira:,}/{len(dosyalar):,}", end="\r", flush=True)
        arsiv.writestr(
            MANIFEST_ADI,
            json.dumps({"dosyalar": kayit, "toplam_bayt": toplam_bayt}, indent=2),
        )

    boyut = cikti.stat().st_size / 1024 / 1024
    print(f"\n\nYazildi: {cikti}  ({boyut:.0f} MB)")
    print("\n" + "=" * 70)
    # Kurucuyu zip'in YANINA koy. Boylece hedef makinede tek komut yeter:
    # kur.py depoyu klonlar, ortami kurar, bu zip'i bulup acar ve dogrular.
    kurucu = ROOT / "kur.py"
    if kurucu.is_file():
        hedef = cikti.parent / "kur.py"
        shutil.copy2(kurucu, hedef)
        print(f"Yazildi: {hedef}")

    print("=" * 70)
    print("YENI MAKINEDE TEK KOMUT:")
    print("\n    python kur.py\n")
    print(f"  ({cikti.parent} icindeki kur.py'yi calistir -- zip'i kendisi bulur)")
    print("\nUSB'ye/buluta su UC dosyayi koy:")
    print(f"  kur.py  ·  {cikti.name}  ·  .env")
    print("\n.env pakete BILEREK konmadi -- FIRMS anahtari ve EPIAS kimlik")
    print("  bilgisi tasir. Bir arsiv kazara paylasilirsa sizinti kalici olur.")
    print("  kur.py onu yaninda bulursa depoya kendisi yerlestirir.")
    return 0


def ac(arsiv_yolu: Path) -> int:
    if not arsiv_yolu.is_file():
        print(f"HATA: paket yok: {arsiv_yolu}")
        return 1
    print(f"PAKET ACILIYOR  <-  {arsiv_yolu}")
    print("=" * 70)
    with zipfile.ZipFile(arsiv_yolu) as arsiv:
        if MANIFEST_ADI not in arsiv.namelist():
            print(f"HATA: pakette {MANIFEST_ADI} yok -- bu betikle uretilmemis.")
            return 1
        arsiv.extractall(ROOT)
    print("  acildi.\n")
    return dogrula(arsiv_yolu)


def dogrula(arsiv_yolu: Path | None = None) -> int:
    """Diskteki dosyalari pakete yazilan hash'lerle karsilastirir."""
    kayit: dict[str, dict[str, object]] | None = None
    if arsiv_yolu and arsiv_yolu.is_file():
        with zipfile.ZipFile(arsiv_yolu) as arsiv:
            kayit = json.loads(arsiv.read(MANIFEST_ADI))["dosyalar"]
    else:
        yerel = ROOT / MANIFEST_ADI
        if yerel.is_file():
            kayit = json.loads(yerel.read_text(encoding="utf-8"))["dosyalar"]
    if kayit is None:
        print(f"HATA: dogrulama kaydi bulunamadi ({MANIFEST_ADI}).")
        return 1

    print(f"DOGRULAMA  ({len(kayit):,} dosya)")
    print("=" * 70)
    eksik: list[str] = []
    bozuk: list[str] = []
    for sira, (gorece, beklenen) in enumerate(sorted(kayit.items()), start=1):
        yol = ROOT / gorece
        if not yol.is_file():
            eksik.append(gorece)
        elif yol.stat().st_size != beklenen["bayt"]:
            bozuk.append(f"{gorece} (boyut)")
        elif _dosya_hash(yol) != beklenen["sha256"]:
            bozuk.append(f"{gorece} (sha256)")
        if sira % 50 == 0 or sira == len(kayit):
            print(f"    {sira:,}/{len(kayit):,}", end="\r", flush=True)

    print()
    if eksik:
        print(f"\nHATA -- {len(eksik)} dosya EKSIK:")
        for ad in eksik[:10]:
            print(f"  {ad}")
        if len(eksik) > 10:
            print(f"  ... ve {len(eksik) - 10} tane daha")
    if bozuk:
        print(f"\nHATA -- {len(bozuk)} dosya BOZUK:")
        for ad in bozuk[:10]:
            print(f"  {ad}")

    print("\n" + "=" * 70)
    if eksik or bozuk:
        print("TASIMA EKSIK. Paketi yeniden kopyala -- yarim veri sessizce yanlis")
        print("sonuc uretir, hata vermez.")
        return 1

    print(f"{len(kayit):,} dosyanin hepsi birebir dogru.")
    print("\nSIRADAKI ADIM -- kendi kapilarini kos:")
    print("  python scripts/veri_sagligi.py")
    print("  python scripts/kapsam_deseni.py")
    print("  python -m pytest -q")
    print("\n.env kopyalandi mi? Yoksa EPIAS ve FIRMS cekicileri calismaz.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    alt = ap.add_subparsers(dest="komut", required=True)

    p_paket = alt.add_parser("paketle", help="Bu makinedeki veriyi paketle")
    p_paket.add_argument("--cikti", default=VARSAYILAN_CIKTI)
    p_paket.add_argument(
        "--yalin",
        action="store_true",
        help="Yeniden uretilebilirleri disla (data/interim + eski gonderimler)",
    )

    p_ac = alt.add_parser("ac", help="Paketi ac ve dogrula")
    p_ac.add_argument("paket")

    p_dog = alt.add_parser("dogrula", help="Diskteki veriyi hash'lerle karsilastir")
    p_dog.add_argument("--paket", default=None)

    args = ap.parse_args()
    if args.komut == "paketle":
        return paketle(Path(args.cikti).resolve(), yalin=args.yalin)
    if args.komut == "ac":
        return ac(Path(args.paket).resolve())
    return dogrula(Path(args.paket).resolve() if args.paket else None)


if __name__ == "__main__":
    raise SystemExit(main())
