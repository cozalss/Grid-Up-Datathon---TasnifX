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
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Pakete girecek dizinler. ``data/`` altindaki her sey -- gizli kontrol
#: noktasi dizinleri DAHIL (pahali olan onlar).
KAYNAK_DIZINLER = ("data",)

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


def _tasinacaklar() -> list[Path]:
    """Pakete girecek dosyalarin listesi (koke gore gorece)."""
    secilen: list[Path] = []
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
    return secilen


def paketle(cikti: Path) -> int:
    dosyalar = _tasinacaklar()
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
    print("YENI MAKINEDE SIRASIYLA:")
    print("  1. git clone https://github.com/cozalss/Grid-Up-Datathon---TasnifX.git")
    print("  2. cd Grid-Up-Datathon---TasnifX")
    print("  3. python -m pip install --require-hashes -r requirements/uv-bootstrap.txt")
    print("  4. uv sync --locked --extra full --extra dev")
    print(f"  5. python scripts/tasima_paketi.py ac {cikti.name}")
    print("\nAYRICA ELLE KOPYALA:  .env")
    print("  Pakete BILEREK konmadi -- FIRMS anahtari ve EPIAS kimlik bilgisi")
    print("  tasir. Bir arsiv kazara paylasilirsa sizinti kalici olur.")
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

    p_ac = alt.add_parser("ac", help="Paketi ac ve dogrula")
    p_ac.add_argument("paket")

    p_dog = alt.add_parser("dogrula", help="Diskteki veriyi hash'lerle karsilastir")
    p_dog.add_argument("--paket", default=None)

    args = ap.parse_args()
    if args.komut == "paketle":
        return paketle(Path(args.cikti).resolve())
    if args.komut == "ac":
        return ac(Path(args.paket).resolve())
    return dogrula(Path(args.paket).resolve() if args.paket else None)


if __name__ == "__main__":
    raise SystemExit(main())
