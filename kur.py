"""TEK KOMUTLA KURULUM -- depoyu klonlar, ortami kurar, veriyi acar, dogrular.

    python kur.py

BU BETIK TEK BASINA CALISIR
---------------------------
Depodan HICBIR SEY import etmez ve yalnizca standart kutuphaneyi kullanir --
cunku calistigi anda depo henuz yoktur. ``scripts/tasima_paketi.py paketle``
bu dosyayi zip'in yanina kopyalar; USB'de su uc sey bulunur::

    kur.py            <- bunu calistir
    datahon_*.zip     <- veri paketi (otomatik bulunur)
    .env              <- FIRMS/EPIAS kimlikleri (varsa yerlestirilir)
    kaggle.json       <- Kaggle API belirteci (varsa ~/.kaggle/ altina konur)

NEDEN AYRI BIR BETIK
--------------------
Kurulum alti adimdi ve her adim ayri bir hata yuzeyi: yanlis Python surumu,
uv bulunamamasi, zip'in yanlis dizine acilmasi, ``.env``in unutulmasi,
kapilarin hic kosulmamasi. Alti adimin besini dogru yapip altinciyi atlamak,
"kurulum bitti" sanip eksik veriyle calismak demektir.

Bu betik altisini da yapar ve HER BIRINI dogrular. Yarim biten bir adim
sonrakini baslatmaz.

TEKRAR CALISTIRILABILIR
-----------------------
Yarida kalirsa yeniden calistir. Klonlanmis depo, kurulu ortam ve acilmis
veri yeniden yapilmaz; kaldigi yerden devam eder.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEPO_URL = "https://github.com/cozalss/Grid-Up-Datathon---TasnifX.git"
DEPO_DIZIN = "Grid-Up-Datathon---TasnifX"

#: Kod bu surumlerde sinandi. Altindaki surumler paketleri kuramaz.
ASGARI_PYTHON = (3, 11)

#: Sanal ortamin SABITLENDIGI surum.
#:
#: NEDEN SABIT: ``uv sync`` surum verilmezse makinedeki EN YENI Python'u
#: secer. Prova kurulumunda (2026-08-22) 3.13.15 secti -- oysa CI matrisi
#: 3.11 ve 3.12'yi kosuyor ve gelistirme makinesi 3.12. Yani kurulum,
#: HIC SINANMAMIS bir yorumlayicida ortam kurmus olurdu.
#:
#: Bu sinsi bir hata sinifidir: paketler kurulur, kod calisir, testler bile
#: gecebilir -- ama sayisal davranis (numpy/pandas surum grafigi) farkli
#: olabilir ve fark yalnizca skorlarda gorunur.
HEDEF_PYTHON = "3.12"

BURASI = Path(__file__).resolve().parent


def basla(mesaj: str) -> None:
    print(f"\n{'=' * 68}\n>>> {mesaj}\n{'=' * 68}")


def kos(komut: list[str], *, cwd: Path | None = None, zorunlu: bool = True) -> int:
    """Komutu calistirir, ciktisini AKTARIR. Zorunluysa hatada durdurur."""
    print(f"    $ {' '.join(str(k) for k in komut)}\n", flush=True)
    sonuc = subprocess.run(komut, cwd=cwd, check=False)  # noqa: S603
    if sonuc.returncode != 0 and zorunlu:
        print(f"\n!! ADIM BASARISIZ (exit {sonuc.returncode}): {' '.join(komut[:3])}")
        print("   Yukaridaki hata mesajini oku; sonraki adimlar buna dayaniyor.")
        raise SystemExit(sonuc.returncode)
    return int(sonuc.returncode)


def komut_var_mi(ad: str) -> bool:
    return shutil.which(ad) is not None


def venv_python(depo: Path) -> Path:
    """Sanal ortamin Python'u -- Windows'ta Scripts/, digerlerinde bin/."""
    return (
        depo
        / ".venv"
        / ("Scripts" if os.name == "nt" else "bin")
        / ("python.exe" if os.name == "nt" else "python")
    )


def paketi_bul() -> Path | None:
    """Bu betigin YANINDAKI veri paketini bulur."""
    adaylar = sorted(BURASI.glob("datahon*.zip"), key=lambda p: p.stat().st_size, reverse=True)
    return adaylar[0] if adaylar else None


def on_kosullar() -> None:
    basla("1/6  On kosullar")
    surum = sys.version_info
    print(f"    python {surum.major}.{surum.minor}.{surum.micro}")
    if (surum.major, surum.minor) < ASGARI_PYTHON:
        print(
            f"\n!! Python {ASGARI_PYTHON[0]}.{ASGARI_PYTHON[1]}+ gerekli. "
            "python.org'dan kur ve kurulumda 'Add Python to PATH' kutusunu isaretle."
        )
        raise SystemExit(1)
    if not komut_var_mi("git"):
        print("\n!! git bulunamadi. https://git-scm.com/downloads adresinden kur.")
        raise SystemExit(1)
    print("    git         bulundu")
    paket = paketi_bul()
    print(f"    veri paketi {paket.name if paket else 'YOK -- veri adimi atlanacak'}")
    for ad in (".env", "kaggle.json"):
        if (BURASI / ad).is_file():
            print(f"    {ad:11s} bulundu")


def depoyu_al() -> Path:
    basla("2/6  Depo")
    depo = BURASI / DEPO_DIZIN
    if (depo / ".git").is_dir():
        print(f"    zaten var: {depo}")
        kos(["git", "pull", "--ff-only"], cwd=depo, zorunlu=False)
        return depo
    kos(["git", "clone", DEPO_URL, str(depo)])
    return depo


def ortami_kur(depo: Path) -> Path:
    basla("3/6  Python ortami")
    py = venv_python(depo)
    if py.is_file():
        print(f"    sanal ortam zaten var: {py}")
    else:
        onyukleme = depo / "requirements" / "uv-bootstrap.txt"
        if onyukleme.is_file():
            kos([sys.executable, "-m", "pip", "install", "--require-hashes", "-r", str(onyukleme)])
        elif not komut_var_mi("uv"):
            kos([sys.executable, "-m", "pip", "install", "uv"])
        kos(
            [
                "uv",
                "sync",
                "--locked",
                "--python",
                HEDEF_PYTHON,
                "--extra",
                "full",
                "--extra",
                "dev",
            ],
            cwd=depo,
        )
    if not py.is_file():
        print(f"\n!! Sanal ortam kurulmadi: {py} yok.")
        raise SystemExit(1)
    return py


def veriyi_ac(depo: Path, py: Path) -> bool:
    basla("4/6  Veri")
    paket = paketi_bul()
    if paket is None:
        print("    Veri paketi bulunamadi -- ATLANDI.")
        print("    datahon_*.zip dosyasini bu betigin yanina koyup tekrar calistir.")
        return False
    hedef = depo / paket.name
    if not hedef.is_file():
        print(f"    kopyalaniyor: {paket.name}")
        shutil.copy2(paket, hedef)
    kos([str(py), "scripts/tasima_paketi.py", "ac", paket.name], cwd=depo)
    return True


def _kopyala_gizli(kaynak: Path, hedef: Path, ad: str, eksik_notu: str) -> None:
    """Gizli bir ayar dosyasini yerine koyar; yoksa NE KAYBEDILDIGINI soyler."""
    if hedef.is_file():
        print(f"    {ad:14s} zaten yerinde")
        return
    if not kaynak.is_file():
        print(f"    {ad:14s} BULUNAMADI -- {eksik_notu}")
        print(f"    {'':14s} sonra eklemek icin: {hedef}")
        return
    hedef.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(kaynak, hedef)
    if os.name != "nt":
        hedef.chmod(0o600)
    print(f"    {ad:14s} yerlestirildi -> {hedef}")


def gizliyi_yerlestir(depo: Path) -> None:
    """Iki gizli dosyayi yerine koyar. Ikisi de PAKETE KONMAZ.

    ``.env``        -> depo koku      (FIRMS anahtari, EPIAS kimligi)
    ``kaggle.json`` -> ~/.kaggle/     (Kaggle API belirteci)

    ``kaggle.json`` OZELLIKLE onemli ve kolay unutulur: depo icinde DEGIL,
    ev dizininde durur. Yani ``git clone`` onu getirmez, veri paketi de
    tasimaz. Onsuz ``kaggle competitions submit`` calismaz -- yani laptoptan
    GONDERIM YAPILAMAZ. Bunun en kotu ogrenilme ani, gonderim hakki
    yanarken oldugu andir.
    """
    basla("5/6  Gizli ayarlar")
    _kopyala_gizli(
        BURASI / ".env",
        depo / ".env",
        ".env",
        "EPIAS ve FIRMS cekicileri calismaz (gerisi calisir)",
    )
    _kopyala_gizli(
        BURASI / "kaggle.json",
        Path.home() / ".kaggle" / "kaggle.json",
        "kaggle.json",
        "kaggle CLI ile GONDERIM YAPILAMAZ",
    )


def kapilari_kos(depo: Path, py: Path, veri_var: bool) -> int:
    basla("6/6  Dogrulama")
    if not veri_var:
        print("    Veri yok; yalnizca testler kosulacak.")
    adimlar: list[tuple[str, list[str]]] = []
    if veri_var:
        adimlar += [
            ("veri sagligi", [str(py), "scripts/veri_sagligi.py"]),
            ("kapsam deseni", [str(py), "scripts/kapsam_deseni.py"]),
        ]
    adimlar.append(("testler", [str(py), "-m", "pytest", "-q", "-p", "no:cacheprovider"]))

    basarisiz: list[str] = []
    for ad, komut in adimlar:
        print(f"\n--- {ad}")
        if kos(komut, cwd=depo, zorunlu=False) != 0:
            basarisiz.append(ad)
    return len(basarisiz)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--atla-dogrulama", action="store_true", help="6. adimi atla (hizli kurulum)")
    args = ap.parse_args()

    print("DATAHON KURULUMU")
    print(f"  calisma dizini: {BURASI}")

    on_kosullar()
    depo = depoyu_al()
    py = ortami_kur(depo)
    veri_var = veriyi_ac(depo, py)
    gizliyi_yerlestir(depo)

    hata = 0 if args.atla_dogrulama else kapilari_kos(depo, py, veri_var)

    print("\n" + "=" * 68)
    if hata:
        print(f"KURULUM BITTI ama {hata} dogrulama adimi BASARISIZ.")
        print("Yukaridaki ciktilari oku -- kurulum eksik olabilir.")
        return 1
    print("KURULUM TAMAM.")
    print(f"\n  cd {DEPO_DIZIN}")
    if not veri_var:
        print("\n  UYARI: veri paketi yoktu. datahon_*.zip'i yanina koyup tekrar calistir.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
