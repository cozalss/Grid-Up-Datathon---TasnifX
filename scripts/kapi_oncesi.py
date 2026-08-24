"""COMMIT ONCESI KAPI -- CI'in kirmizi olmasini yapisal olarak imkansiz kilar.

NEDEN BU BETIK
--------------
2026-08-23/24 boyunca CI DORT KEZ kirmizi oldu ve dordunde de ayni sebep:
``ruff format`` kapisi. Birikme sekli sinsi -- hizli calisirken biçim kapisi
atlanir, dosyalar birikir, sonra CI'da 22 dosya birden patlar.

Elle yakalamak ise yaradi ama surdurulemez: kapi seti 6 SANIYE suruyor,
oysa kirmizi CI'i gormek + duzeltmek + push etmek her seferinde dakikalar
aldi. Insanin hatirlamasina birakilan bir adim, unutulan bir adimdir.

NE YAPAR
--------
Yalnizca SAHNELENMIS ``.py`` dosyalarina bakar (tum agaci taramaz -- hizli):

    1. ``ruff format``   -- bicim bozuksa DUZELTIR ve commit'i reddeder
    2. ``ruff check``    -- lint hatasi varsa reddeder (otomatik duzeltmez)
    3. ``scan_secrets``  -- sir bulursa reddeder

NEDEN DUZELTIP YINE REDDEDIYOR
-------------------------------
``git commit --only <yollar>`` gecici bir indeks kullanir. Kanca icinden
``git add`` yapmak GERCEK indekse yazar, gecici olana degil -- yani duzeltme
commit'e GIRMEZ, sessizce disarda kalir. Bu, "bicimlendirdim" diyip
bicimlenmemis kod gondermek olurdu.

Bu yuzden: dosyayi duzeltir, ama commit'i REDDEDER. Ikinci denemede gecer.
Iki adim, ama hicbir zaman yanlis bir commit uretmez.

NEDEN OTOMATIK BICIM GUVENLI, OTOMATIK LINT DEGIL
--------------------------------------------------
``ruff format`` yalnizca bosluk ve satir kirilimi degistirir; bugun 23 dosyada
AST karsilastirmasiyla dogrulandi -- davranis degismiyor. ``ruff check --fix``
ise kod SILEBILIR (olu degisken, kullanilmayan import) ve o silme bir hatayi
maskeleyebilir. O yuzden lint hatasi insana bildirilir, otomatik duzeltilmez.

KURULUM (bir kez)
-----------------
    git config core.hooksPath .githooks
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]

#: Sanal ortamin Python'u -- kanca kabuk ortamindan cagrildigi icin
#: ``sys.executable`` sistem Python'u olabilir; ruff orada kurulu olmayabilir.
VENV = (
    KOK
    / ".venv"
    / ("Scripts" if sys.platform == "win32" else "bin")
    / ("python.exe" if sys.platform == "win32" else "python")
)


def python_yolu() -> str:
    return str(VENV) if VENV.is_file() else sys.executable


def sahnelenmis_py() -> list[str]:
    """Commit'e girecek ``.py`` dosyalari. Silinenler haric (--diff-filter=ACM)."""
    c = subprocess.run(  # noqa: S603
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        cwd=KOK,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    return [
        y for y in c.stdout.split("\n") if y.strip().endswith(".py") and (KOK / y.strip()).is_file()
    ]


def kos(argv: list[str]) -> tuple[int, str]:
    c = subprocess.run(  # noqa: S603
        argv, cwd=KOK, capture_output=True, text=True, encoding="utf-8", check=False
    )
    return c.returncode, (c.stdout or "") + (c.stderr or "")


def main() -> int:
    dosyalar = sahnelenmis_py()
    if not dosyalar:
        return 0
    py = python_yolu()
    print(f"[kapi] {len(dosyalar)} sahnelenmis .py dosyasi denetleniyor")

    # 1) BICIM -- bozuksa duzelt ve reddet
    kod, _ = kos([py, "-m", "ruff", "format", "--check", *dosyalar])
    if kod != 0:
        kos([py, "-m", "ruff", "format", *dosyalar])
        print("\n[kapi] BICIM BOZUKTU -- DUZELTILDI, ama commit reddedildi.")
        print("[kapi] Sebebi: --only kipinde kanca icinden yapilan duzeltme commit'e girmez.")
        print("[kapi] Dosyalari tekrar sahneleyip ayni commit'i yeniden calistir.\n")
        return 1

    # 2) LINT -- otomatik duzeltilmez, kod silebilir
    kod, cikti = kos([py, "-m", "ruff", "check", "--output-format=concise", *dosyalar])
    if kod != 0:
        print("\n[kapi] LINT HATASI -- commit reddedildi (otomatik duzeltilmez):")
        print(cikti.strip()[:2000])
        print("\n[kapi] Duzeltip tekrar dene. Silme onerilerini once OKU.\n")
        return 1

    # 3) SIR -- gecmise yazilirsa kalici, en tehlikeli kapi
    kod, cikti = kos([py, str(KOK / "security" / "scan_secrets.py")])
    if kod != 0:
        print("\n[kapi] GIZLI TARAMA PATLADI -- commit reddedildi:")
        print(cikti.strip()[:2000])
        print("\n[kapi] Bir sir commit'lenirse git GECMISINE kalici yazilir.\n")
        return 1

    print("[kapi] bicim + lint + sir: TEMIZ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
