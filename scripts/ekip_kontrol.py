"""EKIP KURULUM DOKTORU -- tek komutla "her sey calisiyor mu?" raporu.

NEDEN BU BETIK VAR
------------------
Repo'yu yeni klonlayan bir ekip uyesinin "bende calismiyor" demesi en pahali
gecikme turudur: sorunlarin yarisi kurulum (eksik paket, eksik veri dosyasi,
cp1254 konsolu), diger yarisi bunlarin GEC fark edilmesidir. Bu betik yedi
kontrolu tek komutta kosar ve her FAIL'in yanina DUZELTME KOMUTUNU basar --
teshis ile tedavi ayni satirda.

Kontroller:
  1. python surumu >= 3.11   ('python' komutu; python3 Windows'ta Store stub'i)
  2. zorunlu paketler        (liste pyproject.toml'dan okunur, elle tutulmaz)
  3. gridup import           (kurulu paket veya src/ uzerinden)
  4. veri varliklari         (parquet'ler; eksikse indirme komutu)
  5. kaggle.json             (yalnizca kullanici adi basilir, anahtar ASLA)
  6. mini duman              (500 satir sentetik, 1 fold LightGBM, < 30 sn)
  7. konsol encoding         (cp1254 tuzagi + PYTHONIOENCODING onerisi)

OLCULDU (bu makinede): 7/7 PASS, toplam 2.5 sn; paketler 16/16 (13'u
pyproject'ten, 3'u test altyapisi), veri 231.648 + 233.760 + 96 satir,
duman MAE=8.03 (< 0.1 sn), EXIT=0.

KULLANIM
    python scripts/ekip_kontrol.py

CIKIS KODU: hepsi PASS ise 0, degilse 1.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import locale
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]
PYPROJECT = KOK / "pyproject.toml"
CONFTEST = KOK / "tests" / "conftest.py"

# pip dagitim adi != import adi olan tek paketimiz.
IMPORT_ADI = {"scikit-learn": "sklearn"}

# Bilincli olarak kontrol DISI birakilan opsiyonel gruplar:
#   neural: torch Kaggle imajinda ZATEN var, yerelde zorunlu degil (yuzlerce MB)
#   viz   : matplotlib hatti dogrulamak icin gerekmez
#   full  : diger gruplarin birlesimi -- ayni paketleri iki kez sayardik
ZORUNLU_GRUPLAR = ("models", "search", "features", "io")

# Eksik dosyanin yanina basilacak indirme komutu. gunes_gunluk yerelde uretilmez;
# Kaggle offline paketinden gelir (build_kaggle_package.py onu oraya koyar).
VERI_VARLIKLARI: tuple[tuple[str, str], ...] = (
    ("data/external/hava_gunluk.parquet", "python scripts/fetch_weather.py"),
    (
        "data/external/gunes_gunluk.parquet",
        "python -m kaggle datasets download -d cemzal/gridup-offline-paket"
        " -p data/external --unzip",
    ),
    ("data/reference/ilceler_gdz_adm.parquet", "python scripts/fetch_districts.py"),
)

DUMAN_SURE_SINIRI_SN = 30.0


@dataclass(frozen=True)
class Kontrol:
    """Tek bir kontrolun sonucu: durum + tek satir ozet + istege bagli detay."""

    ad: str
    gecti: bool
    detay: str
    satirlar: tuple[str, ...] = ()
    duzeltme: str | None = None


# ------------------------------------------------------------------ yardimcilar


def _surum_demeti(metin: str) -> tuple[int, int, int]:
    """'3.0.3', '14.0', '2.0.2.post1' gibi metinleri kiyaslanabilir demete cevirir."""
    parcalar = [int(parca) for parca in re.findall(r"\d+", metin)[:3]]
    while len(parcalar) < 3:
        parcalar.append(0)
    return (parcalar[0], parcalar[1], parcalar[2])


def _kurulu_surum(dagitim_adi: str, modul: object) -> str:
    try:
        return importlib.metadata.version(dagitim_adi)
    except importlib.metadata.PackageNotFoundError:
        return str(getattr(modul, "__version__", "bilinmiyor"))


def paket_listesi(pyproject_yolu: Path) -> list[tuple[str, str | None]]:
    """Zorunlu paket listesini pyproject.toml'dan turetir -- elle liste YOK.

    Elle tutulan liste pyproject degisince sessizce eskir; buradaki tek dogruluk
    kaynagi pyproject'tir. Cekirdek bagimliliklar + ZORUNLU_GRUPLAR okunur
    (olculdu: 13 paket).
    """
    import tomllib

    veri = tomllib.loads(pyproject_yolu.read_text(encoding="utf-8"))
    girisler = list(veri["project"].get("dependencies", []))
    gruplar = veri["project"].get("optional-dependencies", {})
    for grup in ZORUNLU_GRUPLAR:
        girisler.extend(gruplar.get(grup, []))

    paketler: list[tuple[str, str | None]] = []
    gorulen: set[str] = set()
    for giris in girisler:
        eslesme = re.match(r"\s*([A-Za-z0-9_.-]+)\s*(?:>=\s*([0-9][0-9.]*))?", giris)
        if eslesme is None or eslesme.group(1) in gorulen:
            continue
        gorulen.add(eslesme.group(1))
        paketler.append((eslesme.group(1), eslesme.group(2)))
    return paketler


def gelistirme_araclari(pyproject_yolu: Path, conftest_yolu: Path) -> list[str]:
    """Test altyapisini repodan turetir: pytest/ruff [tool.*] bolumlerinden,
    hypothesis conftest.py'nin import'undan. Bunlar pyproject bagimliligi DEGIL
    cunku Kaggle imajina kurulmazlar -- ama testleri kosacak ekip uyesine sart.
    """
    import tomllib

    veri = tomllib.loads(pyproject_yolu.read_text(encoding="utf-8"))
    araclar = [ad for ad in ("pytest", "ruff") if ad in veri.get("tool", {})]
    if conftest_yolu.is_file() and "hypothesis" in conftest_yolu.read_text(encoding="utf-8"):
        araclar.append("hypothesis")
    return araclar


def _parquet_satir_sayisi(yol: Path) -> int:
    """Yalnizca dosya ayak notunu okur -- 10 MB'lik parquet'i yuklemeye gerek yok."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        import pandas as pd

        return len(pd.read_parquet(yol))
    return pq.read_metadata(yol).num_rows


# ------------------------------------------------------------------ kontroller


def python_surumu_kontrol() -> Kontrol:
    surum = sys.version.split()[0]
    return Kontrol(
        ad="1) python surumu",
        # pyproject.toml requires-python = ">=3.10" -- doktor daha kati
        # olamaz: 3.10 kuran uye pip kurulumunu tamamlayip FAIL alirdi
        # (denetimde yakalandi).
        gecti=sys.version_info >= (3, 10),
        detay=f"{surum} (gerekli >= 3.10)",
        satirlar=(f"yorumlayici: {sys.executable}",),
        duzeltme=(
            "python.org'dan 3.10+ kur; 'python' komutu onu gostermeli "
            "(python3 DEGIL -- Windows'ta Microsoft Store stub'i acilir)"
        ),
    )


def paket_kontrol() -> Kontrol:
    ad = "2) zorunlu paketler"
    try:
        paketler = paket_listesi(PYPROJECT)
        paketler += [(arac, None) for arac in gelistirme_araclari(PYPROJECT, CONFTEST)]
    except (ImportError, OSError, KeyError, ValueError) as hata:
        return Kontrol(ad, gecti=False, detay=f"pyproject.toml okunamadi: {hata}")

    satirlar: list[str] = []
    eksikler: list[str] = []
    for dagitim, sinir in paketler:
        try:
            modul = importlib.import_module(IMPORT_ADI.get(dagitim, dagitim))
        except ImportError:
            eksikler.append(f"{dagitim}>={sinir}" if sinir else dagitim)
            satirlar.append(f"{dagitim:<14} EKSIK")
            continue
        surum = _kurulu_surum(dagitim, modul)
        if sinir and _surum_demeti(surum) < _surum_demeti(sinir):
            eksikler.append(f"{dagitim}>={sinir}")
            satirlar.append(f"{dagitim:<14} {surum:<12} ESKI (gerekli >= {sinir})")
        else:
            kosul = f"(>= {sinir})" if sinir else "(test araci)"
            satirlar.append(f"{dagitim:<14} {surum:<12} {kosul}")

    hazir = len(paketler) - len(eksikler)
    duzeltme = None
    if eksikler:
        duzeltme = "python -m pip install " + " ".join(f'"{eksik}"' for eksik in eksikler)
    return Kontrol(
        ad=ad,
        gecti=not eksikler,
        detay=f"{hazir}/{len(paketler)} paket hazir (liste pyproject.toml'dan)",
        satirlar=tuple(satirlar),
        duzeltme=duzeltme,
    )


def gridup_kontrol() -> Kontrol:
    ad = "3) gridup paketi"
    kaynak = "kurulu paket"
    try:
        import gridup
    except ImportError:
        # Repodaki butun betiklerin kullandigi mesru yol: src/'yi path'e ekle.
        sys.path.insert(0, str(KOK / "src"))
        try:
            import gridup
        except ImportError as hata:
            return Kontrol(
                ad=ad,
                gecti=False,
                detay=f"import edilemedi: {hata}",
                duzeltme="repo kokunde: python -m pip install -e .",
            )
        kaynak = "src/ sys.path uzerinden (kalici cozum: python -m pip install -e .)"
    return Kontrol(
        ad=ad,
        gecti=True,
        detay=f"v{gridup.__version__}, {len(gridup.__all__)} disa acik isim, {kaynak}",
    )


def veri_kontrol(
    varliklar: tuple[tuple[str, str], ...] = VERI_VARLIKLARI, kok: Path = KOK
) -> Kontrol:
    satirlar: list[str] = []
    eksik = 0
    for gorece_yol, komut in varliklar:
        yol = kok / gorece_yol
        if not yol.is_file():
            eksik += 1
            satirlar.append(f"{yol.name:<28} EKSIK -> {komut}")
            continue
        try:
            satirlar.append(f"{yol.name:<28} {_parquet_satir_sayisi(yol):,} satir")
        except (OSError, ValueError) as hata:  # pyarrow ArrowInvalid, ValueError'dur
            eksik += 1
            satirlar.append(f"{yol.name:<28} OKUNAMADI ({hata}) -> {komut}")
    return Kontrol(
        ad="4) veri varliklari",
        gecti=eksik == 0,
        detay=f"{len(varliklar) - eksik}/{len(varliklar)} dosya hazir",
        satirlar=tuple(satirlar),
        duzeltme="yukaridaki '->' komutlarini repo kokunde kos" if eksik else None,
    )


def kaggle_kontrol() -> Kontrol:
    """GUVENLIK: kaggle.json'in 'key' alani ASLA basilmaz -- yalnizca kullanici adi."""
    ad = "5) kaggle.json"
    dizin = Path(os.environ.get("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle")))
    yol = dizin / "kaggle.json"
    duzeltme = (
        "kaggle.com/settings -> API -> 'Create New Token'; inen dosyayi "
        f"{Path.home() / '.kaggle' / 'kaggle.json'} yoluna koy"
    )
    if not yol.is_file():
        return Kontrol(ad, gecti=False, detay=f"{yol} yok", duzeltme=duzeltme)
    try:
        icerik = json.loads(yol.read_text(encoding="utf-8"))
    except (OSError, ValueError) as hata:
        return Kontrol(ad, gecti=False, detay=f"{yol} okunamadi: {hata}", duzeltme=duzeltme)
    kullanici = icerik.get("username")
    if not kullanici:
        return Kontrol(
            ad, gecti=False, detay=f"{yol} icinde 'username' alani yok", duzeltme=duzeltme
        )
    return Kontrol(ad, gecti=True, detay=f"kullanici '{kullanici}' ({yol})")


def duman_kontrol() -> Kontrol:
    """500 satirlik sentetik panelde 1 fold LightGBM: hatti ucuca kanitlar.

    Gercek panelin minyaturu: 10 ilce x 50 gun, zaman sirali; fold ilk 40 gunu
    train, son 10 gunu valid yapar (ileri zincirleme -- rastgele bolme DEGIL).
    """
    ad = "6) mini duman"
    try:
        import numpy as np
        import pandas as pd

        from gridup import cross_validate
        from gridup.models import starter_params
    except ImportError as hata:
        return Kontrol(
            ad=ad,
            gecti=False,
            detay=f"import edilemedi: {hata}",
            duzeltme='python -m pip install -e ".[models]"',
        )

    basla = time.perf_counter()
    rng = np.random.default_rng(42)
    n_ilce, n_gun = 10, 50  # 500 satir
    gun = np.repeat(np.arange(n_gun), n_ilce)
    sicaklik = 20.0 + 8.0 * np.sin(2 * np.pi * gun / 365.0) + rng.normal(0.0, 2.0, n_gun * n_ilce)
    ruzgar = rng.gamma(2.0, 3.0, n_gun * n_ilce)
    hedef = np.maximum(
        0.0, 5.0 * ruzgar + 3.0 * np.abs(sicaklik - 20.0) + rng.normal(0.0, 5.0, n_gun * n_ilce)
    )
    ozellikler = pd.DataFrame(
        {
            "sicaklik": sicaklik,
            "ruzgar": ruzgar,
            "gun_sira": gun,
            "ilce_kodu": np.tile(np.arange(n_ilce), n_gun),
        }
    )
    fold = (np.arange(0, 40 * n_ilce), np.arange(40 * n_ilce, n_gun * n_ilce))
    params = starter_params("lightgbm", "regression", objective="mae")
    params["n_estimators"] = 80  # duman testi: dogrulugu degil CALISMAYI olcuyoruz

    try:
        sonuc = cross_validate(
            ozellikler, hedef, [fold], kind="lightgbm", metric="mae",
            params=params, early_stopping_rounds=25, verbose=False,
        )
    except Exception as hata:  # doktor cokmemeli: her hata teshise donusur
        return Kontrol(ad, gecti=False, detay=f"cross_validate patladi: {hata!r}")

    sure = time.perf_counter() - basla
    if not np.isfinite(sonuc.overall_score):
        return Kontrol(ad, gecti=False, detay=f"MAE sonlu degil: {sonuc.overall_score}")
    return Kontrol(
        ad=ad,
        gecti=sure < DUMAN_SURE_SINIRI_SN,
        detay=(
            f"500 satir, 1 fold LightGBM: MAE={sonuc.overall_score:.2f}, "
            f"{sure:.1f} sn (sinir {DUMAN_SURE_SINIRI_SN:.0f} sn)"
        ),
        duzeltme="makine cok yavas veya antivirus python'u tariyor; tekrar dene",
    )


def encoding_kontrol(
    *,
    ortam: str | None = None,
    stdout_kodlamasi: str | None = None,
    utf8_modu: bool | None = None,
    tercih: str | None = None,
) -> Kontrol:
    """cp1254 tuzagi: konsol Turkce kod sayfasindayken dosya/altsurec ciktisi bozulur.

    Parametreler test edilebilirlik icin enjekte edilebilir; None ise gercek
    ortamdan okunur.
    """
    if ortam is None:
        ortam = os.environ.get("PYTHONIOENCODING", "")
    if stdout_kodlamasi is None:
        stdout_kodlamasi = getattr(sys.stdout, "encoding", "") or ""
    if utf8_modu is None:
        utf8_modu = bool(sys.flags.utf8_mode)
    if tercih is None:
        tercih = locale.getpreferredencoding(False)

    utf8_aktif = utf8_modu or "utf" in ortam.lower() or "utf" in stdout_kodlamasi.lower()
    satirlar: list[str] = []
    if tercih.lower().startswith("cp125"):
        satirlar.append(
            f"locale {tercih}: Turkce karakterli dosya/altsurec ciktisi bozulabilir"
        )
    return Kontrol(
        ad="7) konsol encoding",
        gecti=utf8_aktif,
        detay=(
            f"stdout={stdout_kodlamasi or '?'}  locale={tercih}  "
            f"PYTHONIOENCODING={ortam or 'AYARSIZ'}"
        ),
        satirlar=tuple(satirlar),
        duzeltme=(
            "kalici: setx PYTHONIOENCODING utf-8  (yeni terminalde gecerli) | "
            'bu oturum: $env:PYTHONIOENCODING="utf-8"'
        ),
    )


# ------------------------------------------------------------------ rapor


def yazdir(kontrol: Kontrol) -> None:
    durum = "PASS" if kontrol.gecti else "FAIL"
    print(f"[{durum}] {kontrol.ad:<22} {kontrol.detay}")
    for satir in kontrol.satirlar:
        print(f"{'':7}{satir}")
    if not kontrol.gecti and kontrol.duzeltme:
        print(f"{'':7}duzeltme: {kontrol.duzeltme}")


def main() -> int:
    basla = time.perf_counter()
    print("=" * 78)
    print("EKIP KURULUM DOKTORU -- yedi kontrol, her FAIL'in yaninda duzeltme komutu")
    print(f"repo: {KOK}")
    print("=" * 78)
    print()

    kontroller: list[Kontrol] = []
    for uretici in (
        python_surumu_kontrol,
        paket_kontrol,
        gridup_kontrol,  # duman testinden ONCE: sys.path fallback'ini o kurar
        veri_kontrol,
        kaggle_kontrol,
        duman_kontrol,
        encoding_kontrol,
    ):
        kontrol = uretici()
        kontroller.append(kontrol)
        yazdir(kontrol)

    gecen = sum(1 for kontrol in kontroller if kontrol.gecti)
    sure = time.perf_counter() - basla
    print("-" * 78)
    if gecen == len(kontroller):
        print(f"SONUC: {gecen}/{len(kontroller)} PASS -- ortam hazir. ({sure:.1f} sn)")
        return 0
    print(
        f"SONUC: {gecen}/{len(kontroller)} PASS, {len(kontroller) - gecen} FAIL -- "
        f"yukaridaki 'duzeltme:' satirlarini kos. ({sure:.1f} sn)"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
