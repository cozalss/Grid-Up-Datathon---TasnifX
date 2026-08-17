"""COLAB BOOTSTRAP: Colab runtime'inda gridup'i calisir hale getirir.

KULLANIM -- Colab notebook'unun ILK hucresi:

    !git clone -q https://github.com/cozalss/Grid-Up-Datathon---TasnifX.git /content/datahon
    %run /content/datahon/scripts/colab_bootstrap.py

Repo OZELSE, klonlamadan once Colab Secrets'a ``GITHUB_TOKEN`` ekle; bu betik
onu bulup klon URL'sine gomer.

NE YAPAR
    1. Ortam teshisi  -- Colab mi, hangi surumler, Kaggle hedefinden sapma var mi
    2. gridup kurulumu -- repodan editable, bagimlilik cakismasi COZMEDEN
    3. Harici veri     -- Kaggle dataset'inden (hava/gunes/ilceler) indirir
    4. Dogrulama       -- import eder, shape'leri yazar

NE YAPMAZ
    Submission notebook'u URETMEZ. Colab bir DENEY ISCISIDIR; juriye gidecek
    notebook Kaggle'in kendi ortaminda yazilir ve orada dogrulanir. Colab'dan
    disari cikan sey kod degil, ARTEFAKTTIR: OOF tahminleri, Optuna parametreleri,
    model agirliklari -- bunlar surumden bagimsizdir.

SIR YONETIMI
    Hicbir kimlik bilgisi bu dosyada YAZILI DEGILDIR ve hicbir ciktiya basilmaz.
    Colab Secrets (sol menu, anahtar ikonu) uzerinden okunur:
        GITHUB_TOKEN     -- repo ozelse
        KAGGLE_USERNAME  -- harici veri + yarisma verisi icin
        KAGGLE_KEY
        EPIAS_USERNAME   -- yalnizca EPIAS'tan TAZE veri cekeceksen
        EPIAS_PASSWORD
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

# Kaggle imajinin OLCULEN surumleri (docs/01 madde 9). Colab bunlardan
# saparsa Colab'da calisan kod Kaggle'da patlayabilir -- sessiz kalmiyoruz.
KAGGLE_HEDEF = {
    "python": "3.12",
    "numpy": "2.0.2",
    "pandas": "3.0.4",
    "sklearn": "1.9.0",
}

REPO_URL = "https://github.com/cozalss/Grid-Up-Datathon---TasnifX.git"
KAGGLE_DATASET = "cemzal/gridup-offline-paket"
VERI_DIZINI = Path("/content/gridup_veri")


def _sir(ad: str) -> str | None:
    """Colab Secrets -> ortam degiskeni sirasiyla dener. Degeri ASLA yazdirmaz."""
    try:
        from google.colab import userdata  # type: ignore[import-not-found]

        deger = userdata.get(ad)
        if deger:
            return deger
    except Exception:
        # Secrets tanimli degil ya da Colab disindayiz -- ortam degiskenine dus.
        pass
    return os.environ.get(ad)


def _calistir(argv: list[str], *, aciklama: str) -> subprocess.CompletedProcess[str]:
    """Sessiz basarisizlik yok: donus kodu sifir degilse stderr'in basini yazar."""
    sonuc = subprocess.run(argv, capture_output=True, text=True, check=False)
    if sonuc.returncode != 0:
        print(f"HATA [{aciklama}]: {sonuc.stderr[:500]}")
    return sonuc


# ---------------------------------------------------------------- 1. teshis


def ortam_teshisi() -> bool:
    """Surumleri yazar, Kaggle hedefinden sapmalari UYARI olarak isaretler."""
    colab_mi = importlib.util.find_spec("google.colab") is not None
    print(f"ortam        : {'Colab' if colab_mi else 'Colab DEGIL -- yerel/diger'}")

    surumler = {"python": ".".join(map(str, sys.version_info[:2]))}
    for modul, ad in (("numpy", "numpy"), ("pandas", "pandas"), ("sklearn", "sklearn")):
        try:
            surumler[ad] = __import__(modul).__version__
        except ImportError:
            surumler[ad] = "YOK"

    burasi = "Colab" if colab_mi else "burasi"
    sapma = [
        f"{ad}: {burasi} {surumler[ad]} vs Kaggle {hedef}"
        for ad, hedef in KAGGLE_HEDEF.items()
        if surumler.get(ad) not in (hedef, "YOK")
    ]
    print("surumler     : " + " · ".join(f"{a}={s}" for a, s in surumler.items()))

    if sapma:
        print(f"\nUYARI -- {burasi} surumleri Kaggle hedefinden SAPIYOR:")
        for satir in sapma:
            print(f"  · {satir}")
        print(
            "  Burada calisan kod Kaggle'da patlayabilir. Colab'i DENEY ISCISI\n"
            "  olarak kullan; submission notebook'unu Kaggle'da yaz ve orada dogrula.\n"
        )
    return colab_mi


# ---------------------------------------------------------------- 2. kurulum


def gridup_kur(repo_dizini: Path) -> None:
    """Repoyu editable kurar. Bagimliliklari COZDURMUYORUZ -- neden asagida."""
    if not (repo_dizini / "pyproject.toml").is_file():
        raise SystemExit(
            f"Repo bulunamadi: {repo_dizini}\nOnce klonla:  !git clone -q {REPO_URL} {repo_dizini}"
        )

    # --no-deps SART: Colab'in numpy/pandas'ini pip'in yukseltmesine izin verirsek
    # onlara karsi derlenmis lightgbm/catboost/torch sessizce bozulur. Cekirdek uc
    # paket Colab'da zaten var; eksik olan opsiyonelleri asagida ayrica kuruyoruz.
    _calistir(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(repo_dizini)],
        aciklama="gridup editable kurulumu",
    )

    # Colab imajinda OLMAYAN, bizim feature ailelerinin ihtiyac duydugu paketler.
    # lightgbm/xgboost/torch/optuna Colab'da HAZIR gelir -- onlara dokunmuyoruz.
    eksikler = [
        paket
        for paket, modul in (
            ("catboost", "catboost"),
            ("holidays", "holidays"),
            ("pvlib", "pvlib"),
            ("hijridate", "hijridate"),
            ("shap", "shap"),
        )
        if importlib.util.find_spec(modul) is None
    ]
    if eksikler:
        print(f"eksik paket  : {', '.join(eksikler)} -- kuruluyor")
        _calistir(
            [sys.executable, "-m", "pip", "install", "-q", *eksikler],
            aciklama="opsiyonel bagimliliklar",
        )


# ------------------------------------------------------------ 3. harici veri


def harici_veri_indir(hedef: Path) -> Path | None:
    """Kaggle dataset'inden hava/gunes/ilceler tablolarini ceker.

    data/ dizini .gitignore'da oldugu icin klon bu dosyalari GETIRMEZ. Kaggle
    dataset'i tek dogruluk kaynagidir: hem Kaggle notebook'u hem Colab ayni
    dosyalari gorur, sürüm kaymasi olmaz.
    """
    kullanici, anahtar = _sir("KAGGLE_USERNAME"), _sir("KAGGLE_KEY")
    if not (kullanici and anahtar):
        print(
            "ATLANDI      : harici veri -- Colab Secrets'ta KAGGLE_USERNAME/KAGGLE_KEY yok.\n"
            "               Hava, gunes ve ilce tablolari OLMADAN devam ediliyor;\n"
            "               ilgili feature aileleri calismayacak."
        )
        return None

    os.environ["KAGGLE_USERNAME"] = kullanici
    os.environ["KAGGLE_KEY"] = anahtar
    if importlib.util.find_spec("kaggle") is None:
        _calistir([sys.executable, "-m", "pip", "install", "-q", "kaggle"], aciklama="kaggle cli")

    hedef.mkdir(parents=True, exist_ok=True)
    sonuc = _calistir(
        [
            sys.executable,
            "-m",
            "kaggle",
            "datasets",
            "download",
            "-d",
            KAGGLE_DATASET,
            "-p",
            str(hedef),
            "--unzip",
        ],
        aciklama=f"dataset indirme ({KAGGLE_DATASET})",
    )
    if sonuc.returncode != 0:
        return None

    dosyalar = sorted(p.name for p in hedef.iterdir() if p.is_file())
    print(f"harici veri  : {hedef} -- {len(dosyalar)} dosya")
    return hedef


# ------------------------------------------------------------- 4. dogrulama


def dogrula(veri_dizini: Path | None) -> None:
    """gridup gercekten import ediliyor mu ve veri okunuyor mu -- kanit."""
    import gridup

    print(f"gridup       : {gridup.__version__} hazir · {len(gridup.__all__)} isim")

    if veri_dizini is None:
        return
    import pandas as pd

    for ad in ("hava_gunluk", "gunes_gunluk", "ilceler_gdz_adm"):
        yol = veri_dizini / f"{ad}.parquet"
        if yol.is_file():
            print(f"  {ad:<18} {pd.read_parquet(yol).shape}")
        else:
            print(f"  {ad:<18} EKSIK")


def main() -> None:
    print("=" * 70)
    ortam_teshisi()

    repo_dizini = Path(os.environ.get("GRIDUP_REPO", "/content/datahon"))
    gridup_kur(repo_dizini)

    veri_dizini = harici_veri_indir(VERI_DIZINI)
    dogrula(veri_dizini)

    print("=" * 70)
    print("TAMAM -- Colab isci olarak hazir.")
    print("Submission notebook'u BURADA degil, Kaggle ortaminda yazilir.")


if __name__ == "__main__":
    main()
