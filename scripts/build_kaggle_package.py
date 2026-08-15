"""Kaggle'da INTERNETSIZ calisacak paketi hazirlar.

NEDEN GEREKLI
-------------
Grid Up Datathon'da notebook juriye gidiyor. Kaggle notebook'unda internet
kapali oldugunda su ucu de calismaz:

  * ``pip install gridup``      -- PyPI'ya erisemez
  * ``requests.get(...)``       -- hava/EPIAS API'sine erisemez
  * ``pvlib``, ``hijridate``    -- kurulu degilse kurulamaz

Cozum: hepsini **tek bir Kaggle Dataset**'e koyup notebook'a girdi olarak
baglamak. Dataset'ler internetsiz notebook'ta da ``/kaggle/input/`` altinda
gorunur -- bu, internetsiz calismanin resmi yoludur.

NE PAKETLENIR
-------------
  1. ``gridup-*.whl``           -- kendi paketimiz (pure python, py3-none-any)
  2. Harici veri parquet'leri   -- hava, gunes, ilce referansi
  3. ``tekerlekler/`` (opsiyonel) -- pvlib/hijridate gibi Kaggle'da OLMAYAN
     paketlerin wheel'leri; ``--wheels`` ile indirilir

KULLANIM
--------
::

    python scripts/build_kaggle_package.py                 # sadece paketle
    python scripts/build_kaggle_package.py --wheels        # bagimlilik wheel'lerini de indir
    python scripts/build_kaggle_package.py --upload        # Kaggle'a yukle

Yukleme icin ``~/.kaggle/kaggle.json`` gerekir (zaten var).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CIKTI = ROOT / "kaggle_paket"

#: Dataset icine kopyalanacak veri dosyalari. Eksik olan SESSIZ atlanmaz,
#: raporlanir -- yarisma gunu "veri neden yok" diye aramak istemeyiz.
VERI_DOSYALARI = (
    "data/external/hava_gunluk.parquet",
    "data/external/gunes_gunluk.parquet",
    "data/reference/ilceler_gdz_adm.parquet",
    "data/reference/ilceler_gdz_adm.csv",
)

#: Kaggle imajinda OLMAYAN, bizim kullandigimiz paketler. Bunlarin wheel'ini
#: yanimizda goturmezsek internetsiz notebook'ta ilgili feature ailesi calismaz.
#: (Kaggle'da lightgbm/xgboost/catboost/optuna/shap/holidays ZATEN vardir.)
#:
#: tabm: cekirdek degil, OPSIYON -- 2024-25 Kaggle kanitli MLP-ansambl
#: (CIBMTR 1.'si kullandi). Yarisma sirasinda harman uyesi olarak denemek
#: istersek internetsiz ortamda elimizin altinda olsun diye tasiniyor;
#: agirlik dosyasi yok, kendi verimizle egitilir (torch Kaggle'da zaten var).
#: NOT: tabm'in kendi bagimliligi rtdl_num_embeddings de Kaggle'da yok --
#: --no-deps indirdigimiz icin acikca listelenmek zorunda (olculdu:
#: METADATA Requires-Dist: torch [var], rtdl_num_embeddings [YOK],
#: typing_extensions [var]).
EKSIK_PAKETLER = ("pvlib", "hijridate", "tabm", "rtdl_num_embeddings")

#: Bu paketleri **KASITLI OLARAK PAKETLEMIYORUZ.** Kaggle imajinda zaten
#: varlar ve farkli bir surumu kurmak AKTIF ZARARLIDIR: numpy/scipy ikili
#: (ABI) uyumu gerektirir; Kaggle'in numpy 2.0.2'sinin uzerine 2.2.6 kurmak,
#: eski ABI'ye karsi derlenmis lightgbm/catboost/shap'i calismaz hale
#: getirebilir. Olculdu: --no-deps olmadan pip bunlarin 64 MB'ini indiriyordu.
KAGGLE_DA_MEVCUT = ("numpy", "pandas", "scipy", "requests", "h5py", "pytz", "six")

DATASET_SLUG = "gridup-offline-paket"
DATASET_BASLIK = "Grid Up Offline Paket"


def _kos(komut: list[str], *, aciklama: str) -> subprocess.CompletedProcess[str]:
    """Alt surec calistirir, hatayi yutmaz."""
    print(f"  $ {' '.join(komut)}")
    # check=False KASITLI: donus kodunu asagida kendimiz raporluyoruz. Istisna
    # firlatmak, kismi basariyi (orn. 2 wheel'den 1'i indi) gizlerdi.
    sonuc = subprocess.run(
        komut, capture_output=True, text=True, encoding="utf-8", cwd=ROOT, check=False
    )
    if sonuc.returncode != 0:
        print(f"  HATA ({aciklama}):")
        print("   ", (sonuc.stderr or sonuc.stdout or "")[:800])
    return sonuc


def wheel_uret() -> Path | None:
    """Paketin wheel'ini uretir ve yolunu dondurur."""
    print("\n[1/4] Wheel uretiliyor")
    dist = ROOT / "dist"
    if dist.exists():
        shutil.rmtree(dist)
    sonuc = _kos(
        [sys.executable, "-m", "build", "--wheel", "--outdir", "dist", "."],
        aciklama="wheel uretimi",
    )
    if sonuc.returncode != 0:
        print("  Once 'pip install build' calistir.")
        return None
    tekerlekler = sorted(dist.glob("*.whl"))
    if not tekerlekler:
        print("  dist/ altinda wheel bulunamadi.")
        return None
    wheel = tekerlekler[-1]
    print(f"  OK: {wheel.name} ({wheel.stat().st_size / 1024:.0f} KB)")
    return wheel


def bagimlilik_wheel_indir(hedef: Path) -> int:
    """Kaggle'da olmayan paketlerin wheel'lerini indirir.

    Iki bayrak da ZORUNLU ve ikisi de farkli bir hatayi onler:

    ``--platform manylinux2014_x86_64``
        Kaggle Linux, biz Windows'tayiz. Bu bayrak olmadan pip Windows
        wheel'i indirir; Kaggle'da kurulmaz. Yarisma gunu fark edilmesi
        gec bir hatadir.

    ``--no-deps``
        Bagimliliklari CEKMEZ. Cekseydi numpy/pandas/scipy'nin YENI surumleri
        inerdi (olculdu: 64 MB) ve bunlari Kaggle'da kurmak, eski numpy
        ABI'sine karsi derlenmis lightgbm/catboost/shap'i bozabilirdi.
        Bu paketler Kaggle imajinda zaten var -- bkz. ``KAGGLE_DA_MEVCUT``.
    """
    print("\n[2/4] Kaggle'da olmayan paketlerin wheel'leri indiriliyor")
    hedef.mkdir(parents=True, exist_ok=True)
    sonuc = _kos(
        [
            sys.executable, "-m", "pip", "download",
            *EKSIK_PAKETLER,
            "--dest", str(hedef),
            "--only-binary=:all:",
            "--no-deps",
            "--platform", "manylinux2014_x86_64",
            "--python-version", "311",
        ],
        aciklama="wheel indirme",
    )
    adet = len(list(hedef.glob("*.whl")))
    if sonuc.returncode == 0:
        print(f"  OK: {adet} wheel (bagimliliklar KASITLI atlandi)")
        print(f"     Kaggle'da hazir varsayilanlar: {', '.join(KAGGLE_DA_MEVCUT)}")
    else:
        print(f"  Kismi: {adet} wheel indirildi. Eksikler internetsiz calismaz.")
    return adet


def veri_kopyala(hedef: Path) -> tuple[int, list[str]]:
    """Harici veri dosyalarini paket klasorune kopyalar."""
    print("\n[3/4] Harici veri kopyalaniyor")
    hedef.mkdir(parents=True, exist_ok=True)
    kopyalanan, eksik = 0, []
    for gorece in VERI_DOSYALARI:
        kaynak = ROOT / gorece
        if not kaynak.exists():
            eksik.append(gorece)
            continue
        shutil.copy2(kaynak, hedef / kaynak.name)
        kopyalanan += 1
        print(f"  + {kaynak.name} ({kaynak.stat().st_size / 1024:.0f} KB)")
    for gorece in eksik:
        print(f"  - EKSIK: {gorece}")
    return kopyalanan, eksik


def metadata_yaz(klasor: Path, kullanici: str) -> Path:
    """Kaggle Dataset metadata dosyasini yazar."""
    yol = klasor / "dataset-metadata.json"
    yol.write_text(
        json.dumps(
            {
                "title": DATASET_BASLIK,
                "id": f"{kullanici}/{DATASET_SLUG}",
                "licenses": [{"name": "CC0-1.0"}],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return yol


def kaggle_kullanici() -> str | None:
    """~/.kaggle/kaggle.json icinden kullanici adini okur."""
    yol = Path.home() / ".kaggle" / "kaggle.json"
    if not yol.exists():
        return None
    try:
        return json.loads(yol.read_text(encoding="utf-8")).get("username")
    except (json.JSONDecodeError, OSError):
        return None


BOOTSTRAP = '''# ==== KAGGLE INTERNETSIZ BOOTSTRAP -- notebook'un ILK hucresi ====
# Dataset'i notebook'a "Add Input" ile ekledikten sonra bu hucre yeter.
# Her adim SESSIZ BASARISIZ OLMAZ: ne kurulduğunu ve neyin eksik oldugunu yazar.
import importlib, os, subprocess, sys
from pathlib import Path

GIRDI = "/kaggle/input/{slug}"
assert os.path.isdir(GIRDI), f"Dataset bagli degil: {{GIRDI}} -- 'Add Input' ile ekle."

# 0) Paketledigimiz wheel'ler bu paketlerin Kaggle'da HAZIR oldugunu varsayar.
#    Varsayim yanlissa simdi ogrenmek, model egitiminin ortasinda ogrenmekten iyidir.
eksik_temel = [m for m in {kaggle_da_mevcut!r} if importlib.util.find_spec(m) is None]
if eksik_temel:
    print("UYARI: Kaggle'da bekledigimiz paketler yok:", eksik_temel)
    print("       Ilgili feature aileleri calismayabilir.")

# 1) Kaggle'da olmayan paketler -- yerel wheel'den, bagimlilik COZMEDEN.
#    --no-deps sart: aksi halde pip numpy/scipy'yi yukseltmeye calisir ve
#    onlara karsi derlenmis lightgbm/catboost bozulur.
tekerlekler = sorted(str(w) for w in Path(GIRDI, "tekerlekler").glob("*.whl"))
if tekerlekler:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                        *tekerlekler, "-q"], capture_output=True, text=True)
    durum = "OK" if r.returncode == 0 else "HATA"
    print(f"tekerlekler: {{len(tekerlekler)}} wheel, kurulum {{durum}}")
    if r.returncode != 0:
        print(r.stderr[:400])

# 2) gridup paketi
gridup_whl = sorted(str(w) for w in Path(GIRDI).glob("gridup-*.whl"))
assert gridup_whl, "gridup wheel'i dataset'te yok."
r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                    gridup_whl[0], "-q"], capture_output=True, text=True)
assert r.returncode == 0, f"gridup kurulamadi: {{r.stderr[:400]}}"

import gridup
print("gridup", gridup.__version__, "hazir")

# 3) Harici veri -- internet YOK, API cagirma; parquet'ten oku
import pandas as pd
HAVA    = pd.read_parquet(os.path.join(GIRDI, "hava_gunluk.parquet"))
GUNES   = pd.read_parquet(os.path.join(GIRDI, "gunes_gunluk.parquet"))
ILCELER = pd.read_parquet(os.path.join(GIRDI, "ilceler_gdz_adm.parquet"))
print(f"hava={{HAVA.shape}} gunes={{GUNES.shape}} ilceler={{ILCELER.shape}}")
# ==== BOOTSTRAP SONU ====
'''


def main() -> int:
    ayristirici = argparse.ArgumentParser(description=__doc__)
    ayristirici.add_argument("--wheels", action="store_true",
                             help="Kaggle'da olmayan paketlerin wheel'lerini de indir")
    ayristirici.add_argument("--upload", action="store_true",
                             help="Hazirlandiktan sonra Kaggle'a yukle")
    args = ayristirici.parse_args()

    print("=" * 68)
    print("KAGGLE INTERNETSIZ PAKET HAZIRLIGI")
    print("=" * 68)

    if CIKTI.exists():
        shutil.rmtree(CIKTI)
    CIKTI.mkdir(parents=True)

    wheel = wheel_uret()
    if wheel is None:
        return 1
    shutil.copy2(wheel, CIKTI / wheel.name)

    wheel_adedi = bagimlilik_wheel_indir(CIKTI / "tekerlekler") if args.wheels else 0
    if not args.wheels:
        print("\n[2/4] Bagimlilik wheel'leri ATLANDI (--wheels ile indirilir)")
        print(f"      Etkilenen feature aileleri: {', '.join(EKSIK_PAKETLER)}")

    kopyalanan, eksik = veri_kopyala(CIKTI)

    print("\n[4/4] Metadata ve bootstrap")
    kullanici = kaggle_kullanici()
    if kullanici is None:
        print("  ~/.kaggle/kaggle.json okunamadi -- metadata KULLANICI_ADI ile yazildi.")
        kullanici = "KULLANICI_ADI"
    metadata_yaz(CIKTI, kullanici)
    print(f"  + dataset-metadata.json  (id: {kullanici}/{DATASET_SLUG})")

    bootstrap_yolu = CIKTI / "notebook_bootstrap.py"
    bootstrap_yolu.write_text(
        BOOTSTRAP.format(slug=DATASET_SLUG, kaggle_da_mevcut=list(KAGGLE_DA_MEVCUT)),
        encoding="utf-8",
    )
    print(f"  + {bootstrap_yolu.name}")

    toplam_mb = sum(p.stat().st_size for p in CIKTI.rglob("*") if p.is_file()) / 1024 / 1024

    print("\n" + "=" * 68)
    print(f"HAZIR: {CIKTI}")
    print(f"  wheel        : {wheel.name}")
    print(f"  bagimlilik   : {wheel_adedi} wheel")
    print(f"  veri dosyasi : {kopyalanan}/{len(VERI_DOSYALARI)}")
    print(f"  toplam       : {toplam_mb:.1f} MB  (Kaggle Dataset siniri 20 GB)")
    if eksik:
        print(f"  EKSIK        : {', '.join(eksik)}")
    print("=" * 68)

    if args.upload:
        print("\nKaggle'a yukleniyor...")
        sonuc = _kos(
            ["kaggle", "datasets", "create", "-p", str(CIKTI), "--dir-mode", "zip"],
            aciklama="dataset olusturma",
        )
        if sonuc.returncode == 0:
            print("  OK. Notebook'ta 'Add Input' ile bagla.")
        else:
            print("  Dataset zaten varsa guncelle:")
            print(f"    kaggle datasets version -p {CIKTI} -m 'guncelleme' --dir-mode zip")
    else:
        print("\nYUKLEME (hazir oldugunda):")
        print(f"  kaggle datasets create -p {CIKTI} --dir-mode zip")
        print("\nSonra notebook'un ilk hucresine notebook_bootstrap.py icerigini yapistir.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
