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
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from security.verify_sources import verify_manifest

CIKTI = ROOT / "kaggle_paket"
WHEEL_MANIFEST = ROOT / "security" / "wheel-manifest.json"
SOURCE_MANIFEST = ROOT / "data" / "sources.yml"


#: Dataset icine kopyalanacak veri dosyalari -- KAYNAK MANIFESTINDEN turetilir.
#: Elle liste tutulunca manifest ve paket ayristi (olculdu, 2026-08-18: izsu
#: manifestte var pakette yok; turizm_aylik_il yalnizca listede). Manifestte
#: olmayan bir dosya pakete GIREMEZ: lisans/hash kaydi olmayan veri
#: dagitilmaz (verify_sources kapisi da ayni kumeye bakar).
def _manifest_veri_dosyalari(manifest_yolu: Path = SOURCE_MANIFEST) -> tuple[str, ...]:
    """sources.yml artefakt yollarini (manifest sirasiyla) dondurur."""
    manifest = json.loads(manifest_yolu.read_text(encoding="utf-8"))
    yollar = tuple(str(kayit["path"]) for kayit in manifest.get("artifacts", []))
    if not yollar:
        raise RuntimeError(f"{manifest_yolu}: artefakt listesi bos; paket kurulamaz.")
    return yollar


VERI_DOSYALARI = _manifest_veri_dosyalari()

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


class SupplyChainError(RuntimeError):
    """Dogrulanmamis artifact veya provenance yayin/kurulumu durdurdu."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_oku(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SupplyChainError(f"Manifest okunamadi: {path} ({type(error).__name__})") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise SupplyChainError(f"Manifest schema_version=1 olmali: {path}")
    return payload


def _wheel_specs(manifest_path: Path = WHEEL_MANIFEST) -> list[dict[str, str]]:
    payload = _json_oku(manifest_path)
    specs = payload.get("wheels")
    if not isinstance(specs, list) or not specs:
        raise SupplyChainError(f"Wheel manifesti bos: {manifest_path}")
    return specs


def dogrula_wheel_manifesti(wheels: list[Path], manifest_path: Path = WHEEL_MANIFEST) -> list[Path]:
    """Wheel kumesini tam dosya adi, surum ve SHA256 ile fail-closed dogrular."""
    specs = _wheel_specs(manifest_path)
    expected: dict[str, dict[str, str]] = {}
    for spec in specs:
        name = str(spec.get("name", "")).strip()
        version = str(spec.get("version", "")).strip()
        filename = str(spec.get("filename", "")).strip()
        digest = str(spec.get("sha256", "")).strip().lower()
        if digest == "unverified" or not digest:
            raise SupplyChainError(f"dogrulanmamis wheel hash'i: {filename or name}")
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise SupplyChainError(f"gecersiz SHA256: {filename or name}")
        normalized = filename.lower().replace("-", "_")
        prefix = f"{name.lower().replace('-', '_')}_{version.lower().replace('-', '_')}"
        if not name or not version or not filename or not normalized.startswith(prefix):
            raise SupplyChainError(f"gecersiz wheel ad/surum sozlesmesi: {filename}")
        if filename in expected:
            raise SupplyChainError(f"tekrarlanan wheel manifest girdisi: {filename}")
        expected[filename] = spec

    actual = {path.name: path for path in wheels}
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise SupplyChainError(f"Wheel kumesi manifestle uyusmuyor; eksik={missing}, fazla={extra}")

    for filename, spec in expected.items():
        actual_digest = _sha256(actual[filename])
        if actual_digest != spec["sha256"].lower():
            raise SupplyChainError(
                f"SHA256 uyusmazligi: {filename}; beklenen={spec['sha256']}, gercek={actual_digest}"
            )
    return [actual[name] for name in sorted(actual)]


def offline_manifest_yaz(gridup_wheel: Path, dependency_wheels: list[Path], output: Path) -> Path:
    """Dogrulanmis ucuncu taraf + yeni uretilen gridup wheel'i icin manifest yazar."""
    if dependency_wheels:
        dependency_wheels = dogrula_wheel_manifesti(dependency_wheels)
    static = {spec["filename"]: spec for spec in _wheel_specs()}
    entries: list[dict[str, str]] = []
    for wheel in sorted(dependency_wheels):
        spec = static[wheel.name]
        entries.append({**spec, "path": f"tekerlekler/{wheel.name}"})

    match = re.match(r"(?P<name>.+)-(?P<version>\d[^-]*)-", gridup_wheel.name)
    if match is None:
        raise SupplyChainError(f"gridup wheel adi taninamadi: {gridup_wheel.name}")
    entries.append(
        {
            "name": match.group("name"),
            "version": match.group("version"),
            "filename": gridup_wheel.name,
            "path": gridup_wheel.name,
            "sha256": _sha256(gridup_wheel),
        }
    )
    output.write_text(
        json.dumps({"schema_version": 1, "wheels": entries}, indent=2),
        encoding="utf-8",
    )
    return output


def yayin_kapisini_dogrula(manifest_path: Path = SOURCE_MANIFEST, *, root: Path = ROOT) -> None:
    """Kanonik verifier ile hash, sema, lisans ve immutable kaynagi zorlar."""
    result = verify_manifest(
        manifest_path,
        root=root,
        publication=True,
        check_files=True,
    )
    if result.errors:
        raise SupplyChainError("yayin kapisi reddetti: " + "; ".join(result.errors))


def atomik_dizin_yayinla(staging: Path, target: Path) -> None:
    """Tam hazir staging dizinini geri alinabilir bir dizin takasiyla yayinlar."""
    if not staging.is_dir():
        raise FileNotFoundError(f"staging dizini yok: {staging}")
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = target.with_name(f".{target.name}.backup-{uuid.uuid4().hex}")
    had_target = target.exists()
    if had_target:
        os.replace(target, backup)  # noqa: PTH105 -- atomik dizin primitive'i
    try:
        os.replace(staging, target)  # noqa: PTH105 -- atomik dizin primitive'i
    except BaseException:
        if had_target and backup.exists() and not target.exists():
            os.replace(backup, target)  # noqa: PTH105 -- rollback ayni primitive
        raise
    if backup.exists():
        shutil.rmtree(backup)


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


#: Kaggle CLI'nin basarisizken bile stdout'a yazip cikis kodu 0 dondurdugu
#: metinler. Cikis koduna guvenmek OLCULDU ve yaniltti: mevcut bir dataset'te
#: ``datasets create`` "dataset already exists" yazip 0 ile cikiyor.
_KAGGLE_HATA_IZLERI = ("already exists", "error", "not found", "403", "401", "traceback")


def _kaggle_ciktisi_basarili(sonuc: subprocess.CompletedProcess[str]) -> bool:
    """Cikis kodu VE cikti metni birlikte degerlendirilir."""
    if sonuc.returncode != 0:
        return False
    metin = ((sonuc.stdout or "") + (sonuc.stderr or "")).lower()
    return not any(iz in metin for iz in _KAGGLE_HATA_IZLERI)


def _kaggle_son_guncelleme(ref: str) -> str | None:
    """Dataset'in Kaggle'daki ``lastUpdated`` degeri; bulunamazsa ``None``."""
    sonuc = subprocess.run(
        ["kaggle", "datasets", "list", "-m", "-s", ref.rsplit("/", maxsplit=1)[-1]],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=ROOT,
        check=False,
    )
    for satir in (sonuc.stdout or "").splitlines():
        if satir.startswith(ref):
            parcalar = satir.split()
            # ref title... size lastUpdated(tarih saat) indirme oy puan
            return " ".join(parcalar[-5:-3]) if len(parcalar) >= 5 else None
    return None


def _kaggle_yayinla() -> bool:
    """Dataset'i olusturur ya da SURUMLER; sonucu Kaggle'a SORARAK dogrular.

    Onceki hali ``datasets create``in donus kodunu basari sayiyordu. Kaggle CLI
    mevcut bir dataset'te "dataset already exists" yazip yine de 0 ile cikiyor;
    bu yuzden ``version`` yedegine hic dusulmuyor ve betik "OK" basarken
    dataset GUNCELLENMEMIS kaliyordu. Yarisma gunu internetsiz notebook'un eski
    wheel'i kurmasi demektir bu -- tam olarak bu deponun her yerde kacindigi
    sessiz basarisizlik bicimi.
    """
    kimlik = json.loads((CIKTI / "dataset-metadata.json").read_text(encoding="utf-8"))
    ref = str(kimlik["id"])
    onceki = _kaggle_son_guncelleme(ref)
    mevcut = onceki is not None
    print(
        f"  dataset {'VAR' if mevcut else 'YOK'}: {ref}" + (f" (son: {onceki})" if onceki else "")
    )

    if mevcut:
        sonuc = _kos(
            [
                "kaggle",
                "datasets",
                "version",
                "-p",
                str(CIKTI),
                "-m",
                f"dogrulanmis guncelleme {datetime.now(timezone.utc):%Y-%m-%d %H:%M}",
                "--dir-mode",
                "zip",
            ],
            aciklama="dataset surumleme",
        )
    else:
        sonuc = _kos(
            ["kaggle", "datasets", "create", "-p", str(CIKTI), "--dir-mode", "zip"],
            aciklama="dataset olusturma",
        )

    if not _kaggle_ciktisi_basarili(sonuc):
        print("  HATA: Kaggle yukleme basarisiz.")
        print("   ", ((sonuc.stdout or "") + (sonuc.stderr or ""))[:600])
        return False

    # Kaggle surum olusturmayi ASENKRON isler: komut doner, dataset birkac
    # saniye sonra guncellenir. Bu yuzden ANINDA lastUpdated karsilastirmasi
    # yanlis negatif verir (olculdu). Once Kaggle'in kendi kabul cumlesini
    # arariz, sonra kisa sure yoklariz.
    kabul = "being created" in ((sonuc.stdout or "") + (sonuc.stderr or "")).lower()
    if not kabul:
        print("  HATA: Kaggle yuklemeyi kabul ettigini bildirmedi.")
        print("   ", ((sonuc.stdout or "") + (sonuc.stderr or ""))[-600:])
        return False

    for _ in range(12):  # ~60 sn
        sonraki = _kaggle_son_guncelleme(ref)
        if not mevcut or sonraki != onceki:
            print(f"  OK. Dataset guncellendi (lastUpdated: {sonraki}).")
            print("      Notebook'ta 'Add Input' ile bagla.")
            return True
        time.sleep(5)

    print(
        "  UYARI: Kaggle yuklemeyi KABUL ETTI ama 60 sn icinde yeni surum gorunmedi.\n"
        f"         Bu normal olabilir (asenkron isleme). ELLE DOGRULA:\n"
        f"         https://www.kaggle.com/datasets/{ref}"
    )
    return True


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
    specs = _wheel_specs()
    pinned = [f"{spec['name']}=={spec['version']}" for spec in specs]
    sonuc = _kos(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            *pinned,
            "--dest",
            str(hedef),
            "--only-binary=:all:",
            "--no-deps",
            "--platform",
            "manylinux2014_x86_64",
            "--python-version",
            "311",
        ],
        aciklama="wheel indirme",
    )
    if sonuc.returncode != 0:
        raise SupplyChainError("Wheel indirme basarisiz; kismi artifact yayinlanmayacak.")
    wheels = list(hedef.glob("*.whl"))
    verified = dogrula_wheel_manifesti(wheels)
    print(f"  OK: {len(verified)} wheel; tam surum + SHA256 dogrulandi")
    print(f"     Kaggle'da hazir varsayilanlar: {', '.join(KAGGLE_DA_MEVCUT)}")
    return len(verified)


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
                # Paket farkli lisanslara sahip veri snapshot'lari tasir. Kaggle
                # metadata'sinda tek ve yanlis bir CC0 iddiasi yerine provenance
                # dosyasina yonlendiren fail-safe sinif kullanilir.
                "licenses": [{"name": "other"}],
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


BOOTSTRAP = """# ==== KAGGLE INTERNETSIZ BOOTSTRAP -- notebook'un ILK hucresi ====
# Dataset'i notebook'a "Add Input" ile ekledikten sonra bu hucre yeter.
# Her adim SESSIZ BASARISIZ OLMAZ: ne kurulduğunu ve neyin eksik oldugunu yazar.
import hashlib, importlib, json, os, subprocess, sys
from pathlib import Path

GIRDI = "/kaggle/input/{slug}"
assert os.path.isdir(GIRDI), f"Dataset bagli degil: {{GIRDI}} -- 'Add Input' ile ekle."

# 0) Hicbir wheel glob ile guvenilir sayilmaz. Manifest tam dosya adi ve
#    SHA256'i dogrulamadan pip CALISMAZ.
manifest_yolu = Path(GIRDI, "wheel-manifest.json")
assert manifest_yolu.is_file(), "wheel-manifest.json yok; kurulum fail-closed durdu."
manifest = json.loads(manifest_yolu.read_text(encoding="utf-8"))
assert manifest.get("schema_version") == 1, "wheel manifest semasi desteklenmiyor."
dogrulanmis = []
for kayit in manifest.get("wheels", []):
    wheel = Path(GIRDI, kayit["path"])
    beklenen = kayit.get("sha256", "")
    assert len(beklenen) == 64 and beklenen != "unverified", f"hash dogrulanmamis: {{wheel.name}}"
    assert wheel.is_file(), f"manifest wheel'i yok: {{wheel}}"
    gercek = hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert gercek == beklenen, f"wheel SHA256 uyusmazligi: {{wheel.name}}"
    dogrulanmis.append(wheel)

# 1) Paketledigimiz wheel'ler bu paketlerin Kaggle'da HAZIR oldugunu varsayar.
#    Varsayim yanlissa simdi ogrenmek, model egitiminin ortasinda ogrenmekten iyidir.
eksik_temel = [m for m in {kaggle_da_mevcut!r} if importlib.util.find_spec(m) is None]
if eksik_temel:
    print("UYARI: Kaggle'da bekledigimiz paketler yok:", eksik_temel)
    print("       Ilgili feature aileleri calismayabilir.")

# 2) Kaggle'da olmayan paketler -- yerel wheel'den, bagimlilik COZMEDEN.
#    --no-deps sart: aksi halde pip numpy/scipy'yi yukseltmeye calisir ve
#    onlara karsi derlenmis lightgbm/catboost bozulur.
tekerlekler = sorted(str(w) for w in dogrulanmis if w.parent.name == "tekerlekler")
if tekerlekler:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                        *tekerlekler, "-q"], capture_output=True, text=True)
    durum = "OK" if r.returncode == 0 else "HATA"
    print(f"tekerlekler: {{len(tekerlekler)}} wheel, kurulum {{durum}}")
    if r.returncode != 0:
        print(r.stderr[:400])

# 3) gridup paketi -- o da ayni manifest dogrulamasindan gecmistir.
gridup_whl = [
    str(w)
    for w in dogrulanmis
    if w.parent == Path(GIRDI) and w.name.startswith("gridup-")
]
assert len(gridup_whl) == 1, "manifestte tam bir gridup wheel'i olmali."
r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                    gridup_whl[0], "-q"], capture_output=True, text=True)
assert r.returncode == 0, f"gridup kurulamadi: {{r.stderr[:400]}}"

import gridup
print("gridup", gridup.__version__, "hazir")

# 4) Harici veri -- internet YOK, API cagirma; parquet'ten oku
import pandas as pd
HAVA    = pd.read_parquet(os.path.join(GIRDI, "hava_gunluk.parquet"))
GUNES   = pd.read_parquet(os.path.join(GIRDI, "gunes_gunluk.parquet"))
ILCELER = pd.read_parquet(os.path.join(GIRDI, "ilceler_gdz_adm.parquet"))
print(f"hava={{HAVA.shape}} gunes={{GUNES.shape}} ilceler={{ILCELER.shape}}")
# ==== BOOTSTRAP SONU ====
"""


def _staging_paketi_hazirla(
    paket: Path, *, wheels: bool
) -> tuple[Path, int, int, list[str]] | None:
    """Paketi hedefe dokunmadan staging dizininde eksiksiz hazirlar."""
    wheel = wheel_uret()
    if wheel is None:
        return None
    shutil.copy2(wheel, paket / wheel.name)

    wheel_adedi = bagimlilik_wheel_indir(paket / "tekerlekler") if wheels else 0
    if not wheels:
        print("\n[2/4] Bagimlilik wheel'leri ATLANDI (--wheels ile indirilir)")
        print(f"      Etkilenen feature aileleri: {', '.join(EKSIK_PAKETLER)}")

    kopyalanan, eksik = veri_kopyala(paket)

    print("\n[4/4] Metadata ve bootstrap")
    kullanici = kaggle_kullanici()
    if kullanici is None:
        print("  ~/.kaggle/kaggle.json okunamadi -- metadata KULLANICI_ADI ile yazildi.")
        kullanici = "KULLANICI_ADI"
    metadata_yaz(paket, kullanici)
    print(f"  + dataset-metadata.json  (id: {kullanici}/{DATASET_SLUG})")

    dependency_wheels = sorted((paket / "tekerlekler").glob("*.whl"))
    offline_manifest_yaz(paket / wheel.name, dependency_wheels, paket / "wheel-manifest.json")
    shutil.copy2(SOURCE_MANIFEST, paket / "sources.yml")
    print("  + wheel-manifest.json  (tam surum + SHA256)")
    print("  + sources.yml          (kaynak/lisans/provenance)")

    bootstrap_yolu = paket / "notebook_bootstrap.py"
    bootstrap_yolu.write_text(
        BOOTSTRAP.format(slug=DATASET_SLUG, kaggle_da_mevcut=list(KAGGLE_DA_MEVCUT)),
        encoding="utf-8",
    )
    print(f"  + {bootstrap_yolu.name}")
    return wheel, wheel_adedi, kopyalanan, eksik


def main() -> int:
    ayristirici = argparse.ArgumentParser(description=__doc__)
    ayristirici.add_argument(
        "--wheels", action="store_true", help="Kaggle'da olmayan paketlerin wheel'lerini de indir"
    )
    ayristirici.add_argument(
        "--upload", action="store_true", help="Hazirlandiktan sonra Kaggle'a yukle"
    )
    args = ayristirici.parse_args()

    print("=" * 68)
    print("KAGGLE INTERNETSIZ PAKET HAZIRLIGI")
    print("=" * 68)

    with tempfile.TemporaryDirectory(prefix=f".{CIKTI.name}.staging-", dir=ROOT) as staging:
        hazirlanan = _staging_paketi_hazirla(Path(staging), wheels=args.wheels)
        if hazirlanan is None:
            return 1
        wheel, wheel_adedi, kopyalanan, eksik = hazirlanan
        atomik_dizin_yayinla(Path(staging), CIKTI)

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
        if eksik:
            raise SupplyChainError(f"yayin kapisi: eksik veri artifactleri: {eksik}")
        if wheel_adedi != len(_wheel_specs()):
            raise SupplyChainError(
                "yayin kapisi: tum dogrulanmis dependency wheel'leri --wheels ile paketlenmeli"
            )
        yayin_kapisini_dogrula()
        print("\nKaggle'a yukleniyor...")
        if not _kaggle_yayinla():
            return 1
    else:
        print("\nYUKLEME: Tum yayin kapilari icin betigi --wheels --upload ile yeniden calistir.")
        print("\nSonra notebook'un ilk hucresine notebook_bootstrap.py icerigini yapistir.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
