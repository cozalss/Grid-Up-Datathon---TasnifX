"""EKIP KURULUM DOKTORU testleri: ``scripts/ekip_kontrol.py``.

Betik, repoyu yeni klonlayan ekip uyesinin TEK KOMUTLA kosacagi seydir;
yalan soylerse kurulum hatalari veri gunune kadar gizli kalir. Bu yuzden
uc sey kilitlenir:

  1. SOZLESME: exit kodu cikti ile tutarli (FAIL yoksa 0, varsa 1),
     yedi kontrolun yedisi de ciktida gorunur.
  2. GIZLILIK: kaggle.json'in 'key' alani HICBIR kosulda basilmaz.
  3. TURETME: zorunlu paket listesi pyproject.toml'dan okunur -- elle
     tutulan liste pyproject degisince sessizce eskirdi.

OLCULDU (bu makinede): betik 7/7 PASS, EXIT=0, 2.5 sn; paketler 16/16
(13 pyproject + pytest/ruff/hypothesis), duman MAE=8.03 (< 0.1 sn).
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))


def _betigi_yukle():
    """``scripts/ekip_kontrol.py``yi modul olarak yukler (paket disinda duruyor).

    sys.modules kaydi SART: betik ``from __future__ import annotations`` +
    ``@dataclass`` kullanir; dataclasses alan tiplerini cozmek icin modulu
    sys.modules'ta arar, bulamayinca AttributeError verir (olculdu).
    """
    spec = importlib.util.spec_from_file_location(
        "ekip_kontrol_betigi", KOK / "scripts" / "ekip_kontrol.py"
    )
    modul = importlib.util.module_from_spec(spec)
    sys.modules["ekip_kontrol_betigi"] = modul
    spec.loader.exec_module(modul)
    return modul


DOKTOR = _betigi_yukle()

VERI_HAZIR = all((KOK / gorece).is_file() for gorece, _ in DOKTOR.VERI_VARLIKLARI)
KAGGLE_YOLU = (
    Path(os.environ.get("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle"))) / "kaggle.json"
)

KONTROL_BASLIKLARI = (
    "1) python surumu",
    "2) zorunlu paketler",
    "3) gridup paketi",
    "4) veri varliklari",
    "5) kaggle.json",
    "6) mini duman",
    "7) konsol encoding",
)


def _betigi_kos(ek_ortam: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    ortam = {**os.environ, "PYTHONIOENCODING": "utf-8", **(ek_ortam or {})}
    return subprocess.run(
        [sys.executable, str(KOK / "scripts" / "ekip_kontrol.py")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(KOK),
        timeout=300,
        check=False,
        env=ortam,
    )


# ---------------------------------------------------------------------------
# TURETME -- paket listesi pyproject.toml'dan gelir, elle yazilmaz
# ---------------------------------------------------------------------------


def test_paket_listesi_pyprojecttan_turetiliyor():
    """Cekirdek + models/search/features/io gruplari girer; neural/viz girmez.

    OLCULDU: 13 paket (numpy..requests); torch ve matplotlib listede YOK --
    torch Kaggle imajinda zaten var, matplotlib hatti dogrulamak icin gerekmez.
    """
    paketler = DOKTOR.paket_listesi(KOK / "pyproject.toml")
    adlar = [ad for ad, _ in paketler]

    for zorunlu in (
        "numpy",
        "pandas",
        "scikit-learn",
        "lightgbm",
        "xgboost",
        "catboost",
        "optuna",
        "shap",
        "pvlib",
        "hijridate",
    ):
        assert zorunlu in adlar, f"{zorunlu} pyproject'ten okunmali"
    assert "torch" not in adlar, "neural grubu bilincli olarak kontrol disi"
    assert "matplotlib" not in adlar, "viz grubu bilincli olarak kontrol disi"
    assert len(adlar) == len(set(adlar)), "full grubu cift sayima yol acmamali"


def test_paket_listesi_alt_sinirlari_tasiyor():
    """Surum kiyasi icin alt sinir pyproject'teki '>=' degerinden gelir."""
    paketler = dict(DOKTOR.paket_listesi(KOK / "pyproject.toml"))

    assert paketler["pandas"] == "2.0"
    assert paketler["lightgbm"] == "4.0"


def test_gelistirme_araclari_repodan_turetiliyor():
    """pytest/ruff [tool.*] bolumlerinden, hypothesis conftest import'undan.

    Bunlar pyproject bagimliligi degildir (Kaggle imajina kurulmazlar) ama
    test paketini kosacak ekip uyesi icin sarttir.
    """
    araclar = DOKTOR.gelistirme_araclari(KOK / "pyproject.toml", KOK / "tests" / "conftest.py")

    assert araclar == ["pytest", "ruff", "hypothesis"]


def test_surum_demeti_posta_ekli_ve_kisa_surumleri_kiyasliyor():
    """'2.0.2.post1' gibi ekler ve '14.0' gibi kisa surumler tuple'a oturmali."""
    assert DOKTOR._surum_demeti("2.0.2.post1") == (2, 0, 2)
    assert DOKTOR._surum_demeti("14.0") == (14, 0, 0)
    assert DOKTOR._surum_demeti("3.0.3") > DOKTOR._surum_demeti("2.1")


# ---------------------------------------------------------------------------
# TEKIL KONTROLLER
# ---------------------------------------------------------------------------


def test_python_surumu_bu_ortamda_geciyor():
    kontrol = DOKTOR.python_surumu_kontrol()

    assert kontrol.gecti, "test 3.11+ ile kosuluyor; kontrol de gecmeli"
    assert sys.version.split()[0] in kontrol.detay
    assert "python3 DEGIL" in kontrol.duzeltme, "Windows Store stub tuzagi anlatilmali"


def test_veri_kontrol_eksik_dosyada_indirme_komutu_basiyor(tmp_path):
    """Eksik dosya sadece 'yok' demez -- NEREDEN gelecegini soyler."""
    varliklar = (("data/olmayan.parquet", "SAHTE_INDIRME_KOMUTU"),)

    kontrol = DOKTOR.veri_kontrol(varliklar=varliklar, kok=tmp_path)

    assert not kontrol.gecti
    assert any("EKSIK" in satir and "SAHTE_INDIRME_KOMUTU" in satir for satir in kontrol.satirlar)
    assert kontrol.duzeltme is not None


@pytest.mark.skipif(not VERI_HAZIR, reason="veri varliklari bu makinede indirilmemis")
def test_veri_kontrol_gercek_dosyalarda_satir_sayisi_olcuyor():
    """OLCULDU: hava 231,648 / gunes 233,760 / ilceler 96 satir."""
    kontrol = DOKTOR.veri_kontrol()

    assert kontrol.gecti
    assert kontrol.detay == "3/3 dosya hazir"
    for satir in kontrol.satirlar:
        assert "satir" in satir, f"her dosya icin satir sayisi basilmali: {satir!r}"


def test_kaggle_kontrol_anahtari_asla_basmiyor(tmp_path, monkeypatch):
    """GIZLILIK SOZLESMESI: 'key' alani kontrol ciktisinin hicbir yerine sizamaz."""
    (tmp_path / "kaggle.json").write_text(
        json.dumps({"username": "takim_uyesi", "key": "COK_GIZLI_ANAHTAR_1234"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))

    kontrol = DOKTOR.kaggle_kontrol()
    tum_cikti = " ".join([kontrol.detay, *kontrol.satirlar, kontrol.duzeltme or ""])

    assert kontrol.gecti
    assert "takim_uyesi" in kontrol.detay
    assert "COK_GIZLI_ANAHTAR_1234" not in tum_cikti


def test_kaggle_kontrol_dosya_yokken_token_talimati_veriyor(tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path / "bos"))

    kontrol = DOKTOR.kaggle_kontrol()

    assert not kontrol.gecti
    assert "Create New Token" in kontrol.duzeltme


def test_encoding_cp1254_ortaminda_fail_ve_setx_onerisi():
    """PYTHONIOENCODING ayarsiz + cp1254 konsol = FAIL; tedavi tek komut."""
    kontrol = DOKTOR.encoding_kontrol(
        ortam="", stdout_kodlamasi="cp1254", utf8_modu=False, tercih="cp1254"
    )

    assert not kontrol.gecti
    assert "cp1254" in kontrol.detay
    assert "setx PYTHONIOENCODING utf-8" in kontrol.duzeltme
    assert any("cp1254" in satir for satir in kontrol.satirlar), "locale tuzagi anlatilmali"


def test_encoding_pythonioencoding_ayarliyken_geciyor():
    """YANLIS-POZITIF KORUMASI: env degiskeni ayarliysa cp1254 locale FAIL uretmez."""
    kontrol = DOKTOR.encoding_kontrol(
        ortam="utf-8", stdout_kodlamasi="cp1254", utf8_modu=False, tercih="cp1254"
    )

    assert kontrol.gecti


# ---------------------------------------------------------------------------
# SOZLESME -- betik subprocess ile: exit kodu + cikti yapisi + gizlilik
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not (VERI_HAZIR and KAGGLE_YOLU.is_file()),
    reason="tam PASS ancak veri + kaggle.json hazir makinede beklenir",
)
def test_betik_hazir_makinede_yedi_kontrolu_de_gecip_sifir_donuyor():
    """OLCULDU: 'SONUC: 7/7 PASS', EXIT=0, 2.5 sn."""
    kosu = _betigi_kos()

    assert kosu.returncode == 0, kosu.stdout[-2000:] + kosu.stderr[-500:]
    for baslik in KONTROL_BASLIKLARI:
        assert baslik in kosu.stdout, f"kontrol basligi eksik: {baslik}"
    assert kosu.stdout.count("[PASS]") == 7
    assert "[FAIL]" not in kosu.stdout
    assert "SONUC: 7/7 PASS" in kosu.stdout

    # GIZLILIK: gercek kaggle anahtari ciktinin hicbir yerinde olamaz.
    anahtar = json.loads(KAGGLE_YOLU.read_text(encoding="utf-8")).get("key", "")
    if anahtar:
        assert anahtar not in kosu.stdout
        assert anahtar not in kosu.stderr


@pytest.mark.slow
def test_eksik_kaggle_json_exit_bir_ve_fail_satiri_basiyor(tmp_path):
    """Tek bir FAIL bile exit kodunu 1 yapmali -- CI bu kodu okuyacak.

    kaggle.json kontrolu KAGGLE_CONFIG_DIR ile bos dizine yonlendirilerek
    deterministik bicimde dusuruluyor; kalan alti kontrol makineye bagli
    kalmaya devam eder ama sozlesme onlardan bagimsizdir.
    """
    kosu = _betigi_kos({"KAGGLE_CONFIG_DIR": str(tmp_path / "bos")})

    assert kosu.returncode == 1, kosu.stdout[-2000:]
    assert "[FAIL] 5) kaggle.json" in kosu.stdout
    assert "Create New Token" in kosu.stdout
    assert "FAIL" in kosu.stdout.split("SONUC:")[-1], "ozet satiri FAIL sayisini soylemeli"


@pytest.mark.slow
def test_exit_kodu_cikti_ile_tutarli(tmp_path):
    """SOZLESMENIN KENDISI: [FAIL] yoksa 0, varsa 1 -- ikisi asla ayrisamaz."""
    hazir = _betigi_kos()
    bozuk = _betigi_kos({"KAGGLE_CONFIG_DIR": str(tmp_path / "bos")})

    for kosu in (hazir, bozuk):
        assert (kosu.returncode == 0) == ("[FAIL]" not in kosu.stdout), kosu.stdout[-1500:]
        assert kosu.stdout.count("[PASS]") + kosu.stdout.count("[FAIL]") == 7
        assert "SONUC:" in kosu.stdout
