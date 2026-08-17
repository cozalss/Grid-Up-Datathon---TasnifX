"""VERI GUNUNUN ILK BETIGI: ``scripts/day_one.py`` regresyon testleri.

Bu betik 21 Agustos'ta CALISAN ILK SEYDIR. Buradaki bir hata 12 gunun ilk
saatini yer, o yuzden her duzeltme OLCULMUS sayilarla kilitlenir.

Kapsanan bulgular:
  1. Zaman kolonu tespiti (ZATEN KAPALI -- geri gitmesin diye korunuyor)
  2. LOG uzayindaki CV skoru ile HAM uzaydaki baseline'in yan yana basilmasi
  3. ``find_files``: 'test' alt dizgisinin 'latest' icinde eslesmesi
  4. ``--yes`` bayraginin KRITIK sizinti kapisini da acmasi
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))

from gridup.models import CVResult  # noqa: E402
from gridup.profiling import profile  # noqa: E402
from gridup.validation import parse_time_series, purged_time_series_split  # noqa: E402


def _betigi_yukle():
    """``scripts/day_one.py``yi modul olarak yukler (paket disinda duruyor)."""
    spec = importlib.util.spec_from_file_location("gun_bir_betigi", KOK / "scripts" / "day_one.py")
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul


def test_explicit_task_is_never_overwritten_by_profile_inference():
    report_summary = {"gorev_tahmini": "regression"}

    assert GUN1.resolve_task("binary", report_summary) == "binary"
    assert GUN1.resolve_task(None, report_summary) == "regression"


def test_official_metric_is_required_fail_closed():
    with pytest.raises(ValueError, match="resmi metrik"):
        GUN1.require_official_metric(None)

    assert GUN1.require_official_metric("MAE") == "mae"


GUN1 = _betigi_yukle()


def _dizin_kur(taban: Path, adlar: list[str]) -> Path:
    """Verilen adlarla bos ama gecerli CSV'ler yazar."""
    taban.mkdir(parents=True, exist_ok=True)
    for ad in adlar:
        (taban / ad).write_text("a,b\n1,2\n", encoding="utf-8")
    return taban


def _adlar(taban: Path) -> dict[str, str]:
    return {rol: yol.name for rol, yol in GUN1.find_files(taban).items()}


# ---------------------------------------------------------------------------
# BULGU 3 -- find_files ad eslestirmesi
# ---------------------------------------------------------------------------


def test_latest_train_dosyasi_test_olarak_secilmiyor(tmp_path):
    """'latest' kelimesi 'test' alt dizgisini ICERIR -- ayni dosya iki role gidiyordu.

    OLCULDU (['latest_train.csv', 'test.csv', 'sample_submission.csv']):
      ONCE : {'train': 'latest_train.csv', 'test': 'latest_train.csv', ...}
      SONRA: {'train': 'latest_train.csv', 'test': 'test.csv', ...}
    """
    kelime = "latest"
    assert "test" in kelime, "bulgunun dayanagi: alt dizgi gercekten esliyor"

    dizin = _dizin_kur(tmp_path / "s2", ["latest_train.csv", "test.csv", "sample_submission.csv"])
    bulunan = _adlar(dizin)

    assert bulunan["train"] == "latest_train.csv"
    assert bulunan["test"] == "test.csv"
    assert bulunan["train"] != bulunan["test"], "ayni dosya iki role atanamaz"


def test_sample_dosyasi_yokken_de_gercek_test_dosyasi_seciliyor(tmp_path):
    """Sessiz hasarin gercek yolu: sample_submission yoksa dogrulama yakalamiyordu.

    sample varken ``validate_submission`` gurultuyle duruyordu; YOKKEN betik
    500 satirlik (olmasi gereken 120) bir submission uretip EXIT=0 veriyordu.

    OLCULDU (['latest_train.csv', 'test.csv']):
      ONCE : {'train': 'latest_train.csv', 'test': 'latest_train.csv'}
      SONRA: {'train': 'latest_train.csv', 'test': 'test.csv'}
    """
    dizin = _dizin_kur(tmp_path / "s8", ["latest_train.csv", "test.csv"])
    bulunan = _adlar(dizin)

    assert bulunan == {"train": "latest_train.csv", "test": "test.csv"}


def test_turkce_harfli_egitim_dosyasi_bulunuyor(tmp_path):
    """OLCULDU (['EĞİTİM.csv', 'TEST.csv', 'ÖRNEK.csv']):

      ONCE : {'test': 'TEST.csv'}  -- train BULUNAMADI, betik EXIT=1
      SONRA: {'train': 'EĞİTİM.csv', 'test': 'TEST.csv', 'sample': 'ÖRNEK.csv'}

    ``join_key`` Turkce katlamasi ("EĞİTİM" -> "egitim") olmadan veri gununde
    dosya adi Turkce ise betik ilk adimda oluyordu.
    """
    dizin = _dizin_kur(tmp_path / "s7", ["EĞİTİM.csv", "TEST.csv", "ÖRNEK.csv"])
    bulunan = _adlar(dizin)

    assert bulunan["train"] == "EĞİTİM.csv"
    assert bulunan["test"] == "TEST.csv"
    assert bulunan["sample"] == "ÖRNEK.csv"


@pytest.mark.parametrize(
    "adlar,beklenen",
    [
        (
            ["train.csv", "test.csv", "sample_submission.csv"],
            {"train": "train.csv", "test": "test.csv", "sample": "sample_submission.csv"},
        ),
        (
            ["train.csv", "test.csv", "sampleSubmission.csv"],
            {"train": "train.csv", "test": "test.csv", "sample": "sampleSubmission.csv"},
        ),
        (
            ["GDZ_train_2023.csv", "GDZ_test_2023.csv", "ornek_gonderim.csv"],
            {
                "train": "GDZ_train_2023.csv",
                "test": "GDZ_test_2023.csv",
                "sample": "ornek_gonderim.csv",
            },
        ),
        (
            ["EGITIM.csv", "TEST.csv", "ORNEK.csv"],
            {"train": "EGITIM.csv", "test": "TEST.csv", "sample": "ORNEK.csv"},
        ),
    ],
)
def test_masum_dosya_adlari_hala_dogru_cozuluyor(tmp_path, adlar, beklenen):
    """YANLIS-POZITIF KORUMASI: kelime bazli eslesme normal adlari bozmamali.

    ``sampleSubmission.csv`` kelime ayraci ICERMEZ; kelime eslesmesi bosa
    duserse alt dizgi yedegi devreye girer. Bu satir olmadan duzeltme klasik
    Kaggle adini kaybediyordu (olculdu: SONRA {'train','test'} -- sample YOK).
    """
    dizin = _dizin_kur(tmp_path / "masum", adlar)

    assert _adlar(dizin) == beklenen


def test_ikinci_test_adayi_sessizce_atilmiyor(tmp_path, capsys):
    """Sessiz duzeltme yok: elenen aday kullaniciya SOYLENIR.

    OLCULDU (['train.csv', 'test_features.csv', 'test_labels.csv']):
      ONCE : test=test_features.csv, test_labels.csv SESSIZCE yok sayildi
      SONRA: ayni secim + "yok sayilanlar: test_labels.csv" satiri basiliyor
    """
    dizin = _dizin_kur(tmp_path / "s6", ["train.csv", "test_features.csv", "test_labels.csv"])
    bulunan = _adlar(dizin)
    cikti = capsys.readouterr().out

    assert bulunan["test"] == "test_features.csv"
    assert "yok sayilanlar" in cikti
    assert "test_labels.csv" in cikti


def test_alt_dizgi_yedegi_kullanildiginda_uyari_basiliyor(tmp_path, capsys):
    """Alt dizgi dali bir TAHMINDIR; kullanilinca sessiz kalmaz."""
    dizin = _dizin_kur(tmp_path / "camel", ["train.csv", "sampleSubmission.csv"])
    GUN1.find_files(dizin)
    cikti = capsys.readouterr().out

    assert "ALT DIZGI" in cikti
    assert "sampleSubmission.csv" in cikti


# ---------------------------------------------------------------------------
# BULGU 2 -- baseline ile CV skoru ayni uzayda mi
# ---------------------------------------------------------------------------


def _sahte_sonuc(oof: np.ndarray, kapsam: np.ndarray, skor: float) -> CVResult:
    """Yalnizca ``covered_predictions()`` + ``overall_score`` tasiyan CVResult."""
    return CVResult(
        oof_predictions=oof,
        oof_covered=kapsam,
        test_predictions=None,
        fold_scores=[skor],
        overall_score=skor,
        feature_importance=pd.DataFrame({"feature": [], "importance": []}),
        metric_name="mae",
        model_kind="lightgbm",
    )


def test_baseline_cv_skoruyla_ayni_uzayda_ve_ayni_satirlarda_olculuyor(capsys):
    """LOG uzayindaki CV skoru, HAM uzaydaki baseline ile yan yana basiliyordu.

    OLCULDU (scripts/day_one.py uctan uca, --metric mae, carpiklik 3.70):
      ONCE : 'Hep sifir' baseline mae 24.131778 (HAM, tum veri)
             CV skoru (OOF)           0.803790 (LOG, kapsanan satirlar)
             ekranda okunan iyilesme  30.02 kat  <-- tamamen sahte
      SONRA: baseline mae 0.697950 ve CV mae 0.803790 -- IKISI DE log1p
             uzayinda ve AYNI 1,500 OOF satirinda; sonuc %-15.2, GECEMEDI.
    """
    ham = np.concatenate([np.zeros(600), np.full(400, 50.0)])
    y_log = np.log1p(ham)
    kapsam = np.zeros(len(ham), dtype=bool)
    kapsam[400:] = True  # purged TimeSeriesSplit ilk donemi kapsamaz

    ham_baseline = float(np.mean(np.abs(ham)))
    sonuc = _sahte_sonuc(np.zeros(len(ham)), kapsam, skor=0.803790)
    baseline = GUN1.baseline_karsilastir(
        y_log, sonuc, metric="mae", zero_share=0.6, log_uzayinda=True
    )
    cikti = capsys.readouterr().out

    # Baseline artik HAM degil LOG uzayinda: iki sayi ayni birimde.
    assert baseline == pytest.approx(float(np.mean(np.abs(y_log[kapsam]))))
    assert baseline < ham_baseline / 5, "ham uzay baseline'i geri sizmis olmamali"
    assert "log1p uzayinda" in cikti
    assert f"ayni {int(kapsam.sum()):,} OOF satirinda" in cikti


def test_baseline_yalnizca_kapsanan_satirlarda_olculuyor():
    """Kapsam esitlenmezse iki sayi FARKLI satir kumesinde karsilastirilir.

    OLCULDU: uctan uca kosuda OOF kapsami %27.5; baseline tum veride
    olculunce 24.131778, kapsanan satirlarda 0.697950 cikiyor.
    """
    y = np.concatenate([np.full(600, 10.0), np.zeros(400)])
    kapsam = np.zeros(len(y), dtype=bool)
    kapsam[600:] = True  # yalnizca sifirlarin oldugu blok kapsanmis

    sonuc = _sahte_sonuc(np.zeros(len(y)), kapsam, skor=0.5)
    baseline = GUN1.baseline_karsilastir(y, sonuc, metric="mae", zero_share=0.4, log_uzayinda=False)

    assert baseline == pytest.approx(0.0), "kapsanan satirlarin hepsi sifir"
    assert float(np.mean(np.abs(y))) == pytest.approx(6.0), "tum veride 6.0 olurdu"


def test_ham_uzayda_kosuda_baseline_ham_kalir_ve_gecen_model_gecti_der(capsys):
    """YANLIS-POZITIF KORUMASI: log1p uygulanmayan kosu bozulmamali.

    ``log_uzayinda=False`` iken ekranda "ham uzayda" yazmali ve baseline'i
    GERCEKTEN gecen bir model "GECTI" almali -- duzeltme masum kosuyu
    kotumser hale getirmemeli.
    """
    y = np.concatenate([np.zeros(500), np.full(500, 8.0)])
    kapsam = np.ones(len(y), dtype=bool)

    sonuc = _sahte_sonuc(np.zeros(len(y)), kapsam, skor=1.0)
    baseline = GUN1.baseline_karsilastir(y, sonuc, metric="mae", zero_share=0.5, log_uzayinda=False)
    cikti = capsys.readouterr().out

    assert baseline == pytest.approx(4.0)
    assert "ham uzayda" in cikti
    assert "log1p" not in cikti
    assert "GECTI" in cikti and "GECEMEDI" not in cikti
    assert "UYARI" not in cikti


def test_baseline_gecilemediginde_acikca_uyariliyor(capsys):
    """Skor baseline'in gerisindeyse kullanici bunu OKUYARAK anlamali."""
    y = np.concatenate([np.zeros(800), np.full(200, 5.0)])
    kapsam = np.ones(len(y), dtype=bool)

    sonuc = _sahte_sonuc(np.zeros(len(y)), kapsam, skor=2.5)
    GUN1.baseline_karsilastir(y, sonuc, metric="mae", zero_share=0.8, log_uzayinda=False)
    cikti = capsys.readouterr().out

    assert "GECEMEDI" in cikti
    assert "two_stage" in cikti


def test_buyuk_daha_iyi_metrikte_karsilastirma_yonu_ters_cevriliyor(capsys):
    """``r2`` gibi metriklerde BUYUK skor iyidir -- yon sabit kodlanmamali."""
    y = np.concatenate([np.zeros(500), np.full(500, 4.0)])
    kapsam = np.ones(len(y), dtype=bool)

    sonuc = _sahte_sonuc(np.zeros(len(y)), kapsam, skor=0.75)
    baseline = GUN1.baseline_karsilastir(y, sonuc, metric="r2", zero_share=0.5, log_uzayinda=False)
    cikti = capsys.readouterr().out

    assert baseline < 0.75, "hep-sifir tahmininin r2'si modelinkinden dusuk"
    assert "GECTI" in cikti


# ---------------------------------------------------------------------------
# BULGU 4 -- --yes kritik sizinti kapisini acmamali
# ---------------------------------------------------------------------------


def _kucuk_veri(dizin: Path, *, sizintili: bool) -> Path:
    """TR tarihli (gg.aa.yyyy), saat damgali, kucuk bir olay kaydi yazar."""
    dizin.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    gunler = pd.date_range("2025-01-01", periods=90, freq="D")
    ilceler = [f"ILCE_{i:02d}" for i in range(4)]
    satirlar = []
    for sira, gun in enumerate(gunler):
        for ilce in ilceler:
            satirlar.append(
                {
                    "ID": f"R{len(satirlar):05d}",
                    "TARIH": f"{gun.day:02d}.{gun.month:02d}.{gun.year} 08:{sira % 60:02d}",
                    "ILCE": ilce,
                    "SICAKLIK": float(rng.normal(18.0, 5.0)),
                    "HEDEF": float(rng.gamma(2.0, 20.0)),
                }
            )
    frame = pd.DataFrame(satirlar)
    egitim = frame.iloc[: len(frame) - 40].copy()
    deneme = frame.iloc[len(frame) - 40 :].copy()
    if sizintili:
        egitim["GELECEK_OKUMA"] = egitim["HEDEF"] * 3.0
        deneme["GELECEK_OKUMA"] = np.linspace(1.0, 9.0, len(deneme))
    egitim.to_csv(dizin / "train.csv", index=False, encoding="utf-8")
    deneme.drop(columns=["HEDEF"]).to_csv(dizin / "test.csv", index=False, encoding="utf-8")
    return dizin


def _betigi_kos(dizin: Path, ek: list[str]) -> subprocess.CompletedProcess:
    komut = [
        sys.executable,
        str(KOK / "scripts" / "day_one.py"),
        "--data",
        str(dizin),
        "--target",
        "HEDEF",
        "--id",
        "ID",
        "--time",
        "TARIH",
        "--group",
        "ILCE",
        "--metric",
        "mae",
        "--yes",
        *ek,
    ]
    return subprocess.run(
        komut,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(KOK),
        timeout=600,
        check=False,
    )


@pytest.mark.slow
def test_yes_bayragi_kritik_sizinti_kapisini_acmiyor(tmp_path):
    """``--yes`` "onaylari gec" demektir, "sizintiyi gormezden gel" demek DEGIL.

    Hedefle 1.0000 korelasyonlu bir kolon enjekte edilip betik ``--yes`` ile
    kosuldu. OLCULDU:
      ONCE : "-> Kritik bulgu var. Yine de devam?  [otomatik: evet]"
             GECERLI / Yazildi: gun1_baseline.csv / EXIT = 0
      SONRA: "DURDURULDU. Once sizintiyi coz." / EXIT = 1, submission YOK
    """
    dizin = _kucuk_veri(tmp_path / "sizintili", sizintili=True)

    kosu = _betigi_kos(dizin, [])

    assert kosu.returncode == 1, kosu.stdout[-2000:]
    assert "[CRITICAL]" in kosu.stdout
    assert "DURDURULDU" in kosu.stdout
    assert "--sizintiyi-kabul-ediyorum" in kosu.stdout
    assert "Yazildi:" not in kosu.stdout, "kritik bulguda submission uretilmemeli"


@pytest.mark.slow
def test_ayri_bayrak_verildiginde_kritik_bulguya_ragmen_devam_ediyor(tmp_path):
    """Kapi kilitli degil, AYRI bir anahtari var -- karar komut satirinda yazili kalir."""
    dizin = _kucuk_veri(tmp_path / "sizintili2", sizintili=True)

    kosu = _betigi_kos(dizin, ["--sizintiyi-kabul-ediyorum"])

    assert kosu.returncode == 0, kosu.stdout[-2000:]
    assert "ragmen devam" in kosu.stdout
    assert "Yazildi:" in kosu.stdout


@pytest.mark.slow
def test_temiz_veride_yes_bayragi_hala_uctan_uca_calisiyor(tmp_path):
    """YANLIS-POZITIF KORUMASI: kritik bulgu yokken ``--yes`` eskisi gibi gecmeli.

    Kapi yalnizca KRITIK bulguda kapanir; panel doldurma gibi rutin onaylar
    ``--yes`` ile otomatik gecilmeye devam eder (OLCULDU: EXIT=0, submission
    uretildi).
    """
    dizin = _kucuk_veri(tmp_path / "temiz", sizintili=False)

    kosu = _betigi_kos(dizin, [])

    assert kosu.returncode == 0, kosu.stdout[-3000:]
    assert "DURDURULDU" not in kosu.stdout
    assert "Yazildi:" in kosu.stdout


# ---------------------------------------------------------------------------
# BULGU 1 -- zaman kolonu tespiti (ZATEN KAPALI, geri gitmesin diye korunuyor)
# ---------------------------------------------------------------------------


def test_metin_tarih_kolonu_profil_tarafindan_goruluyor():
    """ZATEN KAPALI (profiling.py duzeltildi) -- bu test geri gitmeyi engeller.

    OLCULDU (TR bicimli, saat damgali metin tarih kolonu):
      profile().time_columns  ONCE []  ->  SONRA ['tarih']

    day_one.py bu listeye bakip ``time_column``i seciyor. Liste bos kalirsa
    ekranda "TimeSeriesSplit oneriyorum" yazarken CV rastgele KFold ile
    boluyor -- sessiz bir ZAMAN SIZINTISI.
    """
    gunler = pd.date_range("2025-01-01", periods=120, freq="D")
    frame = pd.DataFrame(
        {
            "tarih": [f"{g.day:02d}.{g.month:02d}.{g.year} 08:30" for g in gunler],
            "ilce": ["ILCE_01", "ILCE_02"] * 60,
            "hedef": np.linspace(1.0, 50.0, 120),
        }
    )

    rapor = profile(frame, target="hedef")

    assert frame["tarih"].dtype == object or str(frame["tarih"].dtype) == "str"
    assert "tarih" in rapor.time_columns
    assert "--- ZAMAN KOLONLARI ---" in rapor.report()


def test_zaman_kolonu_bulununca_foldlar_ileri_zincirleme_oluyor():
    """Bolme GERCEKTEN zamana gore mi? Fold train boylari ARTMALI.

    OLCULDU: purged_time_series_split ile [1213, 1319, 1425, 1529, 1639]
    (artan = ileri zincirleme). Rastgele KFold(shuffle=True) olsaydi bes
    fold da esit olurdu (bulgunun kaniti: 5 x train=4368 / valid=1092).
    """
    gunler = pd.date_range("2025-01-01", periods=200, freq="D")
    seri = parse_time_series(pd.Series([f"{g.day:02d}.{g.month:02d}.{g.year}" for g in gunler]))

    folds = purged_time_series_split(
        seri, n_splits=5, embargo=pd.Timedelta(days=5), test_span=pd.Timedelta(days=15)
    )
    boylar = [len(tr) for tr, _ in folds]

    assert len(folds) == 5
    assert boylar == sorted(boylar)
    assert len(set(boylar)) > 1, "esit boylar rastgele KFold demektir"
