"""VERI GUNU: ham dosyadan ilk submission'a tek komut.

21 Agustos'ta veri geldiginde calistirilacak ilk sey budur. Amac ilk saatte
leaderboard'da bir skor gormek -- iyi bir skor degil, CALISAN bir hat.

    python scripts/day_one.py --data data/raw --target HEDEF --id ID

Betik yedi asamada ilerler ve her asamada DURUP ne bulduğunu soyler. Bir
asama karar gerektiriyorsa (CV semasi, hedef tipi) onerisini yazar ve
``--yes`` verilmediyse onay bekler.

TASARIM: Her asama kendi basina anlamlidir. Sonuna kadar calismasa bile
her calisan asama bir sey ogretir -- yarisma gununde en degerli sey budur.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gridup import (  # noqa: E402
    build_panel,
    cross_validate,
    environment_report,
    leakage_report,
    panel_coverage,
    postprocess_predictions,
    profile,
    read_any,
    set_global_seed,
    sniff_dialect_shared,
    suggest_scheme,
    write_submission,
    zero_baseline_score,
)
from gridup.compat import categorical_columns  # noqa: E402
from gridup.features import (  # noqa: E402
    add_calendar_features,
    add_frequency_encoding,
    add_turkish_holiday_features,
    shared_origin,
)
from gridup.metrics import get_metric  # noqa: E402
from gridup.models import starter_params  # noqa: E402
from gridup.panel import PANEL_FLAG_COLUMN  # noqa: E402
from gridup.turkish import join_key, normalize_columns  # noqa: E402
from gridup.validation import (  # noqa: E402
    build_splitter,
    parse_time_series,
    purged_time_series_split,
)

ROOT = Path(__file__).resolve().parents[1]


def banner(step: str, title: str) -> None:
    print(f"\n{'=' * 78}\n{step}  {title}\n{'=' * 78}")


def confirm(question: str, *, auto: bool) -> bool:
    if auto:
        print(f"  -> {question}  [otomatik: evet]")
        return True
    answer = input(f"  -> {question} [E/h] ").strip().lower()
    return answer in ("", "e", "evet", "y", "yes")


def _kolonu_coz(ad: str | None, frame: pd.DataFrame) -> str | None:
    """Kullanicinin yazdigi ham kolon adini normalize edilmis ada cevirir.

    Kullanici CSV'de ne goruyorsa onu yazar ("TARIH"); ``read_any`` ise
    kolonlari normalize eder ("tarih"). Ikisini burada baglıyoruz.
    """
    if ad is None or ad in frame.columns:
        return ad
    aday = normalize_columns([ad]).get(ad)
    if aday and aday in frame.columns:
        return aday
    # Ters yon: kullanici normalize edilmis adi yazdi ama frame ham duruyor.
    for ham, normal in (frame.attrs.get("original_columns") or {}).items():
        if ad in (ham, normal):
            return normal if normal in frame.columns else ham
    return ad


#: Dosya adi kaliplari -- kelime listesi BILEREK degistirilmedi; duzeltme
#: eslesme YONTEMINDE (alt dizgi -> kelime), kelime kumesinde degil.
#: Anahtar sirasi ONEM SIRASIDIR: ayni dosya birden fazla role uyarsa
#: listede once gelen rolu kazanir.
_AD_KALIPLARI = {
    "train": ("train", "egitim"),
    "test": ("test",),
    "sample": ("sample", "submission", "ornek"),
}
_VERI_UZANTILARI = {".csv", ".parquet", ".xlsx", ".txt"}


def _ad_belirtecleri(katlanmis: str) -> set[str]:
    """Katlanmis dosya adini KELIMELERE ayirir ("gdz_train_2023" -> {gdz, train, 2023})."""
    return {parca for parca in re.split(r"[^a-z0-9]+", katlanmis) if parca}


def _adaylari_bul(
    dosyalar: list[tuple[Path, set[str], str]], needles: tuple[str, ...]
) -> tuple[list[Path], bool]:
    """``(aday_listesi, kelime_eslesmesi_mi)``.

    ONCE kelime eslesmesi denenir; hicbir dosya kelime duzeyinde uymuyorsa
    alt dizgiye DUSULUR. Bu sira kritik: ``latest_train.csv`` alt dizgi
    duzeyinde "test" icerir ama gercek ``test.csv`` varken ona hic bakilmaz.
    """
    kesin = [path for path, belirtecler, _ in dosyalar if belirtecler.intersection(needles)]
    if kesin:
        return kesin, True
    zayif = [path for path, _, ad in dosyalar if any(needle in ad for needle in needles)]
    return zayif, False


def find_files(data_dir: Path) -> dict[str, Path]:
    """train/test/sample dosyalarini ad kalibina gore bulur.

    ESLESME ONCE KELIME BAZLIDIR. Eskiden yalnizca ``"test" in name`` alt
    dizgi denetimi vardi ve ``"latest"`` kelimesi ``"test"`` iceriyor
    (olculdu: ``'test' in 'latest'`` -> True). Sonuc olculdu:
    ``['latest_train.csv', 'test.csv', 'sample_submission.csv']`` dizininde
    ``{'train': 'latest_train.csv', 'test': 'latest_train.csv'}`` -- train ve
    test AYNI dosya, gercek ``test.csv`` sessizce elendi. sample_submission
    yoksa betik 500 satirlik (olmasi gereken 120) submission uretip EXIT=0
    ile bitiyordu.

    Alt dizgi dali ``sampleSubmission.csv`` gibi kelime ayraci olmayan klasik
    Kaggle adlari icin YEDEK olarak korunur, ama kullanildiginda sessiz
    kalmaz. Ayni dosya iki role atanamaz; elenen adaylar da ekrana yazilir,
    cunku veri gununde yanlis dosyayi okumak saatlere mal olur.
    """
    dosyalar: list[tuple[Path, set[str], str]] = []
    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() not in _VERI_UZANTILARI:
            continue
        katlanmis = join_key(path.stem)
        dosyalar.append((path, _ad_belirtecleri(katlanmis), katlanmis))

    found: dict[str, Path] = {}
    kullanilan: set[Path] = set()
    for key, needles in _AD_KALIPLARI.items():
        liste, kelime_eslesmesi = _adaylari_bul(dosyalar, needles)
        kalan = [path for path in liste if path not in kullanilan]
        if not kalan:
            continue
        found[key] = kalan[0]
        kullanilan.add(kalan[0])
        if not kelime_eslesmesi:
            print(f"  UYARI: '{key}' icin kelime eslesmesi yok; "
                  f"'{kalan[0].name}' ALT DIZGI ile secildi -- dogru dosya mi kontrol et.")
        if len(kalan) > 1:
            digerleri = ", ".join(path.name for path in kalan[1:])
            print(f"  UYARI: '{key}' icin {len(kalan)} aday var. "
                  f"'{kalan[0].name}' secildi, yok sayilanlar: {digerleri}")
    return found


def baseline_karsilastir(
    y: np.ndarray,
    result: Any,
    *,
    metric: str,
    zero_share: float,
    log_uzayinda: bool,
) -> float:
    """'Hep sifir' baseline'ini CV skoruyla AYNI UZAYDA ve AYNI SATIRLARDA basar.

    Eskiden baseline HAM hedeften, CV skoru ise ``log1p`` uzayindan geliyordu
    ve ikisi ard arda ayni birimmis gibi yaziliyordu. Olculdu (bu betigin
    uctan uca kosusu, --metric mae):

        'Hep sifir' baseline mae : 24.131778   (HAM uzay, TUM veri)
        CV skoru (OOF)           :  0.803790   (LOG uzay, kapsanan satirlar)
        ekranda okunan "iyilesme": 30.02 kat   <-- tamamen sahte

    Iki duzeltme birlikte gerekiyor:
    1) UZAY. ``log1p(0) == 0`` oldugu icin "hep sifir" tahmini log uzayinda da
       gecerlidir; baseline'i DONUSTURULMUS hedefe karsi olcmek yeter.
    2) KAPSAM. purged TimeSeriesSplit'te ilk donem hicbir fold'un valid
       tarafinda degildir (olculdu: kapsam %27.5). ``overall_score`` yalnizca
       kapsanan satirlarda hesaplandigi icin baseline de orada olculmeli.

    Dondurulen deger baseline skorudur (test edilebilir olsun diye).
    """
    kapsanan, _ = result.covered_predictions()
    baseline = zero_baseline_score(np.asarray(y)[kapsanan], metric=metric)
    _, buyuk_daha_iyi, _ = get_metric(metric)
    uzay = "log1p uzayinda" if log_uzayinda else "ham uzayda"
    skor = float(result.overall_score)

    print(f"\n  KARSILASTIRMA -- ikisi de {uzay}, ayni {len(kapsanan):,} OOF satirinda:")
    print(f"    'Hep sifir' baseline {metric}: {baseline:.6f}")
    print(f"    Model CV skoru       {metric}: {skor:.6f}")
    print(f"    (sifir orani %{zero_share * 100:.1f} -- modelin baseline'i GECMESI gerek)")

    gecti = skor > baseline if buyuk_daha_iyi else skor < baseline
    if baseline == 0:
        # Sifira bolme yok: oransal fark tanimsiz, mutlak farki soyleriz.
        print(f"    Baseline tam 0 -- oransal fark tanimsiz. Fark: {skor - baseline:+.6f}")
    else:
        oran = (baseline - skor) / abs(baseline) * (-1 if buyuk_daha_iyi else 1)
        print(f"    Fark: %{oran * 100:+.1f}  ->  {'GECTI' if gecti else 'GECEMEDI'}")
    if not gecti:
        print("    UYARI: Model 'hep sifir' baseline'ini GECMIYOR. Sorun modelde")
        print("    degil yaklasimdadir -- iki asamali model (two_stage) dene.")
    if log_uzayinda:
        print("    NOT: Bu sayilar log1p uzayindadir; leaderboard skoruyla")
        print("    dogrudan karsilastirilamaz (LB ham uzayda olculur).")
    return baseline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw", help="Ham veri dizini")
    parser.add_argument("--target", help="Hedef kolon adi (bilinmiyorsa profil sonrasi sorulur)")
    parser.add_argument("--id", dest="id_column", help="ID kolonu")
    parser.add_argument("--time", dest="time_column", help="Zaman kolonu")
    parser.add_argument("--group", dest="group_column", help="Grup/varlik kolonu")
    parser.add_argument("--metric", default="rmse", help="Resmi metrik")
    parser.add_argument("--task", default="regression",
                        choices=("regression", "binary", "multiclass"))
    parser.add_argument("--yes", action="store_true",
                        help="Rutin onaylari otomatik gec (KRITIK sizinti kapisi HARIC)")
    parser.add_argument("--sizintiyi-kabul-ediyorum", dest="accept_leakage",
                        action="store_true",
                        help="Kritik sizinti bulgusuna RAGMEN devam et. Ayri bayrak, "
                             "cunku --yes 'onaylari gec' demektir, 'sizintiyi gormezden "
                             "gel' demek degil.")
    args = parser.parse_args()

    started = time.perf_counter()
    set_global_seed(42)

    # ---------------------------------------------------------------- 0
    banner("0/7", "ORTAM")
    for key, value in environment_report().items():
        print(f"  {key:<26} {value}")

    # ---------------------------------------------------------------- 1
    banner("1/7", "DOSYALARI BUL VE OKU")
    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"  HATA: {data_dir} yok. --data ile dogru dizini ver.")
        return 1

    files = find_files(data_dir)
    if "train" not in files:
        print(f"  HATA: train dosyasi bulunamadi. {data_dir} icindekiler:")
        for path in sorted(data_dir.rglob("*"))[:20]:
            print(f"    {path.name}")
        return 1

    for key, path in files.items():
        print(f"  {key:<8} {path.name}")

    # ORTAK BICIM KARARI. sniff_dialect her dosyaya BAGIMSIZ bakar ve ayni
    # kolon train'de float64, test'te str olabilir (olculdu). Metin dosyalarinin
    # basliklarini havuzlayip TEK karar veriyoruz.
    metin_yollari = [
        yol for yol in files.values()
        if yol.suffix.lower() in {".csv", ".tsv", ".txt", ".dat", ""}
    ]
    ortak = sniff_dialect_shared(metin_yollari) if len(metin_yollari) > 1 else None
    if ortak is not None:
        print(f"  ortak bicim: {ortak}")

    def _oku(yol: Path) -> pd.DataFrame:
        if ortak is not None and yol in metin_yollari:
            return read_any(yol, dialect=ortak)
        return read_any(yol)

    train = _oku(files["train"])
    test = _oku(files["test"]) if "test" in files else None
    sample = _oku(files["sample"]) if "sample" in files else None

    # Okuyucu kolon adlarini normalize eder ("TARIH" -> "tarih",
    # "Dagitilan Enerji (MWh)" -> "dagitilan_enerji_mwh") ama kullanici
    # dosyada GORDUGU adi yazar. Eskiden bu ikisi eslesmiyordu ve belgelenen
    # veri gunu komutu ("--time TARIH --group ILCE") ilk adimda hata veriyordu:
    #     HATA: hedef kolon belirlenemedi. Kolonlar: ['id','ilce','tarih',...]
    # Yarismanin ilk saatinde kaybedilecek en pahali dakikalar bunlar.
    for alan in ("target", "id_column", "time_column", "group_column"):
        ham = getattr(args, alan)
        cozulen = _kolonu_coz(ham, train)
        if cozulen != ham:
            print(f"  --{alan}: '{ham}' -> '{cozulen}' (normalize edilmis ad)")
        setattr(args, alan, cozulen)

    print(f"\n  train {train.shape}" + (f"   test {test.shape}" if test is not None else ""))
    if sample is not None:
        print(f"  sample_submission kolonlari: {list(sample.columns)}")
        if args.id_column is None:
            args.id_column = str(sample.columns[0])
        if args.target is None:
            args.target = str(sample.columns[-1])
            print(f"  -> sample'dan cikarildi: id={args.id_column}  hedef={args.target}")

    if args.target is None or args.target not in train.columns:
        print(f"\n  HATA: hedef kolon belirlenemedi. Kolonlar: {list(train.columns)}")
        return 1

    # ---------------------------------------------------------------- 2
    banner("2/7", "PROFIL")
    report = profile(train, test, target=args.target)
    print(report.report())

    target_kind = report.target_summary.get("gorev_tahmini", args.task)
    zero_share = float(report.target_summary.get("sifir_orani", 0.0) or 0.0)
    skew = float(report.target_summary.get("carpiklik", 0.0) or 0.0)

    # ---------------------------------------------------------------- 3
    banner("3/7", "SIZINTI TARAMASI")
    time_column = args.time_column or (report.time_columns[0] if report.time_columns else None)
    findings = leakage_report(train, args.target, test=test, time_column=time_column)
    print(f"  {findings['summary']}\n")
    for severity in ("critical", "warning", "info"):
        for message in findings[severity][:6]:
            print(f"  [{severity.upper()}] {message}")

    # KRITIK SIZINTI KAPISI --yes ILE ACILMAZ.
    #
    # --yes "rutin onaylari gec" demektir (panel doldurma gibi, geri
    # alinabilir kararlar). Kritik sizinti geri alinabilir bir karar DEGIL:
    # olculdu, hedefle 1.0000 korelasyonlu bir kolon enjekte edildiginde
    # --yes'li kosu "[otomatik: evet]" yazip EXIT=0 ile submission uretti.
    # Kaggle'a gonderilen dosya LB'de coker, sebebi gorunmez.
    #
    # Bilerek devam etmek icin AYRI ve uzun bir bayrak gerekir; boylece
    # "sizintiyi kabul ettim" karari komut satirinda YAZILI kalir.
    if findings["critical"]:
        if args.accept_leakage:
            print("\n  --sizintiyi-kabul-ediyorum verildi -- KRITIK bulguya ragmen devam.")
        elif args.yes or not confirm("Kritik bulgu var. Yine de devam?", auto=False):
            print("\n  DURDURULDU. Once sizintiyi coz.")
            print("  Bilerek devam etmek istiyorsan: --sizintiyi-kabul-ediyorum")
            print("  (--yes bu kapiyi ACMAZ -- kritik sizinti rutin bir onay degildir.)")
            return 1

    # ---------------------------------------------------------------- 4
    banner("4/7", "PANEL KONTROLU")
    if time_column and args.group_column:
        coverage = panel_coverage(
            train, entity_columns=[args.group_column], time_column=time_column
        )
        ratio = coverage.get("coverage", float("nan"))
        print(f"  Beklenen satir: {coverage.get('expected_rows', 0):,.0f}")
        print(f"  Gercek satir:   {coverage.get('actual_rows', 0):,.0f}")
        print(f"  Doluluk:        %{ratio * 100:.1f}")
        if ratio < 0.95 and confirm("Panel seyrek. Sifirla doldurulsun mu?", auto=args.yes):
            train = build_panel(
                train, entity_columns=[args.group_column], time_column=time_column
            )
            print(f"  Panel kuruldu: {train.shape}")
    else:
        print("  Zaman veya grup kolonu verilmedi -- panel kontrolu atlandi.")
        print("  (--time ve --group ile belirtirsen 'olay olmadi' gunleri doldurulur)")

    # ---------------------------------------------------------------- 5
    banner("5/7", "CV SEMASI")
    suggestion = suggest_scheme(
        train, target=args.target, task_type=target_kind,
        known_time=time_column, known_group=args.group_column,
    )
    print(suggestion)

    horizon = 1
    if time_column and test is not None and time_column in test.columns:
        test_times = parse_time_series(test[time_column])
        horizon = int((test_times.max() - test_times.min()).days) + 1
        print(f"\n  Tahmin ufku (test blok uzunlugu): {horizon} gun")
        print("  -> lag/rolling feature'lari bu ufka gore kaydirilmali")

    if time_column:
        train[time_column] = parse_time_series(train[time_column])
        embargo = pd.Timedelta(days=max(horizon, 30))
        # DOGRULAMA PENCERESI = TEST BLOGU UZUNLUGU.
        #
        # 2023 GDZ birincisi TimeSeriesSplit(n_splits=3, test_size=744)
        # kullandi -- 744 saat, yani test blogunun TAM boyu. CV, tahmin
        # edilecek ufku birebir taklit etmelidir.
        #
        # Satir sayisina gore esit bolme PANEL veride yanlis pencere uretir:
        # 20 ilcelik bir "ay" 620 satirdir ve fold uzunlugu veri yogunluguna
        # gore kayar. test_span zamana gore boler, satira gore degil.
        test_span = pd.Timedelta(days=horizon) if horizon else None
        # n_splits=4: ablasyon, benchmark ve notebook'lar ayni veride 4 ile
        # olculdu -- fold sayisi farkli olursa skorlar KARSILASTIRILAMAZ
        # (denetim bulgusu). Sema degisecekse her yerde birden degisir.
        folds = purged_time_series_split(
            train[time_column], n_splits=4, embargo=embargo, test_span=test_span
        )
    elif args.group_column:
        splitter = build_splitter("GroupKFold", n_splits=5)
        folds = list(splitter.split(train, groups=train[args.group_column]))
    else:
        scheme = "StratifiedKFold" if target_kind != "regression" else "KFold"
        splitter = build_splitter(scheme, n_splits=5, seed=42)
        stratify = train[args.target] if target_kind != "regression" else None
        folds = list(splitter.split(train, stratify))

    for index, (tr, va) in enumerate(folds, start=1):
        print(f"    fold {index}: train={len(tr):>8,}  valid={len(va):>8,}")

    # ---------------------------------------------------------------- 6
    banner("6/7", "FEATURE + BASELINE")
    origin = None
    if time_column and test is not None:
        test[time_column] = parse_time_series(test[time_column])
        origin = shared_origin(train, test, time_column=time_column)
        print(f"  Ortak zaman baslangici: {origin.date()}")

    def build(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        if time_column:
            out = add_calendar_features(out, time_column, include_year=False, origin=origin)
            out = add_turkish_holiday_features(out, time_column)
        categorical = [c for c in categorical_columns(out) if c != args.target]
        if categorical:
            out = add_frequency_encoding(out, categorical[:15])
        return out

    train_features = build(train)
    test_features = build(test) if test is not None else None

    # PANEL_FLAG_COLUMN feature OLAMAZ: fill_value=0 iken hedefin sifir
    # olmasiyla birebir ayni seydir (olculdu: %100 ortusme, Spearman -0.9810).
    # Test kumesinde yoksa zaten elenirdi, ama test verilmediginde listeye
    # giriyordu -- adiyla dislamak tek guvenilir yol.
    drop = {args.target, args.id_column, time_column, PANEL_FLAG_COLUMN} - {None}
    columns = [
        c for c in train_features.columns
        if c not in drop and (test_features is None or c in test_features.columns)
    ]
    print(f"  {len(columns)} feature")

    y = train_features[args.target].to_numpy()
    use_log = target_kind == "regression" and skew > 2
    if use_log:
        from gridup.metrics import log_transform_target

        print(f"  Hedef carpikligi {skew:.2f} -> log1p donusumu uygulaniyor")
        y = log_transform_target(y)

    params = starter_params("lightgbm", target_kind)
    result = cross_validate(
        train_features[columns], y, folds,
        kind="lightgbm", task_type=target_kind, metric=args.metric,
        params=params, test=test_features[columns] if test_features is not None else None,
    )
    print("\n" + result.summary())

    if target_kind == "regression" and zero_share > 0.4:
        baseline_karsilastir(
            y, result, metric=args.metric, zero_share=zero_share, log_uzayinda=use_log
        )

    # ---------------------------------------------------------------- 7
    banner("7/7", "SUBMISSION")
    if test_features is None or result.test_predictions is None:
        print("  Test kumesi yok -- submission uretilmedi.")
        return 0

    predictions = result.test_predictions
    if use_log:
        from gridup.metrics import inverse_log_transform

        predictions = inverse_log_transform(predictions)

    predictions = postprocess_predictions(
        predictions,
        round_to_integer=(args.metric == "mae" and float(np.mod(
            train_features[args.target].dropna(), 1
        ).max()) == 0.0),
        clip_min=0.0,
    )

    path = write_submission(
        test_features[args.id_column].to_numpy(), predictions,
        ROOT / "submissions" / "gun1_baseline.csv",
        sample=sample, id_column=args.id_column, target_column=args.target,
    )

    elapsed = time.perf_counter() - started
    banner("TAMAM", f"{elapsed / 60:.1f} dakika")
    print(f"  Submission: {path}")
    print("\n  SIRADAKI ADIMLAR:")
    print("   1. Bu dosyayi Kaggle'a gonder -- format dogru mu gor")
    print("   2. LB skorunu deney defterine yaz: log.record_lb(...)")
    print("   3. adversarial_validation ile train/test kaymasini olc")
    print(f"   4. Ufuk-farkindalikli lag/rolling ekle (horizon={horizon})")
    print("   5. Hava verisini birlestir (data/external/hava_gunluk.parquet)")
    print("   6. Komsu ilce sinyali (data/reference/ilceler_gdz_adm.parquet)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
