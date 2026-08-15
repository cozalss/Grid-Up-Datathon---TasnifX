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
import sys
import time
from pathlib import Path

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
from gridup.models import starter_params  # noqa: E402
from gridup.validation import build_splitter, purged_time_series_split  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def banner(step: str, title: str) -> None:
    print(f"\n{'=' * 78}\n{step}  {title}\n{'=' * 78}")


def confirm(question: str, *, auto: bool) -> bool:
    if auto:
        print(f"  -> {question}  [otomatik: evet]")
        return True
    answer = input(f"  -> {question} [E/h] ").strip().lower()
    return answer in ("", "e", "evet", "y", "yes")


def find_files(data_dir: Path) -> dict[str, Path]:
    """train/test/sample dosyalarini ad kalibina gore bulur."""
    found: dict[str, Path] = {}
    patterns = {
        "train": ("train", "egitim"),
        "test": ("test",),
        "sample": ("sample", "submission", "ornek"),
    }
    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() not in {".csv", ".parquet", ".xlsx", ".txt"}:
            continue
        name = path.name.lower()
        for key, needles in patterns.items():
            if key not in found and any(needle in name for needle in needles):
                found[key] = path
    return found


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
    parser.add_argument("--yes", action="store_true", help="Tum onaylari otomatik gec")
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

    train = read_any(files["train"])
    test = read_any(files["test"]) if "test" in files else None
    sample = read_any(files["sample"]) if "sample" in files else None

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

    if findings["critical"] and not confirm("Kritik bulgu var. Yine de devam?", auto=args.yes):
        print("  Durduruldu. Once sizintiyi coz.")
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
        test_times = pd.to_datetime(test[time_column], errors="coerce")
        horizon = int((test_times.max() - test_times.min()).days) + 1
        print(f"\n  Tahmin ufku (test blok uzunlugu): {horizon} gun")
        print("  -> lag/rolling feature'lari bu ufka gore kaydirilmali")

    if time_column:
        train[time_column] = pd.to_datetime(train[time_column], errors="coerce")
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
        folds = purged_time_series_split(
            train[time_column], n_splits=5, embargo=embargo, test_span=test_span
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
        test[time_column] = pd.to_datetime(test[time_column], errors="coerce")
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

    drop = {args.target, args.id_column, time_column} - {None}
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

    if target_kind == "regression" and zero_share > 0.4:
        baseline = zero_baseline_score(
            train_features[args.target].to_numpy(), metric=args.metric
        )
        print(f"  'Hep sifir' baseline {args.metric}: {baseline:.6f}")
        print(f"  (sifir orani %{zero_share * 100:.1f} -- modelin bunu GECMESI gerek)")

    params = starter_params("lightgbm", target_kind)
    result = cross_validate(
        train_features[columns], y, folds,
        kind="lightgbm", task_type=target_kind, metric=args.metric,
        params=params, test=test_features[columns] if test_features is not None else None,
    )
    print("\n" + result.summary())

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
