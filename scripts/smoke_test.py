"""Uctan uca duman testi: pipeline'in TAMAMINI sentetik veri uzerinde calistirir.

AMAC: Veri gelmeden once "her sey calisiyor" iddiasini KANITLAMAK. Bu betik
basariyla bitiyorsa, 21 Agustos'ta yapman gereken tek sey config.py'yi
guncellemek ve gercek dosyayi okumak.

Calistirma::

    python scripts/smoke_test.py

Adimlar: veri uret -> profille -> CV semasi sec -> sizinti tara -> feature uret
-> egit -> esik/harman -> submission yaz -> dogrula.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gridup import (  # noqa: E402
    cross_validate,
    environment_report,
    leakage_report,
    profile,
    set_global_seed,
    suggest_scheme,
    write_submission,
)
from gridup.ensemble import correlation_matrix, hill_climb_weights  # noqa: E402
from gridup.experiment import (  # noqa: E402
    DataArtifact,
    ExperimentProvenance,
    ExperimentRecord,
)
from gridup.features import (  # noqa: E402
    add_calendar_features,
    add_turkish_holiday_features,
    oof_target_encode,
)
from gridup.features.temporal import shared_origin  # noqa: E402
from gridup.pipeline import (  # noqa: E402
    FoldPlan,
    build_paired_history_features,
    runtime_recipe_fingerprint,
)
from gridup.recipe import CVRecipe, FeatureRecipe, ModelRecipe, PipelineRecipe  # noqa: E402
from gridup.stores import SQLiteExperimentStore  # noqa: E402
from gridup.synthetic import SyntheticSpec, make_distribution_dataset  # noqa: E402
from gridup.turkish import diagnose_join, join_key  # noqa: E402
from gridup.validation import adversarial_validation, purged_time_series_split  # noqa: E402

TARGET = "kesinti_suresi_dk"
TIME_COLUMN = "tarih"
GROUP_COLUMN = "trafo_id"
ID_COLUMN = "id"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "submissions"


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def main() -> int:
    started = time.perf_counter()
    set_global_seed(42)

    banner("0. ORTAM")
    for key, value in environment_report().items():
        print(f"  {key:<26} {value}")

    # ------------------------------------------------------------------
    banner("1. SENTETIK VERI URET (gercek verinin muhtemel seklini taklit eder)")
    spec = SyntheticSpec(n_transformers=120, start_date="2024-01-01", end_date="2025-12-31")
    train_raw, test_raw, solution = make_distribution_dataset(spec)
    print(f"  train {train_raw.shape}   test {test_raw.shape}")
    print(f"  Ham kolon adlari (Turkce): {list(train_raw.columns[:6])}")

    # Gercek akista read_any() bunu yapar; burada dogrudan normalize ediyoruz.
    from gridup.turkish import normalize_columns

    mapping = normalize_columns(train_raw.columns)
    train = train_raw.rename(columns=mapping)
    test = test_raw.rename(columns=normalize_columns(test_raw.columns))
    solution = solution.rename(columns=normalize_columns(solution.columns))
    print(f"  Normalize edilmis: {list(train.columns[:6])}")

    # ------------------------------------------------------------------
    banner("2. TURKCE JOIN TUZAGI TESTI (bu sessiz hatanin kaniti)")
    provinces_upper = ["İZMİR", "MUĞLA", "AYDIN"]
    provinces_mixed = ["İzmir", "Mugla", "Aydın"]

    naive_matches = len({p.lower() for p in provinces_upper} & set(provinces_mixed))
    smart_matches = len(
        {join_key(p) for p in provinces_upper} & {join_key(p) for p in provinces_mixed}
    )
    print(f"  Naif .lower() eslesme:  {naive_matches}/3   <- SESSIZ HATA")
    print(f"  join_key() eslesme:     {smart_matches}/3   <- DOGRU")
    if smart_matches != 3:
        print("  BASARISIZ: join_key beklendigi gibi calismadi.")
        return 1

    diagnosis = diagnose_join(provinces_upper, provinces_mixed)
    print(f"  Teshis: normalizasyon {diagnosis['recovered']} eslesme kurtardi.")

    # ------------------------------------------------------------------
    banner("3. VERI PROFILI")
    dataset_profile = profile(train, test, target=TARGET)
    print(dataset_profile.report())

    # ------------------------------------------------------------------
    banner("4. CV SEMASI SECIMI")
    suggestion = suggest_scheme(
        train,
        target=TARGET,
        task_type="regression",
        known_group=GROUP_COLUMN,
        known_time=TIME_COLUMN,
    )
    print(suggestion)

    # ------------------------------------------------------------------
    banner("5. SIZINTI TARAMASI")
    findings = leakage_report(train, TARGET, test=test, time_column=TIME_COLUMN)
    print(f"  {findings['summary']}")
    for severity in ("critical", "warning", "info"):
        for message in findings[severity][:4]:
            print(f"  [{severity.upper()}] {message}")

    # ------------------------------------------------------------------
    banner("6. FEATURE URETIMI")
    drop_columns = [TARGET, "ariza_var_mi", "ariza_tipi"]

    train[TIME_COLUMN] = pd.to_datetime(train[TIME_COLUMN])
    test[TIME_COLUMN] = pd.to_datetime(test[TIME_COLUMN])

    # ORTAK origin: train ve test icin ayri ayri hesaplanirsa test'in gun
    # sayaci yeniden 0'dan baslar ve model test'i train'in gecmisi sanir.
    origin = shared_origin(train, test, time_column=TIME_COLUMN)
    print(f"  Ortak zaman baslangici: {origin.date()}")

    # Test bloğu 3 ay ileride -> tahmin aninda son 90 gunun verisi YOK.
    # Lag/rolling'ler bu ufka gore kaydirilmali, aksi halde model uretimde
    # sahip olmayacagi bir sinyale bagimli olur.
    horizon = int((test[TIME_COLUMN].max() - test[TIME_COLUMN].min()).days) + 1
    print(f"  Tahmin ufku (test blok uzunlugu): {horizon} gun")

    def build_base(frame: pd.DataFrame) -> pd.DataFrame:
        out = add_calendar_features(frame, TIME_COLUMN, include_year=False, origin=origin)
        return add_turkish_holiday_features(out, TIME_COLUMN)

    # Global grup/frekans mapping'i erken temporal fold'lara gelecekteki
    # validation dagilimini tasirdi; fold-ici encoder olmadan bu aile kapali.
    history = build_paired_history_features(
        build_base(train),
        build_base(test),
        value_column="tuketim_kwh",
        time_column=TIME_COLUMN,
        group_columns=[GROUP_COLUMN],
        horizon=horizon,
        shifts=[horizon, horizon + 6, horizon + 27],
        rolling_windows=[7, 28],
        rolling_aggregations=("mean", "std"),
        # ACIKCA veriliyor: bu history hedef degil kovaryat ("tuketim_kwh")
        # uzerinde. Nobetci, ikisinin ayni olmadigini dogrular.
        target_column=TARGET,
    )
    train_features, test_features = history.train, history.test

    print(f"  train feature sayisi: {train_features.shape[1]}")
    print(f"  test  feature sayisi: {test_features.shape[1]}")

    # ------------------------------------------------------------------
    banner("7. CV FOLD'LARI (ambargolu zaman serisi -- kayan pencere sizintisini keser)")
    times = pd.to_datetime(train_features[TIME_COLUMN])
    # Ambargo, en uzun kayan pencereden (28) BUYUK secildi.
    folds = purged_time_series_split(times, n_splits=4, embargo=pd.Timedelta(days=35))
    for index, (train_idx, valid_idx) in enumerate(folds, start=1):
        print(f"  fold {index}: train={len(train_idx):>7,}  valid={len(valid_idx):>7,}")

    # ------------------------------------------------------------------
    banner("8. FOLD-DISI HEDEF KODLAMA (sizintisiz)")
    y = train_features[TARGET].to_numpy(dtype="float64")
    train_encoded, test_encoded = oof_target_encode(
        train_features,
        pd.Series(y),
        ["ilce", "abone_grubu"],
        folds,
        test=test_features,
        smoothing=30.0,
        uncovered_policy="nan",
    )
    print("  Kodlanan kolonlar: ilce, abone_grubu -> +2 feature")

    # ------------------------------------------------------------------
    banner("9. ADVERSARIAL VALIDATION (train ve test ayni dagilimdan mi?)")
    feature_columns = [
        column
        for column in train_encoded.columns
        if column not in drop_columns + [ID_COLUMN, TIME_COLUMN] and column in test_encoded.columns
    ]
    adversarial = adversarial_validation(
        train_encoded[feature_columns].sample(n=min(20000, len(train_encoded)), random_state=42),
        test_encoded[feature_columns].sample(n=min(20000, len(test_encoded)), random_state=42),
        n_splits=3,
    )
    print(f"  AUC = {adversarial['auc']:.4f}")
    print(f"  Karar: {adversarial['verdict']}")
    print("  En ayirt edici feature'lar:")
    for name, importance in adversarial["top_features"][:5]:
        print(f"    {name:<44} {importance:>10.1f}")

    # ------------------------------------------------------------------
    banner("10. MODEL EGITIMI (hedef carpik -> log1p donusumu)")
    x_train = train_encoded[feature_columns]
    x_test = test_encoded[feature_columns]
    results = {}
    params_by_kind = {}
    for kind in ("lightgbm", "catboost"):
        print(f"\n  --- {kind} ---")
        params = None
        if kind == "lightgbm":
            from gridup.models import starter_params

            params = starter_params("lightgbm", "regression")
            params["n_estimators"] = 400  # duman testi: hizli olsun
        else:
            from gridup.models import starter_params

            params = starter_params("catboost", "regression")
            params["iterations"] = 400

        params_by_kind[kind] = dict(params)

        results[kind] = cross_validate(
            x_train,
            y,
            folds,
            kind=kind,
            task_type="regression",
            metric="rmsle",
            params=params,
            test=x_test,
            early_stopping_rounds=50,
            target_transform="log1p",
            early_stopping_metric="rmsle",
        )
        print(results[kind].summary()[:900])

    # ------------------------------------------------------------------
    banner("11. HARMANLAMA")
    covered = results["lightgbm"].oof_covered
    oof_map = {name: result.oof_predictions[covered] for name, result in results.items()}

    print("  Model korelasyonu:")
    print(correlation_matrix(oof_map).round(4).to_string())

    weights = hill_climb_weights(oof_map, y[covered], metric="rmsle", step=0.02)

    blended_test = sum(weight * results[name].test_predictions for name, weight in weights.items())

    # ------------------------------------------------------------------
    banner("12. SUBMISSION YAZ VE DOGRULA")
    predictions = blended_test
    submission_path = write_submission(
        test_encoded[ID_COLUMN].to_numpy(),
        predictions,
        OUTPUT_DIR / "smoke_test_harman.csv",
        id_column="id",
        target_column=TARGET,
    )

    # ------------------------------------------------------------------
    banner("13. SIMULE EDILMIS PRIVATE LEADERBOARD")
    from gridup.metrics import rmsle

    merged = pd.read_csv(submission_path).merge(solution, on="id", suffixes=("_tahmin", "_gercek"))
    true_values = merged[f"{TARGET}_gercek"].to_numpy()
    predicted = merged[f"{TARGET}_tahmin"].to_numpy()

    holdout_rmsle = rmsle(true_values, predicted)
    baseline_rmsle = rmsle(true_values, np.full_like(true_values, np.median(y)))

    print(f"  Harman RMSLE:            {holdout_rmsle:.5f}")
    print(f"  Medyan baseline RMSLE:   {baseline_rmsle:.5f}")
    improvement = (baseline_rmsle - holdout_rmsle) / baseline_rmsle * 100
    print(f"  Baseline'a gore kazanc:  %{improvement:.1f}")

    # ------------------------------------------------------------------
    banner("14. DENEY DEFTERI")
    root = Path(__file__).resolve().parents[1]
    store = SQLiteExperimentStore(root / "experiments" / "experiments.db")
    fold_plan = FoldPlan.from_folds(folds, n_rows=len(train_encoded))
    for name, result in results.items():
        estimator_count_key = "iterations" if name == "catboost" else "n_estimators"
        recipe = PipelineRecipe(
            seed=42,
            cv=CVRecipe(
                n_splits=len(folds),
                splitter="purged_time_series",
                embargo_days=35,
            ),
            features=FeatureRecipe(
                horizon=horizon,
                target_shifts=(),
                rolling_windows=(),
                history_value_columns=("tuketim_kwh",),
                history_shifts=(horizon, horizon + 6, horizon + 27),
                history_rolling_windows=(7, 28),
                history_rolling_aggregations=("mean", "std"),
                families=(
                    "calendar",
                    "holiday",
                    "consumption_history",
                    "oof_target_encoding",
                ),
            ),
            model=ModelRecipe(
                kind=name,
                objective=str(params_by_kind[name].get("objective", "regression")),
                metric="rmsle",
                early_stopping_metric="rmsle",
                n_estimators=int(params_by_kind[name][estimator_count_key]),
                early_stopping_rounds=50,
            ),
        )
        runtime_fingerprint = runtime_recipe_fingerprint(
            recipe.to_dict(),
            backend=name,
            target_transform="log1p",
            n_estimators=400,
            early_stopping_rounds=50,
        )
        provenance = ExperimentProvenance.capture(
            recipe_fingerprint=runtime_fingerprint,
            data_artifacts=[DataArtifact.from_path(submission_path)],
            feature_names=feature_columns,
            fold_fingerprint=fold_plan.fingerprint,
        )
        store.add(
            ExperimentRecord(
                name=f"duman_{name}",
                cv_score=result.overall_score,
                metric="rmsle",
                model_kind=name,
                n_features=len(feature_columns),
                fold_scores=result.fold_scores,
                params=params_by_kind[name],
                features=list(feature_columns),
                notes="duman testi: takvim+tatil+grup+lag+rolling+OOF hedef kodlama",
                submission_path=str(submission_path),
                provenance=provenance,
            )
        )
    print(pd.DataFrame(store.load()).tail(len(results))[["name", "cv_score", "metric"]])

    elapsed = time.perf_counter() - started
    banner(f"TAMAM -- pipeline uctan uca calisiyor.  Toplam sure: {elapsed:.1f} sn")

    if improvement <= 0:
        print("UYARI: model baseline'i gecemedi. Duman testi icin bu bir hatadir.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
