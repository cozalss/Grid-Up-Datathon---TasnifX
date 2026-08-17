"""TAM PIPELINE PROVASI -- her modulun birlikte calistigini kanitlar.

``smoke_test.py`` cekirdek akisi dogrular. Bu betik ILERI moduleri de devreye
sokar ve gercek dis veriyle birlestirir:

    panel · weather · spatial · outage_reason · selection · two_stage
    refit · reporting

NEDEN AYRI BIR BETIK: bu moduller birim testleriyle dogrulanmisti ama
BIRLIKTE calistiklari hic gosterilmemisti. Yarisma gunu ilk kez denemek
istemedigimiz sey tam olarak budur.

Calistirma::

    python scripts/full_pipeline.py
    python scripts/full_pipeline.py --hizli    # kucuk veri, az iterasyon
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
    postprocess_predictions,
    set_global_seed,
    write_submission,
    zero_baseline_score,
)
from gridup.experiment import (  # noqa: E402
    DataArtifact,
    ExperimentProvenance,
    ExperimentRecord,
)
from gridup.features import (  # noqa: E402
    add_calendar_features,
    add_neighbour_target_lag,
    add_physical_derivatives,
    add_regional_aggregates,
    add_turkish_holiday_features,
    nearest_neighbours,
    oof_target_encode,
    shared_origin,
)
from gridup.features.outage_reason import add_reason_features, reason_family_report  # noqa: E402
from gridup.metrics import inverse_log_transform, log_transform_target, rmsle  # noqa: E402
from gridup.models import starter_params  # noqa: E402
from gridup.pipeline import (  # noqa: E402
    FoldPlan,
    build_paired_history_features,
    runtime_recipe_fingerprint,
)
from gridup.recipe import CVRecipe, FeatureRecipe, ModelRecipe, PipelineRecipe  # noqa: E402
from gridup.refit import (  # noqa: E402
    estimate_full_data_rounds,
    extract_best_iterations,
    fold_train_fraction,
    multi_seed_refit,
)
from gridup.reporting import (  # noqa: E402
    business_impact,
    cv_fold_table,
    error_by_segment,
    feature_importance_table,
    model_footprint,
    prediction_vs_actual_table,
)
from gridup.selection import null_importance_filter, shap_backward_selection  # noqa: E402
from gridup.stores import SQLiteExperimentStore  # noqa: E402
from gridup.synthetic import SyntheticSpec, make_distribution_dataset  # noqa: E402
from gridup.turkish import join_key, normalize_columns  # noqa: E402
from gridup.validation import adversarial_validation, purged_time_series_split  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WEATHER = ROOT / "data" / "external" / "hava_gunluk.parquet"
DISTRICTS = ROOT / "data" / "reference" / "ilceler_gdz_adm.parquet"

TARGET = "kesinti_suresi_dk"
TIME = "tarih"
GROUP = "trafo_id"
ID = "id"


def banner(step: str, text: str) -> None:
    print(f"\n{'=' * 78}\n{step}  {text}\n{'=' * 78}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hizli", action="store_true", help="Kucuk veri, az iterasyon")
    args = parser.parse_args()

    started = time.perf_counter()
    set_global_seed(42)
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append((name, passed, detail))
        mark = "OK  " if passed else "HATA"
        print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail else ""))

    # ------------------------------------------------------------------ 1
    banner("1/9", "SENTETIK VERI + PANEL")
    spec = SyntheticSpec(
        n_transformers=60 if args.hizli else 150,
        start_date="2024-01-01",
        end_date="2025-09-30" if args.hizli else "2025-12-31",
    )
    train_raw, test_raw, solution = make_distribution_dataset(spec)
    train = train_raw.rename(columns=normalize_columns(train_raw.columns))
    test = test_raw.rename(columns=normalize_columns(test_raw.columns))
    solution = solution.rename(columns=normalize_columns(solution.columns))

    train[TIME] = pd.to_datetime(train[TIME])
    test[TIME] = pd.to_datetime(test[TIME])
    check("veri uretildi", len(train) > 0, f"train {train.shape} test {test.shape}")

    # Panel: sentetik veri zaten tam ama fonksiyonun calistigini dogrula
    panel = build_panel(
        train[[GROUP, TIME, TARGET]],
        entity_columns=[GROUP],
        time_column=TIME,
        verbose=False,
    )
    check("build_panel", len(panel) >= len(train), f"{len(panel):,} satir")

    # ------------------------------------------------------------------ 2
    banner("2/9", "ARIZA SEBEBI TAKSONOMISI")
    with_reason = add_reason_features(train, "ariza_tipi")
    families = reason_family_report(train["ariza_tipi"].dropna())
    check(
        "add_reason_features",
        "sebep_kod" in with_reason.columns,
        f"{len(families)} aile, en sik: {families.iloc[0]['aile']}",
    )

    # ------------------------------------------------------------------ 3
    banner("3/9", "GERCEK DIS VERI: HAVA + ILCE")
    weather_ok = WEATHER.exists()
    districts_ok = DISTRICTS.exists()

    neighbours = None
    if districts_ok:
        districts = pd.read_parquet(DISTRICTS)
        neighbours = nearest_neighbours(
            districts,
            key_column="ilce_key",
            latitude_column="lat",
            longitude_column="lon",
            k=3,
            max_distance_km=120,
        )
        check(
            "komsuluk grafigi",
            len(neighbours) > 0,
            f"{neighbours['ilce_key'].nunique()} ilce, {len(neighbours)} baglanti",
        )
    else:
        check("ilce tablosu", False, "data/reference/ilceler_gdz_adm.parquet yok")

    if weather_ok:
        weather = pd.read_parquet(WEATHER)
        weather = add_physical_derivatives(
            weather, group_columns=["konum_key"], time_column="tarih"
        )
        weather = add_regional_aggregates(
            weather,
            time_column="tarih",
            value_columns=["ruzgar_max", "sicaklik_ort"],
            quantiles=(0.9,),
        )
        check(
            "hava turevleri",
            "islak_ruzgar" in weather.columns,
            f"{len(weather):,} satir, {weather.shape[1]} kolon",
        )
    else:
        check("hava verisi", False, "data/external/hava_gunluk.parquet yok")

    # ------------------------------------------------------------------ 4
    banner("4/9", "FEATURE URETIMI")
    origin = shared_origin(train, test, time_column=TIME)
    horizon = int((test[TIME].max() - test[TIME].min()).days) + 1
    print(f"  ortak origin={origin.date()}  ufuk={horizon} gun")

    def build_base(frame: pd.DataFrame) -> pd.DataFrame:
        out = add_calendar_features(frame, TIME, include_year=False, origin=origin)
        out = add_turkish_holiday_features(out, TIME)
        # Gercek hava verisini join et -- IL bazinda, join_key ile.
        #
        # DIKKAT: hava verisi artik ILCE cozunurlugunde (tarih basina 96
        # satir). Il bazli bir paneli dogrudan merge etmek satir sayisini
        # 96 KATINA cikarirdi. Once ile indirgiyoruz.
        #
        # OLCULDU: hava konumlari il merkezinden ilce anahtarina gecince
        # ("izmir" -> "izmir-konak") bu join'in eslesme orani %0.0'a dustu.
        # Cozum iki parcali: (a) hava verisine il_key/ilce_key kolonlari
        # eklendi, (b) burada il duzeyine indirgeniyor.
        if weather_ok:
            hava_kolonlari = ["isitma_derece_gun", "sogutma_derece_gun", "bolge_ruzgar_max_q90"]
            mevcut = [k for k in hava_kolonlari if k in weather.columns]
            il_havasi = (
                weather.groupby(["il_key", "tarih"], observed=True)[mevcut].mean().reset_index()
            )
            onceki_satir = len(out)
            out = out.assign(_il_key=out["il"].map(join_key))
            out = out.merge(
                il_havasi,
                left_on=["_il_key", TIME],
                right_on=["il_key", "tarih"],
                how="left",
                suffixes=("", "_hava"),
                validate="many_to_one",
            ).drop(columns=["_il_key", "il_key"], errors="ignore")
            if len(out) != onceki_satir:
                raise AssertionError(
                    f"Hava join satir sayisini degistirdi: {onceki_satir} -> {len(out)}"
                )
        return out

    train_base = build_base(train)
    test_base = build_base(test)
    # Temporal CV'den once tum train uzerinde grup/frekans mapping'i fit
    # etmiyoruz: erken fold'lar gelecekteki validation dagilimini gorurdu.
    # Bu aile ancak fold-ici stateful transformer ile geri eklenebilir.
    history = build_paired_history_features(
        train_base,
        test_base,
        value_column="tuketim_kwh",
        time_column=TIME,
        group_columns=[GROUP],
        horizon=horizon,
        shifts=[horizon, horizon + 6],
        rolling_windows=[7],
        rolling_aggregations=("mean", "std"),
        # ACIKCA veriliyor: bu history hedef degil kovaryat ("tuketim_kwh")
        # uzerinde. Nobetci, ikisinin ayni olmadigini dogrular.
        target_column=TARGET,
    )
    train_features, test_features = history.train, history.test

    if weather_ok:
        matched = train_features["isitma_derece_gun"].notna().mean()
        check("hava join", matched > 0.9, f"eslesme orani %{matched * 100:.1f}")

    # Komsu ilce sinyali -- hedef gecmisinden, ufuk kaydirmali
    if neighbours is not None:
        district_daily = (
            train_features.assign(_ilce=train_features["ilce"].map(join_key))
            .groupby(["_ilce", TIME], observed=True)[TARGET]
            .sum()
            .reset_index()
            .rename(columns={"_ilce": "ilce_key"})
        )
        district_daily = add_neighbour_target_lag(
            district_daily,
            neighbours,
            key_column="ilce_key",
            time_column=TIME,
            target_column=TARGET,
            horizon=horizon,
            statistics=("mean",),
        )
        neighbour_column = f"komsu_{TARGET}_ufuk{horizon}_mean"
        check(
            "komsu sinyali",
            neighbour_column in district_daily.columns,
            f"{district_daily[neighbour_column].notna().mean():.1%} dolu",
        )

    print(f"  train {train_features.shape}   test {test_features.shape}")

    # ------------------------------------------------------------------ 5
    banner("5/9", "CV + SIZINTI KONTROLU")
    folds = purged_time_series_split(
        train_features[TIME],
        n_splits=3 if args.hizli else 4,
        embargo=pd.Timedelta(days=max(horizon, 30)),
        verbose=False,
    )
    check("fold uretimi", len(folds) >= 2, f"{len(folds)} fold")

    drop = {TARGET, "ariza_var_mi", "ariza_tipi", ID, TIME, "tarih_hava"}
    columns = [
        c
        for c in train_features.columns
        if c not in drop
        and c in test_features.columns
        and pd.api.types.is_numeric_dtype(train_features[c])
    ]

    y = train_features[TARGET].to_numpy(dtype="float64")
    train_encoded, test_encoded = oof_target_encode(
        train_features,
        pd.Series(y),
        ["ilce"],
        folds,
        test=test_features,
        smoothing=30.0,
        uncovered_policy="nan",
    )
    columns = columns + ["ilce_hedef_kod"]
    check("OOF hedef kodlama", "ilce_hedef_kod" in train_encoded.columns)

    sample_size = min(8000, len(train_encoded), len(test_encoded))
    adversarial = adversarial_validation(
        train_encoded[columns].sample(sample_size, random_state=1),
        test_encoded[columns].sample(sample_size, random_state=1),
        n_splits=3,
    )
    check("adversarial validation", True, f"AUC={adversarial['auc']:.4f}")

    # ------------------------------------------------------------------ 6
    banner("6/9", "FEATURE SECIMI")
    y_log = log_transform_target(y)
    params = starter_params("lightgbm", "regression")
    params["n_estimators"] = 150 if args.hizli else 300

    null_result = null_importance_filter(
        train_encoded[columns], y_log, params=dict(params), n_runs=2, verbose=False
    )
    check(
        "null importance",
        len(null_result["keep"]) > 0,
        f"{len(null_result['keep'])} tutuldu / {len(null_result['drop'])} atildi",
    )

    selected = null_result["keep"] or columns

    selection = shap_backward_selection(
        train_encoded[selected],
        y_log,
        folds,
        metric="rmse",
        params=dict(params),
        drop_per_step=5,
        min_features=8,
        max_steps=2 if args.hizli else 3,
        patience=2,
        shap_sample=800,
        progress=None,
    )
    check(
        "SHAP geri eleme",
        len(selection.best_features) > 0,
        f"{len(selected)} -> {len(selection.best_features)} feature",
    )
    final_columns = selection.best_features

    # ------------------------------------------------------------------ 7
    banner("7/9", "EGITIM + TAM VERI REFIT")
    result = cross_validate(
        train_encoded[final_columns],
        y,
        folds,
        kind="lightgbm",
        metric="rmsle",
        params=dict(params),
        test=test_encoded[final_columns],
        verbose=False,
        target_transform="log1p",
        early_stopping_metric="rmsle",
        early_stopping_rounds=200,
    )
    check(
        "cross_validate",
        np.isfinite(result.overall_score),
        f"CV RMSLE (ham hedef)={result.overall_score:.5f}",
    )

    # mean_train_fraction OLCULEREK veriliyor: purged_time_series_split
    # genisleyen pencere kullanir, yani (k-1)/k varsayimi bu repoda YANLIS
    # (olculdu: gercek ortalama 0.550, varsayim 0.800 -> %52 eksik agac).
    rounds = estimate_full_data_rounds(
        extract_best_iterations(result.models) or [params["n_estimators"]],
        n_folds=len(folds),
        mean_train_fraction=fold_train_fraction(folds, len(train_encoded)),
    )
    refit = multi_seed_refit(
        train_encoded[final_columns],
        y_log,
        test_encoded[final_columns],
        params=dict(params),
        n_estimators=rounds,
        seeds=(0, 1, 2),
        verbose=False,
    )
    check(
        "multi_seed_refit",
        refit.per_seed_predictions.shape[0] == 3,
        f"{rounds} agac, tohum sapmasi {refit.seed_disagreement:.5f}",
    )

    # ------------------------------------------------------------------ 8
    banner("8/9", "SUBMISSION + HOLDOUT")
    refit_raw = inverse_log_transform(refit.predictions)
    blended = 0.5 * result.test_predictions + 0.5 * refit_raw
    predictions = postprocess_predictions(blended, clip_min=0.0, verbose=False)
    path = write_submission(
        test_encoded[ID].to_numpy(),
        predictions,
        ROOT / "submissions" / "tam_pipeline.csv",
        id_column=ID,
        target_column=TARGET,
        validate=True,
    )

    fold_plan = FoldPlan.from_folds(folds, n_rows=len(train_encoded))
    run_recipe = PipelineRecipe(
        seed=42,
        cv=CVRecipe(
            n_splits=len(folds),
            splitter="purged_time_series",
            embargo_days=max(horizon, 30),
        ),
        features=FeatureRecipe(
            horizon=horizon,
            target_shifts=(),
            rolling_windows=(),
            history_value_columns=("tuketim_kwh",),
            history_shifts=(horizon, horizon + 6),
            history_rolling_windows=(7,),
            history_rolling_aggregations=("mean", "std"),
            families=(
                "calendar",
                "holiday",
                "consumption_history",
                "oof_target_encoding",
            )
            + (("weather",) if weather_ok else ()),
        ),
        model=ModelRecipe(
            kind="lightgbm",
            objective=str(params.get("objective", "regression")),
            metric="rmsle",
            early_stopping_metric="rmsle",
            n_estimators=int(params["n_estimators"]),
            early_stopping_rounds=200,
        ),
    )
    run_fingerprint = runtime_recipe_fingerprint(
        run_recipe.to_dict(),
        target_transform="log1p",
        refit_rounds=rounds,
        refit_seeds=(0, 1, 2),
    )
    provenance = ExperimentProvenance.capture(
        recipe_fingerprint=run_fingerprint,
        data_artifacts=[DataArtifact.from_path(path)],
        feature_names=final_columns,
        fold_fingerprint=fold_plan.fingerprint,
    )
    record = SQLiteExperimentStore(ROOT / "experiments" / "experiments.db").add(
        ExperimentRecord(
            name=f"tam_pipeline_lightgbm_{run_fingerprint[:8]}",
            cv_score=result.overall_score,
            metric="rmsle",
            model_kind="lightgbm",
            n_features=len(final_columns),
            fold_scores=list(result.fold_scores),
            params=dict(params),
            features=list(final_columns),
            notes="full_pipeline; score_space=raw; target_transform=log1p",
            submission_path=str(path),
            provenance=provenance,
        )
    )
    check("deney provenance", bool(record.run_id), f"run_id={record.run_id}")

    merged = pd.read_csv(path).merge(solution, on=ID, suffixes=("_p", "_g"))
    truth = merged[f"{TARGET}_g"].to_numpy()
    score = rmsle(truth, merged[f"{TARGET}_p"].to_numpy())
    baseline = rmsle(truth, np.full_like(truth, float(np.median(y))))
    gain = (baseline - score) / baseline * 100
    check(
        "holdout baseline'i geciyor",
        score < baseline,
        f"RMSLE {score:.5f} vs {baseline:.5f}  (%{gain:.1f} kazanc)",
    )

    zero_base = zero_baseline_score(truth, metric="mae")
    print(f"  'hep sifir' MAE baseline: {zero_base:.4f}")

    # ------------------------------------------------------------------ 9
    banner("9/9", "JURI CIKTILARI")
    fold_table = cv_fold_table(result, name="lightgbm")
    check("cv_fold_table", len(fold_table) == len(folds) + 3)

    segments = test_encoded["il"].reset_index(drop=True)
    segment_table = error_by_segment(
        truth, merged[f"{TARGET}_p"].to_numpy(), segments, metric="mae", min_count=10
    )
    check("error_by_segment", len(segment_table) > 0, f"{len(segment_table)} segment")
    print("\n  En kotu segmentler:")
    print(segment_table.tail(3).to_string(index=False))

    calibration = prediction_vs_actual_table(truth, merged[f"{TARGET}_p"].to_numpy(), bins=5)
    check("kalibrasyon tablosu", len(calibration) > 0)

    families_table = feature_importance_table(
        result, group_prefixes=("tarih_", "tatil_", "tuketim_", "ilce", "bolge_")
    )
    check("feature aile tablosu", len(families_table) > 0)
    print("\n  Sinyal kaynagi:")
    print(families_table.to_string(index=False))

    footprint = model_footprint(result.models, elapsed_seconds=result.elapsed_seconds)
    check(
        "model ayak izi",
        footprint["toplam_boyut_mb"] >= 0,
        f"{footprint['toplam_boyut_mb']} MB, {footprint['toplam_agac']} agac",
    )

    impact = business_impact(truth, merged[f"{TARGET}_p"].to_numpy(), unit_label="dakika")
    print(f"\n  Is degeri: {impact['ozet']}")

    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - started
    failed = [name for name, ok, _ in checks if not ok]
    banner(
        "SONUC", f"{len(checks) - len(failed)}/{len(checks)} kontrol gecti  ({elapsed / 60:.1f} dk)"
    )
    if failed:
        print("  BASARISIZ:")
        for name in failed:
            print(f"    - {name}")
        return 1

    print("  Tum moduller BIRLIKTE calisiyor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
