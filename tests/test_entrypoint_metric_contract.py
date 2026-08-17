"""Executable pipelines must report the official metric in raw target space."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_smoke_pipeline_uses_raw_rmsle_contract() -> None:
    source = (ROOT / "scripts/smoke_test.py").read_text(encoding="utf-8")

    assert "x_train, y_log" not in source
    assert 'metric="rmsle"' in source
    assert 'target_transform="log1p"' in source
    assert 'early_stopping_metric="rmsle"' in source


def test_full_pipeline_cv_and_blend_stay_in_raw_space() -> None:
    source = (ROOT / "scripts/full_pipeline.py").read_text(encoding="utf-8")

    assert (
        "result = cross_validate(\n"
        "        train_encoded[final_columns],\n"
        "        y,\n"
        "        folds," in source
    )
    assert 'metric="rmsle"' in source
    assert 'target_transform="log1p"' in source
    assert "inverse_log_transform(refit.predictions)" in source
    assert "inverse_log_transform(blended)" not in source


def test_generated_baseline_notebook_keeps_raw_target_for_scoring() -> None:
    source = (ROOT / "scripts/build_notebooks.py").read_text(encoding="utf-8")

    assert "y = log_transform_target(y)" not in source
    assert 'target_transform=("log1p" if LOG_TARGET else None)' in source
    assert "predictions = inverse_log_transform(predictions)" not in source


def test_day_one_only_auto_logs_for_the_equivalent_rmsle_contract() -> None:
    source = (ROOT / "scripts/day_one.py").read_text(encoding="utf-8")

    assert 'args.metric.lower() == "rmsle"' in source


def test_benchmark_declares_mae_early_stopping_or_fixed_rounds() -> None:
    source = (ROOT / "scripts/benchmark_gercek.py").read_text(encoding="utf-8")

    assert 'early_stopping_metric="mae"' in source
    assert 'target_transform="sqrt"' in source
    assert "sqrt_transform_target(y)" not in source
    assert "inverse_sqrt_transform(sonuc.oof_predictions)" not in source


def test_benchmark_json_publish_is_atomic() -> None:
    source = (ROOT / "scripts/benchmark_gercek.py").read_text(encoding="utf-8")

    assert "atomic_write_bytes(" in source
    assert "CIKTI.write_text(" not in source


def test_full_and_smoke_avoid_global_distribution_fit_before_temporal_cv() -> None:
    full = (ROOT / "scripts/full_pipeline.py").read_text(encoding="utf-8")
    smoke = (ROOT / "scripts/smoke_test.py").read_text(encoding="utf-8")

    for source in (full, smoke):
        assert "build_paired_distribution_features(" not in source
        assert "build_paired_history_features(" in source


def test_day_one_does_not_fit_frequency_counts_before_cv() -> None:
    source = (ROOT / "scripts/day_one.py").read_text(encoding="utf-8")

    assert "build_frequency_features(" not in source
    assert 'families=("calendar", "holiday")' in source
    assert 'families=("calendar", "holiday", "frequency")' not in source


def test_all_temporal_entrypoints_keep_global_frequency_encoding_disabled() -> None:
    for relative in (
        "scripts/real_data_rehearsal.py",
        "scripts/ablation_gercek.py",
        "scripts/benchmark_gercek.py",
        "scripts/build_notebooks.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "add_frequency_encoding(" not in source, relative


def test_generated_notebook_does_not_declare_an_unverified_winner() -> None:
    source = (ROOT / "scripts/build_notebooks.py").read_text(encoding="utf-8")

    assert "Kazanan tekil" not in source
    assert "kazanan tekil" not in source
    assert "bağımsız outer kanıt olmadan bilimsel kazanan değildir" in source


def test_full_pipeline_records_recipe_fold_plan_and_provenance() -> None:
    source = (ROOT / "scripts/full_pipeline.py").read_text(encoding="utf-8")

    for token in (
        "PipelineRecipe(",
        "FoldPlan.from_folds(",
        "ExperimentProvenance.capture(",
        "SQLiteExperimentStore(",
    ):
        assert token in source
