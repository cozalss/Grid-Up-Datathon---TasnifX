from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from gridup.experiment import (
    DataArtifact,
    ExperimentProvenance,
    _git_diff_fingerprint,
    sha256_file,
)
from gridup.recipe import FeatureRecipe, PipelineRecipe, default_pipeline_recipe


def test_recipe_json_roundtrip_and_fingerprint_are_deterministic():
    recipe = default_pipeline_recipe()

    restored = PipelineRecipe.from_json(recipe.to_json())

    assert restored == recipe
    assert restored.fingerprint == recipe.fingerprint
    assert json.loads(recipe.to_json())["schema_version"] == 1


def test_entrypoints_fingerprint_resolved_runtime_not_default_recipe() -> None:
    root = Path(__file__).resolve().parents[1]
    day_one = (root / "scripts/day_one.py").read_text(encoding="utf-8")
    smoke = (root / "scripts/smoke_test.py").read_text(encoding="utf-8")

    assert "runtime_recipe_fingerprint(" in day_one
    assert "target_transform=target_transform" in day_one
    assert "runtime_recipe_fingerprint(" in smoke
    assert "n_estimators=400" in smoke
    assert "early_stopping_rounds=50" in smoke


def test_recipe_override_changes_only_resulting_fingerprint():
    recipe = default_pipeline_recipe()
    changed = replace(recipe, seed=recipe.seed + 1)

    assert changed.cv == recipe.cv
    assert changed.features == recipe.features
    assert changed.fingerprint != recipe.fingerprint


def test_absolute_target_shifts_cannot_cross_forecast_horizon():
    with pytest.raises(ValueError, match="shift.*horizon"):
        FeatureRecipe(horizon=31, target_shifts=(30, 62, 93))


def test_non_target_history_has_an_explicit_serialized_contract() -> None:
    recipe = FeatureRecipe(
        horizon=31,
        target_shifts=(),
        rolling_windows=(),
        history_value_columns=("tuketim_kwh",),
        history_shifts=(31, 37, 58),
        history_rolling_windows=(7, 28),
        history_rolling_aggregations=("mean", "std"),
        families=("calendar", "holiday", "consumption_history", "oof_target_encoding"),
    )
    restored = PipelineRecipe.from_json(PipelineRecipe(features=recipe).to_json())

    assert restored.features == recipe
    assert restored.features.target_shifts == ()


def test_smoke_and_full_provenance_describe_the_executed_feature_graph() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in ("scripts/smoke_test.py", "scripts/full_pipeline.py"):
        source = (root / relative).read_text(encoding="utf-8")
        recipe_block = source[source.index("features=FeatureRecipe(") :]

        assert 'history_value_columns=("tuketim_kwh",)' in recipe_block
        assert '"consumption_history"' in recipe_block
        assert '"oof_target_encoding"' in recipe_block
        assert '"group"' not in recipe_block
        assert '"frequency"' not in recipe_block


def test_provenance_contains_content_hash_and_no_secret_values(tmp_path, monkeypatch):
    data = tmp_path / "train.csv"
    data.write_bytes(b"id,target\n1,2\n")
    monkeypatch.setenv("EPIAS_PASSWORD", "super-secret-value")

    artifact = DataArtifact.from_path(data)
    provenance = ExperimentProvenance.capture(
        recipe_fingerprint="a" * 64,
        data_artifacts=[artifact],
        feature_names=["x"],
        fold_fingerprint="b" * 64,
    )
    payload = json.dumps(provenance.to_dict(), ensure_ascii=False)

    assert artifact.sha256 == sha256_file(data)
    assert artifact.size_bytes == data.stat().st_size
    assert "super-secret-value" not in payload
    assert provenance.recipe_fingerprint == "a" * 64
    assert provenance.fold_fingerprint == "b" * 64
    assert provenance.feature_names == ("x",)
    assert provenance.command
    assert "numpy" in provenance.package_versions
    assert provenance.git_diff_fingerprint is None or len(provenance.git_diff_fingerprint) == 64


def test_git_diff_fingerprint_includes_untracked_files(tmp_path, monkeypatch):
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("baseline", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=tmp_path, check=True)
    monkeypatch.chdir(tmp_path)

    assert _git_diff_fingerprint() is None
    (tmp_path / "new.py").write_text("print('new')\n", encoding="utf-8")

    fingerprint = _git_diff_fingerprint()
    assert fingerprint is not None and len(fingerprint) == 64


def test_juri_notebooku_gercek_ambargoyu_kaydeder() -> None:
    """Notebook'un PROVENANCE kaydi, uretilen fold'larla ayni ambargoyu tasimalidir.

    ``CVRecipe.embargo_days`` varsayilani 0'dir ve bu "henuz yapilandirilmamis
    sablon" demektir. Notebook jeneratoru bu alani bos birakirsa, juriye giden
    notebook fold'lari ``embargo=max(HORIZON, 30)`` ile uretir ama kaydinda
    "0 gun ambargo" yazar -- yani belgelenen sema, kosan semadan BASKA olur.
    Bu bir sizinti degil, bir KAYIT YALANIDIR ve Kapi 2 (notebook
    degerlendirmesi) tam olarak bunu okur.
    """
    from pathlib import Path

    kaynak = Path(__file__).resolve().parents[1] / "scripts" / "build_notebooks.py"
    metin = kaynak.read_text(encoding="utf-8")

    assert "cv=CVRecipe(n_splits=len(folds))" not in metin, (
        "Notebook jeneratoru CVRecipe'i ambargosuz kuruyor: provenance kaydi "
        "gercekte kosan semadan farkli olur."
    )
    assert "embargo_days=max(HORIZON, 30)" in metin, (
        "Notebook'un CVRecipe kaydi, fold uretimindeki "
        "embargo=pd.Timedelta(days=max(HORIZON, 30)) ile ayni degeri tasimali."
    )
