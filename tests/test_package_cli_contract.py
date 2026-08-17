"""Wheel metadata, tekil surum ve hafif CLI sozlesmeleri."""

from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_build_metadata_exposes_wheel_console_scripts() -> None:
    metadata = _pyproject()
    scripts = metadata["project"]["scripts"]

    assert scripts == {
        "gridup-doctor": "gridup.cli:doctor_main",
        "gridup-validate-submission": "gridup.cli:validate_submission_main",
    }
    for target in scripts.values():
        module_name, function_name = target.split(":", maxsplit=1)
        assert callable(getattr(importlib.import_module(module_name), function_name))


def test_package_version_has_one_source_of_truth() -> None:
    metadata = _pyproject()

    assert "version" not in metadata["project"]
    assert metadata["project"]["dynamic"] == ["version"]
    assert metadata["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "gridup._version.__version__"
    }

    import gridup
    from gridup._version import __version__

    assert gridup.__version__ == __version__


def test_neural_dependency_remains_explicitly_optional_and_separately_tested() -> None:
    metadata = _pyproject()
    core = metadata["project"]["dependencies"]
    optional = metadata["project"]["optional-dependencies"]
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    neural_workflow = (ROOT / ".github" / "workflows" / "neural.yml").read_text(encoding="utf-8")

    assert all(not dependency.startswith("torch") for dependency in core)
    assert optional["neural"] == ["torch>=2.0"]
    assert ".[neural" not in workflow
    assert ".[full" not in workflow
    assert "--extra neural" in neural_workflow


def test_doctor_runs_with_core_dependencies_only(capsys) -> None:
    from gridup.cli import doctor_main

    assert doctor_main([]) == 0
    output = capsys.readouterr().out.lower()
    assert "gridup" in output
    assert "python" in output
    assert "torch" in output and "optional" in output


def test_validate_submission_cli_exit_codes_and_messages(tmp_path, capsys) -> None:
    from gridup.cli import validate_submission_main

    sample = tmp_path / "sample.csv"
    valid = tmp_path / "valid.csv"
    invalid = tmp_path / "invalid.csv"
    pd.DataFrame({"ID": [2, 1], "hedef": [0.0, 0.0]}).to_csv(sample, index=False)
    pd.DataFrame({"ID": [2, 1], "hedef": [20.0, 10.0]}).to_csv(valid, index=False)
    pd.DataFrame({"ID": [1, 2], "hedef": [10.0, 20.0]}).to_csv(invalid, index=False)

    assert validate_submission_main([str(valid), "--sample", str(sample)]) == 0
    assert "gecerli" in capsys.readouterr().out.lower()

    assert validate_submission_main([str(invalid), "--sample", str(sample)]) == 1
    failure = capsys.readouterr().out.lower()
    assert "gecersiz" in failure and "id sirasi" in failure

    assert validate_submission_main([str(tmp_path / "missing.csv")]) == 2
    assert "bulunamadi" in capsys.readouterr().err.lower()
