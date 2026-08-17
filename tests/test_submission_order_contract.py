"""Sample submission ID sirasi, coklugu ve guvenli hizalama kontratlari."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridup import io_utils
from gridup.submission import blend_submissions, validate_submission, write_submission


def test_sample_id_order_is_strict_by_default() -> None:
    sample = pd.DataFrame({"ID": [3, 1, 2], "hedef": [0.0, 0.0, 0.0]})
    submission = pd.DataFrame({"ID": [1, 2, 3], "hedef": [10.0, 20.0, 30.0]})

    check = validate_submission(submission, sample=sample)

    assert not check.is_valid
    assert any("sira" in error.lower() for error in check.errors)


def test_sample_id_multiplicity_is_strict() -> None:
    sample = pd.DataFrame({"ID": [1, 1, 2], "hedef": [0.0, 0.0, 0.0]})
    submission = pd.DataFrame({"ID": [1, 2, 2], "hedef": [10.0, 20.0, 30.0]})

    check = validate_submission(submission, sample=sample)

    assert not check.is_valid
    assert any("cokluk" in error.lower() for error in check.errors)
    assert any("eksik" in error.lower() for error in check.errors)
    assert any("fazladan" in error.lower() for error in check.errors)


def test_sample_column_set_and_order_are_strict() -> None:
    sample = pd.DataFrame({"ID": [1, 2], "hedef": [0.0, 0.0]})
    extra = pd.DataFrame({"ID": [1, 2], "hedef": [1.0, 2.0], "debug": [9, 9]})
    reversed_columns = pd.DataFrame({"hedef": [1.0, 2.0], "ID": [1, 2]})

    extra_check = validate_submission(extra, sample=sample)
    order_check = validate_submission(reversed_columns, sample=sample)

    assert not extra_check.is_valid
    assert any("fazladan kolon" in error.lower() for error in extra_check.errors)
    assert not order_check.is_valid
    assert any("kolon sirasi" in error.lower() for error in order_check.errors)


def test_write_submission_explicit_alignment_keeps_id_prediction_pairs(tmp_path) -> None:
    sample = pd.DataFrame({"ID": [3, 1, 2], "hedef": [0.0, 0.0, 0.0]})

    path = write_submission(
        np.array([1, 2, 3]),
        np.array([10.0, 20.0, 30.0]),
        tmp_path / "aligned.csv",
        sample=sample,
        align_to_sample=True,
    )

    written = pd.read_csv(path)
    assert written["ID"].tolist() == [3, 1, 2]
    assert written["hedef"].tolist() == [30.0, 10.0, 20.0]


def test_write_submission_alignment_rejects_ambiguous_duplicate_ids(tmp_path) -> None:
    sample = pd.DataFrame({"ID": [1, 1, 2], "hedef": [0.0, 0.0, 0.0]})
    with pytest.raises(ValueError, match="benzersiz|tekrar"):
        write_submission(
            np.array([1, 2, 1]),
            np.array([10.0, 20.0, 30.0]),
            tmp_path / "ambiguous.csv",
            sample=sample,
            align_to_sample=True,
        )


def test_write_submission_publish_failure_preserves_previous_file(tmp_path, monkeypatch) -> None:
    """Atomik swap basarisizsa onceki gonderim kaybedilemez."""
    path = tmp_path / "submission.csv"
    path.write_bytes(b"ID,hedef\n1,old\n")

    def fail_replace(_source, _target):
        raise OSError("simulated submission swap failure")

    monkeypatch.setattr(io_utils.os, "replace", fail_replace)
    with pytest.raises(OSError, match="swap failure"):
        write_submission(
            np.array([1, 2]),
            np.array([10.0, 20.0]),
            path,
            validate=False,
        )

    assert path.read_bytes() == b"ID,hedef\n1,old\n"
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_blend_publish_failure_preserves_previous_file(tmp_path, monkeypatch) -> None:
    """Harman cikti swap'i kesilirse eski harman hedefi korunur."""
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    pd.DataFrame({"ID": [1, 2], "hedef": [1.0, 2.0]}).to_csv(first, index=False)
    pd.DataFrame({"ID": [1, 2], "hedef": [3.0, 4.0]}).to_csv(second, index=False)
    output = tmp_path / "blend.csv"
    output.write_bytes(b"ID,hedef\n1,old\n")

    def fail_replace(_source, _target):
        raise OSError("simulated blend swap failure")

    monkeypatch.setattr(io_utils.os, "replace", fail_replace)
    with pytest.raises(OSError, match="swap failure"):
        blend_submissions([first, second], output_path=output)

    assert output.read_bytes() == b"ID,hedef\n1,old\n"
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []
