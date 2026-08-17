"""Atomik cikti ve mutable kaynak dogrulama sozlesmeleri."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

from gridup import io_utils
from gridup.io_utils import (
    atomic_write_bytes,
    publish_bytes,
    publish_dataframe,
    validate_cached_file,
    validate_published_dataframe,
)


def test_atomic_write_failure_preserves_previous_target(tmp_path, monkeypatch):
    """os.replace basarisizsa eski hedef korunur ve temp dosya kalmaz."""
    target = tmp_path / "dataset.json"
    target.write_bytes(b'{"state":"old"}')

    def fail_replace(_source, _target):
        raise OSError("simulated publish interruption")

    monkeypatch.setattr(io_utils.os, "replace", fail_replace)

    with pytest.raises(OSError, match="interruption"):
        atomic_write_bytes(target, b'{"state":"new"}')

    assert target.read_bytes() == b'{"state":"old"}'
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


@pytest.mark.parametrize("suffix", [".csv", ".json", ".parquet"])
def test_dataframe_publish_is_atomic_and_has_verifiable_metadata(tmp_path, suffix):
    """Her ana tablo formati hash/sema/satir metadata'siyla yayinlanir."""
    target = tmp_path / f"districts{suffix}"
    frame = pd.DataFrame({"id": [1, 2], "district": ["Buca", "Cesme"]})

    metadata = publish_dataframe(
        frame,
        target,
        required_columns=("id", "district"),
        min_rows=2,
        source="unit-test://districts",
    )
    loaded = validate_published_dataframe(
        target,
        required_columns=("id", "district"),
        min_rows=2,
    )

    assert loaded.to_dict(orient="records") == frame.to_dict(orient="records")
    assert metadata.sha256 == io_utils.sha256_file(target)
    sidecar = json.loads(io_utils.metadata_path(target).read_text(encoding="utf-8"))
    assert sidecar["rows"] == 2
    assert sidecar["columns"] == ["id", "district"]
    assert sidecar["source"] == "unit-test://districts"


def test_dataframe_validation_fails_before_replacing_valid_target(tmp_path):
    """Eksik sema veya yetersiz satir eski gecerli yayini bozamaz."""
    target = tmp_path / "weather.parquet"
    valid = pd.DataFrame({"konum": ["Izmir", "Aydin"], "tarih": ["a", "b"]})
    publish_dataframe(
        valid,
        target,
        required_columns=("konum", "tarih"),
        min_rows=2,
        source="unit-test://weather",
    )
    old_hash = io_utils.sha256_file(target)

    with pytest.raises(ValueError, match="gerekli kolon"):
        publish_dataframe(
            pd.DataFrame({"konum": ["Izmir"]}),
            target,
            required_columns=("konum", "tarih"),
            min_rows=2,
            source="unit-test://weather",
        )

    assert io_utils.sha256_file(target) == old_hash


def test_mutable_binary_cache_requires_matching_hash_metadata(tmp_path):
    """Indirilmis ham dosya metadata yoksa veya degismisse cache sayilmaz."""
    target = tmp_path / "bulletin.xlsx"
    target.write_bytes(b"untracked mutable response")

    with pytest.raises(ValueError, match="metadata"):
        validate_cached_file(target, min_bytes=10)

    content = b"validated mutable response"
    publish_bytes(
        content,
        target,
        source="https://example.test/bulletin.xlsx",
        min_bytes=10,
    )
    metadata = validate_cached_file(target, min_bytes=10)
    assert metadata.sha256 == io_utils.sha256_file(target)

    target.write_bytes(b"X" + content[1:])
    with pytest.raises(ValueError, match="SHA-256"):
        validate_cached_file(target, min_bytes=10)


def test_primary_publishers_do_not_write_dataframe_directly() -> None:
    """Ana cikti ureticilerinde pandas hedef dosyayi dogrudan acamaz."""
    root = Path(__file__).resolve().parents[1]
    relative_paths = (
        "src/gridup/submission.py",
        "scripts/fetch_deprem.py",
        "scripts/fetch_yangin.py",
        "scripts/fetch_hourly_weather.py",
    )
    violations: list[str] = []
    for relative in relative_paths:
        tree = ast.parse((root / relative).read_text(encoding="utf-8"), filename=relative)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            attribute = node.func.attr
            direct_csv = attribute == "to_csv" and (
                bool(node.args) or any(item.arg == "path_or_buf" for item in node.keywords)
            )
            if attribute == "to_parquet" or direct_csv:
                violations.append(f"{relative}:{node.lineno}:{attribute}")

    assert not violations, "Dogrudan pandas yazimi atomik degil: " + ", ".join(violations)
