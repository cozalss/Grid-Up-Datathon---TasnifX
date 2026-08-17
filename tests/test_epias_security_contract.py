"""EPIAS sir, redaksiyon, retry ve atomik yazma sozlesmeleri."""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import requests

from gridup.epias import EpiasAuthError, EpiasClient, EpiasRequestError, load_env_file

ROOT = Path(__file__).resolve().parents[1]


class _Response:
    def __init__(self, status: int, text: str, payload=None):
        self.status_code = status
        self.text = text
        self._payload = payload

    @property
    def ok(self):
        return 200 <= self.status_code < 400

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _load_fetcher():
    name = "fetch_epias_security_contract"
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/fetch_epias_load.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_env_file_never_overwrites_existing_credentials(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "EPIAS_USERNAME=stale@example.com\nEPIAS_PASSWORD=STALE-PASSWORD\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EPIAS_USERNAME", "runtime@example.com")
    monkeypatch.setenv("EPIAS_PASSWORD", "RUNTIME-PASSWORD")

    loaded = load_env_file(env_file)

    assert loaded["EPIAS_PASSWORD"] == "STALE-PASSWORD"
    assert os.environ["EPIAS_USERNAME"] == "runtime@example.com"
    assert os.environ["EPIAS_PASSWORD"] == "RUNTIME-PASSWORD"


def test_fetcher_does_not_reapply_loaded_env_over_runtime_secret(tmp_path, monkeypatch):
    fetcher = _load_fetcher()
    monkeypatch.setenv("EPIAS_USERNAME", "runtime@example.com")
    monkeypatch.setenv("EPIAS_PASSWORD", "RUNTIME-PASSWORD")
    monkeypatch.setattr(
        fetcher,
        "load_env_file",
        lambda _path: {
            "EPIAS_USERNAME": "stale@example.com",
            "EPIAS_PASSWORD": "STALE-PASSWORD",
        },
    )
    observed = {}

    class _Client:
        realtime_consumption = object()
        realtime_generation = object()

        @classmethod
        def from_env(cls):
            observed["username"] = os.environ["EPIAS_USERNAME"]
            observed["password"] = os.environ["EPIAS_PASSWORD"]
            return cls()

    monkeypatch.setattr(fetcher, "EpiasClient", _Client)
    monkeypatch.setattr(fetcher, "_cek", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr(
        sys,
        "argv",
        ["fetch_epias_load.py", "--out", str(tmp_path / "out")],
    )

    assert fetcher.main() == 0
    assert observed == {
        "username": "runtime@example.com",
        "password": "RUNTIME-PASSWORD",
    }


def test_401_retry_connection_failure_uses_same_redacted_error_contract(monkeypatch):
    secret = "REMOTE-EXCEPTION-SECRET"
    responses = [
        _Response(401, "expired"),
        requests.ConnectionError(f"socket failed {secret}"),
    ]

    def fake_post(*_args, **_kwargs):
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr("gridup.epias.requests.post", fake_post)
    client = EpiasClient("user@example.com", "LOCAL-SECRET")
    client._tgt = "old-ticket"
    client._tgt_issued_at = time.time()
    monkeypatch.setattr(client, "_fetch_tgt", lambda: "new-ticket")

    with pytest.raises(EpiasRequestError) as captured:
        client.post("resource", {"x": 1})

    message = str(captured.value)
    assert secret not in message
    assert "LOCAL-SECRET" not in message
    assert "baglanti hatasi" in message
    assert responses == []


@pytest.mark.parametrize(
    ("status", "payload"),
    [
        (500, None),
        (200, ValueError("REMOTE-JSON-SECRET")),
    ],
)
def test_remote_response_body_is_never_copied_into_errors(monkeypatch, status, payload):
    remote_secret = "password=REMOTE-BODY-SECRET"
    monkeypatch.setattr(
        "gridup.epias.requests.post",
        lambda *_args, **_kwargs: _Response(status, remote_secret, payload),
    )
    client = EpiasClient("user@example.com", "LOCAL-SECRET")
    client._tgt = "ticket"
    client._tgt_issued_at = time.time()

    with pytest.raises(EpiasRequestError) as captured:
        client.post("resource")

    message = str(captured.value)
    assert remote_secret not in message
    assert "REMOTE-BODY-SECRET" not in message
    assert "sha256=" in message


def test_auth_failure_body_is_redacted(monkeypatch):
    secret = "password=REMOTE-AUTH-SECRET"
    monkeypatch.setattr(
        "gridup.epias.requests.post",
        lambda *_args, **_kwargs: _Response(403, secret),
    )

    with pytest.raises(EpiasAuthError) as captured:
        EpiasClient("user@example.com", "LOCAL-SECRET")._fetch_tgt()

    assert secret not in str(captured.value)
    assert "REMOTE-AUTH-SECRET" not in str(captured.value)
    assert "sha256=" in str(captured.value)


def test_atomic_parquet_preserves_previous_file_when_write_fails(tmp_path, monkeypatch):
    fetcher = _load_fetcher()
    output = tmp_path / "result.parquet"
    output.write_bytes(b"onceki-gecerli-dosya")
    frame = pd.DataFrame({"x": [1]})

    def fail_write(_self, path, **_kwargs):
        Path(path).write_bytes(b"yarim")
        raise OSError("disk dolu")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_write)

    with pytest.raises(OSError, match="disk dolu"):
        fetcher._atomic_parquet(frame, output)

    assert output.read_bytes() == b"onceki-gecerli-dosya"
    assert not list(tmp_path.glob(".result.parquet.*.tmp"))


def test_chunked_fetch_fails_closed_when_any_chunk_is_missing(monkeypatch):
    fetcher = _load_fetcher()
    monkeypatch.setattr(fetcher, "CHUNK_DAYS", 1)
    monkeypatch.setattr(fetcher, "MAX_RETRY", 1)
    monkeypatch.setattr(fetcher, "PAUSE_SECONDS", 0.0)

    def partly_failing_fetch(*, start, end):
        del end
        if "2024-01-02" in start:
            raise EpiasRequestError("simulated chunk failure")
        return pd.DataFrame({"date": [start], "value": [1.0]})

    with pytest.raises(RuntimeError, match="eksik parca|tamamlanamadi"):
        fetcher._cek(
            object(),
            "tuketim",
            partly_failing_fetch,
            date(2024, 1, 1),
            date(2024, 1, 3),
        )
