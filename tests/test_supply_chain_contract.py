"""Tedarik zinciri, paketleme ve yayin kapisi sozlesmeleri."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _requirement_names(path: Path) -> set[str]:
    names: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.split("#", 1)[0].strip()
        if not value or value.startswith("-"):
            continue
        names.add(value.split(">", 1)[0].split("=", 1)[0].strip().lower())
    return names


def _load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


#: CI kalite matrisinde KOSMASI gereken Python surumleri.
#: 2026-08-18: dort surumden ikiye indirildi. Tutulanlarin gercek tuketicisi
#: var -- 3.11 gelistirici makinesi, 3.12 KAGGLE (juri notebook'unun kostugu
#: ortam). 3.10/3.13 kaldirildi: kullanan yok, her push'ta gurultu uretiyordu.
BEKLENEN_CI_SURUMLERI = ("3.11", "3.12")


def test_ci_matrix_and_required_gates_are_explicit():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    # Matris girdileri SATIR BAZINDA aranir, metin icinde degil.
    # Onceki hali tum dosyada alt dize ariyordu ve bir YORUM satirinda gecen
    # "3.10" bile testi geciriyordu -- olculdu: matris 3.10'u kaldirdiktan
    # sonra test yine GECTI, cunku gerekce yorumunda surum adi geciyordu.
    # Sahte gecen bir kapi, kapi olmayandan daha tehlikelidir.
    matris = {
        satir.split('"')[1]
        for satir in workflow.splitlines()
        if satir.strip().startswith("- python:") and '"' in satir
    }
    assert matris == set(BEKLENEN_CI_SURUMLERI), (
        f"CI matrisi {sorted(matris)} kosuyor, beklenen {sorted(BEKLENEN_CI_SURUMLERI)}. "
        "Surum ekler/cikarirsan BEKLENEN_CI_SURUMLERI'ni de guncelle."
    )
    assert "3.12" in matris, (
        "Kaggle Python 3.12 kullaniyor. Bu surum matristen CIKARILAMAZ: "
        "yerelde calisip Kaggle'da patlayan kod eleme sebebidir."
    )
    for gate in (
        "ruff check",
        "pytest",
        "--cov-fail-under",
        "python -m build",
        "twine check",
        "pip-audit",
        "scan_secrets.py",
        "verify_sources.py",
        "cyclonedx",
        "ruff format --check",
        "mypy",
        "windows-latest",
        "gridup-doctor --version",
        "gridup-validate-submission --help",
        "scripts/smoke_test.py",
    ):
        assert gate in workflow


def test_manual_release_workflow_enforces_publication_gate() -> None:
    workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch" in workflow
    assert "verify_sources.py" in workflow
    assert "--publication" in workflow
    assert "python -m build" in workflow


def test_cross_platform_transitive_lock_is_hash_backed_and_ci_checked() -> None:
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "[[package]]" in lock
    assert "sdist = { url =" in lock
    assert 'hash = "sha256:' in lock
    assert "uv lock --check" in workflow
    assert "uv sync --locked" in workflow
    assert "uv run python -m pytest" in workflow
    assert "python -m pip install\n          --constraint" not in workflow


def test_kaggle_documentation_exposes_only_the_guarded_upload_path() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    package_script = (ROOT / "scripts/build_kaggle_package.py").read_text(encoding="utf-8")

    assert "build_kaggle_package.py --wheels --upload" in readme
    assert "kaggle datasets version -p kaggle_paket" not in readme
    assert 'print(f"  kaggle datasets create' not in package_script


def test_defence_in_depth_and_neural_workflows_are_explicit() -> None:
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    neural = (ROOT / ".github/workflows/neural.yml").read_text(encoding="utf-8")
    release = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")

    assert "gitleaks/gitleaks-action@" in ci
    assert "uv sync --locked --extra neural --extra dev" in neural
    assert "tests/test_neural.py" in neural
    assert "attest-build-provenance@" in release
    assert "sbom-path:" in release
    assert "artifact-metadata: write" in release
    assert "id-token: write" in release
    assert "attestations: write" in release


def test_yarisma_penceresinde_otomatik_tetikleyici_yok() -> None:
    """Yarisma boyunca (21 Agustos - 1 Eylul) kendiliginden kosan is OLMAMALI.

    2026-08-17 karari: ``dependabot.yml`` silindi (haftalik bagimlilik PR'lari)
    ve ``neural.yml``in haftalik cron'u kaldirildi. Gerekce, gurultunun kendisi
    degil dikkat maliyetidir: yarisma penceresinde acilan her otomatik PR ve
    her pazartesi sabahi dusen kirmizi/yesil bildirim, hicbir kapiyi
    korumadigi halde bakilmayi talep eder.

    KORUNANLAR: ``ci.yml`` her push'ta kosar (gercek kapi) ve ``release.yml``
    yalnizca ``workflow_dispatch`` ile tetiklenir (kendiliginden kosmaz).

    Yarisma bittiginde ikisi de geri konabilir; bu test o zaman guncellenir.
    """
    workflows = ROOT / ".github/workflows"

    assert not (ROOT / ".github/dependabot.yml").exists(), (
        "dependabot.yml yarisma penceresinde bilincli olarak kaldirildi."
    )
    assert "schedule:" not in (workflows / "neural.yml").read_text(encoding="utf-8"), (
        "neural.yml yalnizca elle tetiklenmeli; haftalik cron kaldirildi."
    )
    assert "schedule:" not in (workflows / "release.yml").read_text(encoding="utf-8")
    # ci.yml push'ta kosmaya DEVAM etmeli -- kaldirilan gurultu, korunan kapi.
    assert "push:" in (workflows / "ci.yml").read_text(encoding="utf-8")


def test_all_supported_constraints_pin_patched_jupyterlab():
    for constraint in sorted((ROOT / "requirements").glob("constraints-py*.txt")):
        assert "jupyterlab==4.5.10" in constraint.read_text(encoding="utf-8")


def test_local_setup_uses_the_same_exact_lock_as_ci():
    setup = (ROOT / "setup.ps1").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "uv sync --locked --extra full --extra dev" in setup
    assert "uv run python -m pytest" in setup
    assert "uv sync --locked --extra full --extra dev" in readme
    assert "--require-hashes -r requirements/uv-bootstrap.txt" in setup
    assert "--require-hashes -r requirements/uv-bootstrap.txt" in readme

    bootstrap = (ROOT / "requirements/uv-bootstrap.txt").read_text(encoding="utf-8")
    assert "uv==0.12.5" in bootstrap
    assert bootstrap.count("--hash=sha256:") >= 10


def test_each_supported_python_graph_is_audited() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    quality = workflow[workflow.index("jobs:") : workflow.index("windows-smoke:")]

    assert "Audit locked graph for this Python" in quality
    assert "uv export --locked" in quality
    assert "uv run pip-audit" in quality


def test_pyproject_exposes_dev_and_security_extras():
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 CI yolu
        import tomli as tomllib

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    extras = project["optional-dependencies"]
    assert {"dev", "security"}.issubset(extras)
    assert any(item.startswith("pytest") for item in extras["dev"])
    assert any(item.startswith("pip-audit") for item in extras["security"])
    assert not any(item.startswith("torch") for item in extras["full"])
    assert any(item.startswith("torch") for item in extras["neural"])

    doctor = (ROOT / "scripts/ekip_kontrol.py").read_text(encoding="utf-8")
    assert "import tomli as tomllib" in doctor


def test_documented_and_powershell_setups_install_the_test_contract() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    requirements = _requirement_names(ROOT / "requirements.txt")

    assert "uv sync --locked --extra full --extra dev" in readme
    assert {"pytest", "pytest-cov", "hypothesis", "ruff"} <= requirements


def test_exact_wheel_manifest_accepts_only_expected_hash(tmp_path):
    package = _load_script("build_kaggle_supply_contract", "scripts/build_kaggle_package.py")
    wheel = tmp_path / "ornek_paket-1.2.3-py3-none-any.whl"
    wheel.write_bytes(b"gercek-wheel-baytlari")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "wheels": [
                    {
                        "name": "ornek-paket",
                        "version": "1.2.3",
                        "filename": wheel.name,
                        "sha256": _sha256(wheel),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    verified = package.dogrula_wheel_manifesti([wheel], manifest)
    assert verified == [wheel]

    wheel.write_bytes(b"degistirilmis-wheel")
    with pytest.raises(package.SupplyChainError, match="SHA256"):
        package.dogrula_wheel_manifesti([wheel], manifest)


@pytest.mark.parametrize("digest", ["unverified", "", "0" * 63])
def test_wheel_manifest_fails_closed_for_unverified_or_invalid_digest(tmp_path, digest):
    package = _load_script("build_kaggle_unverified_contract", "scripts/build_kaggle_package.py")
    wheel = tmp_path / "ornek_paket-1.2.3-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "wheels": [
                    {
                        "name": "ornek-paket",
                        "version": "1.2.3",
                        "filename": wheel.name,
                        "sha256": digest,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(package.SupplyChainError, match="dogrulanmamis|gecersiz"):
        package.dogrula_wheel_manifesti([wheel], manifest)


def test_dataset_metadata_does_not_mislabel_mixed_bundle_as_cc0(tmp_path):
    package = _load_script("build_kaggle_license_contract", "scripts/build_kaggle_package.py")
    metadata = package.metadata_yaz(tmp_path, "kullanici")
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    licenses = {item["name"] for item in payload["licenses"]}

    assert "CC0-1.0" not in licenses
    assert licenses == {"other"}


def test_publication_gate_rejects_unverified_source(tmp_path):
    package = _load_script("build_kaggle_publication_contract", "scripts/build_kaggle_package.py")
    artifact = tmp_path / "veri.parquet"
    artifact.write_bytes(b"veri")
    manifest = tmp_path / "sources.yml"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bundle_license": "NOASSERTION",
                "artifacts": [
                    {
                        "path": str(artifact),
                        "sha256": "unverified",
                        "license": "NOASSERTION",
                        "redistribution": "unverified",
                        "schema": {"format": "parquet", "required_columns": []},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(package.SupplyChainError, match="yayin"):
        package.yayin_kapisini_dogrula(manifest, root=tmp_path)


def test_publication_gate_uses_full_immutable_and_schema_contract(tmp_path):
    package = _load_script(
        "build_kaggle_full_publication_contract", "scripts/build_kaggle_package.py"
    )
    artifact = tmp_path / "veri.csv"
    artifact.write_text("id,deger\n1,2\n", encoding="utf-8")
    manifest = tmp_path / "sources.yml"
    payload = {
        "schema_version": 1,
        "artifacts": [
            {
                "path": artifact.name,
                "sha256": _sha256(artifact),
                "license": "MIT",
                "redistribution": "allowed",
                "source": {
                    "uri": "https://example.invalid/veri.csv",
                    "snapshot_ref": "v1",
                    "immutable": False,
                },
                "schema": {
                    "format": "csv",
                    "min_rows": 1,
                    "required_columns": ["id", "eksik"],
                },
            }
        ],
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(package.SupplyChainError, match="immutable|sema|kolon"):
        package.yayin_kapisini_dogrula(manifest, root=tmp_path)


def test_staged_directory_publish_preserves_previous_output_on_swap_failure(tmp_path, monkeypatch):
    package = _load_script(
        "build_kaggle_atomic_publish_contract", "scripts/build_kaggle_package.py"
    )
    target = tmp_path / "kaggle_paket"
    target.mkdir()
    (target / "old.txt").write_text("old", encoding="utf-8")
    staging = tmp_path / ".kaggle_paket.staging"
    staging.mkdir()
    (staging / "new.txt").write_text("new", encoding="utf-8")
    real_replace = os.replace
    calls = 0

    def fail_second_replace(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated publish failure")
        return real_replace(source, destination)

    monkeypatch.setattr(package.os, "replace", fail_second_replace)
    with pytest.raises(OSError, match="simulated"):
        package.atomik_dizin_yayinla(staging, target)

    assert (target / "old.txt").read_text(encoding="utf-8") == "old"
    assert not (target / "new.txt").exists()


def test_secret_scanner_detects_high_risk_token_formats_without_leaking_value():
    scanner = _load_script("secret_scan_contract", "security/scan_secrets.py")
    samples = (
        "GITLAB_TOKEN=" + "glpat-" + "abcdefghijklmnopqrstuvwxyz",
        "SLACK_TOKEN=" + "xoxb-" + "123456789012-abcdefghijklmnopqrstuvwx",
        "NPM_TOKEN=" + "npm_" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        "PYPI_TOKEN=" + "pypi-AgEI" + "cHlwaS5vcmcCJGFiY2RlZmdoaWprbG1ub3BxcnN0dXZ3eHl6",
    )

    for sample in samples:
        findings = scanner.scan_text(sample, "config.txt")
        assert findings, sample
        assert all(finding.fingerprint not in sample for finding in findings)


def test_secret_scanner_detects_unquoted_assignments_and_jwts():
    scanner = _load_script("secret_scan_entropy_contract", "security/scan_secrets.py")
    samples = (
        "SERVICE_TOKEN=" + "Q7v9X2m4K8p6R3t5W1y0Z9a8",
        "AUTH=" + "eyJ" + "hbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signaturevalue123",
    )

    for sample in samples:
        findings = scanner.scan_text(sample, "config.env")
        assert findings, sample


def test_repository_source_manifest_hashes_and_schemas_are_valid():
    verifier = _load_script("verify_sources_contract", "security/verify_sources.py")
    manifest = ROOT / "data/sources.yml"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not all((ROOT / item["path"]).is_file() for item in payload["artifacts"]):
        pytest.skip("harici artifactler bu checkout'ta yok; CI metadata kapisi ayri calisir")

    result = verifier.verify_manifest(manifest, root=ROOT, publication=False)

    assert result.checked_artifacts >= 8
    assert not result.errors


def _manifest_yaz(tmp_path, artefakt: dict) -> Path:
    """Tek artefaktlik gecici manifest.

    ``check_files=False`` ile kullanilir: burada olculen sey lisans/immutable
    KARAR MANTIGI, dosya semasi degil. Hash gecerliligi kontrolu zaten
    ``check_files``tan bagimsiz calisir, o yuzden gecerli bir digest veriyoruz.
    """
    manifest = tmp_path / "sources.yml"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifacts": [{"path": "artefakt.bin", "sha256": "0" * 64, **artefakt}],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_yayin_kapisi_lisanssiz_artefakti_hala_reddeder(tmp_path) -> None:
    """Kapinin HUKUKI katiligi degismedi: lisans/izin eksikse yayin BLOKE."""
    from security.verify_sources import verify_manifest

    for bozuk in (
        {"license": "NOASSERTION", "redistribution": "allowed"},
        {"license": "MIT", "redistribution": "unverified"},
    ):
        manifest = _manifest_yaz(
            tmp_path,
            {**bozuk, "source": {"snapshot_ref": "x", "immutable": True}},
        )
        sonuc = verify_manifest(manifest, root=tmp_path, publication=True, check_files=False)
        assert sonuc.errors, f"Yayin kapisi bunu reddetmeliydi: {bozuk}"


def test_yayin_kapisi_degisebilir_ust_kaynagi_bloke_etmez(tmp_path) -> None:
    """``immutable=false`` tek basina yayini engellememelidir -- UYARI kalir.

    2026-08-17 karari. Bu kosul bir YENIDEN URETILEBILIRLIK ozelligidir,
    dagitilan baytlarin hukuki durumu degil: ``sha256`` gonderdigimiz snapshot'i
    zaten sabitliyor ve ayni fonksiyonda dosyaya karsi dogrulaniyor. Birlesik
    haldeyken kapi, lisansi yeniden dagitima ACIKCA izin veren CC-BY-4.0
    Open-Meteo verisinin yayinini engelliyordu -- yanlis pozitif.
    """
    from security.verify_sources import verify_manifest

    manifest = _manifest_yaz(
        tmp_path,
        {
            "license": "CC-BY-4.0",
            "redistribution": "allowed",
            "source": {"snapshot_ref": "2020..2026", "immutable": False},
        },
    )
    sonuc = verify_manifest(manifest, root=tmp_path, publication=True, check_files=False)

    assert not sonuc.errors, f"immutable=false tek basina yayini bloke etmemeli: {sonuc.errors}"
    assert any("immutable=false" in u for u in sonuc.warnings), (
        "Uyari KAYBOLMAMALI -- bilgi korunmali, yalnizca bloke etmemeli."
    )


def test_paket_veri_listesi_kaynak_manifestinden_turetiliyor():
    """VERI_DOSYALARI == sources.yml artefaktlari; elle liste ayrismasi kapali.

    2026-08-18 denetimi: izsu manifestte var pakette yok, turizm_aylik_il
    yalnizca listede vardi; Kaggle'a yuklenen paket bayat kaldi.
    """
    package = _load_script("build_kaggle_manifest_contract", "scripts/build_kaggle_package.py")
    manifest = json.loads((ROOT / "data" / "sources.yml").read_text(encoding="utf-8"))
    beklenen = tuple(a["path"] for a in manifest["artifacts"])
    assert beklenen == package.VERI_DOSYALARI
    assert len(beklenen) >= 10
