"""Harici veri snapshot'larinin hash, sema, lisans ve provenance kapisi."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VerificationResult:
    checked_artifacts: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema(path: Path, file_format: str) -> tuple[int, set[str]]:
    if file_format == "parquet":
        import pyarrow.parquet as pq

        metadata = pq.read_metadata(path)
        return metadata.num_rows, set(metadata.schema.names)
    if file_format == "csv":
        import pandas as pd

        frame = pd.read_csv(path)
        return len(frame), set(frame.columns)
    raise ValueError(f"desteklenmeyen format: {file_format}")


def verify_manifest(
    manifest_path: Path,
    *,
    root: Path,
    publication: bool = False,
    check_files: bool = True,
) -> VerificationResult:
    """Manifesti dogrular; publication=True lisans ve immutable kaynaklari zorlar."""
    result = VerificationResult()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        result.errors.append(f"manifest okunamadi: {type(error).__name__}")
        return result
    if payload.get("schema_version") != 1:
        result.errors.append("schema_version=1 olmali")
        return result
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        result.errors.append("artifacts bos veya liste degil")
        return result

    seen: set[str] = set()
    for artifact in artifacts:
        result.checked_artifacts += 1
        relative = str(artifact.get("path", ""))
        if not relative or relative in seen:
            result.errors.append(f"gecersiz/tekrarlanan artifact yolu: {relative!r}")
            continue
        seen.add(relative)
        path = Path(relative)
        path = path if path.is_absolute() else root / path
        digest = str(artifact.get("sha256", "")).lower()
        valid_digest = re.fullmatch(r"[0-9a-f]{64}", digest) is not None
        if digest == "unverified" or not valid_digest:
            message = f"{relative}: hash dogrulanmamis"
            (result.errors if publication else result.warnings).append(message)

        license_name = str(artifact.get("license", ""))
        redistribution = str(artifact.get("redistribution", ""))
        source = artifact.get("source", {})
        # IKI AYRI SORU, IKI AYRI SONUC.
        #
        # (1) "Bunu dagitma HAKKIMIZ var mi ve dagittigimiz sey BUTUN mu?"
        #     Lisans, yeniden dagitim izni, snapshot_ref ve hash. Bunlar hukuki
        #     ve butunluk kosullaridir; publication'da BLOKE EDER. Katilik
        #     degismedi.
        #
        # (2) "Ust kaynagi birebir YENIDEN URETEBILIR miyiz?"
        #     ``immutable``. Bu bir yeniden-uretilebilirlik ozelligidir,
        #     dagittigimiz baytlarin hukuki durumu degil. Yayini bloke ETMEZ,
        #     her zaman UYARI kalir.
        #
        # Neden ayrildi (2026-08-17): ikisi tek kosulda birlesikken kapi,
        # lisansi yeniden dagitima ACIKCA izin veren CC-BY-4.0 Open-Meteo
        # verisinin yayinini engelliyordu -- yanlis pozitif. Gonderdigimiz
        # baytlari zaten ``sha256`` garantiliyor ve o hash bu fonksiyonda
        # dosyaya karsi DOGRULANIYOR; ust kaynagin sonradan degismis olmasi
        # gonderdigimiz snapshot'in ne butunlugunu ne de lisansini etkiler.
        # Bu bir gevsetme degil, kapinin dogru soruyu sormasidir: lisanssiz
        # veya izinsiz hicbir artefakt hala yayinlanamaz.
        engelleyici: list[str] = []
        if license_name in ("", "NOASSERTION"):
            engelleyici.append("lisans belirsiz")
        if redistribution != "allowed":
            engelleyici.append(f"yeniden dagitim={redistribution or 'bos'}")
        if source.get("snapshot_ref") in (None, "", "unverified"):
            engelleyici.append("snapshot_ref yok")
        if engelleyici:
            message = f"{relative}: {', '.join(engelleyici)}"
            (result.errors if publication else result.warnings).append(message)

        if source.get("immutable") is not True:
            result.warnings.append(
                f"{relative}: ust kaynak degisebilir (immutable=false) -- dagitilan "
                "snapshot hash ile sabit, ancak ayni sorgu bugun farkli cevap verebilir"
            )
        if not check_files:
            continue
        if not path.is_file():
            result.errors.append(f"{relative}: dosya yok")
            continue
        if valid_digest:
            actual = sha256_file(path)
            if actual != digest:
                result.errors.append(f"{relative}: SHA256 uyusmuyor")

        schema = artifact.get("schema", {})
        try:
            rows, columns = _schema(path, str(schema.get("format", "")))
        except (ImportError, OSError, ValueError) as error:
            result.errors.append(f"{relative}: sema okunamadi ({type(error).__name__})")
            continue
        minimum = schema.get("min_rows")
        if not isinstance(minimum, int) or rows < minimum:
            result.errors.append(f"{relative}: satir sayisi {rows}, gereken >= {minimum}")
        required = set(schema.get("required_columns", []))
        missing = sorted(required - columns)
        if missing:
            result.errors.append(f"{relative}: zorunlu kolonlar eksik: {missing}")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/sources.yml"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--publication", action="store_true")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Artifact dosyalari CI checkout'unda yoksa yalnizca manifest sozlesmesini denetle",
    )
    args = parser.parse_args()
    result = verify_manifest(
        args.manifest,
        root=args.root,
        publication=args.publication,
        check_files=not args.metadata_only,
    )
    for warning in result.warnings:
        print(f"UYARI: {warning}")
    for error in result.errors:
        print(f"HATA: {error}")
    print(
        f"Provenance: {result.checked_artifacts} artifact, "
        f"{len(result.errors)} hata, {len(result.warnings)} uyari"
    )
    return 1 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
