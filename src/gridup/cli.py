"""Hafif, wheel ile kurulan GridUp komut satiri araclari."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from collections.abc import Sequence
from pathlib import Path

from ._version import __version__
from .submission import validate_submission

_CORE_DISTRIBUTIONS = ("numpy", "pandas", "scikit-learn", "packaging")


def doctor_main(argv: Sequence[str] | None = None) -> int:
    """Cekirdek kurulumun surumlerini denetler; optional paket istemez."""
    parser = argparse.ArgumentParser(
        prog="gridup-doctor",
        description="GridUp cekirdek kurulumunu ve calisma ortamini denetler.",
    )
    parser.add_argument("--version", action="version", version=f"gridup {__version__}")
    parser.parse_args(argv)

    print(f"gridup {__version__}")
    print(f"python {sys.version.split()[0]}")
    missing: list[str] = []
    for distribution in _CORE_DISTRIBUTIONS:
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            missing.append(distribution)
            print(f"HATA  {distribution}: kurulu degil")
        else:
            print(f"OK    {distribution}: {version}")

    print("BILGI torch: optional; neural modeller icin 'pip install gridup[neural]'")
    if missing:
        print(
            "HATA: eksik cekirdek bagimliliklar: " + ", ".join(missing),
            file=sys.stderr,
        )
        return 1
    print("OK: GridUp cekirdek ortami hazir.")
    return 0


def _read_submission_csv(path: Path, label: str):
    """CLI CSV girdisini anlasilir hata mesaji ile okur."""
    if not path.is_file():
        raise FileNotFoundError(f"{label} dosyasi bulunamadi: {path}")
    import pandas as pd

    separator = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        return pd.read_csv(path, sep=separator)
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        raise ValueError(f"{label} CSV okunamadi: {error}") from error


def validate_submission_main(argv: Sequence[str] | None = None) -> int:
    """Bir submission CSV'sini sample sirasi ve semasina gore dogrular."""
    parser = argparse.ArgumentParser(
        prog="gridup-validate-submission",
        description="Submission CSV dosyasini Kaggle'a yuklemeden once dogrula.",
    )
    parser.add_argument("submission", type=Path, help="Dogrulanacak CSV/TSV yolu")
    parser.add_argument("--sample", type=Path, help="Ornek submission CSV/TSV yolu")
    parser.add_argument("--id-column", default="ID", help="ID kolon adi")
    parser.add_argument("--target-column", default="hedef", help="Hedef kolon adi")
    parser.add_argument("--expected-rows", type=int, help="Beklenen satir sayisi")
    parser.add_argument(
        "--allow-negative",
        action="store_true",
        help="Negatif tahmin uyarisini fiziksel kisit olarak uygulama.",
    )
    args = parser.parse_args(argv)

    try:
        submission = _read_submission_csv(args.submission, "Submission")
        sample = (
            _read_submission_csv(args.sample, "Sample submission")
            if args.sample is not None
            else None
        )
        check = validate_submission(
            submission,
            sample=sample,
            id_column=args.id_column,
            target_column=args.target_column,
            allow_negative=args.allow_negative,
            expected_rows=args.expected_rows,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"HATA: {error}", file=sys.stderr)
        return 2

    print(check)
    return 0 if check.is_valid else 1


if __name__ == "__main__":  # pragma: no cover - console scripts asil giris yoludur
    raise SystemExit(doctor_main())
