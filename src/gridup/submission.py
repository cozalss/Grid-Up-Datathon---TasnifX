"""Submission dosyasi uretimi ve dogrulamasi.

Yarismalarda kaybedilen puanlarin sasirtici bir kismi MODELDEN DEGIL, bozuk
submission dosyasindan gelir: eksik satir, yanlis kolon adi, NaN, yanlis sira,
bilimsel gosterim, negatif tahmin.

Bu modul her submission'i yazmadan ONCE dogrular ve sorunu acikca soyler.
Kaggle'in "Submission Scoring Error" mesaji sana hicbir sey soylemez.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["SubmissionCheck", "validate_submission", "write_submission", "blend_submissions"]


@dataclass(frozen=True)
class SubmissionCheck:
    """Submission dogrulama sonucu."""

    is_valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    stats: dict[str, float]

    def __str__(self) -> str:
        lines = ["GECERLI" if self.is_valid else "GECERSIZ"]
        for error in self.errors:
            lines.append(f"  HATA: {error}")
        for warning in self.warnings:
            lines.append(f"  UYARI: {warning}")
        if self.stats:
            summary = "  ".join(f"{key}={value:.4g}" for key, value in self.stats.items())
            lines.append(f"  Istatistik: {summary}")
        return "\n".join(lines)

    def raise_if_invalid(self) -> None:
        if not self.is_valid:
            raise ValueError("Submission gecersiz:\n" + "\n".join(self.errors))


def validate_submission(
    submission: pd.DataFrame,
    *,
    sample: pd.DataFrame | None = None,
    id_column: str = "ID",
    target_column: str = "hedef",
    allow_negative: bool = False,
    expected_rows: int | None = None,
) -> SubmissionCheck:
    """Submission dosyasini Kaggle'a yuklemeden once dogrular.

    Args:
        submission: Uretilen submission.
        sample: ``sample_submission.csv``. Verilirse kolon adlari, satir sayisi
            ve ID kumesi ONA GORE kontrol edilir -- en guvenilir dogrulama budur.
        allow_negative: Negatif tahminlere izin ver. Kesinti suresi, tuketim,
            abone sayisi gibi fiziksel buyuklukler icin ``False`` birak.

    Returns:
        ``SubmissionCheck``.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if sample is not None:
        id_column = sample.columns[0]
        target_column = sample.columns[-1]

        missing_columns = [column for column in sample.columns if column not in submission.columns]
        if missing_columns:
            errors.append(
                f"Eksik kolon(lar): {missing_columns}. "
                f"Beklenen kolonlar: {list(sample.columns)}"
            )

        if len(submission) != len(sample):
            errors.append(
                f"Satir sayisi uyusmuyor: submission={len(submission)}, "
                f"beklenen={len(sample)}"
            )

        if id_column in submission.columns and id_column in sample.columns:
            submission_ids = set(submission[id_column])
            sample_ids = set(sample[id_column])
            missing_ids = sample_ids - submission_ids
            extra_ids = submission_ids - sample_ids
            if missing_ids:
                errors.append(f"{len(missing_ids)} ID eksik. Ornek: {list(missing_ids)[:5]}")
            if extra_ids:
                errors.append(f"{len(extra_ids)} fazladan ID var. Ornek: {list(extra_ids)[:5]}")

    if expected_rows is not None and len(submission) != expected_rows:
        errors.append(f"Satir sayisi {len(submission)}, beklenen {expected_rows}")

    stats: dict[str, float] = {}

    if target_column in submission.columns:
        predictions = pd.to_numeric(submission[target_column], errors="coerce")

        nan_count = int(predictions.isna().sum())
        if nan_count:
            errors.append(
                f"{nan_count} NaN/sayisal olmayan tahmin var. Kaggle bunu reddeder."
            )

        infinite_count = int(np.isinf(predictions.dropna()).sum())
        if infinite_count:
            errors.append(f"{infinite_count} sonsuz (inf) tahmin var.")

        finite = predictions.replace([np.inf, -np.inf], np.nan).dropna()
        if len(finite):
            stats = {
                "min": float(finite.min()),
                "max": float(finite.max()),
                "ortalama": float(finite.mean()),
                "std": float(finite.std()),
                "benzersiz": float(finite.nunique()),
            }

            if not allow_negative and (finite < 0).any():
                negative_count = int((finite < 0).sum())
                warnings.append(
                    f"{negative_count} negatif tahmin var. Fiziksel buyuklukler "
                    "(sure, tuketim) negatif olamaz -- np.clip(pred, 0, None) uygula."
                )

            if finite.nunique() == 1:
                warnings.append(
                    "Tum tahminler AYNI deger. Model ogrenmemis veya yanlis "
                    "kolon yazilmis olabilir."
                )

            if float(finite.std()) == 0.0:
                warnings.append("Tahminlerin standart sapmasi 0.")
    else:
        errors.append(f"Hedef kolon '{target_column}' submission icinde yok.")

    return SubmissionCheck(
        is_valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        stats=stats,
    )


def write_submission(
    ids: np.ndarray | pd.Series,
    predictions: np.ndarray,
    path: str | Path,
    *,
    sample: pd.DataFrame | None = None,
    id_column: str = "ID",
    target_column: str = "hedef",
    clip_negative: bool = True,
    float_format: str = "%.6f",
    validate: bool = True,
) -> Path:
    """Submission dosyasini dogrulayarak yazar ve yolunu dondurur.

    ``float_format`` bilerek acikca veriliyor: pandas varsayilani buyuk/kucuk
    sayilarda bilimsel gosterime gecebilir (``1.2e-05``) ve bazi Kaggle
    ayristiricilari bunu reddeder.

    Raises:
        ValueError: Dogrulama basarisizsa (``validate=True`` iken).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    raw_values = np.asarray(predictions, dtype="float64").ravel()

    if sample is not None:
        id_column = sample.columns[0]
        target_column = sample.columns[-1]

    identifiers = np.asarray(ids).ravel()

    # DOGRULAMA KIRPMADAN ONCE yapilir. Once kirpip sonra dogrulamak,
    # "negatif tahmin var" uyarisini ASLA tetiklenemez hale getirir -- yani
    # ters log donusumundeki bir isaret hatasi veya olcek hatasi sessizce
    # 0.0'a bastirilir ve iz birakmaz.
    if validate:
        check = validate_submission(
            pd.DataFrame({id_column: identifiers, target_column: raw_values}),
            sample=sample,
            id_column=id_column,
            target_column=target_column,
            allow_negative=not clip_negative,
        )
        print(check)
        check.raise_if_invalid()

    values = raw_values
    if clip_negative:
        negative = raw_values < 0
        negative_count = int(negative.sum())
        if negative_count:
            worst = float(raw_values[negative].min())
            share = negative_count / len(raw_values) * 100
            print(
                f"  KIRPILDI: {negative_count:,} tahmin (%{share:.2f}) "
                f"negatifti ve 0'a cekildi. En kucuk deger: {worst:.4f}. "
                "Bu oran yuksekse model olcegi veya ters donusum hatalidir -- "
                "kirpma bir cozum degil, bir semptom ortbasidir."
            )
        values = np.clip(raw_values, 0, None)

    submission = pd.DataFrame({id_column: identifiers, target_column: values})

    submission.to_csv(path, index=False, float_format=float_format, encoding="utf-8")
    print(f"Yazildi: {path}  ({len(submission)} satir)")
    return path


def blend_submissions(
    paths: list[str | Path],
    weights: list[float] | None = None,
    *,
    method: str = "mean",
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Birden fazla submission'i harmanlar.

    Args:
        method: ``mean`` (deger ortalamasi) veya ``rank`` (sira ortalamasi).
            AUC ile puanlanan yarismalarda ``rank`` genellikle daha iyidir:
            modellerin olasilik olcekleri farkli olsa bile siralamalari
            birlestirilebilir. RMSE gibi deger metriklerinde ``mean`` kullan.

    Returns:
        Harmanlanmis submission DataFrame'i.
    """
    frames = [pd.read_csv(path) for path in paths]
    if not frames:
        raise ValueError("Harmanlanacak dosya yok.")

    id_column, target_column = frames[0].columns[0], frames[0].columns[-1]

    reference_ids = frames[0][id_column].to_numpy()
    for index, frame in enumerate(frames[1:], start=1):
        if not np.array_equal(frame[id_column].to_numpy(), reference_ids):
            raise ValueError(
                f"{paths[index]}: ID sirasi ilk dosyayla ayni degil. "
                "Harmanlamadan once ID'ye gore sirala."
            )

    if weights is None:
        weights = [1.0 / len(frames)] * len(frames)
    if len(weights) != len(frames):
        raise ValueError(
            f"Agirlik sayisi ({len(weights)}) dosya sayisiyla ({len(frames)}) uyusmuyor."
        )

    total = sum(weights)
    normalized = [weight / total for weight in weights]

    if method == "rank":
        columns = [frame[target_column].rank(pct=True).to_numpy() for frame in frames]
    elif method == "mean":
        columns = [frame[target_column].to_numpy() for frame in frames]
    else:
        raise ValueError(f"Bilinmeyen yontem '{method}'. 'mean' veya 'rank' kullan.")

    blended = sum(
        weight * column for weight, column in zip(normalized, columns, strict=True)
    )
    result = pd.DataFrame({id_column: reference_ids, target_column: blended})

    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False, float_format="%.6f", encoding="utf-8")
        shown = [round(weight, 3) for weight in normalized]
        print(f"Harmanlandi -> {output}  ({method}, agirliklar={shown})")

    return result
