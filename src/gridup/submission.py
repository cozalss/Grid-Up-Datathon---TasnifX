"""Submission dosyasi uretimi ve dogrulamasi.

Yarismalarda kaybedilen puanlarin sasirtici bir kismi MODELDEN DEGIL, bozuk
submission dosyasindan gelir: eksik satir, yanlis kolon adi, NaN, yanlis sira,
bilimsel gosterim, negatif tahmin.

Bu modul her submission'i yazmadan ONCE dogrular ve sorunu acikca soyler.
Kaggle'in "Submission Scoring Error" mesaji sana hicbir sey soylemez.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .io_utils import atomic_write_bytes

__all__ = ["SubmissionCheck", "validate_submission", "write_submission", "blend_submissions"]

_MISSING_ID = ("__gridup_missing_id__",)


def _id_key(value: object) -> object:
    """NaN/NA kimliklerini de cokluk ve sira karsilastirmasinda esit sayar."""
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and missing:
        return _MISSING_ID
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _id_keys(values: pd.Series | np.ndarray) -> list[object]:
    return [_id_key(value) for value in np.asarray(values, dtype=object).ravel()]


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
            ve ID sirasi/coklugu ONA GORE kontrol edilir -- en guvenilir
            dogrulama budur.
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

        sample_columns = list(sample.columns)
        submission_columns = list(submission.columns)
        missing_columns = [column for column in sample_columns if column not in submission_columns]
        extra_columns = [column for column in submission_columns if column not in sample_columns]
        if missing_columns:
            errors.append(
                f"Eksik kolon(lar): {missing_columns}. Beklenen kolonlar: {sample_columns}"
            )
        if extra_columns:
            errors.append(
                f"Fazladan kolon(lar): {extra_columns}. Beklenen kolonlar: {sample_columns}"
            )
        if not missing_columns and not extra_columns and submission_columns != sample_columns:
            errors.append(
                "Kolon sirasi sample submission ile birebir uyusmuyor: "
                f"submission={submission_columns}, beklenen={sample_columns}."
            )

        if len(submission) != len(sample):
            errors.append(
                f"Satir sayisi uyusmuyor: submission={len(submission)}, beklenen={len(sample)}"
            )

        if id_column in submission.columns and id_column in sample.columns:
            submission_ids = _id_keys(submission[id_column])
            sample_ids = _id_keys(sample[id_column])
            submission_counts = Counter(submission_ids)
            sample_counts = Counter(sample_ids)
            missing_ids = sample_counts - submission_counts
            extra_ids = submission_counts - sample_counts
            if missing_ids or extra_ids:
                missing_count = sum(missing_ids.values())
                extra_count = sum(extra_ids.values())
                errors.append(
                    "ID cokluk uyusmazligi: "
                    f"{missing_count} ID eksik (ornek: {list(missing_ids)[:5]}), "
                    f"{extra_count} fazladan ID var (ornek: {list(extra_ids)[:5]})."
                )
            elif submission_ids != sample_ids:
                errors.append(
                    "ID sirasi sample submission ile birebir uyusmuyor. "
                    "Tahmin-ID ciftlerini koruyarak acik align_to_sample=True kullan."
                )

    if expected_rows is not None and len(submission) != expected_rows:
        errors.append(f"Satir sayisi {len(submission)}, beklenen {expected_rows}")

    stats: dict[str, float] = {}

    if target_column in submission.columns:
        predictions = pd.to_numeric(submission[target_column], errors="coerce")

        nan_count = int(predictions.isna().sum())
        if nan_count:
            errors.append(f"{nan_count} NaN/sayisal olmayan tahmin var. Kaggle bunu reddeder.")

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


def _orijinal_basliklar(
    sample: pd.DataFrame | None, id_column: str, target_column: str
) -> tuple[str, str]:
    """Ornek submission'daki HAM kolon adlarini geri getirir.

    NEDEN KRITIK
    ------------
    ``read_any`` kolon adlarini normalize eder: ``"Dagitilan Enerji (MWh)"``
    -> ``"dagitilan_enerji_mwh"``. Bu ic isleyis icin dogrudur (Turkce
    karakter ve bosluk her katmanda risk tasir) ama **submission dosyasi
    Kaggle'a gider ve orada kolon adi BIREBIR eslesmek zorundadir.**

    2023 GDZ yarismasinda hedef kolonu tam olarak ``Dagitilan Enerji (MWh)``
    olmak zorundaydi. OLCULDU: bu koruma olmadan yazilan dosyanin basligi
    ``id,dagitilan_enerji_mwh`` cikiyordu -- Kaggle REDDEDER, ama bizim
    dogrulamamiz "GECERLI" diyordu.

    ``read_any`` esleme sozlugunu ``frame.attrs["original_columns"]`` icinde
    saklar; buradan tersine cevirip kullaniyoruz. Ornek verilmemisse veya
    esleme yoksa adlar OLDUGU GIBI kalir.
    """
    if sample is None:
        return id_column, target_column
    esleme = sample.attrs.get("original_columns")
    if not isinstance(esleme, dict) or not esleme:
        return id_column, target_column
    # {ham: normalize} -> {normalize: ham}
    ters = {normalize: ham for ham, normalize in esleme.items()}
    return ters.get(id_column, id_column), ters.get(target_column, target_column)


def _iki_kolon_sartini_dogrula(sample: pd.DataFrame | None) -> None:
    """Ornek submission 2 kolondan genisse ACIK hata firlatir.

    NEDEN AYRI BIR KAPI
    -------------------
    ``write_submission`` yapisal olarak ``{id_column: ids, target_column:
    values}`` yani TAM IKI kolonluk bir frame kurar; ortadaki kolonlari
    uretebilecegi bir girdisi yoktur. 3+ kolonlu bir ``sample_submission``
    (bilesik anahtar veya coklu hedef) geldiginde iki kotu yol vardi:

      * ``validate=True``  -> "Eksik kolon(lar): ['Ilce']" -- kullaniciyi
        KENDI hatasini aramaya yollayan yaniltici mesaj, ustelik pipeline'in
        EN SON adiminda.
      * ``validate=False`` -> OLCULDU: 3 kolonlu ornege karsi diske 2 kolonlu
        dosya yazildi ve hicbir uyari cikmadi. Kaggle bunu reddeder, biz de
        neden reddettigini goremeyiz.

    Bu yuzden kapi dogrulamadan ONCE ve ``validate`` bayragindan BAGIMSIZ
    calisir: eksik olan sey kullanicinin verisi degil, bu fonksiyonun
    yetenegidir.
    """
    if sample is None or len(sample.columns) <= 2:
        return
    raise ValueError(
        f"Ornek submission {len(sample.columns)} kolonlu: {list(sample.columns)}. "
        "write_submission yalnizca (id, hedef) ikilisi uretebilir -- ortadaki "
        "kolonlari uretecek bir girdisi yok.\n"
        "Bilesik anahtar veya coklu hedef varsa frame'i KENDIN kur ve "
        "dogrulamayi ayrica cagir:\n"
        "    gonderim = sample[list(sample.columns[:-1])].assign(**{sample.columns[-1]: tahmin})\n"
        "    validate_submission(gonderim, sample=sample).raise_if_invalid()\n"
        "    gonderim.to_csv(yol, index=False, float_format='%.6f', encoding='utf-8')"
    )


def _sample_sirasina_hizala(
    identifiers: np.ndarray,
    predictions: np.ndarray,
    sample: pd.DataFrame,
    id_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Tahmin-ID ciftini bozmadan sample sirasina getirir.

    Tekrarlanan ID'lerde hangi tahminin hangi ornek satirina ait oldugu ek bir
    anahtar olmadan bilinemez. Bu durumda tahmin uydurmak yerine kapali hata
    veririz.
    """
    if len(identifiers) != len(predictions):
        raise ValueError(
            f"ID ({len(identifiers)}) ve tahmin ({len(predictions)}) uzunluklari farkli."
        )
    sample_ids = np.asarray(sample[id_column]).ravel()
    input_keys = _id_keys(identifiers)
    sample_keys = _id_keys(sample_ids)
    if len(set(input_keys)) != len(input_keys) or len(set(sample_keys)) != len(sample_keys):
        raise ValueError(
            "align_to_sample icin ID'ler benzersiz olmali; tekrar eden ID'ler "
            "tahmin eslesmesini belirsiz yapar."
        )
    input_counts = Counter(input_keys)
    sample_counts = Counter(sample_keys)
    if input_counts != sample_counts:
        missing = sample_counts - input_counts
        extra = input_counts - sample_counts
        raise ValueError(
            "align_to_sample ID cokluk uyusmazligi: "
            f"{sum(missing.values())} eksik, {sum(extra.values())} fazladan."
        )
    prediction_by_id = dict(zip(input_keys, predictions, strict=True))
    aligned = np.asarray([prediction_by_id[key] for key in sample_keys], dtype="float64")
    return sample_ids, aligned


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
    align_to_sample: bool = False,
    original_header: bool = True,
) -> Path:
    """Submission dosyasini dogrulayarak yazar ve yolunu dondurur.

    ``float_format`` bilerek acikca veriliyor: pandas varsayilani buyuk/kucuk
    sayilarda bilimsel gosterime gecebilir (``1.2e-05``) ve bazi Kaggle
    ayristiricilari bunu reddeder.

    Raises:
        ValueError: Ornek submission 2 kolondan genisse (``validate``
            bayragindan bagimsiz), guvenli hizalama mumkun degilse veya
            dogrulama basarisizsa.
    """
    _iki_kolon_sartini_dogrula(sample)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    raw_values = np.asarray(predictions, dtype="float64").ravel()

    if sample is not None:
        id_column = sample.columns[0]
        target_column = sample.columns[-1]

    identifiers = np.asarray(ids).ravel()
    if align_to_sample:
        if sample is None:
            raise ValueError("align_to_sample=True icin sample submission zorunlu.")
        identifiers, raw_values = _sample_sirasina_hizala(
            identifiers, raw_values, sample, id_column
        )

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
        # ravel(): np.clip 'n boyutlu' tip dondurur; raw_values 1 boyutlu
        # oldugu icin yeniden atama eski numpy stub'larinda tip catismasi
        # yaratir. ravel zaten 1B olan diziyi degistirmez, tipi netlestirir.
        values = np.clip(raw_values, 0, None).ravel()

    submission = pd.DataFrame({id_column: identifiers, target_column: values})

    # Baslik YALNIZCA yazma aninda orijinale cevrilir -- tum dogrulama ve
    # hesaplama normalize adlarla yapildi, degistirmiyoruz.
    if original_header:
        ham_id, ham_hedef = _orijinal_basliklar(sample, id_column, target_column)
        if (ham_id, ham_hedef) != (id_column, target_column):
            submission = submission.rename(columns={id_column: ham_id, target_column: ham_hedef})
            print(f"  Baslik orijinale cevrildi: {ham_id!r}, {ham_hedef!r}")

    csv_content = submission.to_csv(index=False, float_format=float_format).encode("utf-8")
    atomic_write_bytes(path, csv_content)
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

    blended = sum(weight * column for weight, column in zip(normalized, columns, strict=True))
    result = pd.DataFrame({id_column: reference_ids, target_column: blended})

    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        csv_content = result.to_csv(index=False, float_format="%.6f").encode("utf-8")
        atomic_write_bytes(output, csv_content)
        shown = [round(weight, 3) for weight in normalized]
        print(f"Harmanlandi -> {output}  ({method}, agirliklar={shown})")

    return result
