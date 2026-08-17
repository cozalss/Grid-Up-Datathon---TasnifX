"""Otomatik veri profilleme: veri geldigi ilk 10 dakikada calistirilacak modul.

AMAC: ``profile(train, test)`` cagirdiginda, elle 2 saat surecek kesif
calismasinin yerini alan tek bir metin raporu uretmek.

Rapor su sorulari cevaplar:
  * Kolonlar hangi tipte, kac tanesi kategorik/sayisal/tarih?
  * Nerede eksik veri var ve orani ne?
  * Hedef nasil dagilmis (carpik mi, dengesiz mi, sifir yigilmasi var mi)?
  * Hangi kolonlar sabit, hangileri neredeyse benzersiz (ID)?
  * Train ve test kolonlari uyusuyor mu?
  * Kategorik kardinaliteler yonetilebilir mi?
  * Zaman ekseni var mi, train/test nasil bolunmus?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .compat import is_categorical_like
from .turkish import has_combining_dot
from .validation import _detect_time_columns

__all__ = ["ColumnProfile", "DatasetProfile", "profile_columns", "profile", "quick_look"]

# Bir kolonun ID sayilmasi icin gereken benzersizlik orani.
_ID_UNIQUE_RATIO = 0.98
# Kategorik sayilmasi icin ust kardinalite siniri (bunun ustu "yuksek kardinalite").
_HIGH_CARDINALITY = 100
# TEKRARLI MONOTON isareti icin alt sinirlar. Kucuk frame'de veya iki-uc
# degerli bir kolonda monotonluk tesadufen olusur; kanit sayilmaz.
_MONOTON_MIN_SATIR = 20
_MONOTON_MIN_BENZERSIZ = 3


@dataclass(frozen=True)
class ColumnProfile:
    """Tek bir kolonun profili."""

    name: str
    dtype: str
    kind: str  # sayisal | kategorik | tarih | metin | bos
    missing_count: int
    missing_ratio: float
    unique_count: int
    unique_ratio: float
    sample_values: tuple[Any, ...]
    stats: dict[str, float] = field(default_factory=dict)
    flags: tuple[str, ...] = ()


def _classify(series: pd.Series) -> str:
    if series.isna().all():
        return "bos"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "tarih"
    if pd.api.types.is_bool_dtype(series):
        return "kategorik"
    if pd.api.types.is_numeric_dtype(series):
        return "sayisal"
    if isinstance(series.dtype, pd.CategoricalDtype):
        return "kategorik"
    if not is_categorical_like(series):
        return "metin"
    unique_ratio = series.nunique(dropna=True) / max(series.notna().sum(), 1)
    return "metin" if unique_ratio > 0.5 else "kategorik"


def _tamsayi_gibi(series: pd.Series) -> bool:
    """Kolon TAMSAYI kimligi tasiyor mu (int dtype veya tamsayi degerli float)?

    ``1.0, 2.0, 3.0`` seklinde okunmus bir ID kolonu (CSV'de NaN varsa pandas
    onu float yapar) hala ID'dir; ``18.4, 21.7`` gibi surekli bir olcum
    degildir. Bool disarida: ``True/False`` ne kimlik ne sayactir.
    """
    if pd.api.types.is_bool_dtype(series):
        return False
    if pd.api.types.is_integer_dtype(series):
        return True
    if not pd.api.types.is_float_dtype(series):
        return False
    finite = series.replace([np.inf, -np.inf], np.nan).dropna()
    return bool(len(finite)) and bool((finite == finite.round()).all())


def _id_benzeri_mi(series: pd.Series, kind: str, unique: int, row_count: int) -> bool:
    """ID-BENZERI isaretinin kosulu: benzersiz VE kimlik tasiyabilen bir tip.

    NEDEN TIP KAPISI EKLENDI (olculdu)
    ----------------------------------
    Eski kosul yalnizca benzersizlik oranina bakiyordu. Surekli bir float
    kolonu tanim geregi neredeyse %100 benzersizdir, yani her sicaklik/nem
    olcumu ve cogu regresyon HEDEFI "ID" isaretlenir::

        96 ilce x 240 gun paneli, isaret alanlar:
          ONCE : ['kayit_no', 'sicaklik', 'nem', 'kesinti']   <- 3 yanlis pozitif
          SONRA: ['kayit_no']

    ``kesinti`` hedefin kendisiydi. Uc yanlis pozitif arasinda kaybolan bir
    uyari, uyari degildir.
    """
    if unique / max(row_count, 1) < _ID_UNIQUE_RATIO:
        return False
    if kind in {"kategorik", "metin"}:
        return True
    return kind == "sayisal" and _tamsayi_gibi(series)


def _tekrarli_monoton_mu(series: pd.Series, kind: str, unique: int, row_count: int) -> bool:
    """Satir sirasiyla artan, TEKRARLI bir tamsayi sayaci mi?

    NEDEN AYRI BIR ISARET (olculdu)
    -------------------------------
    Gercek zaman sizintisi tasiyan kolon genellikle benzersiz DEGILDIR:
    ``parti_no = gun_no // 7`` gibi bir yukleme sayaci 23.040 satirda yalnizca
    35 farkli deger alir, ama satir sirasiyla monoton artar ve train/holdout
    araliklari neredeyse hic ortusmez (train [0, 28] / holdout [28, 34]).
    Yalnizca benzersizlige bakan ID-BENZERI kapisi bunu TAMAMEN kaciriyordu::

        ONCE : parti_no icin uretilen isaret sayisi = 0
        SONRA: 'TEKRARLI MONOTON' isareti uretiliyor
    """
    if kind != "sayisal" or row_count < _MONOTON_MIN_SATIR:
        return False
    if unique < _MONOTON_MIN_BENZERSIZ or unique / max(row_count, 1) >= _ID_UNIQUE_RATIO:
        return False
    if not _tamsayi_gibi(series):
        return False
    return bool(series.is_monotonic_increasing or series.is_monotonic_decreasing)


def _column_flags(series: pd.Series, kind: str, row_count: int) -> tuple[str, ...]:
    """Kolon hakkinda dikkat edilmesi gereken isaretler."""
    flags: list[str] = []
    unique = series.nunique(dropna=False)
    missing_ratio = float(series.isna().mean())

    if unique <= 1:
        flags.append("SABIT -- bilgi tasimaz, cikar")
    if _id_benzeri_mi(series, kind, unique, row_count):
        flags.append("ID-BENZERI -- feature yapma, sirali ise zaman sizdirir")
    if _tekrarli_monoton_mu(series, kind, unique, row_count):
        flags.append(
            "TEKRARLI MONOTON -- satir sirasiyla artan sayac (parti/yukleme no); "
            "benzersiz olmadigi icin ID kapisina takilmaz ama ZAMAN SIZDIRIR"
        )
    if missing_ratio > 0.5:
        flags.append(f"COK EKSIK (%{missing_ratio * 100:.0f})")
    if kind == "kategorik" and unique > _HIGH_CARDINALITY:
        flags.append(f"YUKSEK KARDINALITE ({unique}) -- hedef/frekans kodlama gerekli")

    if kind == "sayisal":
        finite = series.replace([np.inf, -np.inf], np.nan).dropna()
        if len(finite) > 10:
            skewness = float(finite.skew())
            if abs(skewness) > 3:
                flags.append(f"COK CARPIK (skew={skewness:.1f}) -- log1p dusun")
            if (finite == 0).mean() > 0.5:
                flags.append(f"SIFIR YIGILMASI (%{(finite == 0).mean() * 100:.0f})")
            if (finite < 0).any() and (finite >= 0).mean() > 0.95:
                flags.append("Az sayida NEGATIF -- veri hatasi olabilir")

    if kind in {"kategorik", "metin"}:
        sample = series.dropna().astype(str).head(500)
        if any(has_combining_dot(value) for value in sample):
            flags.append("BIRLESIK NOKTA (U+0307) -- yanlis .lower() kullanilmis, join'ler bozulur")

    return tuple(flags)


def profile_columns(frame: pd.DataFrame) -> list[ColumnProfile]:
    """Her kolon icin profil uretir."""
    row_count = len(frame)
    profiles: list[ColumnProfile] = []

    for name in frame.columns:
        series = frame[name]
        kind = _classify(series)
        unique = int(series.nunique(dropna=False))

        stats: dict[str, float] = {}
        if kind == "sayisal":
            finite = series.replace([np.inf, -np.inf], np.nan).dropna()
            if len(finite):
                stats = {
                    "min": float(finite.min()),
                    "p25": float(finite.quantile(0.25)),
                    "medyan": float(finite.median()),
                    "p75": float(finite.quantile(0.75)),
                    "max": float(finite.max()),
                    "ortalama": float(finite.mean()),
                    "std": float(finite.std()),
                    "carpiklik": float(finite.skew()),
                }
        elif kind == "tarih":
            valid = pd.to_datetime(series, errors="coerce").dropna()
            if len(valid):
                stats = {"gun_araligi": float((valid.max() - valid.min()).days)}

        profiles.append(
            ColumnProfile(
                name=name,
                dtype=str(series.dtype),
                kind=kind,
                missing_count=int(series.isna().sum()),
                missing_ratio=float(series.isna().mean()),
                unique_count=unique,
                unique_ratio=unique / max(row_count, 1),
                sample_values=tuple(series.dropna().head(3).tolist()),
                stats=stats,
                flags=_column_flags(series, kind, row_count),
            )
        )

    return profiles


@dataclass
class DatasetProfile:
    """Veri setinin tam profili."""

    train_shape: tuple[int, int]
    test_shape: tuple[int, int] | None
    columns: list[ColumnProfile]
    target_summary: dict[str, Any]
    schema_diff: dict[str, list[str]]
    time_columns: list[str]
    memory_mb: float

    def flagged(self) -> list[ColumnProfile]:
        """Yalnizca isaretlenmis kolonlar -- once bunlara bak."""
        return [column for column in self.columns if column.flags]

    def by_kind(self, kind: str) -> list[str]:
        return [column.name for column in self.columns if column.kind == kind]

    def report(self) -> str:
        """Insan tarafindan okunacak tam rapor."""
        lines: list[str] = []
        add = lines.append

        add("=" * 78)
        add("VERI PROFILI")
        add("=" * 78)
        add(f"Train: {self.train_shape[0]:,} satir x {self.train_shape[1]} kolon")
        if self.test_shape:
            add(f"Test:  {self.test_shape[0]:,} satir x {self.test_shape[1]} kolon")
            ratio = self.test_shape[0] / max(self.train_shape[0], 1)
            add(f"Test/Train orani: {ratio:.2f}")
        add(f"Bellek (train): {self.memory_mb:.1f} MB")

        add("\n--- KOLON TIPLERI ---")
        for kind in ("sayisal", "kategorik", "tarih", "metin", "bos"):
            names = self.by_kind(kind)
            if names:
                shown = ", ".join(names[:12])
                more = f" (+{len(names) - 12} tane)" if len(names) > 12 else ""
                add(f"{kind:<12} ({len(names):>3}): {shown}{more}")

        if self.schema_diff.get("train_only") or self.schema_diff.get("test_only"):
            add("\n--- SEMA FARKI ---")
            if self.schema_diff.get("train_only"):
                add(f"Yalniz train'de: {self.schema_diff['train_only']}")
                add("  -> Bunlar tahmin aninda YOK. Hedef ve ID disindakiler sizintidir.")
            if self.schema_diff.get("test_only"):
                add(f"Yalniz test'te:  {self.schema_diff['test_only']}")

        if self.time_columns:
            add(f"\n--- ZAMAN KOLONLARI --- {self.time_columns}")
            add("  -> Rastgele KFold GELECEGI SIZDIRIR. validation.suggest_scheme() calistir.")

        flagged = self.flagged()
        if flagged:
            add(f"\n--- ISARETLENMIS KOLONLAR ({len(flagged)}) ---")
            for column in flagged:
                add(f"\n{column.name}  [{column.kind}, {column.dtype}]")
                add(
                    f"  eksik=%{column.missing_ratio * 100:.1f}  "
                    f"benzersiz={column.unique_count:,} (%{column.unique_ratio * 100:.1f})"
                )
                for flag in column.flags:
                    add(f"  ! {flag}")

        if self.target_summary:
            add("\n--- HEDEF ---")
            for key, value in self.target_summary.items():
                add(f"  {key}: {value}")

        add("\n" + "=" * 78)
        return "\n".join(lines)


def _summarize_target(series: pd.Series) -> dict[str, Any]:
    """Hedef degiskeni ozetler ve gorev tipini tahmin eder."""
    summary: dict[str, Any] = {
        "ad": series.name,
        "dtype": str(series.dtype),
        "eksik": int(series.isna().sum()),
        "benzersiz": int(series.nunique(dropna=True)),
    }

    unique = summary["benzersiz"]

    if pd.api.types.is_numeric_dtype(series) and unique > 20:
        finite = series.replace([np.inf, -np.inf], np.nan).dropna()
        summary["gorev_tahmini"] = "regression"
        summary["min"] = float(finite.min())
        summary["max"] = float(finite.max())
        summary["ortalama"] = round(float(finite.mean()), 4)
        summary["medyan"] = round(float(finite.median()), 4)
        skewness = float(finite.skew())
        summary["carpiklik"] = round(skewness, 3)
        summary["sifir_orani"] = round(float((finite == 0).mean()), 4)

        if abs(skewness) > 2:
            summary["oneri"] = (
                "Carpiklik yuksek -> log1p donusumu dene (metrics.log_transform_target). "
                "Metrik RMSLE ise bu ZORUNLU."
            )
        if float((finite == 0).mean()) > 0.4:
            summary["oneri_2"] = (
                "Sifir yigilmasi yuksek -> iki asamali model dusun: "
                "once 'sifir mi degil mi' siniflandirmasi, sonra pozitiflerde regresyon."
            )
    else:
        summary["gorev_tahmini"] = "binary" if unique == 2 else "multiclass"
        distribution = series.value_counts(normalize=True, dropna=True)
        summary["sinif_dagilimi"] = {
            str(key): round(float(value), 4) for key, value in distribution.head(10).items()
        }
        if len(distribution) and float(distribution.min()) < 0.05:
            summary["oneri"] = (
                f"Dengesiz (en nadir sinif %{distribution.min() * 100:.2f}) -> "
                "StratifiedKFold + esik optimizasyonu (metrics.optimize_threshold) sart. "
                "0.5 esigi neredeyse kesin yanlis."
            )

    return summary


def profile(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    *,
    target: str | None = None,
) -> DatasetProfile:
    """Veri setinin tam profilini cikarir.

    Veri geldiginde CALISTIRILACAK ILK FONKSIYON budur::

        from gridup.profiling import profile
        print(profile(train, test, target="hedef").report())
    """
    columns = profile_columns(train)

    schema_diff: dict[str, list[str]] = {}
    if test is not None:
        schema_diff = {
            "train_only": [c for c in train.columns if c not in test.columns],
            "test_only": [c for c in test.columns if c not in train.columns],
        }

    # ZAMAN KOLONU TESPITI TEK KAYNAKTAN.
    #
    # Onceki surum yalnizca dtype siniflandirmasina bakiyordu
    # (``column.kind == "tarih"``) ve METIN olarak saklanmis tarihleri
    # KACIRIYORDU. validation._detect_time_columns ise onlari buluyordu --
    # yani iki ayri "hangi kolon zamandir" cevabi vardi ve birbirini
    # tutmuyordu.
    #
    # OLCULDU: ISO bicimli string tarih kolonunda
    #   profiling.time_columns       = []
    #   _detect_time_columns()       = ['tarih']
    # day_one.py:151 birincisine baktigi icin time_column=None kaliyor,
    # ekranda "TimeSeriesSplit oneriyorum" yazarken CV rastgele boluyor --
    # sessiz bir ZAMAN SIZINTISI.
    #
    # Not: dtype siniflandirmasina (column.kind) DOKUNMUYORUZ; onu
    # degistirmek frekans kodlama aday listesini de degistirirdi.
    dtype_tabanli = [column.name for column in columns if column.kind == "tarih"]
    tespit = _detect_time_columns(train)
    # Sirayi koru, tekrarlari at.
    time_columns = list(dict.fromkeys(dtype_tabanli + tespit))

    target_summary: dict[str, Any] = {}
    if target:
        # Yanlis yazilmis hedef adini SESSIZCE atlamayiz: rapordan "--- HEDEF ---"
        # bolumu kaybolur, kullanici raporun eksiksiz oldugunu sanir ve
        # carpiklik/dengesizlik uyarilarini hic gormez.
        if target not in train.columns:
            raise KeyError(
                f"Hedef kolon '{target}' train icinde yok. "
                f"Mevcut kolonlar: {list(train.columns)[:20]}"
            )
        target_summary = _summarize_target(train[target])

    return DatasetProfile(
        train_shape=train.shape,
        test_shape=test.shape if test is not None else None,
        columns=columns,
        target_summary=target_summary,
        schema_diff=schema_diff,
        time_columns=time_columns,
        memory_mb=float(train.memory_usage(deep=True).sum() / 1024**2),
    )


def quick_look(frame: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Kolon bazli kompakt ozet tablosu -- notebook'ta hizli bakis icin."""
    rows = []
    for column in profile_columns(frame):
        rows.append(
            {
                "kolon": column.name,
                "tip": column.kind,
                "dtype": column.dtype,
                "eksik_%": round(column.missing_ratio * 100, 2),
                "benzersiz": column.unique_count,
                "ornek": str(column.sample_values[:n])[:60],
                "isaret": " | ".join(column.flags)[:80],
            }
        )
    return pd.DataFrame(rows)
