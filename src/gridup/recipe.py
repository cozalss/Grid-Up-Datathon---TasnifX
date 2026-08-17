"""Immutable, serializable pipeline recipes.

Every executable entry point should resolve its behaviour from the same recipe
instead of repeating fold, horizon, lag and model constants in scripts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "CVRecipe",
    "ExecutionPolicy",
    "FeatureRecipe",
    "ModelRecipe",
    "PipelineRecipe",
    "default_pipeline_recipe",
]


@dataclass(frozen=True)
class CVRecipe:
    """CV semasinin KAYDI -- fold URETMEZ, yalnizca ne kosuldugunu belgeler.

    ``embargo_days`` varsayilani 0'dir ve bu bir sema onerisi DEGILDIR; sifir
    ambargo purge'un amacini ortadan kaldirir. Gercek bir kosuyu kaydederken bu
    alan ACIKCA doldurulmalidir, aksi halde provenance kaydi gercekten kosandan
    baska bir semayi belgeler. ``purged_time_series_split`` uretim tarafinda
    ambargoyu zaten varsayilansiz zorunlu tutar; buradaki 0 yalnizca "henuz
    yapilandirilmamis sablon" anlamina gelir.

    Kural: ``embargo_days >= en uzun kayan pencere``. Gercek cagri yerleri
    (day_one, full_pipeline, smoke_test, build_notebooks) bu alani acikca yazar
    ve ``tests/test_recipe_and_provenance_contract`` bunu kilitler.
    """

    n_splits: int = 4
    splitter: str = "purged_time_series"
    test_span_days: int | None = None
    embargo_days: int = 0

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ValueError("n_splits en az 2 olmali")
        if self.test_span_days is not None and self.test_span_days < 1:
            raise ValueError("test_span_days pozitif olmali")
        if self.embargo_days < 0:
            raise ValueError("embargo_days negatif olamaz")


@dataclass(frozen=True)
class FeatureRecipe:
    """Feature availability contract.

    ``target_shifts`` are absolute row offsets. A target-derived value cannot
    be newer than the forecast origin, therefore every shift must be at least
    ``horizon``.
    """

    horizon: int = 1
    target_shifts: tuple[int, ...] = (1, 7, 14, 28)
    rolling_windows: tuple[int, ...] = (7, 28)
    history_value_columns: tuple[str, ...] = ()
    history_shifts: tuple[int, ...] = ()
    history_rolling_windows: tuple[int, ...] = ()
    history_rolling_aggregations: tuple[str, ...] = ()
    families: tuple[str, ...] = ("calendar", "holiday", "frequency", "lag")

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon en az 1 olmali")
        invalid = [shift for shift in self.target_shifts if shift < self.horizon]
        if invalid:
            raise ValueError(
                f"target shift degerleri horizon ({self.horizon}) altinda olamaz: {invalid}"
            )
        if any(shift < 1 for shift in self.target_shifts):
            raise ValueError("target shift degerleri pozitif olmali")
        if any(window < 1 for window in self.rolling_windows):
            raise ValueError("rolling window degerleri pozitif olmali")
        if any(shift < 1 for shift in self.history_shifts):
            raise ValueError("history shift degerleri pozitif olmali")
        if any(window < 1 for window in self.history_rolling_windows):
            raise ValueError("history rolling window degerleri pozitif olmali")
        if self.history_shifts and not self.history_value_columns:
            raise ValueError("history shift icin history_value_columns zorunludur")
        if self.history_rolling_windows and not self.history_value_columns:
            raise ValueError("history rolling icin history_value_columns zorunludur")


@dataclass(frozen=True)
class ModelRecipe:
    kind: str = "lightgbm"
    objective: str | None = None
    metric: str = "rmse"
    early_stopping_metric: str | None = None
    n_estimators: int = 2000
    early_stopping_rounds: int = 100

    def __post_init__(self) -> None:
        if self.n_estimators < 1:
            raise ValueError("n_estimators pozitif olmali")
        if self.early_stopping_rounds < 1:
            raise ValueError("early_stopping_rounds pozitif olmali")


@dataclass(frozen=True)
class ExecutionPolicy:
    keep_models: bool = True
    n_jobs: int = 1
    max_memory_mb: int | None = None

    def __post_init__(self) -> None:
        if self.n_jobs < 1:
            raise ValueError("n_jobs pozitif olmali")
        if self.max_memory_mb is not None and self.max_memory_mb < 1:
            raise ValueError("max_memory_mb pozitif olmali")


@dataclass(frozen=True)
class PipelineRecipe:
    schema_version: int = 1
    seed: int = 42
    cv: CVRecipe = field(default_factory=CVRecipe)
    features: FeatureRecipe = field(default_factory=FeatureRecipe)
    model: ModelRecipe = field(default_factory=ModelRecipe)
    execution: ExecutionPolicy = field(default_factory=ExecutionPolicy)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Desteklenmeyen recipe schema_version: {self.schema_version}")
        if self.seed < 0:
            raise ValueError("seed negatif olamaz")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PipelineRecipe:
        # Ic ice alanlar ACIKCA dogrulanir. ``CVRecipe(**payload["cv"])`` bir
        # liste/metin geldiginde de basarisiz olur, ama mesaji
        # "argument after ** must be a mapping, not list" olur -- bozuk bir
        # recipe dosyasini eline alan kisiye HANGI alanin bozuk oldugunu
        # soylemez. Fail-fast zaten calisiyordu; eksik olan tesbisti.
        for alan in ("cv", "features", "model", "execution"):
            if alan in payload and not isinstance(payload[alan], dict):
                raise ValueError(
                    f"Recipe alani '{alan}' bir nesne (mapping) olmali, "
                    f"{type(payload[alan]).__name__} geldi."
                )
        feature_payload = payload.get("features", {})
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            seed=int(payload.get("seed", 42)),
            cv=CVRecipe(**payload.get("cv", {})),
            features=FeatureRecipe(
                **{
                    **feature_payload,
                    "target_shifts": tuple(feature_payload.get("target_shifts", (1, 7, 14, 28))),
                    "rolling_windows": tuple(feature_payload.get("rolling_windows", (7, 28))),
                    "history_value_columns": tuple(
                        feature_payload.get("history_value_columns", ())
                    ),
                    "history_shifts": tuple(feature_payload.get("history_shifts", ())),
                    "history_rolling_windows": tuple(
                        feature_payload.get("history_rolling_windows", ())
                    ),
                    "history_rolling_aggregations": tuple(
                        feature_payload.get("history_rolling_aggregations", ())
                    ),
                    "families": tuple(
                        feature_payload.get("families", ("calendar", "holiday", "frequency", "lag"))
                    ),
                }
            ),
            model=ModelRecipe(**payload.get("model", {})),
            execution=ExecutionPolicy(**payload.get("execution", {})),
        )

    @classmethod
    def from_json(cls, payload: str) -> PipelineRecipe:
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError("Pipeline recipe bir JSON nesnesi olmali")
        return cls.from_dict(value)


def default_pipeline_recipe() -> PipelineRecipe:
    """Return a new immutable default recipe."""

    return PipelineRecipe()
