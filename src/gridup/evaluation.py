"""Statistically honest model-selection decisions for temporal backtests."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np

__all__ = [
    "BenchmarkDecision",
    "OuterAnchor",
    "OuterEvidence",
    "paired_model_decision",
]


@dataclass(frozen=True)
class OuterAnchor:
    """One untouched temporal validation anchor with reproducibility evidence."""

    anchor_id: str
    train_end: str
    validation_start: str
    validation_end: str
    scores: Mapping[str, float]
    recipe_fingerprint: str
    fold_fingerprint: str

    def __post_init__(self) -> None:
        if not self.anchor_id:
            raise ValueError("Outer anchor kimligi ve provenance fingerprint'leri zorunludur")
        for name, fingerprint in (
            ("recipe", self.recipe_fingerprint),
            ("fold", self.fold_fingerprint),
        ):
            if re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
                raise ValueError(f"Outer anchor {name} fingerprint SHA-256 olmali")
        train_end = datetime.fromisoformat(self.train_end)
        validation_start = datetime.fromisoformat(self.validation_start)
        validation_end = datetime.fromisoformat(self.validation_end)
        if train_end >= validation_start:
            raise ValueError("Outer anchor train_end, validation_start'tan once olmali")
        if validation_start > validation_end:
            raise ValueError("Outer anchor validation_start, validation_end'i gecemez")
        if len(self.scores) < 2 or not all(np.isfinite(float(v)) for v in self.scores.values()):
            raise ValueError("Outer anchor en az iki sonlu aday skoru tasimalidir")


@dataclass(frozen=True)
class OuterEvidence:
    """Ordered independent anchors; a boolean attestation cannot replace this."""

    anchors: tuple[OuterAnchor, ...]

    def __post_init__(self) -> None:
        anchor_ids = [anchor.anchor_id for anchor in self.anchors]
        fold_ids = [anchor.fold_fingerprint for anchor in self.anchors]
        if len(set(anchor_ids)) != len(anchor_ids):
            raise ValueError("Outer anchor_id degerleri benzersiz olmali")
        if len(set(fold_ids)) != len(fold_ids):
            raise ValueError("Outer fold fingerprint'leri benzersiz olmali")
        candidate_sets = {frozenset(anchor.scores) for anchor in self.anchors}
        if len(candidate_sets) > 1:
            raise ValueError("Her outer anchor ayni adaylari skorlamalidir")
        starts = [datetime.fromisoformat(anchor.validation_start) for anchor in self.anchors]
        if starts != sorted(starts):
            raise ValueError("Outer anchor'lar kronolojik sirada olmali")
        for previous, current in zip(self.anchors, self.anchors[1:], strict=False):
            previous_end = datetime.fromisoformat(previous.validation_end)
            current_start = datetime.fromisoformat(current.validation_start)
            if previous_end >= current_start:
                raise ValueError("Outer validation araliklari ayrik olmali; ortusme bulundu")

    def score_map(self) -> dict[str, np.ndarray]:
        if not self.anchors:
            return {}
        names = tuple(self.anchors[0].scores)
        return {
            name: np.asarray([anchor.scores[name] for anchor in self.anchors], dtype="float64")
            for name in names
        }


@dataclass(frozen=True)
class BenchmarkDecision:
    candidate_name: str
    baseline_name: str
    winner: str | None
    statistically_conclusive: bool
    decision_reason: str
    mean_paired_difference: float
    ci_low: float
    ci_high: float
    practical_effect: float
    n_anchors: int

    def to_dict(self) -> dict[str, str | float | int | bool | None]:
        return asdict(self)


def paired_model_decision(
    *,
    candidate_scores: list[float] | np.ndarray,
    baseline_scores: list[float] | np.ndarray,
    candidate_name: str,
    baseline_name: str,
    practical_effect: float = 0.0,
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
    seed: int = 42,
    greater_is_better: bool = False,
    block_length: int = 2,
) -> BenchmarkDecision:
    """Compare models using paired temporal anchors and a bootstrap interval.

    A winner is declared only when the full confidence interval clears both
    zero and the configured practical-effect threshold. This prevents a tiny
    same-OOF improvement from being presented as a proven winner.
    """

    candidate = np.asarray(candidate_scores, dtype="float64")
    baseline = np.asarray(baseline_scores, dtype="float64")
    if candidate.shape != baseline.shape or candidate.ndim != 1:
        raise ValueError("candidate ve baseline skorlari ayni uzunlukta 1B olmali")
    if len(candidate) < 6:
        raise ValueError("Guvenilir model karari icin en az 6 eslesmis anchor gerekli")
    if not np.isfinite(candidate).all() or not np.isfinite(baseline).all():
        raise ValueError("Anchor skorlarinda NaN veya sonsuz deger olamaz")
    if not 0 < confidence < 1:
        raise ValueError("confidence 0 ile 1 arasinda olmali")
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap en az 100 olmali")
    if block_length < 2 or block_length > len(candidate):
        raise ValueError(f"Temporal bootstrap block_length 2..{len(candidate)} arasinda olmali")
    if practical_effect < 0:
        raise ValueError("practical_effect negatif olamaz")

    difference = candidate - baseline
    rng = np.random.default_rng(seed)
    blocks_per_sample = int(np.ceil(len(difference) / block_length))
    starts = rng.integers(
        0,
        len(difference) - block_length + 1,
        size=(n_bootstrap, blocks_per_sample),
    )
    offsets = np.arange(block_length)
    indices = (starts[..., None] + offsets).reshape(n_bootstrap, -1)[:, : len(difference)]
    sampled = difference[indices]
    means = sampled.mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    ci_low, ci_high = np.quantile(means, [alpha, 1.0 - alpha])

    if greater_is_better:
        candidate_wins = ci_low > practical_effect
        baseline_wins = ci_high < -practical_effect
    else:
        candidate_wins = ci_high < -practical_effect
        baseline_wins = ci_low > practical_effect

    if candidate_wins:
        winner, reason = candidate_name, "candidate_better"
    elif baseline_wins:
        winner, reason = baseline_name, "baseline_better"
    else:
        winner, reason = None, "inconclusive"

    return BenchmarkDecision(
        candidate_name=candidate_name,
        baseline_name=baseline_name,
        winner=winner,
        statistically_conclusive=winner is not None,
        decision_reason=reason,
        mean_paired_difference=float(difference.mean()),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        practical_effect=float(practical_effect),
        n_anchors=len(difference),
    )
