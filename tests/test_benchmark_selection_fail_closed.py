"""Gercek benchmark'in ayni-OOF kazanan yanilgisina karsi karar kontrati."""

from __future__ import annotations

import hashlib

import pytest
from scripts.benchmark_gercek import bilimsel_kazanan_karari

from gridup.evaluation import OuterAnchor, OuterEvidence


def _evidence(n: int = 6) -> OuterEvidence:
    anchors = tuple(
        OuterAnchor(
            anchor_id=f"a{index}",
            train_end=f"2025-{index + 1:02d}-01",
            validation_start=f"2025-{index + 1:02d}-02",
            validation_end=f"2025-{index + 1:02d}-20",
            scores={"tek_model": 10.0, "harman": 8.0},
            recipe_fingerprint=hashlib.sha256(f"recipe-{index}".encode()).hexdigest(),
            fold_fingerprint=hashlib.sha256(f"fold-{index}".encode()).hexdigest(),
        )
        for index in range(n)
    )
    return OuterEvidence(anchors)


def test_same_oof_blend_is_not_a_winner_without_outer_evidence() -> None:
    decision = bilimsel_kazanan_karari(
        {"tek_model": 10.0, "harman": 9.0},
    )

    assert decision["apparent_oof_best"] == "harman"
    assert decision["winner"] is None
    assert decision["statistically_conclusive"] is False
    reason = decision["decision_reason"].lower()
    assert "ayni oof" in reason
    assert "en az 6" in reason
    assert "outer" in reason


def test_six_anchors_are_not_enough_without_independence_attestation() -> None:
    decision = bilimsel_kazanan_karari(
        {"tek_model": 10.0, "harman": 9.0},
        outer_anchor_scores={
            "tek_model": [10.0] * 6,
            "harman": [8.0] * 6,
        },
        independent_outer=False,
    )

    assert decision["winner"] is None
    assert decision["statistically_conclusive"] is False
    assert "bagimsiz" in decision["decision_reason"].lower()


def test_boolean_attestation_cannot_replace_structured_outer_evidence() -> None:
    decision = bilimsel_kazanan_karari(
        {"tek_model": 10.0, "harman": 9.0},
        outer_anchor_scores={"tek_model": [10.0] * 6, "harman": [8.0] * 6},
        independent_outer=True,
    )

    assert decision["winner"] is None
    assert "provenance" in decision["decision_reason"].lower()


def test_fewer_than_six_independent_paired_anchors_fail_closed() -> None:
    decision = bilimsel_kazanan_karari(
        {"tek_model": 10.0, "harman": 9.0},
        outer_evidence=_evidence(5),
    )

    assert decision["winner"] is None
    assert decision["statistically_conclusive"] is False
    assert decision["n_anchors"] == 5
    assert "en az 6" in decision["decision_reason"].lower()


def test_consistent_independent_outer_evidence_can_declare_the_preselected_winner() -> None:
    decision = bilimsel_kazanan_karari(
        {"tek_model": 10.0, "harman": 9.0},
        outer_evidence=_evidence(),
        practical_effect=0.5,
        n_bootstrap=2_000,
    )

    assert decision["winner"] == "harman"
    assert decision["statistically_conclusive"] is True
    assert decision["n_anchors"] == 6
    assert decision["pairwise_decisions"][0]["winner"] == "harman"


def test_outer_evidence_rejects_overlapping_validation_intervals() -> None:
    first, second = _evidence(2).anchors
    overlapping = OuterAnchor(
        anchor_id="overlap",
        train_end="2025-01-05",
        validation_start="2025-01-10",
        validation_end="2025-01-25",
        scores=second.scores,
        recipe_fingerprint=second.recipe_fingerprint,
        fold_fingerprint=second.fold_fingerprint,
    )

    with pytest.raises(ValueError, match="ortus|ayrik|overlap"):
        OuterEvidence((first, overlapping))
