from __future__ import annotations

import pytest

from gridup.evaluation import paired_model_decision


def test_benchmark_requires_at_least_six_paired_anchors():
    with pytest.raises(ValueError, match="en az 6"):
        paired_model_decision(
            candidate_scores=[9, 9, 9, 9],
            baseline_scores=[10, 10, 10, 10],
            candidate_name="blend",
            baseline_name="single",
        )


def test_uncertain_small_gain_does_not_declare_a_winner():
    result = paired_model_decision(
        candidate_scores=[9.0, 11.0, 9.0, 11.0, 9.0, 11.0],
        baseline_scores=[10.0] * 6,
        candidate_name="blend",
        baseline_name="single",
        n_bootstrap=2000,
        seed=7,
    )

    assert result.winner is None
    assert result.statistically_conclusive is False
    assert result.decision_reason == "inconclusive"


def test_consistent_practical_gain_declares_candidate():
    result = paired_model_decision(
        candidate_scores=[8.0, 8.2, 7.9, 8.1, 8.0, 8.1],
        baseline_scores=[10.0, 10.2, 9.9, 10.1, 10.0, 10.1],
        candidate_name="blend",
        baseline_name="single",
        practical_effect=0.5,
        n_bootstrap=2000,
        seed=11,
    )

    assert result.winner == "blend"
    assert result.statistically_conclusive is True
    assert result.ci_high < -0.5
    assert result.n_anchors == 6


def test_temporal_bootstrap_uses_blocks_larger_than_one():
    with pytest.raises(ValueError, match="block"):
        paired_model_decision(
            candidate_scores=[8.0] * 6,
            baseline_scores=[10.0] * 6,
            candidate_name="blend",
            baseline_name="single",
            block_length=1,
        )
