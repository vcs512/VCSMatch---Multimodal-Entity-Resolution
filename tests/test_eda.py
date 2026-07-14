from __future__ import annotations

import pandas as pd
import pytest

from src.core.datasets.dataset_split import assign_stratum, stratified_group_split


class TestAssignStratum:
    def test_higher_variance_when_std_exceeds_q75(self):
        groups = pd.DataFrame(
            {
                "label_group": list(range(10)),
                "multiples_score_std": [0.1] * 9 + [0.9],
                "indonesian_score_std": [0.1] * 10,
                "product_count": [1] * 10,
            }
        )
        result = assign_stratum(group_summary=groups)
        assert result.loc[9, "stratum"] == "higher_variance"
        assert result.loc[0, "stratum"] == "lower_variance"

    def test_higher_variance_when_either_std_exceeds_q75(self):
        groups = pd.DataFrame(
            {
                "label_group": list(range(10)),
                "multiples_score_std": [0.1] * 10,
                "indonesian_score_std": [0.1] * 9 + [0.9],
                "product_count": [1] * 10,
            }
        )
        result = assign_stratum(group_summary=groups)
        assert result.loc[9, "stratum"] == "higher_variance"
        assert result.loc[0, "stratum"] == "lower_variance"

    def test_higher_variance_when_product_count_exceeds_q75(self):
        groups = pd.DataFrame(
            {
                "label_group": list(range(10)),
                "multiples_score_std": [0.1] * 10,
                "indonesian_score_std": [0.1] * 10,
                "product_count": [1] * 9 + [10],
            }
        )
        result = assign_stratum(group_summary=groups)
        assert result.loc[9, "stratum"] == "higher_variance"
        assert result.loc[0, "stratum"] == "lower_variance"


class TestStratifiedGroupSplit:
    def test_returns_all_splits(self):
        groups = pd.DataFrame(
            {
                "label_group": list(range(100)),
                "multiples_score_std": [0.1] * 25 + [0.2] * 25 + [0.8] * 25 + [0.9] * 25,
                "indonesian_score_std": [0.1] * 50 + [0.9] * 50,
                "product_count": [1] * 100,
            }
        )
        groups = assign_stratum(group_summary=groups)
        result = stratified_group_split(groups=groups, random_seed=42)
        assert "split" in result.columns
        assert set(result["split"].unique()) == {"train", "val", "test"}
        train_pct = result["split"].value_counts()["train"] / len(result)
        assert train_pct == pytest.approx(0.7, abs=0.05)
