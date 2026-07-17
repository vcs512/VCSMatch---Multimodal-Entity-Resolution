from __future__ import annotations

import json

import pandas as pd
import pytest

from src.core.evaluation.metrics import (
    compute_per_row_metrics,
    summarize_by_stratum,
    summarize_metrics,
)


def _make_retrieval_df(
    posting_ids: list[str],
    label_groups: list[int],
    strata: list[str],
    retrieved_lists: list[list[str]],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "posting_id": posting_ids,
            "label_group": label_groups,
            "stratum": strata,
            "retrieved_products": [json.dumps(rl) for rl in retrieved_lists],
        }
    )


class TestComputePerRowMetrics:
    def test_perfect_retrieval(self):
        df = _make_retrieval_df(
            posting_ids=["a", "b", "c"],
            label_groups=[1, 1, 2],
            strata=["low", "low", "high"],
            retrieved_lists=[
                ["a", "b"],
                ["b", "a"],
                ["c"],
            ],
        )
        result = compute_per_row_metrics(df=df, recall_ks=[1, 5])
        assert result.loc[0, "precision"] == 1.0
        assert result.loc[0, "recall"] == 1.0
        assert result.loc[0, "f1"] == 1.0
        assert result.loc[0, "recall@1"] == 0.5
        assert result.loc[0, "recall@5"] == 1.0

    def test_no_relevant_retrieved(self):
        df = _make_retrieval_df(
            posting_ids=["a", "b"],
            label_groups=[1, 2],
            strata=["low", "high"],
            retrieved_lists=[["b"], ["a"]],
        )
        result = compute_per_row_metrics(df=df, recall_ks=[1])
        assert result.loc[0, "precision"] == 0.0
        assert result.loc[0, "recall"] == 0.0
        assert result.loc[0, "f1"] == 0.0
        assert result.loc[0, "recall@1"] == 0.0

    def test_partial_relevance(self):
        df = _make_retrieval_df(
            posting_ids=["a", "b", "c", "d"],
            label_groups=[1, 1, 1, 2],
            strata=["low", "low", "low", "high"],
            retrieved_lists=[
                ["a", "b", "d"],
                ["b", "a"],
                ["c", "a"],
                ["d"],
            ],
        )
        result = compute_per_row_metrics(df=df, recall_ks=[2])
        row_a = result[result["posting_id"] == "a"].iloc[0]
        assert row_a["precision"] == pytest.approx(2 / 3)
        assert row_a["recall"] == pytest.approx(2 / 3)
        assert row_a["recall@2"] == pytest.approx(2 / 3)

    def test_empty_retrieved_list(self):
        df = _make_retrieval_df(
            posting_ids=["a"],
            label_groups=[1],
            strata=["low"],
            retrieved_lists=[[]],
        )
        result = compute_per_row_metrics(df=df, recall_ks=[5])
        assert result.loc[0, "precision"] == 0.0
        assert result.loc[0, "recall"] == 0.0
        assert result.loc[0, "f1"] == 0.0
        assert result.loc[0, "recall@5"] == 0.0

    def test_retrieved_less_than_k(self):
        df = _make_retrieval_df(
            posting_ids=["a", "b"],
            label_groups=[1, 1],
            strata=["low", "low"],
            retrieved_lists=[["a"], ["b"]],
        )
        result = compute_per_row_metrics(df=df, recall_ks=[5, 10])
        assert result.loc[0, "recall@5"] == 0.5
        assert result.loc[0, "recall@10"] == 0.5

    def test_custom_recall_ks(self):
        df = _make_retrieval_df(
            posting_ids=["a", "b", "c"],
            label_groups=[1, 1, 1],
            strata=["low", "low", "low"],
            retrieved_lists=[["a", "b", "c"], ["b", "a", "c"], ["c", "a", "b"]],
        )
        result = compute_per_row_metrics(df=df, recall_ks=[1, 2, 3])
        assert result.loc[0, "recall@1"] == pytest.approx(1 / 3)
        assert result.loc[0, "recall@2"] == pytest.approx(2 / 3)
        assert result.loc[0, "recall@3"] == 1.0


class TestSummarizeMetrics:
    def test_mean_of_metric_columns(self):
        df = pd.DataFrame(
            {
                "posting_id": ["a", "b"],
                "label_group": [1, 1],
                "stratum": ["low", "low"],
                "precision": [1.0, 0.5],
                "recall": [1.0, 0.5],
                "f1": [1.0, 0.5],
                "recall@5": [1.0, 0.5],
            }
        )
        summary = summarize_metrics(df_metrics=df)
        assert summary["precision"] == 0.75
        assert summary["recall"] == 0.75
        assert summary["f1"] == 0.75
        assert summary["recall@5"] == 0.75


class TestSummarizeByStratum:
    def test_grouped_means(self):
        df = pd.DataFrame(
            {
                "posting_id": ["a", "b", "c", "d"],
                "label_group": [1, 1, 2, 2],
                "stratum": ["low", "low", "high", "high"],
                "precision": [1.0, 0.5, 0.0, 1.0],
                "recall": [1.0, 0.5, 0.0, 1.0],
                "f1": [1.0, 0.5, 0.0, 1.0],
                "recall@5": [1.0, 0.5, 0.0, 1.0],
            }
        )
        result = summarize_by_stratum(df_metrics=df)
        low = result[result["stratum"] == "low"].iloc[0]
        high = result[result["stratum"] == "high"].iloc[0]
        assert low["precision"] == 0.75
        assert high["precision"] == 0.5
