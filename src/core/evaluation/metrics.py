from __future__ import annotations

import json

import pandas as pd
from tqdm import tqdm


def _compute_group_sizes(
    df: pd.DataFrame,
    group_col: str = "label_group",
) -> dict[int, int]:
    """Count the number of products per label group.

    Args:
        df: DataFrame with a label_group column.
        group_col: Name of the group column.

    Returns:
        Dict mapping each group ID to its product count.
    """
    return df[group_col].value_counts().to_dict()


def compute_per_row_metrics(
    df: pd.DataFrame,
    recall_ks: list[int],
    retrieved_col: str = "retrieved_products",
    group_col: str = "label_group",
) -> pd.DataFrame:
    """Compute precision, recall, f1, and recall@k for every query row.

    Args:
        df: DataFrame with columns posting_id, label_group, stratum,
            and retrieved_products (JSON list of posting_ids).
        recall_ks: List of k values for recall@k computation.
        retrieved_col: Name of the column with JSON retrieved lists.
        group_col: Name of the ground-truth group column.

    Returns:
        DataFrame with per-row metrics: precision, recall, f1,
        and recall@{k} for each k in recall_ks.
    """
    group_sizes = _compute_group_sizes(df=df, group_col=group_col)
    lookup = df.set_index("posting_id")[group_col].to_dict()

    rows = []
    for _, row in tqdm(df.iterrows(), desc="Computing metrics", total=len(df)):
        query_group = row[group_col]
        retrieved = json.loads(row[retrieved_col])
        total_group_size = group_sizes[query_group]

        cumulative = 0
        recall_at_k = {}
        for i, pid in enumerate(retrieved):
            if lookup.get(pid) == query_group:
                cumulative += 1
            if (i + 1) in recall_ks:
                recall_at_k[i + 1] = (
                    cumulative / total_group_size
                    if total_group_size > 0
                    else 0.0
                )

        for k in recall_ks:
            if k not in recall_at_k:
                recall_at_k[k] = (
                    cumulative / total_group_size
                    if total_group_size > 0
                    else 0.0
                )

        matches = cumulative
        precision = matches / len(retrieved) if retrieved else 0.0
        recall = matches / total_group_size if total_group_size > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        row = {
            "posting_id": row["posting_id"],
            group_col: query_group,
            "stratum": row["stratum"],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        for k in recall_ks:
            row[f"recall-{k}"] = recall_at_k.get(k, 0.0)

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_metrics(df_metrics: pd.DataFrame) -> pd.Series:
    """Compute the mean of all metric columns across all rows.

    Args:
        df_metrics: DataFrame with per-row metrics (output of
            compute_per_row_metrics).

    Returns:
        Series with the mean value of each metric column.
    """
    metric_cols = [
        c
        for c in df_metrics.columns
        if c not in ("posting_id", "label_group", "stratum")
    ]
    return df_metrics[metric_cols].mean()


def summarize_by_stratum(df_metrics: pd.DataFrame) -> pd.DataFrame:
    """Compute the mean of each metric grouped by stratum.

    Args:
        df_metrics: DataFrame with per-row metrics (output of
            compute_per_row_metrics).

    Returns:
        DataFrame with one row per stratum and the mean of each
        metric column.
    """
    metric_cols = [
        c
        for c in df_metrics.columns
        if c not in ("posting_id", "label_group", "stratum")
    ]
    return df_metrics.groupby("stratum")[metric_cols].mean().reset_index()
