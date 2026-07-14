from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def assign_stratum(
    group_summary: pd.DataFrame,
    vision_std_col: str = "multiples_score_std",
    language_std_col: str = "indonesian_score_std",
    product_count_col: str = "product_count",
) -> pd.DataFrame:
    """Label groups as higher_variance or lower_variance based on std dev and group size.

    A group is considered *higher_variance* if its standard deviation on
    either score column, or its product count, exceeds the 75th percentile
    (Q3) of that column's distribution across all groups.  Otherwise it is
    *lower_variance*.

    Args:
        group_summary: DataFrame with std columns and a product_count column.
        vision_std_col: Column for the vision-based std.
        language_std_col: Column for the language-based std.
        product_count_col: Column for the product count per group.

    Returns:
        The same DataFrame with a new ``stratum`` column added.
    """
    q75_vision = group_summary[vision_std_col].quantile(0.75)
    q75_language = group_summary[language_std_col].quantile(0.75)
    q75_product_count = group_summary[product_count_col].quantile(0.75)

    higher_vision = group_summary[vision_std_col] > q75_vision
    higher_language = group_summary[language_std_col] > q75_language
    higher_product_count = group_summary[product_count_col] > q75_product_count

    group_summary["stratum"] = "lower_variance"
    group_summary.loc[
        higher_vision | higher_language | higher_product_count, "stratum"
    ] = "higher_variance"
    return group_summary


def stratified_group_split(
    groups: pd.DataFrame,
    stratify_col: str = "stratum",
    split_ratios: list[float] = [0.7, 0.15, 0.15],
    random_seed: int = 42,
) -> pd.DataFrame:
    """Split groups into train / val / test while preserving stratum proportions.

    The first split produces train and a temporary set; the temporary
    set is split 50/50 into val and test.

    Args:
        groups: DataFrame with at least a ``stratum`` column.
        stratify_col: Column used for stratification.
        split_ratios: Three-element list [train, val, test].
        random_seed: Random seed for reproducibility.

    Returns:
        The input DataFrame with a ``split`` column added
        (values: "train", "val", "test").
    """
    train_ratio = split_ratios[0]
    temp_ratio = 1.0 - train_ratio

    train_groups, temp_groups = train_test_split(
        groups,
        test_size=temp_ratio,
        stratify=groups[stratify_col],
        random_state=random_seed,
    )

    val_groups, test_groups = train_test_split(
        temp_groups,
        test_size=0.5,
        stratify=temp_groups[stratify_col],
        random_state=random_seed,
    )

    train_groups = train_groups.copy()
    val_groups = val_groups.copy()
    test_groups = test_groups.copy()
    train_groups["split"] = "train"
    val_groups["split"] = "val"
    test_groups["split"] = "test"

    return pd.concat([train_groups, val_groups, test_groups], axis=0)
