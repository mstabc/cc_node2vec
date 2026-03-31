import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_DATASETS = [
    "sepsis",
    "bpic_2012_A",
    "bpic_2017_O",
    "bpic_2020_Domestic_declarations",
    "bpic_2020_International_declarations",
    "bpic_2020_Permit_log",
    "bpic_2020_Prepaid_travel_cost",
    "bpic_2020_Request_for_payment",
    "roadtraffic",
]
EXPECTED_METHODS = ["ilp", "inductive", "heuristics", "split", "gnn"]
EXPECTED_SCENARIOS = ["static", "context_aware"]
EXPECTED_ALPHAS = [0.5, 1.0, 2.0]
EXPECTED_DIMS = [32, 64, 128]
EXPECTED_MIN_COUNTS = [50, 100, 200]
EXPECTED_TOTAL = 2430

TIME_COLUMNS = [
    "time_petrinetgraph_construction",
    "time_random_walk_generation",
    "time_word2vec_training",
    "time_procrustes_alignment",
    "time_distance_computation",
    "time_total",
]

NUMERIC_COLUMNS = [
    "alpha",
    "dim",
    "min_count",
    "lambda_val",
    "D_weighted",
    "matched_nodes",
    "unmatched_nodes",
    "procrustes_residual_total",
    "procrustes_residual_per_anchor",
    "anchor_count",
    "embedding_dim",
    "ratio",
    "gt_nodes_dropped",
    "gt_nodes_dropped_pct",
    "model_nodes_dropped",
    "model_nodes_dropped_pct",
    *TIME_COLUMNS,
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=str,
        default="results/results_master.csv",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def load_master_csv(path):
    dataframe = pd.read_csv(path)
    for column in NUMERIC_COLUMNS:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
    return dataframe


def match_numeric(series, value):
    numeric_series = pd.to_numeric(series, errors="coerce")
    if np.isinf(value):
        return np.isinf(numeric_series)
    return np.isclose(numeric_series, value, equal_nan=False)


def filter_rows(
    dataframe,
    dataset=None,
    method=None,
    scenario=None,
    alpha=None,
    dim=None,
    min_count=None,
    lambda_val=None,
):
    mask = pd.Series(True, index=dataframe.index)

    if dataset is not None:
        if isinstance(dataset, (list, tuple, set)):
            mask &= dataframe["dataset"].isin(dataset)
        else:
            mask &= dataframe["dataset"] == dataset

    if method is not None:
        if isinstance(method, (list, tuple, set)):
            mask &= dataframe["method"].isin(method)
        else:
            mask &= dataframe["method"] == method

    if scenario is not None:
        if isinstance(scenario, (list, tuple, set)):
            mask &= dataframe["scenario"].isin(scenario)
        else:
            mask &= dataframe["scenario"] == scenario

    if alpha is not None:
        mask &= match_numeric(dataframe["alpha"], alpha)

    if dim is not None:
        mask &= match_numeric(dataframe["dim"], dim)

    if min_count is not None:
        mask &= match_numeric(dataframe["min_count"], min_count)

    if lambda_val is not None:
        mask &= match_numeric(dataframe["lambda_val"], lambda_val)

    return dataframe.loc[mask].copy()


def prefer_lambda_inf(dataframe, subset_columns):
    if dataframe.empty:
        return dataframe

    if np.isinf(dataframe["lambda_val"]).any():
        dataframe = dataframe.loc[np.isinf(dataframe["lambda_val"])].copy()

    return (
        dataframe.sort_values(subset_columns + ["lambda_val"])
        .drop_duplicates(subset=subset_columns, keep="first")
        .copy()
    )


def derive_nodes_total(dataframe):
    nodes_total = pd.Series(np.nan, index=dataframe.index, dtype=float)
    positive_pct_mask = dataframe["gt_nodes_dropped_pct"] > 0
    nodes_total.loc[positive_pct_mask] = (
        dataframe.loc[positive_pct_mask, "gt_nodes_dropped"] * 100.0
        / dataframe.loc[positive_pct_mask, "gt_nodes_dropped_pct"]
    )
    return nodes_total


def save_table(dataframe, output_path):
    dataframe.to_csv(output_path, index=False, float_format="%.6f")


def normalize_key_float(value):
    if np.isinf(value):
        return float("inf")
    return round(float(value), 6)


def build_completed_key_set(dataframe):
    filtered = dataframe[
        dataframe["dataset"].isin(EXPECTED_DATASETS)
        & dataframe["method"].isin(EXPECTED_METHODS)
        & dataframe["scenario"].isin(EXPECTED_SCENARIOS)
        & np.isinf(dataframe["lambda_val"])
    ].copy()

    filtered = filtered[
        filtered["alpha"].isin(EXPECTED_ALPHAS)
        & filtered["dim"].isin(EXPECTED_DIMS)
        & filtered["min_count"].isin(EXPECTED_MIN_COUNTS)
    ]

    return {
        (
            row.dataset,
            row.method,
            row.scenario,
            normalize_key_float(row.alpha),
            int(row.dim),
            int(row.min_count),
        )
        for row in filtered.itertuples()
    }


def build_expected_key_set():
    return set(
        product(
            EXPECTED_DATASETS,
            EXPECTED_METHODS,
            EXPECTED_SCENARIOS,
            EXPECTED_ALPHAS,
            EXPECTED_DIMS,
            EXPECTED_MIN_COUNTS,
        )
    )


def format_summary_value(value):
    if pd.isna(value):
        return "nan"
    if np.isinf(value):
        return "inf"
    return f"{value:.6f}"


def main():
    args = parse_args()
    input_path = Path(args.input_csv).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Master CSV not found: {input_path}")

    output_dir = (
        Path(args.output_dir).resolve() if args.output_dir else input_path.parent.resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = load_master_csv(input_path)

    runtime_df = filter_rows(dataframe, alpha=1.0, dim=128, min_count=200)
    runtime_df = prefer_lambda_inf(runtime_df, ["dataset", "method", "scenario"])
    runtime_df["nodes_total"] = derive_nodes_total(runtime_df)
    runtime_df = runtime_df[
        [
            "dataset",
            "method",
            "scenario",
            "nodes_total",
            *TIME_COLUMNS,
        ]
    ].sort_values(["dataset", "method", "scenario"])
    save_table(runtime_df, output_dir / "table_runtime.csv")

    residual_df = filter_rows(
        dataframe,
        alpha=1.0,
        dim=128,
        min_count=200,
        scenario="context_aware",
    )
    residual_df = residual_df[
        [
            "dataset",
            "method",
            "lambda_val",
            "D_weighted",
            "matched_nodes",
            "unmatched_nodes",
            "procrustes_residual_total",
            "procrustes_residual_per_anchor",
            "ratio",
        ]
    ].sort_values(["dataset", "method", "lambda_val"])
    save_table(residual_df, output_dir / "table_procrustes_residuals.csv")

    alpha_df = filter_rows(
        dataframe,
        dataset=["sepsis", "bpic_2020_Permit_log"],
        dim=128,
        min_count=200,
        lambda_val=float("inf"),
        scenario="context_aware",
    )
    alpha_df = alpha_df[
        [
            "dataset",
            "method",
            "alpha",
            "D_weighted",
        ]
    ].sort_values(["dataset", "method", "alpha"])
    save_table(alpha_df, output_dir / "table_alpha_ablation.csv")

    dim_df = filter_rows(
        dataframe,
        alpha=1.0,
        min_count=200,
        lambda_val=float("inf"),
        scenario="context_aware",
    )
    dim_df = dim_df[
        [
            "dataset",
            "method",
            "dim",
            "D_weighted",
            "procrustes_residual_per_anchor",
            "ratio",
            "matched_nodes",
        ]
    ].sort_values(["dataset", "method", "dim"])
    save_table(dim_df, output_dir / "table_dim_ablation.csv")

    # The requested schema omits scenario here; context-aware keeps one row per dataset/method/min_count.
    mincount_df = filter_rows(
        dataframe,
        alpha=1.0,
        dim=128,
        lambda_val=float("inf"),
        scenario="context_aware",
    )
    mincount_df = mincount_df[
        [
            "dataset",
            "method",
            "dim",
            "min_count",
            "gt_nodes_dropped",
            "gt_nodes_dropped_pct",
            "model_nodes_dropped",
            "model_nodes_dropped_pct",
        ]
    ].sort_values(["dataset", "method", "min_count"])
    save_table(mincount_df, output_dir / "table_mincount_dropout.csv")

    expected_keys = build_expected_key_set()
    completed_keys = build_completed_key_set(dataframe)
    missing_keys = sorted(expected_keys - completed_keys)

    dim_128_df = filter_rows(dataframe, dim=128)
    dim_128_residuals = dim_128_df["procrustes_residual_per_anchor"].replace(
        [np.inf, -np.inf], np.nan
    )

    default_runtime_df = filter_rows(dataframe, alpha=1.0, dim=128, min_count=200)
    default_runtime_df = prefer_lambda_inf(
        default_runtime_df, ["dataset", "method", "scenario"]
    )
    mean_stage_times = default_runtime_df[TIME_COLUMNS].mean(numeric_only=True)

    print(f"Total rows in master CSV: {len(dataframe)}")
    print(f"Completed expected configs: {len(completed_keys)} / {EXPECTED_TOTAL}")
    if missing_keys:
        print("Missing combinations:")
        for dataset, method, scenario, alpha, dim, min_count in missing_keys:
            print(
                "  "
                f"dataset={dataset}, method={method}, scenario={scenario}, "
                f"alpha={alpha:.1f}, dim={dim}, min_count={min_count}, lambda_val=inf"
            )
    else:
        print("Missing combinations: none")

    print(
        "Mean procrustes_residual_per_anchor at dim=128: "
        f"{format_summary_value(dim_128_residuals.mean())}"
    )
    print(
        "Max procrustes_residual_per_anchor at dim=128: "
        f"{format_summary_value(dim_128_residuals.max())}"
    )
    print("Mean runtime per stage at default config:")
    for column in TIME_COLUMNS:
        print(f"  {column}: {format_summary_value(mean_stage_times.get(column, np.nan))}")


if __name__ == "__main__":
    main()
