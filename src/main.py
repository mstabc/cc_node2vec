import argparse
import csv
import glob
import math
import os
import time

import numpy as np
import pm4py
from gensim.models import Word2Vec
from pm4py.objects.petri_net.importer import importer as pnml_importer
from scipy.linalg import orthogonal_procrustes

from distance_cal import compare_vectors_with_penalty
from graph_encoding import global_encoder, update_node_and_edge_encodings
from petri_builder import PetriNetGraph, PetriNetMinedGraph
from random_walks import calculate_max_walk_length, generate_walks
from utils import get_xes_file


MASTER_COLUMNS = [
    "dataset",
    "method",
    "scenario",
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
    "time_petrinetgraph_construction",
    "time_random_walk_generation",
    "time_word2vec_training",
    "time_procrustes_alignment",
    "time_distance_computation",
    "time_total",
]

STAGE_KEYS = [
    "petrinetgraph_construction",
    "random_walk_generation",
    "word2vec_training",
    "procrustes_alignment",
    "distance_computation",
]

DATASET_REGISTRY = [
    {
        "name": "sepsis",
        "relative_path": ("data", "01-Sepsis"),
        "aliases": {"01-sepsis", "01_sepsis", "sepsis_log"},
    },
    {
        "name": "bpic_2012_A",
        "relative_path": ("data", "2012_A"),
        "aliases": {"2012_a", "bpic_2012_a", "bpic_2012_A"},
    },
    {
        "name": "bpic_2017_O",
        "relative_path": ("data", "2017_O"),
        "aliases": {"2017_o", "bpic_2017_o", "bpic_2017_O"},
    },
    {
        "name": "bpic_2020_Domestic_declarations",
        "relative_path": ("data", "2020_Domestic_declarations"),
        "aliases": {
            "2020_domestic_declarations",
            "bpic_2020_domestic_declarations",
            "bpic_2020_Domestic_declarations",
        },
    },
    {
        "name": "bpic_2020_International_declarations",
        "relative_path": ("data", "2020_International_declarations"),
        "aliases": {
            "2020_international_declarations",
            "bpic_2020_international_declarations",
            "bpic_2020_International_declarations",
        },
    },
    {
        "name": "bpic_2020_Permit_log",
        "relative_path": ("data", "2020_Permit_log"),
        "aliases": {
            "2020_permit_log",
            "bpic_2020_permit_log",
            "bpic_2020_Permit_log",
        },
    },
    {
        "name": "bpic_2020_Prepaid_travel_cost",
        "relative_path": ("data", "2020_Prepaid_travel_cost"),
        "aliases": {
            "2020_prepaid_travel_cost",
            "bpic_2020_prepaid_travel_cost",
            "bpic_2020_Prepaid_travel_cost",
        },
    },
    {
        "name": "bpic_2020_Request_for_payment",
        "relative_path": ("data", "2020_Request_for_payment"),
        "aliases": {
            "2020_request_for_payment",
            "bpic_2020_request_for_payment",
            "bpic_2020_Request_for_payment",
        },
    },
    {
        "name": "roadtraffic",
        "relative_path": ("data", "RoadTraffic"),
        "aliases": {"roadtraffic", "road_traffic", "RoadTraffic"},
    },
]

METHOD_FILE_PATTERNS = {
    "ilp": "data_ilp*.pnml",
    "inductive": "data_inductive*.pnml",
    "heuristics": "data_heuristics*.pnml",
    "split": "data_split*.pnml",
    "gnn": "data_gcn*.pnml",
}

METHOD_ALIASES = {
    "ilp": "ilp",
    "inductive": "inductive",
    "heuristics": "heuristics",
    "split": "split",
    "gnn": "gnn",
}

DESIRED_TYPES = {"sink", "source", "transition"}


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    if seconds < 3600:
        return f"{seconds / 60:.2f} minutes"
    return f"{seconds / 3600:.2f} hours"


def format_number(value):
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def summarize_log(log):
    case_lengths = log.groupby("case:concept:name").size()
    return {
        "cases": int(case_lengths.shape[0]),
        "events": int(len(log)),
        "activities": int(log["concept:name"].nunique()),
        "trace_len_min": int(case_lengths.min()),
        "trace_len_mean": float(case_lengths.mean()),
        "trace_len_max": int(case_lengths.max()),
    }


def summarize_graphs(graphs):
    graph_count = len(graphs)
    total_nodes = sum(graph.number_of_nodes() for _, graph in graphs)
    total_edges = sum(graph.number_of_edges() for _, graph in graphs)
    return {
        "graph_count": graph_count,
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "avg_nodes": (total_nodes / graph_count) if graph_count else 0.0,
        "avg_edges": (total_edges / graph_count) if graph_count else 0.0,
    }


def float_to_csv(value):
    if value is None:
        return ""
    if isinstance(value, (np.floating, float)):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.6f}"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    return str(value)


def make_row_key(dataset, method, scenario, alpha, dim, min_count, lambda_val):
    return (
        dataset,
        method,
        scenario,
        float_to_csv(float(alpha)),
        float_to_csv(int(dim)),
        float_to_csv(int(min_count)),
        float_to_csv(float(lambda_val)),
    )


def master_csv_has_header(master_csv_path):
    if not os.path.exists(master_csv_path):
        return False

    with open(master_csv_path, "r", encoding="utf-8", newline="") as csv_file:
        for line in csv_file:
            stripped = line.strip()
            if not stripped:
                continue
            return stripped.split(",") == MASTER_COLUMNS
    return False


def ensure_master_csv_header(master_csv_path):
    if not os.path.exists(master_csv_path):
        return

    if master_csv_has_header(master_csv_path):
        return

    with open(master_csv_path, "r", encoding="utf-8", newline="") as csv_file:
        existing_lines = csv_file.readlines()

    with open(master_csv_path, "w", encoding="utf-8", newline="") as csv_file:
        csv_file.write(",".join(MASTER_COLUMNS) + "\n")
        for line in existing_lines:
            if line.strip():
                csv_file.write(line)


def load_existing_keys(master_csv_path):
    if not os.path.exists(master_csv_path):
        return set()

    ensure_master_csv_header(master_csv_path)
    existing_keys = set()
    with open(master_csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if not row or not row.get("dataset"):
                continue
            existing_keys.add(
                make_row_key(
                    row["dataset"],
                    row["method"],
                    row["scenario"],
                    float(row["alpha"]),
                    int(float(row["dim"])),
                    int(float(row["min_count"])),
                    float(row["lambda_val"]),
                )
            )
    return existing_keys


def append_result_row(master_csv_path, row):
    ensure_master_csv_header(master_csv_path)
    write_header = not os.path.exists(master_csv_path)
    with open(master_csv_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=MASTER_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({column: float_to_csv(row.get(column)) for column in MASTER_COLUMNS})


def resolve_output_dir(base_dir, output_dir):
    if os.path.isabs(output_dir):
        return output_dir
    return os.path.join(base_dir, output_dir)


def resolve_datasets(base_dir, dataset_arg):
    dataset_specs = []
    alias_lookup = {}

    for spec in DATASET_REGISTRY:
        dataset_path = os.path.join(base_dir, *spec["relative_path"])
        if not os.path.isdir(dataset_path):
            continue

        dataset_spec = {"name": spec["name"], "path": dataset_path}
        dataset_specs.append(dataset_spec)

        aliases = {spec["name"], *spec["aliases"]}
        for alias in aliases:
            alias_lookup[alias.lower()] = dataset_spec

    if dataset_arg == "all":
        return dataset_specs

    requested = dataset_arg.lower()
    if requested not in alias_lookup:
        available = sorted(spec["name"] for spec in dataset_specs)
        raise ValueError(f"Unknown dataset '{dataset_arg}'. Available datasets: {available}")

    return [alias_lookup[requested]]


def resolve_method_files(dataset_path):
    method_files = {}
    for method_name, pattern in METHOD_FILE_PATTERNS.items():
        matches = sorted(glob.glob(os.path.join(dataset_path, pattern)))
        if not matches:
            raise FileNotFoundError(
                f"Missing PNML file for method '{method_name}' in {dataset_path}."
            )
        method_files[method_name] = matches[0]
    return method_files


def resolve_methods(method_arg):
    if method_arg == "all":
        return list(METHOD_FILE_PATTERNS.keys())

    normalized = method_arg.lower()
    if normalized not in METHOD_ALIASES:
        available = sorted(METHOD_FILE_PATTERNS.keys())
        raise ValueError(f"Unknown method '{method_arg}'. Available methods: {available}")

    return [METHOD_ALIASES[normalized]]


def load_log(dataset_path):
    xes_file = get_xes_file(dataset_path)
    if not xes_file:
        raise FileNotFoundError(f"No XES file found in {dataset_path}.")

    log = pm4py.read_xes(xes_file)
    if not hasattr(log, "groupby"):
        log = pm4py.convert_to_dataframe(log)
    return log, xes_file


def build_truth_trace_graphs(log, dataset_name=None):
    truth_trace_graphs = []
    total_cases = int(log["case:concept:name"].nunique())
    progress_interval = max(1, total_cases // 10)
    build_start = time.perf_counter()

    for index, (case_id, log_case) in enumerate(log.groupby("case:concept:name"), start=1):
        net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(log_case)
        petri_net_graph = PetriNetGraph(log_case, net, initial_marking, final_marking)
        truth_trace_graphs.extend(petri_net_graph.generate_trace_graphs())

        if index % progress_interval == 0 or index == total_cases:
            elapsed = time.perf_counter() - build_start
            prefix = f"[{dataset_name}] " if dataset_name else ""
            print(
                f"  {prefix}truth graph progress: {index}/{total_cases} cases "
                f"({elapsed:.2f}s elapsed)"
            )

    truth_trace_graphs.sort(key=lambda item: item[0])
    return truth_trace_graphs


def build_trace_graphs(file_path, log, scenario):
    construction_time = 0.0

    t0 = time.perf_counter()
    net, initial_marking, final_marking = pnml_importer.apply(file_path)
    construction_time += time.perf_counter() - t0

    if scenario == "context_aware":
        t0 = time.perf_counter()
        petri_net_graph = PetriNetGraph(log, net, initial_marking, final_marking)
        construction_time += time.perf_counter() - t0

        trace_graphs = []
        for case_id, trace_df in log.groupby("case:concept:name"):
            t0 = time.perf_counter()
            trace_graph = petri_net_graph.create_trace_graph(trace_df)
            construction_time += time.perf_counter() - t0
            trace_graphs.append((case_id, trace_graph))
    else:
        t0 = time.perf_counter()
        petri_net_graph = PetriNetMinedGraph(net, initial_marking, final_marking)
        construction_time += time.perf_counter() - t0

        t0 = time.perf_counter()
        trace_graph = petri_net_graph.get_graph()
        construction_time += time.perf_counter() - t0
        trace_graphs = [("mined_graph", trace_graph)]

    trace_graphs.sort(key=lambda item: item[0])
    return trace_graphs, construction_time


def build_word2vec_model(walks, dim, window, min_count):
    return Word2Vec(
        sentences=walks,
        sg=1,
        vector_size=dim,
        window=window,
        workers=1,
        min_count=min_count,
        hs=0,
        negative=1,
        hashfxn=hash,
        epochs=1000,
        alpha=0.02,
        seed=42,
    )


def compute_dropout_stats(graphs, model):
    graph_nodes = set()
    for _, graph in graphs:
        graph_nodes.update(graph.nodes())

    vocab_nodes = set(model.wv.key_to_index)
    nodes_total = len(graph_nodes)
    nodes_in_vocab = len(graph_nodes & vocab_nodes)
    nodes_dropped = nodes_total - nodes_in_vocab
    nodes_dropped_pct = (100.0 * nodes_dropped / nodes_total) if nodes_total else 0.0

    return {
        "nodes_total": nodes_total,
        "nodes_in_vocab": nodes_in_vocab,
        "nodes_dropped": nodes_dropped,
        "nodes_dropped_pct": nodes_dropped_pct,
    }


def is_empty_vocabulary_error(exception):
    message = str(exception).lower()
    markers = (
        "empty vocabulary",
        "you must first build vocabulary",
        "you must supply at least one word",
        "you must supply at least one sentence",
    )
    return any(marker in message for marker in markers)


def sanitize_float_for_filename(value):
    text = float_to_csv(float(value))
    return text.replace("-", "neg_").replace(".", "p")


def save_pca_visualization(
    output_dir,
    dataset,
    method,
    scenario,
    alpha,
    dim,
    min_count,
    trace_model,
    truth_model,
    rotation_matrix,
):
    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError:
        return

    if trace_model.wv.vectors.shape[0] == 0 or truth_model.wv.vectors.shape[0] == 0:
        return

    combined_embeddings = np.vstack(
        (trace_model.wv.vectors @ rotation_matrix, truth_model.wv.vectors)
    )
    if combined_embeddings.shape[0] < 3 or combined_embeddings.shape[1] < 3:
        return

    pca = PCA(n_components=3)
    reduced_embeddings = pca.fit_transform(combined_embeddings)

    trace_reduced = reduced_embeddings[: len(trace_model.wv.vectors)]
    truth_reduced = reduced_embeddings[len(trace_model.wv.vectors) :]

    trace_labels = trace_model.wv.index_to_key
    truth_labels = truth_model.wv.index_to_key
    trace_label_index = {label: index for index, label in enumerate(trace_labels)}
    truth_label_index = {label: index for index, label in enumerate(truth_labels)}

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        trace_reduced[:, 0],
        trace_reduced[:, 1],
        trace_reduced[:, 2],
        color="blue",
        alpha=0.6,
        label="Trace Embeddings",
    )
    ax.scatter(
        truth_reduced[:, 0],
        truth_reduced[:, 1],
        truth_reduced[:, 2],
        color="red",
        alpha=0.6,
        label="Truth Embeddings",
    )

    for label in set(trace_labels) & set(truth_labels):
        trace_index = trace_label_index[label]
        truth_index = truth_label_index[label]
        ax.plot(
            [trace_reduced[trace_index, 0], truth_reduced[truth_index, 0]],
            [trace_reduced[trace_index, 1], truth_reduced[truth_index, 1]],
            [trace_reduced[trace_index, 2], truth_reduced[truth_index, 2]],
            c="gray",
            linestyle="--",
            linewidth=0.9,
        )

    ax.set_title(f"3D PCA Visualization - {method}")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    ax.legend()

    plot_path = os.path.join(
        output_dir,
        (
            f"pca_visualization_{dataset}_{method}_{scenario}"
            f"_alpha_{sanitize_float_for_filename(alpha)}"
            f"_dim_{dim}_mincount_{min_count}.png"
        ),
    )
    plt.savefig(plot_path)
    plt.close()


def nan_result_row(dataset, method, scenario, alpha, dim, min_count, lambda_val):
    row = {
        "dataset": dataset,
        "method": method,
        "scenario": scenario,
        "alpha": alpha,
        "dim": dim,
        "min_count": min_count,
        "lambda_val": lambda_val,
        "D_weighted": np.nan,
        "matched_nodes": np.nan,
        "unmatched_nodes": np.nan,
        "procrustes_residual_total": np.nan,
        "procrustes_residual_per_anchor": np.nan,
        "anchor_count": np.nan,
        "embedding_dim": dim,
        "ratio": np.nan,
        "gt_nodes_dropped": np.nan,
        "gt_nodes_dropped_pct": np.nan,
        "model_nodes_dropped": np.nan,
        "model_nodes_dropped_pct": np.nan,
        "time_petrinetgraph_construction": np.nan,
        "time_random_walk_generation": np.nan,
        "time_word2vec_training": np.nan,
        "time_procrustes_alignment": np.nan,
        "time_distance_computation": np.nan,
        "time_total": np.nan,
    }
    return row


def run_configuration(
    dataset,
    method,
    scenario,
    file_path,
    log,
    truth_trace_graphs,
    alpha,
    dim,
    min_count,
    lambda_val,
    output_dir,
    disable_pca_plots,
):
    stage_times = {stage_key: 0.0 for stage_key in STAGE_KEYS}

    trace_graphs, stage_times["petrinetgraph_construction"] = build_trace_graphs(
        file_path, log, scenario
    )
    trace_graph_summary = summarize_graphs(trace_graphs)
    print(
        "    Trace graphs: "
        f"{trace_graph_summary['graph_count']} graphs, "
        f"{trace_graph_summary['total_nodes']} nodes, "
        f"{trace_graph_summary['total_edges']} edges"
    )

    labels_mapping, type_mapping = global_encoder(truth_trace_graphs, trace_graphs)
    updated_truth_graphs = update_node_and_edge_encodings(
        truth_trace_graphs, labels_mapping, type_mapping
    )
    updated_trace_graphs = update_node_and_edge_encodings(
        trace_graphs, labels_mapping, type_mapping
    )
    truth_graph_summary = summarize_graphs(updated_truth_graphs)
    encoded_trace_graph_summary = summarize_graphs(updated_trace_graphs)
    walk_length = calculate_max_walk_length(updated_trace_graphs + updated_truth_graphs)
    window = walk_length / 2
    allowed_types = {type_mapping[key] for key in DESIRED_TYPES if key in type_mapping}
    print(
        "    Encoded graphs: "
        f"truth={truth_graph_summary['graph_count']} graphs / "
        f"{truth_graph_summary['total_nodes']} nodes, "
        f"model={encoded_trace_graph_summary['graph_count']} graphs / "
        f"{encoded_trace_graph_summary['total_nodes']} nodes"
    )
    print(
        "    Walk setup: "
        f"walk_length={walk_length}, window={window:.2f}, "
        f"allowed_types={len(allowed_types)}"
    )

    t0 = time.perf_counter()
    trace_walks = generate_walks(
        updated_trace_graphs,
        10000  if scenario == "context_aware" else 50000,
        walk_length,
        allowed_types=allowed_types,
    )
    stage_times["random_walk_generation"] += time.perf_counter() - t0
    print(
        "    Trace walk corpus prepared: "
        f"{len(trace_walks)} walks"
    )

    t0 = time.perf_counter()
    truth_walks = generate_walks(
        updated_truth_graphs,
        10000 if scenario == "context_aware" else 50000,
        walk_length,
        allowed_types=allowed_types,
    )
    stage_times["random_walk_generation"] += time.perf_counter() - t0
    print(
        "    Truth walk corpus prepared: "
        f"{len(truth_walks)} walks"
    )

    try:
        t0 = time.perf_counter()
        print(
            "    Training truth Word2Vec: "
            f"dim={dim}, min_count={min_count}, window={window:.2f}"
        )
        truth_model = build_word2vec_model(truth_walks, dim, window, min_count)
        stage_times["word2vec_training"] += time.perf_counter() - t0
        print(
            "    Truth Word2Vec done: "
            f"vocab={len(truth_model.wv)}, time={stage_times['word2vec_training']:.2f}s cumulative"
        )

        t0 = time.perf_counter()
        print(
            "    Training model Word2Vec: "
            f"dim={dim}, min_count={min_count}, window={window:.2f}"
        )
        trace_model = build_word2vec_model(trace_walks, dim, window, min_count)
        stage_times["word2vec_training"] += time.perf_counter() - t0
        print(
            "    Model Word2Vec done: "
            f"vocab={len(trace_model.wv)}, time={stage_times['word2vec_training']:.2f}s cumulative"
        )
    except Exception as exception:
        if is_empty_vocabulary_error(exception):
            print(
                "    Word2Vec produced an empty vocabulary. "
                f"Returning NaN row for dataset={dataset}, method={method}, scenario={scenario}."
            )
            return nan_result_row(dataset, method, scenario, alpha, dim, min_count, lambda_val)
        raise

    gt_dropout = compute_dropout_stats(updated_truth_graphs, truth_model)
    model_dropout = compute_dropout_stats(updated_trace_graphs, trace_model)
    print(
        "    Dropout stats: "
        f"gt_dropped={gt_dropout['nodes_dropped']}/{gt_dropout['nodes_total']} "
        f"({gt_dropout['nodes_dropped_pct']:.2f}%), "
        f"model_dropped={model_dropout['nodes_dropped']}/{model_dropout['nodes_total']} "
        f"({model_dropout['nodes_dropped_pct']:.2f}%)"
    )

    expected_labels = set(labels_mapping.values())
    shared_labels = sorted(
        expected_labels
        & set(trace_model.wv.index_to_key)
        & set(truth_model.wv.index_to_key),
        key=str,
    )

    anchor_count = len(shared_labels)
    ratio = float("inf") if anchor_count == 0 else dim / anchor_count
    unmatched_nodes = len(expected_labels) - anchor_count
    print(
        "    Alignment inputs: "
        f"expected_labels={len(expected_labels)}, anchors={anchor_count}, "
        f"unmatched={unmatched_nodes}, ratio={ratio}"
    )

    if anchor_count == 0:
        print("    No shared anchors after Word2Vec; returning NaN distance row.")
        row = {
            "dataset": dataset,
            "method": method,
            "scenario": scenario,
            "alpha": alpha,
            "dim": dim,
            "min_count": min_count,
            "lambda_val": lambda_val,
            "D_weighted": np.nan,
            "matched_nodes": 0,
            "unmatched_nodes": unmatched_nodes,
            "procrustes_residual_total": np.nan,
            "procrustes_residual_per_anchor": np.nan,
            "anchor_count": 0,
            "embedding_dim": dim,
            "ratio": ratio,
            "gt_nodes_dropped": gt_dropout["nodes_dropped"],
            "gt_nodes_dropped_pct": gt_dropout["nodes_dropped_pct"],
            "model_nodes_dropped": model_dropout["nodes_dropped"],
            "model_nodes_dropped_pct": model_dropout["nodes_dropped_pct"],
            "time_petrinetgraph_construction": stage_times["petrinetgraph_construction"],
            "time_random_walk_generation": stage_times["random_walk_generation"],
            "time_word2vec_training": stage_times["word2vec_training"],
            "time_procrustes_alignment": stage_times["procrustes_alignment"],
            "time_distance_computation": stage_times["distance_computation"],
            "time_total": sum(stage_times.values()),
        }
        return row

    trace_shared = np.array([trace_model.wv[label] for label in shared_labels])
    truth_shared = np.array([truth_model.wv[label] for label in shared_labels])

    t0 = time.perf_counter()
    rotation_matrix, _ = orthogonal_procrustes(trace_shared, truth_shared)
    aligned_trace_shared = trace_shared @ rotation_matrix
    stage_times["procrustes_alignment"] = time.perf_counter() - t0

    residual_total = float(np.linalg.norm(aligned_trace_shared - truth_shared, "fro"))
    residual_per_anchor = residual_total / anchor_count
    print(
        "    Procrustes: "
        f"residual_total={residual_total:.6f}, "
        f"residual_per_anchor={residual_per_anchor:.6f}, "
        f"time={stage_times['procrustes_alignment']:.4f}s"
    )
    aligned_trace_vectors = {
        label: vector for label, vector in zip(shared_labels, aligned_trace_shared)
    }

    t0 = time.perf_counter()
    distance_result = compare_vectors_with_penalty(
        trace_model=trace_model,
        truth_model=truth_model,
        expected_labels=expected_labels,
        penalty_factor=alpha,
        lambda_values=[lambda_val],
        aligned_trace_vectors=aligned_trace_vectors,
        shared_labels=shared_labels,
    )[0]
    stage_times["distance_computation"] = time.perf_counter() - t0
    print(
        "    Distance: "
        f"D_weighted={distance_result['weighted_distance']:.6f}, "
        f"matched={distance_result['shared_count']}, "
        f"missing={distance_result['missing_count']}, "
        f"time={stage_times['distance_computation']:.4f}s"
    )

    if not disable_pca_plots:
        save_pca_visualization(
            output_dir,
            dataset,
            method,
            scenario,
            alpha,
            dim,
            min_count,
            trace_model,
            truth_model,
            rotation_matrix,
        )

    row = {
        "dataset": dataset,
        "method": method,
        "scenario": scenario,
        "alpha": alpha,
        "dim": dim,
        "min_count": min_count,
        "lambda_val": lambda_val,
        "D_weighted": distance_result["weighted_distance"],
        "matched_nodes": distance_result["shared_count"],
        "unmatched_nodes": distance_result["missing_count"],
        "procrustes_residual_total": residual_total,
        "procrustes_residual_per_anchor": residual_per_anchor,
        "anchor_count": anchor_count,
        "embedding_dim": dim,
        "ratio": ratio,
        "gt_nodes_dropped": gt_dropout["nodes_dropped"],
        "gt_nodes_dropped_pct": gt_dropout["nodes_dropped_pct"],
        "model_nodes_dropped": model_dropout["nodes_dropped"],
        "model_nodes_dropped_pct": model_dropout["nodes_dropped_pct"],
        "time_petrinetgraph_construction": stage_times["petrinetgraph_construction"],
        "time_random_walk_generation": stage_times["random_walk_generation"],
        "time_word2vec_training": stage_times["word2vec_training"],
        "time_procrustes_alignment": stage_times["procrustes_alignment"],
        "time_distance_computation": stage_times["distance_computation"],
        "time_total": sum(stage_times.values()),
    }
    print(
        "    Stage times: "
        f"construct={stage_times['petrinetgraph_construction']:.2f}s, "
        f"walks={stage_times['random_walk_generation']:.2f}s, "
        f"w2v={stage_times['word2vec_training']:.2f}s, "
        f"proc={stage_times['procrustes_alignment']:.2f}s, "
        f"dist={stage_times['distance_computation']:.2f}s, "
        f"total={row['time_total']:.2f}s"
    )
    return row


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy_mode", nargs="?", choices=["case", "mined"], help=argparse.SUPPRESS)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--min_count", type=int, default=200)
    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--method", type=str, default="all")
    parser.add_argument(
        "--scenario",
        choices=["static", "context_aware", "both"],
        default="both",
    )
    parser.add_argument("--lambda_val", type=float, default=float("inf"))
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--disable_pca_plots", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.legacy_mode:
        args.scenario = "context_aware" if args.legacy_mode == "case" else "static"

    overall_start = time.perf_counter()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = resolve_output_dir(base_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    master_csv_path = os.path.join(output_dir, "results_master.csv")
    existing_keys = load_existing_keys(master_csv_path)

    dataset_specs = resolve_datasets(base_dir, args.dataset)
    selected_methods = resolve_methods(args.method)
    selected_scenarios = (
        ["static", "context_aware"] if args.scenario == "both" else [args.scenario]
    )

    print(
        "Running configurations with "
        f"alpha={args.alpha}, dim={args.dim}, min_count={args.min_count}, "
        f"lambda_val={float_to_csv(args.lambda_val)}, "
        f"disable_pca_plots={args.disable_pca_plots}"
    )

    for dataset_spec in dataset_specs:
        pending_runs = []
        method_files = resolve_method_files(dataset_spec["path"])
        for method in selected_methods:
            for scenario in selected_scenarios:
                key = make_row_key(
                    dataset_spec["name"],
                    method,
                    scenario,
                    args.alpha,
                    args.dim,
                    args.min_count,
                    args.lambda_val,
                )
                if key in existing_keys:
                    print(
                        "Skipping existing row: "
                        f"dataset={dataset_spec['name']} method={method} scenario={scenario}"
                    )
                    continue
                pending_runs.append((method, scenario, method_files[method]))

        if not pending_runs:
            continue

        dataset_start = time.perf_counter()
        print(f"Processing dataset: {dataset_spec['name']}")
        log_load_start = time.perf_counter()
        log, xes_file = load_log(dataset_spec["path"])
        log_load_time = time.perf_counter() - log_load_start
        log_summary = summarize_log(log)
        print(
            "  XES loaded: "
            f"path={xes_file}, time={log_load_time:.2f}s"
        )
        print(
            "  Log summary: "
            f"cases={log_summary['cases']}, events={log_summary['events']}, "
            f"activities={log_summary['activities']}, "
            f"trace_len[min/mean/max]={log_summary['trace_len_min']}/"
            f"{format_number(log_summary['trace_len_mean'])}/"
            f"{log_summary['trace_len_max']}"
        )
        truth_build_start = time.perf_counter()
        truth_trace_graphs = build_truth_trace_graphs(log, dataset_name=dataset_spec["name"])
        truth_build_time = time.perf_counter() - truth_build_start
        truth_summary = summarize_graphs(truth_trace_graphs)
        print(
            "  Truth graphs ready: "
            f"{truth_summary['graph_count']} graphs, "
            f"{truth_summary['total_nodes']} nodes, "
            f"{truth_summary['total_edges']} edges, "
            f"time={truth_build_time:.2f}s"
        )

        for method, scenario, file_path in pending_runs:
            print(
                f"  Running dataset={dataset_spec['name']} method={method} "
                f"scenario={scenario}"
            )
            print(f"    PNML path: {file_path}")
            row = run_configuration(
                dataset=dataset_spec["name"],
                method=method,
                scenario=scenario,
                file_path=file_path,
                log=log,
                truth_trace_graphs=truth_trace_graphs,
                alpha=args.alpha,
                dim=args.dim,
                min_count=args.min_count,
                lambda_val=args.lambda_val,
                output_dir=output_dir,
                disable_pca_plots=args.disable_pca_plots,
            )
            append_result_row(master_csv_path, row)
            existing_keys.add(
                make_row_key(
                    row["dataset"],
                    row["method"],
                    row["scenario"],
                    row["alpha"],
                    row["dim"],
                    row["min_count"],
                    row["lambda_val"],
                )
            )

        dataset_time = time.perf_counter() - dataset_start
        print(
            f"Completed dataset {dataset_spec['name']} in {format_time(dataset_time)}"
        )

    overall_time = time.perf_counter() - overall_start
    print(f"Finished in {format_time(overall_time)}")
    print(f"Master results written to {master_csv_path}")


if __name__ == "__main__":
    main()
