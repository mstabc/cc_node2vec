import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def entropy_regularized_distance(z, d, lambda_val):
    z = np.clip(z, 1e-10, 1)
    return np.sum(z * d) + lambda_val * np.sum(z * np.log(z))


def compare_vectors_with_penalty(
    trace_model,
    truth_model,
    expected_labels,
    penalty_factor,
    lambda_values=None,
    rotation_matrix=None,
    aligned_trace_vectors=None,
    shared_labels=None,
):
    if lambda_values is None:
        lambda_values = [0.0, 0.5, 1.0, 10.0, float("inf")]

    results = []
    distance_metrics = ["cosine"]

    trace_labels = set(trace_model.wv.index_to_key)
    truth_labels = set(truth_model.wv.index_to_key)

    if shared_labels is None:
        shared_labels = sorted(expected_labels & trace_labels & truth_labels, key=str)
    else:
        shared_labels = list(shared_labels)

    if not shared_labels:
        raise ValueError("No shared labels found between trace and truth models.")

    if aligned_trace_vectors is None:
        if rotation_matrix is None:
            trace_shared = np.array([trace_model.wv[label] for label in shared_labels])
            truth_shared = np.array([truth_model.wv[label] for label in shared_labels])
            rotation_matrix, _ = orthogonal_procrustes(trace_shared, truth_shared)

        aligned_trace_vectors = {
            label: np.asarray(trace_model.wv[label] @ rotation_matrix, dtype=float)
            for label in shared_labels
        }
    else:
        aligned_trace_vectors = {
            label: np.asarray(vector, dtype=float)
            for label, vector in aligned_trace_vectors.items()
        }

    for metric in distance_metrics:
        shared_dists = []
        present_mask = []
        for label in expected_labels:
            if label in trace_labels and label in truth_labels:
                trace_vec = aligned_trace_vectors[label]
                truth_vec = np.asarray(truth_model.wv[label], dtype=float)

                dist = cdist(
                    trace_vec.reshape(1, -1),
                    truth_vec.reshape(1, -1),
                    metric=metric,
                )[0, 0]

                if not np.isfinite(dist):
                    dist = 0.0

                shared_dists.append(dist)
                present_mask.append(True)
            else:
                present_mask.append(False)

        if shared_dists:
            max_shared = float(np.max(shared_dists))
            if max_shared == 0.0:
                max_shared = 1.0
        else:
            max_shared = 1.0

        d_v = []
        shared_index = 0
        for is_present in present_mask:
            if is_present:
                d_v.append(shared_dists[shared_index])
                shared_index += 1
            else:
                d_v.append(max_shared * penalty_factor)

        d_v = np.asarray(d_v, dtype=float)
        n = len(d_v)

        for lambda_val in lambda_values:
            if np.isinf(lambda_val):
                weights = np.full(n, 1.0 / n)
                weighted_distance = np.sum(weights * d_v)
                entropy = -np.sum(weights * np.log(weights))
            else:
                z0 = np.full(n, 1.0 / n)
                constraints = {"type": "eq", "fun": lambda z: np.sum(z) - 1}
                bounds = [(0, 1)] * n

                result = minimize(
                    entropy_regularized_distance,
                    z0,
                    args=(d_v, float(lambda_val)),
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                )

                if not result.success:
                    raise ValueError(
                        f"Optimization failed for lambda={lambda_val}: {result.message}"
                    )

                weights = result.x
                weighted_distance = np.sum(weights * d_v)
                entropy = -np.sum(weights * np.log(np.clip(weights, 1e-10, 1)))

            results.append(
                {
                    "metric": metric,
                    "lambda": lambda_val,
                    "weighted_distance": weighted_distance,
                    "normalized_distance": (
                        weighted_distance / (penalty_factor * 2)
                        if penalty_factor != 0
                        else weighted_distance / 2
                    ),
                    "entropy": entropy,
                    "weights": weights,
                    "shared_count": len(shared_labels),
                    "missing_count": len(expected_labels) - len(shared_labels),
                }
            )

    return results
