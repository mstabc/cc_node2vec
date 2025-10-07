import networkx as nx
import copy
import os
import glob
from collections import defaultdict, OrderedDict
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from scipy.spatial.distance import cdist
from scipy.stats import t
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import minimize

def entropy_regularized_distance(z, d, λ):
    z = np.clip(z, 1e-10, 1)  # Avoid log(0)
    return np.sum(z * d) + λ * np.sum(z * np.log(z))

def compare_vectors_with_penalty(trace_model, truth_model, expected_labels, penalty_factor, lambda_values=None):
    if lambda_values is None:
        lambda_values = [0, 0.5, 1, 10, 'inf']

    results = []
    distance_metrics = ['cosine']

    trace_labels = set(trace_model.wv.index_to_key)
    truth_labels = set(truth_model.wv.index_to_key)
    shared_labels = list(expected_labels & trace_labels & truth_labels)

    if not shared_labels:
        raise ValueError("No shared labels found between trace and truth models.")

    trace_shared = np.array([trace_model.wv[label] for label in shared_labels])
    truth_shared = np.array([truth_model.wv[label] for label in shared_labels])
    R, _ = orthogonal_procrustes(trace_shared, truth_shared)

    for metric in distance_metrics:
        # --- Pass 1: compute distances for labels that exist in BOTH models ---
        shared_dists = []
        present_mask = []   # True if label present in both; False if missing in at least one
        for label in expected_labels:
            if label in trace_labels and label in truth_labels:
                trace_vec = trace_model.wv[label] @ R  # align trace vector
                truth_vec = truth_model.wv[label]

                dist = cdist(
                    np.asarray(trace_vec, dtype=float).reshape(1, -1),
                    np.asarray(truth_vec, dtype=float).reshape(1, -1),
                    metric=metric
                )[0, 0]

                # guard against NaN (can happen with zero vectors for cosine)
                if not np.isfinite(dist):
                    dist = 0.0

                shared_dists.append(dist)
                present_mask.append(True)
            else:
                present_mask.append(False)

        # Determine the max among the actually observed distances
        if len(shared_dists) > 0:
            max_shared = float(np.max(shared_dists))
            # If all shared distances are 0 (identical), avoid a degenerate penalty base
            if max_shared == 0.0:
                max_shared = 1.0
        else:
            # Fallback if nothing shared for this metric:
            # use a conservative baseline rather than a theoretical cap
            # (you can change this to 1.0 or any small positive constant)
            max_shared = 1.0

        # --- Pass 2: build d_v by inserting penalties for missing labels based on max_shared ---
        d_v = []
        i_shared = 0
        for is_present in present_mask:
            if is_present:
                d_v.append(shared_dists[i_shared])
                i_shared += 1
            else:
                # Penalty is proportional to what we actually observed for this metric
                d_v.append(max_shared * penalty_factor)

        d_v = np.asarray(d_v, dtype=float)
        n = len(d_v)

        # If you still need a "max_distance" for normalization/ reporting,
        # use the ACTUAL maximum in d_v (which now includes penalty inserts).
        max_distance = float(np.max(d_v)) if n > 0 else 1.0

        for λ in lambda_values:
            if λ == 'inf':
                z_uniform = np.full(n, 1.0 / n)
                weighted_distance = np.sum(z_uniform * d_v)
                entropy = -np.sum(z_uniform * np.log(z_uniform))
                weights = z_uniform
            else:
                z0 = np.full(n, 1.0 / n)
                constraints = {'type': 'eq', 'fun': lambda z: np.sum(z) - 1}
                bounds = [(0, 1)] * n

                result = minimize(entropy_regularized_distance, z0,
                                  args=(d_v, λ),
                                  method='SLSQP',
                                  bounds=bounds,
                                  constraints=constraints)

                if not result.success:
                    raise ValueError(f"Optimization failed for λ={λ}: {result.message}")

                weights = result.x
                weighted_distance = np.sum(weights * d_v)
                entropy = -np.sum(weights * np.log(np.clip(weights, 1e-10, 1)))

            results.append({
                "metric": metric,
                "lambda": λ,
                "weighted_distance": weighted_distance,
                "normalized_distance": weighted_distance / (penalty_factor * 2) if penalty_factor != 0 else weighted_distance / 2,
                "entropy": entropy,
                "weights": weights,
                "shared_count": len(shared_labels),
                "missing_count": len(expected_labels) - len(shared_labels)
            })

    return results

# def compare_vectors_with_penalty(trace_model, truth_model, expected_labels, penalty_factor):
#     results = []
#     distance_metrics = ['cosine', 'euclidean']

#     trace_labels = set(trace_model.wv.index_to_key)
#     truth_labels = set(truth_model.wv.index_to_key)
#     shared_labels = list(expected_labels & trace_labels & truth_labels)

#     if not shared_labels:
#         raise ValueError("No shared labels found between trace and truth models.")

#     # Step 1: Align trace embeddings to truth space using shared labels
#     trace_shared = np.array([trace_model.wv[label] for label in shared_labels])
#     truth_shared = np.array([truth_model.wv[label] for label in shared_labels])
#     R, _ = orthogonal_procrustes(trace_shared, truth_shared)

#     for metric in distance_metrics:
#         distances = []

#         for label in expected_labels:
#             if label in trace_labels and label in truth_labels:
#                 trace_vec = trace_model.wv[label] @ R  # align trace vector
#                 truth_vec = truth_model.wv[label]
#                 dist = cdist(trace_vec.reshape(1, -1), truth_vec.reshape(1, -1), metric=metric)[0][0]
#                 distances.append(dist)
#             else:
#                 max_distance = 2 if metric == 'cosine' else np.sqrt(2)
#                 distances.append(max_distance * penalty_factor)

#         distances = np.array(distances)
#         average_distance = np.mean(distances)
#         std_dev = np.std(distances)
#         n = len(distances)
#         conf_interval = t.ppf(0.975, n - 1) * std_dev / np.sqrt(n) if n > 1 else 0

#         results.append({
#             "metric": metric,
#             "average_distance": average_distance,
#             "std_dev": std_dev,
#             "confidence_interval": conf_interval,
#             "penalty_factor": penalty_factor,
#             "compared_count": len(shared_labels),
#             "missing_count": len(expected_labels) - len(shared_labels),
#             "missing_in_trace": len(expected_labels - trace_labels),
#             "missing_in_truth": len(expected_labels - truth_labels)
#         })

#     return results
