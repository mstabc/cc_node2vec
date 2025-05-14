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

def compare_vectors_with_penalty(trace_model, truth_model, expected_labels, penalty_factor):
    results = []
    
    distance_metrics = ['cosine', 'euclidean']
    
    for metric in distance_metrics:
        distances = []
        total_distance = 0
        compared_count = 0
        missing_count = 0
        trace_labels = set(trace_model.wv.index_to_key)
        truth_labels = set(truth_model.wv.index_to_key)
        

        
        missing_in_trace = expected_labels - trace_labels
        missing_in_truth = expected_labels - truth_labels

        max_distance = 2 if metric == 'cosine' else np.sqrt(2)
        
        for label in expected_labels:
            
            if label in trace_labels and label in truth_labels:
                trace_vector = trace_model.wv[label].reshape(1, -1)  
                truth_vector = truth_model.wv[label].reshape(1, -1)
            

                distance = cdist(trace_vector, truth_vector, metric=metric)[0][0]
                distances.append(distance)
            else:
                distances.append(max_distance * penalty_factor)
        
        distances = np.array(distances)
        average_distance = np.mean(distances)
        std_dev = np.std(distances)
        n = len(distances)
        conf_interval = t.ppf(0.975, n - 1) * std_dev / np.sqrt(n) if n > 1 else 0

        results.append({
            "metric": metric,
            "average_distance": average_distance,
            "std_dev": std_dev,
            "confidence_interval": conf_interval,
            "penalty_factor": penalty_factor,
            "compared_count": sum(label in trace_labels and label in truth_labels for label in expected_labels),
            "missing_count": sum(label not in trace_labels or label not in truth_labels for label in expected_labels),
            "missing_in_trace": len(missing_in_trace),
            "missing_in_truth": len(missing_in_truth)
        })
    
    return results