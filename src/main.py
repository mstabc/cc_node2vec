import os
import glob
import pm4py
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from petri_builder import PetriNetGraph, PetriNetMinedGraph
from pm4py.objects.petri_net.importer import importer as pnml_importer
from graph_encoding import global_encoder, update_node_and_edge_encodings
from random_walks import calculate_max_walk_length, generate_walks
from datetime import datetime
from random_walks import calculate_max_walk_length
from pm4py.objects.petri_net.importer import importer as pnml_importer
from utils import get_dataset_folders, get_xes_file, get_pnml_files
from scipy.spatial.distance import cdist
from gensim.models import Word2Vec
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.log.importer.xes.variants import iterparse, line_by_line, iterparse_mem_compressed, iterparse_20, chunk_regex, rustxes
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.streaming.importer.xes.variants import xes_trace_stream, xes_event_stream
from distance_cal import compare_vectors_with_penalty 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
# imploggingort 
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
import numpy as np
from pm4py.objects.log import obj, exporter, importer, util


def main(mode):
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  
    DATA_DIR = os.path.join(BASE_DIR, "data")  
    RESULTS_DIR = os.path.join(BASE_DIR, "results" if mode == "case" else "mined_results")
    # logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    dataset_folders = get_dataset_folders(DATA_DIR)
    distance_metrics = ['cosine', 'euclidean']
    desired_types = {'sink', 'source', 'transition'}
    for dataset in dataset_folders:
        
        # logging.info(f"Processing dataset: {dataset}")
        print(f"Processing dataset: {dataset}")
        dataset_path = os.path.join(DATA_DIR, dataset)


        xes_file = get_xes_file(dataset_path)
        pnml_files = get_pnml_files(dataset_path)
        log = pm4py.read_xes(xes_file)


        truth_trace_graphs = []
        unique_cases = log['case:concept:name'].unique()

        for case in unique_cases:
            log_case = log[log['case:concept:name'] == case]
            
            net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(log_case)
            petri_net_graph = PetriNetGraph(log_case, net, initial_marking, final_marking)
            truth_graph = petri_net_graph.generate_trace_graphs()
            

            truth_trace_graphs.append(truth_graph)
            
        
        truth_trace_graphs1 = []
        for sublist in truth_trace_graphs:
            truth_trace_graphs1.extend(sublist)

        truth_trace_graphs = truth_trace_graphs1

        truth_trace_graphs = sorted(truth_trace_graphs, key=lambda x: x[0])
        
        
        results = []
        all_results = []
        temp = 0
        for file_path in pnml_files:
            
            method_name = os.path.basename(file_path).replace(".pnml", "")
            
            # logging.info(f"Processing Petri net: {method_name}")
            print(f"Processing Petri net: {method_name}")
            net, initial_marking, final_marking = pnml_importer.apply(file_path)
            petri_net_graph = PetriNetGraph(log, net, initial_marking, final_marking)
            if mode == "case":
                petri_net_graph = PetriNetGraph(log, net, initial_marking, final_marking)
                trace_graphs = petri_net_graph.generate_trace_graphs()
            else:
                petri_net_graph1 = PetriNetMinedGraph(net, initial_marking, final_marking)
                trace_graphs = petri_net_graph1.get_graph()
                trace_graphs = [("mined_graph", trace_graphs)]
            
            trace_graphs = sorted(trace_graphs, key=lambda x: x[0])
            if temp == 0:
                labels_mapping, type_mapping = global_encoder(truth_trace_graphs, trace_graphs)
                updated_truth_graphs = update_node_and_edge_encodings(truth_trace_graphs, labels_mapping, type_mapping)
                updated_trace_graphs = update_node_and_edge_encodings(trace_graphs, labels_mapping, type_mapping)
                walk_length = calculate_max_walk_length(updated_trace_graphs + updated_truth_graphs)
            else:
                updated_trace_graphs = update_node_and_edge_encodings(trace_graphs, labels_mapping, type_mapping)
            
            walk_count = 50000 if mode == "case" else 600000

            

            
            trace_walks = generate_walks(updated_trace_graphs, walk_count, walk_length * 2, type_mapping)
            if temp == 0:
                truth_walks = generate_walks(updated_truth_graphs, 100000, walk_length * 2, type_mapping)


            
            allowed_types = {type_mapping[key] for key in desired_types if key in type_mapping}
            trace_walks = [[node for node, node_type in walk if node_type in allowed_types] for walk in trace_walks]
            if temp == 0:
                    truth_walks = [[node for node, node_type in walk if node_type in allowed_types] for walk in truth_walks]
            windows = [4, 5, 6, 7, 8, 3]
            penalties = [0.5, 1, 1.5, 2.5, 2]
            
            for window in windows:
                if temp == 0:
                    truth_node2vec = Word2Vec(truth_walks,
                                sg=1,  
                                vector_size=8,
                                window= window,
                                workers=24,
                                min_count=0,
                                hs=0 , 
                                negative = 5,
                                hashfxn=hash, epochs=500,
                                alpha = 0.005,
                                seed=42)
                    truth_vectors = truth_node2vec.wv.vectors
                trace_node2vec = Word2Vec(trace_walks,
                        sg=1,
                        vector_size=8,
                        window=window,
                        workers=24,
                        min_count=0,
                        hs=0,
                        negative = 5,
                        hashfxn=hash, epochs=500,
                        alpha = 0.001,
                        seed=42)
                
                expected_labels = set(labels_mapping.values())
                
                vector_size = trace_node2vec.vector_size
                expected_labels = set(labels_mapping.values())

                for penalty in penalties:
                    results = compare_vectors_with_penalty(
                        trace_model=trace_node2vec, 
                        truth_model=truth_node2vec,
                        expected_labels=expected_labels,
                        penalty_factor= penalty
                        )
                    for result in results:
                        result.update({
                            "method": method_name,
                            "window": window,
                            "penalty_factor": penalty
                        })  

                    all_results.extend(results)
            trace_embeddings = trace_node2vec.wv.vectors
            truth_embeddings = truth_node2vec.wv.vectors
            combined_embeddings = np.vstack((trace_embeddings, truth_embeddings))

            pca = PCA(n_components=3)
            reduced_embeddings = pca.fit_transform(combined_embeddings)

            trace_reduced = reduced_embeddings[:len(trace_embeddings)]
            truth_reduced = reduced_embeddings[len(trace_embeddings):]

            trace_labels = trace_node2vec.wv.index_to_key
            truth_labels = truth_node2vec.wv.index_to_key

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(trace_reduced[:, 0], trace_reduced[:, 1], trace_reduced[:, 2], 
                    color='blue', alpha=0.6, label="Trace Embeddings")
            ax.scatter(truth_reduced[:, 0], truth_reduced[:, 1], truth_reduced[:, 2], 
                    color='red', alpha=0.6, label="Truth Embeddings")

            for i in range(min(len(trace_embeddings), len(truth_embeddings))):
                ax.plot([trace_reduced[i, 0], truth_reduced[i, 0]], 
                        [trace_reduced[i, 1], truth_reduced[i, 1]], 
                        [trace_reduced[i, 2], truth_reduced[i, 2]], 
                        c='gray', linestyle='--', linewidth=0.9)

            ax.set_title(f"3D PCA Visualization - {method_name}")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_zlabel("Component 3")
            ax.legend()
            plt_path = os.path.join(RESULTS_DIR, f"pca_visualization_{dataset}_{method_name}.png")
            plt.savefig(plt_path)
            plt.close()
            print(f"Saved PCA visualization: {plt_path}")

        
            temp = 1
        df = pd.DataFrame(all_results)
        result_filename = os.path.join(RESULTS_DIR, f"average_distance_{dataset}.csv")
        df.to_csv(result_filename, index=False)
        print(f"Results saved for dataset {dataset} in {result_filename}")
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["case", "mined"], help="Choose between 'case' and 'mined' modes")
    args = parser.parse_args()
    main(args.mode)
