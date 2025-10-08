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
from scipy.linalg import orthogonal_procrustes
from datetime import datetime
import time


# imploggingort 
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
import numpy as np
from pm4py.objects.log import obj, exporter, importer, util
def format_time(seconds):
    """Format time in a readable format"""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.2f} minutes"
    else:
        return f"{seconds/3600:.2f} hours"

def main(mode):
    overall_start = time.time()
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  
    DATA_DIR = os.path.join(BASE_DIR, "data")  
    RESULTS_DIR = os.path.join(BASE_DIR, "results" if mode == "case" else "mined_results")
    # logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    dataset_folders = get_dataset_folders(DATA_DIR)
    distance_metrics = ['cosine', 'euclidean']
    desired_types = {'sink', 'source', 'transition'}
    init_start = time.time()
    init_time = time.time() - init_start
    print(f"Initialization completed in {format_time(init_time)}")
    
    timing_results = []
    for dataset in dataset_folders:
        dataset_start = time.time()
        # logging.info(f"Processing dataset: {dataset}")
        print(f"Processing dataset: {dataset}")
        dataset_path = os.path.join(DATA_DIR, dataset)
        file_load_start = time.time()

        xes_file = get_xes_file(dataset_path)
        pnml_files = get_pnml_files(dataset_path)
        log = pm4py.read_xes(xes_file)
        file_load_time = time.time() - file_load_start
        print(f"  File loading completed in {format_time(file_load_time)}")
        truth_gen_start = time.time()
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
        
        truth_gen_time = time.time() - truth_gen_start
        print(f"  Truth graph generation completed in {format_time(truth_gen_time)}")
        
        results = []
        all_results = []
        temp = 0
        pnml_processing_times = []
        for file_path in pnml_files:
            
            method_name = os.path.basename(file_path).replace(".pnml", "")
            pnml_start = time.time()
            # logging.info(f"Processing Petri net: {method_name}")
            print(f"Processing Petri net: {method_name}")
            net_gen_start = time.time()
            
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
            net_gen_time = time.time() - net_gen_start
            print(f"    trace graphs generation: {format_time(net_gen_time)}")
            encoding_start = time.time()
            if temp == 0:
                labels_mapping, type_mapping = global_encoder(truth_trace_graphs, trace_graphs)
                updated_truth_graphs = update_node_and_edge_encodings(truth_trace_graphs, labels_mapping, type_mapping)
                updated_trace_graphs = update_node_and_edge_encodings(trace_graphs, labels_mapping, type_mapping)
                walk_length = calculate_max_walk_length(updated_trace_graphs + updated_truth_graphs)
                print(f"    Max walk length: {walk_length}")
            else:
                updated_trace_graphs = update_node_and_edge_encodings(trace_graphs, labels_mapping, type_mapping)
            encoding_time = time.time() - encoding_start
            print(f"    Encoding: {format_time(encoding_time)}")
            
            
            walk_count = 10000 if mode == "case" else 50000
            walk_gen_start = time.time()
            

            
            trace_walks = generate_walks(updated_trace_graphs, walk_count, walk_length, type_mapping)
            if temp == 0:
                truth_walks = generate_walks(updated_truth_graphs, walk_count, walk_length, type_mapping)


            
            allowed_types = {type_mapping[key] for key in desired_types if key in type_mapping}
            trace_walks = [[node for node, node_type in walk if node_type in allowed_types] for walk in trace_walks]
            if temp == 0:
                    truth_walks = [[node for node, node_type in walk if node_type in allowed_types] for walk in truth_walks]
                    
            walk_gen_time = time.time() - walk_gen_start
            print(f"    Walk generation: {format_time(walk_gen_time)}")

            windows = [walk_length / 2]
            penalties = [1]
            
            w2v_start = time.time()
            for window in windows:
                if temp == 0:
                    truth_node2vec = Word2Vec(truth_walks,
                                sg=1,  
                                vector_size=128,
                                window= window,
                                workers=1,
                                min_count=200,
                                hs=0 , 
                                negative = 1,
                                hashfxn=hash, epochs=1000,
                                alpha = 0.02,
                                seed=42)
                    truth_vectors = truth_node2vec.wv.vectors
                trace_node2vec = Word2Vec(trace_walks,
                        sg=1,
                        vector_size=128,
                        window=window,
                        workers=1,
                        min_count=200,
                        hs=0,
                        negative = 1,
                        hashfxn=hash, epochs=1000,
                        alpha = 0.02,
                        seed=42)
                w2v_time = time.time() - w2v_start
                print(f"    Word2Vec training: {format_time(w2v_time)}")
                comparison_start = time.time()
                
                
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
                comparison_time = time.time() - comparison_start
                print(f"    Distance comparison: {format_time(comparison_time)}")    
            
            shared_labels = list(set(trace_node2vec.wv.index_to_key) & set(truth_node2vec.wv.index_to_key))
            

            trace_shared = np.array([trace_node2vec.wv[label] for label in shared_labels])
            truth_shared = np.array([truth_node2vec.wv[label] for label in shared_labels])
            R, _ = orthogonal_procrustes(trace_shared, truth_shared)

            trace_embeddings = trace_node2vec.wv.vectors @ R
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

            # Draw lines between matching labels
            trace_label_index = {label: i for i, label in enumerate(trace_labels)}
            truth_label_index = {label: i for i, label in enumerate(truth_labels)}
            for label in set(trace_labels) & set(truth_labels):
                i = trace_label_index[label]
                j = truth_label_index[label]
                ax.plot([trace_reduced[i, 0], truth_reduced[j, 0]], 
                        [trace_reduced[i, 1], truth_reduced[j, 1]], 
                        [trace_reduced[i, 2], truth_reduced[j, 2]], 
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
        results_save_start = time.time()
        df = pd.DataFrame(all_results)
        result_filename = os.path.join(RESULTS_DIR, f"average_distance_{dataset}.csv")
        df.to_csv(result_filename, index=False)
        
        # Save timing results for this dataset
        timing_df = pd.DataFrame(pnml_processing_times)
        timing_filename = os.path.join(RESULTS_DIR, f"timing_results_{dataset}.csv")
        timing_df.to_csv(timing_filename, index=False)
        results_save_time = time.time() - results_save_start
        
        dataset_total_time = time.time() - dataset_start
        
        # Store dataset timing summary
        timing_results.append({
            'dataset': dataset,
            'file_loading': file_load_time,
            'truth_generation': truth_gen_time,
            'pnml_processing': sum(t['total'] for t in pnml_processing_times),
            'results_saving': results_save_time,
            'total': dataset_total_time
        })
        print(f"Results saved for dataset {dataset} in {result_filename}")
        print(f"Timing results saved in {timing_filename}")
        print(f"Dataset {dataset} completed in {format_time(dataset_total_time)}")
        print("-" * 50)
    overall_timing_df = pd.DataFrame(timing_results)
    overall_timing_filename = os.path.join(RESULTS_DIR, "overall_timing_summary.csv")
    overall_timing_df.to_csv(overall_timing_filename, index=False)
    
    overall_time = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"OVERALL EXECUTION COMPLETED IN {format_time(overall_time)}")
    print(f"Overall timing summary saved in {overall_timing_filename}")
    print(f"{'='*60}")
    
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["case", "mined"], help="Choose between 'case' and 'mined' modes")
    args = parser.parse_args()
    main(args.mode)
