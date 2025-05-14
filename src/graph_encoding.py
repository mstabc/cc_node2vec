import networkx as nx
import random
import numpy as np


def global_encoder(trace_graphs1, trace_graphs2):
    combined_trace_graphs = trace_graphs1 + trace_graphs2
    all_node_labels = []
    all_node_types = []

    
    for _, graph in combined_trace_graphs:
        for node, attributes in graph.nodes(data=True):
            
            if 'label' in attributes and attributes['label'] is not None:
                all_node_labels.append(attributes['label'])
            all_node_types.append(attributes['type'])

    unique_labels = list(set(all_node_labels))
    unique_types = list(set(all_node_types))

    labels_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    type_mapping = {typ: idx for idx, typ in enumerate(unique_types)}

    return labels_mapping, type_mapping


def update_node_and_edge_encodings(graphs, labels_mapping, type_mapping):
    updated_graphs = []

    for case_name, graph in graphs:
        new_graph = nx.DiGraph()
        new_graph.graph['label'] = graph.graph.get('label', 0)
        
        local_auxiliary_mapping = {}
        auxiliary_counter = max(labels_mapping.values(), default=0) + 1

        for node, attributes in graph.nodes(data=True):
            node_label = attributes.get('label')
            node_type = attributes.get('type')

            if node_label in labels_mapping:
                encoded_label = labels_mapping[node_label]
            else:
                if node not in local_auxiliary_mapping:

                    local_auxiliary_mapping[node] = auxiliary_counter
                    auxiliary_counter += 1

                encoded_label = local_auxiliary_mapping[node]

            encoded_type = type_mapping.get(node_type, -1)
            new_graph.add_node(encoded_label, type=encoded_type)

        for src, dst, attributes in graph.edges(data=True):
            src_encoded = labels_mapping.get(graph.nodes[src]['label'], local_auxiliary_mapping.get(src))
            dst_encoded = labels_mapping.get(graph.nodes[dst]['label'], local_auxiliary_mapping.get(dst))
            
            if src_encoded is not None and dst_encoded is not None:
                new_graph.add_edge(src_encoded, dst_encoded, weight=attributes.get('weight', 1.0))

        updated_graphs.append((case_name, new_graph))
    
    return updated_graphs
