import random
import networkx as nx
import numpy as np
from graph_encoding import global_encoder

def random_walk(graph, start_node, walk_length):
    """
    Perform a directed random walk on the graph starting from the given node.

    """
    walk = []
    current_node = start_node

    for _ in range(walk_length):
        current_type = graph.nodes[current_node].get('type')
        walk.append((current_node, current_type))


        neighbors = list(graph.successors(current_node))
        # print(neighbors)
        if not neighbors:
            # print("No outgoing edges")
            break 
        
        current_node = random.choice(neighbors)

    return walk

def generate_walks(graphs, num_walks, walk_length, type_mapping):
    """
    Generate directed random walks for a list of graphs.

    """
    walks = []
    for case_name, graph in graphs:
        for node, attributes in graph.nodes(data=True):
            node_type = attributes.get('type')
            desired_types = {'sink', 'source', 'transition'}
            allowed_types = {type_mapping[key] for key in desired_types if key in type_mapping}

            if node_type in allowed_types:
                
                for _ in range(num_walks):
                    walk = random_walk(graph, node, walk_length)
                    walks.append(walk)
    return walks

def calculate_max_walk_length(graphs):
    """
    Calculate the maximum walk length across all graphs.
    
    """
    max_length = 0
    for _, graph in graphs:
        if nx.is_connected(graph.to_undirected()):
            diameter = nx.diameter(graph.to_undirected())
        else:
            
            lengths = dict(nx.all_pairs_shortest_path_length(graph.to_undirected()))
            max_path_length = max(max(length.values()) for length in lengths.values())
            diameter = max_path_length
        max_length = max(max_length, diameter)
    return max_length