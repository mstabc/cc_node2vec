import random

import networkx as nx


class WalkCorpus:
    """Re-iterable walk corpus that generates filtered walks on demand."""

    def __init__(self, graphs, num_walks, walk_length, allowed_types, seed=None):
        self.graphs = graphs
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.allowed_types = set(allowed_types)
        # Lock one seed per corpus so repeated iterations replay the same walks.
        self.seed = random.randrange(2**32) if seed is None else seed
        self.total_walks = sum(
            self.num_walks
            for _, graph in self.graphs
            for _, attributes in graph.nodes(data=True)
            if attributes.get("type") in self.allowed_types
        )

    def __iter__(self):
        rng = random.Random(self.seed)
        for _, graph in self.graphs:
            for node, attributes in graph.nodes(data=True):
                if attributes.get("type") in self.allowed_types:
                    for _ in range(self.num_walks):
                        walk = random_walk(
                            graph=graph,
                            start_node=node,
                            walk_length=self.walk_length,
                            allowed_types=self.allowed_types,
                            rng=rng,
                        )
                        if walk:
                            yield walk

    def __len__(self):
        return self.total_walks


def random_walk(graph, start_node, walk_length, allowed_types=None, rng=None):
    """
    Perform a directed random walk on the graph starting from the given node.

    """
    walk = []
    current_node = start_node
    rng = rng or random

    for _ in range(walk_length):
        current_type = graph.nodes[current_node].get("type")
        if allowed_types is None or current_type in allowed_types:
            walk.append(current_node)

        neighbors = list(graph.successors(current_node))
        if not neighbors:
            break

        current_node = rng.choice(neighbors)

    return walk


def generate_walks(graphs, num_walks, walk_length, type_mapping=None, allowed_types=None, seed=None):
    """
    Return a re-iterable corpus that streams directed random walks.

    """
    if allowed_types is None:
        desired_types = {"sink", "source", "transition"}
        if type_mapping is None:
            raise ValueError("type_mapping or allowed_types must be provided.")
        allowed_types = {type_mapping[key] for key in desired_types if key in type_mapping}

    return WalkCorpus(
        graphs=graphs,
        num_walks=num_walks,
        walk_length=walk_length,
        allowed_types=allowed_types,
        seed=seed,
    )


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
