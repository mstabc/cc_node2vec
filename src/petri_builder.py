
import pm4py
from pm4py.objects.petri_net.importer import importer as pnml_importer
import networkx as nx
import copy
import os
import glob
from collections import defaultdict, OrderedDict
from pm4py.streaming.importer.xes.variants import xes_trace_stream, xes_event_stream
import matplotlib.pyplot as plt
import logging
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.log.importer.xes.variants import iterparse, line_by_line, iterparse_mem_compressed, iterparse_20, chunk_regex, rustxes
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.objects.log import obj, exporter, importer, util
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from pm4py.algo.discovery.ilp import algorithm as ilp_miner
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import networkx as nx
import re



class PetriNetGraph:
    def __init__(self, log, net, initial_marking, final_marking):
        self.graph = nx.DiGraph()
        self.net = net
        self.log = log
        self.initial_marking = initial_marking
        self.final_marking = final_marking
        self._initialize_petri_net()

    def _initialize_petri_net(self):
        """Convert the Petri net (places, transitions, arcs) to a graph structure."""

        for place in self.net.places:
            place_type = 'place'
            if place in self.initial_marking:
                place_type = 'source'
            elif place in self.final_marking:
                place_type = 'sink'
                
            self.graph.add_node(
                place.name,
                label='source' if place_type == 'source' else 'sink' if place_type == 'sink' else None,
                type=place_type,
                feature={"count": 0}
            )


        for transition in self.net.transitions:
            transition_label = transition.label
            node_type = 'silent' if transition_label is None else 'transition'
            self.graph.add_node(
                transition.name,
                label=transition_label,
                type=node_type,
                feature={"count": 0}
            )


        for arc in self.net.arcs:
            self.graph.add_edge(arc.source.name, arc.target.name, feature={"count": 0})

    def create_trace_graph(self, trace_df):
        """Create a specific graph for the trace by cloning the Petri net and updating node features."""

        trace_graph = copy.deepcopy(self.graph)
        

        events = trace_df['concept:name'].tolist()
        

        label_to_nodes = {}
        for node, data in trace_graph.nodes(data=True):
            if data.get('type') == 'transition' and data.get('label') is not None:
                label = data.get('label')
                if label not in label_to_nodes:
                    label_to_nodes[label] = []
                label_to_nodes[label].append(node)
        

        transitions_to_remove = []
        for node, data in trace_graph.nodes(data=True):
            if data.get('type') == 'transition':
                label = data.get('label')
                if label is not None and label not in events:
                    transitions_to_remove.append(node)
        

        trace_graph.remove_nodes_from(transitions_to_remove)
        

        source_nodes = []
        if self.initial_marking:
            for place in self.initial_marking:
                source_node = str(place).replace('\n', '/')
                if source_node in trace_graph:
                    source_nodes.append(source_node)
        

        if not source_nodes:
            source_node = "source"
            trace_graph.add_node(source_node, label='source', type='source', feature={"count": 0})
            source_nodes.append(source_node)
            

        sink_nodes = []
        if self.final_marking:
            for place in self.final_marking:
                sink_node = str(place).replace('\n', '/')
                if sink_node in trace_graph:
                    sink_nodes.append(sink_node)
        

        if not sink_nodes:
            sink_node = "sink"
            trace_graph.add_node(sink_node, label='sink', type='sink', feature={"count": 0})
            sink_nodes.append(sink_node)
            

        if not trace_df.empty:
            first_event_name = trace_df.iloc[0]['concept:name']
            last_event_name = trace_df.iloc[-1]['concept:name']
            

            if not self.initial_marking or all(trace_graph.out_degree(sn) == 0 for sn in source_nodes):
                first_event_nodes = label_to_nodes.get(first_event_name, [])
                for source_node in source_nodes:
                    if first_event_nodes:
                        for node in first_event_nodes:
                            if node in trace_graph:
                                trace_graph.add_edge(source_node, node, feature={"count": 0})
                    else:

                        first_event_node = f"t_{first_event_name}"
                        if first_event_node not in trace_graph:
                            trace_graph.add_node(first_event_node, label=first_event_name, 
                                              type='transition', feature={"count": 0})
                        trace_graph.add_edge(source_node, first_event_node, feature={"count": 0})
            
            if not self.final_marking or all(trace_graph.in_degree(sn) == 0 for sn in sink_nodes):
                last_event_nodes = label_to_nodes.get(last_event_name, [])
                for sink_node in sink_nodes:
                    if last_event_nodes:
                        for node in last_event_nodes:
                            if node in trace_graph:  
                                trace_graph.add_edge(node, sink_node, feature={"count": 0})
                    else:

                        last_event_node = f"t_{last_event_name}"
                        if last_event_node not in trace_graph:
                            trace_graph.add_node(last_event_node, label=last_event_name, 
                                              type='transition', feature={"count": 0})
                        trace_graph.add_edge(last_event_node, sink_node, feature={"count": 0})

        for _, event in trace_df.iterrows():
            activity_name = event.get('concept:name', 'Unknown Activity')

            matching_transitions = [node for node, data in trace_graph.nodes(data=True) 
                                   if data.get('type') == 'transition' and data.get('label') == activity_name]
            

            for transition in matching_transitions:
                feature = trace_graph.nodes[transition].get('feature', {})
                feature["count"] = feature.get("count", 0) + 1
                trace_graph.nodes[transition]['feature'] = feature
        

        self._clean_isolated_places(trace_graph, source_nodes, sink_nodes)
        

        self._handle_disconnected_components(trace_graph, source_nodes, sink_nodes)
        
        return trace_graph
    
    def _clean_isolated_places(self, graph, source_nodes, sink_nodes):
        """Remove isolated places (degree < 2) except source and sink nodes."""
        removed = True
        while removed:
            removed = False
            nodes_to_remove = []
            
            for node in graph.nodes:
                if (graph.degree[node] < 2 and 
                    node not in source_nodes and 
                    node not in sink_nodes and 
                    graph.nodes[node].get('type') == 'place'):
                    nodes_to_remove.append(node)
            
            if nodes_to_remove:
                graph.remove_nodes_from(nodes_to_remove)
                removed = True
    
    def _handle_disconnected_components(self, graph, source_nodes, sink_nodes):
        """Identify and handle disconnected components in the graph."""

        undirected = graph.to_undirected()
        components = list(nx.connected_components(undirected))
        
        if len(components) <= 1:
            return
        

        main_component = None
        max_source_sink_count = -1
        
        for component in components:
            source_sink_count = sum(1 for node in component if node in source_nodes or node in sink_nodes)
            if source_sink_count > max_source_sink_count:
                max_source_sink_count = source_sink_count
                main_component = component
        

        if max_source_sink_count == 0:
            main_component = max(components, key=len)
        
        for component in components:
            if component != main_component:
                
                for node in component:
                    graph.nodes[node]['disconnected'] = True
                
  
                graph.remove_nodes_from(component)
                
    
    def generate_trace_graphs(self):
        """Generate a graph for each trace in the log."""
        trace_graphs = []
        for case_id, trace_df in self.log.groupby('case:concept:name'):
            trace_graph = self.create_trace_graph(trace_df)
            
            trace_graphs.append((case_id, trace_graph))
        
        return trace_graphs
    
class PetriNetMinedGraph:
    def __init__(self, net, initial_marking, final_marking):
        self.graph = nx.DiGraph()
        self.net = net
        self.initial_marking = initial_marking
        self.final_marking = final_marking
        self._initialize_petri_net()
        
    def _initialize_petri_net(self):
        """Convert the Petri net (places, transitions, arcs) to a graph structure."""

        added_nodes = set()
        

        for place in self.net.places:
            place_type = 'place'
            place_label = None
            if place in self.initial_marking:
                place_type = 'source'
                place_label = 'source'
            elif place in self.final_marking:
                place_type = 'sink'
                place_label = 'sink'
            
            self.graph.add_node(
                place.name,
                label=place_label,
                type=place_type,
                feature={"count": 0}
            )
            added_nodes.add(place.name)

        for transition in self.net.transitions:
            transition_label = transition.label if transition.label else None
            node_type = 'transition' if transition.label else 'silent'

            self.graph.add_node(
                transition.name,
                label=transition_label,
                type=node_type,
                feature={"count": 0}
            )
            added_nodes.add(transition.name)

        for arc in self.net.arcs:
            if arc.source.name in added_nodes and arc.target.name in added_nodes:
                self.graph.add_edge(
                    arc.source.name, 
                    arc.target.name, 
                    feature={"count": 0},
                    weight=1.0
                )
            else:
                print(f"Warning: Missing nodes in arc {arc.source.name} -> {arc.target.name}")

    def get_graph(self):
        """Return the complete Petri net graph."""
        return self.graph