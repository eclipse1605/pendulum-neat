"""
Network Visualization Module for NEAT

This module provides functionality for visualizing neural networks
evolved by the NEAT algorithm, showing their structure and connections.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from typing import Dict, List, Tuple, Any, Optional
import os

from neat.genome import Genome, NodeGene
from visualization.colors import interpolate_color


def draw_network(genome: Genome, ax=None, node_radius: float = 0.1,
                show_disabled: bool = False, node_colors: Optional[Dict] = None) -> plt.Figure:
    """
    Draw a neural network from a genome
    
    Args:
        genome: The genome to visualize
        ax: Matplotlib axes to draw on (creates new one if None)
        node_radius: Radius of node circles
        show_disabled: Whether to show disabled connections
        node_colors: Optional dict mapping node types to colors
        
    Returns:
        plt.Figure: Matplotlib figure of the network
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure
    
                        
    if node_colors is None:
        node_colors = {
            NodeGene.INPUT: (0.7, 0.85, 1.0),                
            NodeGene.HIDDEN: (1.0, 0.85, 0.7),                 
            NodeGene.OUTPUT: (0.7, 1.0, 0.7)                  
        }
    
                            
    input_nodes = [node for node_id, node in genome.nodes.items() if node.type == NodeGene.INPUT]
    hidden_nodes = [node for node_id, node in genome.nodes.items() if node.type == NodeGene.HIDDEN]
    output_nodes = [node for node_id, node in genome.nodes.items() if node.type == NodeGene.OUTPUT]
    
                                   
    node_positions = {}
    
                                             
    input_count = len(input_nodes)
    for i, node in enumerate(input_nodes):
        if input_count > 1:
            y = 0.9 - 0.8 * (i / (input_count - 1))
        else:
            y = 0.5
        node_positions[node.id] = (0.1, y)
    
                                               
    output_count = len(output_nodes)
    for i, node in enumerate(output_nodes):
        if output_count > 1:
            y = 0.9 - 0.8 * (i / (output_count - 1))
        else:
            y = 0.5
        node_positions[node.id] = (0.9, y)
    
                                         
                                          
    if hidden_nodes:
        layers = _organize_hidden_nodes_in_layers(genome, hidden_nodes)
        layer_count = len(layers)
        
                                   
        for layer_idx, layer_nodes in enumerate(layers):
            x = 0.1 + 0.8 * ((layer_idx + 1) / (layer_count + 1))
            node_count = len(layer_nodes)
            
            for i, node in enumerate(layer_nodes):
                if node_count > 1:
                    y = 0.9 - 0.8 * (i / (node_count - 1))
                else:
                    y = 0.5
                node_positions[node.id] = (x, y)
    
                      
    for conn in genome.connections.values():
        if not conn.enabled and not show_disabled:
            continue
        
                       
        if conn.input_node not in node_positions or conn.output_node not in node_positions:
            continue
        
        input_pos = node_positions[conn.input_node]
        output_pos = node_positions[conn.output_node]
        
                         
                                                                                            
        if conn.weight > 0:
            color = interpolate_color((0, 0, 0), (0, 0.8, 0), min(1.0, conn.weight))
        elif conn.weight < 0:
            color = interpolate_color((0, 0, 0), (0.8, 0, 0), min(1.0, -conn.weight))
        else:
            color = (0, 0, 0)
        
                                               
        linewidth = 0.5 + 2.0 * min(1.0, abs(conn.weight))
        
                                                
        linestyle = '-' if conn.enabled else ':'
        
                             
        ax.add_patch(FancyArrowPatch(
            input_pos, output_pos,
            arrowstyle='-|>', shrinkA=node_radius*72, shrinkB=node_radius*72,
            connectionstyle='arc3,rad=0.1',
            linewidth=linewidth, color=color, linestyle=linestyle
        ))
    
                
    for node_id, pos in node_positions.items():
        node = genome.nodes[node_id]
        color = node_colors[node.type]
        
                          
        circle = Circle(pos, radius=node_radius, facecolor=color, edgecolor='black', zorder=10)
        ax.add_patch(circle)
        
                          
        if node.type == NodeGene.INPUT:
                                                        
            ax.text(pos[0] - 0.15, pos[1], f"Input {node_id}", 
                   ha='right', va='center', fontsize=9)
        elif node.type == NodeGene.OUTPUT:
                                                          
            ax.text(pos[0] + 0.15, pos[1], f"Output {node_id - len(input_nodes)}", 
                   ha='left', va='center', fontsize=9)
        else:
                                                    
            ax.text(pos[0], pos[1], str(node_id), 
                   ha='center', va='center', fontsize=8, color='black')
    
                 
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Neural Network (Fitness: {genome.fitness:.2f})")
    ax.axis('off')
    
    return fig


def _organize_hidden_nodes_in_layers(genome: Genome, hidden_nodes: List[NodeGene]) -> List[List[NodeGene]]:
    """
    Organize hidden nodes into layers for visualization
    
    Arranges hidden nodes based on their position in the network topology.
    
    Args:
        genome: The genome containing the nodes
        hidden_nodes: List of hidden nodes to organize
        
    Returns:
        List[List[NodeGene]]: Nodes organized into layers
    """
                       
    layers = []
    
                      
    input_nodes = set(node_id for node_id, node in genome.nodes.items() 
                     if node.type == NodeGene.INPUT)
    
                       
    output_nodes = set(node_id for node_id, node in genome.nodes.items() 
                      if node.type == NodeGene.OUTPUT)
    
                                                 
    connections_to = {node.id: set() for node in hidden_nodes}
    connections_from = {node.id: set() for node in hidden_nodes}
    
    for conn in genome.connections.values():
        if not conn.enabled:
            continue
        
        if conn.output_node in connections_to:
            connections_to[conn.output_node].add(conn.input_node)
        
        if conn.input_node in connections_from:
            connections_from[conn.input_node].add(conn.output_node)
    
                                         
    remaining_nodes = set(node.id for node in hidden_nodes)
    
                                                 
    current_layer = []
    for node in hidden_nodes:
        if node.id in remaining_nodes:
                                                                  
            input_sources = connections_to[node.id]
            if input_sources and input_sources.issubset(input_nodes):
                current_layer.append(node)
                remaining_nodes.remove(node.id)
    
    if current_layer:
        layers.append(current_layer)
        current_sources = set(node.id for node in current_layer)
    else:
                                                    
        layers.append([])
        current_sources = set()
    
                   
    while remaining_nodes:
        current_layer = []
        next_remaining = set(remaining_nodes)
        
        for node_id in remaining_nodes:
                                                                                
            input_sources = connections_to[node_id]
            if input_sources and input_sources.issubset(input_nodes | current_sources):
                node = genome.nodes[node_id]
                current_layer.append(node)
                next_remaining.remove(node_id)
        
                                                                             
                                                              
        if not current_layer and next_remaining:
            node_id = next(iter(next_remaining))
            current_layer.append(genome.nodes[node_id])
            next_remaining.remove(node_id)
        
        if current_layer:
            layers.append(current_layer)
            current_sources.update(node.id for node in current_layer)
        
        remaining_nodes = next_remaining
        
                                             
        if len(layers) > 10:
                                                         
            if remaining_nodes:
                remaining_layer = [genome.nodes[node_id] for node_id in remaining_nodes]
                layers.append(remaining_layer)
            break
    
    return layers


def save_network_visualization(genome: Genome, filename: str) -> None:
    """
    Save a visualization of a neural network to a file
    
    Args:
        genome: The genome to visualize
        filename: Path to save the visualization to
    """
                                          
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
                            
    fig, ax = plt.subplots(figsize=(10, 8))
    
                  
    draw_network(genome, ax)
    
                   
    plt.tight_layout()
    
                 
    plt.savefig(filename, dpi=150)
    plt.close(fig)


def visualize_genome_set(genomes: List[Genome], directory: str) -> None:
    """
    Visualize and save a set of genomes to a directory
    
    Args:
        genomes: List of genomes to visualize
        directory: Directory to save the visualizations to
    """
                                          
    os.makedirs(directory, exist_ok=True)
    
                                        
    for i, genome in enumerate(genomes):
        filename = os.path.join(directory, f"genome_{i}_fitness_{genome.fitness:.2f}.png")
        save_network_visualization(genome, filename)
        print(f"Saved visualization for genome {i} to {filename}")
