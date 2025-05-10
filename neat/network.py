"""
Neural Network Module for NEAT

This module implements the neural network functionality for the NEAT algorithm,
providing a more efficient implementation for network evaluation.
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Any, Optional
import math

from neat.genome import Genome, NodeGene


class FeedForwardNetwork:
    """
    Feed-Forward Neural Network
    
    An efficient implementation of a feed-forward neural network for
    evaluating genomes during evolution.
    """
    
    def __init__(self, genome: Genome):
        """
        Initialize a feed-forward neural network from a genome
        
        Args:
            genome: The genome to convert to a neural network
        """
        self.input_nodes = []
        self.hidden_nodes = []
        self.output_nodes = []
        self.node_evals = []
        
                                 
        self.activation = self._get_activation_function(genome.config)
        
                                   
        for node_id, node in genome.nodes.items():
            if node.type == NodeGene.INPUT:
                self.input_nodes.append(node_id)
            elif node.type == NodeGene.HIDDEN:
                self.hidden_nodes.append(node_id)
            elif node.type == NodeGene.OUTPUT:
                self.output_nodes.append(node_id)
        
                                               
        self.input_nodes.sort()
        self.output_nodes.sort()
        
                                                             
        self._build_node_evaluations(genome)
    
    def _get_activation_function(self, config: Dict[str, Any]):
        """
        Get the activation function specified in the config
        
        Args:
            config: Configuration dictionary
            
        Returns:
            function: The activation function
        """
        activation_name = config['neat']['activation_function']
        
        if activation_name == 'sigmoid':
                                              
            def sigmoid(x):
                try:
                    if x < -100:
                        return 0.0
                    elif x > 100:
                        return 1.0
                    else:
                        return 1.0 / (1.0 + math.exp(-x))
                except OverflowError:
                    return 0.0 if x < 0 else 1.0
            return sigmoid
        elif activation_name == 'tanh':
                                         
            def safe_tanh(x):
                try:
                    if x < -20:
                        return -1.0
                    elif x > 20:
                        return 1.0
                    else:
                        return math.tanh(x)
                except (OverflowError, ValueError):
                    return -1.0 if x < 0 else 1.0
            return safe_tanh
        elif activation_name == 'relu':
            return lambda x: max(0.0, min(x, 100.0))                                   
        elif activation_name == 'leaky_relu':
            return lambda x: min(x, 100.0) if x > 0 else max(0.01 * x, -100.0)
        else:
                                  
            def safe_tanh(x):
                try:
                    if x < -20:
                        return -1.0
                    elif x > 20:
                        return 1.0
                    else:
                        return math.tanh(x)
                except (OverflowError, ValueError):
                    return -1.0 if x < 0 else 1.0
            return safe_tanh
    
    def _build_node_evaluations(self, genome: Genome) -> None:
        """
        Build the node evaluation data structure for efficient network evaluation
        
        This creates a list of (node_id, activation_function, bias, inputs),
        where inputs is a list of (input_node_id, weight) tuples.
        
        Args:
            genome: The genome to convert
        """
                                                                
        node_inputs = {}
        
                                     
        for conn_gene in genome.connections.values():
            if conn_gene.enabled:
                                     
                output_node = conn_gene.output_node
                
                                                                 
                if conn_gene.input_node not in genome.nodes or output_node not in genome.nodes:
                                              
                    continue
                
                                                          
                if output_node not in node_inputs:
                    node_inputs[output_node] = []
                
                                                       
                weight = max(-8.0, min(8.0, conn_gene.weight))
                node_inputs[output_node].append((conn_gene.input_node, weight))
        
                                                             
        nodes_to_evaluate = []
        
                                                                                  
        for node_type in [NodeGene.HIDDEN, NodeGene.OUTPUT]:
            for node_id, node in genome.nodes.items():
                if node.type == node_type:
                    inputs = node_inputs.get(node_id, [])
                                                         
                    bias = max(-8.0, min(8.0, node.bias))
                    nodes_to_evaluate.append((node_id, self.activation, bias, inputs))
        
                            
        self.node_evals = []
        
                                                           
        max_iterations = max(100, len(nodes_to_evaluate) * 2)
        iteration = 0
        
                                                                                    
        while nodes_to_evaluate and iteration < max_iterations:
            iteration += 1
                                                                             
            for i, (node_id, activation, bias, inputs) in enumerate(nodes_to_evaluate):
                input_ids = [input_id for input_id, _ in inputs]
                
                                                                                            
                ready = all(
                    input_id in self.input_nodes or
                    any(eval_node_id == input_id for eval_node_id, _, _, _ in self.node_evals)
                    for input_id in input_ids
                )
                
                if ready:
                    self.node_evals.append(nodes_to_evaluate.pop(i))
                    break
            else:
                                                        
                                                                                        
                if nodes_to_evaluate:
                                                                    
                    best_node_idx = 0
                    min_missing = float('inf')
                    
                    for i, (node_id, activation, bias, inputs) in enumerate(nodes_to_evaluate):
                        input_ids = [input_id for input_id, _ in inputs]
                        
                                                    
                        missing = sum(
                            1 for input_id in input_ids
                            if input_id not in self.input_nodes and
                            not any(eval_node_id == input_id for eval_node_id, _, _, _ in self.node_evals)
                        )
                        
                        if missing < min_missing:
                            min_missing = missing
                            best_node_idx = i
                    
                                                                   
                    self.node_evals.append(nodes_to_evaluate.pop(best_node_idx))
        
                                                     
        if iteration >= max_iterations:
            print(f"Warning: Hit maximum iterations ({max_iterations}) when building network. Possible cycle in neural network.")
    
    def activate(self, inputs: List[float]) -> List[float]:
        """
        Activate the neural network with the given inputs
        
        Args:
            inputs: Input values for the network
            
        Returns:
            List[float]: Output values from the network
        """
                               
        if len(inputs) != len(self.input_nodes):
            raise ValueError(
                f"Expected {len(self.input_nodes)} inputs, got {len(inputs)}"
            )
        
                                                
        sanitized_inputs = []
        for i, val in enumerate(inputs):
            try:
                float_val = float(val)
                                           
                if math.isnan(float_val) or math.isinf(float_val):
                                                     
                    sanitized_inputs.append(0.0)
                else:
                                              
                    sanitized_inputs.append(max(-1000.0, min(1000.0, float_val)))
            except (ValueError, TypeError):
                                              
                sanitized_inputs.append(0.0)
        
                                                           
        node_values = {}
        for i, node_id in enumerate(self.input_nodes):
            node_values[node_id] = sanitized_inputs[i]
        
                                             
        for node_id, activation, bias, node_inputs in self.node_evals:
            try:
                                                                      
                node_sum = bias
                for input_id, weight in node_inputs:
                    if input_id not in node_values:
                                                             
                        continue
                    node_sum += node_values[input_id] * weight
                
                                                               
                node_sum = max(-1000.0, min(1000.0, node_sum))
                
                                                                   
                try:
                    node_values[node_id] = activation(node_sum)
                except Exception as e:
                                                                  
                    print(f"Activation error: {e}, using fallback value")
                    node_values[node_id] = 0.0
            except Exception as e:
                                                                      
                print(f"Node evaluation error: {e}, using fallback value")
                node_values[node_id] = 0.0
        
                                             
        output_values = []
        for node_id in self.output_nodes:
            if node_id in node_values:
                                                 
                val = max(-1.0, min(1.0, node_values[node_id]))
                output_values.append(val)
            else:
                                            
                output_values.append(0.0)
        
        return output_values


def create_feed_forward_network(genome: Genome) -> FeedForwardNetwork:
    """
    Create a feed-forward neural network from a genome
    
    Args:
        genome: The genome to convert
        
    Returns:
        FeedForwardNetwork: A feed-forward neural network
        
    Raises:
        ValueError: If the genome is invalid or cannot be converted to a network
    """
                                             
    if not genome:
        raise ValueError("Cannot create network from empty genome")
    
                                                             
    input_count = sum(1 for node in genome.nodes.values() if node.type == NodeGene.INPUT)
    output_count = sum(1 for node in genome.nodes.values() if node.type == NodeGene.OUTPUT)
    
    if input_count == 0:
        raise ValueError(f"Genome has no input nodes (needs {genome.config['neat']['num_inputs']})")
    
    if output_count == 0:
        raise ValueError(f"Genome has no output nodes (needs {genome.config['neat']['num_outputs']})")
    
                          
    invalid_connections = []
    for conn in genome.connections.values():
        if conn.input_node not in genome.nodes or conn.output_node not in genome.nodes:
            invalid_connections.append(conn)
    
    if invalid_connections:
                                                   
        for conn in invalid_connections:
            del genome.connections[conn.innovation]
        print(f"Removed {len(invalid_connections)} invalid connections from genome")
    
    try:
        return FeedForwardNetwork(genome)
    except Exception as e:
        raise ValueError(f"Failed to create network: {e}")
