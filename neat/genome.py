"""
Genome Module for NEAT

This module implements the Genome class which represents a neural network
in the NEAT algorithm. Each genome consists of node genes and connection genes.
"""

import numpy as np
import random
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from copy import deepcopy

from neat.innovation import InnovationTracker


class NodeGene:
    """
    Node Gene in a NEAT Neural Network
    
    Represents a neuron in the neural network with a unique ID,
    type (input, hidden, output), and a bias value.
    """
    
                
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2
    
    def __init__(self, node_id: int, node_type: int, bias: float = 0.0):
        """
        Initialize a node gene
        
        Args:
            node_id: Unique identifier for this node
            node_type: Type of node (INPUT, HIDDEN, or OUTPUT)
            bias: Bias value for this node
        """
        self.id = node_id
        self.type = node_type
        self.bias = bias
    
    def copy(self) -> 'NodeGene':
        """
        Create a copy of this node gene
        
        Returns:
            NodeGene: A copy of this node gene
        """
        return NodeGene(self.id, self.type, self.bias)
    
    def __str__(self) -> str:
        """String representation of the node gene"""
        types = ["INPUT", "HIDDEN", "OUTPUT"]
        return f"Node: {self.id} (type={types[self.type]}, bias={self.bias:.3f})"


class ConnectionGene:
    """
    Connection Gene in a NEAT Neural Network
    
    Represents a connection between two nodes with a weight value,
    enabled/disabled status, and an innovation number for tracking history.
    """
    
    def __init__(self, input_node: int, output_node: int, weight: float,
                 enabled: bool = True, innovation: int = 0):
        """
        Initialize a connection gene
        
        Args:
            input_node: ID of the input node
            output_node: ID of the output node
            weight: Weight of the connection
            enabled: Whether the connection is enabled
            innovation: Historical marking for crossover
        """
        self.input_node = input_node
        self.output_node = output_node
        self.weight = weight
        self.enabled = enabled
        self.innovation = innovation
    
    def copy(self) -> 'ConnectionGene':
        """
        Create a copy of this connection gene
        
        Returns:
            ConnectionGene: A copy of this connection gene
        """
        return ConnectionGene(
            self.input_node,
            self.output_node,
            self.weight,
            self.enabled,
            self.innovation
        )
    
    def __str__(self) -> str:
        """String representation of the connection gene"""
        status = "enabled" if self.enabled else "disabled"
        return f"Connection: {self.input_node} -> {self.output_node} " \
               f"(weight={self.weight:.3f}, {status}, innov={self.innovation})"


class Genome:
    """
    Genome Class for NEAT
    
    Represents a complete neural network genome with node genes and connection genes.
    Includes methods for mutation, crossover, and network evaluation.
    """
    
    def __init__(self, config: Dict[str, Any], innovation_tracker: Optional[InnovationTracker] = None):
        """
        Initialize a genome
        
        Args:
            config: Configuration dictionary
            innovation_tracker: Tracker for innovations
        """
        self.config = config
        self.nodes: Dict[int, NodeGene] = {}
        self.connections: Dict[int, ConnectionGene] = {}
        self.fitness: float = 0.0
        self.adjusted_fitness: float = 0.0
        self.species_id: Optional[int] = None
        
                                                                     
        if innovation_tracker is None:
            return
        
                                                                     
        self.initialize_minimal(innovation_tracker)
    
    def initialize_minimal(self, innovation_tracker: InnovationTracker) -> None:
        """
        Initialize a minimal network with only input and output nodes
        
        Args:
            innovation_tracker: Tracker for innovations
        """
                         
        input_count = self.config['neat']['num_inputs']
        for i in range(input_count):
            self.nodes[i] = NodeGene(i, NodeGene.INPUT)
            innovation_tracker.register_node(i)
            
                          
        output_count = self.config['neat']['num_outputs']
        for i in range(output_count):
            node_id = i + input_count
            self.nodes[node_id] = NodeGene(node_id, NodeGene.OUTPUT)
            innovation_tracker.register_node(node_id)
            
                                                                                   
        for i in range(input_count):
            for j in range(output_count):
                input_node = i
                output_node = j + input_count
                
                                               
                weight = np.random.normal(0, self.config['neat']['initial_weight_stdev'])
                
                                       
                innovation = innovation_tracker.get_connection_innovation(input_node, output_node)
                
                                   
                self.connections[innovation] = ConnectionGene(
                    input_node, output_node, weight, True, innovation
                )
    
    def copy(self) -> 'Genome':
        """
        Create a copy of this genome
        
        Returns:
            Genome: A copy of this genome
        """
                             
        new_genome = Genome(self.config)
        
                    
        for node_id, node in self.nodes.items():
            new_genome.nodes[node_id] = node.copy()
        
                          
        for innov, conn in self.connections.items():
            new_genome.connections[innov] = conn.copy()
        
                      
        new_genome.fitness = self.fitness
        new_genome.adjusted_fitness = self.adjusted_fitness
        new_genome.species_id = self.species_id
        
        return new_genome
    
    def mutate(self, innovation_tracker: InnovationTracker) -> None:
        """
        Mutate this genome
        
        Args:
            innovation_tracker: Tracker for innovations
        """
                                     
        self._mutate_connection_weights()
        
                                      
        self._mutate_add_connection(innovation_tracker)
        
                                
        self._mutate_add_node(innovation_tracker)
        
                       
        self._mutate_bias()
        
                                  
        self._mutate_enable_disable()
    
    def _mutate_connection_weights(self) -> None:
        """Mutate connection weights"""
                                                                             
        for conn in self.connections.values():
            if random.random() < self.config['neat']['weight_mutation_rate']:
                                                                                  
                if random.random() < 0.9:                              
                                    
                    conn.weight += random.uniform(-1, 1) * self.config['neat']['weight_mutation_power']
                else:
                                              
                    conn.weight = random.uniform(-1, 1)
    
    def _mutate_add_connection(self, innovation_tracker: InnovationTracker) -> None:
        """
        Add a new connection through mutation
        
        Args:
            innovation_tracker: Tracker for innovations
        """
        if random.random() >= self.config['neat']['connection_mutation_rate']:
            return
        
                                               
        for _ in range(20):                
                                                 
            source_id = random.choice(list(self.nodes.keys()))
            target_id = random.choice(list(self.nodes.keys()))
            
            source_node = self.nodes[source_id]
            target_node = self.nodes[target_id]
            
                                  
                                            
                                             
                                                                             
                                        
            if (target_node.type == NodeGene.INPUT or
                source_node.type == NodeGene.OUTPUT or
                source_id == target_id or
                self._is_cycle(source_id, target_id)):
                continue
            
                                                     
            exists = False
            for conn in self.connections.values():
                if conn.input_node == source_id and conn.output_node == target_id:
                    exists = True
                    break
            
            if exists:
                continue
            
                                            
            innovation_number = innovation_tracker.get_connection_innovation(source_id, target_id)
            weight = np.random.normal(0, self.config['neat']['initial_weight_stdev'])
            self.connections[innovation_number] = ConnectionGene(
                source_id, target_id, weight, True, innovation_number
            )
            return
    
    def _is_cycle(self, source_id: int, target_id: int) -> bool:
        """
        Check if adding a connection from source_id to target_id would create a cycle
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            
        Returns:
            bool: True if a cycle would be created, False otherwise
        """
                                                                                         
        return self._can_reach(target_id, source_id)
    
    def _can_reach(self, start_id: int, end_id: int) -> bool:
        """
        Check if there's a path from start_id to end_id in the network
        
        Args:
            start_id: Starting node ID
            end_id: Target node ID
            
        Returns:
            bool: True if a path exists, False otherwise
        """
                              
        visited = set()
        queue = [start_id]
        
        while queue:
            node_id = queue.pop(0)
            
            if node_id == end_id:
                return True
            
            if node_id in visited:
                continue
            
            visited.add(node_id)
            
                                                          
            for conn in self.connections.values():
                if conn.input_node == node_id and conn.enabled:
                    queue.append(conn.output_node)
        
        return False
    
    def _mutate_add_node(self, innovation_tracker: InnovationTracker) -> None:
        """
        Add a new node through mutation
        
        Args:
            innovation_tracker: Tracker for innovations
        """
        if random.random() >= self.config['neat']['node_mutation_rate']:
            return
        
                                                     
        enabled_connections = [conn for conn in self.connections.values() if conn.enabled]
        if not enabled_connections:
            return
        
        conn = random.choice(enabled_connections)
        
                                         
        conn.enabled = False
        
                           
        new_node_id = innovation_tracker.get_node_innovation(conn.input_node, conn.output_node)
        self.nodes[new_node_id] = NodeGene(new_node_id, NodeGene.HIDDEN)
        
                                    
                                                              
        input_to_new = innovation_tracker.get_connection_innovation(conn.input_node, new_node_id)
        self.connections[input_to_new] = ConnectionGene(
            conn.input_node, new_node_id, 1.0, True, input_to_new
        )
        
                                                                    
        new_to_output = innovation_tracker.get_connection_innovation(new_node_id, conn.output_node)
        self.connections[new_to_output] = ConnectionGene(
            new_node_id, conn.output_node, conn.weight, True, new_to_output
        )
    
    def _mutate_bias(self) -> None:
        """Mutate node bias values"""
        for node in self.nodes.values():
            if node.type != NodeGene.INPUT and random.random() < self.config['neat']['bias_mutation_rate']:
                                                                                   
                if random.random() < 0.9:                              
                    node.bias += random.uniform(-1, 1) * self.config['neat']['weight_mutation_power']
                else:
                    node.bias = random.uniform(-1, 1)
    
    def _mutate_enable_disable(self) -> None:
        """Mutate enabling/disabling connections"""
        for conn in self.connections.values():
                                            
            if conn.enabled and random.random() < self.config['neat']['disable_mutation_rate']:
                conn.enabled = False
            
                                              
            elif not conn.enabled and random.random() < self.config['neat']['enable_mutation_rate']:
                conn.enabled = True
    
    @staticmethod
    def crossover(parent1: 'Genome', parent2: 'Genome') -> 'Genome':
        """
        Create a new genome by crossing over two parent genomes
        
        The more fit parent is considered the primary parent.
        Matching genes are inherited randomly, while disjoint and
        excess genes are inherited from the more fit parent.
        
        Args:
            parent1: First parent genome
            parent2: Second parent genome
            
        Returns:
            Genome: A new genome created by crossover
        """
                                       
        if parent1.fitness > parent2.fitness:
            fit_parent, less_fit_parent = parent1, parent2
        else:
            fit_parent, less_fit_parent = parent2, parent1
        
                             
        child = Genome(fit_parent.config)
        
                                                    
        for node_id, node_gene in fit_parent.nodes.items():
            child.nodes[node_id] = node_gene.copy()
        
                                  
        for innovation, conn_gene in fit_parent.connections.items():
                                                                       
            if innovation in less_fit_parent.connections:
                                                  
                if random.random() < 0.5:
                    child.connections[innovation] = conn_gene.copy()
                else:
                    child.connections[innovation] = less_fit_parent.connections[innovation].copy()
            else:
                                                                            
                child.connections[innovation] = conn_gene.copy()
        
        return child
    
    def distance(self, other: 'Genome') -> float:
        """
        Calculate the compatibility distance between this genome and another
        
        This distance is used for speciation. It considers:
        - Disjoint genes
        - Excess genes
        - Weight differences in matching genes
        
        Args:
            other: Another genome to compare with
            
        Returns:
            float: The compatibility distance between the two genomes
        """
                                                  
        innovations1 = set(self.connections.keys())
        innovations2 = set(other.connections.keys())
        
                                                   
        matching = innovations1.intersection(innovations2)
        
                                         
        max_innov1 = max(innovations1) if innovations1 else 0
        max_innov2 = max(innovations2) if innovations2 else 0
        
                                                               
        disjoint1 = sum(1 for inn in innovations1 if inn <= max_innov2 and inn not in innovations2)
        
                                                               
        disjoint2 = sum(1 for inn in innovations2 if inn <= max_innov1 and inn not in innovations1)
        
                                            
        excess1 = sum(1 for inn in innovations1 if inn > max_innov2)
        
                                            
        excess2 = sum(1 for inn in innovations2 if inn > max_innov1)
        
        disjoint = disjoint1 + disjoint2
        excess = excess1 + excess2
        
                                                                
        weight_diff = 0.0
        if matching:
            weight_diff = sum(
                abs(self.connections[inn].weight - other.connections[inn].weight)
                for inn in matching
            ) / len(matching)
        
                               
        c1 = self.config['neat']['compatibility_disjoint_coefficient']
        c2 = self.config['neat']['compatibility_weight_coefficient']
        
                                                                        
        n = max(len(innovations1), len(innovations2))
        n = 1 if n < 20 else n                                     
        
                                          
        distance = (c1 * (disjoint + excess) / n) + (c2 * weight_diff)
        
        return distance
    
    def activate(self, inputs: List[float]) -> List[float]:
        """
        Activate the neural network with the given inputs
        
        Args:
            inputs: Input values for the network
            
        Returns:
            List[float]: Output values from the network
        """
        if len(inputs) != sum(1 for node in self.nodes.values() if node.type == NodeGene.INPUT):
            raise ValueError("Input dimensions don't match network inputs")
        
                                 
        activation_func = self._get_activation_function()
        
                                
        node_values = {}
        
                          
        input_nodes = [node_id for node_id, node in self.nodes.items() if node.type == NodeGene.INPUT]
        for i, node_id in enumerate(input_nodes):
            node_values[node_id] = inputs[i]
        
                                                                       
        sorted_nodes = self._get_nodes_in_layers()
        
                            
        for layer in sorted_nodes[1:]:                    
            for node_id in layer:
                node = self.nodes[node_id]
                
                                          
                node_sum = node.bias
                for conn in self.connections.values():
                    if conn.output_node == node_id and conn.enabled:
                        if conn.input_node in node_values:
                            node_sum += node_values[conn.input_node] * conn.weight
                
                                           
                node_values[node_id] = activation_func(node_sum)
        
                              
        output_nodes = [node_id for node_id, node in self.nodes.items() if node.type == NodeGene.OUTPUT]
        return [node_values[node_id] for node_id in output_nodes]
    
    def _get_activation_function(self):
        """
        Get the activation function specified in the config
        
        Returns:
            function: The activation function
        """
        activation_name = self.config['neat']['activation_function']
        
        if activation_name == 'sigmoid':
            return lambda x: 1.0 / (1.0 + np.exp(-x))
        elif activation_name == 'tanh':
            return lambda x: np.tanh(x)
        elif activation_name == 'relu':
            return lambda x: max(0, x)
        else:
                             
            return lambda x: np.tanh(x)
    
    def _get_nodes_in_layers(self) -> List[List[int]]:
        """
        Return nodes organized by layers for feed-forward processing
        
        Returns:
            List[List[int]]: Lists of node IDs organized by layer
        """
                     
        input_nodes = [node_id for node_id, node in self.nodes.items() if node.type == NodeGene.INPUT]
        
                      
        output_nodes = [node_id for node_id, node in self.nodes.items() if node.type == NodeGene.OUTPUT]
        
                                                
        hidden_nodes = [node_id for node_id, node in self.nodes.items() if node.type == NodeGene.HIDDEN]
        
                                 
        layers = [input_nodes]
        
                                       
        processed = set(input_nodes)
        
                                                       
        remaining = set(hidden_nodes)
        while remaining:
                                                       
            current_layer = []
            for node_id in list(remaining):
                                                                      
                inputs_processed = True
                for conn in self.connections.values():
                    if conn.output_node == node_id and conn.enabled and conn.input_node not in processed:
                        inputs_processed = False
                        break
                
                if inputs_processed:
                    current_layer.append(node_id)
                    remaining.remove(node_id)
            
                                                        
            if current_layer:
                layers.append(current_layer)
                processed.update(current_layer)
            else:
                                                                       
                                                                               
                if remaining:
                                                                
                    current_layer = list(remaining)[:10]
                    layers.append(current_layer)
                    processed.update(current_layer)
                    for node_id in current_layer:
                        remaining.remove(node_id)
                else:
                    break
        
                          
        layers.append(output_nodes)
        
        return layers
    
    def __str__(self) -> str:
        """String representation of the genome"""
        s = f"Genome (fitness={self.fitness:.2f}, species={self.species_id}):\n"
        s += "  Nodes:\n"
        for node in self.nodes.values():
            s += f"    {node}\n"
        s += "  Connections:\n"
        for conn in self.connections.values():
            s += f"    {conn}\n"
        return s
