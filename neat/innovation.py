"""
Innovation Tracker Module for NEAT

This module tracks innovation numbers for new connections and nodes in NEAT.
Innovation numbers are used to track the historical origin of genes for
proper crossover between genomes of different topologies.
"""

from typing import Dict, Tuple, Set


class InnovationTracker:
    """
    Innovation Tracker for NEAT algorithm
    
    Tracks innovation numbers for new connections and nodes to ensure
    consistent historical markings across the population.
    """
    
    def __init__(self):
        """Initialize the innovation tracker"""
                                                                                     
        self.connection_innovations: Dict[Tuple[int, int], int] = {}
        
                                                                   
        self.node_innovations: Dict[Tuple[int, int], int] = {}
        
                                  
        self.node_ids: Set[int] = set()
        
                                   
        self.innovation_number = 0
        
                                                       
        self.next_node_id = 0
    
    def get_connection_innovation(self, in_node: int, out_node: int) -> int:
        """
        Get the innovation number for a connection gene
        
        If the connection is new, assign a new innovation number.
        If the connection already exists, return its innovation number.
        
        Args:
            in_node: Input node ID
            out_node: Output node ID
            
        Returns:
            innovation_number: The innovation number for this connection
        """
        key = (in_node, out_node)
        
                                                                               
        if key in self.connection_innovations:
            return self.connection_innovations[key]
        
                                                   
        self.connection_innovations[key] = self.innovation_number
        self.innovation_number += 1
        
        return self.connection_innovations[key]
    
    def get_node_innovation(self, in_node: int, out_node: int) -> int:
        """
        Get the node ID for splitting a connection between in_node and out_node
        
        If this connection has been split before, return the same node ID.
        If this is a new split, create a new node ID.
        
        Args:
            in_node: Input node ID
            out_node: Output node ID
            
        Returns:
            node_id: The ID for the new node in this position
        """
        key = (in_node, out_node)
        
                                                                           
        if key in self.node_innovations:
            return self.node_innovations[key]
        
                                         
                                                             
        while self.next_node_id in self.node_ids:
            self.next_node_id += 1
        
                               
        self.node_innovations[key] = self.next_node_id
        self.node_ids.add(self.next_node_id)
        
        node_id = self.next_node_id
        self.next_node_id += 1
        
        return node_id
    
    def register_node(self, node_id: int) -> None:
        """
        Register an existing node ID
        
        Args:
            node_id: The node ID to register
        """
        self.node_ids.add(node_id)
        if node_id >= self.next_node_id:
            self.next_node_id = node_id + 1
    
    def reset(self) -> None:
        """Reset the innovation tracker"""
        self.connection_innovations = {}
        self.node_innovations = {}
        self.node_ids = set()
        self.innovation_number = 0
        self.next_node_id = 0
