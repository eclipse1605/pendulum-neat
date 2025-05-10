"""
Species Module for NEAT

This module handles species management for the NEAT algorithm, including
creating, updating, and managing species throughout evolution.
"""

import random
from typing import Dict, List, Set, Optional, Any

from neat.genome import Genome


class Species:
    """
    Species Class for NEAT
    
    Represents a species of genomes that are similar in structure.
    Species are used to promote diversity in the population through
    speciation, which groups similar genomes together for fitness sharing.
    """
    
    def __init__(self, species_id: int, representative: Genome):
        """
        Initialize a species
        
        Args:
            species_id: Unique identifier for this species
            representative: Genome that represents this species
        """
        self.id = species_id
        self.representative = representative
        self.members: List[Genome] = [representative]
        self.fitness_history: List[float] = []
        self.generations_no_improvement = 0
        self.adjusted_fitness_sum = 0.0
    
    def add_member(self, genome: Genome) -> None:
        """
        Add a genome to this species
        
        Args:
            genome: Genome to add to this species
        """
        genome.species_id = self.id
        self.members.append(genome)
    
    def calculate_adjusted_fitness(self) -> None:
        """
        Calculate adjusted fitness for all members using fitness sharing
        """
                                                    
                                                        
        for genome in self.members:
            genome.adjusted_fitness = genome.fitness / len(self.members)
        
                                     
        self.adjusted_fitness_sum = sum(genome.adjusted_fitness for genome in self.members)
    
    def update_fitness_history(self) -> bool:
        """
        Update fitness history with current max fitness
        
        Returns:
            bool: True if fitness improved, False otherwise
        """
        if not self.members:
            return False
        
                               
        max_fitness = max(genome.fitness for genome in self.members)
        
                                   
        improved = False
        if not self.fitness_history or max_fitness > max(self.fitness_history):
            improved = True
            self.generations_no_improvement = 0
        else:
            self.generations_no_improvement += 1
        
                                
        self.fitness_history.append(max_fitness)
        
        return improved
    
    def sort_members(self) -> None:
        """Sort members by fitness (descending)"""
        self.members.sort(key=lambda genome: genome.fitness, reverse=True)
    
    def cull(self, survival_threshold: float) -> None:
        """
        Remove less fit members of the species
        
        Args:
            survival_threshold: Portion of the species to keep
        """
        if len(self.members) > 2:
                                     
            self.sort_members()
            
                                        
            keep = max(2, int(survival_threshold * len(self.members)))
            
                                     
            self.members = self.members[:keep]
    
    def select_parent(self) -> Genome:
        """
        Select a parent for reproduction
        
        Uses tournament selection to select a parent
        
        Returns:
            Genome: The selected parent
        """
        if not self.members:
            raise RuntimeError("Cannot select parent from empty species")
        
        if len(self.members) == 1:
            return self.members[0]
        
                              
        k = min(3, len(self.members))
        candidates = random.sample(self.members, k)
        return max(candidates, key=lambda genome: genome.fitness)
    
    def choose_new_representative(self) -> None:
        """Choose a new representative for this species"""
        if self.members:
            self.representative = random.choice(self.members)
    
    def __str__(self) -> str:
        """String representation of the species"""
        avg_fitness = sum(genome.fitness for genome in self.members) / len(self.members) if self.members else 0
        return f"Species {self.id}: {len(self.members)} members, avg fitness: {avg_fitness:.2f}"


class SpeciesManager:
    """
    Species Manager for NEAT
    
    Manages all species in the population, including creating new species,
    assigning genomes to species, and culling stagnant species.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the species manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.species: Dict[int, Species] = {}
        self.next_species_id = 0
    
    def speciate(self, genomes: List[Genome]) -> None:
        """
        Assign each genome to a species
        
        Args:
            genomes: List of genomes to assign to species
        """
                               
        for species in self.species.values():
            species.members = []
        
                                   
        for genome in genomes:
            assigned = False
            
                                              
            for species in self.species.values():
                if genome.distance(species.representative) < self.config['neat']['species_threshold']:
                    species.add_member(genome)
                    assigned = True
                    break
            
                                                                           
            if not assigned:
                new_species = Species(self.next_species_id, genome)
                self.species[self.next_species_id] = new_species
                genome.species_id = self.next_species_id
                self.next_species_id += 1
        
                              
        self.species = {id: species for id, species in self.species.items() if species.members}
    
    def calculate_adjusted_fitness(self) -> None:
        """Calculate adjusted fitness for all species"""
        for species in self.species.values():
            species.calculate_adjusted_fitness()
    
    def update_fitness_history(self) -> None:
        """Update fitness history for all species"""
        for species in self.species.values():
            species.update_fitness_history()
    
    def sort_species_members(self) -> None:
        """Sort members in each species by fitness"""
        for species in self.species.values():
            species.sort_members()
    
    def get_species_reproduction_allocations(self) -> Dict[int, int]:
        """
        Calculate how many offspring each species should produce
        
        Returns:
            Dict[int, int]: Mapping of species ID to number of offspring
        """
                                          
        total_adjusted_fitness = sum(species.adjusted_fitness_sum for species in self.species.values())
        
                                                                  
        allocations = {}
        
        if total_adjusted_fitness > 0:
                                                                   
            population_size = self.config['neat']['population_size']
            for species_id, species in self.species.items():
                offspring = int(round(species.adjusted_fitness_sum / total_adjusted_fitness * population_size))
                allocations[species_id] = offspring
        else:
                                                              
            offspring_per_species = max(1, self.config['neat']['population_size'] // len(self.species))
            for species_id in self.species.keys():
                allocations[species_id] = offspring_per_species
        
                                                           
        total_offspring = sum(allocations.values())
        
        if total_offspring < self.config['neat']['population_size']:
                                                                             
            sorted_species = sorted(
                self.species.values(),
                key=lambda s: max(g.fitness for g in s.members) if s.members else 0,
                reverse=True
            )
            for species in sorted_species:
                if total_offspring >= self.config['neat']['population_size']:
                    break
                allocations[species.id] += 1
                total_offspring += 1
        elif total_offspring > self.config['neat']['population_size']:
                                                                              
            sorted_species = sorted(
                self.species.values(),
                key=lambda s: max(g.fitness for g in s.members) if s.members else 0
            )
            for species in sorted_species:
                if total_offspring <= self.config['neat']['population_size']:
                    break
                if allocations[species.id] > 1:
                    allocations[species.id] -= 1
                    total_offspring -= 1
        
        return allocations
    
    def cull_species(self) -> None:
        """Cull members within each species to keep only the strongest"""
        for species in self.species.values():
            species.cull(self.config['neat']['survival_threshold'])
    
    def remove_stagnant_species(self) -> None:
        """Remove species that haven't improved in a long time"""
                                               
        species_to_remove = []
        
                                      
        if len(self.species) <= 1:
            return
        
                                            
        stagnation_threshold = 15                                                              
        for species_id, species in self.species.items():
            if species.generations_no_improvement > stagnation_threshold:
                species_to_remove.append(species_id)
        
                                 
        for species_id in species_to_remove:
            del self.species[species_id]
    
    def choose_new_representatives(self) -> None:
        """Choose new representatives for each species"""
        for species in self.species.values():
            species.choose_new_representative()
    
    def get_all_members(self) -> List[Genome]:
        """
        Get all genomes from all species
        
        Returns:
            List[Genome]: List of all genomes
        """
        members = []
        for species in self.species.values():
            members.extend(species.members)
        return members
    
    def get_best_genome(self) -> Optional[Genome]:
        """
        Get the best genome from all species
        
        Returns:
            Optional[Genome]: The best genome, or None if no genomes
        """
        best_genome = None
        best_fitness = float('-inf')
        
        for species in self.species.values():
            for genome in species.members:
                if genome.fitness > best_fitness:
                    best_fitness = genome.fitness
                    best_genome = genome
        
        return best_genome
    
    def __str__(self) -> str:
        """String representation of the species manager"""
        s = f"Species Manager ({len(self.species)} species):\n"
        for species in sorted(self.species.values(), key=lambda s: s.id):
            s += f"  {species}\n"
        return s
