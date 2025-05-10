"""
Population Module for NEAT

This module implements the Population class which manages the entire
evolutionary process for the NEAT algorithm, including selection,
reproduction, and fitness evaluation.
"""

import random
import pickle
import os
from typing import Dict, List, Set, Tuple, Callable, Any, Optional
import numpy as np

from neat.genome import Genome
from neat.species import SpeciesManager
from neat.innovation import InnovationTracker


class Population:
    """
    Population Class for NEAT
    
    Manages a population of genomes that evolve using the NEAT algorithm,
    including speciation, selection, reproduction, and evaluation.
    """
    
    def __init__(self, config: Dict[str, Any], eval_function: Callable[[Genome], float]):
        """
        Initialize a population
        
        Args:
            config: Configuration dictionary
            eval_function: Function to evaluate a genome's fitness
        """
        self.config = config
        self.eval_function = eval_function
        self.population_size = config['neat']['population_size']
        self.generation = 0
        
                                   
        self.innovation_tracker = InnovationTracker()
        
                                
        self.species_manager = SpeciesManager(config)
        
                                   
        self.genomes = [Genome(config, self.innovation_tracker) for _ in range(self.population_size)]
        
                               
        self.best_genome = None
        self.best_fitness = float('-inf')
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
                            
        self.species_manager.speciate(self.genomes)
    
    def evolve(self) -> Tuple[float, float]:
        """
        Evolve the population one generation
        
        Returns:
            Tuple[float, float]: Average fitness and best fitness of the generation
        """
                                             
        self._evaluate_fitness()
        
                           
        self._update_statistics()
        
                                                          
        if (not self.config['neat']['no_fitness_termination'] and
            self.best_fitness >= self.config['neat']['fitness_threshold']):
            return self.avg_fitness_history[-1], self.best_fitness_history[-1]
        
                                        
        self.species_manager.update_fitness_history()
        
                                 
        self.species_manager.remove_stagnant_species()
        
                                                     
        self.species_manager.calculate_adjusted_fitness()
        
                                                   
        offspring_allocations = self.species_manager.get_species_reproduction_allocations()
        
                                                       
        self.species_manager.cull_species()
        
                                        
        new_genomes = self._reproduce(offspring_allocations)
        
                                    
        self.genomes = new_genomes
        
                                                     
        self.species_manager.choose_new_representatives()
        
                                       
        self.species_manager.speciate(self.genomes)
        
                                      
        self.generation += 1
        
        return self.avg_fitness_history[-1], self.best_fitness_history[-1]
    
    def _evaluate_fitness(self) -> None:
        """Evaluate the fitness of all genomes in the population"""
                                                    
        import signal
        from contextlib import contextmanager
        import time
        
        @contextmanager
        def timeout(seconds):
            def handler(signum, frame):
                raise TimeoutError(f"Evaluation timed out after {seconds} seconds")
            
                                     
            original_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                yield
            finally:
                                                                   
                signal.signal(signal.SIGALRM, original_handler)
                signal.alarm(0)
        
                                                                               
        eval_timeout = 5
        
        for i, genome in enumerate(self.genomes):
            try:
                with timeout(eval_timeout):
                    genome.fitness = self.eval_function(genome)
            except TimeoutError as e:
                print(f"Warning: Genome {i} evaluation timed out. Assigning minimal fitness.")
                                                            
                genome.fitness = 0.1
            except Exception as e:
                print(f"Error evaluating genome {i}: {str(e)}")
                genome.fitness = 0.1
    
    def _update_statistics(self) -> None:
        """Update population statistics"""
                                   
        avg_fitness = sum(genome.fitness for genome in self.genomes) / len(self.genomes)
        self.avg_fitness_history.append(avg_fitness)
        
                          
        current_best = max(self.genomes, key=lambda genome: genome.fitness)
        current_best_fitness = current_best.fitness
        self.best_fitness_history.append(current_best_fitness)
        
                                     
        if current_best_fitness > self.best_fitness:
            self.best_fitness = current_best_fitness
            self.best_genome = current_best.copy()
    
    def _reproduce(self, offspring_allocations: Dict[int, int]) -> List[Genome]:
        """
        Create the next generation of genomes
        
        Args:
            offspring_allocations: Dict mapping species ID to number of offspring
            
        Returns:
            List[Genome]: List of genomes for the next generation
        """
                                 
        new_genomes = []
        
                               
        elitism = self.config['neat']['elitism']
        
                          
        for species_id, offspring_count in offspring_allocations.items():
            species = self.species_manager.species[species_id]
            
                                         
            if species.members:
                                         
                species.sort_members()
                
                                                          
                elite_count = min(elitism, len(species.members), offspring_count)
                for i in range(elite_count):
                    new_genomes.append(species.members[i].copy())
                
                                                                            
                for i in range(elite_count, offspring_count):
                                           
                    child = self._create_child(species)
                    new_genomes.append(child)
        
                                                                  
        while len(new_genomes) < self.population_size:
                                        
            genome = Genome(self.config, self.innovation_tracker)
            new_genomes.append(genome)
        
                                                     
        if len(new_genomes) > self.population_size:
            new_genomes = new_genomes[:self.population_size]
        
        return new_genomes
    
    def _create_child(self, species) -> Genome:
        """
        Create a child genome through either mutation or crossover
        
        Args:
            species: The species to create a child from
            
        Returns:
            Genome: A new child genome
        """
                                                                           
        if len(species.members) > 1 and random.random() < 0.75:
                       
            parent1 = species.select_parent()
            parent2 = species.select_parent()
            
                                             
            while parent2 is parent1:
                parent2 = species.select_parent()
            
                                            
            child = Genome.crossover(parent1, parent2)
        else:
                                                        
            parent = species.select_parent()
            child = parent.copy()
        
                          
        child.mutate(self.innovation_tracker)
        
        return child
    
    def get_best_genomes(self, n: int = 5) -> List[Genome]:
        """
        Get the top n genomes in the population
        
        Args:
            n: Number of top genomes to return
            
        Returns:
            List[Genome]: List of the top n genomes
        """
        sorted_genomes = sorted(self.genomes, key=lambda genome: genome.fitness, reverse=True)
        return sorted_genomes[:n]
    
    def save_checkpoint(self, filename: str) -> None:
        """
        Save the current state of the population to a file
        
        Args:
            filename: Path to save the checkpoint to
        """
                                        
        checkpoint = {
            'generation': self.generation,
            'innovation_tracker': self.innovation_tracker,
            'species_manager': self.species_manager,
            'genomes': self.genomes,
            'best_genome': self.best_genome,
            'best_fitness': self.best_fitness,
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'config': self.config
        }
        
                      
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    @classmethod
    def load_checkpoint(cls, filename: str, eval_function: Callable[[Genome], float]) -> 'Population':
        """
        Load a population from a checkpoint file
        
        Args:
            filename: Path to the checkpoint file
            eval_function: Function to evaluate a genome's fitness
            
        Returns:
            Population: A new population object loaded from the checkpoint
        """
                        
        with open(filename, 'rb') as f:
            checkpoint = pickle.load(f)
        
                                 
        population = cls(checkpoint['config'], eval_function)
        
                       
        population.generation = checkpoint['generation']
        population.innovation_tracker = checkpoint['innovation_tracker']
        population.species_manager = checkpoint['species_manager']
        population.genomes = checkpoint['genomes']
        population.best_genome = checkpoint['best_genome']
        population.best_fitness = checkpoint['best_fitness']
        population.best_fitness_history = checkpoint['best_fitness_history']
        population.avg_fitness_history = checkpoint['avg_fitness_history']
        
        return population
    
    def __str__(self) -> str:
        """String representation of the population"""
        avg_fitness = self.avg_fitness_history[-1] if self.avg_fitness_history else 0
        best_fitness = self.best_fitness_history[-1] if self.best_fitness_history else 0
        
        s = f"Population (gen={self.generation}, size={len(self.genomes)}):\n"
        s += f"  Avg fitness: {avg_fitness:.2f}\n"
        s += f"  Best fitness: {best_fitness:.2f}\n"
        s += f"  Species: {len(self.species_manager.species)}\n"
        
        return s
