"""
Training Module for NEAT Pendulum

This module implements the training process for the NEAT algorithm
to evolve neural networks capable of balancing the double pendulum on a cart.
"""

import os
import numpy as np
import time
import toml
from typing import Dict, List, Any, Tuple, Optional
import logging
import matplotlib.pyplot as plt
from datetime import datetime

from neat.genome import Genome
from neat.population import Population
from neat.network import create_feed_forward_network
from physics.pendulum import DoublePendulumCart


def setup_logger(log_dir: str = 'logs/training') -> logging.Logger:
    """
    Set up a logger for training process
    
    Args:
        log_dir: Directory to store log files
        
    Returns:
        logging.Logger: Configured logger instance
    """
                                          
    os.makedirs(log_dir, exist_ok=True)
    
                                        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
                      
    logger = logging.getLogger('neat_training')
    logger.setLevel(logging.INFO)
    
                         
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
                            
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
                      
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
                            
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def evaluate_genome(genome: Genome, config: Dict, render: bool = False) -> float:
    """
    Evaluate a genome's fitness by testing its controller on the pendulum
    
    Args:
        genome: Genome to evaluate
        config: Configuration dictionary
        render: Whether to render the evaluation
        
    Returns:
        float: Fitness score for the genome
    """
                                                                        
    try:
        pendulum = DoublePendulumCart(config)
    except Exception as e:
        logging.error(f"Failed to create pendulum environment: {e}")
        return 0.001                                                     
    
                                                                          
    try:
        network = create_feed_forward_network(genome)
    except Exception as e:
        logging.error(f"Network creation failed: {e}")
                                                                     
        return 0.01
    
                                                                                        
    try:
        observation = pendulum.reset()
    except Exception as e:
        logging.error(f"Environment reset failed: {e}")
        return 0.02
    
                                                                                     
    max_steps = config['simulation']['steps']
    step_count = 0
    total_reward = 0
    
                                                                          
    while not pendulum.done and step_count < max_steps:
                                                                        
        try:
            action = network.activate(observation)[0]
                                                                                                             
            action = max(-1.0, min(1.0, action))
        except Exception as e:
            logging.error(f"Network activation error: {e}")
                                                          
            return max(0.05, total_reward * 0.9)  
        
                                                                 
        try:
            observation, reward, done, info = pendulum.step(action)
            total_reward += reward
        except Exception as e:
            logging.error(f"Simulation step error: {e}")
                                                                
            return max(0.05, total_reward * 0.8)
        
        step_count += 1
    
                                                                  
    if step_count >= max_steps and not pendulum.done:
                                                            
        total_reward = pendulum.total_reward + 10.0
    else:
        total_reward = pendulum.total_reward
    
                                    
    return max(0, total_reward)                                    


def train_neat(config_path: str = 'config.toml', checkpoint_dir: str = 'output/models',
              checkpoint_frequency: int = 10, render_frequency: int = 0, 
              generation_timeout: int = 300, patience: int = 15) -> Tuple[Population, List[Genome]]:
    """
    Train the NEAT algorithm to solve the double pendulum on cart problem
    
    Args:
        config_path: Path to configuration file
        checkpoint_dir: Directory to save checkpoints to
        checkpoint_frequency: Save checkpoint every n generations
        render_frequency: Render best genome every n generations (0 to disable)
        
    Returns:
        Tuple[Population, List[Genome]]: Final population and list of best genomes
    """
                        
    config = toml.load(config_path)
    
                  
    logger = setup_logger()
    logger.info("Starting NEAT training with configuration from %s", config_path)
    
                                                     
    os.makedirs(checkpoint_dir, exist_ok=True)
    
                                                  
    def eval_function(genome: Genome) -> float:
        return evaluate_genome(genome, config)
    
                                      
    population = Population(config, eval_function)
    
                                
    best_genomes = []
                 
    start_time = time.time()
    
                                             
    max_generations = config['neat']['max_generations']
                                      
    os.makedirs(checkpoint_dir, exist_ok=True)
    
                               
    generations = []
    avg_fitness_history = []
    best_fitness_history = []
    best_genomes = []
    
                              
    stagnation_counter = 0
    best_fitness_ever = float('-inf')
                                                      
    plateau_threshold = config.get('neat', {}).get('plateau_threshold', 0.01)                            
                                                                                  
    prev_best_fitness = float('-inf')
    
                             
    for generation in range(max_generations):
        logger.info("Generation %d of %d", generation + 1, max_generations)
        
                                           
        start_gen_time = time.time()
        
        try:
                                                           
            avg_fitness, best_fitness = population.evolve()
            
                                                    
            gen_time = time.time() - start_gen_time
            if gen_time > generation_timeout:
                logger.warning("Generation %d took %.2f seconds (exceeded timeout of %d seconds)", 
                             generation + 1, gen_time, generation_timeout)
        except Exception as e:
            logger.error("Error during evolution: %s", str(e))
                                       
            emergency_path = os.path.join(checkpoint_dir, f"emergency_checkpoint_gen_{generation + 1}.pkl")
            population.save_checkpoint(emergency_path)
            logger.info("Saved emergency checkpoint to %s", emergency_path)
                                           
            continue
        
                     
        logger.info("  Average fitness: %.2f", avg_fitness)
        logger.info("  Best fitness: %.2f", best_fitness)
        logger.info("  Species count: %d", len(population.species_manager.species))
        
                                      
        generations.append(generation)
        avg_fitness_history.append(avg_fitness)
        best_fitness_history.append(best_fitness)
        
                                                         
        current_best = population.get_best_genomes(1)[0]
        
                                                      
                                           
        if current_best.fitness > best_fitness_ever * (1 + plateau_threshold):
            best_fitness_ever = current_best.fitness
            stagnation_counter = 0
            logger.info("  New best fitness: %.2f (significantly improved)", best_fitness_ever)
                                       
        elif prev_best_fitness > 0 and current_best.fitness < prev_best_fitness * (1 - plateau_threshold * 2):
                                                                                                 
            logger.warning("  Fitness decreased significantly from %.2f to %.2f", prev_best_fitness, current_best.fitness)
                                                                                              
            if current_best.fitness < prev_best_fitness * 0.7:            
                stagnation_counter = max(0, stagnation_counter - 1)                                       
                logger.info("  Reducing stagnation counter to %d due to large fitness drop", stagnation_counter)
        else:
                                   
            stagnation_counter += 1
            logger.info(f"  Stagnation counter: {stagnation_counter}/{patience}")
            
                                                         
        prev_best_fitness = current_best.fitness
        
                                  
        if not best_genomes or current_best.fitness > best_genomes[0].fitness:
            best_genomes = [current_best.copy()]
            logger.info("  New best genome with fitness %.2f", current_best.fitness)
        
                                      
        if checkpoint_frequency > 0 and (generation + 1) % checkpoint_frequency == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"neat_checkpoint_gen_{generation + 1}.pkl")
            population.save_checkpoint(checkpoint_path)
            logger.info("  Saved checkpoint to %s", checkpoint_path)
        
                                         
        if render_frequency > 0 and (generation + 1) % render_frequency == 0:
            logger.info("  Rendering best genome...")
                                                   
            from visualization.renderer import create_renderer
            renderer = create_renderer(config_path)
            renderer.visualize_genome(current_best)
        
                                      
        
                                                         
        if not config['neat']['no_fitness_termination'] and best_fitness >= fitness_threshold:
            logger.info("Reached fitness threshold of %.2f at generation %d", 
                      fitness_threshold, generation + 1)
            break
            
                                                       
        if patience > 0 and stagnation_counter >= patience:
            logger.info(f"Early stopping triggered after {stagnation_counter} generations without improvement")
                                                       
            early_stop_path = os.path.join(checkpoint_dir, f"early_stop_checkpoint_gen_{generation + 1}.pkl")
            population.save_checkpoint(early_stop_path)
            logger.info(f"Saved early stopping checkpoint to {early_stop_path}")
            break
    
                            
    elapsed = time.time() - start_time
    logger.info("Training completed in %.2f seconds (%.2f generations/sec)",
              elapsed, (generation + 1) / elapsed)
    
                           
    final_checkpoint_path = os.path.join(checkpoint_dir, "neat_checkpoint_final.pkl")
    population.save_checkpoint(final_checkpoint_path)
    logger.info("Saved final checkpoint to %s", final_checkpoint_path)
    
                           
    best_genomes_path = os.path.join(checkpoint_dir, "best_genomes.pkl")
    import pickle
    with open(best_genomes_path, 'wb') as f:
        pickle.dump(best_genomes, f)
    logger.info("Saved best genomes to %s", best_genomes_path)
    
                                                  
                                                                                
    fig = plt.figure(figsize=(12, 8))
    try:
                                
        plt.plot(generations, avg_fitness_history, label='Average Fitness', linewidth=2, alpha=0.8)
        plt.plot(generations, best_fitness_history, label='Best Fitness', linewidth=2)
        
                                                                           
        if len(generations) > 5:
            try:
                                                                 
                window = min(5, len(generations) // 4)
                if window > 0:
                    avg_smooth = np.convolve(avg_fitness_history, np.ones(window)/window, mode='valid')
                    best_smooth = np.convolve(best_fitness_history, np.ones(window)/window, mode='valid')
                    plt.plot(generations[window-1:], avg_smooth, 'k--', alpha=0.5, linewidth=1)
                    plt.plot(generations[window-1:], best_smooth, 'k--', alpha=0.5, linewidth=1)
            except Exception as e:
                logger.warning(f"Could not generate trend lines: {e}")
        
                                
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Fitness', fontsize=12)
        plt.title('NEAT Training Progress', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
                         
        plot_path = os.path.join('output', 'training_progress.png')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        logger.info("Saved training progress plot to %s", plot_path)
    finally:
                                                             
        plt.close(fig)
    
    return population, best_genomes


if __name__ == "__main__":
                              
    population, best_genomes = train_neat(
        config_path='config.toml',
        checkpoint_frequency=10,
        render_frequency=20,
        generation_timeout=120                                
    )
    
                                   
    if best_genomes:
        best_genome = best_genomes[0]
        print(f"Best genome fitness: {best_genome.fitness:.2f}")
        print(f"Best genome has {len(best_genome.nodes)} nodes and {len(best_genome.connections)} connections")
    else:
        print("No genomes evolved!")
