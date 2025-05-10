"""
Main Entry Point for NEAT Pendulum

This module provides the main entry point for training, testing, and visualizing
neural networks evolved by NEAT to balance a double pendulum on a cart.
"""

import os
import argparse
import toml
import pickle
import numpy as np
from typing import List, Dict, Any

from neat.genome import Genome
from neat.network import create_feed_forward_network
from physics.pendulum import DoublePendulumCart
from visualization.renderer import create_renderer


def main():
    """Main entry point for the NEAT Pendulum application"""
                                  
    parser = argparse.ArgumentParser(description='NEAT Pendulum on Cart')
    parser.add_argument('--config', type=str, default='config.toml',
                      help='Path to configuration file')
    parser.add_argument('--mode', type=str, default='train',
                      choices=['train', 'test', 'visualize'],
                      help='Mode to run in (train, test, or visualize)')
    parser.add_argument('--checkpoint', type=str,
                      help='Path to checkpoint file for testing or visualization')
    parser.add_argument('--output', type=str, default='output/videos/pendulum_evolution.mp4',
                      help='Path to output video file for visualization')
    args = parser.parse_args()
    
                        
    config = toml.load(args.config)
    
                           
    if args.mode == 'train':
        train(config, args.config)
    elif args.mode == 'test':
        test(config, args.checkpoint)
    elif args.mode == 'visualize':
        visualize(config, args.checkpoint, args.output)
    else:
        print(f"Invalid mode: {args.mode}")


def train(config: Dict[str, Any], config_path: str):
    """
    Train the NEAT algorithm to balance the pendulum
    
    Args:
        config: Configuration dictionary
        config_path: Path to configuration file
    """
    from train import train_neat
    
    print("Starting training...")
    population, best_genomes = train_neat(
        config_path=config_path,
        checkpoint_frequency=10,
        render_frequency=20
    )
    
                               
    if best_genomes:
        print("Visualizing best genome...")
        renderer = create_renderer(config_path)
        renderer.visualize_genome(best_genomes[0])
    else:
        print("No genomes evolved!")


def test(config: Dict[str, Any], checkpoint_path: str):
    """
    Test a trained genome on the pendulum
    
    Args:
        config: Configuration dictionary
        checkpoint_path: Path to checkpoint file
    """
                                     
                              
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found: {checkpoint_path}")
        return
    
                                
    try:
        print(f"Loading checkpoint from {checkpoint_path}...")
        
                                                
        try:
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
            
                                      
            if isinstance(data, list) and len(data) > 0:
                print(f"Loaded {len(data)} genomes from checkpoint")
                                                               
                genome = data[0]
                if not hasattr(genome, 'fitness') or not hasattr(genome, 'connections'):
                    print("Warning: Loaded genome appears invalid, missing expected attributes")
            else:
                                                                       
                print("Not a list of genomes, attempting to load as Population checkpoint...")
                from train import evaluate_genome
                from neat.population import Population
                population = Population.load_checkpoint(
                    checkpoint_path,
                    lambda g: evaluate_genome(g, config)
                )
                genome = population.best_genome
                print(f"Loaded population checkpoint with {len(population.species_manager.species)} species")
        except Exception as inner_e:
            print(f"Error during initial loading, trying alternative method: {inner_e}")
                                            
            from train import evaluate_genome
            from neat.population import Population
            population = Population.load_checkpoint(
                checkpoint_path,
                lambda g: evaluate_genome(g, config)
            )
            genome = population.best_genome
        
                                    
        if genome is None:
            print("No valid genome found in checkpoint")
            return
        
                             
        print(f"Using genome with fitness: {genome.fitness:.2f}")
        print(f"Genome has {len(genome.nodes)} nodes and {len(genome.connections)} connections")
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Check if the checkpoint file is valid and try again")
        return
    
                                         
    pendulum = DoublePendulumCart(config)
    network = create_feed_forward_network(genome)
    
                    
    observation = pendulum.reset()
    
                    
    total_reward = 0
    while not pendulum.done:
                                 
        action = network.activate(observation)[0]
        
                     
        observation, reward, done, info = pendulum.step(action)
        
        total_reward += reward
    
    print(f"Test complete. Total reward: {total_reward:.2f}")
    print(f"Steps: {pendulum.steps}")
    
                        
    renderer = create_renderer(config)
    
                                             
    pendulum = DoublePendulumCart(config)
    renderer.visualize_genome(genome, pendulum)


def visualize(config: Dict[str, Any], checkpoint_path: str, output_path: str):
    """
    Create a visualization of the best genomes
    
    Args:
        config: Configuration dictionary
        checkpoint_path: Path to checkpoint file
        output_path: Path to output video file
    """
                                          
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
                  
                                                             
    try:
        print(f"Loading checkpoint for visualization from {checkpoint_path}...")
        
                                                
        try:
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
            
                                      
            if isinstance(data, list) and len(data) > 0:
                print(f"Loaded {len(data)} genomes from checkpoint for visualization")
                genomes = data
                                  
                for i, genome in enumerate(genomes):
                    if not hasattr(genome, 'fitness') or not hasattr(genome, 'connections'):
                        print(f"Warning: Genome {i} appears invalid, missing expected attributes")
            else:
                                                                       
                print("Not a list of genomes, loading as Population checkpoint...")
                from train import evaluate_genome
                from neat.population import Population
                population = Population.load_checkpoint(
                    checkpoint_path,
                    lambda g: evaluate_genome(g, config)
                )
                genomes = population.get_best_genomes(
                    config['visualization']['best_networks_to_save']
                )
                print(f"Loaded {len(genomes)} best genomes from population checkpoint")
        except Exception as inner_e:
            print(f"Error during initial loading, trying alternative method: {inner_e}")
                                            
            from train import evaluate_genome
            from neat.population import Population
            population = Population.load_checkpoint(
                checkpoint_path,
                lambda g: evaluate_genome(g, config)
            )
            genomes = population.get_best_genomes(
                config['visualization']['best_networks_to_save']
            )
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Check if the checkpoint file is valid and try again")
        return
    
                                           
    if not genomes:
        print("No genomes to visualize")
        return
    
                     
    renderer = create_renderer(config)
    
                            
    renderer.visualize_best_genomes(
        genomes,
        output_dir=os.path.dirname(output_path),
        video_filename=os.path.basename(output_path)
    )
    
                                      
    from neat.visualization import visualize_genome_set
    visualize_genome_set(genomes, 'output/networks')


if __name__ == "__main__":
    main()
