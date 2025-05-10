# Pendulum-NEAT

A physics simulation of a double pendulum on a cart controlled by neural networks evolved using the NEAT (NeuroEvolution of Augmenting Topologies) algorithm.

## Overview

This project simulates a double inverted pendulum mounted on a cart and uses the NEAT algorithm to evolve neural networks capable of balancing the pendulum system. The goal is to train neural networks that can apply appropriate forces to the cart to keep the double pendulum in an upright position.

The physics simulation uses Runge-Kutta 4th order numerical integration for accurate modeling of the double pendulum dynamics.

## Key Features

- Physics simulation of a double pendulum on a cart
- NEAT algorithm implementation for evolving neural network controllers
- Multiple operating modes: training, testing, and visualization
- Checkpoint system for saving and loading evolved neural networks
- Visualization tools for neural networks and pendulum dynamics
- Early stopping and stagnation detection for efficient training

## Project Structure

- `main.py` - Main entry point for the application (train, test, visualize)
- `train.py` - Implementation of the NEAT training algorithm
- `config.toml` - Configuration file for physics parameters and NEAT settings
- `physics/` - Physics simulation modules
  - `pendulum.py` - Double pendulum on cart physics implementation
- `neat/` - NEAT algorithm implementation
- `visualization/` - Visualization tools
- `output/` - Directory for output files (models, videos, plots)
- `logs/` - Training logs

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- PyTOML
- (Other dependencies as needed)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/pendulum-neat.git
   cd pendulum-neat
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Linux/Mac
   # OR
   .venv\Scripts\activate  # On Windows
   ```

3. Install the dependencies:
   ```
   pip install numpy matplotlib toml
   ```

## Usage

### Training Mode

Train neural networks to balance the pendulum:

```
python main.py --mode train --config config.toml
```

Optional arguments:
- `--config`: Path to configuration file (default: config.toml)

### Testing Mode

Test a previously trained neural network:

```
python main.py --mode test --config config.toml --checkpoint output/models/best_genomes.pkl
```

Required arguments:
- `--checkpoint`: Path to the saved neural network model

### Visualization Mode

Create a video visualization of a trained neural network:

```
python main.py --mode visualize --config config.toml --checkpoint output/models/best_genomes.pkl --output output/videos/pendulum_evolution.mp4
```

Required arguments:
- `--checkpoint`: Path to the saved neural network model

Optional arguments:
- `--output`: Path to the output video file (default: output/videos/pendulum_evolution.mp4)

## Configuration

The `config.toml` file contains all the configuration options for the physics simulation and the NEAT algorithm. Key settings include:

### Physics Settings

```toml
[physics]
gravity = 9.81
timestep = 0.01
cart_mass = 1.0
pendulum1_mass = 0.5
pendulum1_length = 1.0
pendulum2_mass = 0.5
pendulum2_length = 1.0
max_force = 10.0
friction = 0.1
track_limit = 2.5
```

### Simulation Settings

```toml
[simulation]
steps = 1000
```

### NEAT Algorithm Settings

```toml
[neat]
population_size = 150
max_generations = 100
fitness_threshold = 500.0
no_fitness_termination = false
```

### Visualization Settings

```toml
[visualization]
best_networks_to_save = 5
```

## Performance Tips

- Adjust the population size and max generations in the config file to balance between training time and solution quality
- If training is taking too long, consider decreasing the simulation steps
- Use the early stopping feature to avoid wasting computation once fitness plateaus
- Running with visualization during training can significantly slow down the process

## Troubleshooting

- If you encounter numerical instability in the physics simulation, try adjusting the timestep to a smaller value
- For "Population extinction" errors, increase the population size or adjust the NEAT parameters to encourage more diversity
- Check logs in the `logs/` directory for detailed information about training progress and errors

## License

[MIT License](LICENSE)

## Acknowledgments

- The NEAT algorithm is based on the paper by Kenneth O. Stanley and Risto Miikkulainen: "Evolving Neural Networks through Augmenting Topologies" (2002)
