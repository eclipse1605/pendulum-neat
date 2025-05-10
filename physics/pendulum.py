"""
Double Pendulum on Cart Physics Module

This module implements the physics simulation for a double pendulum mounted
on a cart system. It uses Runge-Kutta 4th order method for numerical integration
of the equations of motion.
"""

import numpy as np
import toml
from typing import Tuple, List, Dict, Any


class DoublePendulumCart:
    """
    Double Pendulum on Cart Physics Simulation
    
    This class simulates the physics of a double pendulum mounted on a cart.
    The simulation uses Runge-Kutta 4th order method for accurate numerical integration.
    """
    
    def __init__(self, config_path: str = 'config.toml'):
        """
        Initialize the double pendulum on cart system
        
        Args:
            config_path: Path to the configuration file or config dictionary
        """
                                                                            
        if isinstance(config_path, str):
            config = toml.load(config_path)
        else:
            config = config_path
            
                                        
        self.gravity = config['physics']['gravity']
        self.timestep = config['physics']['timestep']
        self.cart_mass = config['physics']['cart_mass']
        self.pendulum1_mass = config['physics']['pendulum1_mass']
        self.pendulum1_length = config['physics']['pendulum1_length']
        self.pendulum2_mass = config['physics']['pendulum2_mass']
        self.pendulum2_length = config['physics']['pendulum2_length']
        self.max_force = config['physics']['max_force']
        self.friction = config['physics']['friction']
        self.track_limit = config['physics']['track_limit']
        
        self.simulation_steps = config['simulation']['steps']
        
                          
        self.reset()
        
                                                     
        self.history = []
        
    def reset(self) -> np.ndarray:
        """
        Reset the system to a slightly perturbed initial state
        
        Returns:
            observation: The initial observation vector
        """
                                                                          
                                                              
        
                                                          
                                                              
        initial_angle_range = 0.1                               
        
        self.state = np.array([
            np.random.uniform(-0.2, 0.2),                                       
            0.0,                 
            np.pi + np.random.uniform(-initial_angle_range, initial_angle_range),                                   
            np.random.uniform(-0.05, 0.05),                                  
            np.pi + np.random.uniform(-initial_angle_range, initial_angle_range),                                   
            np.random.uniform(-0.05, 0.05)                                   
        ])
        
        self.steps = 0
        self.done = False
        self.reward = 0
        self.total_reward = 0
        self.history = [self.state.copy()]
        
        return self.get_observation()
    
    def get_observation(self) -> np.ndarray:
        """
        Convert the physical state to an observation vector for the neural network
        
        Returns:
            observation: Processed state vector for the neural network
        """
                                                       
        cart_pos = self.state[0] / self.track_limit
        cart_vel = self.state[1] / 10.0                      
        
                                                                                
        theta1 = self.state[2]
        theta2 = self.state[4]
        
        return np.array([
            cart_pos,
            cart_vel,
            np.sin(theta1),
            np.cos(theta1),
            self.state[3] / 10.0,                                 
            np.sin(theta2),
            np.cos(theta2),
            self.state[5] / 10.0                                  
        ])
    
    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Advance the simulation by one timestep
        
        Args:
            action: Force to apply to the cart, normalized in [-1, 1]
            
        Returns:
            observation: The new observation vector
            reward: The reward for this step
            done: Whether the episode is finished
            info: Additional information dictionary
        """
                               
        force = np.clip(action, -1.0, 1.0) * self.max_force
        
                                 
        x, x_dot, theta1, theta1_dot, theta2, theta2_dot = self.state
        
                                                                  
        k1 = self.derivatives(self.state, force)
        k2 = self.derivatives(self.state + 0.5 * self.timestep * k1, force)
        k3 = self.derivatives(self.state + 0.5 * self.timestep * k2, force)
        k4 = self.derivatives(self.state + self.timestep * k3, force)
        
                      
        self.state += self.timestep * (k1 + 2*k2 + 2*k3 + k4) / 6.0
        
                                      
        self.state[2] = ((self.state[2] + np.pi) % (2 * np.pi)) - np.pi
        self.state[4] = ((self.state[4] + np.pi) % (2 * np.pi)) - np.pi
        
                                              
        self.history.append(self.state.copy())
        
                                        
        if abs(self.state[0]) > self.track_limit:
            self.done = True
            self.reward = -10.0
        else:
                                                                              
            angle1_reward = np.cos(self.state[2] - np.pi)                                
            angle2_reward = np.cos(self.state[4] - np.pi)                                
            
                                                 
            pendulum_reward = (angle1_reward + 1.5 * angle2_reward) / 2.5
            
                                                
            position_penalty = abs(self.state[0] / self.track_limit) * 0.05
            
                                      
            velocity_penalty = (abs(self.state[1]) * 0.01 + 
                              abs(self.state[3]) * 0.005 + 
                              abs(self.state[5]) * 0.005)
            
                                      
            self.reward = pendulum_reward - position_penalty - velocity_penalty
        
        self.total_reward += self.reward
        self.steps += 1
        
                                            
        if self.steps >= self.simulation_steps:
            self.done = True
        
        info = {
            'step': self.steps,
            'total_reward': self.total_reward,
            'state': self.state.copy()
        }
        
        return self.get_observation(), self.reward, self.done, info
    
    def derivatives(self, state: np.ndarray, force: float) -> np.ndarray:
        """
        Calculate derivatives of the state variables
        
        Args:
            state: Current state [x, x_dot, theta1, theta1_dot, theta2, theta2_dot]
            force: Force applied to the cart
            
        Returns:
            derivatives: Derivatives of state variables
        """
        x, x_dot, theta1, theta1_dot, theta2, theta2_dot = state
        
                
        m0 = self.cart_mass
        m1 = self.pendulum1_mass
        m2 = self.pendulum2_mass
        
                 
        l1 = self.pendulum1_length
        l2 = self.pendulum2_length
        
                 
        g = self.gravity
        
                            
        b = self.friction
        
                                                                    
                                                                
        
                                       
        sin_theta1 = np.sin(theta1)
        cos_theta1 = np.cos(theta1)
        sin_theta2 = np.sin(theta2)
        cos_theta2 = np.cos(theta2)
        sin_diff = np.sin(theta1 - theta2)
        cos_diff = np.cos(theta1 - theta2)
        
                                                          
                                                               
        
                                                      
        C0 = m0 + m1 + m2
        C1 = m1 * l1 * cos_theta1 + m2 * l1 * cos_theta1
        C2 = m2 * l2 * cos_theta2
        C3 = m1 * l1 * sin_theta1 + m2 * l1 * sin_theta1
        C4 = m2 * l2 * sin_theta2
        C5 = m1 * l1**2 + m2 * l1**2
        C6 = m2 * l2**2
        C7 = m2 * l1 * l2 * cos_diff
        C8 = m2 * l1 * l2 * sin_diff
        
                                    
        N1 = -m1 * l1 * theta1_dot**2 * sin_theta1 - m2 * l1 * theta1_dot**2 * sin_theta1
        N2 = -m2 * l2 * theta2_dot**2 * sin_theta2
        N3 = m1 * g * sin_theta1 + m2 * g * sin_theta1
        N4 = m2 * g * sin_theta2
        N5 = force - b * x_dot                              
        
                                                        
        a11 = C0
        a12 = C1
        a13 = C2
        a21 = C1
        a22 = C5
        a23 = C7
        a31 = C2
        a32 = C7
        a33 = C6
        
                                            
        b1 = N1 + N2 + N5
        b2 = C8 * theta2_dot**2 - N3
        b3 = -C8 * theta1_dot**2 - N4
        
                                                                                     
                                                          
        A = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
        B = np.array([b1, b2, b3])
        
                                                                   
                                                                                         
        try:
                                                         
            cond = np.linalg.cond(A)
            
                                                             
            if cond < 1e10:                                                                                
                x_ddot, theta1_ddot, theta2_ddot = np.linalg.solve(A, B)
                                                                        
            else:                                           
                x_ddot, theta1_ddot, theta2_ddot = np.linalg.lstsq(A, B, rcond=1e-10)[0]
                                                      
                dampening_factor = 0.98
                x_ddot *= dampening_factor
                theta1_ddot *= dampening_factor
                theta2_ddot *= dampening_factor
                
        except np.linalg.LinAlgError as e:
                                                        
            try:
                A_pinv = np.linalg.pinv(A, rcond=1e-10)
                solution = A_pinv.dot(B)
                x_ddot, theta1_ddot, theta2_ddot = solution
                
                                                                      
                x_ddot *= 0.9
                theta1_ddot *= 0.9
                theta2_ddot *= 0.9
                
            except Exception as pinv_error:
                                                                                              
                print(f"Numerical error in pendulum physics: {pinv_error}")
                x_ddot = np.random.uniform(-0.1, 0.1)
                theta1_ddot = np.random.uniform(-0.1, 0.1)
                theta2_ddot = np.random.uniform(-0.1, 0.1)
        
        return np.array([x_dot, x_ddot, theta1_dot, theta1_ddot, theta2_dot, theta2_ddot])
    
    def get_positions(self) -> Dict[str, np.ndarray]:
        """
        Calculate the positions of the cart and pendulum points for visualization
        
        Returns:
            positions: Dictionary containing cart and pendulum point positions
        """
        x, _, theta1, _, theta2, _ = self.state
        
                       
        cart_pos = np.array([x, 0])
        
                                                
        p1_x = x + self.pendulum1_length * np.sin(theta1)
        p1_y = -self.pendulum1_length * np.cos(theta1)
        pendulum1_pos = np.array([p1_x, p1_y])
        
                                                      
        p2_x = p1_x + self.pendulum2_length * np.sin(theta2)
        p2_y = p1_y - self.pendulum2_length * np.cos(theta2)
        pendulum2_pos = np.array([p2_x, p2_y])
        
        return {
            'cart': cart_pos,
            'pendulum1': pendulum1_pos,
            'pendulum2': pendulum2_pos
        }
    
    def get_history_positions(self) -> List[Dict[str, np.ndarray]]:
        """
        Calculate positions for the entire history of states
        
        Returns:
            history_positions: List of position dictionaries for each state in history
        """
        positions = []
        for state in self.history:
            x, _, theta1, _, theta2, _ = state
            
                           
            cart_pos = np.array([x, 0])
            
                                                    
            p1_x = x + self.pendulum1_length * np.sin(theta1)
            p1_y = -self.pendulum1_length * np.cos(theta1)
            pendulum1_pos = np.array([p1_x, p1_y])
            
                                                          
            p2_x = p1_x + self.pendulum2_length * np.sin(theta2)
            p2_y = p1_y - self.pendulum2_length * np.cos(theta2)
            pendulum2_pos = np.array([p2_x, p2_y])
            
            positions.append({
                'cart': cart_pos,
                'pendulum1': pendulum1_pos,
                'pendulum2': pendulum2_pos
            })
            
        return positions
