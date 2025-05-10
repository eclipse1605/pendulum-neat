"""
Color Utilities Module

This module provides utilities for working with colors in visualizations,
including color interpolation, conversion, and predefined color schemes.
"""

from typing import Tuple, List, Dict, Union
import numpy as np


                                                      
RGB = Tuple[int, int, int]

                                                                                 
RGBA = Tuple[int, int, int, float]


def hex_to_rgb(hex_color: str) -> RGB:
    """
    Convert a hex color string to an RGB tuple
    
    Args:
        hex_color: Hex color string (e.g., '#FF0000', 'FF0000')
        
    Returns:
        RGB: RGB color tuple
    """
                             
    hex_color = hex_color.lstrip('#')
    
                    
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: RGB) -> str:
    """
    Convert an RGB tuple to a hex color string
    
    Args:
        rgb: RGB color tuple
        
    Returns:
        str: Hex color string
    """
    return '#{:02x}{:02x}{:02x}'.format(*rgb)


def interpolate_color(color1: RGB, color2: RGB, factor: float) -> RGB:
    """
    Interpolate between two colors
    
    Args:
        color1: First RGB color tuple
        color2: Second RGB color tuple
        factor: Interpolation factor (0.0 = color1, 1.0 = color2)
        
    Returns:
        RGB: Interpolated RGB color
    """
                                
    factor = max(0.0, min(1.0, factor))
    
                                
    r = int(color1[0] + factor * (color2[0] - color1[0]))
    g = int(color1[1] + factor * (color2[1] - color1[1]))
    b = int(color1[2] + factor * (color2[2] - color1[2]))
    
    return (r, g, b)


def interpolate_colors(color1: RGB, color2: RGB, steps: int) -> List[RGB]:
    """
    Generate a list of colors interpolating from color1 to color2
    
    Args:
        color1: First RGB color tuple
        color2: Second RGB color tuple
        steps: Number of color steps to generate
        
    Returns:
        List[RGB]: List of interpolated RGB colors
    """
    colors = []
    for i in range(steps):
        factor = i / (steps - 1) if steps > 1 else 0
        colors.append(interpolate_color(color1, color2, factor))
    return colors


def brighten_color(color: RGB, factor: float) -> RGB:
    """
    Brighten a color by a factor
    
    Args:
        color: RGB color tuple
        factor: Brightening factor (0.0 = no change, 1.0 = white)
        
    Returns:
        RGB: Brightened RGB color
    """
                               
    return interpolate_color(color, (255, 255, 255), factor)


def darken_color(color: RGB, factor: float) -> RGB:
    """
    Darken a color by a factor
    
    Args:
        color: RGB color tuple
        factor: Darkening factor (0.0 = no change, 1.0 = black)
        
    Returns:
        RGB: Darkened RGB color
    """
                               
    return interpolate_color(color, (0, 0, 0), factor)


def alpha_blend(color: RGB, background: RGB, alpha: float) -> RGB:
    """
    Blend a color with a background color using alpha
    
    Args:
        color: Foreground RGB color tuple
        background: Background RGB color tuple
        alpha: Alpha value (0.0 = fully transparent, 1.0 = fully opaque)
        
    Returns:
        RGB: Blended RGB color
    """
                               
    alpha = max(0.0, min(1.0, alpha))
    
                          
    r = int(color[0] * alpha + background[0] * (1 - alpha))
    g = int(color[1] * alpha + background[1] * (1 - alpha))
    b = int(color[2] * alpha + background[2] * (1 - alpha))
    
    return (r, g, b)


                                  
class ColorScheme:
    """
    Predefined color schemes for visualizations
    """
                                          
    BACKGROUND = (10, 15, 20)
    
                
    GRID = (50, 50, 50)
    
                              
    CART = (0, 191, 255)                 
    PENDULUM1 = (255, 165, 0)          
    PENDULUM2 = (255, 140, 0)               
    
                  
    TRAIL = (255, 69, 0)              
    
                 
    TEXT = (255, 255, 255)         
    TEXT_HIGHLIGHT = (255, 255, 0)          
    
                                                        
    SPECIES_COLORS = [
        (255, 0, 0),         
        (0, 255, 0),           
        (0, 0, 255),          
        (255, 255, 0),          
        (255, 0, 255),           
        (0, 255, 255),        
        (255, 128, 0),          
        (128, 0, 255),          
        (0, 255, 128),        
        (128, 255, 0),        
    ]
    
    @classmethod
    def get_species_color(cls, species_id: int) -> RGB:
        """
        Get a color for a species
        
        Args:
            species_id: Species ID
            
        Returns:
            RGB: Color for the species
        """
        return cls.SPECIES_COLORS[species_id % len(cls.SPECIES_COLORS)]
    
    @classmethod
    def get_fitness_color(cls, fitness: float, max_fitness: float) -> RGB:
        """
        Get a color representing a fitness value
        
        Args:
            fitness: Fitness value
            max_fitness: Maximum fitness value
            
        Returns:
            RGB: Color representing the fitness (red to green gradient)
        """
                                     
        normalized = min(1.0, max(0.0, fitness / max_fitness)) if max_fitness > 0 else 0
        
                               
        return interpolate_color((255, 0, 0), (0, 255, 0), normalized)
    
    @classmethod
    def from_config(cls, config: Dict) -> None:
        """
        Update color scheme from configuration
        
        Args:
            config: Configuration dictionary
        """
                                                        
        if 'visualization' in config:
            vis_config = config['visualization']
            
            if 'background_color' in vis_config:
                cls.BACKGROUND = tuple(vis_config['background_color'])
            
            if 'grid_color' in vis_config:
                cls.GRID = tuple(vis_config['grid_color'])
            
            if 'cart_color' in vis_config:
                cls.CART = tuple(vis_config['cart_color'])
            
            if 'pendulum_color' in vis_config:
                cls.PENDULUM1 = tuple(vis_config['pendulum_color'])
                                                              
                cls.PENDULUM2 = darken_color(cls.PENDULUM1, 0.2)
            
            if 'trail_color' in vis_config:
                cls.TRAIL = tuple(vis_config['trail_color'])
            
            if 'text_color' in vis_config:
                cls.TEXT = tuple(vis_config['text_color'])
                                                           
                cls.TEXT_HIGHLIGHT = brighten_color(cls.TEXT, 0.5)
