"""
Trail Effects Module

This module provides functionality for creating modern motion trails
in the pendulum visualization, giving a more dynamic and visually
appealing representation of movement.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from collections import deque

from visualization.colors import RGB, alpha_blend, interpolate_color


class TrailPoint:
    """
    Represents a single point in a motion trail
    
    Stores position and time information for rendering trails
    with fade effects.
    """
    
    def __init__(self, position: np.ndarray, time: float = 0.0):
        """
        Initialize a trail point
        
        Args:
            position: 2D position of the point
            time: Time when this point was created
        """
        self.position = position.copy()
        self.time = time


class Trail:
    """
    Motion Trail for Dynamic Objects
    
    Maintains a history of positions for an object and provides
    methods for rendering motion trails with fade effects.
    """
    
    def __init__(self, max_length: int = 100, decay_rate: float = 0.95):
        """
        Initialize a motion trail
        
        Args:
            max_length: Maximum number of points in the trail
            decay_rate: Rate at which trail points fade (0-1)
        """
        self.max_length = max_length
        self.decay_rate = decay_rate
        self.points: deque = deque(maxlen=max_length)
        self.time = 0.0
    
    def add_point(self, position: np.ndarray) -> None:
        """
        Add a new point to the trail
        
        Args:
            position: 2D position of the point
        """
        self.time += 1.0
        self.points.append(TrailPoint(position, self.time))
    
    def clear(self) -> None:
        """Clear all points in the trail"""
        self.points.clear()
        self.time = 0.0
    
    def get_points(self) -> List[Tuple[np.ndarray, float]]:
        """
        Get all points with their opacity values
        
        Returns:
            List[Tuple[np.ndarray, float]]: List of (position, opacity) tuples
        """
        result = []
        
        if not self.points:
            return result
        
                                                       
        newest_time = self.points[-1].time
        for point in self.points:
                                                                       
            age = (newest_time - point.time) / self.max_length
            
                                                                           
            opacity = (1.0 - age) ** 2                                         
            
            result.append((point.position, opacity))
        
        return result
    
    def get_line_segments(self, min_opacity: float = 0.05, max_opacity: float = 0.9) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Get line segments for rendering the trail with varying opacity
        
        Args:
            min_opacity: Minimum opacity for oldest trail segments
            max_opacity: Maximum opacity for newest trail segments
            
        Returns:
            List[Tuple[np.ndarray, np.ndarray, float]]: List of (start, end, opacity) tuples
        """
        result = []
        
        if len(self.points) < 2:
            return result
        
                                             
        points_list = list(self.points)
        newest_time = points_list[-1].time
        oldest_time = points_list[0].time
        time_range = max(1.0, newest_time - oldest_time)
        
        for i in range(len(points_list) - 1):
                                            
            time_factor = (points_list[i].time - oldest_time) / time_range
            opacity = min_opacity + (max_opacity - min_opacity) * time_factor
            
                              
            result.append((
                points_list[i].position,
                points_list[i + 1].position,
                opacity
            ))
        
        return result


class MultiColorTrail(Trail):
    """
    Motion Trail with Color Gradient Effects
    
    Extends the basic Trail class to support color gradients along the trail,
    providing even more visually appealing motion effects.
    """
    
    def __init__(self, max_length: int = 100, decay_rate: float = 0.95,
                 start_color: RGB = (255, 0, 0), end_color: RGB = (0, 0, 255)):
        """
        Initialize a multi-color motion trail
        
        Args:
            max_length: Maximum number of points in the trail
            decay_rate: Rate at which trail points fade (0-1)
            start_color: Color for the newest part of the trail
            end_color: Color for the oldest part of the trail
        """
        super().__init__(max_length, decay_rate)
        self.start_color = start_color
        self.end_color = end_color
    
    def get_colored_line_segments(self, min_opacity: float = 0.05, max_opacity: float = 0.9) -> List[Tuple[np.ndarray, np.ndarray, RGB, float]]:
        """
        Get line segments with color gradient for rendering
        
        Args:
            min_opacity: Minimum opacity for oldest trail segments
            max_opacity: Maximum opacity for newest trail segments
            
        Returns:
            List[Tuple[np.ndarray, np.ndarray, RGB, float]]: List of (start, end, color, opacity) tuples
        """
        result = []
        
        if len(self.points) < 2:
            return result
        
                                                       
        points_list = list(self.points)
        newest_time = points_list[-1].time
        oldest_time = points_list[0].time
        time_range = max(1.0, newest_time - oldest_time)
        
        for i in range(len(points_list) - 1):
                                                      
            time_factor = (points_list[i].time - oldest_time) / time_range
            opacity = min_opacity + (max_opacity - min_opacity) * time_factor
            color = interpolate_color(self.end_color, self.start_color, time_factor)
            
                                         
            result.append((
                points_list[i].position,
                points_list[i + 1].position,
                color,
                opacity
            ))
        
        return result
