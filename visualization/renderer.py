"""
Renderer Module

This module implements visualization for the double pendulum on cart system,
including real-time rendering and MP4 video creation with modern effects.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import deque
import subprocess
import toml

from physics.pendulum import DoublePendulumCart
from neat.genome import Genome
from neat.network import create_feed_forward_network
from visualization.trail import Trail, MultiColorTrail
from visualization.colors import ColorScheme, RGB, interpolate_color, alpha_blend


class PendulumRenderer:
    """
    Renderer for the Double Pendulum on Cart system
    
    Provides methods for real-time rendering and video creation with
    modern visual effects like trails and dynamic lighting.
    """
    
    def __init__(self, config_path: str = 'config.toml'):
        """
        Initialize the renderer
        
        Args:
            config_path: Path to the configuration file
        """
                            
        self.config = toml.load(config_path)
        ColorScheme.from_config(self.config)
        
                                
        self.width = self.config['visualization']['width']
        self.height = self.config['visualization']['height']
        self.dpi = 100
        self.fig = plt.figure(figsize=(self.width / self.dpi, self.height / self.dpi), dpi=self.dpi)
        self.ax = self.fig.add_subplot(111)
        
                                             
        self.trail_length = self.config['visualization']['trail_length']
        self.fps = self.config['visualization']['fps']
        
                       
        self.pendulum1_trail = MultiColorTrail(
            max_length=self.trail_length,
            start_color=ColorScheme.PENDULUM1,
            end_color=ColorScheme.TRAIL
        )
        self.pendulum2_trail = MultiColorTrail(
            max_length=self.trail_length,
            start_color=ColorScheme.PENDULUM2,
            end_color=ColorScheme.TRAIL
        )
        
                                      
        self.cart = None
        self.pendulum1 = None
        self.pendulum2 = None
        self.trail_lines = []
        
                                
        self._setup_plot()
    
    def _setup_plot(self) -> None:
        """Set up the plot aesthetics for modern visualization"""
                              
        self.fig.set_facecolor(tuple(c/255.0 for c in ColorScheme.BACKGROUND))
        self.ax.set_facecolor(tuple(c/255.0 for c in ColorScheme.BACKGROUND))
        
                     
        self.ax.grid(True, alpha=0.3, color=tuple(c/255.0 for c in ColorScheme.GRID))
        
                                                                    
        track_limit = self.config['physics']['track_limit']
        pendulum1_length = self.config['physics']['pendulum1_length']
        pendulum2_length = self.config['physics']['pendulum2_length']
        
                          
        max_length = pendulum1_length + pendulum2_length
        x_padding = max(1.0, track_limit * 0.2)
        y_padding = max(0.5, max_length * 0.2)
        
        self.ax.set_xlim(-track_limit - x_padding, track_limit + x_padding)
        self.ax.set_ylim(-max_length - y_padding, y_padding)
        
                            
        self.ax.set_xlabel('Position (m)', color=tuple(c/255.0 for c in ColorScheme.TEXT))
        self.ax.set_ylabel('Height (m)', color=tuple(c/255.0 for c in ColorScheme.TEXT))
        
                         
        self.ax.tick_params(colors=tuple(c/255.0 for c in ColorScheme.TEXT))
        
                         
        self.ax.axhline(y=0, color=tuple(c/255.0 for c in ColorScheme.GRID), linestyle='-', linewidth=2, alpha=0.7)
        
                           
        limit_color = tuple(c/255.0 for c in ColorScheme.GRID)
        self.ax.axvline(x=-track_limit, color=limit_color, linestyle='--', linewidth=1, alpha=0.5)
        self.ax.axvline(x=track_limit, color=limit_color, linestyle='--', linewidth=1, alpha=0.5)
        
                       
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        
                       
        self.fig.tight_layout()
    
    def _init_animation(self) -> List:
        """
        Initialize animation objects
        
        Returns:
            List: List of artists for animation
        """
                                 
        cart_width = 0.5
        cart_height = 0.2
        cart_color = tuple(c/255.0 for c in ColorScheme.CART)
        self.cart = Rectangle(
            (-cart_width/2, -cart_height/2), cart_width, cart_height,
            color=cart_color, ec='black', lw=1, zorder=3
        )
        self.ax.add_patch(self.cart)
        
                                         
        self.pendulum1_joint = Circle(
            (0, 0), radius=0.05, color=cart_color, ec='black', lw=1, zorder=4
        )
        self.ax.add_patch(self.pendulum1_joint)
        
        pendulum1_color = tuple(c/255.0 for c in ColorScheme.PENDULUM1)
        self.pendulum1_rod = FancyArrowPatch(
            (0, 0), (0, -self.config['physics']['pendulum1_length']),
            linewidth=4, color=pendulum1_color, zorder=2,
            arrowstyle='-', shrinkA=0, shrinkB=0
        )
        self.ax.add_patch(self.pendulum1_rod)
        
        self.pendulum1_mass = Circle(
            (0, -self.config['physics']['pendulum1_length']), 
            radius=0.1, color=pendulum1_color, ec='black', lw=1, zorder=4
        )
        self.ax.add_patch(self.pendulum1_mass)
        
                         
        pendulum2_color = tuple(c/255.0 for c in ColorScheme.PENDULUM2)
        self.pendulum2_rod = FancyArrowPatch(
            (0, -self.config['physics']['pendulum1_length']), 
            (0, -self.config['physics']['pendulum1_length'] - self.config['physics']['pendulum2_length']),
            linewidth=3, color=pendulum2_color, zorder=2,
            arrowstyle='-', shrinkA=0, shrinkB=0
        )
        self.ax.add_patch(self.pendulum2_rod)
        
        self.pendulum2_mass = Circle(
            (0, -self.config['physics']['pendulum1_length'] - self.config['physics']['pendulum2_length']), 
            radius=0.08, color=pendulum2_color, ec='black', lw=1, zorder=4
        )
        self.ax.add_patch(self.pendulum2_mass)
        
                           
                                                            
        
                                          
        self.title = self.ax.set_title(
            "Double Pendulum on Cart",
            color=tuple(c/255.0 for c in ColorScheme.TEXT),
            fontsize=12
        )
        
                                                         
        self.info_text = self.ax.text(
            0.02, 0.02, "",
            transform=self.ax.transAxes,
            color=tuple(c/255.0 for c in ColorScheme.TEXT),
            fontsize=10,
            verticalalignment='bottom'
        )
        
                                                    
        return [
            self.cart, self.pendulum1_joint, self.pendulum1_rod, self.pendulum1_mass,
            self.pendulum2_rod, self.pendulum2_mass, self.title, self.info_text
        ]
    
    def _update_animation(self, frame: int, pendulum: DoublePendulumCart, 
                         genome: Optional[Genome] = None, 
                         network = None) -> List:
        """
        Update animation for a frame
        
        Args:
            frame: Frame number
            pendulum: Double pendulum on cart instance
            genome: Genome being visualized (optional)
            network: Neural network for control (optional)
            
        Returns:
            List: List of updated artists
        """
                                                                  
        if frame == 0:
            self.pendulum1_trail.clear()
            self.pendulum2_trail.clear()
            for line in self.trail_lines:
                line.remove()
            self.trail_lines = []
        
                               
        positions = pendulum.get_positions()
        cart_pos = positions['cart']
        p1_pos = positions['pendulum1']
        p2_pos = positions['pendulum2']
        
                              
        self.cart.set_xy((cart_pos[0] - 0.25, -0.1))
        
                                         
        self.pendulum1_joint.center = (cart_pos[0], 0)
        
                              
        self.pendulum1_rod.set_positions((cart_pos[0], 0), (p1_pos[0], p1_pos[1]))
        
                               
        self.pendulum1_mass.center = (p1_pos[0], p1_pos[1])
        
                              
        self.pendulum2_rod.set_positions((p1_pos[0], p1_pos[1]), (p2_pos[0], p2_pos[1]))
        
                               
        self.pendulum2_mass.center = (p2_pos[0], p2_pos[1])
        
                       
        self.pendulum1_trail.add_point(np.array([p1_pos[0], p1_pos[1]]))
        self.pendulum2_trail.add_point(np.array([p2_pos[0], p2_pos[1]]))
        
                                
        for line in self.trail_lines:
            line.remove()
        self.trail_lines = []
        
                                                   
        for start, end, color, opacity in self.pendulum1_trail.get_colored_line_segments(
            min_opacity=self.config['visualization']['trail_opacity_min'],
            max_opacity=self.config['visualization']['trail_opacity_max']
        ):
            rgb_color = tuple(c/255.0 for c in color)
            line = self.ax.plot(
                [start[0], end[0]], [start[1], end[1]],
                color=rgb_color, alpha=opacity, linewidth=2, zorder=1
            )[0]
            self.trail_lines.append(line)
        
        for start, end, color, opacity in self.pendulum2_trail.get_colored_line_segments(
            min_opacity=self.config['visualization']['trail_opacity_min'],
            max_opacity=self.config['visualization']['trail_opacity_max']
        ):
            rgb_color = tuple(c/255.0 for c in color)
            line = self.ax.plot(
                [start[0], end[0]], [start[1], end[1]],
                color=rgb_color, alpha=opacity, linewidth=2, zorder=1
            )[0]
            self.trail_lines.append(line)
        
                                       
        if genome:
            self.title.set_text(f"Double Pendulum on Cart - Genome Fitness: {genome.fitness:.2f}")
        else:
            self.title.set_text(f"Double Pendulum on Cart - Step: {pendulum.steps}")
        
                          
        cart_pos = pendulum.state[0]
        cart_vel = pendulum.state[1]
        theta1 = pendulum.state[2]
        theta2 = pendulum.state[4]
        
                                                         
        theta1_deg = np.degrees(theta1) % 360
        theta2_deg = np.degrees(theta2) % 360
        
        info = (
            f"Cart: {cart_pos:.2f}m, {cart_vel:.2f}m/s\n"
            f"θ1: {theta1_deg:.1f}°, θ2: {theta2_deg:.1f}°\n"
            f"Reward: {pendulum.total_reward:.2f}"
        )
        self.info_text.set_text(info)
        
                                                          
        if frame < pendulum.simulation_steps - 1 and not pendulum.done and network:
            observation = pendulum.get_observation()
            action = network.activate(observation)[0]
            pendulum.step(action)
        
                                              
        artists = [
            self.cart, self.pendulum1_joint, self.pendulum1_rod, self.pendulum1_mass,
            self.pendulum2_rod, self.pendulum2_mass, self.title, self.info_text
        ]
        artists.extend(self.trail_lines)
        
        return artists
    
    def visualize_genome(self, genome: Genome, pendulum: Optional[DoublePendulumCart] = None, 
                        save_path: Optional[str] = None) -> None:
        """
        Visualize a genome controlling the pendulum
        
        Args:
            genome: Genome to visualize
            pendulum: Double pendulum instance (creates new one if None)
            save_path: Path to save the animation to (as MP4)
        """
                                         
        if pendulum is None:
            pendulum = DoublePendulumCart(self.config)
            pendulum.reset()
        
                                           
        try:
            network = create_feed_forward_network(genome)
        except Exception as e:
            print(f"Error creating network: {e}")
            return
        
                                          
        observation = pendulum.reset()
        
                                                            
                                                              
        if self.fig is not None and plt.fignum_exists(self.fig.number):
            plt.close(self.fig)
            
                              
        self.fig = plt.figure(figsize=(self.width / self.dpi, self.height / self.dpi), dpi=self.dpi)
        self.ax = self.fig.add_subplot(111)
        self._setup_plot()
        
                                                            
        if not pendulum.done:
            try:
                action = network.activate(observation)[0]
                pendulum.step(action)
            except Exception as e:
                print(f"Error in initial simulation step: {e}")
                return
        
                                              
        try:
            ani = animation.FuncAnimation(
                self.fig, self._update_animation,
                frames=min(500, self.config['simulation']['steps']),                                   
                fargs=(pendulum, genome, network),
                init_func=self._init_animation,
                blit=True, interval=1000/self.fps
            )
            
                                         
            if save_path:
                                                      
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                try:
                                                  
                    subprocess.run(['which', 'ffmpeg'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    
                                                     
                    writer = animation.FFMpegWriter(
                        fps=self.fps, metadata=dict(artist='NEAT'),
                        bitrate=5000
                    )
                    
                                                            
                    print(f"Saving animation to {save_path}...")
                    ani.save(save_path, writer=writer)
                    print(f"Animation saved to {save_path}")
                except subprocess.CalledProcessError:
                    print("Error: ffmpeg is not installed. Cannot save animation.")
                    print("Please install ffmpeg with 'sudo apt-get install ffmpeg' or equivalent.")
                                                 
                    try:
                        gif_path = save_path.replace('.mp4', '.gif')
                        print(f"Attempting to save as GIF to {gif_path}...")
                        writer = animation.PillowWriter(fps=20)                     
                        ani.save(gif_path, writer=writer)
                        print(f"Animation saved as GIF to {gif_path}")
                    except Exception as e:
                        print(f"Failed to save animation: {e}")
                finally:
                                                 
                    plt.close(self.fig)
            else:
                                
                plt.show()
        except Exception as e:
            print(f"Error in animation: {e}")
            plt.close(self.fig)
    
    def visualize_best_genomes(self, genomes: List[Genome], 
                              output_dir: str = 'output/videos', 
                              video_filename: str = 'pendulum_evolution.mp4') -> None:
        """
        Create a video showcasing the best genomes
        
        Args:
            genomes: List of genomes to visualize (sorted by fitness)
            output_dir: Directory to save the outputs
            video_filename: Filename for the final combined video
        """
                                 
        os.makedirs(output_dir, exist_ok=True)
        
                                                   
        temp_dir = os.path.join(output_dir, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
                                         
        video_paths = []
        for i, genome in enumerate(genomes):
                                             
            pendulum = DoublePendulumCart(self.config)
            pendulum.reset()
            
                               
            video_path = os.path.join(temp_dir, f'genome_{i}_fitness_{genome.fitness:.2f}.mp4')
            video_paths.append(video_path)
            
                            
            self.visualize_genome(genome, pendulum, video_path)
            
                            
            print(f"Generated video {i+1}/{len(genomes)} for genome with fitness {genome.fitness:.2f}")
        
                                          
        combined_path = os.path.join(output_dir, video_filename)
        self._combine_videos(video_paths, combined_path)
        
                                           
                       
                                 
        
        print(f"Combined video saved to {combined_path}")
    
    def _combine_videos(self, video_paths: List[str], output_path: str) -> None:
        """
        Combine multiple videos into a single MP4 file
        
        Args:
            video_paths: List of paths to videos to combine
            output_path: Path to save the combined video
        """
                                            
        try:
            subprocess.run(['which', 'ffmpeg'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError:
            print("Error: ffmpeg is not installed. Cannot combine videos.")
            print("Please install ffmpeg with 'sudo apt-get install ffmpeg' or equivalent.")
            return
        
                                                         
        valid_paths = []
        for path in video_paths:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                valid_paths.append(path)
            else:
                print(f"Warning: Video file {path} does not exist or is empty. Skipping.")
        
        if not valid_paths:
            print("Error: No valid video files to combine.")
            return
        
                                                         
        temp_list_path = os.path.join(os.path.dirname(output_path), 'temp_video_list.txt')
        with open(temp_list_path, 'w') as f:
            for path in valid_paths:
                                                             
                abs_path = os.path.abspath(path).replace("'", "'\\''")
                f.write(f"file '{abs_path}'\n")
        
                                                                 
        cmd = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', temp_list_path, '-c', 'copy', output_path
        ]
        
        try:
            result = subprocess.run(
                cmd, 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            print(f"Videos combined successfully to {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error combining videos: {e}")
            print(f"ffmpeg stderr: {e.stderr}")
                                                                     
            try:
                print("Attempting to combine videos with re-encoding...")
                cmd_alt = [
                    'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                    '-i', temp_list_path, '-c:v', 'libx264', '-crf', '23', 
                    '-preset', 'medium', output_path
                ]
                result = subprocess.run(cmd_alt, check=True)
                print(f"Videos combined with re-encoding to {output_path}")
            except subprocess.CalledProcessError as e2:
                print(f"All attempts to combine videos failed: {e2}")
        finally:
                                          
            if os.path.exists(temp_list_path):
                os.remove(temp_list_path)


                                         
def create_renderer(config_path: str = 'config.toml') -> PendulumRenderer:
    """
    Create a pendulum renderer from a configuration file
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        PendulumRenderer: A new pendulum renderer
    """
    return PendulumRenderer(config_path)
