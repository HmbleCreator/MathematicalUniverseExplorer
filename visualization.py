import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Visualizer:
    def __init__(self):
        self.figure = plt.figure()
        
    def create_3d_plot(self, x, y, z, title='3D Plot', cmap='viridis'):
        """Create a 3D surface plot"""
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(x, y, z, cmap=cmap, linewidth=0, antialiased=True)
        
        # Add color bar
        self.figure.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        return self.figure