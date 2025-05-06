import numpy as np
import matplotlib.pyplot as plt

def plot_3d(func, ax=None, xlim=(-5,5), ylim=(-5,5), resolution=100):
    x = np.linspace(*xlim, resolution)
    y = np.linspace(*ylim, resolution)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        show_plot = True
    else:
        ax.clear()
        show_plot = False
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if show_plot:
        plt.show()
    return ax
