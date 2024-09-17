import numpy as np 
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_cylinder(ax, center, radius, height, rot, **kwargs):

    u = np.linspace(0, 2* np.pi, 50)
    h = np.linspace(0, height, 20)
    x = np.outer(radius * np.sin(u)+center[0], np.ones_like(h))
    y = np.outer(radius * np.cos(u)+center[1], np.ones_like(h))
    z = np.outer(np.ones_like(u), h + center[2] - 0.5 * height)
    
    #ax.plot_surface(x, y, np.ones_like(z) * 0.5*height)
    ax.plot_surface(x, y, z, **kwargs)
    
    # Generate data for the bottom disk
    u = np.linspace(0, 2 * np.pi, 50)
    x_bottom = radius * np.sin(u) + center[0]
    y_bottom = radius * np.cos(u) + center[1]
    z_bottom = np.ones_like(x_bottom) * (center[2] - 0.5 * height)
    
    # Plot the bottom disk
    ax.plot_trisurf(x_bottom, y_bottom, z_bottom, **kwargs)

    # Generate data for the top disk
    x_top = radius * np.sin(u) + center[0]
    y_top = radius * np.cos(u) + center[1]
    z_top = np.ones_like(x_top) * (center[2] + 0.5 * height)
    
    # Plot the top disk
    ax.plot_trisurf(x_top, y_top, z_top, **kwargs)