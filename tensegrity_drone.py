import numpy as np
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt

class TensegrityDrone(object):

    def __init__(self, l=1.0, m=1.0, I=0.1, p=[0, 0, 0]) -> None:

        # Pose
        self.position = np.array(p)
        self.orientation = R.from_euler(seq='xyz',
                                        angles=[0.0, np.pi/4, 0.0])
        
        # Inertial parameters
        self.m = m 
        self.I = I
        
        # Length of rods
        self.l = l

        # Prop Radius (for visualization only)
        self.r  = 0.1*l

        # Vertices of the icosahedron w.r.t. its centroid
        self.vertices = self.orientation.apply(l * (np.array([
                                    [0.00, 0.50, 0.25], 
                                    [1.00, 0.50, 0.25],
                                    [0.00, 0.50, 0.75], 
                                    [1.00, 0.50, 0.75], 
                                    [0.25, 0.00, 0.50], 
                                    [0.25, 1.00, 0.50], 
                                    [0.75, 0.00, 0.50], 
                                    [0.75, 1.00, 0.50], 
                                    [0.50, 0.25, 0.00], 
                                    [0.50, 0.25, 1.00], 
                                    [0.50, 0.75, 0.00], 
                                    [0.50, 0.75, 1.00]
                                ])  - 0.5))   + self.position
        
        assert(np.linalg.norm(self.vertices[0]-self.vertices[2]) == 0.5 * l)
        assert(np.linalg.norm(self.vertices[2]-self.vertices[3]) == l)

        # Locations of propellers w.r.t. the centroid
        self.propellers =  self.orientation.apply(np.array([
            [0.25 * l, 0.25 * l, 0.05 * l],
            [0.25 * l, -0.25 * l, 0.05 * l],
            [-0.25 * l, 0.25 * l, 0.05 * l],
            [-0.25 * l, -0.25 * l, 0.05 * l]
        ])) + self.position

        # Strings indeces (for visualization only)
        self.strings = [ 
                    [0, 4], [0, 5], [0, 8], [0, 10], 
                    [1, 6], [1, 7], [1, 8], [1, 10], 
                    [2, 4], [2, 5], [2, 9], [2, 11], 
                    [3, 6], [3, 7], [3, 9], [3, 11], 
                    [4, 8], [4, 9], [5, 10], [5, 11], 
                    [6, 8], [6, 9], [7, 10], [7, 11]
                    ]
        
        # Setting up plotting stuff
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')

    def plot_tensegrity_drone(self) -> None:
        self.ax.clear()

        self.ax.scatter(self.position[0],
                        self.position[1],
                        self.position[2], 
                        s=45, alpha=0.75,
                        color="black")
        self.ax.text(self.position[0]+0.01,
                     self.position[1]+0.01,
                     self.position[2]+0.01,
                     f"CoM")

        self.ax.scatter(self.vertices[:,0], 
           self.vertices[:,1], 
           self.vertices[:,2])

        for i in range(len(self.propellers)):
            tau = np.linspace(-np.pi, np.pi, 25)
            circle = self.orientation.apply(np.array([self.r *np.sin(tau),
                                                      self.r * np.cos(tau),
                                                      np.zeros_like(tau)]).T) \
                        + self.propellers[i, :]
            self.ax.plot(circle[:, 0], circle[:, 1], circle[:, 2], color="red")

        for i in range(len(self.vertices)):
            self.ax.text(self.vertices[i, 0], 
                    self.vertices[i, 1], 
                    self.vertices[i, 2], 
                    f"Vertex {i+1}")

        for i in range(0, 11, 2):
            segment = np.array([self.vertices[i,:],self.vertices[i+1,:]])
            self.ax.plot(segment[:,0],segment[:,1],segment[:,2], color="black")

        for idx in self.strings:
            string = np.array([self.vertices[idx[0],:], self.vertices[idx[1], :]])
            self.ax.plot(string[:,0], string[:,1], string[:,2], color="grey")

        self.ax.set_xlabel(f"x[m]")
        self.ax.set_ylabel(f"y[m]")
        self.ax.set_zlabel(f"z[m]")

        self.set_axes_equal(self.ax)


    def set_axes_equal(self, ax):
        """
        Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc.

        Input
        ax: a matplotlib axis, e.g., as output from plt.gca().
        """

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


    def save(self, name: str) -> None:
        self.fig.savefig(name)
    def show(self):
        plt.show()

