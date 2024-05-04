import numpy as np
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt

class TensegrityDrone(object):

    def __init__(self, p=[0.0, 0.0, 0.0], angles=[0.0, 0.0, 0.0],
                 l=0.22, g=9.81, th=7.50e-8, yd=0.009975,
                 m=0.315, I=[6.21e-4, 9.62e-4, 9.22e-4],
                 plot=False) -> None:

        # Pose
        self.position = np.array(p)
        self.orientation = R.from_euler(seq='xyz',
                                        angles=angles)
        
        # Completed trajectory
        self.position_hist = np.array([p[0:3]])
        
        # Gravity constant
        self.g = g

        # Speed^2 to Thrust constant
        self.th = th
        # Force to Yaw moment constant
        self.yd = yd

        # Inertial parameters
        self.m = m         # total mass
        self.I = np.array(I)   # Diagonal elements of the inertia tensor of a flat (square) plate      
        
        # Length of rods
        self.l = l

        # Prop Radius (for visualization only)
        self.r  = 0.1*l

        # Vertices of the icosahedron w.r.t. its centroid
        self.vertices_nominal = l * (np.array([
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
                                ]) - 0.5)
        self.vertices = self.orientation.apply(self.vertices_nominal) + self.position
        
        assert np.abs(np.linalg.norm(self.vertices[0]-self.vertices[2]) - 0.5*l) <= 1e-7, \
              f"Distance should be {0.5 * l} but is {np.linalg.norm(self.vertices[0]-self.vertices[2])}"
        assert np.abs(np.linalg.norm(self.vertices[2]-self.vertices[3]) - l) <= 1e-7, \
              f"Distance should be {l} but is {np.linalg.norm(self.vertices[2]-self.vertices[3])}"

        # Locations of propellers w.r.t. the centroid
        self.propllers_nominal = np.array([
            [ 0.0525,  0.040, 0.0],
            [ 0.0525, -0.040, 0.0],
            [-0.0525,  0.040, 0.0],
            [-0.0525, -0.040, 0.0]
        ])
        self.propellers =  self.orientation.apply(self.propllers_nominal) + self.position

        # The resulting mapping matrix for rotational accelerations
        self.rotational_acc_mat = np.array([[-0.25 * l / self.I[0], 0.25 * l / self.I[0], 0.25 * l / self.I[0], -0.25 * l / self.I[0]], # roll
                                            [-0.25 * l / self.I[1], 0.25 * l / self.I[1], -0.25 * l / self.I[1], 0.25 * l / self.I[1]], # pitch
                                            [- yd / self.I[2], -yd / self.I[2],  yd / self.I[2],  yd / self.I[2]]  # yaw
                                            ])

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
        if plot:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(projection='3d')

    
    def get_A(self, x):
        """ Returns the matrix for the state x that maps control inputs 
            u to state accelerations x_ddot. In this case 6 x 4 matrix.
            (Positional accelerations + Rotational Accelerations x 4 actuation forces) """
        
        orientation = R.from_euler(seq='xyz',
                                   angles=x[3:6])
        linear_acc_mat = 1.0 / self.m * orientation.apply([0.0, 0.0, 1.0]).T

        return np.concatenate((np.array(4 * [linear_acc_mat]).T,
                               self.rotational_acc_mat))

    def dynamics(self, x, u, f_ext=np.array([0,0,0]), tau_ext=np.array([0, 0, 0]),
                 update_internal_state=False) -> np.array:
        
        if update_internal_state:
            self.position = x[0:3]
            self.orientation = R.from_euler(seq='xyz',
                                            angles=x[3:6])

        # Init the derivative array
        x_ddot = np.array([
                            0, 0, 0, # Positional derivatives
                            0, 0, 0  # Rotational derivatives
                        ], dtype=float)
        
        # Add the actuation contribution
        x_ddot += self.get_A(x) @ (self.th * u**2) 

        # Add gravity contribution
        x_ddot += np.array([0, 0, -self.g,
                            0, 0, 0])
        
        # Add external forces and torques contribution
        x_ddot += np.concatenate((f_ext / self.m, tau_ext / self.I))

        return x_ddot

    def set_pose(self, p: np.array) -> None:
        self.position = p[0:3]
        self.orientation = R.from_euler("xyz", p[3:6])

        self.vertices = self.orientation.apply(self.vertices_nominal) + self.position
        self.propellers = self.orientation.apply(self.propllers_nominal) + self.position

        self.position_hist = np.append(self.position_hist, [self.position], axis=0)

    def plot_tensegrity_drone(self, t=0.0) -> None:
            self.ax.clear()

            # Plot the CoM
            self.ax.scatter(self.position[0],
                            self.position[1],
                            self.position[2], 
                            s=45, alpha=0.75,
                            color="black")
            self.ax.text(self.position[0]+0.01,
                        self.position[1]+0.01,
                        self.position[2]+0.01,
                        f"CoM")

            # Plot Timestamp
            self.ax.text(-0.1,-0.1,-0.1,
                         f"Time: {t: .2f} s",
                         fontsize=15)
            # Plot all the tensegrity vertices
            self.ax.scatter(self.vertices[:,0], 
                            self.vertices[:,1], 
                            self.vertices[:,2])
            for i in range(len(self.vertices)):
                self.ax.text(self.vertices[i, 0], 
                            self.vertices[i, 1], 
                            self.vertices[i, 2], 
                            rf"$v_{{{i+1}}}$")
                
            # Plot the trajectory of the CoM
            self.ax.plot(self.position_hist[:, 0],
                        self.position_hist[:, 1],
                        self.position_hist[:, 2], color="orange")

            # Plot the Propellors
            for i in range(len(self.propellers)):
                tau = np.linspace(-np.pi, np.pi, 25)
                circle = self.orientation.apply(np.array([self.r * np.sin(tau),
                                                        self.r * np.cos(tau),
                                                        np.zeros_like(tau)]).T) \
                            + self.propellers[i, :]
                self.ax.plot(circle[:, 0],
                            circle[:, 1],
                            circle[:, 2], color="red")

            # Plot the carbon fibre rods
            for i in range(0, 11, 2):
                segment = np.array([self.vertices[i,:],self.vertices[i+1,:]])
                self.ax.plot(segment[:,0],
                            segment[:,1],
                            segment[:,2], color="black")

            # Plot the strings
            for idx in self.strings:
                string = np.array([self.vertices[idx[0],:], self.vertices[idx[1], :]])
                self.ax.plot(string[:,0],
                            string[:,1],
                            string[:,2], color="grey")

            # Set labels
            self.ax.set_xlabel(f"x [m]")
            self.ax.set_ylabel(f"y [m]")
            self.ax.set_zlabel(f"z [m]")

            # For equal aspect ratio
            #self.ax.set_box_aspect([ub - lb for lb, ub in (getattr(self.ax, f'get_{a}lim')() for a in 'xyz')])

    def plot_trajectory(self, t, x,
                        name, u=lambda t, x: np.zeros(4)) -> None:
        ## Plot x versus t
        fig = plt.figure()
        ax = fig.subplots(3, sharex=True)
        ax[0].plot(t, x[:, 0:3])
        ax[0].set_ylabel(r'Position [$m$]')
        ax[0].legend([r"$x$",r"$y$",r"$z$"], ncol=3, loc="upper right")
        ax2 = ax[0].twinx()
        ax2.plot(t, x[:, 6:9], "--")
        ax2.set_ylabel(r'Velocity [$m / s$]')
        ax2.legend([r"$\dot{x}$",r"$\dot{y}$",r"$\dot{z}$"],
                   ncol=3, loc="lower right")
        
        ax[1].plot(t, np.rad2deg(x[:, 3:6]))
        ax[1].set_ylabel(r'Orientation [$^\circ$]')
        ax[1].legend([
                r"$\phi$",r"$\theta$",r"$\psi$",
                ], ncol=3, loc="upper right")
        ax3 = ax[1].twinx()
        ax3.plot(t, np.rad2deg(x[:, 9:12]), "--")
        ax3.set_ylabel(r'Angular Vel [$^\circ / s$]')
        ax3.legend([
                r"$\dot{\phi}$",r"$\dot{\theta}$",r"$\dot{\psi}$"
                ], ncol=3, loc="lower right")
        
        ctrl = np.zeros([len(t), 4])
        for i in range(len(t)):
            ctrl[i, :] = u(t[i], x[i, :])
        ax[2].plot(t, self.th * ctrl ** 2)
        ax[2].legend([r"$u_1$",r"$u_2$",r"$u_3$",r"$u_4$"],
                     bbox_to_anchor=(1.0, 1.0))
        ax[2].set_xlim([t[0], t[-1]])
        ax[2].set_xlabel(r't [$s$]')
        ax[2].set_ylabel(r'Thrust [$N$]')

        plt.savefig(name, bbox_inches='tight')

    def set_limits(self, xlim: tuple, ylim: tuple, zlim: tuple) -> None:
        self.ax.set_xlim(xlim[0], xlim[1])
        self.ax.set_ylim(ylim[0], ylim[1])
        self.ax.set_zlim(zlim[0], zlim[1])
        
    def save(self, name: str) -> None:
        self.fig.savefig(name)

    def show(self) -> None:
        plt.show()

