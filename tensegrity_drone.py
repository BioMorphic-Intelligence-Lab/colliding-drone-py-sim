import numpy as np
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from rectoid import plot_rectoid

class TensegrityDrone(object):

    def __init__(self, p=[0.0, 0.0, 0.0], angles=[0.0, 0.0, 0.0],
                 l=0.22, g=9.81, th=7.50e-8, yd=0.009975,
                 m=0.315, I=[6.21e-4, 9.62e-4, 9.22e-4],
                 barrier_loc=[], barrier_sidelength=[],
                 barrier_orientation=[0, 0, 0],
                 n = [0, -1, 0],
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
        
        # Save all barriers present in the world
        self.barrier_loc = np.array(barrier_loc)
        self.barrier_sidelength = np.array(barrier_sidelength)
        self.barrier_orientation = R.from_euler(seq="xyz", angles=barrier_orientation)
        # Normal vector and plane parameter from the normal form parametrization
        #   n1 * x +  n2 * y  + n3 * z + d = 0
        if len(barrier_loc) > 0:
            self.n = self.barrier_orientation.apply(n)
            self.d = np.dot(self.n, self.barrier_loc
                                        + 0.5 * self.barrier_sidelength[2]
                                            * self.barrier_orientation.apply(self.n))

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

    def dynamics(self, x, u, update_internal_state=False) -> np.array:
        
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
        
        # Find external wrench if object is specified
        if len(self.barrier_loc) > 0:
            tau_ext = self.get_contact_wrench(x)

            # Add external forces and torques contribution
            x_ddot += np.concatenate((tau_ext[0:3] / self.m, tau_ext[3:6] / self.I))

        return x_ddot

    def set_pose(self, p: np.array) -> None:
        self.position = p[0:3]
        self.orientation = R.from_euler("xyz", p[3:6])

        self.vertices = self.orientation.apply(self.vertices_nominal) + self.position
        self.propellers = self.orientation.apply(self.propllers_nominal) + self.position

        self.position_hist = np.append(self.position_hist, [self.position], axis=0)

    def is_in_contact(self, vertices: np.array) -> np.array:
        """Function the returns a binary array indicating
           which of vertices are in contact and which ones are not"""
        # Currently only works if the barrier rectoid is not rotated
        return np.array([(
            (self.barrier_orientation.apply(vertices[i, :])
                >= (self.barrier_loc - 0.5 * self.barrier_sidelength)).all()
            and
            (self.barrier_orientation.apply(vertices[i, :])
                <= (self.barrier_loc + 0.5 * self.barrier_sidelength)).all()
                )
            for i in range(len(vertices))
        ])

    def get_vertex_velocities(self, x: np.array, selector: np.array) -> np.array:

        # Init velocity vector
        vel = np.zeros([len(selector), 3])
        orientation = R.from_euler(seq="xyz", angles=x[3:6])

        for i, idx in enumerate(selector):
            # Add the linear contribution
            vel[i,:] += x[6:9]

            # Add the rotational contribution
            vel[i, :] += orientation.apply(x[9:12] * self.vertices_nominal[idx, :])

        return vel

    def get_contact_wrench(self, x: np.array) -> np.array:
        """Function that gets the total contact wrench acting on the system
           by summing up the indiviual wrenches from each vertex"""

        # Initialize the wrench vector
        wrench = np.zeros(6)

        # Find vertex locations from state
        vertices = np.zeros([len(self.vertices_nominal), 3])
        rot = R.from_euler(seq="xyz", angles=x[3:6])
        for i in range(len(self.vertices_nominal)):
            vertices[i, :] = rot.apply(self.vertices_nominal[i, :]) + x[0:3]

        # Find which vertices are in contact
        in_contact = self.is_in_contact(vertices)
        in_contact_idx = np.flatnonzero(in_contact)

        # Extract their velocities
        vertex_velocities = self.get_vertex_velocities(x,
                                selector=in_contact_idx)

        assert len(in_contact_idx) == np.shape(vertex_velocities)[0], \
               "More velocities extracted that vertices are in contact!"

        # Compute and add the wrench contribution for each one
        for i, idx in enumerate(in_contact_idx):
            wrench +=  self.get_contact_wrench_by_vertex(x, vertices[idx],
                                                         vertex_velocities[i])

        return wrench

    def get_contact_wrench_by_vertex(self, x: np.array,
                                     vertex: np.array, vertex_velocity: np.array) -> np.array:
        """Function that finds the wrench acting on the CoM due to the contact
           of the given vertex """

        # Get the impact force at the vertex
        force = self.get_contact_force(vertex, vertex_velocity)

        # Concatenate to full 6d wrench
        wrench = np.concatenate((force, np.cross(vertex - x[0:3], force)))

        return wrench

    def get_contact_force(self, vertex: np.array, vertex_speed: np.array,
                          k=10000.0, d=100.0 # Spring and damping coefficients
                          ) -> np.array:
        """ Function that finds the contact force acting on a single vertex
            that is in contact """
        # Find the current penetration depth
        penetration_depth = np.dot(self.n, vertex) - self.d
        # Find the current penetration speed, we only do damping
        # on the way in, ergo saturate
        penetration_speed = min([np.dot(self.n, vertex_speed), 0])

        # Initialize force vector
        f = np.zeros(3)

        # Sanity check, we should not be in contact when this is called
        if penetration_depth > 0:
            raise Warning("Contact force requested when vertex is not in contact!")
        else:
            # Spring damper model of the contact force.
            # The contact force always acts normal to the plane
            f = self.n * np.abs(k * penetration_depth + d * penetration_speed)

        return f

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

            # Find which vertices are in contact
            if (not self.barrier_loc.size == 0
                and not self.barrier_sidelength.size == 0):

                # Plot the barriers if the list is not empty
                plot_rectoid(self.ax, self.barrier_loc, self.barrier_sidelength, rot=self.barrier_orientation,
                            facecolors='xkcd:grey', edgecolor="black", alpha=0.5)

                # Find which vertices are in contact
                in_contact = self.is_in_contact(self.vertices)
            else:
                in_contact = np.array(len(self.vertices) * [False])

            # Plot all the tensegrity vertices adjusting
            # the color base on contact
            self.ax.scatter(self.vertices[in_contact,0],
                            self.vertices[in_contact,1],
                            self.vertices[in_contact,2], color="orange")
            self.ax.scatter(self.vertices[~in_contact,0],
                            self.vertices[~in_contact,1],
                            self.vertices[~in_contact,2], color="blue")

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

        fig.set_size_inches((5, 10))
        plt.savefig(name, bbox_inches='tight', dpi=500)

    def set_limits(self, xlim: tuple, ylim: tuple, zlim: tuple) -> None:
        self.ax.set_xlim(xlim[0], xlim[1])
        self.ax.set_ylim(ylim[0], ylim[1])
        self.ax.set_zlim(zlim[0], zlim[1])
        
    def save(self, name: str) -> None:
        self.fig.savefig(name)

    def show(self) -> None:
        plt.show()

