import numpy as np
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

from rectoid import plot_rectoid

class TensegrityDrone(object):

    def __init__(self, p=[0.0, 0.0, 0.0], angles=[0.0, 0.0, 0.0],
                 l=0.22, g=9.81, th=7.50e-8, yd=0.009975,
                 m=0.315, I=[6.21e-4, 9.62e-4, 9.22e-4],
                 barrier_loc=[], barrier_sidelength=[],
                 barrier_orientation=[0, 0, 0],
                 n = [0, -1, 0],
                 # Spring and damping coefficients for contact dynamics
                 k=10000.0, damp=2000.0,
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
        self.I = np.diag(I)   # Diagonal elements of the inertia tensor of a flat (square) plate      
        
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
        self.propellers_nominal = np.array([
            [ 0.0525,  0.040, 0.0],
            [ 0.0525, -0.040, 0.0],
            [-0.0525,  0.040, 0.0],
            [-0.0525, -0.040, 0.0]
        ])
        self.propellers =  self.orientation.apply(self.propellers_nominal) + self.position

        # The resulting mapping matrix for rotational accelerations
        self.rotational_acc_mat = (np.linalg.inv(self.I) @
                                    np.array([[-0.25 * l, 0.25 * l, 0.25 * l, -0.25 * l], # roll
                                              [-0.25 * l, 0.25 * l, -0.25 * l, 0.25 * l], # pitch
                                              [- yd, -yd,  yd,  yd]  # yaw
                                            ]))

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

        # Contact Dynamics Parameters
        self.k = k
        self.damp = damp

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
            x_ddot += np.concatenate((tau_ext[0:3] / self.m, np.linalg.inv(self.I) @ tau_ext[3:6]))

        return x_ddot

    def set_pose(self, p: np.array) -> None:
        self.position = p[0:3]
        self.orientation = R.from_euler("xyz", p[3:6])

        self.vertices = self.orientation.apply(self.vertices_nominal) + self.position
        self.propellers = self.orientation.apply(self.propellers_nominal) + self.position

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
            vel[i, :] += np.cross(x[9:12], 
                                  orientation.apply(self.vertices_nominal[idx, :]))

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
        scale = 1.0 / max(len(in_contact_idx), 1)
        for i, idx in enumerate(in_contact_idx):
            wrench +=  scale * self.get_contact_wrench_by_vertex(x, vertices[idx],
                                                         vertex_velocities[i])

        return wrench

    def get_contact_wrench_by_vertex(self, x: np.array,
                                     vertex: np.array, vertex_velocity: np.array) -> np.array:
        """Function that finds the wrench acting on the CoM due to the contact
           of the given vertex """

        # Get the impact force at the vertex
        force = self.get_contact_force(vertex, vertex_velocity)
        
        # Compute the torque resulting on the body
        torque = np.cross(vertex - x[0:3], force)

        # Concatenate to full 6d wrench
        wrench = np.concatenate((force,
                                 torque
                                 ))

        return wrench

    def get_contact_force(self, vertex: np.array, vertex_speed: np.array) -> np.array:
        """ Function that finds the contact force acting on a single vertex
            that is in contact """
        # Find the current penetration depth
        penetration_depth = np.dot(self.n, vertex) - self.d
        # Find the current penetration speed
        # (both ways to simulate loss of energy)
        penetration_speed = np.dot(self.n, vertex_speed)

        # Initialize force vector
        f = np.zeros(3)

        # Sanity check, we should be in contact when this is called
        if penetration_depth > 0:
            raise Warning("Contact force requested when vertex is not in contact!")
        else:
            # Spring damper model of the contact force.
            # The contact force always acts normal to the plane
            f = self.n * np.abs(self.k * penetration_depth + self.damp * penetration_speed)

        return f

    def plot_tensegrity_drone(self, t=0.0, axis=False,
                              x_des=np.array([])) -> None:

            self.ax.clear()

            # Plot body axis
            if axis:
                self.ax.quiver(self.position[0],
                               self.position[1],
                               self.position[2],
                               1, 0, 0,
                               length=0.1, normalize=True, color="red")
                self.ax.quiver(self.position[0],
                               self.position[1],
                               self.position[2],
                               0, 1, 0,
                               length=0.1, normalize=True, color="green")
                self.ax.quiver(self.position[0],
                               self.position[1],
                               self.position[2],
                               0, 0, 1,
                               length=0.1, normalize=True, color="blue")

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

            # Plot Reference Position if give
            if x_des.size != 0:
                self.ax.scatter(x_des[0],
                                x_des[1],
                                x_des[2],
                                s=45, color="black", marker="x")
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
            self.ax.set_box_aspect([ub - lb for lb, ub in (getattr(self.ax, f'get_{a}lim')() for a in 'xyz')])

    def bring_angle_to_range(self, angle):
        """Function that brings arbitrary angles to the range [-pi, pi]"""
        angle -= np.sign(angle) * (np.abs(angle) // (2*np.pi)) * 2 * np.pi

        angle[angle >  np.pi] -= 2*np.pi
        angle[angle < -np.pi] += 2*np.pi

        return angle

    def plot_trajectory(self, t, x,
                        name,
                        u=lambda t, x: np.zeros(4),
                        downsample=1.0,
                        pred=None,
                        controller=None) -> None:
        # Find downsampling step
        step = int(1.0 / downsample)
        
        # Plot x versus t
        fig = plt.figure()
        ax = fig.subplots(4, sharex=True)
        ax[0].plot(t[0:-1:step], x[0:-1:step, 0], label=r"$x$")
        ax[0].plot(t[0:-1:step], x[0:-1:step, 1], label=r"$y$")
        ax[0].plot(t[0:-1:step], x[0:-1:step, 2], label=r"$z$")
        ax[0].set_ylabel(r'Position [$m$]')
        ax[0].legend(ncol=3, loc="upper right")
        ax2 = ax[0].twinx()
        ax2.plot(t[0:-1:step], x[0:-1:step, 6],
                 label=r"$\dot{x}$", linestyle="--")
        ax2.plot(t[0:-1:step], x[0:-1:step, 7],
                 label=r"$\dot{y}$", linestyle="--")
        ax2.plot(t[0:-1:step], x[0:-1:step, 8],
                 label=r"$\dot{z}$", linestyle="--")
        ax2.set_ylabel(r'Velocity [$m / s$]')
        
        ax[1].plot(t[0:-1:step],
                   np.rad2deg(self.bring_angle_to_range(x[0:-1:step, 3])),
                   label=r"$\phi$")
        ax[1].plot(t[0:-1:step],
                   np.rad2deg(self.bring_angle_to_range(x[0:-1:step, 4])),
                   label=r"$\theta$")
        ax[1].plot(t[0:-1:step],
                   np.rad2deg(self.bring_angle_to_range(x[0:-1:step, 5])),
                   label=r"$\psi$")
        ax[1].set_ylabel(r'Orientation [$^\circ$]')
        ax[1].legend(ncol=3, loc="upper right")
        ax3 = ax[1].twinx()
        ax3.plot(t[0:-1:step], np.rad2deg(x[0:-1:step, 9]), "--",
                 label=r"$\dot{\phi}$")
        ax3.plot(t[0:-1:step], np.rad2deg(x[0:-1:step, 10]), "--",
                 label=r"$\dot{\theta}$")
        ax3.plot(t[0:-1:step], np.rad2deg(x[0:-1:step, 11]), "--",
                 label=r"$\dot{\psi}$")
        ax3.set_ylabel(r'Angular Vel [$^\circ / s$]')
        
        ctrl = np.zeros([len(t), 4])
        for i in range(len(ctrl)):
            ctrl[i, :] = u(t[i], x[i, :])
        ax[2].plot(t, self.th * ctrl ** 2)
        ax[2].legend([r"$u_1$",r"$u_2$",r"$u_3$",r"$u_4$"],
                     bbox_to_anchor=(1.0, 1.0))
        ax[2].set_xlim([t[0], t[-1]])
        ax[2].set_ylabel(r'Thrust [$N$]')

        # Find which vertices are in contact
        if (not self.barrier_loc.size == 0
            and not self.barrier_sidelength.size == 0):

            # First index of contact initialization
            contact_idx = np.nan

            # Find which vertices are in contact
            in_contact = np.array(len(t) * [12 * [True]])
            for i in range(len(x)):
                rot = R.from_euler(seq="xyz", angles=x[i, 3:6])
                vertices = rot.apply(self.vertices_nominal) + x[i, 0:3]
                in_contact[i, :] = self.is_in_contact(vertices)

                if np.isnan(contact_idx) and in_contact[i,:].any():
                    contact_idx = i

            if (not np.isnan(contact_idx)) and pred is not None:

                # Reset Controller
                if controller is not None:
                    controller.reset()

                # Find velocity prediction after contact
                t_contact = t[contact_idx-1:]
                vel_pred = np.zeros([len(t_contact), 6])
                vel_pred[0,: ] = np.concatenate(
                    (x[contact_idx-1,6:9], x[contact_idx-1,9:12]))

                # Iterate over time since collision and find the
                #  velocity Prediction
                for i in range(1, len(t_contact)):
                    delta_prediction = pred(x[contact_idx + i-2, 0:6],
                                            x[contact_idx + i-2, 6:12],
                                            ctrl[contact_idx + i-1,:],
                                            in_contact[contact_idx + i-1, :],
                                            dt=t_contact[i]-t_contact[i-1])
                    vel_pred[i, :] = (vel_pred[i-1, :] 
                                      + delta_prediction)
                
                ax2.plot(t_contact, vel_pred[:, 0], color="blue",
                           linestyle=":", label=r"$\dot{x}_{{pred}}$")
                ax2.plot(t_contact, vel_pred[:, 1], color="orange",
                           linestyle=":", label=r"$\dot{y}_{{pred}}$")
                ax2.plot(t_contact, vel_pred[:, 2], color="green",
                           linestyle=":", label=r"$\dot{z}_{{pred}}$")
                ax3.plot(t_contact, vel_pred[:, 3], color="blue",
                           linestyle=":", label=r"$\dot{\phi}_{{pred}}$")
                ax3.plot(t_contact, vel_pred[:, 4], color="orange",
                           linestyle=":",label=r"$\dot{\theta}_{{pred}}$")
                ax3.plot(t_contact, vel_pred[:, 5], color="green",
                           linestyle=":", label=r"$\dot{\psi}_{{pred}}$")
        else:
            in_contact = np.array(len(t) * [12 * [False]])

        # Do legends for velocities after we plotted the prediction
        ax2.legend(ncol=3, loc="lower right")
        ax3.legend(ncol=3, loc="lower right")

        for i in range(12):
            time = t[in_contact[:, i]]
            ax[3].scatter(time, (i + 1) * np.ones_like(time), color="orange")

        # Set plot ranges and labels
        ax[3].set_ylim((0.5, 12.5))
        ax[3].grid(axis="y")
        ax[3].set_yticks([i + 1 for i in range(12)])
        ax[3].set_yticklabels([fr'$v_{{{i+1}}}$' for i in range(12)])
        ax[3].set_xlabel(r't [$s$]')

        # Save the figure
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

