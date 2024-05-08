from tqdm import tqdm
import scipy.integrate
import matplotlib.pyplot as plt
import numpy as np

from optparse import OptionParser
from tensegrity_drone import TensegrityDrone
from controller import *
from animate_traj import animate

def add_po() -> dict:        
    parser = OptionParser()
    parser.add_option("-p", "--plot", dest="plot_path", default="",
                    help="Plot the trajectory to file PLOT_PATH", metavar="PLOT_PATH")
    parser.add_option("-a", "--animate", dest="anim_path", default="",
                    help="Save animation to file PATH")
    parser.add_option("-s", "--save", dest="save_path", default="",
                    help="Save the trajectory to file SAVE_PATH", metavar="SAVE_PATH")
    parser.add_option("-l", "--load", dest="load_path", default="",
                    help="Load the trajectory from file LOAD_PATH", metavar="LOAD_PATH")

    (options, args) = parser.parse_args()

    return options

def u(x, des_p, des_yaw):
    # Find desired attitude and total thrust
    acc = (position_ctrl(x, des_p, np.zeros(3)))

    des_attitude, tot_thrust = get_attitude_and_thrust(
            acc,
            yaw_des=des_yaw,
            att=x[3:6]
        )
    
    # Return the motor speeds for the above
    return motor_speeds_from_forces(
                    motor_forces_from_torques(
                        torques=angular_vel_ctrl(x[9:12], 
                                         attitude_ctrl(x[3:6],
                                                       des_attitude.as_euler("xyz"))
                                        ),
                            tot_thrust=tot_thrust                                        
                        )
                    )

def main():

    # Add program options
    options = add_po()

    p0 = np.array([2.0,0,0])
    # Init drone objet and define desired attitude
    drone = TensegrityDrone(plot=True, angles=[0.0, 0, 0],
                            p = p0,
                            barrier_loc=[0.70, 2.0, 0.0],
                            barrier_sidelength=[1.4, 2.0, 4.0],
                            barrier_orientation=np.deg2rad([0, 0, 0]),
                            n=[0,-1,0])
    
    t_end = 2.5
    speed_y = 1.0 # meters per second
    speed_x = -2.3
    des_p = lambda t: p0 + np.array([speed_x * t,
                                    speed_y * t,
                                    0.0])
    des_yaw = lambda t:np.deg2rad(0)

    # Set control law
    ctrl = lambda t, x: u(x, des_p=des_p(t), des_yaw=des_yaw(t))

    x0 = np.array([
                    p0[0], p0[1], p0[2], 0, 0, 0,  # Pose
                    0, 0, 0, 0, 0, 0   # Pose derivative
                ], dtype=float)
    
    t = np.linspace(0, t_end, int(1e5))
    
    f = lambda t, y : np.concatenate((y[6:12],
                                    drone.dynamics(x=y, 
                                                    u=(ctrl(t, y)),
                                                    update_internal_state=True)))

    if options.load_path != "":
        r = np.loadtxt(options.load_path)
        t = r[0, :]
        x = r[1:, :]
    else:
        ## Set up the ODE object
        print("Solve ode ...")
        r = scipy.integrate.solve_ivp(f, (0, t_end), x0, method='BDF',
                                    t_eval=t, max_step=0.001)
        print("... done!")

        t = r.t
        x = r.y

    if options.save_path != "":
        traj = np.concatenate(([t], x), axis=0)
        np.savetxt(options.save_path, traj)

    if options.plot_path != "":
        drone.plot_trajectory(t, x.T, options.plot_path, u=ctrl,
                              downsample=0.01)

    if options.anim_path != "":
        ## Animate
        traj = x[0:6, :].T
        animate(t, traj, name=options.anim_path,
                drone=drone, speed_factor=1, downsample=0.01)

if __name__ == '__main__':
    main()