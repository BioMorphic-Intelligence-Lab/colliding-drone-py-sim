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

    # Init drone objet and define desired attitude
    drone = TensegrityDrone(plot=True, angles=[0.0, 0, 0],
                            p = [0,0,0],
                            barrier_loc=[0.0, 3.0, 1.0],
                            barrier_sidelength=[4.0, 2.0, 4.0],
                            barrier_orientation=np.deg2rad([0, 0, 0]),
                            n=[0,-1,0])
    t_end = 3.0
    speed = 2 # meters per second
    des_p = lambda t: np.array([0.0,
                                speed * t,
                                1])
    des_yaw = lambda t:np.deg2rad(0)

    # Set control law
    ctrl = lambda t, x: u(x, des_p=des_p(t), des_yaw=des_yaw(t))

    x0 = np.array([
                    0, 0, 0, 0, 0, 0,  # Pose
                    0, 0, 0, 0, 0, 0   # Pose derivative
                ], dtype=float)
    
    t = np.linspace(0, t_end, 1000)
    
    f = lambda t, y : np.concatenate((y[6:12],
                                      drone.dynamics(x=y, 
                                                     u=(ctrl(t, y)),
                                                     update_internal_state=True)))

        ## Set up the ODE object
    print("Solve ode ...")
    r = scipy.integrate.solve_ivp(f, (0, t_end), x0, method='BDF',
                                  t_eval=t, max_step=0.1)
    print("... done!")

    if options.plot_path != "":
        drone.plot_trajectory(r.t, r.y.T, options.plot_path, u=ctrl)

    if options.anim_path != "":
        ## Animate
        traj = r.y[0:6, :].T
        animate(r.t, traj, name=options.anim_path,
                drone=drone, speed_factor=1)

if __name__ == '__main__':
    main()