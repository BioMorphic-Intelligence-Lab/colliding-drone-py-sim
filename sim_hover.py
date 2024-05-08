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

def main():

    options = add_po()

    drone = TensegrityDrone(plot=True)

    h_des = 10.0
    u = lambda t, x: motor_speeds_from_forces(
                    motor_forces_from_torques(
                        angular_vel_ctrl(x[9:12], np.zeros(3)),
                        2.0 * (h_des - x[2]) + 1.0 * (-x[8]) +  drone.m * drone.g
                    )
    )

    x0 = np.array([
                    0,0,0,0,0,0,  # Pose
                    0,0,0,0,0,0   # Pose derivative
                ], dtype=float)
    
    t_end = 15
    t = np.linspace(0, t_end, 200)
    
    f = lambda t, y : np.concatenate(
                                        (y[6:12],
                                            drone.dynamics(x=y, 
                                                           u=u(t, y))
                                        )
                                    )
    ## Set up the ODE object
    print("Solve ode ...")
    r = scipy.integrate.solve_ivp(f, (0, t_end), x0, method='BDF',
                                  t_eval=t, max_step=0.001)
    print("... done!")

    if options.plot_path != "":
        drone.plot_trajectory(r.t, r.y.T, options.plot_path, u=u)

    if options.anim_path != "":
        ## Animate
        traj = r.y[0:6, :].T
        animate(r.t, traj, name=options.anim_path,
                drone=drone, speed_factor=1)

if __name__ == '__main__':
    main()