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

    angular_vel_des = np.array([0.1,0.2,0.3])

    u = lambda t, x: motor_speeds_from_forces(
                    motor_forces_from_torques(
                        angular_vel_ctrl(x[9:12], angular_vel_des),
                        drone.m * drone.g
                    )
    )

    x0 = np.array([
                    0,0,0,0,0,0,  # Pose
                    0,0,0,0,0,0   # Pose derivative
                ], dtype=float)
    
    t = np.linspace(0, 15, 500)
    x = np.zeros([len(t), 12])
    x[0, :] = x0
    
    f = lambda t, y : np.concatenate((y[6:12],
                                      drone.dynamics(x=y, 
                                                     u=(u(t, y)))))

    ## Set up the ODE object
    r = scipy.integrate.ode(f)
    r.set_integrator('dopri5')    # A Runge-Kutta solver
    r.set_initial_value(x0)

    for n in range(1,len(t)):
        r.integrate(t[n])
        assert r.successful()
        x[n] = r.y

    if options.plot_path != "":
        drone.plot_trajectory(t, x, options.plot_path, u=u)

    if options.anim_path != "":
        ## Animate
        traj = x[:, 0:6]
        animate(t, traj, name=options.anim_path, drone=drone)

if __name__ == '__main__':
    main()