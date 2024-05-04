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
    drone = TensegrityDrone()
    t_end = 20
    des_p = lambda t: np.array([-2 * np.sin(2 * 0.1 * t * 2 * np.pi),
                                2 * np.cos(0.1 * t * 2 * np.pi),
                                2.0]) 
    des_yaw = lambda t:np.deg2rad(0)

    # Set control law
    ctrl = lambda t, x: u(x, des_p=des_p(t), des_yaw=des_yaw(t))

    x0 = np.array([
                    0,0,0,0,0,0,  # Pose
                    0,0,0,0,0,0   # Pose derivative
                ], dtype=float)
    
    t = np.linspace(0, t_end, 1000)
    x = np.zeros([len(t), 12])
    x[0, :] = x0
    
    f = lambda t, y : np.concatenate((y[6:12],
                                      drone.dynamics(x=y, 
                                                     u=(ctrl(t, y)))))

    ## Set up the ODE object
    r = scipy.integrate.ode(f)
    r.set_integrator('dopri5')    # A Runge-Kutta solver
    r.set_initial_value(x0)

    for n in range(1,len(t)):
        r.integrate(t[n])
        assert r.successful()
        x[n] = r.y

    if options.plot_path != "":
        drone.plot_trajectory(t, x, options.plot_path, u=ctrl)

    if options.anim_path != "":
        ## Animate
        traj = x[:, 0:6]
        animate(t, traj, name=options.anim_path)

if __name__ == '__main__':
    main()