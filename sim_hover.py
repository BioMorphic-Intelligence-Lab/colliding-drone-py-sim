import numpy as np

from tensegrity_drone import TensegrityDrone
from controller import *
from misc import *

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
    
    run(options, f, x0, t, drone, u, des_p=None,
        speed_factor=1, downsample=1)

if __name__ == '__main__':
    main()