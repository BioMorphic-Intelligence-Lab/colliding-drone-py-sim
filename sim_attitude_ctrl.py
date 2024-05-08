import numpy as np

from tensegrity_drone import TensegrityDrone
from controller import Controller
from misc import *

def main():

    # Add program options
    options = add_po()

    # Init drone objet and define desired attitude
    drone = TensegrityDrone(plot=True)
    controller = Controller()
    attitude_des = np.deg2rad([2, 5, 10])

    u = lambda t, x: controller.motor_speeds_from_forces(
                    controller.motor_forces_from_torques(
                        controller.angular_vel_ctrl(x[9:12], 
                                         controller.attitude_ctrl(
                                             x[3:6], attitude_des)
                                        ),
                        drone.m * drone.g
                        )
                    )

    x0 = np.array([
                    0,0,0,0,0,0,  # Pose
                    0,0,0,0,0,0   # Pose derivative
                ], dtype=float)
    t_end = 15
    t = np.linspace(0, t_end, 500)
    
    f = lambda t, y : np.concatenate((y[6:12],
                                      drone.dynamics(x=y, 
                                                     u=(u(t, y)))))

    run(options, f, x0, t, drone, controller, 
        ctrl=u, des_p=None,
        speed_factor=1, downsample=1)

if __name__ == '__main__':
    main()