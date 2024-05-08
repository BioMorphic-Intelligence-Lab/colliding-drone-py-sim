import numpy as np

from tensegrity_drone import TensegrityDrone
from controller import CollisionRecoveryController
from misc import *

def main():

    # Add program options
    options = add_po()

    p0 = np.array([1.0,0,0])
    # Init drone objet and define desired attitude
    drone = TensegrityDrone(plot=True, angles=[0.0, 0, 0],
                            p = p0,
                            barrier_loc=[0.70, 2.0, 0.0],
                            barrier_sidelength=[1.4, 2.0, 4.0],
                            barrier_orientation=np.deg2rad([0, 0, 0]),
                            n=[0,-1,0])
    
    controller = CollisionRecoveryController(drone)

    t_end = 4.5
    speed_y = 2.0 # meters per second
    speed_x = -2.25 # meters per second
    des_p = lambda t: p0 + np.array([speed_x * t,
                                    speed_y * t,
                                    0.0])
    des_v = lambda t: np.array([0,0,0])
    des_yaw = lambda t:np.deg2rad(0)

    # Set control law
    ctrl = lambda t, x: controller.u(x, x_des=des_p(t), v_des=des_v(t),
                                     yaw_des=des_yaw(t))

    x0 = np.array([
                    p0[0], p0[1], p0[2], 0, 0, 0,  # Pose
                    0, 0, 0, 0, 0, 0   # Pose derivative
                ], dtype=float)
    
    t = np.linspace(0, t_end, int(1e5))
    
    f = lambda t, y : np.concatenate((y[6:12],
                                    drone.dynamics(x=y, 
                                                    u=(ctrl(t, y)),
                                                    update_internal_state=True)))

    run(options, f, x0, t, drone, controller,
        ctrl=ctrl, des_p=des_p,
        speed_factor=0.1, downsample=0.01)

if __name__ == '__main__':
    main()