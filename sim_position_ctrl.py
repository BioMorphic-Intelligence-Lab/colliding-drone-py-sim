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
    t_end = 20
    speed = 0.1 # Revolution per second
    amplitude = 0.5
    height = 0.5
    des_p = lambda t: np.array([amplitude * np.sin(2 * speed * t * 2 * np.pi),
                                amplitude * np.cos(speed* t * 2 * np.pi),
                                height])
    des_v = lambda t: np.array([4*speed*np.pi*amplitude*np.cos(4*speed*np.pi*t),
                                -2*speed*np.pi*amplitude*np.sin(2*speed*np.pi*t),
                                0])
    des_yaw = lambda t: np.arctan2(-amplitude*np.sin(speed * t * 2 * np.pi) * 2 * speed * np.pi,
                                    amplitude*np.cos(2*speed*t*2*np.pi) * 4 * speed*np.pi)

    # Set control law
    ctrl = lambda t, x: controller.u(x, x_des=des_p(t), v_des=des_v(t),
                                     yaw_des=des_yaw(t))

    x0 = np.array([
                    0,0,0,0,0,0,  # Pose
                    0,0,0,0,0,0   # Pose derivative
                ], dtype=float)
    
    t = np.linspace(0, t_end, 1000)
    
    f = lambda t, y : np.concatenate((y[6:12],
                                      drone.dynamics(x=y, 
                                                     u=(ctrl(t, y)))))

    run(options, f, x0, t, drone, controller, 
        ctrl=ctrl, des_p=des_p,
        speed_factor=1, downsample=1)

if __name__ == '__main__':
    main()