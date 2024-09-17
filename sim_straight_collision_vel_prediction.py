import numpy as np

from tensegrity_drone import TensegrityDrone
from controller import VelocityPredictionRecoveryController
from misc import *

def main():

    # Add program options
    options = add_po()

    options.load_path = "./python_sim/data/straight_collision_velocity_prediction.npy"
    #options.save_path = "./python_sim/data/straight_collision_velocity_prediction.npy"
    options.plot_path = "./python_sim/plots/straight_collision_velocity_prediction.png"

    # Init drone objet and define desired attitude
    drone = TensegrityDrone(plot=True, angles=[0.0, 0, 0],
                            p = [0,0,0],
                            barrier_loc=[0.0, 4.0, 1.0],
                            barrier_sidelength=[4.0, 2.0, 4.0],
                            barrier_orientation=np.deg2rad([0, 0, 0]),
                            n=[0,-1,0])
    
    # Init Controller
    controller = VelocityPredictionRecoveryController(drone)

    t_end = 2.0
    speed = 2 # meters per second
    des_p = lambda t: np.array([0.0,
                                speed * t,
                                1])
    des_v = lambda t: np.array([0, speed, 0])
    des_yaw = lambda t:np.deg2rad(0)

    # Set control law
    ctrl = lambda t, x: controller.u(t, x, x_des=des_p(t), v_des=des_v(t),
                                     yaw_des=des_yaw(t))

    x0 = np.array([
                    0, 0, 0, 0, 0, 0,  # Pose
                    0, 0, 0, 0, 0, 0   # Pose derivative
                ], dtype=float)
    
    t = np.linspace(0, t_end, int(1e5))
    
    f = lambda t, y : np.concatenate((y[6:12],
                                      drone.dynamics(x=y, 
                                                     u=(ctrl(t, y)),
                                                     update_internal_state=True)))
    
    run(options, f, x0, t, drone, controller, ctrl=ctrl, des_p=None,
        speed_factor=0.1, downsample=0.1, max_step=0.0001,
        pred=controller.predict_delta_vel)

if __name__ == '__main__':
    main()