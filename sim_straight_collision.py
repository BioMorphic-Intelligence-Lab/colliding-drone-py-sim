import numpy as np

from tensegrity_drone import TensegrityDrone
from controller import *
from misc import *

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
    
    t = np.linspace(0, t_end, int(1e5))
    
    f = lambda t, y : np.concatenate((y[6:12],
                                      drone.dynamics(x=y, 
                                                     u=(ctrl(t, y)),
                                                     update_internal_state=True)))
    run(options, f, x0, t, drone, ctrl, des_p=None,
        speed_factor=0.1, downsample=0.01, max_step=0.1)

if __name__ == '__main__':
    main()