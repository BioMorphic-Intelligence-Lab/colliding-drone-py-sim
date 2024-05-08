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
    drone = TensegrityDrone(plot=True,
                            p = [0, 0.85, 0.5], angles=[0.0, 0.0, 0],
                            barrier_loc=[0.0, 1.0, -1],
                            barrier_sidelength=[2.0, 1.9, 2.0],
                            barrier_orientation=np.deg2rad([0, 0, 0]),
                            n=[0,0,1],
                            k=100, damp=100)
    # Define final time
    t_end = 0.5

    # Set control law
    ctrl = lambda t, x: np.array([0,0,0,0], dtype=float) 

    # Define initial state
    x0 = np.array([
                    0, 0.85, 0.5,  0.0,  0.0, 0,  # Pose
                    0, -3.0,   0,  0.0,  0.0, 0   # Pose derivative
                ], dtype=float)
    
    # Define time vector
    t = np.linspace(0, t_end, 1000)
    
    # Put the dynamics function together
    f = lambda t, y : np.concatenate((y[6:12],
                                      drone.dynamics(x=y, 
                                                     u=(ctrl(t, y)),
                                                     update_internal_state=True)))

    run(options, f, x0, t, drone, ctrl, des_p=None,
        speed_factor=0.1, downsample=1)

if __name__ == '__main__':
    main()