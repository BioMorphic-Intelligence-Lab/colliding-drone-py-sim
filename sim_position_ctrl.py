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
    drone = TensegrityDrone(plot=True)
    t_end = 20
    speed = 0.1 # Revolution per second
    amplitude = 0.5
    height = 0.5
    des_p = lambda t: np.array([amplitude * np.sin(2 * speed * t * 2 * np.pi),
                                amplitude * np.cos(speed* t * 2 * np.pi),
                                height])
    des_yaw = lambda t: np.arctan2(-amplitude*np.sin(speed * t * 2 * np.pi) * 2 * speed * np.pi,
                                    amplitude*np.cos(2*speed*t*2*np.pi) * 4 * speed*np.pi)

    # Set control law
    ctrl = lambda t, x: u(x, des_p=des_p(t), des_yaw=des_yaw(t))

    x0 = np.array([
                    0,0,0,0,0,0,  # Pose
                    0,0,0,0,0,0   # Pose derivative
                ], dtype=float)
    
    t = np.linspace(0, t_end, 1000)
    
    f = lambda t, y : np.concatenate((y[6:12],
                                      drone.dynamics(x=y, 
                                                     u=(ctrl(t, y)))))

    run(options, f, x0, t, drone, ctrl, des_p=des_p,
        speed_factor=1, downsample=1)

if __name__ == '__main__':
    main()