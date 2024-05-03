import numpy as np 

def position_ctrl(x: np.array, p_des: np.array, kp=1.0, kd=0.1) -> np.array:
    """Function that takes the current state 
       (x,y,z, roll, pitch, yaw, xdot, ydot, zdot, rollrat, pitchrate, yawrate)
       and outputs a PD control action in desired body accelerations
       (a_x, a_y, a_z, a_yaw_)"""
    p = x[[0, 1, 2, 5]]
    v = x[[6, 7, 8, 11]]
    return kp * (p_des - p) + kd * v

def get_attitude(acc: np.array) -> np.array:
    """Function that takes the currently desired body acceleration and outputs
       the desired total thrust magnitude and the body attitude """
    pass

def attitude_ctrl(att: np.array, att_des: np.array) -> np.array:
    """Function that takes the current attitude and desired attitude and outputs
       the body rates needed to achieve the orientation that aligns the thrust 
       vector with the desired acceleration. """
    pass
    
def angular_vel_ctrl(ang_vel: np.array, ang_vel_des: np.array) -> np.array:
    """Function that takes the current and desired angular velocity and outputs
       the body torques needed to achieve it. """
    pass
    
def motor_forces_from_torqus(torques: np.array) -> np.array:
    """Function that takes the desired body torques and outputs
       the forces of each motor to achieve it."""
    pass 

def motor_speeds_from_forces(forces: np.array) -> np.array:
    """Function that take the desired motor forces and outputs
       the corresponding motor speeds."""
    pass
    