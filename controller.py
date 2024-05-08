import numpy as np 
from scipy.spatial.transform import Rotation as R

class Controller(object):
   def __init__(self) -> None:
      pass

   def position_ctrl(self, x: np.array, p_des: np.array, v_des: np.array,
                     acc_des=np.zeros(3), nat_freq=2.0, damping=0.7) -> np.array:
      """Function that takes the current state 
         (x,y,z, roll, pitch, yaw, xdot, ydot, zdot, rollrate, pitchrate, yawrate)
         and desired position and outputs a PD control action in desired body
         accelerations (a_x, a_y, a_z)"""
      p = x[0:3]
      v = x[6:9]

      p_err = (p_des - p)
      v_err = (v_des - v)

      acc = (p_err * nat_freq ** 2
            + v_err * 2 * nat_freq * damping + acc_des)
      return acc

   def get_attitude_and_thrust(self, acc: np.array,
                              yaw_des: float,
                              att: np.array,
                              mass=0.315,
                              g=9.81) -> np.array:
      """Function that takes the currently desired body acceleration and outputs
         the desired total thrust magnitude and the body attitude """

      #Compute total thrust, thrust direction:
      acc += np.array([0, 0, mass * g])

      # Saturate negative z axis acceleration
      if acc[2] < 0:
         acc[2] = 0

      normDesProperAcc = np.linalg.norm(acc)
      if normDesProperAcc < 1e-6:
         desThrustDir = np.array([0, 0, 1])
      else:
         desThrustDir = acc / normDesProperAcc

      # this ensures that we always produce the required z-acceleration (up to some maximum)
      MIN_THRUST_CORR_FAC = 1.00
      thrustCorrFactor = (R.from_euler(seq="xyz", angles=att).apply([0, 0, 1]))[2]
      thrustCorrFactorSaturated = ((thrustCorrFactor <  MIN_THRUST_CORR_FAC) * MIN_THRUST_CORR_FAC 
                                 + (thrustCorrFactor >= MIN_THRUST_CORR_FAC) * thrustCorrFactor)
      totalNormThrust = normDesProperAcc / thrustCorrFactorSaturated


      # Construct the attitude that would match this desired thrust direction
      e3 = np.array([0, 0, 1], dtype=float)
      cosAngle = np.dot(desThrustDir, e3)
      angle = np.arccos(cosAngle)

      rotAx = np.cross(e3, desThrustDir)
      n = np.linalg.norm(rotAx)

      if n < 1e-6:
         desAtt = R.identity()
      else:
         desAtt = R.from_rotvec(rotAx * (angle / n))

      desAtt = desAtt * R.from_euler(seq="xyz", angles=[0, 0, yaw_des])

      return desAtt, totalNormThrust

   def attitude_ctrl(self, att: np.array, att_des: np.array,
                     time_constant=np.array([0.08, 0.08, 0.08])) -> np.array:
      """Function that takes the current attitude and desired attitude and outputs
         the body rates needed to achieve the orientation that aligns the thrust 
         vector with the desired acceleration. """
      attR = R.from_euler(seq="xyz", angles=att)
      att_desR = R.from_euler(seq="xyz", angles=att_des)
      att_err = (att_desR.inv() * attR)


      # Find the rotation axis
      desRedAttRotAx = np.cross(att_err.inv().apply([0, 0, 1]),
                              [0,0,1])
      
      # Find the cos of the rotation angle about the above axis
      desRedAttRotAn_cos = np.dot(att_err.inv().apply([0, 0, 1]),
                                 [0, 0, 1])
      # Handle angle crossover
      if desRedAttRotAn_cos >= 1.0:
         desRedAttRotAn = 0
      elif desRedAttRotAn_cos <= -1.0:
         desRedAttRotAn = np.pi
      else:
         desRedAttRotAn = np.arccos(desRedAttRotAn_cos)
      
      # normalize
      n = np.linalg.norm(desRedAttRotAx)
      if n < 1e-12:
         desRedAttRotAx = np.zeros(3)
      else:
         desRedAttRotAx = desRedAttRotAx / n

      k3 = 1.0 / time_constant[2]
      k12 = 1.0 / time_constant[0]

      desAngVel = (-k3 * att_err.as_euler(seq="xyz") 
                  - (k12 - k3) * desRedAttRotAn * desRedAttRotAx)
      return desAngVel

   def angular_vel_ctrl(self, ang_vel: np.array, ang_vel_des: np.array,
                        time_constant=np.array([0.04, 0.04, 0.14]),
                        i=np.array([6.21e-4, 9.62e-4, 9.22e-4])) -> np.array:
      """Function that takes the current and desired angular velocity and outputs
         the body torques needed to achieve it. """
      ang_vel_error = (ang_vel_des - ang_vel)
      des_accel = ang_vel_error / time_constant
      inertia = np.diag(i)
      nonlinear_corr = np.cross(ang_vel, (inertia @ ang_vel))
      
      return inertia @ des_accel + nonlinear_corr
      
   def motor_forces_from_torques(self, torques: np.array, tot_thrust: float,
                                 prop=[0.0525, 0.040], yd=0.009975,
                                 max_thrust_per_propeller=2.6,
                                 min_thrust_per_propeller=0.08,
                                 ) -> np.array:
      """Function that takes the desired body torques and total thrust
         and outputs the forces of each motor to achieve it."""
      f = 0.25 * np.array([
         -torques[0] / prop[1] - torques[1] / prop[0] - torques[2] / yd + tot_thrust,
         torques[0] / prop[1] + torques[1] / prop[0] - torques[2] / yd + tot_thrust,
         torques[0] / prop[1] - torques[1] / prop[0] + torques[2] / yd + tot_thrust,
         -torques[0] / prop[1] + torques[1] / prop[0] + torques[2] / yd + tot_thrust,
      ])

      # Saturate thrust of each proellor
      f[f > max_thrust_per_propeller] = max_thrust_per_propeller
      f[f < min_thrust_per_propeller] = min_thrust_per_propeller

      return f
      
   def motor_speeds_from_forces(self, forces: np.array, th=7.50e-8) -> np.array:
      """Function that take the desired motor forces and outputs
         the corresponding motor speeds."""
      
      return np.sqrt(forces / th)
   
   def reset(self):
      pass

   def u(self, x, x_des, v_des, yaw_des):
      # Find the desired (linear) acceleration
      acc = (self.position_ctrl(x, x_des, v_des))

      # Find desired attitude and total thrust
      des_attitude, tot_thrust = self.get_attitude_and_thrust(
               acc,
               yaw_des=yaw_des,
               att=x[3:6]
            )
      
      # Return the motor speeds for the above
      return self.motor_speeds_from_forces(
                        self.motor_forces_from_torques(
                           torques=self.angular_vel_ctrl(x[9:12], 
                                          self.attitude_ctrl(x[3:6],
                                                      des_attitude.as_euler("xyz"))
                                          ),
                              tot_thrust=tot_thrust                                        
                           )
                        )
      
class CollisionRecoveryController(Controller):
   def __init__(self, drone) -> None:
      super().__init__()
      self.drone = drone
      self.collision_occured = False
      self.new_ref = np.zeros(3)

   def reset(self):
      self.collision_occured = False

   def u(self, x, x_des, v_des, yaw_des):
      # Find current position of vertices
      rot = R.from_euler(seq="xyz", angles=x[3:6])
      vertices = rot.apply(self.drone.vertices_nominal) + x[0:3]

      # Find which vertices are in contact
      in_contact = self.drone.is_in_contact(vertices)

      if in_contact.any() or self.collision_occured:
         if self.collision_occured == False:
            self.new_ref = x[0:3] - np.array([0, 0.5, 0])
         
         self.collision_occured = True
         return super().u(x, self.new_ref, np.zeros(3), yaw_des)
      else:
         return super().u(x,x_des,v_des,yaw_des)
