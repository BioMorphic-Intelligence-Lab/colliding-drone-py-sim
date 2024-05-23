import numpy as np 
from enum import Enum
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
                        i=np.array([6.21e-4, 9.62e-4, 9.22e-4]),
                        feed_forward=np.zeros(3)) -> np.array:
      """Function that takes the current and desired angular velocity and outputs
         the body torques needed to achieve it. """
      ang_vel_error = (ang_vel_des - ang_vel)
      des_accel = ang_vel_error / time_constant
      inertia = np.diag(i)
      nonlinear_corr = np.cross(ang_vel, (inertia @ ang_vel))
      
      return inertia @ (des_accel + feed_forward) + nonlinear_corr
      
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

class TorqueBasedCollisionRecoveryController(Controller):

   class State(Enum):
      POSITION_CTRL = 0
      RECOVERING = 1
      HOVER = 2

   def __init__(self, drone) -> None:
      super().__init__()
      self.drone = drone
      self.state = self.State.POSITION_CTRL
      self.new_ref = np.zeros(3)
      self.new_p_ref = np.zeros(3)

   def reset(self):
      self.state = self.State.POSITION_CTRL
      self.new_ref = np.zeros(3)

   def find_feed_forward(self, x, in_contact):
      feed_forward_torque = np.zeros(3)

      feed_forward_torque += np.array([10, 0, 0], dtype=float)

      return feed_forward_torque
   
   def attitude_ctrl_with_ff(self, x: np.array, att_des: np.array, ff=0.0,
                             m=0.315, g=9.81):
      att = x[3:6]
      ang_vel = x[9:12]
      ang_vel_des = super().attitude_ctrl(att, att_des)

      # Find Total Thrust in order to maintain altitude
      total_thrust = m * g
      # this ensures that we always produce the required z-acceleration (up to some maximum)
      MIN_THRUST_CORR_FAC = 1.00
      thrustCorrFactor = (R.from_euler(seq="xyz", angles=att).apply([0, 0, 1]))[2]
      thrustCorrFactorSaturated = ((thrustCorrFactor <  MIN_THRUST_CORR_FAC) * MIN_THRUST_CORR_FAC 
                                 + (thrustCorrFactor >= MIN_THRUST_CORR_FAC) * thrustCorrFactor)
      totalNormThrust = total_thrust / thrustCorrFactorSaturated
     
      return self.motor_speeds_from_forces(
                        self.motor_forces_from_torques(
                              torques=self.angular_vel_ctrl(
                                             ang_vel=ang_vel, 
                                             ang_vel_des=ang_vel_des,
                                             feed_forward=ff),
                              tot_thrust=totalNormThrust                                        
                           )
                        )
   
   def find_target_attitude(self, in_contact: np.array) -> np.array:
      attitude = np.zeros(3)

      attitude += np.array([np.deg2rad(5), 0, 0])

      return attitude

   def u(self, x, x_des, v_des, yaw_des, m=0.315, g=9.81):

      # Find current position of vertices
      rot = R.from_euler(seq="xyz", angles=x[3:6])
      vertices = rot.apply(self.drone.vertices_nominal) + x[0:3]

      # Find which vertices are in contact
      in_contact = self.drone.is_in_contact(vertices)

      # Check if we are in contact at all
      if in_contact.any():
         # Find the feed forward term
         feed_forward_term = self.find_feed_forward(x, in_contact)

         # Compute Target Attitude
         self.new_ref = self.find_target_attitude(in_contact)

         # Compute Hover position
         self.new_p_ref = x[0:3] - np.array([0, 0.25, 0])

         # We're now in the state of recovering
         self.state = self.State.RECOVERING

         # Return control action Attitude Control + Feed Forward
         return self.attitude_ctrl_with_ff(x=x, att_des=self.new_ref,
                                           ff=feed_forward_term)

      # We're not in immediate contact, but are still recovering
      elif self.state == self.State.RECOVERING:

         # If the angular velocity falls under a threshold 
         # We have recovered and can go back to position control
         if np.linalg.norm(x[9:12]) < np.deg2rad(1):
            self.state = self.State.HOVER
            return self.attitude_ctrl_with_ff(x=x, att_des=[0, 0, 0])
         # Otherwise keep attitude control
         else:
            # Find Total Thrust in order to maintain altitude
            total_thrust = m * g
            # this ensures that we always produce the required z-acceleration (up to some maximum)
            MIN_THRUST_CORR_FAC = 1.00
            thrustCorrFactor = (R.from_euler(seq="xyz", angles=x[3:6]).apply([0, 0, 1]))[2]
            thrustCorrFactorSaturated = ((thrustCorrFactor <  MIN_THRUST_CORR_FAC) * MIN_THRUST_CORR_FAC 
                                       + (thrustCorrFactor >= MIN_THRUST_CORR_FAC) * thrustCorrFactor)
            totalNormThrust = total_thrust / thrustCorrFactorSaturated
         
            return self.motor_speeds_from_forces(
                              self.motor_forces_from_torques(
                                    torques=self.angular_vel_ctrl(x[9:12], 
                                                   self.attitude_ctrl(x[3:6], att_des=self.new_ref)),
                                    tot_thrust=totalNormThrust                                        
                                 )
                              )

      elif self.state == self.State.HOVER:
         return super().u(x, self.new_p_ref, np.zeros(3), 0)
      # We're not in contact and aren't recovering
      else:
         # Simply do position control
         return super().u(x,x_des,v_des,yaw_des)
      
class VelocityPredictionRecoveryController(Controller):
   def __init__(self, drone) -> None:
      super().__init__()
      self.drone = drone
      self.collision_occured = False
      self.new_ref = np.zeros(3)

   def reset(self):
      self.collision_occured = False

   def predict_delta_vel(self, x, v, u, in_contact, dt,
                   e=0.8, # coefficient of restitution
                   g=9.81):
      
      # Find the rotation matrix of the drone
      rot = R.from_euler(seq="xyz", angles=x[3:6])

      # Find the linear velocity vector in the body frame
      v_b = rot.apply(v[0:3])

      # We ignore all contact points that are not in the 
      # the direction of travel
      if v_b[0] <= 0: 
         # Traveling in negative x
         in_contact[1] = False
         in_contact[3] = False
      else:
         # Traveling in positive x
         in_contact[0] = False
         in_contact[2] = False
             
      if v_b[1] <= 0:
         # Traveling in negaitve y
         in_contact[5] = False
         in_contact[7] = False
      else:
         # Traveling in positive y
         in_contact[4] = False
         in_contact[6] = False
      

      delta_vel = np.array([0, 0, -g * dt, 0, 0, 0], dtype=float)
      if in_contact.any():
         scale = 1.0 / np.sum(1 * in_contact)
         normal = np.array([0, -1, 0], dtype=float)
         inv_I = np.linalg.inv(self.drone.I)
         for i in range(len(in_contact)):
            if in_contact[i]:
               if not self.collision_occured:
                  r = rot.apply(self.drone.vertices_nominal[i])
                  rel_vel = np.dot(normal,
                                   v[0:3] + np.cross(v[3:6], r))
                  
                  j = (-(1 + e) * rel_vel
                        / (1.0 / self.drone.m 
                           + np.dot(normal,
                                    inv_I @ np.cross(np.cross(r, normal),
                                                     r)
                                    )
                           )
                     )

                  delta_lin_vel = scale * j / self.drone.m * normal
                  delta_rot_vel = scale * inv_I @ (j * np.cross(r, normal))

                  delta_vel += np.concatenate((
                     delta_lin_vel,
                     delta_rot_vel)
                  )
               else:
                  delta_vel += np.concatenate((
                     [0, 0, 0],
                     [0, 0, 0])
                  )
         
         self.collision_occured = True
      else:
         if self.collision_occured:
            self.collision_occured = False

         delta_vel += np.concatenate((
                  [0, 0, 0],
                  [0, 0, 0])
               )
      return delta_vel

   def u(self, t, x, x_des, v_des, yaw_des):
      # Find current position of vertices
      rot = R.from_euler(seq="xyz", angles=x[3:6])
      vertices = rot.apply(self.drone.vertices_nominal) + x[0:3]

      # Find which vertices are in contact
      in_contact = self.drone.is_in_contact(vertices)

      if in_contact.any() or self.collision_occured:
         
         self.collision_occured = True
         return np.zeros(4)
      else:
         return super().u(x,x_des,v_des,yaw_des)