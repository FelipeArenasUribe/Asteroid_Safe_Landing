import numpy as np
from src import math_utils

class TrackingController:
    def __init__(self, spacecraft, K_r, K_v, tau, c_1, w_bar_0, K_R, K_omega):
        self.spacecraft = spacecraft

        # Feedback Linearizing Gains
        self.K_r = K_r # position gain
        self.K_v = K_v # velocity gain

        # Attitude controller gains
        self.K_R = K_R
        self.K_omega = K_omega

        # Disturbance observer parameters
        self.tau = tau # Observer gain
        self.nu_0 = -self.tau * self.spacecraft.x_0[3:6] # Initial condition

        # Disturbance estimate error bound params
        self.c_1 = c_1
        self.w_bar_0 = w_bar_0

    def desired_force_control(self, x, x_r, u_r, nu):
        # --- Extract states ---
        r = x[0:3]
        v = x[3:6]
        R = x[6:15].reshape(3, 3)
        
        r_d = x_r[0:3]
        v_d = x_r[3:6]
        R_d = x_r[6:15].reshape(3, 3)

        # --- Position and velocity errors ---
        e_r = r - r_d
        e_v = v - v_d

        # --- Desired force ---
        u_F_d = self.spacecraft.m * (R.T @ (
            - self.K_r @ e_r
            - self.K_v @ e_v
            - self.w_estimate(x, nu)
            - self.spacecraft.f(x)[3:6]
            + self.spacecraft.dynamics(x_r,u_r, np.zeros(3))[3:6]
        ))

        return u_F_d
    
    def desired_torque_control(self, x, x_r, u_r):
        # --- Extract states ---
        R = x[6:15].reshape(3, 3)
        omega = x[15:18]

        R_d = x_r[6:15].reshape(3, 3)
        omega_d = x_r[15:18]

        x_r_dot = self.spacecraft.dynamics(x_r,u_r)

        R_d_dot = x_r_dot[6:15].reshape(3, 3)
        omega_d_dot = x_r_dot[15:18]
        
        # --- Attitude control ---
        e_R = math_utils.attitude_error_vector(R, R_d)
        e_omega = math_utils.angular_velocity_error_vector(R, omega, R_d, omega_d)

        # Control torque
        u_T_d = self.spacecraft.J_inv @ (
            - self.K_R @ e_R
            - self.K_omega @ e_omega
            + math_utils.skew_symmetric(R.T @ R_d @ omega_d) @ self.spacecraft.J @ R.T @ R_d @ omega_d
            + self.spacecraft.J @ (R.T @ R_d @ omega_d_dot)
        )

        return u_T_d
    
    def desired_virtual_control(self, x, x_r, u_r, nu):
        """
        Computes the virtual control inputs for the spacecraft
        Includes both translational (forces) and rotational (torques).
        """

        x = x.flatten()
        x_r = x_r.flatten()
        u_r = u_r.flatten()
        nu = nu.flatten()

        u_F_d = self.desired_force_control(x, x_r, u_r, nu)
        u_T_d = self.desired_torque_control(x, x_r, u_r)

        return np.append(u_F_d,u_T_d)
    
    def w_estimate(self, x, nu):
        return self.tau*x[3:6] + nu
    
    def w_estimate_error(self, x, nu, w):
        w_tilde = w - self.w_estimate(x, nu)
        return w_tilde
    
    def nu_dynamics(self, x, u, nu):
        nu_dot = - self.tau * (self.spacecraft.dynamics(x,u)[3:6] + self.w_estimate(x, nu))
        return nu_dot
    
    def dist_error_bound_dyn(self, w_bar, x):
        v = x[3:6]
        return -self.tau * w_bar + (self.c_1**2 / self.tau) * np.linalg.norm(v)**2
    
class ControllerDynamics:
    def __init__(self, spacecraft, tracking_controller, A_c, B_c):
        self.spacecraft = spacecraft

        # Disturbance observer parameters
        self.tau = tracking_controller.tau # Observer gain
        self.nu_0 = tracking_controller.nu_0 # Initial condition
        self.w_bar_0 = tracking_controller.w_bar_0

        # Disturbance estimate error bound params
        self.c_1 = tracking_controller.c_1

        # Control dynamics
        self.A_c = A_c
        self.B_c = B_c

    def w_estimate(self, x, nu):
        return self.tau*x[3:6] + nu
    
    def w_estimate_error(self, x, nu, w):
        w_tilde = w - self.w_estimate(x, nu)
        return w_tilde
    
    def nu_dynamics(self, x, u, nu):
        nu_dot = - self.tau * (self.spacecraft.dynamics(x,u)[3:6] + self.w_estimate(x, nu))
        return nu_dot
    
    def dist_error_bound_dyn(self, w_bar, x):
        v = x[3:6]
        return -self.tau * w_bar + (self.c_1**2 / self.tau) * np.linalg.norm(v)**2

    def control_dynamics(self, u, zeta):
        return self.A_c @ u + self.B_c @ zeta