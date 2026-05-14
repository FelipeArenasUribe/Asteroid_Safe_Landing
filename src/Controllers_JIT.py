import numpy as np
from src import math_utils
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.scipy.special import logsumexp

@partial(jit, static_argnums=())
def _compute_safe_control_jit(
    # --- system state ---
    x, u, nu, w_d, x_r, u_r, u_r_dot,
    # --- controller dynamics ---
    A_c, B_c, B_c_inv,
    # --- tracking gains ---
    K_r, K_v, K_R, K_omega,
    # --- disturbance observer ---
    tau, nu_0, c_1, w_bar_0,
    # --- spacecraft properties ---
    m, J, J_inv, D,
    # --- celestial body ---
    mu, omega_A, omega_A_hat,
    # --- surrogate filter ---
    sigma,
    # --- safety filter ---
    alpha, rho, gamma, gamma_inv,
    k_state, k_input,
    beta_1_0, beta_1_1,
    beta_2_0, beta_2_1,
    beta_3_0, beta_4_0,
    # --- constraints ---
    x_f, e_s, theta_gs, beta_gs, theta_b,
    v_max, v_l, omega_max, omega_l,
    A_tilde, b_tilde,
):
    # ------------------------------------------------------------------ helpers
    def _skew(v):
        return jnp.array([
            [ 0.0,  -v[2],  v[1]],
            [ v[2],   0.0, -v[0]],
            [-v[1],  v[0],   0.0],
        ])

    def _skew_to_vec(S):
        return jnp.array([S[2, 1], S[0, 2], S[1, 0]])

    def smooth_norm(xi, eps=1e-3):
        return jnp.sqrt(eps + jnp.dot(xi, xi))

    def eta_transition(a, k=1.0e-5):
        '''
        Parameters:
        a : float (The input, typically ||r - rf||^2)
        k : float (The inverse of the transition radius squared)
            
        Returns:
        float : The value of eta(a).

        Example tuning parameters:
        Start Braking Distance |    k Value
        200 m                       2.5*10^-5
        100 m                       1.0*10^-4
        50 m                        4.0*10^-4
        10 m                        0.01
        '''
        ka = jnp.clip(k * a, 0.0, 1.0)
        return ka - jnp.sin(2 * jnp.pi * ka) / (2 * jnp.pi)

    # ------------------------------------------------------------------ dynamics
    def f(x):
        r, v  = x[0:3], x[3:6]
        R     = x[6:15].reshape(3, 3)
        omega = x[15:18]

        grav  = -(mu / jnp.linalg.norm(r)**3) * r
        v_dot = grav - 2.0 * omega_A_hat @ v - omega_A_hat @ omega_A_hat @ r
        R_dot = R @ _skew(omega)

        omega_total = R.T @ omega_A + omega
        omega_dot   = -J_inv @ jnp.cross(omega_total, J @ omega_total)

        return jnp.concatenate([v, v_dot, R_dot.flatten(), omega_dot])

    def g(x):
        R = x[6:15].reshape(3, 3)
        G = jnp.zeros((18, 6))
        G = G.at[3:6,  0:3].set((1.0 / m) * R)
        G = G.at[15:18, 3:6].set(J_inv)
        return G

    def dynamics(x, u_tilde, w=jnp.zeros(3)):
        return f(x) + g(x) @ u_tilde + D @ w

    def w_estimate(x, nu):
        return tau * x[3:6] + nu

    def nu_dynamics(x, u, nu):
        return -tau * (dynamics(x, u)[3:6] + w_estimate(x, nu))

    def dist_error_bound_dyn(w_bar, x):
        v = x[3:6]
        return -tau * w_bar + (c_1**2 / tau) * jnp.dot(v, v)

    # ------------------------------------------------------------------ desired virtual control (needed for zeta_d)
    def desired_force_control(x, x_r, x_r_dot, u_r, nu):
        r, v     = x[0:3], x[3:6]
        r_d, v_d = x_r[0:3], x_r[3:6]
        R_d      = x_r[6:15].reshape(3, 3)
        e_r = r - r_d
        e_v = v - v_d
        return m * (R_d.T @ (
            - K_r @ e_r
            - K_v @ e_v
            - w_estimate(x, nu)
            - f(x)[3:6]
            + dynamics(x_r, u_r)[3:6]
        ))

    def desired_torque_control(x, x_r, x_r_dot):
        R, omega = x[6:15].reshape(3, 3), x[15:18]
        R_d          = x_r[6:15].reshape(3, 3)
        omega_d      = x_r[15:18]
        omega_d_dot  = x_r_dot[15:18]

        RtRd      = R_d.T @ R
        trace_val = jnp.trace(RtRd)
        scale     = 1.0 / (2.0 * jnp.sqrt(jnp.maximum(1e-6, 1.0 + trace_val)))
        S         = RtRd - RtRd.T
        e_R       = scale * _skew_to_vec(S)
        e_omega   = omega - R.T @ R_d @ omega_d
        omega_ref = R.T @ R_d @ omega_d

        return J_inv @ (
            - K_R @ e_R
            - K_omega @ e_omega
            + jnp.cross(omega_ref, J @ omega_ref)
            + J @ (R.T @ R_d @ omega_d_dot)
        )

    def desired_virtual_control(x, x_r, u_r, nu):
        x_r_dot = dynamics(x_r, u_r)
        return jnp.concatenate([
            desired_force_control(x, x_r, x_r_dot, u_r, nu),
            desired_torque_control(x, x_r, x_r_dot),
        ])

    def desired_surrogate_control(x, u, x_r, u_r, nu, u_d, u_r_dot):
        du_dx     = jax.jacobian(desired_virtual_control, argnums=0)(x, x_r, u_r, nu)
        du_dx_ref = jax.jacobian(desired_virtual_control, argnums=1)(x, x_r, u_r, nu)
        du_du_ref = jax.jacobian(desired_virtual_control, argnums=2)(x, x_r, u_r, nu)
        du_dnu    = jax.jacobian(desired_virtual_control, argnums=3)(x, x_r, u_r, nu)

        x_dot   = dynamics(x, u, w_estimate(x, nu))
        nu_dot  = nu_dynamics(x, u, nu)
        x_r_dot = dynamics(x_r, u_r)

        u_d_dot = (
            du_dx     @ x_dot
          + du_dnu    @ nu_dot
          + du_dx_ref @ x_r_dot
          + du_du_ref @ u_r_dot
        )
        return B_c_inv @ (-A_c @ u + u_d_dot - sigma * (u - u_d))

    # ------------------------------------------------------------------ CBF barrier functions
    def psi_1_0(x):
        r = x[0:3]
        delta_r = r - beta_gs * x_f[0:3].squeeze()
        dist = jnp.sqrt(jnp.sum(jnp.square(delta_r)) + 1e-12)
        return jnp.dot(e_s, delta_r) - dist * jnp.cos(theta_gs)

    def psi_1_1(x):
        return jax.grad(psi_1_0)(x) @ f(x) + beta_1_0 * psi_1_0(x)

    def psi_1_2_bar(x, u, nu, w_d):
        dpsi = jax.grad(psi_1_1)(x)
        return (
            dpsi @ f(x)
          + dpsi @ (g(x) @ u)
          + dpsi @ (D @ (tau * x[3:6] + nu))
          - smooth_norm(dpsi @ D) * jnp.sqrt(w_d + 1e-12)
          + beta_1_1 * psi_1_1(x)
        )

    def psi_2_0(x):
        r  = x[0:3]
        R  = x[6:15].reshape(3, 3)
        e_b     = -R[:, 2]
        rel_pos = beta_gs * x_f[0:3].squeeze() - r
        dist    = jnp.sqrt(jnp.sum(jnp.square(rel_pos)) + 1e-12)
        return e_b @ rel_pos - dist * jnp.cos(theta_b)

    def psi_2_1(x):
        return jax.grad(psi_2_0)(x) @ f(x) + beta_2_0 * psi_2_0(x)

    def psi_2_2_bar(x, u, nu, w_d):
        dpsi = jax.grad(psi_2_1)(x)
        return (
            dpsi @ f(x)
          + dpsi @ (g(x) @ u)
          + dpsi @ (D @ (tau * x[3:6] + nu))
          - smooth_norm(dpsi @ D) * jnp.sqrt(w_d + 1e-12)
          + beta_2_1 * psi_2_1(x)
        )

    def psi_3_0(x):
        r, v = x[0:3], x[3:6]
        r_f, v_f = x_f[0:3], x_f[3:6]
        sqrd_pos = jnp.sum(jnp.square(r - r_f))
        sqrd_vel = jnp.sum(jnp.square(v - v_f))
        eta = eta_transition(sqrd_pos)
        v_limit_sq = eta * v_max**2 + (1 - eta) * v_l**2
        return v_limit_sq - sqrd_vel

    def psi_3_1_bar(x, u, nu, w_d):
        dpsi = jax.grad(psi_3_0)(x)
        return (
            dpsi @ f(x)
          + dpsi @ (g(x) @ u)
          + dpsi @ (D @ (tau * x[3:6] + nu))
          - smooth_norm(dpsi @ D) * jnp.sqrt(w_d + 1e-12)
          + beta_3_0 * psi_3_0(x)
        )

    def psi_4_0(x):
        r, omega = x[0:3], x[15:18]
        r_f = x_f[0:3]
        sqrd_pos    = jnp.sum(jnp.square(r - r_f))
        sqrd_ang_vel = jnp.sum(jnp.square(omega))
        eta = eta_transition(sqrd_pos)
        omega_limit_sq = eta * omega_max**2 + (1 - eta) * omega_l**2
        return omega_limit_sq - sqrd_ang_vel

    def psi_4_1(x, u):
        dpsi = jax.grad(psi_4_0)(x)
        return (
            dpsi @ f(x)
          + dpsi @ (g(x) @ u)
          + beta_4_0 * psi_4_0(x)
        )

    def phi(u_tilde):
        return A_tilde @ u_tilde + b_tilde

    # ------------------------------------------------------------------ softmin CBF
    def softmin_h(x, u, nu, w_d):
        b_state = jnp.array([
            psi_1_2_bar(x, u, nu, w_d),
            psi_2_2_bar(x, u, nu, w_d),
            psi_3_1_bar(x, u, nu, w_d),
            psi_4_1(x, u),
        ])
        b_state = jnp.multiply(k_state, b_state)
        b_input = jnp.multiply(k_input, phi(u))
        b = jnp.append(b_state, b_input)
        return -(1.0 / rho) * logsumexp(-rho * b)

    # ------------------------------------------------------------------ CBF condition
    def b_func(x, u, nu, w_d, zeta, kappa):
        dh_dx   = jax.grad(softmin_h, argnums=0)(x, u, nu, w_d)
        dh_du   = jax.grad(softmin_h, argnums=1)(x, u, nu, w_d)
        dh_dnu  = jax.grad(softmin_h, argnums=2)(x, u, nu, w_d)
        dh_dw_d = jax.grad(softmin_h, argnums=3)(x, u, nu, w_d)

        L_f_h = (
            dh_dx   @ dynamics(x, u, w=jnp.zeros(3))
          + dh_du   @ (A_c @ u)
          + dh_dnu  @ nu_dynamics(x, u, nu)
          + dh_dw_d * dist_error_bound_dyn(w_d, x)
        )
        L_g_h = dh_du @ B_c
        L_d_h = dh_dx @ D

        return (
            L_f_h
          + L_g_h @ zeta
          + L_d_h @ w_estimate(x, nu)
          - smooth_norm(L_d_h) * jnp.sqrt(w_d)
          + alpha  * softmin_h(x, u, nu, w_d)
          + kappa  * softmin_h(x, u, nu, w_d)
        )

    # ------------------------------------------------------------------ safe control law
    def safe_control(x, u, nu, w_d, zeta_d):
        varphi = b_func(x, u, nu, w_d, zeta_d, jnp.array(0.0))
        h      = softmin_h(x, u, nu, w_d)
        dh_du  = jax.grad(softmin_h, argnums=1)(x, u, nu, w_d)

        L_g_h  = dh_du @ B_c

        norm_sq = jnp.dot(L_g_h, L_g_h)

        denom   = norm_sq + gamma_inv * h**2

        var     = (-varphi) / denom

        lam     = jnp.maximum(0, var)

        kappa   = (h * lam) / gamma

        zeta    = zeta_d + lam * jnp.transpose((L_g_h))

        return zeta, kappa, lam

    # ------------------------------------------------------------------ main computation
    u_d    = desired_virtual_control(x, x_r, u_r, nu)
    zeta_d = desired_surrogate_control(x, u, x_r, u_r, nu, u_d, u_r_dot)
    zeta, kappa, lam = safe_control(x, u, nu, w_d, zeta_d)

    return zeta, u_d, lam
    
class JIT_SafetyController:
    def __init__(self, spacecraft, controller_dynamics, tracking_controller, sigma, betas, k_state, k_input, rho, alpha, gamma):

        self.spacecraft = spacecraft
        self.controller_dynamics = controller_dynamics
        self.tracking_controller = tracking_controller

        
        # --- Safety Filter Parameters ---
        self.sigma = jnp.asarray(sigma)     # Desired surrogate control dynamic constant

        self.beta_1_0 = jnp.asarray(betas["beta_1_0"])
        self.beta_1_1 = jnp.asarray(betas["beta_1_1"])

        self.beta_2_0 = jnp.asarray(betas["beta_2_0"])
        self.beta_2_1 = jnp.asarray(betas["beta_2_1"])

        self.beta_3_0 = jnp.asarray(betas["beta_3_0"])

        self.beta_4_0 = jnp.asarray(betas["beta_4_0"])

        self.alpha = jnp.asarray(alpha)

        self.k_state = jnp.asarray(k_state)
        self.k_input = jnp.asarray(k_input)

        self.rho = jnp.asarray(rho)

        self.gamma = jnp.asarray(gamma)

        self.gamma_inv = 1/self.gamma

        # --- Control Dynamics ---

        self.A_c = jnp.asarray(controller_dynamics.A_c)
        self.B_c = jnp.asarray(controller_dynamics.B_c)
        self.B_c_inv = jnp.asarray(np.linalg.inv(controller_dynamics.B_c))

        # --- Tracking Controller Params ---

        # Feedback Linearizing Gains
        self.K_r = jnp.asarray(tracking_controller.K_r)
        self.K_v = jnp.asarray(tracking_controller.K_v) # velocity gain

        # Attitude controller gains
        self.K_R = jnp.asarray(tracking_controller.K_R)
        self.K_omega = jnp.asarray(tracking_controller.K_omega)

        # Disturbance observer parameters
        self.tau = jnp.asarray(tracking_controller.tau) # Observer gain
        self.nu_0 = jnp.asarray(tracking_controller.nu_0) # Initial condition

        # Disturbance estimate error bound params
        self.c_1 = jnp.asarray(tracking_controller.c_1)
        self.w_bar_0 = jnp.asarray(tracking_controller.w_bar_0)
    
        # --- System Properties ---
        # Known Celestial body parameters
        self.omega_A = spacecraft.CB_omega
        self.omega_A_hat = jnp.array(math_utils.skew_symmetric(spacecraft.CB_omega))
        mass = sum(body.mass for body in spacecraft.known_CB.sub_bodies) # total mass of celestial body
        G = 6.67430e-11
        self.mu = jnp.asarray(G * mass)

        # Spacecraft properties
        self.m = jnp.asarray(spacecraft.m)
        self.J = jnp.asarray(spacecraft.J)
        self.J_inv = jnp.asarray(spacecraft.J_inv)

        self.D = jnp.asarray(spacecraft.D)

        # Input constraint and allocation matrices
        self.M = jnp.asarray(spacecraft.M)
        self.A_tilde = jnp.asarray(spacecraft.A_tilde)
        self.b_tilde = jnp.asarray(spacecraft.b_tilde)
    
        # Initial and final conditions
        self.x_0 = jnp.asarray(spacecraft.x_0)
        self.x_f = jnp.asarray(spacecraft.x_f)

        # Landing parameters
        self.r_l = jnp.asarray(spacecraft.r_l)
        self.v_l = jnp.asarray(spacecraft.v_l)
        self.theta_l = jnp.asarray(spacecraft.theta_l)
        self.omega_l = jnp.asarray(spacecraft.omega_l)

        # Glideslope constraint
        self.theta_gs = jnp.asarray(spacecraft.theta_gs)
        self.e_s = jnp.asarray(spacecraft.e_s)
        self.beta_gs = jnp.asarray(spacecraft.beta)

        # Field-of-view constraint
        self.theta_b = jnp.asarray(spacecraft.theta_b)

        # Speed constraint
        self.v_max = jnp.asarray(spacecraft.v_max)

        # Angular speed constraint
        self.omega_max = jnp.asarray(spacecraft.omega_max)

    def f(self, x):
        r = x[0:3]
        v = x[3:6]
        R = x[6:15].reshape(3, 3)
        omega = x[15:18]

        r_dot = v

        grav = - (self.mu / jnp.linalg.norm(r)**3) * r

        v_dot = (grav - 2.0 * self.omega_A_hat @ v - self.omega_A_hat @ self.omega_A_hat @ r)

        # Ensure math_utils uses jnp-based skew symmetric logic
        R_dot = R @ math_utils.jnp_skew_symmetric(omega)

        omega_total = R.T @ self.omega_A + omega

        omega_dot = -self.J_inv @ jnp.cross(omega_total, self.J @ omega_total)

        return jnp.concatenate([
            r_dot,
            v_dot,
            R_dot.flatten(),
            omega_dot
        ])

    def g(self, x):
        R = x[6:15].reshape(3, 3)
        
        # Initialize as JAX array
        G = jnp.zeros((18, 6))

        # Use functional updates instead of assignment
        G = G.at[3:6, 0:3].set((1.0 / self.m) * R)
        G = G.at[15:18, 3:6].set(self.J_inv)
        
        return G
 
    def dynamics(self, x, u_tilde, w=jnp.zeros(3)):
        return self.f(x) + self.g(x) @ u_tilde + self.D @ w
    
    def desired_force_control(self, x, x_r, x_r_dot, u_r, nu):
        r, v = x[0:3], x[3:6]
        r_d, v_d = x_r[0:3], x_r[3:6]
        R_d = x_r[6:15].reshape(3, 3)

        e_r = r - r_d
        e_v = v - v_d
        
        u_F_d = self.m * (R_d.T @ (
            - self.K_r @ e_r
            - self.K_v @ e_v
            - self.w_estimate(x, nu)
            - self.f(x)[3:6]
            + self.dynamics(x_r,u_r)[3:6]
        ))
        return u_F_d

    def desired_torque_control(self, x, x_r, x_r_dot):
        R = x[6:15].reshape(3, 3)
        omega = x[15:18]
        R_d = x_r[6:15].reshape(3, 3)
        omega_d = x_r[15:18]
        omega_d_dot = x_r_dot[15:18]
        
        # --- Attitude control ---
        RtRd = R_d.T @ R
        # Added epsilon for numerical stability in gradients
        trace_val = jnp.trace(RtRd)
        scale = 1.0 / (2.0 * jnp.sqrt(jnp.maximum(1e-6, 1.0 + trace_val)))
        
        S = RtRd - RtRd.T
        e_R = scale * math_utils.jnp_skew_symmetric_to_vector(S)
        e_omega = omega - R.T @ R_d @ omega_d

        # Ensure all math_utils calls use jnp internally
        omega_ref = R.T @ R_d @ omega_d
        
        u_T_d = self.J_inv @ (
            - self.K_R @ e_R
            - self.K_omega @ e_omega
            + jnp.cross(omega_ref, self.J @ omega_ref) # Often cleaner than skew @ J @ omega
            + self.J @ (R.T @ R_d @ omega_d_dot)
        )
        return u_T_d

    def desired_virtual_control(self, x, x_r, u_r, nu):
        x = x.ravel()
        x_r = x_r.ravel()
        u_r = u_r.ravel()
        nu = nu.ravel()

        # Compute reference dynamics once
        x_r_dot = self.dynamics(x_r, u_r)

        u_F_d = self.desired_force_control(x, x_r, x_r_dot, u_r, nu)
        u_T_d = self.desired_torque_control(x, x_r, x_r_dot)

        return jnp.concatenate([u_F_d, u_T_d])    
    
    def w_estimate(self, x, nu):
        w_hat = self.tau*x[3:6] + nu
        return w_hat
    
    def nu_dynamics(self, x, u, nu):
        nu_dot = - self.tau * (self.dynamics(x,u)[3:6] + self.w_estimate(x, nu))
        return nu_dot
    
    def dist_error_bound_dyn(self, w_bar, x):
        v = x[3:6]
        return -self.tau * w_bar + (self.c_1**2 / self.tau) * jnp.dot(v,v)
    
    def control_dynamics(self, u, zeta):
        return self.A_c @ u + self.B_c @ zeta

    def desired_surrogate_control(self, x, u, x_r, u_r, nu, u_d, u_r_dot):

        # --- Compute gradients ---
        f = partial(self.desired_virtual_control)

        du_dx     = jax.jacobian(f, argnums=0)(x, x_r, u_r, nu)
        du_dx_ref = jax.jacobian(f, argnums=1)(x, x_r, u_r, nu)
        du_du_ref = jax.jacobian(f, argnums=2)(x, x_r, u_r, nu)
        du_dnu    = jax.jacobian(f, argnums=3)(x, x_r, u_r, nu)

        # --- Compute derivatives ---
        x_dot = self.dynamics(x, u, self.w_estimate(x, nu))
        nu_dot = self.nu_dynamics(x, u, nu)
        x_r_dot = self.dynamics(x_r, u_r)


        # --- Compute u_d_dot ---
        u_d_dot = (
            du_dx     @ x_dot
        + du_dnu    @ nu_dot
        + du_dx_ref @ x_r_dot
        + du_du_ref @ u_r_dot
        )

        # --- Surrogate control ---
        return self.B_c_inv @ (- self.A_c @ u + u_d_dot - self.sigma * (u - u_d))

    def psi_1_0(self, x):
        r = x[0:3]
        x_f = self.x_f[0:3].squeeze()

        delta_r = r - self.beta_gs * x_f

        # Safe norm calculation to avoid NaN gradients at delta_r = 0
        dist = jnp.sqrt(jnp.sum(jnp.square(delta_r)) + 1e-12)     

        return jnp.dot(self.e_s, delta_r) - dist * jnp.cos(self.theta_gs)
    
    def psi_1_1(self, x):
        dpsi1_dx = jax.grad(self.psi_1_0)(x)
        return dpsi1_dx @ self.f(x) + self.beta_1_0 * self.psi_1_0(x)
    
    def psi_1_2_bar(self, x, u, nu, w_d):
        
        psi_1_1_dx = jax.grad(self.psi_1_1)(x)

        term1 = psi_1_1_dx @ self.f(x)
        term2 = psi_1_1_dx @ (self.g(x) @ u)
        term3 = psi_1_1_dx @ (self.D @ (self.tau * x[3:6] + nu))
        term4 = -self.smooth_norm(psi_1_1_dx @ self.D) * jnp.sqrt(w_d + 1e-12)

        term5 = self.beta_1_1 * self.psi_1_1(x)

        return term1 + term2 + term3 + term4 + term5
        
    def psi_2_0(self, x):
        r = x[0:3]
        R = x[6:15].reshape(3, 3)
        
        e_b = -R[:, 2] 
        
        rel_pos = self.beta_gs * self.x_f[0:3].squeeze() - r 

        dist = jnp.sqrt(jnp.sum(jnp.square(rel_pos)) + 1e-12)

        return e_b @ rel_pos - dist * jnp.cos(self.theta_b)
        
    def psi_2_1(self, x):
        dpsi2_dx = jax.grad(self.psi_2_0)(x)
        return dpsi2_dx @ self.f(x) + self.beta_2_0 * self.psi_2_0(x)
    
    def psi_2_2_bar(self, x, u, nu, w_d):
        
        psi_2_1_dx = jax.grad(self.psi_2_1)(x)

        term1 = psi_2_1_dx @ self.f(x)
        term2 = psi_2_1_dx @ (self.g(x) @ u)
        term3 = psi_2_1_dx @ (self.D @ (self.tau * x[3:6] + nu))
        term4 = -self.smooth_norm(psi_2_1_dx @ self.D) * jnp.sqrt(w_d + 1e-12)

        term5 = self.beta_2_1 * self.psi_2_1(x)
        
        return term1 + term2 + term3 + term4 + term5

    def psi_3_0(self, x):
        # Ensure 1D slices
        r = x[0:3]
        v = x[3:6]
        r_f = self.x_f[0:3]
        v_f = self.x_f[3:6]

        # Use sum of squares directly for better stability and performance
        sqrd_pos = jnp.sum(jnp.square(r - r_f))
        sqrd_vel = jnp.sum(jnp.square(v - v_f))

        # JAX-compliant transition function
        eta = self.eta_transition(sqrd_pos)
        
        # Calculate the squared velocity limit transition
        v_limit_sq = eta * self.v_max**2 + (1 - eta) * self.v_l**2
    
        return v_limit_sq - sqrd_vel
    
    def psi_3_1_bar(self, x, u, nu, w_d):
        
        psi_3_0_dx = jax.grad(self.psi_3_0)(x)

        term1 = psi_3_0_dx @ self.f(x)
        term2 = psi_3_0_dx @ (self.g(x) @ u)
        term3 = psi_3_0_dx @ (self.D @ (self.tau * x[3:6] + nu))
        term4 = -self.smooth_norm(psi_3_0_dx @ self.D) * jnp.sqrt(w_d + 1e-12)

        term5 = self.beta_3_0 * self.psi_3_0(x)
        
        return term1 + term2 + term3 + term4 + term5

    def psi_4_0(self, x):
        # Extract states
        r = x[0:3]
        omega = x[15:18]
        r_f = self.x_f[0:3]

        sqrd_pos = jnp.sum(jnp.square(r - r_f))
        sqrd_ang_vel = jnp.sum(jnp.square(omega))

        eta = self.eta_transition(sqrd_pos)

        omega_limit_sq = eta * self.omega_max**2 + (1 - eta) * self.omega_l**2

        return omega_limit_sq - sqrd_ang_vel
    
    def psi_4_1(self, x, u):
        
        psi_4_0_dx = jax.grad(self.psi_4_0)(x)

        term1 = psi_4_0_dx @ self.f(x)
        term2 = psi_4_0_dx @ (self.g(x) @ u)

        term5 = self.beta_4_0 * self.psi_4_0(x)
        
        return term1 + term2 + term5
    
    def smooth_norm(self, xi, eps=1e-3):
        return jnp.sqrt(eps + jnp.dot(xi, xi))

    def eta_transition(self, a, k=1.0e-5):
        """
        JAX-compliant implementation of the transition function eta(a).

        Parameters:
        a : float (The input, typically ||r - rf||^2)
        k : float (The inverse of the transition radius squared)
            
        Returns:
        float : The value of eta(a).

        Example tuning parameters:
        Start Braking Distance |    k Value
        200 m                       2.5*10^-5
        100 m                       1.0*10^-4
        50 m                        4.0*10^-4
        10 m                        0.01
        """
        # 1. Handle the boundaries using jnp.clip 
        ka = jnp.clip(k * a, 0.0, 1.0)
        
        # 2. Calculate the transition
        # When ka is 0, the result is 0. 
        # When ka is 1, (1 - sin(2pi)/2pi) = 1.
        result = ka - jnp.sin(2 * jnp.pi * ka) / (2 * jnp.pi)

        return result

    def phi(self, u_tilde):
        return self.A_tilde @ u_tilde + self.b_tilde
    
    def softmin_h(self, x, u, nu, w_d):

        b_state = jnp.array([
            self.psi_1_2_bar(x, u, nu, w_d),
            self.psi_2_2_bar(x, u, nu, w_d),
            self.psi_3_1_bar(x, u, nu, w_d),
            self.psi_4_1(x, u)
        ])

        b_state = jnp.multiply(self.k_state, b_state)

        b_input = jnp.multiply(self.k_input, self.phi(u))

        b = jnp.append(b_state, b_input)

        h = -(1.0 / self.rho) * logsumexp(-self.rho * b)

        return h
    
    def b(self, x, u, nu, w_d, zeta, kappa):

        dh_dx  = jax.grad(self.softmin_h, argnums=0)(x, u, nu, w_d)
        dh_du  = jax.grad(self.softmin_h, argnums=1)(x, u, nu, w_d)
        dh_dnu = jax.grad(self.softmin_h, argnums=2)(x, u, nu, w_d)
        dh_dw_d = jax.grad(self.softmin_h, argnums=3)(x, u, nu, w_d)

        L_f_h = dh_dx @ self.dynamics(x, u, w=jnp.zeros(3)) + dh_du @ (self.A_c @ u) + dh_dnu @ self.nu_dynamics(x, u, nu) + dh_dw_d * self.dist_error_bound_dyn(w_d, x)

        L_g_h = dh_du @ self.B_c

        L_d_h = dh_dx @ self.D

        return  L_f_h + L_g_h @ zeta + L_d_h @ self.w_estimate(x, nu) - self.smooth_norm(L_d_h) * jnp.sqrt(w_d) +  + self.alpha * self.softmin_h(x, u, nu, w_d) + kappa * self.softmin_h(x, u, nu, w_d)

    # def safe_control(self, x, u, nu, w_d, zeta_d):
    
    #     varphi = self.b(x, u, nu, w_d, zeta_d, jnp.array(0.0))

    #     h = self.softmin_h(x, u, nu, w_d)

    #     dh_du = jax.grad(self.softmin_h, argnums=1)(x, u, nu, w_d)

    #     L_g_h = dh_du @ self.B_c

    #     norm_sq = jnp.dot(L_g_h, L_g_h)

    #     denom = norm_sq + self.gamma_inv * h**2

    #     var = (-varphi) / denom

    #     lam = jnp.where(varphi < 0, var, 0.0)

    #     kappa = (h * lam) / self.gamma

    #     zeta = zeta_d + lam * (dh_du @ self.A_c)

    #     return zeta, kappa

    def compute_safe_control(self, x, u, nu, w_d, x_r, u_r, u_r_dot):
        zeta, u_d, lam = _compute_safe_control_jit(
            # state
            jnp.asarray(x), jnp.asarray(u), jnp.asarray(nu), jnp.asarray(w_d),
            jnp.asarray(x_r), jnp.asarray(u_r), jnp.asarray(u_r_dot),
            # controller dynamics
            self.A_c, self.B_c, self.B_c_inv,
            # tracking gains
            self.K_r, self.K_v, self.K_R, self.K_omega,
            # disturbance observer
            self.tau, self.nu_0, self.c_1, self.w_bar_0,
            # spacecraft
            self.m, self.J, self.J_inv, self.D,
            # celestial body
            self.mu, self.omega_A, self.omega_A_hat,
            # surrogate filter
            self.sigma,
            # safety filter
            self.alpha, self.rho, self.gamma, self.gamma_inv,
            self.k_state, self.k_input,
            self.beta_1_0, self.beta_1_1,
            self.beta_2_0, self.beta_2_1,
            self.beta_3_0, self.beta_4_0,
            # constraints
            self.x_f, self.e_s, self.theta_gs, self.beta_gs, self.theta_b,
            self.v_max, self.v_l, self.omega_max, self.omega_l,
            self.A_tilde, self.b_tilde,
        )
        return np.asarray(zeta), np.asarray(u_d), np.asarray(lam)