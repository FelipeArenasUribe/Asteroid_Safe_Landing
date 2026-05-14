import math
import numpy as np
import cdd
import cvxpy as cp

from src import utils

from src import math_utils

from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline, PchipInterpolator

class Simulation:
    def __init__(self, spacecraft, tracking_controller, X_ref, U_ref, dt_control, t_final):
        self.spacecraft = spacecraft
        self.tracking_controller = tracking_controller
        self.dt_control = dt_control
        self.t_final = t_final
        
        ref_n_steps = X_ref.shape[1]
        self.n_x = X_ref.shape[0]
        self.n_u = U_ref.shape[0]
        self.n_steps = int(np.ceil(self.t_final / self.dt_control))

        t_original = np.linspace(0.0, self.t_final, ref_n_steps)
        t_target = np.linspace(0.0, self.t_final, self.n_steps)

        if ref_n_steps == self.n_steps:
            self.X_ref = X_ref
            self.U_ref = U_ref
        else:
            # Option A: CubicSpline for maximum smoothness (C2 continuity)
            # bc_type='natural' ensures second derivatives are zero at endpoints
            cs_x = CubicSpline(t_original, X_ref, axis=1, bc_type='natural')
            self.X_ref = cs_x(t_target)

            # Option B: PCHIP for Control Inputs (prevents U_ref overshooting limits)
            pch_u = PchipInterpolator(t_original, U_ref, axis=1)
            self.U_ref = pch_u(t_target)

    def augmented_ode(self, t, z, u, w):
        # Unpack augmented state
        x = z[0:self.n_x]          # physical state
        nu = z[self.n_x:self.n_x+3]        # observer state
        w_bar = z[-1]       # disturbance bound

        xdot = self.spacecraft.dynamics(x, u, w)

        nu_dot = self.tracking_controller.nu_dynamics(x, u, nu)

        w_bar_dot = self.tracking_controller.dist_error_bound_dyn(w_bar, x)

        dzdt = np.concatenate([
            xdot,
            nu_dot,
            np.array([w_bar_dot])
        ])

        return dzdt

    def run(self, print_process=False):
        t_current = 0.0
        n_steps = self.n_steps

        n_x = self.n_x
        n_u = self.n_u
        n_nu = 3
        n_w = 3

        # -------------------------
        # Preallocate histories (time along columns)
        self.X_hist = np.zeros((n_x, n_steps + 1))
        self.nu_hist = np.zeros((n_nu, n_steps + 1))
        self.w_hist = np.zeros((n_w, n_steps + 1))
        self.w_hat_hist = np.zeros((n_w, n_steps + 1))
        self.w_bar_hist = np.zeros((1, n_steps + 1))
        self.u_tilde_hist = np.zeros((n_u, n_steps))
        self.u_tilde_d_hist = np.zeros((n_u, n_steps))
        self.t_hist = np.zeros(n_steps + 1)

        # -------------------------
        # Initial conditions
        self.X_hist[:, 0] = self.spacecraft.x_0.flatten()
        self.nu_hist[:, 0] = self.tracking_controller.nu_0.flatten()
        self.w_hat_hist[:, 0] = self.tracking_controller.w_estimate(self.X_hist[:, 0], self.nu_hist[:, 0])
        self.w_bar_hist[0, 0] = self.tracking_controller.w_bar_0
        self.t_hist[0] = t_current

        # Initial augmented state
        z = np.concatenate([
            self.X_hist[:, 0],
            self.nu_hist[:, 0],
            self.w_bar_hist[0, 0:1]
        ])

        # Printer
        printer = (lambda t: print(f"Current time: {t:.2f}s")) if print_process else (lambda t: None)

        # -------------------------
        # Main simulation loop
        for k in range(n_steps):
            printer(t_current)
            
            # -------------------------
            # Sample control (ZOH)
            x = z[0:n_x]
            nu = z[n_x:n_x + n_nu]
            w_bar = z[n_x + n_nu]

            w_hat = self.tracking_controller.w_estimate(x, nu)
            self.w_hat_hist[:, k] = w_hat

            w = self.spacecraft.grav_disturbance(x)
            self.w_hist[:, k] = w

            u_tilde_d = self.tracking_controller.desired_virtual_control(
                x, self.X_ref[:, k], self.U_ref[:, k], nu
            )

            self.u_tilde_d_hist[:, k] = u_tilde_d

            u_tilde = self.spacecraft.Saturate_Input(u_tilde_d)

            self.u_tilde_hist[:, k] = u_tilde

            # -------------------------
            # Integrate augmented system
            t_span = (t_current, t_current + self.dt_control)
            sol = solve_ivp(
                fun=lambda t, zz: self.augmented_ode(t, zz, u_tilde, w),
                t_span=t_span,
                y0=z,
                method="RK45",
                rtol=1e-5,
                atol=1e-8,
                max_step=0.05
            )

            # -------------------------
            # Update augmented state
            z = sol.y[:, -1]
            x_new = z[0:n_x]

            # Project rotation matrix back to SO(3)
            R_mat = x_new[6:15].reshape((3, 3))
            x_new[6:15] = math_utils.project_to_SO3(R_mat).ravel()

            self.X_hist[:, k + 1] = x_new
            self.nu_hist[:, k + 1] = z[n_x:n_x + n_nu]
            self.w_bar_hist[0, k + 1] = z[n_x + n_nu]

            # -------------------------
            # Collision check
            if self.spacecraft.CB.is_inside(x_new[0:3]):
                # print(f"Collision detected at t = {t_current:.2f}s, step {k}")
                break

            # -------------------------
            # Landing check (if not collided)
            if self.spacecraft.landing_condition(x_new):
                print(f"Landing successful at t = {t_current:.2f}s, step {k}")
                break

            # -------------------------
            # Advance time
            t_current += self.dt_control
            self.t_hist[k + 1] = t_current

            # Final landing check
            if self.spacecraft.landing_condition(self.X_hist[:, -1]):
                print('Landing achieved.')

        # Keep only strictly increasing time entries
        valid_idx = np.where(np.diff(self.t_hist) > 0)[0]
        last_valid = valid_idx[-1] + 1 if len(valid_idx) > 0 else 0

        self.t_hist = self.t_hist[:last_valid+1]
        self.X_hist = self.X_hist[:, :last_valid+1]
        self.nu_hist = self.nu_hist[:, :last_valid+1]
        self.w_hist = self.w_hist[:, :last_valid+1]
        self.w_hat_hist = self.w_hat_hist[:, :last_valid+1]
        self.w_bar_hist = self.w_bar_hist[:, :last_valid+1]
        self.u_tilde_hist = self.u_tilde_hist[:, :last_valid]
        self.u_tilde_d_hist = self.u_tilde_d_hist[:, :last_valid]

        return self.t_hist, self.X_hist, self.u_tilde_hist, self.nu_hist, self.w_hat_hist, self.w_bar_hist, self.w_hist, self.u_tilde_d_hist
     
class Simulation_Safe:
    def __init__(self, spacecraft, safety_controller, controller_dynamics, X_ref, U_ref, dt_control, t_final):
        self.spacecraft = spacecraft
        self.safety_controller = safety_controller
        self.controller_dynamics = controller_dynamics
        self.dt_control = dt_control
        self.t_final = t_final

        # Calculate target simulation steps
        self.n_steps = int(np.ceil(self.t_final / self.dt_control))
        t_target = np.linspace(0.0, self.t_final, self.n_steps)

        # Identify input dimensions (assuming input is [Dim x Steps])
        n_ref_steps = X_ref.shape[1]
        self.n_x = X_ref.shape[0]
        self.n_u = U_ref.shape[0]

        # Time grid for the original reference data
        t_ref = np.linspace(0.0, self.t_final, n_ref_steps)

        if n_ref_steps == self.n_steps:
            self.X_ref = X_ref
            self.U_ref = U_ref
            # We still need a spline if we want a smooth U_ref_dot
            interp_u = CubicSpline(t_ref, U_ref, axis=1, bc_type='intermediate')
        else:
            # 1. Create Spline Objects (C2 continuity: smooth position and velocity)
            # Using CubicSpline with 'clamped' or 'not-a-knot' is great for dynamics
            interp_x = CubicSpline(t_ref, X_ref, axis=1)
            interp_u = CubicSpline(t_ref, U_ref, axis=1)

            # 2. Resample onto the control time grid
            self.X_ref = interp_x(t_target)
            self.U_ref = interp_u(t_target)

        # 3. ANALYTIC DERIVATIVE 
        self.U_ref_dot = interp_u.derivative()(t_target)
    
    def augmented_ode(self, t, z, zeta, w):
        # Unpack augmented state
        x = z[0:self.n_x]          # physical state
        nu = z[self.n_x:self.n_x+3]        # observer state
        w_bar = z[self.n_x+3]       # disturbance bound
        u = z[self.n_x+4:]          # Actuator input


        xdot = self.spacecraft.dynamics(x, u, w)

        nu_dot = self.controller_dynamics.nu_dynamics(x, u, nu)

        w_bar_dot = self.controller_dynamics.dist_error_bound_dyn(w_bar, x)

        u_dot = self.controller_dynamics.control_dynamics(u, zeta)

        dzdt = np.concatenate([
            xdot,
            nu_dot,
            np.array([w_bar_dot]),
            u_dot
        ])

        return dzdt

    def run(self, print_process=False):
        t_current = 0.0
        n_steps = self.n_steps

        n_x = self.n_x
        n_u = 6
        n_nu = 3
        n_w = 3

        # -------------------------
        # Preallocate histories (time along columns)
        self.X_hist = np.zeros((n_x, n_steps + 1))
        self.nu_hist = np.zeros((n_nu, n_steps + 1))
        self.w_hist = np.zeros((n_w, n_steps + 1))
        self.w_hat_hist = np.zeros((n_w, n_steps + 1))
        self.w_bar_hist = np.zeros((1, n_steps + 1))
        self.u_hist = np.zeros((n_u, n_steps + 1))
        self.u_d_hist = np.zeros((n_u, n_steps))
        self.zeta_hist = np.zeros((n_u, n_steps))
        self.lambda_hist = np.zeros((n_steps))
        self.t_hist = np.zeros(n_steps + 1)

        # -------------------------
        # Initial conditions
        self.X_hist[:, 0] = self.spacecraft.x_0.flatten()
        self.nu_hist[:, 0] = self.controller_dynamics.nu_0.flatten()
        self.w_hat_hist[:, 0] = self.controller_dynamics.w_estimate(
            self.X_hist[:, 0], self.nu_hist[:, 0]
        )
        self.w_bar_hist[0, 0] = self.controller_dynamics.w_bar_0
        self.u_hist[:, 0] = self.U_ref[:, 0]
        self.t_hist[0] = t_current

        # Initial augmented state
        z = np.concatenate([
            self.X_hist[:, 0],
            self.nu_hist[:, 0],
            self.w_bar_hist[0, 0:1],
            self.u_hist[:, 0]
        ])

        # Printer
        printer = (lambda t: print(f"Current time: {t:.2f}s")) if print_process else (lambda t: None)

        # -------------------------
        # Main simulation loop
        for k in range(n_steps):
            printer(t_current)

            # -------------------------
            # Sample control (ZOH)
            x = z[0:n_x]
            nu = z[n_x:n_x + n_nu]
            w_bar = z[n_x + n_nu]
            u = z[n_x + n_nu + 1:]

            w_hat = self.controller_dynamics.w_estimate(x, nu)
            self.w_hat_hist[:, k] = w_hat

            w = self.spacecraft.grav_disturbance(x)
            self.w_hist[:, k] = w

            zeta, u_d, lam = self.safety_controller.compute_safe_control(
                x, u, nu, w_bar, self.X_ref[:, k], self.U_ref[:, k], self.U_ref_dot[:, k]
            )
            
            self.u_d_hist[:, k] = u_d
            self.zeta_hist[:, k] = zeta
            self.lambda_hist[k] = lam

            # -------------------------
            # Integrate augmented system
            t_span = (t_current, t_current + self.dt_control)
            sol = solve_ivp(
                fun=lambda t, zz: self.augmented_ode(t, zz, zeta, w),
                t_span=t_span,
                y0=z,
                method="RK45",
                rtol=1e-5,
                atol=1e-8,
                max_step=0.05
            )

            # -------------------------
            # Update augmented state
            z = sol.y[:, -1]
            x_new = z[0:n_x]

            self.X_hist[:, k + 1] = x_new
            self.nu_hist[:, k + 1] = z[n_x:n_x + n_nu]
            self.w_bar_hist[0, k + 1] = z[n_x + n_nu]

            u = self.spacecraft.Saturate_Input(z[n_x + n_nu + 1:])
            
            self.u_hist[:, k + 1] = u

            # -------------------------
            # Collision check
            if self.spacecraft.CB.is_inside(x_new[0:3]):
                # print(f"Collision detected at t = {t_current:.2f}s, step {k}")
                print(np.linalg.norm(x_new[0:3] - self.spacecraft.x_f[0:3].reshape(-1)))
                # Speed condition
                print(np.linalg.norm(x_new[3:6] - self.spacecraft.x_f[3:6].reshape(-1)))
                # Attitude condition
                print(math_utils.d_SO3(x_new[6:15].reshape(3, 3), self.spacecraft.x_f[6:15].reshape(3, 3)))
                # Angular speed condition
                print(np.linalg.norm(x_new[15:18] - self.spacecraft.x_f[15:18].reshape(-1)))
                break

            # -------------------------
            # Landing check
            if self.spacecraft.landing_condition(x_new):
                print(f"Landing successful at t = {t_current:.2f}s, step {k}")
                # Position condition
                print(np.linalg.norm(x_new[0:3] - self.spacecraft.x_f[0:3].reshape(-1)))
                # Speed condition
                print(np.linalg.norm(x_new[3:6] - self.spacecraft.x_f[3:6].reshape(-1)))
                # Attitude condition
                print(math_utils.d_SO3(x_new[6:15].reshape(3, 3), self.spacecraft.x_f[6:15].reshape(3, 3)))
                # Angular speed condition
                print(np.linalg.norm(x_new[15:18] - self.spacecraft.x_f[15:18].reshape(-1)))
                break

            # -------------------------
            # Advance time
            t_current += self.dt_control
            self.t_hist[k + 1] = t_current

        # -------------------------
        # Keep only valid, strictly increasing time entries
        valid_idx = np.where(np.diff(self.t_hist) > 0)[0]
        last_valid = valid_idx[-1] + 1 if len(valid_idx) > 0 else 0

        self.t_hist = self.t_hist[:last_valid + 1]
        self.X_hist = self.X_hist[:, :last_valid + 1]
        self.nu_hist = self.nu_hist[:, :last_valid + 1]
        self.w_hist = self.w_hist[:, :last_valid + 1]
        self.w_hat_hist = self.w_hat_hist[:, :last_valid + 1]
        self.w_bar_hist = self.w_bar_hist[:, :last_valid + 1]
        self.u_hist = self.u_hist[:, :last_valid + 1]
        self.u_d_hist = self.u_d_hist[:, :last_valid]
        self.zeta_hist = self.zeta_hist[:, :last_valid]
        self.lambda_hist = self.lambda_hist[:last_valid + 1]

        # Final landing check
        if self.spacecraft.landing_condition(self.X_hist[:, -1]):
            print('Landing achieved.')

        return (
            self.t_hist,
            self.X_hist,
            self.u_hist,
            self.nu_hist,
            self.w_hat_hist,
            self.w_bar_hist,
            self.w_hist,
            self.u_d_hist,
            self.zeta_hist,
            self.lambda_hist
        )

class spacecraft:
    def __init__(self, x_0, r_f, R_f, mass, inertia, M, A, b, theta_gs, theta_b, e_s, CB, mass_uncertainty = 0.0, omega_uncertainty = 0.0, r_l = 1.0, v_l = 0.1, theta_l= 0.2, omega_l = 0.2):
        '''
        Spacecraft object setup

        Inputs:
        x_0: Initial conditions
        r_f: Desired final position
        R_f Desired final attitude
        mass: spacecraft mass [kg]
        '''
        # Spacecraft properties
        self.m = mass
        self.J = inertia
        self.J_inv = np.linalg.inv(inertia)

        # Input constraint and allocation matrices
        self.A = A
        self.b = b
        self.M = M
        self.A_tilde, self.b_tilde = self.polytope_projection(A, b, M)

        self.ell = M.shape[1] # Number of actuators
        self.num_virtual_control_inputs = 6
        self.n_u_tilde = np.size(self.b_tilde)


        # Gravitational disturbance setup
        self.CB = CB # Real celestial body
        self.known_CB = utils.generate_known_CB(CB, mass_uncertainty=mass_uncertainty) # Known gravitational model of celestial body
        self.CB_omega = CB.omega * (1.0 - omega_uncertainty) # known angular velocity of Celestial Body
        self.D = np.vstack((np.zeros((3, 3)), np.eye(3), np.zeros((9, 3)), np.zeros((3, 3))))
        
        # Initial and final conditions
        self.x_0 = x_0

        v_f = math_utils.skew_symmetric(self.CB_omega) @ r_f # desired final velocity
        omega_f = np.array([0.0, 0.0, 0.0]) # [rad/s]

        self.x_f = np.concatenate((
                r_f.flatten(),
                v_f.flatten(),
                R_f.flatten(),
                omega_f.flatten()
            )).reshape(-1, 1)  # Reshape to (18, 1)

        # Landing parameters
        self.r_l = r_l # landing radius
        self.v_l = v_l # Speed tolerance
        self.theta_l = theta_l # Attitude tolerance
        self.omega_l = omega_l # Angular speed tolerance

        # State constraint parameters
        if theta_gs >= math.pi/2:
            raise ValueError("Gldieslope angle is greater that pi/2.")
        if theta_b >= math.pi/2:
            raise ValueError("Boresight angle is greater that pi/2.")
        
        # Glideslope constraint
        self.theta_gs = theta_gs
        self.e_s = e_s / np.linalg.norm(e_s) # landing direction unit vector
        self.beta = 1 - (r_l/np.linalg.norm(r_f)*np.atan(self.theta_gs)) # glideslope relaxation constant

        # Field-of-view constraint
        self.theta_b = theta_b

        # Speed constraint
        self.v_max = 12.0 # maximum speed [m/s]

        # Angular speed constraint
        self.omega_max = 0.1 # maximum angular speed [rad/s]

        if self.check_state_constraints(self.x_0) == False:
            raise ValueError("Initial state does not satisfy state constraints.")

    def f(self, x):
        r = x[0:3]
        v = x[3:6]
        R = x[6:15].reshape(3, 3)
        omega = x[15:18]

        r_dot = v

        v_dot = self.known_CB.grav_accel(r) - 2 * math_utils.skew_symmetric(self.CB_omega) @ v - math_utils.skew_symmetric(self.CB_omega) @ math_utils.skew_symmetric(self.CB_omega) @ r

        R_dot = R @ math_utils.skew_symmetric(omega)

        omega_total = R.T @ self.CB_omega + omega
        omega_dot = -self.J_inv @ np.cross(omega_total, self.J @ omega_total)

        xdot = np.zeros(18)
        xdot[0:3] = r_dot
        xdot[3:6] = v_dot
        xdot[6:15] = R_dot.flatten()
        xdot[15:18] = omega_dot
        return xdot
    
    def g(self, x):
        R = x[6:15].reshape(3, 3)
        G = np.zeros((18, 6))

        # Force contribution: (1/m) * R
        G[3:6, 0:3] = (1.0 / self.m) * R

        # Torque contribution: J_inv
        G[15:18, 3:6] = self.J_inv
        return G
    
    def dynamics(self, x, u_tilde, w=np.zeros(3)):

        return self.f(x) + self.g(x) @ u_tilde + self.D @ w

    def grav_disturbance(self, x):
        r = x[0:3]
        return self.CB.Gravity(r) - self.known_CB.grav_accel(r)
    
    def allocation(self, u_tilde):
        """
        Solves the control allocation QP:
        min  u_hat^T * Q * u_hat + c^T * u_hat
        s.t. M * u_hat = u_tilde
            A * u_hat + b >= 0

        Parameters:
        u_tilde : (m,) array - The desired control input
        
        Returns:
        u_star  : (l,) array - The optimal actuator inputs
        """

        Q = np.eye(self.ell) # Minimize energy (L2 norm squared)
        c = np.zeros(self.ell)

        u_hat = cp.Variable(self.ell)

        # Define the objective function
        objective = cp.Minimize(cp.quad_form(u_hat, Q) + c.T @ u_hat)

        # Define the constraints
        constraints = [
            self.M @ u_hat == u_tilde,       # Equality constraint
            self.A @ u_hat + self.b >= 0     # Inequality constraint
        ]

        # Create and solve the problem
        prob = cp.Problem(objective, constraints)
        
        # OSQP is efficient for sparse QPs with inequality constraints
        prob.solve(solver=cp.OSQP)

        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise ValueError(f"Optimization failed with status: {prob.status}")

        return u_hat.value

    def Saturate_actuator_input(self, u):
        '''
        Saturates u into the polytope A * u_sat + b >= 0
        '''
        u_sat = np.clip(u, self.b[self.ell:], self.b[:self.ell])
        return u_sat
            
    def Saturate_Input(self, u_tilde):
        '''
        Saturates u_tilde into the polytope A_tilde * u_sat + b_tilde >= 0
        using CVXPY to minimize the L2-norm distance.
        '''
        # Define the dimensions
        # u_tilde is 6x1, A_tilde is mx6, b_tilde is mx1
        n = 6 
        A = self.A_tilde
        b = self.b_tilde

        # Define the Optimization Variables
        u_sat = cp.Variable(n)

        # Define the Objective
        # Minimize 0.5 * ||u_sat - u_tilde||^2
        objective = cp.Minimize(0.5 * cp.sum_squares(u_sat - u_tilde))

        # Define the Constraints
        # The polytope: A_tilde * u_sat + b_tilde >= 0
        constraints = [A @ u_sat + b >= 0]

        # 5. Solve the problem
        # OSQP is a fast, reliable solver for Quadratic Programs
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP)

        if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
            return u_sat.value
        else:
            # If the problem is infeasible, return a safe fallback (e.g., zeros)
            print(f"Saturation optimization failed with status: {prob.status}")
            return np.zeros(n)

    def landing_condition(self, x):
        # Position condition
        cond_1 = np.linalg.norm(x[0:3] - self.x_f[0:3].reshape(-1)) < self.r_l
        # Speed condition
        cond_2 = np.linalg.norm(x[3:6] - self.x_f[3:6].reshape(-1)) < self.v_l
        # Attitude condition
        cond_3 = math_utils.d_SO3(x[6:15].reshape(3, 3), self.x_f[6:15].reshape(3, 3)) < self.theta_l
        # Angular speed condition
        cond_4 = np.linalg.norm(x[15:18] - self.x_f[15:18].reshape(-1)) < self.omega_l

        return cond_1 and cond_2 and cond_3 and cond_4
    
    def glideslope_constraint(self, x):

        delta_r = x[0:3] - self.beta * self.x_f[0:3]

        return self.e_s.T @ (delta_r) - np.linalg.norm(delta_r) * np.cos(self.theta_gs)

    def field_of_view_constraint(self, x):
        r = x[0:3]
        R = x[6:15].reshape(3, 3)
        
        # e_b is usually the third column of the rotation matrix
        e_b = -R @ np.array([0, 0, 1]) 
        
        rel_pos = self.beta * self.x_f[0:3] - r

        return e_b.T @ rel_pos - np.linalg.norm(rel_pos) * np.cos(self.theta_b)

    def speed_constraint(self, x):

        r = x[0:3]
        v = x[3:6]
        r_f = self.x_f[0:3]
        v_f = self.x_f[3:6]

        sqrd_pos = np.linalg.norm(r - r_f)**2
        sqrd_vel = np.linalg.norm(v - v_f)**2

        eta = self.eta_transition(sqrd_pos)
        return eta * self.v_max**2 + (1 - eta) * self.v_l**2 - sqrd_vel
    
    def ang_speed_constraint(self, x):

        r = x[0:3]
        omega = x[15:18]
        r_f = self.x_f[0:3]

        sqrd_pos = np.linalg.norm(r - r_f)**2

        sqrd_ang_vel = np.linalg.norm(omega)**2 

        eta = self.eta_transition(sqrd_pos)

        return eta * self.omega_max**2 + (1 - eta) * self.omega_l**2 - sqrd_ang_vel
    
    def check_state_constraints(self,x):

        gs_constr = self.glideslope_constraint(x) >= 0
        fv_cons = self.field_of_view_constraint(x) >= 0
        speed_cons = self.speed_constraint(x) >= 0
        ang_speed_cons = self.ang_speed_constraint(x) >= 0

        return (gs_constr and fv_cons and speed_cons and ang_speed_cons)
    
    def check_input_constraint(self, u_tilde, tol=1e-9):
        """
        Checks if the control input u_tilde is within theadmissible control set.
        Defined by: A_tilde @ u_tilde + b_tilde >= 0
        """

        constraints = self.A_tilde @ u_tilde + self.b_tilde
        
        # Check if all elements in the resulting vector are non-negative
        # We use a small negative tolerance to account for floating-point precision
        return np.all(constraints >= -tol)

    def check_actuator_constraint(self, u, tol=1e-9):
        """
        Checks if the actuator input u is within the admissible actuator input set.
        Defined by: A @ u + b >= 0
        """

        constraints = self.A @ u + self.b
        
        # Check if all elements in the resulting vector are non-negative
        # We use a small negative tolerance to account for floating-point precision
        return np.all(constraints >= -tol)
    
    def polytope_projection(self, A: np.ndarray, b: np.ndarray, M: np.ndarray):
        """
        Project P = {u : Au + b >= 0} through u_tilde = M @ u.
        Returns (A_tilde, b_tilde) for P_tilde = {u_tilde : A_tilde u_tilde + b_tilde >= 0}.
        """
        V = self._vertex_enumeration(A, b)
        V_proj = self._project_vertices(V, M)
        A_tilde, b_tilde = self._facet_enumeration(V_proj)
        return A_tilde, b_tilde

    def _vertex_enumeration(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        H = np.hstack((b.reshape(-1, 1), A))
        mat = cdd.matrix_from_array(H, rep_type=cdd.RepType.INEQUALITY)
        poly = cdd.polyhedron_from_matrix(mat)
        V_rep = np.array(cdd.copy_generators(poly).array, dtype=float)
        if V_rep.ndim == 1:
            V_rep = V_rep[np.newaxis, :]
        vertex_mask = V_rep[:, 0] == 1
        return V_rep[vertex_mask, 1:]

    def _project_vertices(self, V: np.ndarray, M: np.ndarray) -> np.ndarray:
        return V @ M.T

    def _remove_redundant_rows(self, mat: cdd.Matrix) -> list[int]:
        essential = []

        H_rep = np.array(mat.array, dtype=float)
        for i in range(np.size(H_rep, 0)):
            if cdd.redundant(mat, i) is not None:
                essential.append(i)
        return essential

    def _facet_enumeration(self, V_proj: np.ndarray):
        k = V_proj.shape[0]
        V_cdd = np.hstack((np.ones((k, 1)), V_proj))
        mat = cdd.matrix_from_array(V_cdd, rep_type=cdd.RepType.GENERATOR)
        poly = cdd.polyhedron_from_matrix(mat)
        H_rep_mat = cdd.copy_inequalities(poly)
        essential = self._remove_redundant_rows(H_rep_mat)
        H_rep = np.array(H_rep_mat.array, dtype=float)[essential, :]
        return H_rep[:, 1:], H_rep[:, 0]

    def eta_transition(self, a, k = 1.0*10**-5):
        """
        Scalar implementation of the piecewise transition function eta(a).
        
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
        # Safety check for negative distance squared
        if a <= 0:
            return 0.0
        
        # Beyond the transition radius, speed limit is constant
        if a >= 1 / k:
            return 1.0
        
        # Transition phase
        ka = k * a
        return ka - math.sin(2 * math.pi * ka) / (2 * math.pi)

class Celestial_Body:
    """
    Class representing a celestial body

    Parameters
    ----------
    mass : float
        Mass of the body [kg].
    radius : float
        Mean radius of the body [m].
    synodic_rotation_period : float, optional
        Rotation period in hours. Default is 0 (non-rotating).
    position : ndarray, optional
        Initial position vector [m].
    orientation : ndarray, optional
        Orientation Euler angles [rad].
    """
    def __init__(self, main_body, synodic_rotation_period=0.0):

        self.main_body = main_body
        self.mass = sum(body.mass for body in self.main_body.sub_bodies) # total mass of celestial body
        G = 6.67430e-11
        self.mu = G * self.mass

        self.synodic_rotation_period = synodic_rotation_period
        if synodic_rotation_period > 0.0:
            self.angular_velocity = (2 * np.pi) / (synodic_rotation_period * 3600)
        else:
            self.angular_velocity = 0.0
        self.omega = np.array([0, 0, self.angular_velocity])  # angular velocity vector [rad/s]
        self.omega_dot = np.array([0, 0, 0])  # angular acceleration [rad/s^2]

    def Gravity(self, r):

        return self.main_body.grav_accel(r)   
    
    def is_inside(self, r):

        r = np.asarray(r).reshape(3)

        for sub_body in self.main_body.sub_bodies:


            # Transform inertial position into body frame
            r_rel = r - sub_body.position

            if hasattr(sub_body, "Rot_matrix"):
                R_IB = sub_body.Rot_matrix
                r_body = R_IB.T @ r_rel
            else:
                r_body = r_rel

            if sub_body.type == "Sphere":
                if np.linalg.norm(r_body) <= sub_body.radius:
                    return True

            elif sub_body.type == "Ellipsoid":
                a, b, c = sub_body.dim
                x, y, z = r_body

                ellipsoid_value = (x/a)**2 + (y/b)**2 + (z/c)**2

                if ellipsoid_value <= 1.0:
                    return True

        return False