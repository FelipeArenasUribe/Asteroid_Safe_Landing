import numpy as np
import control as ct
import control.optimal as obc
from scipy.optimize import LinearConstraint, NonlinearConstraint
from src import math_utils


class TrajectoryGenerator:
    def __init__(self, spacecraft, T_f, N):
        """
        Generate reference trajectories for spacecraft pose dynamics.

        Inputs:
        - spacecraft: object with system parameters
        - Tf: final time horizon
        - N: Sampling grid
        """
        self.spacecraft = spacecraft
        self.CB = self.spacecraft.known_CB
        self.x_0 = spacecraft.x_0.flatten()
        self.x_f = spacecraft.x_f.flatten()
        self.N_x = self.x_0.shape[0]

        self.T_f = T_f
        self.N = N

    def Constant_Traj(self):
        """
        Generate a constant reference trajectory.

        Outputs:
            X_traj     : (N_x, N)   state trajectory
            U_traj     : (n_u_tilde, N) control trajectory
        """

        # -----------------------------
        # Time vector
        t_vec = np.linspace(0.0, self.T_f, self.N)  # (N,)

        f_x_f = self.spacecraft.f(self.x_f) # ARC_Grav expects 1D arrays
        g_x_f = self.spacecraft.g(self.x_f) # ARC_Grav expects 1D arrays


        # Desired force
        u_F = np.linalg.inv(g_x_f[3:6,0:3]) @ (- f_x_f[3:6])

        # Desired torque
        # u_T = np.linalg.inv(g_x_f[15:18,3:6]) @ (- f_x_f[15:18]) # this is virtually zero
        u_T = np.zeros(3)

        u_tilde_d = np.append(u_F, u_T)

        U_traj = u_tilde_d[:, np.newaxis] * np.ones((1, self.N)) # (n_u, N)

        X_traj = self.x_f[:, np.newaxis] * np.ones((1, self.N)) # (18, N)

        return X_traj, U_traj
    
    def Opt_Traj(self):
        """
        Generate a constant reference trajectory.

        Outputs:
            X_traj     : (N_x, N)   state trajectory
            U_traj     : (n_u_tilde, N) control trajectory
        """

        X_traj, U_traj = self.Constant_Traj()

        Att_traj = X_traj[6:,:]
        U_T_traj = U_traj[3:,:]

        pos_traj_gen = PositionTrajectoryGenerator(self.spacecraft, self.T_f)

        X_pos_traj, U_pos_traj = pos_traj_gen.generate_trajectory(N=self.N)

        X_pos_traj = X_pos_traj.T
        U_pos_traj = U_pos_traj.T

        u_F_traj = np.zeros_like(U_pos_traj)
        for i in range(self.N):
            u_F_traj[:,i] = Att_traj[:9,i].reshape(3,3) @ U_pos_traj[:,i]

        X_traj = np.vstack((X_pos_traj,Att_traj))

        U_traj = np.vstack((u_F_traj,U_T_traj))

        return X_traj, U_traj

class PositionTrajectoryGenerator:
    def __init__(self, spacecraft, Tf):
        """
        Generate reference trajectories for spacecraft translational dynamics.

        Inputs:
        - system: object with .dynamics(x,u) and .spacecraft.m
        - controller: controller object (not used directly here)
        - x_0: initial state (6D: pos+vel)
        - x_f: final state (6D: pos+vel)
        - Tf: final time horizon
        """

        self.spacecraft = spacecraft
        self.x_0 = spacecraft.x_0[0:6].flatten()
        self.x_f = spacecraft.x_f[0:6].flatten()
        self.Tf = Tf
        self.n_x = 6
        self.n_u = 3  # 3D force vector

        mass = sum(body.mass for body in spacecraft.known_CB.sub_bodies)
        G = 6.67430e-11
        self.mu = np.asarray(G * mass)

        self.G = np.zeros((6, 3))
        self.G[3:6, 0:3] = (1.0 / self.spacecraft.m) * np.eye(3)

        # Input constraint matrix (project 6-DOF polytope down to forces only)
        M = np.array([
            # Fx     Fy     Fz     Tx      Ty    Tz
            [1.0,   0.0,   0.0,   0.0,    0.0,  0.0],  # Fx
            [0.0,   1.0,   0.0,   0.0,    0.0,  0.0],  # Fy
            [0.0,   0.0,   1.0,   0.0,    0.0,  0.0],  # Fz
        ])

        self.A_F, self.b_F = self.spacecraft.polytope_projection(
            spacecraft.A_tilde, spacecraft.b_tilde, M
        )
      
        # Scaling factors (to improve solver convergence)
        self.x_scale = np.array([1e3, 1e3, 1e3, 10, 10, 10])  # pos ~ km, vel ~ 10 m/s
        self.u_scale = np.array([1.0, 1.0, 1.0])              # control normalized by max accel

        # Wrap spacecraft dynamics in a NonlinearIOSystem
        self.sc_sys = ct.NonlinearIOSystem(
            self.dyn, self.y,
            states=self.n_x,
            name='spacecraft',
            inputs=('ux', 'uy', 'uz'),
            outputs=('rx', 'ry', 'rz', 'vx', 'vy', 'vz')
        )

    def f(self, x):
        r = x[0:3]
        v = x[3:6]

        r_dot = v

        v_dot = self.spacecraft.known_CB.grav_accel(r) - 2 * math_utils.skew_symmetric(self.spacecraft.CB_omega) @ v - math_utils.skew_symmetric(self.spacecraft.CB_omega) @ math_utils.skew_symmetric(self.spacecraft.CB_omega) @ r

        xdot = np.zeros(6)
        xdot[0:3] = r_dot
        xdot[3:6] = v_dot
        return xdot
    
    def dynamics(self, x, u):
        return self.f(x) + self.G @ u

    # --- Dynamics and output functions ---
    def dyn(self, t, x, u, params):
        # Rescale inputs for dynamics
        u_real = u * self.u_scale
        x_real = x * self.x_scale
        dx_real = self.dynamics(x_real, u_real)
        # Return scaled derivative
        return dx_real / self.x_scale

    def y(self, t, x, u, params):
        return x

    # --- Straight line trajectory (initial guess) ---
    def straight_line_trajectory(self, N, dt):
        # Interpolate positions
        r_d = np.zeros((N, 3))
        for i in range(3):
            r_d[:, i] = np.linspace(self.x_0[i], self.x_f[i], N)

        # Velocity by finite differences
        v_d = np.zeros_like(r_d)
        for k in range(1, N-1):
            v_d[k] = (r_d[k+1] - r_d[k-1]) / (2*dt)
        v_d[0] = (r_d[1] - r_d[0]) / dt
        v_d[-1] = (r_d[-1] - r_d[-2]) / dt

        # Acceleration by finite differences of velocity
        a_d = np.zeros_like(r_d)
        for k in range(1, N-1):
            a_d[k] = (v_d[k+1] - v_d[k-1]) / (2*dt)
        a_d[0] = (v_d[1] - v_d[0]) / dt
        a_d[-1] = (v_d[-1] - v_d[-2]) / dt

        # Control guess = m * acceleration
        u_d = self.spacecraft.m * a_d

        # Combine into state trajectory
        x_d = np.hstack((r_d, v_d))

        # Apply scaling
        x_d_scaled = x_d / self.x_scale
        u_d_scaled = u_d / self.u_scale

        return x_d_scaled, u_d_scaled

    def make_glideslope_pyramid_constraint(self):
        """
        Linear approximation using R_f to map from landing frame to reference frame.
        """
        # Define normals in the Landing Frame (F_l) where z is the axis
        c = np.cos(self.spacecraft.theta_gs)
        s = np.sin(self.spacecraft.theta_gs)
        
        n1_l = np.array([ c,  0.0, -s])
        n2_l = np.array([ 0.0,  c, -s])
        n3_l = np.array([-c,  0.0, -s])
        n4_l = np.array([ 0.0, -c, -s])

        # Rotate normals into Reference Frame (F_A)
        R_l = self.spacecraft.x_f[6:15].reshape(3, 3)
        
        n1 = R_l @ n1_l
        n2 = R_l @ n2_l
        n3 = R_l @ n3_l
        n4 = R_l @ n4_l

        H_pos = np.vstack([n1, n2, n3, n4])  # (4,3)
        
        # Pad with zeros for velocity components
        H = np.hstack([H_pos, np.zeros((4, 3))])  # now shape (4,6) to match state vector

        # Upper bounds (h vector = 0)
        h = np.zeros(4)

        # Scale position entries according to x_scale
        H_scaled = H.copy()
        H_scaled[:, :3] /= self.x_scale[:3]  # only scale positions, velocities remain zero

        return [obc.state_poly_constraint(self.sc_sys, H_scaled, h)]

    def make_velocity_constraint(self):
        """
        Creates a linear state constraint to ensure |v_i| <= v_max for i = x, y, z.

        Parameters
        ----------
        sc_sys : NonlinearIOSystem
            The system object containing the state scaling info.
        v_max : array-like of shape (3,)
            Maximum allowed velocities along x, y, z (in real units).

        Returns
        -------
        velocity_constraint : list
            A list containing one state_poly_constraint object suitable for OCP solver.
        """

        v_max = np.asarray(self.spacecraft.v_max, dtype=float)

        # Scale velocities to the normalized state used in OCP
        v_max_scaled = v_max / self.x_scale[3:6]

        # Construct A x <= b
        A = np.zeros((6, self.n_x))
        b = np.zeros(6)

        # v_x
        A[0, 3] = 1.0
        b[0] = v_max_scaled[0]
        A[1, 3] = -1.0
        b[1] = v_max_scaled[0]

        # v_y
        A[2, 4] = 1.0
        b[2] = v_max_scaled[1]
        A[3, 4] = -1.0
        b[3] = v_max_scaled[1]

        # v_z
        A[4, 5] = 1.0
        b[4] = v_max_scaled[2]
        A[5, 5] = -1.0
        b[5] = v_max_scaled[2]

        return [obc.state_poly_constraint(self.sc_sys, A, b)]

    # --- Generate trajectory with OCP ---
    def generate_trajectory(self, N=100, show_progress=False):

        timepts = np.linspace(0, self.Tf, N)

        # --- Create Initial guess --- #
        X_guess, U_guess = self.straight_line_trajectory(N, self.Tf / (N - 1))

        # --- Define Cost Function --- #
        # Cost weights (scaled)
        Q = np.diag([10, 10, 10, 1, 1, 1])
        R = np.diag([0.01, 0.01, 0.01])

        # Q = np.diag([1, 1, 1, 1, 1, 1])
        # R = np.diag([1, 1, 1])

        # Quadratic costs
        terminal_cost = obc.quadratic_cost(self.sc_sys, Q, None, x0=self.x_f / self.x_scale)
        running_cost = obc.quadratic_cost(self.sc_sys, Q, R, x0=self.x_f / self.x_scale, u0=U_guess[-1])

        # --- Define Constraints --- #

        # State Constraints

        # Glideslope Position constraint
        position_constraint = [self.make_glideslope_pyramid_constraint()]

        # Maximum Velocity constriant
        velocity_constraint = self.make_velocity_constraint()

        # Input Constraints
        input_constraint = [obc.input_poly_constraint(self.sc_sys, self.A_F, self.b_F)]

        # Then combine with other constraints
        state_input_constraints = position_constraint + velocity_constraint + input_constraint

        # Terminal state constraint
        terminal_state = [obc.state_range_constraint(self.sc_sys, self.x_f / self.x_scale, self.x_f / self.x_scale)]

        # Solve OCP
        print("Generating Position Trajectory...")

        # Solve OCP (no unsupported kwargs)
        result = obc.solve_ocp(
            self.sc_sys,
            timepts,
            self.x_0 / self.x_scale,
            running_cost,
            terminal_cost=terminal_cost,
            trajectory_constraints = state_input_constraints,
            terminal_constraints=terminal_state,
            initial_guess=(X_guess.T, U_guess.T),
            log=False
        )

        if not result.success:
            raise RuntimeError("Trajectory optimization failed")

        # Extract results and rescale
        X_traj = result.states.T * self.x_scale
        U_traj = result.inputs.T * self.u_scale

        return X_traj, U_traj