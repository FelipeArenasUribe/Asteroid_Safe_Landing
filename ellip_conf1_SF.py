import numpy as np
import matplotlib.pyplot as plt

import ARC_Grav

from src import utils
from src import main
from src import Controllers
from src import Controllers_JIT
from src import plotting

# ------------------------------------------------------------
# Create irregular asteroid model

# Create sub_bodies
ellipsoids = [
    ARC_Grav.main.Ellipsoidal_Body(
        dimensions=np.array([400.0, 400.0, 1000.0]),
        density=1380.0,
        position=np.array([0.0, 0.0, 0.0])
    )
]

# Create asteroid as composition of multiple ellipsoids
asteroid_geom = ARC_Grav.main.main_body(sub_bodies=ellipsoids)

# Create rotatiting Asteroid object
asteroid = main.Celestial_Body(main_body=asteroid_geom, synodic_rotation_period=4.0)

# ------------------------------------------------------------
# Create Spacecraft and Mission Parameters

r_0, r_f, e_s, R_0, R_f = utils.compute_boundary_positions(asteroid, alt = 1500.0, lat_disp= np.array([-300.0, -200.0]), initial_rel_orientation = np.array([np.deg2rad(10.0), np.deg2rad(5.0), 0.0]))

v_0 = np.array([0.0, 4.0, 4.0]) # [m/s]
omega_0 = np.array([0.0, 0.0, 0.0]) # [rad/s]

# Full state vectors
x_0 = np.concatenate((
        r_0.flatten(),
        v_0.flatten(),
        R_0.flatten(),
        omega_0.flatten()
    )).reshape(-1, 1)  # Reshape to (18, 1)

F_max = 20.0 # N
T_max = 0.2 # N * m

# Define the upper bounds
u_max = np.array([T_max, T_max, T_max, T_max, F_max, F_max, F_max])
u_min = -u_max

A = np.vstack([np.eye(7), -np.eye(7)])
b = np.hstack([u_max, -u_min]) 

M = np.array([
    # Reaaction wheels         |  Thrusters
    [ 0.0,   0.0,   0.0,   0.0,    1.0,  0.0,  0.0], # Fx
    [ 0.0,   0.0,   0.0,   0.0,    0.0,  1.0,  0.0], # Fy
    [ 0.0,   0.0,   0.0,   0.0,    0.0,  0.0,  1.0], # Fz
    [ 1.0,  -0.33, -0.33, -0.33,   0.0,  0.0, 0.0], # Tx
    [-0.33,  1.0,  -0.33, -0.33,  0.0,  0.0,  0.0], # Ty
    [-0.33, -0.33,  1.0,  -0.33,   0.0, 0.0,  0.0]  # Tz
])

# Spacecraft setup
J = np.array([
    [ 180.0,   -5.0,   -2.0],
    [ -5.0,   180.0,   -8.0],
    [ -2.0,   -8.0,   150.0]
])

spacecraft = main.spacecraft(x_0=x_0, r_f=r_f, R_f = R_f, mass=600, inertia=J, M=M, A=A, b=b, theta_gs=np.deg2rad(60.0), theta_b=np.deg2rad(60.0), e_s=e_s, CB=asteroid, mass_uncertainty = 0.10, omega_uncertainty = 0.0, r_l = 2.0, v_l = 0.15, theta_l= 0.2, omega_l = 0.05)

# ------------------------------------------------------------
# Create Reference Trajectory
T_f = 500
N = 100

t_hist = np.linspace(0,T_f,N)

# Generate trajectory

data = np.load('trajectories/traj_conf1_ellip.npz')
X_traj = data['X']
U_traj = data['U']

# ------------------------------------------------------------
# Controller setup

# Position Feedback Linearizing
K_r=0.1 * np.eye(3) # Position gain
K_v=2 * np.sqrt(0.1) * np.eye(3) # Velocity gain

# Attitude Feedback Linearizing
# K_R = np.diag([797.78, 797.78, 664.82])
# K_omega = np.diag([720.0, 720.0, 600.0])

# K_R = np.diag([525.0, 525.0, 435.0])
# K_omega = np.diag([615.0, 615.0, 510.0])

K_R = np.diag([400.0, 400.0, 335.0])
K_omega = np.diag([536.0, 536.0, 448.0])

# K_R = np.diag([280.0, 280.0, 235.0])
# K_omega = np.diag([450.0, 450.0, 375.0])

# Disturbance observer and estimation error bound parameters
tau = 10.0
c_1 = 1e-3
w_bar_0 = 1e-10

# Control Dynamics
A_c = -3 * np.eye(6)
B_c = 3 * np.eye(6)
sigma = 3

betas = {
    "beta_1_0": 0.1,
    "beta_1_1": 0.1,
    "beta_2_0": 1.0,
    "beta_2_1": 1.0,
    "beta_3_0": 1.0,
    "beta_4_0": 10.0
}

alpha = 1.0

gamma = 1e20

k_state = np.array([1, 1, 1, 100])

k_input = np.ones(spacecraft.n_u_tilde)

rho = 40

tracking_controller = Controllers.TrackingController(spacecraft, K_r, K_v, tau, c_1, w_bar_0, K_R, K_omega)

controller_dynamics = Controllers.ControllerDynamics(spacecraft, tracking_controller, A_c, B_c)

safety_controller_JIT = Controllers_JIT.JIT_SafetyController(spacecraft, controller_dynamics, tracking_controller, sigma, betas, k_state, k_input, rho, alpha, gamma)

# ------------------------------------------------------------
# Simulation

sim = main.Simulation_Safe(spacecraft, safety_controller_JIT, controller_dynamics, X_traj, U_traj, 0.2, T_f)

t_hist, X_hist, u_tilde_hist, nu_hist, w_hat_hist, w_bar_hist, w_hist, u_tilde_d_hist, zeta_hist, lambda_hist = sim.run(print_process=True)

w_bound_hist = np.sqrt(w_bar_hist)

u_hist = utils.compute_u_hist(spacecraft, u_tilde_hist)

results = utils.compute_constraints_vmap(X_hist, u_tilde_hist, nu_hist, w_bar_hist, safety_controller_JIT)

# Level-0 barriers
psi_1_0_hist    = results["psi_1_0"]
psi_2_0_hist    = results["psi_2_0"]
psi_3_0_hist    = results["psi_3_0"]
psi_4_0_hist    = results["psi_4_0"]

psi_list = [psi_1_0_hist, psi_2_0_hist, psi_3_0_hist, psi_4_0_hist]

# Level-1 barriers
psi_1_1_hist    = results["psi_1_1"]
psi_2_1_hist    = results["psi_2_1"]

# Terminal barriers (what the safety filter enforces)
psi_1_2_bar_hist = results["psi_1_2_bar"]
psi_2_2_bar_hist = results["psi_2_2_bar"]
psi_3_1_bar_hist = results["psi_3_1_bar"]
psi_4_1_hist     = results["psi_4_1"]

# Input constraints
phi_hist        = results["phi"]

# Scaled components (for diagnosing active constraint)
b_state_hist    = results["b_state"]   # shape (T, 4)
b_input_hist    = results["b_input"]   # shape (T, n_input_constraints)

# Aggregate softmin CBF — must stay ≥ 0
h_soft_hist     = results["h_soft"]
h_min_hist     = results["h_min"]

# --- Plots ---

plotting.plot_trajectory_attitude_with_body(X_hist, asteroid.main_body, index=-1, n_points=100, az=-75, el=15, location='upper right', save_fig=True, n_attitude=12, frame_scale=170)

plotting.plot_w_estimates(w_hat_hist, w_bound_hist, w_hist, t_hist, save_fig=False)

plotting.plot_position_attitude_error(t_hist, X_hist, spacecraft.x_f, save_fig=True)

plotting.plot_state_constraints(psi_list, t_hist, save_fig=False)

plotting.plot_virtual_controls(t_hist, u_tilde_hist, u_tilde_d_hist, labels=None, save_fig=False)

plotting.plot_actuator_inputs(t_hist, u_hist, spacecraft, save_fig=False)

plotting.plot_h_softmin_lambda(t_hist, h_soft_hist, h_min_hist, lambda_hist, save_fig=False)

# --- Supplementary Plots ---

# plotting.plot_position_velocity_error(t_hist, X_hist, spacecraft.x_f, save_fig=False)

# plotting.plot_rotation_matrix(t_hist, X_hist.T[:, 6:15], spacecraft.x_f[6:15], save_fig=False)

# plotting.plot_attitude_error_function(t_hist, X_hist.T[:, 6:15], spacecraft.x_f[6:15], save_fig=False)

# plotting.plot_attitude_error_vector(t_hist, X_hist.T[:, 6:15], spacecraft.x_f[6:15], save_fig=False)

# plotting.plot_angular_velocity_error(t_hist, X_hist.T[:, 6:15], X_hist.T[:, 15:18], spacecraft.x_f[6:15], spacecraft.x_f[15:18], save_fig=False)