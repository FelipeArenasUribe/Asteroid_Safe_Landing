import numpy as np
import matplotlib.pyplot as plt

import ARC_Grav

from src import utils
from src import main
from src import traj_gen
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

F_max = 100.0 # N

# Define the upper bounds
u_max = np.array([F_max, F_max, F_max, F_max, F_max, F_max, F_max, F_max])
u_min = -u_max

A = np.vstack([np.eye(8), -np.eye(8)])
b = np.hstack([u_max, -u_min]) 

M = np.array([
    [1.0, 0.0, -1.0,  0.0, 1.0, 0.0,  0.0,  0.0], # Fx
    [0.0, 1.0,  0.0, -1.0, 0.0, 1.0,  0.0,  0.0], # Fy
    [0.0, 0.0,  0.0,  0.0, 1.0, 1.0,  1.0,  1.0], # Fz
    [0.0, 0.0,  0.0,  0.0, 1.0, 0.0, -1.0,  0.0], # Tx
    [0.0, 0.0,  0.0,  0.0, 0.0, 1.0,  0.0, -1.0], # Ty
    [1.0, 1.0,  1.0,  1.0, 0.0, 0.0,  0.0,  0.0]  # Tz
])

# Spacecraft setup
J = np.array([
    [ 4200.0,   80.0,   -40.0],
    [ 80.0,   4000.0,   -60.0],
    [ -40.0,   -60.0,   3200.0]
])

spacecraft = main.spacecraft(x_0=x_0, r_f=r_f, R_f = R_f, mass=3500, inertia=J, M=M, A=A, b=b, theta_gs=np.deg2rad(45.0), theta_b=np.deg2rad(60.0), e_s=e_s, CB=asteroid, mass_uncertainty = 0.10, omega_uncertainty = 0.0, r_l = 2.0, v_l = 0.1, theta_l= 0.2, omega_l = 0.2)

# ------------------------------------------------------------
# Create Reference Trajectory
T_f = 500
N = 100

traj_generator = traj_gen.TrajectoryGenerator(spacecraft, T_f, N)

X_traj, U_traj = traj_generator.Opt_Traj()

t_hist = np.linspace(0,T_f,N)

plotting.plot_trajectory_with_body(X_traj, asteroid.main_body, az=35, el=10, location = 'upper right', save_fig=False)

plotting.plot_position_velocity_error(t_hist, X_traj, spacecraft.x_f, save_fig=False)

plotting.plot_virtual_controls(t_hist, U_traj, U_traj, labels=None, save_fig=False)

# Save multiple arrays into one compressed file
np.savez('traj_conf2_ellip.npz', X=X_traj, U=U_traj)
