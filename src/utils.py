import numpy as np
import ARC_Grav
from src import math_utils
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

def generate_known_CB(CB,mass_uncertainty = 0.0):
    '''
    Generate Newtonian CB with mass uncerainty based on real CB
    '''
    known_mass =  CB.mass*(1.0 - mass_uncertainty)
    known_radius = np.average([body.R for body in CB.main_body.sub_bodies])

    masses = np.array([body.mass for body in CB.main_body.sub_bodies])
    positions = np.array([np.asarray(body.position).reshape(3) for body in CB.main_body.sub_bodies])

    known_position = np.zeros(3)
    known_position[:] = (masses[:, None] * positions).sum(axis=0) / masses.sum() # computed as the center of mass of the main body

    known_spherical = ARC_Grav.main.Spherical_Body(mass = known_mass, radius = known_radius, position = known_position)

    return ARC_Grav.main.main_body(sub_bodies=[known_spherical])

def grav_disturbance(CB, known_CB, x):
        r = x[0:3]
        return CB.Gravity(r) - known_CB.Gravity(r)

def compute_boundary_positions(CB, alt, lat_disp, initial_rel_orientation = np.array([0.0, 0.0, 0.0]), sub_body_index = 0):
    '''
    Function to compute:
    1. Initial and final position
    2. Local perpendicular unit vector
    3. Initial and final attitude

    Inputs:
    CB: Celestial Body object
    alt: altitude with respect to landing position
    lat_disp: lateral displacement 2-vector
    initial_rel_orientation: euler angles of relative orientation with respect to final attitude
    '''

    sub_body = CB.main_body.sub_bodies[sub_body_index]

    rf_body = np.array([sub_body.dim[0], 0.0, 0.0])
    e_s_body = np.array([1, 0, 0])

    r0_body = np.array([sub_body.dim[0]+alt, lat_disp[0], lat_disp[1]])

    # Rot_matrix maps inertial to body
    if hasattr(sub_body, "Rot_matrix"):
        R_IB = sub_body.Rot_matrix
        R_BI = R_IB.T          # body to inertial
    else:
        R_BI = np.eye(3)

    # # --- Transform to inertial frame ---
    # r_0 = sub_body.position + R_BI @ r0_body
    # r_f = sub_body.position + R_BI @ rf_body
    # e_s = R_BI @ e_s_body     # surface normal in inertial frame
    # e_s /= np.linalg.norm(e_s)

    # --- Transform to inertial frame ---
    r_0 = sub_body.position + R_BI.T @ r0_body
    r_f = sub_body.position + R_BI.T @ rf_body
    e_s = R_BI.T @ e_s_body     # surface normal in inertial frame
    e_s /= np.linalg.norm(e_s)

    # Generate final attitude with k_th vector parallel to e_s
    e_k = e_s

    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(e_k, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    e_i = ref - np.dot(ref, e_k) * e_k
    e_i /= np.linalg.norm(e_i)
    e_j = np.cross(e_k, e_i)

    R_f = np.column_stack((e_i, e_j, e_k))

    # Initial attitude computed by relative rotations with respect to final attitude
    R_0f = math_utils.rotation_matrix_from_euler(*initial_rel_orientation)

    R_0 = R_f @ R_0f # Initial orientation with respect to asteroid frame

    return r_0, r_f, e_s, R_0, R_f

def compute_boundary_positions_irreg(CB, alt, lat_disp, initial_rel_orientation = np.array([0.0, 0.0, 0.0]), sub_body_index = 0):
    '''
    Function to compute:
    1. Initial and final position
    2. Local perpendicular unit vector
    3. Initial and final attitude

    Inputs:
    CB: Celestial Body object
    alt: altitude with respect to landing position
    lat_disp: lateral displacement 2-vector
    initial_rel_orientation: euler angles of relative orientation with respect to final attitude
    '''

    sub_body = CB.main_body.sub_bodies[sub_body_index]

    rf_body = np.array([-sub_body.dim[0], 0.0, 0.0])
    e_s_body = np.array([-1, 0, 0])

    r0_body = np.array([sub_body.dim[0]-alt, lat_disp[0], lat_disp[1]])

    print(rf_body)
    print(r0_body)
    input()

    # Rot_matrix maps inertial to body
    if hasattr(sub_body, "Rot_matrix"):
        R_IB = sub_body.Rot_matrix
        R_BI = R_IB.T          # body to inertial
    else:
        R_BI = np.eye(3)

    # --- Transform to inertial frame ---
    r_0 = sub_body.position + R_BI @ r0_body
    r_f = sub_body.position + R_BI @ rf_body
    e_s = R_BI @ e_s_body     # surface normal in inertial frame
    e_s /= np.linalg.norm(e_s)

    # Generate final attitude with k_th vector parallel to e_s
    e_k = e_s

    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(e_k, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    e_i = ref - np.dot(ref, e_k) * e_k
    e_i /= np.linalg.norm(e_i)
    e_j = np.cross(e_k, e_i)

    R_f = np.column_stack((e_i, e_j, e_k))

    # Initial attitude computed by relative rotations with respect to final attitude
    R_0f = math_utils.rotation_matrix_from_euler(*initial_rel_orientation)

    R_0 = R_f @ R_0f # Initial orientation with respect to asteroid frame

    return r_0, r_f, e_s, R_0, R_f

def compute_u_hist(spacecraft, u_tilde_hist):
    num_columns = u_tilde_hist.shape[1]
    u_hist = np.zeros((spacecraft.ell, num_columns))

    for i in range(num_columns):
        u_hist[:, i] = spacecraft.allocation(u_tilde_hist[:, i])

    return u_hist

def compute_state_constraint_functions(x_hist, spacecraft):
    # Determine the number of time steps N
    N = x_hist.shape[1]

    # Pre-allocate histories for state constraints
    psi_1_hist = np.zeros(N)
    psi_2_hist = np.zeros(N)
    psi_3_hist = np.zeros(N)
    psi_4_hist = np.zeros(N)

    def compute_primary(x):
        psi_1 = spacecraft.glideslope_constraint(x.reshape(-1, 1))
        psi_2 = spacecraft.field_of_view_constraint(x.reshape(-1, 1))
        psi_3 = spacecraft.speed_constraint(x)
        psi_4 = spacecraft.ang_speed_constraint(x)

        return psi_1, psi_2, psi_3, psi_4

    # Loop through each time step
    for k in range(N):
        x_k = x_hist[:, k]
        
        p1, p2, p3, p4 = compute_primary(x_k)
        
        # Use .item() to ensure we are grabbing the scalar value
        psi_1_hist[k] = np.array(p1).item()
        psi_2_hist[k] = np.array(p2).item()
        psi_3_hist[k] = np.array(p3).item()
        psi_4_hist[k] = np.array(p4).item()

    return psi_1_hist, psi_2_hist, psi_3_hist, psi_4_hist

def compute_constraints_vmap(x_hist, u_hist, nu_hist, w_bar_hist, safety_controller):

    x_hist     = jnp.asarray(x_hist)
    u_hist     = jnp.asarray(u_hist)
    nu_hist    = jnp.asarray(nu_hist)
    w_bar_hist = jnp.asarray(w_bar_hist)

    if w_bar_hist.ndim > 1:
        w_bar_hist = w_bar_hist.ravel()

    # histories are shaped (state_dim, T) → vmap over axis 0 after transposing
    x_hist_T     = x_hist.T      # (T, 18)
    u_hist_T     = u_hist.T      # (T, 6)
    nu_hist_T    = nu_hist.T     # (T, 3)
    # w_bar_hist is already (T,)

    # ------------------------------------------------------------------
    # 1. Per-timestep CBF evaluation (all psi functions + phi + softmin)
    # ------------------------------------------------------------------
    def compute_all_constraints(x, u, nu, w_bar):

        # --- Level-0 CBFs (state only) ---
        psi_1_0 = safety_controller.psi_1_0(x)
        psi_2_0 = safety_controller.psi_2_0(x)
        psi_3_0 = safety_controller.psi_3_0(x)
        psi_4_0 = safety_controller.psi_4_0(x)

        # --- Level-1 CBFs ---
        psi_1_1 = safety_controller.psi_1_1(x)
        psi_2_1 = safety_controller.psi_2_1(x)
        # psi_3 and psi_4 jump straight to their terminal barrier
        # (psi_3_1_bar and psi_4_1 are the terminal conditions)

        # --- Terminal barrier functions (what the safety filter acts on) ---
        psi_1_2_bar = safety_controller.psi_1_2_bar(x, u, nu, w_bar)
        psi_2_2_bar = safety_controller.psi_2_2_bar(x, u, nu, w_bar)
        psi_3_1_bar = safety_controller.psi_3_1_bar(x, u, nu, w_bar)
        psi_4_1     = safety_controller.psi_4_1(x, u)

        # --- Input constraint ---
        phi_val = safety_controller.phi(u)          # vector, shape (n_constraints,)

        # --- Softmin over all barriers (matches softmin_h exactly) ---
        b_state = jnp.multiply(
            safety_controller.k_state,
            jnp.array([psi_1_2_bar, psi_2_2_bar, psi_3_1_bar, psi_4_1])
        )
        b_input = jnp.multiply(safety_controller.k_input, phi_val)
        b       = jnp.append(b_state, b_input)
        h_soft  = -(1.0 / safety_controller.rho) * logsumexp(-safety_controller.rho * b)
        h_min = jnp.min(b)

        return (
            psi_1_0, psi_2_0, psi_3_0, psi_4_0,
            psi_1_1, psi_2_1,
            psi_1_2_bar, psi_2_2_bar, psi_3_1_bar, psi_4_1,
            phi_val,
            b_state, b_input,
            h_soft, h_min
        )

    # ------------------------------------------------------------------
    # 2. vmap over time
    # ------------------------------------------------------------------
    (
        psi_1_0_hist, psi_2_0_hist, psi_3_0_hist, psi_4_0_hist,
        psi_1_1_hist, psi_2_1_hist,
        psi_1_2_bar_hist, psi_2_2_bar_hist, psi_3_1_bar_hist, psi_4_1_hist,
        phi_hist,
        b_state_hist, b_input_hist,
        h_soft_hist, h_min_hist,
    ) = jax.vmap(compute_all_constraints, in_axes=(0, 0, 0, 0))(
        x_hist_T, u_hist_T, nu_hist_T, w_bar_hist
    )

    # ------------------------------------------------------------------
    # 3. Return everything as NumPy for plotting
    # ------------------------------------------------------------------
    return {
        # Level-0 barriers (glideslope, FOV, speed, angular speed)
        "psi_1_0": np.asarray(psi_1_0_hist),
        "psi_2_0": np.asarray(psi_2_0_hist),
        "psi_3_0": np.asarray(psi_3_0_hist),
        "psi_4_0": np.asarray(psi_4_0_hist),

        # Level-1 barriers (glideslope and FOV chains are degree-2)
        "psi_1_1": np.asarray(psi_1_1_hist),
        "psi_2_1": np.asarray(psi_2_1_hist),

        # Terminal barriers (what the filter directly enforces)
        "psi_1_2_bar": np.asarray(psi_1_2_bar_hist),
        "psi_2_2_bar": np.asarray(psi_2_2_bar_hist),
        "psi_3_1_bar": np.asarray(psi_3_1_bar_hist),
        "psi_4_1":     np.asarray(psi_4_1_hist),

        # Input constraints: shape (T, n_input_constraints)
        "phi": np.asarray(phi_hist),

        # Scaled components fed into softmin — useful for diagnosing which
        # constraint is active (i.e. which component is the softmin minimum)
        "b_state": np.asarray(b_state_hist),   # shape (T, 4)
        "b_input": np.asarray(b_input_hist),   # shape (T, n_input_constraints)

        # Aggregate softmin CBF value — must stay ≥ 0 for safety
        "h_soft": np.asarray(h_soft_hist),
        "h_min": np.asarray(h_min_hist),
    }

def landing_results(spacecraft, x_hist, t_vec):

    landing_index = None

    for k, x in enumerate(x_hist.T):
        if spacecraft.landing_condition(x):
            landing_index = k
            break
    else:
        print("Landing condition was never met.")
        return

    xf = spacecraft.xf.reshape(-1)

    pos_error = np.linalg.norm(x[0:3] - xf[0:3])
    vel_error = np.linalg.norm(x[3:6] - xf[3:6])
    alt_error = np.linalg.norm(
        spacecraft.e_hat.T @ (x[0:3] - xf[0:3])
    )

    landing_time = t_vec[landing_index]

    print('Landing time:    ', landing_time)
    print('Position error:  ', pos_error)
    print('Altitude error:  ', alt_error)
    print('Speed error:     ', vel_error)