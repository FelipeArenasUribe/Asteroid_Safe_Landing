import numpy as np
import matplotlib.pyplot as plt
from src import math_utils
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.ticker as ticker

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 10,
    "font.family": "serif"
})

def plot_trajectory_with_body(X_hist, main_body, index=-1, n_points=100, az=-55, el=10, location = 'upper right', save_fig=False):
    """
    Plot 3D spacecraft trajectory along with a celestial body composed of multiple sub-bodies.
    """

    X_hist_T = X_hist.T
    X = X_hist_T[:, 0:3]

    fig = plt.figure(figsize=(4.0, 3.5))
    ax = fig.add_subplot(111, projection='3d')

    # --- Plot trajectory ---
    ax.plot(X[:, 0], X[:, 1], X[:, 2], lw=2.5, color='mediumblue')
    ax.scatter(X[0, 0], X[0, 1], X[0, 2], color='lime', s=25, label="$r(0)$")
    ax.scatter(X[index, 0], X[index, 1], X[index, 2], color='red', s=25, label="$r(t_f)$")

    # --- Plot each sub-body ---
    u = np.linspace(0, 2*np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    U, V = np.meshgrid(u, v)

    # for sub_body in reversed(main_body.sub_bodies):
    for sub_body in main_body.sub_bodies:
        if sub_body.type == 'Ellipsoid':
            a, b, c = sub_body.dim
            x = a * np.cos(U) * np.sin(V)
            y = b * np.sin(U) * np.sin(V)
            z = c * np.cos(V)
        else:
            x = sub_body.R * np.cos(U) * np.sin(V)
            y = sub_body.R * np.sin(U) * np.sin(V)
            z = sub_body.R * np.cos(V)

        # Apply rotation and translation
        xyz = np.vstack((x.ravel(), y.ravel(), z.ravel()))
        xyz_rot = sub_body.Rot_matrix @ xyz
        xyz_rot[0, :] += sub_body.position[0]
        xyz_rot[1, :] += sub_body.position[1]
        xyz_rot[2, :] += sub_body.position[2]

        x_rot = xyz_rot[0, :].reshape(n_points, n_points)
        y_rot = xyz_rot[1, :].reshape(n_points, n_points)
        z_rot = xyz_rot[2, :].reshape(n_points, n_points)

        # Plot as solid gray surface
        ax.plot_surface(x_rot, y_rot, z_rot, color='gray', alpha=1.0, linewidth=0, antialiased=True)

    # --- Equal aspect ratio ---
    all_points = np.vstack([X] + [sub.position.reshape(1,3) for sub in main_body.sub_bodies])
    x_limits = [all_points[:,0].min(), all_points[:,0].max()]
    y_limits = [all_points[:,1].min(), all_points[:,1].max()]
    z_limits = [all_points[:,2].min(), all_points[:,2].max()]

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    max_range = max(x_range, y_range, z_range)

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim([x_middle - max_range/2, x_middle + max_range/2])
    ax.set_ylim([y_middle - max_range/2, y_middle + max_range/2])
    ax.set_zlim([z_middle - max_range/2, z_middle + max_range/2])

    # --- Labels and style ---
    ax.set_xlabel('$r_x$ [m]', labelpad=-5, fontsize=9)
    ax.set_ylabel('$r_y$ [m]', labelpad=-5, fontsize=9)
    ax.set_zlabel('$r_z$ [m]', labelpad=-5, fontsize=9)
    
    # Reduce tick label font size
    ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(500))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(500))
    ax.tick_params(axis='both', which='major', labelsize=8, pad =-2)

    # --- Legend ---
    # ax.legend(loc=location, fontsize=9)

    # bbox_to_anchor=(x, y) where (1, 1) is the far top-right corner.
    # Lowering these values moves the legend toward the center.
    ax.legend(
        loc=location, 
        bbox_to_anchor=(0.85, 0.85), 
        fontsize=9,
        frameon=True,
        borderaxespad=0.1
    )

    # --- Set view angle ---
    ax.view_init(elev=el, azim=az)

    plt.tight_layout()
    if save_fig:
        filename = 'traj_plot.pdf'
        fig.savefig(filename, bbox_inches='tight', pad_inches=0.3)

    plt.show()

def plot_position_attitude_error(t_vec, X_traj, xf, save_fig=False):
    """
    Plot position and velocity errors over time.

    Inputs:
        t_vec : (N+1,) time array
        X_traj: (n_x, N+1) state trajectory
        xf    : final desired state (length 6) [r(3), v(3)]
    """
    r = X_traj[0:3, :]

    r_d = xf[0:3].reshape(3, 1)

    e_r = r - r_d

    d = np.zeros_like(e_r[0,:])

    for i in range(len(d)):
        d[i] = math_utils.d_SO3(X_traj[6:15, i].reshape(3,3), xf[6:15].reshape(3,3))


    fig, ax = plt.subplots(2, 1, figsize=(3.5, 2.5), sharex=True)

    # Plot position errors
    ax[0].plot(t_vec, e_r[0, :], label='$e_x$')
    ax[0].plot(t_vec, e_r[1, :], label='$e_y$')
    ax[0].plot(t_vec, e_r[2, :], label='$e_z$')
    ax[0].set_ylabel('$e(x)$ [m]')

    # Optimized legend spacing
    ax[0].legend(
        loc='lower right',
        frameon=False,
        handletextpad=0.2,  # Reduces space between line and text
        labelspacing=0.075,    # Reduces vertical space between entries
        borderpad=0.3,       # Reduces whitespace inside the legend box
        borderaxespad=0.05    # Distance from the edge of the plot
    )

    ax[0].grid(False)
    ax[0].axhline(0, color='gray', linestyle='--')
    ax[0].set_xlim(t_vec.min(), t_vec.max())

    # Plot attitude error
    ax[1].plot(t_vec, d[:])
    ax[1].set_ylabel('$d(R,R_{\\rm d})$ [rad]')
    ax[1].grid(False)
    ax[1].set_xlim(t_vec.min(), t_vec.max())  # Set x-limits
    ax[1].axhline(0, color='gray', linestyle='--')

    ax[1].set_xlabel('$t(s)$')

    
    plt.tight_layout()

    if save_fig:
        plt.savefig('Position_Attitude_error_plot.pdf')
    plt.show()

def plot_state_constraints(psi_hists, t_vec, save_fig=False):
    """
    psi_hists: List or tuple containing [psi_1_hist, psi_2_hist, psi_3_hist, psi_4_hist]
    t_vec: Time vector of size N
    """
    t_min, t_max = t_vec[0], t_vec[-1]
    psi_labels = [r'$\psi_{1}(x)$', r'$\psi_{2}(x)$', r'$\psi_{3}(x)$', r'$\psi_{4}(x)$']

    # --- Figure 1: State Constraints (4 Subplots) ---
    fig1, axs1 = plt.subplots(4, 1, figsize=(3.5, 4.0), sharex=True)
    
    for i in range(4):
        axs1[i].plot(t_vec, psi_hists[i], linewidth=1.5, color='tab:blue')
        axs1[i].set_ylabel(psi_labels[i])

        # axs1[i].set_yscale('log')

        # axs1[i].grid(True, which='both', linestyle=':', alpha=0.5)
        
        upper_limit = np.max(psi_hists[i]) * 1.1 
        axs1[i].set_ylim(-0.001, upper_limit)

        axs1[i].set_xlim(t_min, t_max)
    
    axs1[-1].set_xlabel('$t(s)$')
    fig1.tight_layout()

    if save_fig:
        fig1.savefig('state_constraints.pdf', bbox_inches='tight')
    
    plt.show()

def plot_w_estimates(w_hat_hist, w_bar_hist, w_hist, t_vec, save_fig=False):

    # -----------------------------
    t = t_vec[:-1]
    w = w_hist[:, :-1]
    w_hat = w_hat_hist[:, :-1]
    w_bar = w_bar_hist[:, :-1]

    # Compute magnitudes
    w_mag = np.linalg.norm(w, axis=0)
    w_hat_mag = np.linalg.norm(w_hat, axis=0)
    w_bar_mag = np.linalg.norm(w_bar, axis=0)

    t_min, t_max = t[0], t[-1]

    error_mag = np.linalg.norm(w - w_hat, axis=0)

    plt.figure(figsize=(3.5, 1.5))
    plt.plot(t, error_mag, label=r'$\|\tilde{w}\|$', linewidth=2)
    plt.plot(t, w_bar_mag, ':', label=r'$\sqrt{\bar{w}}$', linewidth=2)
    plt.xlabel('$t(s)$', fontsize=9)
    plt.ylabel(r'$\|\tilde{w}\|$ [$\mathrm{m/s^2}$]', fontsize=9)
    plt.yscale('log')
    plt.xlim(t_min, t_max)
    plt.legend(fontsize=8, loc='best', frameon=False)
    # plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.grid(False)
    plt.tight_layout()

    if save_fig:
        filename = 'dist_est_upper_bound_plot.pdf'
        plt.savefig(filename, bbox_inches='tight')

    plt.show()

def plot_virtual_controls(t, u_tilde, U_d_traj, labels=None, save_fig=False):
    
    u_tilde = np.array(u_tilde)
    U_d_traj = np.array(U_d_traj)
    
    n = u_tilde.shape[0]
    
    if labels is None:
        labels = [r'$F_x$', r'$F_y$', r'$F_z$', r'$\tau_x$', r'$\tau_y$', r'$\tau_z$'][:n]

    fig, ax = plt.subplots(n, 1, figsize=(3.5, 0.8*n), sharex=True)
    if n == 1: ax = [ax]

    num_data_points = U_d_traj.shape[1]
    u_tilde = u_tilde[:,:num_data_points]
    t_plot = t[:num_data_points] 

    for i in range(n):
        ax[i].plot(t_plot, u_tilde[i, :], label=r'$\tilde{u}$')
        ax[i].plot(t_plot, U_d_traj[i, :], '--', label=r'$\tilde{u}_{d}$')
        
        ax[i].set_xlim([t_plot[0], t_plot[-1]])
        ax[i].set_ylabel(labels[i])
        if i==0:
            ax[i].legend(
            loc='upper left',
            frameon=False,
            handletextpad=0.2,  # Reduces space between line and text
            labelspacing=0.15,    # Reduces vertical space between entries
            borderpad=0.3,       # Reduces whitespace inside the legend box
            borderaxespad=0.025,    # Distance from the edge of the plot
            ncol=2
            )
        ax[i].grid(False)
        
    
    ax[-1].set_xlabel('$t(s)$')
    plt.tight_layout()
    if save_fig: plt.savefig('virtual_control_inputs_plot.pdf')
    plt.show()

def plot_actuator_inputs(t_hist, u_hist, spacecraft, save_fig=False):
    """
    Plots the history of each actuator and its corresponding physical limits.
    
    Parameters:
    - t_hist: Array of time steps (length N)
    - u_hist: Array of actuator inputs (shape: ell x N)
    - spacecraft: Object containing .ell, .A, and .b
    """

    num_data_points = u_hist.shape[1]
    t_hist = t_hist[:num_data_points]

    n_actuators = spacecraft.ell
    t_hist = np.array(t_hist)
    u_hist = np.array(u_hist)

    # Create subplots: one for each actuator
    fig, axes = plt.subplots(n_actuators, 1, figsize=(3.5, 0.8 * n_actuators), sharex=True)
    
    # Ensure axes is always iterable even if there is only 1 actuator
    if n_actuators == 1:
        axes = [axes]

    for i in range(n_actuators):
        ax = axes[i]
        
        # 1. Plot the actual actuator data
        ax.plot(t_hist, u_hist[i, :], label=f'Actuator {i+1}', color='tab:blue', linewidth=1.5)

        # 2. Extract and plot limits from A*u <= b
        # We look for rows in A that constrain only actuator i
        upper_bound = None
        lower_bound = None
        
        for row_idx in range(spacecraft.A.shape[0]):
            row = spacecraft.A[row_idx, :]
            # Check if row has only one non-zero element at index i
            non_zero_indices = np.where(row != 0)[0]
            
            if len(non_zero_indices) == 1 and non_zero_indices[0] == i:
                val = spacecraft.b[row_idx] / row[i]
                if row[i] > 0:
                    upper_bound = val if upper_bound is None else min(upper_bound, val)
                else:
                    lower_bound = val if lower_bound is None else max(lower_bound, val)

        # Plot horizontal lines for limits if found
        if upper_bound is not None:
            ax.axhline(y=upper_bound, color='r', linestyle='--', alpha=0.3)
        if lower_bound is not None:
            ax.axhline(y=lower_bound, color='r', linestyle='--', alpha=0.3)

        # Formatting
        ax.set_ylabel(f'$u_{i+1}$')
        ax.set_xlim([t_hist[0], t_hist[-1]])
        # ax.grid(True, linestyle=':', alpha=0.6)
        # if i == 0:
        #     ax.legend(loc='upper right', fontsize='small', ncol=3)

    axes[-1].set_xlabel('$t(s)$')

    plt.tight_layout()
    if save_fig:
        plt.savefig('actuator_history.pdf')
    
    plt.show()

def plot_h_softmin(t_vec, h_softmin, h_min, save_fig=False):

    # -----------------------------
    t = t_vec

    t_min, t_max = t[0], t[-1]

    plt.figure(figsize=(3.5, 1.5))
    plt.plot(t, h_softmin, label=r'$h(\tilde{x})$', linewidth=1.5)
    plt.plot(t, h_min, '--', label=r'$\min(\tilde{x})$', linewidth=1.5)
    plt.xlabel('$t(s)$', fontsize=9)
    plt.ylabel(r'$h(\tilde{x})$', fontsize=9)
    plt.xlim(t_min, t_max)
    plt.legend(fontsize=8, loc='upper right', frameon=True)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_fig:
        filename = 'h_softmin_plot.pdf'
        plt.savefig(filename, bbox_inches='tight')

    plt.show()

def plot_h_softmin_log(t_vec, h_softmin, h_min, save_fig=False):

    # -----------------------------
    t = t_vec

    t_min, t_max = t[0], t[-1]

    eps = np.min(h_softmin)

    plt.figure(figsize=(3.5, 1.5))
    plt.plot(t, h_softmin + eps, label=r'$h(\tilde{x})$', linewidth=1.5)
    plt.plot(t, h_min + eps, '--', label=r'$\min(\tilde{x})$', linewidth=1.5)
    
    # Set y-axis to log scale
    plt.yscale('log')
    
    plt.xlabel('$t(s)$', fontsize=9)
    plt.ylabel(r'$h(\tilde{x})$', fontsize=9)
    plt.xlim(t_min, t_max)
    plt.legend(fontsize=8, loc='upper right', frameon=True)
    
    # Updated grid to handle log scale minor ticks
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    plt.tight_layout()

    if save_fig:
        filename = 'h_softmin_plot.pdf'
        plt.savefig(filename, bbox_inches='tight')

    plt.show()

def plot_h_softmin_lambda(t_vec, h_softmin, h_min, lambda_hist, save_fig=False):

    num_data_points = lambda_hist.size
    t_vec = t_vec[:num_data_points]
    h_softmin = h_softmin[:num_data_points]
    h_min = h_min[:num_data_points]

    t_min, t_max = t_vec[0], t_vec[-1]

    # Increased height slightly to prevent cramped labels
    fig, ax = plt.subplots(2, 1, figsize=(3.5, 2.5), sharex=True)

    # Top Plot: h(x) comparison
    ax[0].plot(t_vec, h_softmin, label=r'$h(\tilde{x})$', linewidth=1.5)
    ax[0].plot(t_vec, h_min, '--', label=r'$\min(\tilde{x})$', linewidth=1.5)
    ax[0].set_ylabel(r'$h(\tilde{x})$', fontsize=9)
    ax[0].set_xlim(t_min, t_max)
    upper_limit = np.max(h_min) * 1.1 
    ax[0].set_ylim(-0.1, upper_limit)
    ax[0].legend(fontsize=8, loc='upper right', frameon=False)
    # ax[0].grid(True, which='both', linestyle='--', alpha=0.5)
    # ax[0].axhline(0, color='gray', linestyle='--')

    # Bottom Plot: Lambda history
    ax[1].semilogy(t_vec, lambda_hist + 1) 
    ax[1].set_ylabel(r'$\lambda + 1$', fontsize=9)
    ax[1].set_xlabel('$t$ (s)', fontsize=9)
    ax[1].set_xlim(t_min, t_max)
    ax[1].grid(True, which='both', linestyle='--', alpha=0.5)

    plt.tight_layout()

    if save_fig:
        plt.savefig('h_softmin_plot.pdf', bbox_inches='tight')

    plt.show()

# === Auxiliary Plots ===

def plot_position_velocity_error(t_vec, X_traj, xf, save_fig=False):
    """
    Plot position and velocity errors over time.

    Inputs:
        t_vec : (N+1,) time array
        X_traj: (n_x, N+1) state trajectory
        xf    : final desired state (length 6) [r(3), v(3)]
    """
    r = X_traj[0:3, :]
    v = X_traj[3:6, :]

    r_d = xf[0:3].reshape(3, 1)
    v_d = xf[3:6].reshape(3, 1)

    e_r = r - r_d
    e_v = v - v_d

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Plot position errors
    ax[0].plot(t_vec, e_r[0, :], label='$e_x$')
    ax[0].plot(t_vec, e_r[1, :], label='$e_y$')
    ax[0].plot(t_vec, e_r[2, :], label='$e_z$')
    ax[0].set_ylabel('$e(t)$ [m]')
    ax[0].legend()
    ax[0].grid(True)
    ax[0].set_xlim(t_vec.min(), t_vec.max())  # Set x-limits

    # Plot velocity errors
    ax[1].plot(t_vec, e_v[0, :], label='$\\dot{e}_x$')
    ax[1].plot(t_vec, e_v[1, :], label='$\\dot{e}_y$')
    ax[1].plot(t_vec, e_v[2, :], label='$\\dot{e}_z$')
    ax[1].set_ylabel('$\\dot{e}(t)$ [m/s]')
    ax[1].legend()
    ax[1].grid(True)
    ax[1].set_xlim(t_vec.min(), t_vec.max())  # Set x-limits

    ax[1].set_xlabel('$t(s)$')

    if save_fig:
        plt.savefig('Position_Velocity_error_plot.pdf')
    plt.tight_layout()
    plt.show()

def plot_rotation_matrix(t, R, R_d, save_fig=False):
    """R is (N, 9), R_d is (9,) or (3,3)"""
    fig, ax = plt.subplots(3, 3, figsize=(8, 6), sharex=True)
    R_d_flat = np.array(R_d).flatten()
    
    for i in range(3):
        for j in range(3):
            idx = i * 3 + j
            ax[i, j].plot(t, R[:, idx], label='Actual')
            ax[i, j].axhline(y=R_d_flat[idx], color='r', linestyle='--', alpha=0.7)
            ax[i, j].set_title(f'$R_{{{i+1},{j+1}}}$')
            ax[i, j].grid(True)
            ax[i, j].set_xlim(t.min(), t.max())
            if i == 2:
                ax[i, j].set_xlabel('Time [s]')
                
    plt.suptitle('Rotation Matrix Components')
    plt.tight_layout()
    if save_fig: plt.savefig('Rotation_Matrix_plot.pdf')
    plt.show()

def plot_attitude_error_function(t_hist, R_hist, R_d, save_fig=False):
    Psi_hist = []
    Rd_mat = np.array(R_d).reshape(3, 3)
    
    for R_vec in R_hist:
        Psi = math_utils.attitude_error_function(R_vec.reshape(3, 3), Rd_mat)
        Psi_hist.append(Psi)

    plt.figure(figsize=(8, 4))
    plt.plot(t_hist, Psi_hist, linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel(r"$\Psi(t)$")
    # Corrected: plt.xlim instead of plt.set_xlim
    plt.xlim(t_hist.min(), t_hist.max()) 
    plt.title("Attitude Error Function")
    plt.grid(True)
    if save_fig: plt.savefig('Attitude_error_func_plot.pdf')
    plt.show()

def plot_attitude_error_vector(t_hist, R_hist, R_d, save_fig=False):
    eR_hist = []
    Rd_mat = np.array(R_d).reshape(3, 3)
    
    for R_vec in R_hist:
        eR = math_utils.attitude_error_vector(R_vec.reshape(3, 3), Rd_mat)
        eR_hist.append(eR)

    eR_hist = np.array(eR_hist)
    plt.figure(figsize=(8, 4))
    for i in range(3):
        plt.plot(t_hist, eR_hist[:, i], label=f"$e_{{{i+1}}}$")
    plt.xlabel("Time [s]")
    plt.ylabel(r"$e_R(t)$")
    # Corrected: plt.xlim instead of plt.set_xlim
    plt.xlim(t_hist.min(), t_hist.max())
    plt.legend()
    plt.grid(True)
    if save_fig: plt.savefig('Attitude_error_plot.pdf')
    plt.show()

def plot_angular_velocity_error(t_hist, R_hist, W_hist, R_d, W_d, save_fig=False):
    """R_hist (N,9), W_hist (N,3), R_d constant, W_d constant"""
    ew_hist = []
    
    # Pre-process constants to ensure they are the right shape for the math
    Rd_mat = np.array(R_d).reshape(3, 3)
    Wd_vec = np.array(W_d).flatten()
    
    for R_vec, W in zip(R_hist, W_hist):
        # Calculate error
        ew = math_utils.angular_velocity_error_vector(R_vec.reshape(3, 3), W, Rd_mat, Wd_vec)
        
        # FIX: Force 'ew' to be a 1D array with 3 elements 
        # This handles the case where the math function returns (3,1) or (3,3)
        ew_hist.append(np.array(ew).flatten()[:3])

    ew_hist = np.array(ew_hist)
    
    plt.figure(figsize=(8, 4))
    for i in range(3):
        # Now ew_hist is guaranteed to be (N, 3), so indexing works perfectly
        plt.plot(t_hist, ew_hist[:, i], label=f"$e_{{{i+1}}}$")
        
    plt.xlabel("Time [s]")
    plt.ylabel(r"$e_\omega(t)$")
    plt.xlim(t_hist.min(), t_hist.max())
    plt.legend()
    plt.grid(True)
    
    if save_fig: 
        plt.savefig('Ang_vel_error_plot.pdf')
    plt.show()

def plot_control(U_traj, U_d_traj, t_vec, save_fig=False):
    """
    Plot control trajectories.

    Inputs:
        U_traj   : (n_u, N)   control trajectory
        U_d_traj : (n_u, N)   desired control trajectory
        t_vec    : (N+1,)     time vector
    """

    n_u, N = U_traj.shape
    t_min, t_max = t_vec[0], t_vec[-1]

    fig1, axs = plt.subplots(n_u, 1, figsize=(3.5, 2.7), sharex=True)
    if n_u == 1:
        axs = [axs]

    labels = ['x', 'y', 'z']

    for i in range(n_u):
        axs[i].step(t_vec, U_traj[i, :], where='post', linewidth=1.5)
        axs[i].step(t_vec, U_d_traj[i, :], where='post',
                    linewidth=1.5, linestyle='--')

        axs[i].set_ylabel(fr'$u_{{{labels[i]}}}$ [N]')
        axs[i].set_xlim(t_min, t_max)
        axs[i].grid(True)

    if n_u >= 3:
        axs[0].legend([r'$u$', r'$u_d$'],
                      fontsize=8,
                      loc='best',
                      frameon=False)

    # Helper function for zoom boxes
    

    axs[-1].set_xlabel('$t(s)$')

    plt.tight_layout()

    if save_fig:
        filename = 'control_plot.pdf'
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def plot_safety_analysis(results, t_vec):
    """
    Generates diagnostic plots for CBF levels, input constraints, and softmin aggregate.
    
    Args:
        results (dict): Dictionary containing the psi, phi, and h_soft history arrays.
        t_vec (ndarray): Time vector of size N.
    """
    t_min, t_max = t_vec[0], t_vec[-1]
    
    # Helper to add zero lines and grid
    def format_ax(ax, ylabel):
        ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_ylabel(ylabel)
        ax.grid(True, which='both', linestyle=':', alpha=0.5)
        ax.set_xlim(t_min, t_max)

    # 1. Barrier 1 Hierarchy (Level 0, 1, 2_bar)
    fig1, axs1 = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    fig1.suptitle("Barrier 1 Hierarchy", fontweight='bold')
    axs1[0].plot(t_vec, results["psi_1_0"], color='tab:blue')
    format_ax(axs1[0], r"$\psi_{1,0}$")
    axs1[1].plot(t_vec, results["psi_1_1"], color='tab:blue')
    format_ax(axs1[1], r"$\psi_{1,1}$")
    axs1[2].plot(t_vec, results["psi_1_2_bar"], color='tab:blue')
    format_ax(axs1[2], r"$\bar{\psi}_{1,2}$")
    axs1[-1].set_xlabel("Time (s)")

    # 2. Barrier 2 Hierarchy (Level 0, 1, 2_bar)
    fig2, axs2 = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    fig2.suptitle("Barrier 2 Hierarchy", fontweight='bold')
    axs2[0].plot(t_vec, results["psi_2_0"], color='tab:orange')
    format_ax(axs2[0], r"$\psi_{2,0}$")
    axs2[1].plot(t_vec, results["psi_2_1"], color='tab:orange')
    format_ax(axs2[1], r"$\psi_{2,1}$")
    axs2[2].plot(t_vec, results["psi_2_2_bar"], color='tab:orange')
    format_ax(axs2[2], r"$\bar{\psi}_{2,2}$")
    axs2[-1].set_xlabel("Time (s)")

    # 3. Barrier 3 Hierarchy (Level 0, 1_bar)
    fig3, axs3 = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    fig3.suptitle("Barrier 3 Hierarchy", fontweight='bold')
    axs3[0].plot(t_vec, results["psi_3_0"], color='tab:green')
    format_ax(axs3[0], r"$\psi_{3,0}$")
    axs3[1].plot(t_vec, results["psi_3_1_bar"], color='tab:green')
    format_ax(axs3[1], r"$\bar{\psi}_{3,1}$")
    axs3[-1].set_xlabel("Time (s)")

    # 4. Barrier 4 Hierarchy (Level 0, 1)
    fig4, axs4 = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    fig4.suptitle("Barrier 4 Hierarchy", fontweight='bold')
    axs4[0].plot(t_vec, results["psi_4_0"], color='tab:red')
    format_ax(axs4[0], r"$\psi_{4,0}$")
    axs4[1].plot(t_vec, results["psi_4_1"], color='tab:red')
    format_ax(axs4[1], r"$\psi_{4,1}$")
    axs4[-1].set_xlabel("Time (s)")

    # 5. Input Constraints (All phi)
    fig5, ax5 = plt.subplots(figsize=(8, 4))
    phi = results["phi"]
    for j in range(phi.shape[1]):
        ax5.plot(t_vec, phi[:, j], label=f"$\phi_{{{j+1}}}$")
    format_ax(ax5, r"$\phi(\tilde{u})$")
    ax5.set_title("Input Constraints", fontweight='bold')
    ax5.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax5.set_xlabel("Time (s)")

    # 6. Terminal Filter Components (Composite)
    fig6, axs6 = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    fig6.suptitle("Safety Filter Summary", fontweight='bold')
    
    # Flattening for easy indexing
    flat_axs = axs6.flatten()
    data_keys = ["psi_1_2_bar", "psi_2_2_bar", "psi_3_1_bar", "psi_4_1"]
    labels = [r"$\bar{\psi}_{1,2}$", r"$\bar{\psi}_{2,2}$", r"$\bar{\psi}_{3,1}$", r"$\psi_{4,1}$"]
    
    for i in range(4):
        flat_axs[i].plot(t_vec, results[data_keys[i]], color='black')
        format_ax(flat_axs[i], labels[i])
        
    # Plot phi (min across constraints for clarity in composite)
    flat_axs[4].plot(t_vec, np.min(results["phi"], axis=1), color='purple', label='min($\phi$)')
    format_ax(flat_axs[4], r"$\phi_{min}$")
    
    # Plot h_soft
    flat_axs[5].plot(t_vec, results["h_soft"], color='tab:red', linewidth=2)
    format_ax(flat_axs[5], r"$h_{soft}$")
    
    for i in [4, 5]: flat_axs[i].set_xlabel("Time (s)")
    fig6.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 7. Aggregate h_soft (Standalone)
    fig7, ax7 = plt.subplots(figsize=(8, 4))
    ax7.plot(t_vec, results["h_soft"], color='tab:red', linewidth=2)
    format_ax(ax7, r"$h_{soft}$")
    ax7.set_title("Aggregate Softmin CBF", fontweight='bold')
    ax7.set_xlabel("Time (s)")

    plt.show()