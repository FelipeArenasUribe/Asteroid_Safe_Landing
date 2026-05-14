import numpy as np
import jax.numpy as jnp

def skew_symmetric(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def jnp_skew_symmetric(v):
    return jnp.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def skew_symmetric_to_vector(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])

def jnp_skew_symmetric_to_vector(S):
    return jnp.array([S[2, 1], S[0, 2], S[1, 0]])

def R_x(theta):
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])

def R_y(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])

def R_z(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])

def rotation_matrix_from_euler(roll, pitch, yaw, sequence = "123"):
    """
    Returns the rotation matrix from Euler angles (roll, pitch, yaw) of a 1-2-3 rotation sequence.
    The angles are in radians.
    """

    if sequence == "123":
        return R_z(yaw) @ R_y(pitch) @ R_x(roll)
    elif sequence == "321":
        return R_x(roll) @ R_y(pitch) @ R_z(yaw)
    else:
        raise ValueError("Unsupported rotation sequence")

def euler_from_rotation_matrix(R):
    """
    Returns the Euler angles (roll, pitch, yaw) from a rotation matrix R.
    The angles are in radians.
    """
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0

    return np.array([roll, pitch, yaw])

def d_SO3(P, Q):
    """
    Computes the geodesic distance between two rotation matrices P and Q.
    
    Args:
        P (np.array): A 3x3 orthogonal matrix with det=1.
        Q (np.array): A 3x3 orthogonal matrix with det=1.
        
    Returns:
        float: The distance d(P, Q) in radians [0, pi].
    """
    # Calculate the trace of P.T @ Q
    trace_val = np.trace(np.dot(P.T, Q))
    
    # Compute the argument for arccos
    arg = (trace_val - 1.0) / 2.0
    
    # Clip the argument to handle numerical precision issues
    # preventing NaN results if arg is 1.0000000000000002
    arg = np.clip(arg, -1.0, 1.0)
    
    return np.arccos(arg)

def attitude_error_function(R, R_d):
    """
    Computes the attitude error function between two rotation matrices.
    """
    Psi = 2 - np.sqrt(1 + np.trace(R_d.T @ R))
    return Psi

def attitude_error_vector(R, R_d):
    """
    Computes the attitude error vector between two rotation matrices.
    Returns a 3x1 vector as defined by:
    eR(R, Rd) = (1 / (2 * sqrt(1 + tr(Rd.T @ R)))) * vee(Rd.T @ R - R.T @ Rd)
    """
    S = R_d.T @ R - R.T @ R_d
    scale = 1 / (2 * np.sqrt(1 + np.trace(R_d.T @ R)))
    e_R = scale * skew_symmetric_to_vector(S)
    return e_R

def angular_velocity_error_vector(R, omega, R_d, omega_d):
    """
    Computes the angular velocity error vector between two angular velocities.
    """
    return omega - R.T @ R_d @ omega_d

def project_to_SO3(R):
    U, _, Vt = np.linalg.svd(R)
    R_proj = U @ Vt
    # Ensure det(R) = +1
    if np.linalg.det(R_proj) < 0:
        U[:, -1] *= -1
        R_proj = U @ Vt
    return R_proj