import numpy as np
from COVtoINF import cov_to_inf
from Removal import removal

def tupdate(u, B, V, Phi, gamma, Qk):
    """
    Time update from X(k) to X(k+1) in the Kalman filter.

    Parameters:
    u (numpy.ndarray): An n x 1 vector representing the mean of the state X(k) at time k.
    B (numpy.ndarray): An n x n matrix of Gaussian influence diagram arc coefficients of state X(k) at time k.
    V (numpy.ndarray): An n x 1 vector of Gaussian influence diagram conditional variances of state X(k).
    Phi (numpy.ndarray): The n x n state transition matrix Phi(k).
    gamma (numpy.ndarray): The n x r process noise matrix Gamma(k).
    Qk (numpy.ndarray): The r x r process noise covariance matrix of the process noise vector w(k).

    Returns:
    u (numpy.ndarray): The updated mean vector of the state X(k+1) at time k+1.
    B (numpy.ndarray): The updated matrix of Gaussian influence diagram arc coefficients of state X(k+1).
    V (numpy.ndarray): The updated vector of conditional variances of state X(k+1).
    """

    # Get dimensions
    n = u.shape[0]  # size of u
    r = Qk.shape[0]  # size of Qk

    # Convert process noise covariance matrix to influence diagram form
    Bq, Vq, _ = cov_to_inf(Qk, r)

    # Prepare block matrices
    On = np.zeros((n, n))
    Or = np.zeros((r, r))
    Onr = np.zeros((n, r))
    On1 = np.zeros((n, 1))
    Or1 = np.zeros((r, 1))

    # Create new V and B matrices for the update
    V_new = np.concatenate([Vq, V, On1.flatten()])
    B_new = np.block([
        [Bq, Onr.T, gamma.T],
        [Onr, B, Phi.T],
        [Onr, On, On]
    ])

    # Perform Removal operation to update B and V
    n0 = 0
    n1 = n + r
    n2 = n
    n3 = 0
    B, V = removal(B_new, V_new, n0, n1, n2)

    # Update V and B for the current step
    V = V.flatten()  # Ensure V is a flat vector
    B = B[n+r:n+r+n, n+r:n+r+n]  # Extract the relevant part of B
    V = V[n+r:n+r+n]  # Extract the relevant part of V

    # Update the mean vector
    u_new = Phi @ u  # Update u using the state transition matrix
    u = u_new

    return u, B, V
