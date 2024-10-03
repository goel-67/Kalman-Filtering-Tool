import numpy as np
from Mupdate import mupdate
from INFtoCOV import inf_to_cov

class trackingKF:
    def __init__(self, F, H, state, state_covariance, process_noise):
        self.StateTransitionModel = F
        self.MeasurementModel = H
        self.State = state
        self.StateCovariance = state_covariance
        self.ProcessNoise = process_noise


def id_correct(filter, zmeas, zcov):
    """
    Performs the measurement update process of the Kalman filter for object-oriented programming.

    Parameters:
    filter (object): A trackingKF object that contains State, StateCovariance, MeasurementModel, StateTransitionModel, and ProcessNoise.
    zmeas (numpy.ndarray): Measurement value.
    zcov (numpy.ndarray): Covariance of the measurement value.

    Returns:
    xcorr (numpy.ndarray): Corrected state of the measurement value.
    Pcorr (numpy.ndarray): Corrected state of the covariance.
    """

    # Extract filter parameters
    u = filter.State
    P = filter.StateCovariance
    H = filter.MeasurementModel
    F = filter.StateTransitionModel
    Qk = filter.ProcessNoise
    
    M = Qk.shape[0]
    Vzeros = np.zeros((M, 1))
    I = np.eye(M)

    # Measurement update
    u, V, B = mupdate(0, zmeas, u, P, Vzeros, zcov, H)

    # Convert influence diagram back to covariance form
    B = inf_to_cov(V, B, len(B))

    xcorr = u
    Pcorr = B

    return xcorr, Pcorr

# Test setup
k = 0
zmeas = np.array([0.0101])
u = np.array([[0.0101], [0.1188]])
P = np.array([[0.01071225, 0.017495523],
              [0.017495523, 2.04175521]])

V = np.array([0, 0])
zcov = np.array([0.01])
H = np.array([[1, 0]])
F = np.array([[1.0191, 0.0099],
              [-0.2474, 0.9994]])
Qk = np.array([[0.002, 0.002],
               [0.002, 0.438]])

# Instantiate the Kalman filter
filter = trackingKF(F, H, u, P, Qk)

# Test the id_correct function
xcorr, Pcorr = id_correct(filter, zmeas, zcov)

# Display the results with high precision
np.set_printoptions(precision=10)
print("Corrected State (xcorr):", xcorr)
print("Corrected Covariance (Pcorr):", Pcorr)