import numpy as np
from Mupdate import mupdate
from INFtoCOV import inf_to_cov

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

