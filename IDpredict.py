import numpy as np
from COVtoINF import cov_to_inf
from Tupdate import tupdate
from INFtoCOV import inf_to_cov

def id_predict(filter):
    """
    Performs the Time Update (prediction) portion of the Kalman Filter for object-oriented programming.

    Parameters:
    filter (object): A trackingKF object that contains State, StateCovariance, MeasurementModel, StateTransitionModel, and ProcessNoise.

    Returns:
    xpred (numpy.ndarray): The predicted state.
    Ppred (numpy.ndarray): The predicted state estimation error covariance.
    """

    # Extract filter parameters
    u = filter.State
    B_old = filter.StateCovariance
    H = filter.MeasurementModel
    F = filter.StateTransitionModel
    Qk = filter.ProcessNoise
    
    M = Qk.shape[0]
    I = np.eye(M)

    # Convert covariance to influence diagram form
    B_old, V_old, Precision = cov_to_inf(B_old, M)

    # Perform time update
    u, B, V = tupdate(u, B_old, V_old, F, I, Qk)

    # Convert influence diagram back to covariance form
    Ppred = inf_to_cov(V, B, M)

    xpred = u

    return xpred, Ppred
