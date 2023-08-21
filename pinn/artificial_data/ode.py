# ============================================================================
# Numerical solver of ODEs using Runge-Kutta method
# Author : Valérie Bibeau, Polytechnique Montréal, 2023
# ============================================================================

# ---------------------------------------------------------------------------
# Library
import numpy as np
import torch
# ---------------------------------------------------------------------------

def edo(y, prm):
    """Ordinary differiential equation

    Args:
        y (numpy array): Value of dependant variables
        prm (struct): Kinetic parameters

    Returns:
        numpy array: Right hand side of ODEs
    """

    cA = y[0]
    cB = y[1]
    cC = y[2]
    cD = y[3]

    k1 = prm.k1
    k2 = prm.k2
    k3 = prm.k3
    k4 = prm.k4

    f = np.zeros(4)

    f[0] = - k1 * cA + k2 * cB * cC
    f[1] = + k1 * cA - k2 * cB * cC
    f[2] = + k1 * cA - k2 * cB * cC - k3 * cC + k4 * cD
    f[3] = + k3 * cC - k4 * cD

    return f

def runge_kutta(y0, prm, dt, tf):
    """Runge-Kutta method to solve ODEs

    Args:
        y0 (numpy array): Initial condition
        prm (struct): Kinetic parameters
        dt (float): Time step
        tf (float): Simulation time

    Returns:
        numpy array: Time and results
    """

    t = np.array([0])
    mat_y = np.array([y0])

    while t[-1] < tf:

        k1 = dt * edo(y0, prm)
        k2 = dt * edo(y0+k1/2, prm)
        k3 = dt * edo(y0+k2/2, prm)
        k4 = dt * edo(y0+k3, prm)

        y = y0 + 1/6*(k1 + 2*k2 + 2*k3 + k4)

        mat_y = np.append(mat_y, [y], axis=0)

        y0 = np.copy(y)

        t = np.append(t, [t[-1]+dt], axis=0)

    return t, mat_y

def add_noise(y, percentage):
    """Add noise to the artificial data

    Args:
        y (numpy array): Output tensor
        percentage (float): Relative error on data

    Returns:
        numpy array: Output array with noise applied
    """

    for i in range(y.shape[1]):
        noise = np.random.normal(0, y[1:,i].std(), y.shape[0]-1) * percentage
        y[1:,i] = y[1:,i] + noise

    return y

def make_idx(dt_pinn, dt_data, t_num, y_num, tf, collocation_points, percentage):
    """Find the right indexation of data points in collocation points

    Args:
        dt_pinn (float): Time step of collocation points
        dt_data (float): Time step of data points
        t_num (numpy array): Time
        y_num (numpy array): Numerical results
        tf (float): Simulation time
        collocation_points (int): Collocation points of the PINN model
        percentage (float): Relative error on data

    Raises:
        Exception: dt_pinn needs to be higher than dt_pinn!

    Returns:
        array and list: Input array, output array, list of indexation
    """

    if dt_data < dt_pinn:
        step = int(dt_pinn/dt_data)
    else:
        raise Exception("dt_pinn needs to be higher than dt_pinn!")
    
    y = y_num[::step,:]
    y = add_noise(np.copy(y), percentage)
    t = t_num[::step]

    X = np.linspace(0, tf, collocation_points+1)
    Y = np.zeros((collocation_points+1, y_num.shape[1]))

    vec_idx = []
    for ti in t:
        w = np.where(np.abs(ti - X) < 1e-6)
        idx = w[0][0]
        vec_idx.append(idx)
    
    Y[vec_idx] = y

    return X, Y, vec_idx

def put_in_device(X, Y, device):
    """Put tensors in device

    Args:
        X (numpy array): Input array
        Y (numpy array): Output array
        device (string): CPU or CUDA

    Returns:
        tensor: Input and output tensors
    """

    X = X.reshape(-1,1)

    X = torch.from_numpy(X).float().to(device)
    Y = torch.from_numpy(Y).float().to(device)

    return X, Y