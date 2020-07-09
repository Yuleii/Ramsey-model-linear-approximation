
'''
Linear Approximation of a simple Ramsey model with exogenous growth of
technology and population.
'''

# Import Python Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def cal_steady_state(params):
    '''
    This function takes endogenous and exogenous parameters
    and returns the steady state values in the Euler equation.

    Parameters
    ----------
    alpha: float
        Curvature of production function.
    beta: float
        Discount factor.
    delta: float
        Depreciation rate.
    n: float
        Harrod-Neutral technological growth.
    g: float
        Population growth.
    sigma: float
        The degree of relative risk aversion.

    Returns
    -------
    kss_sharp: float
        Steady state value of normalized capital.
    css_sharp: float
        Steady state value of normalized consumption.
    '''
    # Unpack params.
    [alpha, beta, delta, n, g, sigma] = params

    # Normalized beta.
    betatilde = beta*((1.0+g)*(1.0+n))**(1.0-sigma)
    betatildetilde = betatilde/((1.0+g)*(1.0+n))

    # Compute the steady state.
    kss_sharp = ((1.0/betatildetilde-1.0+delta)/alpha)**(1.0/(alpha-1.0))
    css_sharp = kss_sharp**alpha-kss_sharp*(1.0+g)*(1.0+n)+(1.0-delta)*kss_sharp

    return kss_sharp, css_sharp


def simulate_model(kss_sharp, css_sharp, params, T):
    '''
    This function generate derivatives matrix A in linearized dynamic system and
    simulate the process of linear approximation. It returns vectors for
    consumption and capital deviations, consumption level, capital level, output
    levels and Capital-output-ratio against time.

    Parameters
    ----------
    css_sharp: float
        Value of normalized consumption at stead state.
    kss_sharp: float
        Value normalized capital at stead state.
    params: float
        List of parameter values.
    T: int
        The number of time periods to simulate


    Returns
    -------
    c_dev: array_like
        Consumption deviation over time.
    k_dev: array_like
        Capital deviation over time.
    C_level: array_like
        Consumption level over time.
    K_level: array_like
        Capital level over time.
    '''

    # Unpack params
    [alpha, beta, delta, n, g, sigma] = params

    # Normalized beta.
    betatilde = beta*((1+g)*(1+n))**(1-sigma)
    betatildetilde = betatilde/((1+g)*(1+n))

    # Compute coefficients for matrix_A.
    coeff1 = (-css_sharp*alpha*(alpha-1)*kss_sharp**(alpha-2))/((1+n)*(1+g))

    # Get entries of matrix_A.
    matrix_A = np.zeros((2, 2))
    matrix_A[0, 0] = 1+betatildetilde*coeff1
    matrix_A[0, 1] = -coeff1
    matrix_A[1, 0] = -1/((1+g)*(1+n))
    matrix_A[1, 1] = 1/(betatildetilde*(1+g)*(1+n))

    # Get eigenvectors and eigenvalues of matrix_A.
    D, V = np.linalg.eig(matrix_A)

    # Find out at which position the stable eigenvalue appears
    stab_col_ind = int(np.where(D < 1)[0])

    # Initialize vectors to store variables in deviations.
    x_sharp_dev = np.zeros((2, T))

    # Initial conditions.
    x_sharp_dev[1, 0] = -0.1*kss_sharp

    # Obtain deviative level.
    x_sharp_dev[0, 0] = (V[0, stab_col_ind]/V[1, stab_col_ind])*x_sharp_dev[1, 0]

    # Simulate.
    for t in range(1, T-1):
        x_sharp_dev[1, t] = np.dot(matrix_A[1, :], x_sharp_dev[:, t-1])
        x_sharp_dev[0, t] = (V[0, stab_col_ind]/V[1, stab_col_ind])*x_sharp_dev[1, t]

    # Obtain levels (normalized).
    x_sharp_level = x_sharp_dev+np.array([[css_sharp, kss_sharp], ]*50).transpose()

    # Obtain levels.
    X_level = x_sharp_level*((1+n)*(1+g))**np.array(list(range(T)))

    # Decompose vectors.
    c_sharp_dev = x_sharp_dev[0, :]
    k_sharp_dev = x_sharp_dev[1, :]
    C_level = X_level[0, :]
    K_level = X_level[1, :]

    # Obtain output Y.
    Y_level = x_sharp_level[1, :]**alpha*((1+n)*(1+g))**np.array(list(range(T)))

    return c_sharp_dev, k_sharp_dev, C_level, K_level, Y_level
