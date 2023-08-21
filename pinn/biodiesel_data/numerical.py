# ============================================================================
# Numerical solver
# Author : Valérie Bibeau, Polytechnique Montréal, 2023
# ============================================================================

# ---------------------------------------------------------------------------
# Libraries importation
import numpy as np
# ---------------------------------------------------------------------------

def EDO(y, prm, i):
    """Right hand side of ODEs

    Args:
        y (array): Values of independant variables (concentrations)
        prm (struc): Extra parameters
        i (int): Integer

    Returns:
        array: Evaluation of the right hand side
    """

    f = np.zeros(6)

    cTG = y[0]
    cDG = y[1]
    cMG = y[2]
    cG = y[3]
    cME = y[4]
    Ti = prm.T[i]

    k1 = prm.A1 + prm.E1 * (Ti - prm.T_mean)
    k2 = prm.A2 + prm.E2 * (Ti - prm.T_mean)
    k3 = prm.A3 + prm.E3 * (Ti - prm.T_mean)
    k4 = prm.A4 + prm.E4 * (Ti - prm.T_mean)
    k5 = prm.A5 + prm.E5 * (Ti - prm.T_mean)
    k6 = prm.A6 + prm.E6 * (Ti - prm.T_mean)

    # k1 = prm.A1 * np.exp(-prm.E1 / Ti)
    # k2 = prm.A2 * np.exp(-prm.E2 / Ti)
    # k3 = prm.A3 * np.exp(-prm.E3 / Ti)
    # k4 = prm.A4 * np.exp(-prm.E4 / Ti)
    # k5 = prm.A5 * np.exp(-prm.E5 / Ti)
    # k6 = prm.A6 * np.exp(-prm.E6 / Ti)

    f[0] = - k1 * cTG + k2 * cDG * cME
    f[1] = + k1 * cTG - k2 * cDG * cME - k3 * cDG + k4 * cMG * cME
    f[2] = + k3 * cDG - k4 * cMG * cME - k5 * cMG + k6 * cG * cME
    f[3] = + k5 * cMG - k6 * cG * cME
    f[4] = + k1 * cTG - k2 * cDG * cME + k3 * cDG - k4 * cMG * cME + k5 * cMG - k6 * cG * cME
    f[5] = (prm.e*prm.Q + prm.c1*Ti + prm.c2) / prm.m_Cp

    return f

def runge_kutta(y0, t, prm):
    """Runge-Kutta method

    Args:
        y0 (array): Initial conditions
        t (array): Time
        prm (struct): Extra parameters

    Returns:
        array: Time and numerical solutions of the ODEs
    """

    mat_y = np.array([y0])

    for i in range(len(t)-1):
        dt = t[i+1] - t[i]
        
        k1 = dt * EDO(y0, prm, i)
        k2 = dt * EDO(y0+k1/2, prm, i)
        k3 = dt * EDO(y0+k2/2, prm, i)
        k4 = dt * EDO(y0+k3, prm, i)
        y = y0 + 1/6*(k1 + 2*k2 + 2*k3 + k4)

        mat_y = np.append(mat_y, [y], axis=0)

        y0 = np.copy(y)

    return t, mat_y