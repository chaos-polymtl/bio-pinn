# ============================================================================
# Non linear regression method with artificial data using Scipy
# Author : Valérie Bibeau, Polytechnique Montréal, 2023
# PINN with 1 feature (time) and 4 outputs.
# ============================================================================

# ---------------------------------------------------------------------------
# Imports
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
np.random.seed(1234)
# ----------------------------------------------------------------------------

class CurveFit():
    
    def __init__(self, y0, idx, create):
        """Constructor

        Args:
            y0 (array): Initial condition
            idx (list): Indexation of data points in collocation points
            create (bool): If True, create the database,
                           if False, use it for curve_fit (scipy)
        """
        self.y0 = y0
        self.idx = idx
        self.create = create
        
    def func(self, y, k1, k2, k3, k4):
        """Right hand side of the ODEs

        Args:
            y (array): Values of dependant variables (concentrations)
            k1, k2, k3, k4 (float): Kinetic constants

        Returns:
            array: Evaluation of the right hand side
        """
        f = np.zeros(len(y))
        cA = y[0]
        cB = y[1]
        cC = y[2]
        cD = y[3]
        f[0] = - k1*cA + k2*cB*cC
        f[1] = + k1*cA - k2*cB*cC
        f[2] = + k1*cA - k2*cB*cC - k3*cC + k4*cD
        f[3] = + k3*cC - k4*cD
        return f

    def ode(self, t, k1, k2, k3, k4):
        """Molar balances

        Args:
            t (array): Time
            k1, k2, k3, k4 (float): Kinetic parameters

        Returns:
            array: Solution of the ODEs
        """
        y_out = np.array([])
        for i in range(1):
            y = np.array([self.y0])
            for i in range(len(t)-1):
                dt = t[i+1] - t[i]
                c1 = dt * self.func(y[-1], k1, k2, k3, k4)
                c2 = dt * self.func(y[-1]+c1/2, k1, k2, k3, k4)
                c3 = dt * self.func(y[-1]+c2/2, k1, k2, k3, k4)
                c4 = dt * self.func(y[-1]+c3, k1, k2, k3, k4)
                y = np.append(y,
                              [y[-1] + 1/6 * (c1 + 2*c2 + 2*c3 + c4)],
                              axis=0)
            
            if self.create == 0:
                y = y[self.idx,:]
                for i in range(y.shape[1]):
                    noise = 0.2 * np.random.normal(0, y[1:,i].std(), y.shape[0]-1)
                    y[1:,i] = y[1:,i] + noise
                y = y.flatten()
                y_out = np.append(y_out, y)
            if self.create == 1:
                y = y[self.idx,:]
                y = y.flatten()
                y_out = np.append(y_out, y)
            if self.create == 2:
                y_out = np.append(y_out, y)
                
        return y_out

# Initial conditions
y0 = np.array([1.0, 0.0, 0.2, 0.0])

# For solver
xdata = np.linspace(0,10,101)
# From data
xtrain = np.linspace(0,10,6)

idx = []
for xi in xtrain:
    find = np.where(abs(xi - xdata) < 1e-8)[0][0]
    idx.append(int(find))

# Data to train on
curve = CurveFit(y0, idx, 0)
ynoise = curve.ode(xdata, 1.5, 0.5, 1.0, 0.1)

# True data
curve.create = 2
ytrue = curve.ode(xdata, 1.5, 0.5, 1.0, 0.1)

# Fit
curve.create = 1
p0 = np.array([1., 1., 1., 1.])
popt, pcov = curve_fit(curve.ode, xdata, ynoise, p0=p0, bounds=(0, np.inf), method='trf')
print(popt)
print(pcov)

# Regression constants
curve.create = 2
ypred = curve.ode(xdata, popt[0], popt[1], popt[2], popt[3])

# Error on all data
print(mean_absolute_error(ytrue, ypred))

# Graphs
plt.plot(xdata, ytrue.reshape((101,4))[:,0], '--r')
plt.plot(xdata, ytrue.reshape((101,4))[:,1], '--b')
plt.plot(xdata, ytrue.reshape((101,4))[:,2], '--k')
plt.plot(xdata, ytrue.reshape((101,4))[:,3], '--c')
plt.plot(xdata, ypred.reshape((101,4))[:,0], '-r')
plt.plot(xdata, ypred.reshape((101,4))[:,1], '-b')
plt.plot(xdata, ypred.reshape((101,4))[:,2], '-k')
plt.plot(xdata, ypred.reshape((101,4))[:,3], '-c')
plt.plot(xtrain, ynoise.reshape((6,4))[:,0], 'or', label='[A]')
plt.plot(xtrain, ynoise.reshape((6,4))[:,1], 'ob', label='[B]')
plt.plot(xtrain, ynoise.reshape((6,4))[:,2], 'ok', label='[C]')
plt.plot(xtrain, ynoise.reshape((6,4))[:,3], 'oc', label='[D]')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Concentration (mol/L)')
plt.show()