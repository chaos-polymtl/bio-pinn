import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

class CurveFit():
    
    def __init__(self, y0, idx, create):
        self.y0 = y0
        self.idx = idx
        self.create = create
        self.rng = np.random.default_rng()
        
    def func(self, y, k1, k2):
        f = np.zeros(len(y))
        f[0] = - k1*y[0] + k2*y[1]
        f[1] = + k1*y[0] - k2*y[1]
        return f

    def ode(self, t, k1, k2):
        y_out = np.array([])
        for i in range(3):
            y = np.array([self.y0])
            for i in range(len(t)-1):
                dt = t[i+1] - t[i]
                c1 = dt * self.func(y[-1], k1, k2)
                c2 = dt * self.func(y[-1]+c1/2, k1, k2)
                c3 = dt * self.func(y[-1]+c2/2, k1, k2)
                c4 = dt * self.func(y[-1]+c3, k1, k2)
                y = np.append(y,
                            [y[-1] + 1/6 * (c1 + 2*c2 + 2*c3 + c4)],
                            axis=0)
            
            y = y[self.idx,:]
            y = y.flatten(order='F')
            if self.create:
                y_out = np.append(y_out, y + 0.02 * self.rng.normal(size=y.size))
            if not self.create:
                y_out = np.append(y_out, y)
        
        return y_out

y0 = np.array([1.0, 0.0])

# For solver
xdata = np.linspace(0,1,1001)
# From data
xtrain = np.linspace(0,1,11)

idx = []
for xi in xtrain:
    find = np.where(abs(xi - xdata) < 1e-8)[0][0]
    idx.append(int(find))

# Data to train on
curve = CurveFit(y0, idx, True)
ynoise = curve.ode(xdata, 5, 1)
curve.create = False

popt, pcov = curve_fit(curve.ode, xdata, ynoise)
print(popt)
print(pcov)