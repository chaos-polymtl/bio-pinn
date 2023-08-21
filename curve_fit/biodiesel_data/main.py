# ============================================================================
# Non linear regression method (for comparison with PINN) using Scipy
# Goal : Predict the kinetic constants of a microwave-assisted biodiesel process.
# Author : Valérie Bibeau, Polytechnique Montréal, 2023
# PINN with 1 feature (time) and 4 outputs.
# ============================================================================

# ---------------------------------------------------------------------------
# Imports
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import os
np.random.seed(1234)
PATH = os.getcwd()
# ----------------------------------------------------------------------------

class CurveFit():
    
    def __init__(self, y0, create):
        """Constructor

        Args:
            y0 (array): Initial condition
            idx (list): Indexation of data points in collocation points
            create (bool): If True, create the database,
                           if False, use it for curve_fit (scipy)
        """
        self.y0 = y0
        self.create = create
        self.EVALUATE_Q = None
        
    def func(self, y, k11, k12,
                      k21, k22,
                      k31, k32,
                      k41, k42,
                      k51, k52,
                      k61, k62,
                      e, w1, w2):
        """Right hand side of the ODEs

        Args:
            y (array): Values of dependant variables (concentrations)
            All constants k (float): Kinetic parameters

        Returns:
            array: Evaluation of the right hand side
        """
        f = np.zeros(len(y))

        cTG = y[0]
        cDG = y[1]
        cMG = y[2]
        cG = y[3]
        cME = y[4]
        T = y[5]

        k1 = k11 + k12*(T - self.T_mean)
        k2 = k21 + k22*(T - self.T_mean)
        k3 = k31 + k32*(T - self.T_mean)
        k4 = k41 + k42*(T - self.T_mean)
        k5 = k51 + k52*(T - self.T_mean)
        k6 = k61 + k62*(T - self.T_mean)

        f[0] = (+ k1*cTG - k2*cDG*cME)*-1
        f[1] = (- k1*cTG + k2*cDG*cME + k3*cDG - k4*cMG*cME)*-1
        f[2] = (- k3*cDG + k4*cMG*cME + k5*cMG - k6*cG*cME)*-1
        f[3] = (- k5*cMG + k6*cG*cME)*-1
        f[4] = (- k1*cTG + k2*cDG*cME - k3*cDG + k4*cMG*cME - k5*cMG + k6*cG*cME)*-1
        f[5] = (e*self.Q + w1*T + w2)/10

        return f

    def get_file_data(self, file):
        """Get experimental data points

        Args:
            file (string): Name of file of data points

        Returns:
            array: Portion of database
        """
        df = pd.read_csv(file)
        t = df['t'].to_numpy().reshape(-1,1)
        if 'exp' in file.split('_')[0]:
            c = df.iloc[:,1:4].to_numpy()
            Q = np.ones(c.shape[0],dtype=int)*int(file[-6])
            Q = Q.reshape(-1,1)
            y = np.concatenate((t, c, Q), axis=1)
            z = np.array([])
        if 'T' in file.split('_')[0]:
            T = df['T'].to_numpy().reshape(-1,1)
            Q = df['Q'].to_numpy().reshape(-1,1)
            z = np.concatenate((t, T, Q), axis=1)
            y = np.array([])

        return y, z
    
    def get_data(self, files):
        """Concatenate all data

        Args:
            files (list of string): All data files

        Returns:
            array: Full database
        """
        for file in files:
            y, z = self.get_file_data(file)
            try:
                if len(z) == 0:
                    self.ydata = np.append(self.ydata, y, axis=0)
                elif len(y) == 0:
                    self.zdata = np.append(self.zdata, z, axis=0)
            except:
                if len(z) == 0:
                    self.ydata = np.copy(y)
                elif len(y) == 0:
                    self.zdata = np.copy(z)
        
        self.T_mean = np.mean(self.zdata[:,-2])

        return self.ydata, self.zdata

    def make_idx(self, t):
        """Indexation of data points within collocation points

        Args:
            t (array): Time

        Returns:
            list: Indexation
        """
        self.idx_c = []
        self.idx_T = []
        for ti in self.ydata[:,0]:
            find = list(np.where(abs(ti - t) < 1e-8)[0])
            self.idx_c += find
        for ti in self.zdata[:,0]:
            find = list(np.where(abs(ti - t) < 1e-8)[0])
            self.idx_T += find
        return self.idx_c, self.idx_T

    def ode(self, t, k11, k12,
                     k21, k22,
                     k31, k32,
                     k41, k42,
                     k51, k52,
                     k61, k62,
                     e, w1, w2):
        """Molar balances

        Args:
            t (array): Time
            All constants k (float): Kinetic parameters

        Returns:
            array: Solution of the ODEs
        """
        y_Q = np.empty((3*t.size,len(self.y0)))
        j = 0
        for Q in [4, 5, 6]:
            self.Q = Q
            y = np.array([self.y0])
            for i in range(len(t)-1):
                dt = t[i+1] - t[i]
                c1 = dt * self.func(y[-1], k11, k12,
                                        k21, k22,
                                        k31, k32,
                                        k41, k42,
                                        k51, k52,
                                        k61, k62,
                                        e, w1, w2)
                c2 = dt * self.func(y[-1]+c1/2, k11, k12,
                                                k21, k22,
                                                k31, k32,
                                                k41, k42,
                                                k51, k52,
                                                k61, k62,
                                                e, w1, w2)
                c3 = dt * self.func(y[-1]+c2/2, k11, k12,
                                                k21, k22,
                                                k31, k32,
                                                k41, k42,
                                                k51, k52,
                                                k61, k62,
                                                e, w1, w2)
                c4 = dt * self.func(y[-1]+c3, k11, k12,
                                            k21, k22,
                                            k31, k32,
                                            k41, k42,
                                            k51, k52,
                                            k61, k62,
                                            e, w1, w2)
                y = np.append(y,
                                [y[-1] + 1/6 * (c1 + 2*c2 + 2*c3 + c4)],
                                axis=0)
                
            y_Q[j:j+t.size,:] = y
            j += t.size
            
            if self.create and self.Q == self.EVALUATE_Q:
                ypred = y[:,:3]
                zpred = y[:,-1].reshape(-1,1)
                x_out = np.concatenate((ypred, zpred), axis=1)
        
        if not self.create:
            ypred = y_Q[self.idx_c,:3]
            zpred = y_Q[self.idx_T,-1].reshape(-1,1)
            y_out = ypred.flatten()
            z_out = zpred.flatten()
            x_out = np.concatenate((y_out, z_out))
                
        return x_out

# Initial conditions
y0 = np.array([0.61911421,0.040004937,0.000394678,0.0,0.0,33.0])

# Data
curve = CurveFit(y0, False)
files = np.array([['data/exp1_4W.csv', 'data/exp1_5W.csv', 'data/exp1_6W.csv'],
         ['data/exp2_4W.csv', 'data/exp2_5W.csv', 'data/exp2_6W.csv'],
         ['data/exp3_4W.csv', 'data/exp3_5W.csv', 'data/exp3_6W.csv'],
         ['data/T1_4W.csv', 'data/T1_5W.csv', 'data/T1_6W.csv'],
         ['data/T2_4W.csv', 'data/T2_5W.csv', 'data/T2_6W.csv'],
         ['data/T3_4W.csv', 'data/T3_5W.csv', 'data/T3_6W.csv']])
y_train = np.array([])
for i in range(files.shape[1]):
    yexp, zexp = curve.get_data(files[:,i])
    ydata = yexp[:,1:-1].flatten()
    zdata = zexp[:,1:-1].flatten()
    ytrain = np.concatenate((y_train, ydata, zdata))

# Time
t = np.linspace(0,600,3001)
curve.make_idx(t)

# Fit
p0 = [1/240, 1e-4, 1/240, 1e-4, 1/240, 1e-4, 1/240, 1e-4, 1/240, 1e-4, 1/240, 1e-4, 0.5, -0.1, -0.1]
bounds = ((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -np.inf, -np.inf, -np.inf), (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf))
popt, pcov = curve_fit(curve.ode, t, ytrain, p0=p0, bounds=bounds, method='trf')
print(popt)

# Predictions
k11, k12, k21, k22, k31, k32, k41, k42, k51, k52, k61, k62, e, w1, w2 = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9], popt[10], popt[11], popt[12], popt[13], popt[14]
curve.create = True

curve.EVALUATE_Q = 4
xpred_4W = curve.ode(np.linspace(0,600,3001), k11, k12,
                        k21, k22,
                        k31, k32,
                        k41, k42,
                        k51, k52,
                        k61, k62,
                        e, w1, w2)

curve.EVALUATE_Q = 5
xpred_5W = curve.ode(np.linspace(0,360,1801), k11, k12,
                        k21, k22,
                        k31, k32,
                        k41, k42,
                        k51, k52,
                        k61, k62,
                        e, w1, w2)

curve.EVALUATE_Q = 6
xpred_6W = curve.ode(np.linspace(0,240,1201), k11, k12,
                        k21, k22,
                        k31, k32,
                        k41, k42,
                        k51, k52,
                        k61, k62,
                        e, w1, w2)

# Graphs
plt.plot(yexp[yexp[:,-1]==4][:,0], yexp[yexp[:,-1]==4][:,2], 'ob', label='Expériences 4W')
plt.plot(yexp[yexp[:,-1]==5][:,0], yexp[yexp[:,-1]==5][:,2], 'or', label='Expériences 5W')
plt.plot(yexp[yexp[:,-1]==6][:,0], yexp[yexp[:,-1]==6][:,2], 'ok', label='Expériences 6W')
plt.plot(np.linspace(0,600,3001), xpred_4W[:,1], '-b', label='TRF 4W')
plt.plot(np.linspace(0,360,1801), xpred_5W[:,1], '-r', label='TRF 5W')
plt.plot(np.linspace(0,240,1201), xpred_6W[:,1], '-k', label='TRF 6W')
plt.legend()
plt.xlabel('Temps (s)')
plt.ylabel('[DG] (mol/L)')
plt.show()

plt.plot(zexp[zexp[:,-1]==4][:,0], zexp[zexp[:,-1]==4][:,1], 'ob', label='Expériences 4W')
plt.plot(zexp[zexp[:,-1]==5][:,0], zexp[zexp[:,-1]==5][:,1], 'or', label='Expériences 5W')
plt.plot(zexp[zexp[:,-1]==6][:,0], zexp[zexp[:,-1]==6][:,1], 'ok', label='Expériences 6W')
plt.plot(np.linspace(0,600,3001), xpred_4W[:,-1], '-b', label='TRF 4W')
plt.plot(np.linspace(0,360,1801), xpred_5W[:,-1], '-r', label='TRF 5W')
plt.plot(np.linspace(0,240,1201), xpred_6W[:,-1], '-k', label='TRF 6W')
plt.show()