import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import os
np.random.seed(1234)
PATH = os.getcwd()

class CurveFit():
    
    def __init__(self, y0, Q, create):
        self.y0 = y0
        self.Q = Q
        self.create = create
        
    def func(self, y, k11, k12,
                      k21, k22,
                      k31, k32,
                      k41, k42,
                      k51, k52,
                      k61, k62,
                      e, w1, w2):
        
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
        f[5] = (e*self.Q + w1*T + w2)

        return f

    def get_file_data(self, file):
        df = pd.read_csv(file)
        t = df['t'].to_numpy().reshape(-1,1)
        if 'exp' in file.split('_')[0]:
            c = df.iloc[:,1:4].to_numpy()
            Q = np.ones(c.shape[0],dtype=int)*int(file[-6])
            Q = Q.reshape(-1,1)
            y = np.concatenate((t, c), axis=1)
            z = np.array([])
        if 'T' in file.split('_')[0]:
            T = df['T'].to_numpy().reshape(-1,1)
            Q = df['Q'].to_numpy().reshape(-1,1)
            z = np.concatenate((t, T), axis=1)
            y = np.array([])

        return y, z
    
    def get_data(self, files):
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
        
        self.T_mean = np.mean(self.zdata[:,-1])

        return self.ydata, self.zdata

    def make_idx(self, t):
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
        y_out = np.array([])
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
        
        if not self.create:
            ypred = y[self.idx_c,:3]
            zpred = y[self.idx_T,-1].reshape(-1,1)
            y_out = ypred.flatten()
            z_out = zpred.flatten()
            x_out = np.concatenate((y_out, z_out))
        
        if self.create:
            ypred = y[:,:3]
            zpred = y[:,-1].reshape(-1,1)
            x_out = np.concatenate((ypred, zpred), axis=1)
                
        return x_out

y0 = np.array([0.61911421,0.040004937,0.000394678,0.0,0.0,33.0])

# Data
curve = CurveFit(y0, 4, False)
files = ['data/exp1_4W.csv', #'data/exp1_5W.csv', 'data/exp1_6W.csv',
         'data/exp2_4W.csv', #'data/exp2_5W.csv', 'data/exp2_6W.csv',
         'data/exp3_4W.csv', #'data/exp3_5W.csv', 'data/exp3_6W.csv',
         'data/T1_4W.csv', #'data/T1_5W.csv', 'data/T1_6W.csv',
         'data/T2_4W.csv', #'data/T2_5W.csv', 'data/T2_6W.csv',
         'data/T3_4W.csv'] #'data/T3_5W.csv', 'data/T3_6W.csv']
yexp, zexp = curve.get_data(files)
ydata = yexp[:,1:].flatten()
zdata = zexp[:,1:].flatten()
ytrain = np.concatenate((ydata, zdata))

# Time
t = np.linspace(0,600,3001)
curve.make_idx(t)

# Fit
p0 = [1/240, 1e-4, 1/240, 1e-4, 1/240, 1e-4, 1/240, 1e-4, 1/240, 1e-4, 1/240, 1e-4, 1, -1, -1]
bounds = ((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -np.inf, -np.inf, -np.inf), (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf))
popt, pcov, infodict, mesg, ier = curve_fit(curve.ode, t, ytrain, p0=p0, bounds=bounds, method='trf', full_output=True, max_nfev=200)
print(popt)
print(infodict)
print(mesg)

# Pred
k11, k12, k21, k22, k31, k32, k41, k42, k51, k52, k61, k62, e, w1, w2 = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9], popt[10], popt[11], popt[12], popt[13], popt[14]
curve.create = True
xpred = curve.ode(t, k11, k12,
                     k21, k22,
                     k31, k32,
                     k41, k42,
                     k51, k52,
                     k61, k62,
                     e, w1, w2)

# Graphs
plt.plot(yexp[:,0], yexp[:,2], 'or', label='Expériences')
plt.plot(t, xpred[:,1], '-k', label='Prédictions')
plt.legend()
plt.xlabel('Temps (s)')
plt.ylabel('[DG] (mol/L)')
plt.savefig('main.png',dpi=600)
plt.show()

plt.plot(zexp[:,0], zexp[:,1], 'or')
plt.plot(t, xpred[:,-1], '-k')
plt.show()