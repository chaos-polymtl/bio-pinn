# Importation de librairies
import torch
import numpy as np
import pandas as pd

# Set seed
torch.set_default_dtype(torch.float)
torch.manual_seed(1234)
np.random.seed(1234)

# Read data
def read_data(file):
    data = pd.read_csv('data/'+file, sep=',')
    data = data.replace(np.nan, 0.)
    
    C = data.to_numpy()

    return C

def find_idx_C(t, C):
    idx = []
    t_data = C[:,0]
    for ti in t_data:
        idx.append(np.where(t == ti)[0][0])
        
    return idx

def find_idx_T(t, T):
    idx = []
    t_data = T[:,1]
    for ti in t_data:
        idx.append(np.where(t == ti)[0][0])
        
    return idx

def put_in_device(x, y, z, device):

    X = torch.from_numpy(x).float().to(device)
    Y = torch.from_numpy(y).float().to(device)
    Z = torch.from_numpy(z).float().to(device)

    return X, Y, Z

def gather_data(files, T_files):
    
    C = read_data(files[0])
    P = float(files[0].split('_')[1].split('W')[0])
    t = np.linspace(0,int(C[-1,0]),int(C[-1,0]*2+1)).reshape(-1,1)
    T = read_data(T_files[0])
    Q = np.ones(t.shape)*P

    idx_C = find_idx_C(t, C)
    idx_T = find_idx_T(t, T)
    idx_y0 = [0]

    X = np.concatenate((t, Q), axis=1)
    Y = np.zeros((X.shape[0], C[:,1:].shape[1]))
    Z = np.zeros((X.shape[0], 1))
    for i in range(Y.shape[1]):
        Y[idx_C,i] = C[:,i+1]
        if i == 3:
            Z[idx_T,0] = T[:,i]

    len_t = len(t)
    for i in range(1,len(files)):
        new_C = read_data(files[i])
        P = float(files[i].split('_')[1].split('W')[0])
        new_T = read_data(T_files[i])
        new_t = np.linspace(0,int(new_C[-1,0]),int(new_C[-1,0]*2+1)).reshape(-1,1)
        new_Q = np.ones(new_t.shape)*P

        new_idx_C = find_idx_C(new_t, new_C)
        new_idx_T = find_idx_T(new_t, new_T)

        new_X = np.concatenate((new_t, new_Q), axis=1)
        X = np.concatenate((X, new_X), axis=0)
        new_Y = np.zeros((new_X.shape[0], new_C[:,1:].shape[1]))
        new_Z = np.zeros((new_X.shape[0], 1))
        for k in range(new_Y.shape[1]):
            new_Y[new_idx_C,k] = new_C[:,k+1]
            if k == 3:
                new_Z[new_idx_T,0] = new_T[:,k]
        Y = np.concatenate((Y, new_Y), axis=0)
        Z = np.concatenate((Z, new_Z), axis=0)

        for j in range(len(new_idx_C)):
            new_idx_C[j] = new_idx_C[j] + len_t
        for j in range(len(new_idx_T)):
            new_idx_T[j] = new_idx_T[j] + len_t
        idx_C = idx_C + new_idx_C
        idx_T = idx_T + new_idx_T
        idx_y0 = idx_y0 + [len_t]
        len_t += len(new_t)
        
    return X, Y, Z, idx_C, idx_T, idx_y0