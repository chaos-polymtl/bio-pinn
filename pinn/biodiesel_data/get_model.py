import os
import shutil
import pandas as pd

PATH = os.getcwd()

best_case = 'pinn'

os.chdir(f'{PATH}/{best_case}')
shutil.copy(f'{PATH}/{best_case}/pinn.py', f'{PATH}/')
shutil.copy(f'{PATH}/{best_case}/data.py', f'{PATH}/')

from pinn import *
from data import *
from numerical import *
import torch
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# Modèle
# --------------------------------------------------------------
model = torch.load('model.pt', map_location=torch.device('cpu'))

files = ['exp1_6W.csv', 'exp2_6W.csv', 'exp3_6W.csv',
         'exp1_5W.csv', 'exp2_5W.csv', 'exp3_5W.csv',
         'exp1_4W.csv', 'exp2_4W.csv', 'exp3_4W.csv']

T_files = ['T1_6W.csv', 'T2_6W.csv', 'T3_6W.csv',
           'T1_5W.csv', 'T2_5W.csv', 'T3_5W.csv',
           'T1_4W.csv', 'T2_4W.csv', 'T3_4W.csv']

X, Y, Z, idx, idx_T, idx_y0 = gather_data(files, T_files)

device = torch.device('cpu')
X_train, Y_train, Z_train = put_in_device(X, Y, Z, device)

output = model(X_train)

# --------------------------------------------------------------
# Extraction des données
# --------------------------------------------------------------

dict_output = {}
dict_output['t'] = X_train[:,0].detach().numpy()
dict_output['Q'] = X_train[:,1].detach().numpy()
dict_output['TG'] = output[:,0].detach().numpy()
dict_output['DG'] = output[:,1].detach().numpy()
dict_output['MG'] = output[:,2].detach().numpy()
dict_output['G'] = output[:,3].detach().numpy()
dict_output['ME'] = output[:,4].detach().numpy()
dict_output['T'] = output[:,5].detach().numpy()

df_output = pd.DataFrame.from_dict(dict_output)

df_4W = df_output[df_output['Q'] == 4.0]
df_4W = df_4W.drop_duplicates(subset=['t', 'Q'])
df_4W = df_4W.sort_values('t')

df_5W = df_output[df_output['Q'] == 5.0]
df_5W = df_5W.drop_duplicates(subset=['t', 'Q'])
df_5W = df_5W.sort_values('t')

df_6W = df_output[df_output['Q'] == 6.0]
df_6W = df_6W.drop_duplicates(subset=['t', 'Q'])
df_6W = df_6W.sort_values('t')

y0 = np.array([0.61911421,0.040004937,0.000394678,0.0,0.0,
               np.min(df_6W['T'].to_numpy())])
class parameters():
    e = float(model.e.detach().numpy())
    c1 = float(model.c1.detach().numpy())
    c2 = float(model.c2.detach().numpy())
    A1 = float(model.A1.detach().numpy())
    E1 = float(model.E1.detach().numpy())
    A2 = float(model.A2.detach().numpy())
    E2 = float(model.E2.detach().numpy())
    A3 = float(model.A3.detach().numpy())
    E3 = float(model.E3.detach().numpy())
    A4 = float(model.A4.detach().numpy())
    E4 = float(model.E4.detach().numpy())
    A5 = float(model.A5.detach().numpy())
    E5 = float(model.E5.detach().numpy())
    A6 = float(model.A6.detach().numpy())
    E6 = float(model.E6.detach().numpy())
    T = df_6W['T'].to_numpy()
    T_mean = np.mean(output[:,5].detach().numpy())
    Q = 6
    m_Cp = 10
    
prm = parameters()

t_num, y_num = euler(y0, df_6W['t'].to_numpy(), prm)
y0 = np.array([0.61911421,0.040004937,0.000394678,0.0,0.0,
               np.min(df_5W['T'].to_numpy())])

prm.Q = 5
prm.T = df_5W['T'].to_numpy()
t_num_5, y_num_5 = euler(y0, df_5W['t'].to_numpy(), prm)
y0 = np.array([0.61911421,0.040004937,0.000394678,0.0,0.0,
               np.min(df_4W['T'].to_numpy())])

prm.Q = 4
prm.T = df_4W['T'].to_numpy()
t_num_4, y_num_4 = euler(y0, df_4W['t'].to_numpy(), prm)

# --------------------------------------------------------------
# Graphique
# --------------------------------------------------------------

plt.plot(X_train[idx[:12],0], Y_train[idx[:12],0], 'o', label='Experiments 6W')
plt.plot(X_train[idx[12:27],0], Y_train[idx[12:27],0], 'o', label='Experiments 5W')
plt.plot(X_train[idx[27:45],0], Y_train[idx[27:45],0], 'o', label='Experiments 4W')
plt.plot(df_6W['t'].to_numpy(), df_6W['TG'].to_numpy(), 'o', markersize=1, label='PINN 6W')
plt.plot(df_5W['t'].to_numpy(), df_5W['TG'].to_numpy(), 'o', markersize=1, label='PINN 5W')
plt.plot(df_4W['t'].to_numpy(), df_4W['TG'].to_numpy(), 'o', markersize=1, label='PINN 4W')
plt.plot(t_num, y_num[:,0], '--', label='Numerical 6W')
plt.plot(t_num_5, y_num_5[:,0], '--', label='Numerical 5W')
plt.plot(t_num_4, y_num_4[:,0], '--', label='Numerical 4W')
plt.xlabel('Time [sec]')
plt.ylabel('TG Concentration [mol/L]')
plt.legend()
plt.show()

plt.plot(X_train[idx[:12],0], Y_train[idx[:12],1], 'o', label='Experiments 6W')
plt.plot(X_train[idx[12:27],0], Y_train[idx[12:27],1], 'o', label='Experiments 5W')
plt.plot(X_train[idx[27:45],0], Y_train[idx[27:45],1], 'o', label='Experiments 4W')
plt.plot(df_6W['t'].to_numpy(), df_6W['DG'].to_numpy(), 'o', markersize=1, label='PINN 6W')
plt.plot(df_5W['t'].to_numpy(), df_5W['DG'].to_numpy(), 'o', markersize=1, label='PINN 5W')
plt.plot(df_4W['t'].to_numpy(), df_4W['DG'].to_numpy(), 'o', markersize=1, label='PINN 4W')
plt.plot(t_num, y_num[:,1], '--', label='Numerical 6W')
plt.plot(t_num_5, y_num_5[:,1], '--', label='Numerical 5W')
plt.plot(t_num_4, y_num_4[:,1], '--', label='Numerical 4W')
plt.xlabel('Time [sec]')
plt.ylabel('DG Concentration [mol/L]')
plt.legend()
plt.show()

plt.plot(X_train[idx[:12],0], Y_train[idx[:12],2], 'o', label='Experiments 6W')
plt.plot(X_train[idx[12:27],0], Y_train[idx[12:27],2], 'o', label='Experiments 5W')
plt.plot(X_train[idx[27:45],0], Y_train[idx[27:45],2], 'o', label='Experiments 4W')
plt.plot(df_6W['t'].to_numpy(), df_6W['MG'].to_numpy(), 'o', markersize=1, label='PINN 6W')
plt.plot(df_5W['t'].to_numpy(), df_5W['MG'].to_numpy(), 'o', markersize=1, label='PINN 5W')
plt.plot(df_4W['t'].to_numpy(), df_4W['MG'].to_numpy(), 'o', markersize=1, label='PINN 4W')
plt.plot(t_num, y_num[:,2], '--', label='Numerical 6W')
plt.plot(t_num_5, y_num_5[:,2], '--', label='Numerical 5W')
plt.plot(t_num_4, y_num_4[:,2], '--', label='Numerical 4W')
plt.xlabel('Time [sec]')
plt.ylabel('MG Concentration [mol/L]')
plt.legend()
plt.show()

plt.plot(df_6W['t'].to_numpy(), df_6W['G'].to_numpy(), 'o', markersize=1, label='PINN 6W')
plt.plot(df_5W['t'].to_numpy(), df_5W['G'].to_numpy(), 'o', markersize=1, label='PINN 5W')
plt.plot(df_4W['t'].to_numpy(), df_4W['G'].to_numpy(), 'o', markersize=1, label='PINN 4W')
plt.plot(t_num, y_num[:,3], '--', label='Numerical 6W')
plt.plot(t_num_5, y_num_5[:,3], '--', label='Numerical 5W')
plt.plot(t_num_4, y_num_4[:,3], '--', label='Numerical 4W')
plt.xlabel('Time [sec]')
plt.ylabel('G Concentration [mol/L]')
plt.legend()
plt.show()

plt.plot(df_6W['t'].to_numpy(), df_6W['ME'].to_numpy(), 'o', markersize=1, label='PINN 6W')
plt.plot(df_5W['t'].to_numpy(), df_5W['ME'].to_numpy(), 'o', markersize=1, label='PINN 5W')
plt.plot(df_4W['t'].to_numpy(), df_4W['ME'].to_numpy(), 'o', markersize=1, label='PINN 4W')
plt.plot(t_num, y_num[:,4], '--', label='Numerical 6W')
plt.plot(t_num_5, y_num_5[:,4], '--', label='Numerical 5W')
plt.plot(t_num_4, y_num_4[:,4], '--', label='Numerical 4W')
plt.xlabel('Time [sec]')
plt.ylabel('ME Concentration [mol/L]')
plt.legend()
plt.show()

plt.plot(X_train[idx_T,0], Z_train[idx_T,0], 'o', markersize=1, label='Experiments')
plt.plot(df_6W['t'].to_numpy(), df_6W['T'].to_numpy(), 'o', markersize=1, label='PINN 6W')
plt.plot(df_5W['t'].to_numpy(), df_5W['T'].to_numpy(), 'o', markersize=1, label='PINN 5W')
plt.plot(df_4W['t'].to_numpy(), df_4W['T'].to_numpy(), 'o', markersize=1, label='PINN 4W')
plt.xlabel('Time [sec]')
plt.ylabel(r'Temperature [$\degree$C]')
plt.legend()
plt.show()

print(f'e: {float(model.e.detach().numpy())}')
print(f'c1: {float(model.c1.detach().numpy())}')
print(f'c2: {float(model.c2.detach().numpy())}')
print(f'A1: {float(model.A1.detach().numpy())}')
print(f'E1: {float(model.E1.detach().numpy())}')
print(f'A2: {float(model.A2.detach().numpy())}')
print(f'E2: {float(model.E2.detach().numpy())}')
print(f'A3: {float(model.A3.detach().numpy())}')
print(f'E3: {float(model.E3.detach().numpy())}')
print(f'A4: {float(model.A4.detach().numpy())}')
print(f'E4: {float(model.E4.detach().numpy())}')
print(f'A5: {float(model.A5.detach().numpy())}')
print(f'E5: {float(model.E5.detach().numpy())}')
print(f'A6: {float(model.A6.detach().numpy())}')
print(f'E6: {float(model.E6.detach().numpy())}')

# --------------------------------------------------------------
# Génération des résultats dans des bases de données
# --------------------------------------------------------------

data_PINN_4W = {}
data_PINN_4W['t'] = df_4W['t'].to_numpy()
data_PINN_4W['Q'] = df_4W['Q'].to_numpy()
data_PINN_4W['TG'] = df_4W['TG'].to_numpy()
data_PINN_4W['DG'] = df_4W['DG'].to_numpy()
data_PINN_4W['MG'] = df_4W['MG'].to_numpy()
data_PINN_4W['G'] = df_4W['G'].to_numpy()
data_PINN_4W['ME'] = df_4W['ME'].to_numpy()
df_PINN_4W = pd.DataFrame.from_dict(data_PINN_4W)
df_PINN_4W.to_csv(PATH+'/'+best_case+'/results/'+'data_PINN_4W.csv')

data_PINN_5W = {}
data_PINN_5W['t'] = df_5W['t'].to_numpy()
data_PINN_5W['Q'] = df_5W['Q'].to_numpy()
data_PINN_5W['TG'] = df_5W['TG'].to_numpy()
data_PINN_5W['DG'] = df_5W['DG'].to_numpy()
data_PINN_5W['MG'] = df_5W['MG'].to_numpy()
data_PINN_5W['G'] = df_5W['G'].to_numpy()
data_PINN_5W['ME'] = df_5W['ME'].to_numpy()
df_PINN_5W = pd.DataFrame.from_dict(data_PINN_5W)
df_PINN_5W.to_csv(PATH+'/'+best_case+'/results/'+'data_PINN_5W.csv')

data_PINN_6W = {}
data_PINN_6W['t'] = df_6W['t'].to_numpy()
data_PINN_6W['Q'] = df_6W['Q'].to_numpy()
data_PINN_6W['TG'] = df_6W['TG'].to_numpy()
data_PINN_6W['DG'] = df_6W['DG'].to_numpy()
data_PINN_6W['MG'] = df_6W['MG'].to_numpy()
data_PINN_6W['G'] = df_6W['G'].to_numpy()
data_PINN_6W['ME'] = df_6W['ME'].to_numpy()
df_PINN_6W = pd.DataFrame.from_dict(data_PINN_6W)
df_PINN_6W.to_csv(PATH+'/'+best_case+'/results/'+'data_PINN_6W.csv')

data_num_4 = {}
data_num_4['t'] = t_num_4
data_num_4['TG'] = y_num_4[:,0]
data_num_4['DG'] = y_num_4[:,1]
data_num_4['MG'] = y_num_4[:,2]
data_num_4['G'] = y_num_4[:,3]
data_num_4['ME'] = y_num_4[:,4]
data_num_4['T'] = y_num_4[:,5]
df_num_4 = pd.DataFrame.from_dict(data_num_4)
df_num_4.to_csv(PATH+'/'+best_case+'/results/'+'data_num_4W.csv')

data_num_5 = {}
data_num_5['t'] = t_num_5
data_num_5['TG'] = y_num_5[:,0]
data_num_5['DG'] = y_num_5[:,1]
data_num_5['MG'] = y_num_5[:,2]
data_num_5['G'] = y_num_5[:,3]
data_num_5['ME'] = y_num_5[:,4]
data_num_5['T'] = y_num_5[:,5]
df_num_5 = pd.DataFrame.from_dict(data_num_5)
df_num_5.to_csv(PATH+'/'+best_case+'/results/'+'data_num_5W.csv')

data_num_6 = {}
data_num_6['t'] = t_num
data_num_6['TG'] = y_num[:,0]
data_num_6['DG'] = y_num[:,1]
data_num_6['MG'] = y_num[:,2]
data_num_6['G'] = y_num[:,3]
data_num_6['ME'] = y_num[:,4]
data_num_6['T'] = y_num[:,5]
df_num_6 = pd.DataFrame.from_dict(data_num_6)
df_num_6.to_csv(PATH+'/'+best_case+'/results/'+'data_num_6W.csv')

data_exp = {}
data_exp['t'] = X_train[:,0].detach().numpy().flatten()
data_exp['Q'] = X_train[:,1].detach().numpy().flatten()
data_exp['TG'] = Y_train[:,0].detach().numpy().flatten()
data_exp['DG'] = Y_train[:,1].detach().numpy().flatten()
data_exp['MG'] = Y_train[:,2].detach().numpy().flatten()
data_exp['G'] = Y_train[:,3].detach().numpy().flatten()
data_exp['ME'] = Y_train[:,4].detach().numpy().flatten()
df_exp = pd.DataFrame.from_dict(data_exp)
df_exp = df_exp[df_exp['TG'] != 0.0]
df_exp.to_csv(PATH+'/'+best_case+'/results/'+'data_exp.csv')

data_temp = {}
data_temp['t'] = X_train[:,0].detach().numpy().flatten()
data_temp['Q'] = X_train[:,1].detach().numpy().flatten()
data_temp['T'] = output[:,5].detach().numpy().flatten()
df_temp = pd.DataFrame.from_dict(data_temp)

df_temp_4W = df_temp[df_temp['Q'] == 4.0]
df_temp_4W = df_temp_4W.drop_duplicates(subset=['t', 'Q'])
df_temp_4W = df_temp_4W.sort_values('t')
df_temp_4W.to_csv(PATH+'/'+best_case+'/results/'+'data_temp_4W.csv')

df_temp_5W = df_temp[df_temp['Q'] == 5.0]
df_temp_5W = df_temp_5W.drop_duplicates(subset=['t', 'Q'])
df_temp_5W = df_temp_5W.sort_values('t')
df_temp_5W.to_csv(PATH+'/'+best_case+'/results/'+'data_temp_5W.csv')

df_temp_6W = df_temp[df_temp['Q'] == 6.0]
df_temp_6W = df_temp_6W.drop_duplicates(subset=['t', 'Q'])
df_temp_6W = df_temp_6W.sort_values('t')
df_temp_6W.to_csv(PATH+'/'+best_case+'/results/'+'data_temp_6W.csv')

data_temp_exp = {}
data_temp_exp['t'] = X_train[:,0].detach().numpy()
data_temp_exp['Q'] = X_train[:,1].detach().numpy()
data_temp_exp['T'] = Z_train[:,0].detach().numpy()
data_temp_exp = pd.DataFrame.from_dict(data_temp_exp)

df_temp_exp_4W = data_temp_exp[data_temp_exp['Q'] == 4.0]
df_temp_exp_4W = df_temp_exp_4W[df_temp_exp_4W['T'] != 0]
df_temp_exp_4W.to_csv(PATH+'/'+best_case+'/results/'+'data_temp_exp_4W.csv')

df_temp_exp_5W = data_temp_exp[data_temp_exp['Q'] == 5.0]
df_temp_exp_5W = df_temp_exp_5W[df_temp_exp_5W['T'] != 0]
df_temp_exp_5W.to_csv(PATH+'/'+best_case+'/results/'+'data_temp_exp_5W.csv')

df_temp_exp_6W = data_temp_exp[data_temp_exp['Q'] == 6.0]
df_temp_exp_6W = df_temp_exp_6W[df_temp_exp_6W['T'] != 0]
df_temp_exp_6W.to_csv(PATH+'/'+best_case+'/results/'+'data_temp_exp_6W.csv')

data_norm = {}
C_max = 0.6191142
df_mean = df_exp.groupby(['t', 'Q']).mean() / C_max
df_std = df_exp.groupby(['t', 'Q']).std() / C_max
df_all_PINN = df_output[(df_output['t'] == 0.0) | (df_output['t'] == 60.0) | (df_output['t'] == 120.0) | (df_output['t'] == 240.0) | (df_output['t'] == 360.0) | (df_output['t'] == 600.0)]
df_mean_PINN = df_all_PINN.groupby(['t', 'Q']).mean() / C_max
df_std_PINN = df_all_PINN.groupby(['t', 'Q']).std() / C_max
data_norm['TG'] = df_mean['TG'].to_numpy()
data_norm['TG_error'] = df_std['TG'].to_numpy()
data_norm['DG'] = df_mean['DG'].to_numpy()
data_norm['DG_error'] = df_std['DG'].to_numpy()
data_norm['MG'] = df_mean['MG'].to_numpy()
data_norm['MG_error'] = df_std['MG'].to_numpy()
data_norm['TG_PINN'] = df_mean_PINN['TG'].to_numpy()
data_norm['DG_PINN'] = df_mean_PINN['DG'].to_numpy()
data_norm['MG_PINN'] = df_mean_PINN['MG'].to_numpy()
df_norm = pd.DataFrame.from_dict(data_norm)
df_norm.to_csv(PATH+'/'+best_case+'/results/'+'data_norm.csv')

data_error_4W = {}
df_test = df_exp[df_exp['Q'] == 4.0]
df_mean = df_test.groupby(['t'], as_index=False).mean()
df_std = df_test.groupby(['t'], as_index=False).std()
data_error_4W['t'] = df_mean['t']
data_error_4W['TG'] = df_mean['TG'].to_numpy()
data_error_4W['TG_error'] = df_std['TG'].to_numpy()
data_error_4W['DG'] = df_mean['DG'].to_numpy()
data_error_4W['DG_error'] = df_std['DG'].to_numpy()
data_error_4W['MG'] = df_mean['MG'].to_numpy()
data_error_4W['MG_error'] = df_std['MG'].to_numpy()
df_error_4W = pd.DataFrame.from_dict(data_error_4W)
df_error_4W.to_csv(PATH+'/'+best_case+'/results/'+'data_error_4W.csv')

data_error_5W = {}
df_test = df_exp[df_exp['Q'] == 5.0]
df_mean = df_test.groupby(['t'], as_index=False).mean()
df_std = df_test.groupby(['t'], as_index=False).std()
data_error_5W['t'] = df_mean['t']
data_error_5W['TG'] = df_mean['TG'].to_numpy()
data_error_5W['TG_error'] = df_std['TG'].to_numpy()
data_error_5W['DG'] = df_mean['DG'].to_numpy()
data_error_5W['DG_error'] = df_std['DG'].to_numpy()
data_error_5W['MG'] = df_mean['MG'].to_numpy()
data_error_5W['MG_error'] = df_std['MG'].to_numpy()
df_error_5W = pd.DataFrame.from_dict(data_error_5W)
df_error_5W.to_csv(PATH+'/'+best_case+'/results/'+'data_error_5W.csv')

data_error_6W = {}
df_test = df_exp[df_exp['Q'] == 6.0]
df_mean = df_test.groupby(['t'], as_index=False).mean()
df_std = df_test.groupby(['t'], as_index=False).std()
data_error_6W['t'] = df_mean['t']
data_error_6W['TG'] = df_mean['TG'].to_numpy()
data_error_6W['TG_error'] = df_std['TG'].to_numpy()
data_error_6W['DG'] = df_mean['DG'].to_numpy()
data_error_6W['DG_error'] = df_std['DG'].to_numpy()
data_error_6W['MG'] = df_mean['MG'].to_numpy()
data_error_6W['MG_error'] = df_std['MG'].to_numpy()
df_error_6W = pd.DataFrame.from_dict(data_error_6W)
df_error_6W.to_csv(PATH+'/'+best_case+'/results/'+'data_error_6W.csv')
