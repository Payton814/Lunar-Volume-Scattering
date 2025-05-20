import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from Scat_Class import Efield
import glob

data_list = glob.glob('./VolScatNoiseData/SVII/5a5b10d_randOri_1nsMG_WG/*.csv')
data_list2 = glob.glob('./VolScatNoiseData/SVII/5a5b10d_randOri_1nsMG_FreeSpace/*.csv')
#print(data_list)

dfmf = pd.read_csv('./VolScatNoiseData/Empty_10d_Broadband.csv')
Exmf = np.array(dfmf['Total E x (V/m)'])
tmf = np.array(dfmf['Time (ns)'])
tns = np.linspace(0, 300, 10000)
Ex_imf = np.interp(tns, tmf, Exmf)


Ex = []
for ii in range(len(data_list)):
    df = pd.read_csv(data_list[ii])
    #plt.plot(tns, np.interp(tns, df['Time (ns)'], df['Total E x (V/m)']))
    print(len(df['Total E x (V/m)']))
    #Ex.append(np.interp(tns, df['Time (ns)'], df['Total E x (V/m)']))
    mf = np.convolve(np.interp(tns, df['Time (ns)'], df['Total E x (V/m)']), np.flip(Ex_imf), mode = 'same')
    Ex.append(mf)
    #plt.plot(tns, mf/np.max(np.abs(mf)), linestyle = '-')

Exavg = np.mean(Ex, axis = 0)

#mf = np.convolve(Exavg, np.flip(Ex_imf), mode = 'same')

Ex2 = []
for ii in range(len(data_list2)):
    df = pd.read_csv(data_list2[ii])
    tns2 = np.linspace(0, 300, 10000)
    #plt.plot(tns, np.interp(tns, df['Time (ns)'], df['Total E x (V/m)']))
    #print(len(df['Total E y (V/m)']))
    Ex2.append(np.interp(tns2, df['Time (ns)'], df['Total E y (V/m)']))

Exavg2 = np.mean(Ex2, axis = 0)

plt.plot(tns, Exavg/np.max(np.abs(Exavg)), label = 'WG')
#plt.plot(tns, mf/np.max(np.abs(mf)), label = 'mf', linestyle = ':')
plt.plot(tns2, Exavg2/np.max(np.abs(Exavg2)), label = 'Freespace')
plt.show()




'''fig, axs = plt.subplots(2, 1, sharex = True)
Ex = []
Noise = []
Noise_xpol = []
for ii in range(len(data_list)):
    N = Efield(data_list[ii], Eavg = Exavg)
    #plt.plot(N.getFFT()[1], N.getFFT()[0])
    #axs[0].plot(N.t, N.getNoise(), linestyle = ':')
    Noise.append(N.getNoiseEnv())
    Noise_xpol.append(N.getNoiseEnv(xpol = True))
    Ex.append(N.Ex_i)
    #axs[0].plot(N.t, N.getNoiseEnv(), linestyle = '--')
    #axs[1].plot(N.t, N.Ey_i)
Ex = np.array(Ex)
Noise = np.array(Noise)
Noise_xpol = np.array(Noise_xpol)
mean_env = np.mean(Noise, axis = 0)
env_std = np.std(Noise, axis = 0)
mean_env_xpol = np.mean(Noise_xpol, axis = 0)

#axs[0].plot(N.t, N.Empty, label = "Empty Cavity")
axs[0].plot(N.t, Exavg, label = 'Attenuated Signal')
axs[0].plot(N.t, mean_env, color = 'k', label = 'co-pol envelope')
axs[0].plot(N.t, mean_env_xpol, color = 'r', label = 'x-pol envelope', linestyle = '--')
axs[1].plot(N.t, mean_env_xpol, color = 'k', label = 'x-pol envelope', linestyle = '-')

axs[0].grid()
axs[0].set_ylabel('E-field (V/m)', fontsize = 15)
axs[1].grid()
axs[1].set_ylabel('E-field (V/m)', fontsize = 15)
axs[0].set_title('Scattering Noise', fontsize = 15)
axs[0].legend()
axs[1].legend()
plt.subplots_adjust(wspace=0, hspace=0)
plt.xlabel('time (ns)', fontsize = 15)
plt.show()'''

'''Noise = Efield('./VolScatNoiseData/CE4/1.csv')
plt.plot(Noise.t, Noise.getNoiseEnv(), label = 'class test')
tinterp = np.linspace(0, 300, 10000)

fig, axs = plt.subplots(2, 1)

emptydf = pd.read_csv('./VolScatNoiseData/Empty.csv')

t = emptydf['Time (ns)'][1:]
Ey0 = np.array(emptydf['Total E x (V/m)'][1:])
Ey0_interp = np.interp(tinterp, t, Ey0)
print(len(Ey0))
#plt.plot(t, Ey0, label = 'Empty', marker = 'o')
#plt.plot(tinterp, Ey0_interp, label = 'Empty', marker = 'o')


df1 = pd.read_csv('./VolScatNoiseData/CE4/1.csv')

t = df1['Time (ns)']


Ey = np.array(df1['Total E x (V/m)'])
Ex = np.array(df1['Total E y (V/m)'])
Ey1_interp = np.interp(tinterp, t, Ey)
print(len(Ey))
analytic_signal = hilbert(Ey1_interp - Ey0_interp)
amplitude_envelope = np.abs(analytic_signal)
axs[0].plot(tinterp, amplitude_envelope)
axs[0].plot(tinterp, Ey1_interp - Ey0_interp, label = 't1', linestyle = '--')
axs[1].plot(t, Ex)


df2 = pd.read_csv('./VolScatNoiseData/CE4/2.csv')

t = df2['Time (ns)']
Ey = np.array(df2['Total E x (V/m)'])
Ex = np.array(df2['Total E y (V/m)'])
Ey2_interp = np.interp(tinterp, t, Ey)
print(len(Ey))
axs[0].plot(tinterp, Ey2_interp - Ey0_interp, label = 't2', linestyle = '--')
axs[1].plot(t, Ex, linestyle = '--')

axs[0].grid()
axs[1].grid()
axs[0].legend()
plt.show()'''