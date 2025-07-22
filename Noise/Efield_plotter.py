import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from Scat_ClassV2 import Efield
import glob

data_list = glob.glob('./VolScatNoiseData/SVII/5a5b10d_randOri_1nsMG_WG/*.csv')
#data_list = glob.glob('./VolScatNoiseData/0.4D_400/*.csv')
data_list = glob.glob('./VolScatNoiseData/PEC_0.05D_20DL_1nsMG/*.csv')
data_list = glob.glob('./VolScatNoiseData/PEC_0.05D_20DL_1nsMG_WG/*.csv')

data_list = glob.glob('./VolScatNoiseData/Rock_0.05D_10DL_3000Rocks_1nsMG/*.csv')
data_list = glob.glob('./VolScatNoiseData/Rock_005D_20Dl_4000Rocks_1nsMG_WG/*.csv')
data_list = glob.glob('./VolScatNoiseData/0.04to2D_5a5b10d_RandOri_PBC/*.csv')
data_list2 = glob.glob('../../../Downloads/CE3_dipole/*.csv')
data_list = glob.glob('./VolScatNoiseData/CE3_Sphere_dipole/*.csv')
data_list2 = glob.glob('./VolScatNoiseData/CE3_Sphere_Dipole_15m/*.csv')
#data_list = glob.glob('./VolScatNoiseData/SVII/5a5b10d_randOri_1nsMG_FreeSpace/*.csv')
print(data_list)
dfref2 = pd.read_csv('../../../Downloads/Empty_5ns_18m_dipole_at_0.5m.csv')
dfref = pd.read_csv('./VolScatNoiseData/CE3_5ns_18m_dipole_at_0.5m_Sphere_Empty.csv')

dfmatlab = pd.read_csv('./VolScatNoiseData/Mie_CE3.csv')
f = np.arange(0, 1.2, 0.001)
'''dfref = pd.read_csv('./VolScatNoiseData/CE3_5ns_14m_dipole_at_0.5m_Sphere_10a10b5d.csv')
plt.plot(dfref.iloc[:, 0], dfref.iloc[:, 2])
plt.show()
plt.plot(np.fft.fftfreq(len(dfref.iloc[:, 0]), dfref.iloc[1, 0] - dfref.iloc[0, 0])[:len(dfref.iloc[:, 0])//2], 
         10*np.log10(abs(np.fft.fft(dfref.iloc[:, 2]))[:len(dfref.iloc[:, 0])//2]/(np.max(np.fft.fft(dfref.iloc[:, 2])[:len(dfref.iloc[:, 0])//2]))))
plt.show()'''


t = dfref.iloc[:,0]
Ey_ref = dfref.iloc[:, 2]
fftref = np.fft.fft(Ey_ref)

t2 = dfref2.iloc[:,0]
Ey_ref2 = dfref2.iloc[:, 2]
fftref = np.fft.fft(Ey_ref)
#plt.plot(t, Ey_ref)
Eavg = []
for i in range(len(data_list)):
    df = pd.read_csv(data_list[i])
    t = df.iloc[:, 0]
    Ey = np.array(df.iloc[:, 2])
    Eavg.append(Ey)

Eavg2 = []
for i in range(len(data_list2)):
    df = pd.read_csv(data_list2[i])
    t2 = df.iloc[:, 0]
    Ey = np.array(df.iloc[:, 2])
    Eavg2.append(Ey)
    #print(data_list[i])
    #print(t[210], Ey[210])
    #plt.plot(t, Ey)
#print(len(t))
Eavg = np.mean(np.array(Eavg), axis = 0)
Eavg2 = np.mean(np.array(Eavg2), axis = 0)
plt.plot(t, Eavg)
plt.plot(t2, Eavg2)
plt.show()

fft = np.fft.fft(Eavg)
fft2 = np.fft.fft(Eavg2)
freq = np.fft.fftfreq(len(t), t[1]-t[0])
freq2 = np.fft.fftfreq(len(t2), t2[1]-t2[0])
plt.plot(freq[:len(t)//2], 20*np.log10(abs(fftref[:len(t)//2])/np.max(abs(fftref[:len(t)//2]))))
plt.plot(freq[:len(t)//2], 20*np.log10(abs(fft[:len(t)//2])/np.max(abs(fft[:len(t)//2]))))
plt.plot(freq2[:len(t2)//2], 20*np.log10(abs(fft2[:len(t2)//2])/np.max(abs(fft2[:len(t2)//2]))))
plt.xlim((0, 1))
plt.show()

plt.plot(freq[:len(t)//2], (20*np.log10(abs(fft[:len(t)//2])/np.max(abs(fftref[:len(t)//2]))) - 20*np.log10(abs(fftref[:len(t)//2])/np.max(abs(fftref[:len(t)//2])))))
plt.plot(f, 10*np.array(dfmatlab.iloc[:, 0]))
plt.xlim((0,1))
plt.ylim((-10, 1))
plt.show()

'''dfmf = pd.read_csv('./VolScatNoiseData/Empty_10d_Broadband.csv')
Exmf = np.array(dfmf['Total E x (V/m)'])
tmf = np.array(dfmf['Time (ns)'])
tns = np.linspace(0, 200, 10000)
mf = np.flip(np.interp(tns, tmf, Exmf))
MATCHEDFILT = False
Ecopol = []
for ii in range(len(data_list)):
    df = pd.read_csv(data_list[ii])
    tns = np.linspace(0, 200, 10000)
    #plt.plot(tns, np.interp(tns, df['Time (ns)'], df['Total E x (V/m)']))
    #print(len(df['Total E x (V/m)']))
    if (np.max(np.array(df['Total E x (V/m)'])) > np.max(np.array(df['Total E y (V/m)']))):
        if (MATCHEDFILT == True):
            Ecopol.append(np.convolve(np.interp(tns, df['Time (ns)'], df['Total E x (V/m)']), mf, mode = 'same'))
        else:
            Ecopol.append(np.interp(tns, df['Time (ns)'], df['Total E x (V/m)']))
    else:
        Ecopol.append(np.interp(tns, df['Time (ns)'], df['Total E y (V/m)']))

Ecpavg = np.mean(Ecopol, axis = 0)

fft = np.abs(np.fft.fft(Ecpavg))
freq = np.fft.fftfreq(len(Ecpavg), tns[1] - tns[0])
plt.plot(freq, 10*np.log10(fft/fft.max()))
plt.show()

fig, axs = plt.subplots(2, 1, sharex = True)
Ecopol = []
Noise = []
Noise_xpol = []
for ii in range(len(data_list)):
    N = Efield(data_list[ii], Eavg = Ecpavg, matchedFilt = MATCHEDFILT)
    #plt.plot(N.getFFT()[1], N.getFFT()[0])
    #axs[0].plot(N.t, N.getNoise(), linestyle = ':')
    Noise.append(N.getNoiseEnv())
    Noise_xpol.append(N.getNoiseEnv(xpol = True))
    Ecopol.append(N.Ecopol_i)
    #axs[0].plot(N.t, N.getNoiseEnv(), linestyle = '--')
    #axs[1].plot(N.t, N.Ey_i)
Ex = np.array(Ecopol)
Noise = np.array(Noise)
Noise_xpol = np.array(Noise_xpol)
mean_env = np.mean(Noise, axis = 0)
env_std = np.std(Noise, axis = 0)
mean_env_xpol = np.mean(Noise_xpol, axis = 0)
env_xpol_std = np.std(Noise_xpol, axis = 0)

SNR = np.max(np.abs(N.Eavg_i))**2/np.max(mean_env)**2
print("E amplitude: ", np.max(np.abs(Ecpavg)))
print("Noise Amplitude: ", np.max(mean_env))
print("SNR: ", SNR)

#axs[0].plot(N.t, N.Empty, label = "Empty Cavity")
axs[0].plot(N.t, N.Eavg_i, label = 'Attenuated Signal')
#axs[0].fill_between(N.t,mean_env - env_std, mean_env + env_std, color = 'k', alpha = 0.5)
axs[0].plot(N.t, mean_env, color = 'k', label = 'co-pol envelope')

#axs[0].fill_between(N.t,mean_env_xpol - env_xpol_std, mean_env_xpol + env_xpol_std, color = 'r', alpha = 0.5)
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