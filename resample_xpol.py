import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import resample, correlate
from scipy.interpolate import UnivariateSpline
import glob
import math


NUMSAMPLES = 700
dfref = pd.read_csv('./Noise/VolScatNoiseData/CE3_5ns_18m_dipole_at_0.5m_Sphere_Empty.csv')
dfref = pd.read_csv('./Noise/VolScatNoiseData/CE3_Donut_5ro_3ri_8x0y2.5z_ref.csv')
Ey = dfref.iloc[:, 3]
Ey_resample_ref = resample(Ey, NUMSAMPLES)
fftref = np.fft.fft(Ey_resample_ref)
t2 = np.linspace(0, 100, NUMSAMPLES)
fftfreq = np.fft.fftfreq(len(t2), t2[1] - t2[0])
print(len(fftfreq[:len(fftfreq//2)]))
#plt.plot(t2, Ey_resample_ref)
#plt.show()

#plt.plot(fftfreq[:len(fftfreq)//2], 20*np.log10(abs(fftref[:len(fftfreq)//2])/np.max(abs(fftref[:len(fftfreq)//2]))))
#plt.show()




#data_list = glob.glob('./Noise/VolScatNoiseData/CE3_Donut_5ro_3ri_-8x0y2.5z/*.csv')
#data_list = glob.glob('./Noise/VolScatNoiseData/SVII_Donut_5ro_3ri_0.1D_3Dmax/-8x0y2.5z/*.csv') + glob.glob('./Noise/VolScatNoiseData/SVII_Donut_5ro_3ri_0.1D_3Dmax/8x0y2.5z/*.csv') + glob.glob('./Noise/VolScatNoiseData/SVII_Donut_5ro_3ri_0.1D_3Dmax/0x-8y2.5z/*.csv') + glob.glob('./Noise/VolScatNoiseData/SVII_Donut_5ro_3ri_0.1D_3Dmax/0x8y2.5z/*.csv')
data_list = glob.glob('./Noise/VolScatNoiseData/CE3_Donut_5ro_3ri_0.03D_3Dmax/-8x0y2.5z/*.csv') + glob.glob('./Noise/VolScatNoiseData/CE3_Donut_5ro_3ri_0.03D_3Dmax/8x0y2.5z/*.csv') + glob.glob('./Noise/VolScatNoiseData/CE3_Donut_5ro_3ri_0.03D_3Dmax/0x8y2.5z/*.csv') + glob.glob('./Noise/VolScatNoiseData/CE3_Donut_5ro_3ri_0.03D_3Dmax/0x-8y2.5z/*.csv')
data_list = glob.glob('./Noise/VolScatNoiseData/CE3_Cuboid_Donut_4ro_2ri_0.03D_3Dmax/-8x0y2.5z/*.csv') + glob.glob('./Noise/VolScatNoiseData/CE3_Cuboid_Donut_4ro_2ri_0.03D_3Dmax/8x0y2.5z/*.csv') + glob.glob('./Noise/VolScatNoiseData/CE3_Cuboid_Donut_4ro_2ri_0.03D_3Dmax/0x8y2.5z/*.csv') + glob.glob('./Noise/VolScatNoiseData/CE3_Cuboid_Donut_4ro_2ri_0.03D_3Dmax/0x-8y2.5z/*.csv')

data_list8x = glob.glob('./Noise/VolScatNoiseData/CE3_Cuboid_Donut_4ro_2ri_0.03D_3Dmax/8x0y2.5z/*.csv')
data_listminus8x = glob.glob('./Noise/VolScatNoiseData/CE3_Cuboid_Donut_4ro_2ri_0.03D_3Dmax/-8x0y2.5z/*.csv')
data_list8y = glob.glob('./Noise/VolScatNoiseData/CE3_Cuboid_Donut_4ro_2ri_0.03D_3Dmax/0x8y2.5z/*.csv')
data_listminus8y = glob.glob('./Noise/VolScatNoiseData/CE3_Cuboid_Donut_4ro_2ri_0.03D_3Dmax/0x-8y2.5z/*.csv')
data_list = data_list8x + data_listminus8x + data_list8y + data_listminus8y

#data_list = glob.glob('./Noise/VolScatNoiseData/CE4_Donut_7ro_2ri_0.03D_3Dmax/-8x0y2.5z/*.csv') + glob.glob('./Noise/VolScatNoiseData/CE4_Donut_7ro_2ri_0.03D_3Dmax/8x0y2.5z/*.csv') + glob.glob('./Noise/VolScatNoiseData/CE4_Donut_7ro_2ri_0.03D_3Dmax/0x8y2.5z/*.csv') + glob.glob('./Noise/VolScatNoiseData/CE4_Donut_7ro_2ri_0.03D_3Dmax/0x-8y2.5z/*.csv')
print(len(data_list))
#data_list = glob.glob('./Noise/VolScatNoiseData/SVII_Donut_5ro_3ri_0.1D_3Dmax/8x0y2.5z/*.csv')
#data_list = ['../../Downloads/test.csv']
#df = pd.read_csv(data_list[0])
#Ey = df.iloc[:, 2]

#Ey_resample_ref = resample(Ey, 10000)

print(len(data_list))
Eyavg0 = []
Eyavg = []
Expolavg = []
delay = []
Expol_avg0 = []
for i in range(len(data_list)):
    #print(data_list[i])
    df = pd.read_csv(data_list[i])
    #df = pd.read_csv('./Noise/VolScatNoiseData/CE3_Donut_5ro_3ri_0.03D_3Dmax/-8x0y2.5z/189.csv')
    try:
        t = df['Time (ns)']
    except:
        #print("FUCK")
        t = np.array(df['Time (s)'])/1e-9
    #t = df.iloc[:, 0]
    #print(len(t))
    Ey = df.iloc[:, 3]
    #Expol = df.iloc[:, 2]
    if (len(data_list)%4 == 0):
        Expol = df.iloc[:, 2]
    elif (len(data_list)%4 == 1):
        Expol = -df.iloc[:, 2]
    elif (len(data_list)%4 == 2):
        Expol = -df.iloc[:, 1]
    elif (len(data_list)%4 == 3):
        Expol = df.iloc[:, 1]
    #plt.plot(t, Ey)
    #print(data_list[i])
    #print(i)
    if (math.isnan(Ey[400]) or np.max(Ey) > 1):
        print(data_list[i])
    else:
        #print(t[400], Ey[400])
        #print(' ')

        Eyavg0.append(Ey)
        Expol_avg0.append(Expol)

        Ey_resample = resample(Ey, NUMSAMPLES)
        Eyxpol_resample = resample(Expol, NUMSAMPLES)
        t2 = np.linspace(0, 100, NUMSAMPLES)

        corr = correlate(Ey_resample_ref, Ey_resample, mode = 'same')
        delay = np.array(corr).argmax() - NUMSAMPLES//2

        Ey_resample = np.roll(Ey_resample, delay)
        #print(delay)
        if (i == 0):
            Eyxpol_ref = Eyxpol_resample
            plt.plot(t2, Eyxpol_ref)
            plt.show()
        corr = correlate(Eyxpol_ref, Eyxpol_resample, mode = 'same')
        delay = np.array(corr).argmax() - NUMSAMPLES//2
        Eyxpol_resample = np.roll(Eyxpol_resample, delay)
        #print(delay)
        #plt.plot(corr)
        #plt.show()
        Eyavg.append(Ey_resample)
        Expolavg.append(Eyxpol_resample)

        #plt.plot(t, Ey, marker = 'o', linestyle = '--')
        #plt.plot(t2, Ey_resample, marker = 'o', linestyle = '--')
#plt.show()
Eyavg = np.mean(Eyavg, axis = 0)
Expolavg = np.mean(Expolavg, axis = 0)
#Eyavg0 = np.mean(Eyavg0, axis = 0)
Expol_avg0 = np.mean(Expol_avg0, axis = 0)

fig, axs = plt.subplots(1, 2)
axs[0].plot(t, Expol_avg0)
axs[0].set_xlabel('time (ns)', fontsize = 15)
axs[0].set_ylabel('E-field (V/m)', fontsize = 15)
axs[0].set_title('Cross correlated Xpol field', fontsize = 15, fontweight = 'bold')
axs[0].grid()

axs[1].plot(t2, Ey_resample_ref, label = 'reference waveform')
axs[1].plot(t2, Eyavg, label = 'attenuated waveform')
axs[1].set_xlabel('time (ns)', fontsize = 15)
axs[1].set_ylabel('E-field (V/m)', fontsize = 15)
axs[1].set_title('Cross correlated Copol field', fontsize = 15, fontweight = 'bold')
axs[1].grid()
axs[1].legend()
#axs[1].plot(t, Eyavg0, linestyle = '--')
#axs[1].set_ylim(-0.002, 0.002)
#axs[0].set_ylim(-0.002, 0.002)
#plt.plot(t2, Ey_resample_ref)
plt.show()

fft = abs(np.fft.fft(Eyavg))
f = np.fft.fftfreq(len(t2), t2[1] - t2[0])
fft0 = abs(np.fft.fft(Eyavg0))
f0 = np.fft.fftfreq(len(t), t[1] - t[0])

spline = UnivariateSpline(f[:len(f)//2], 20*np.log10(abs(fft[:len(f)//2])/np.max(abs(fft[:len(f)//2]))))
#plt.plot(f0[:len(f0)//2], 20*np.log10(fft0[:len(f0)//2]/np.max(fft0[:len(f0)//2])))
plt.plot(f[:len(f)//2], 20*np.log10(abs(fft[:len(f)//2])/np.max(abs(fft[:len(f)//2]))))
plt.plot(f[:len(f)//2], 20*np.log10(abs(fftref[:len(f)//2])/np.max(abs(fftref[:len(f)//2]))))
plt.plot(f[:len(f)//2], spline(f[:len(f)//2]))
plt.xlim((0, 1))
plt.show()


dfmatlab = pd.read_csv('./Noise/VolScatNoiseData/Mie_CE3_Cuboid.csv')
fm = np.arange(0, 1.2, 0.001)
plt.plot(fm, 2*(np.array(dfmatlab.iloc[:, 0])))
plt.plot(f[:len(f)//2], 20*np.log10(abs(fft[:len(f)//2])/abs(fftref[:len(f)//2])))
#plt.plot(f[:len(f)//2], spline(f[:len(f)//2]) - 20*np.log10(abs(fftref[:len(f)//2])/np.max(abs(fft[:len(f)//2]))))
plt.xlim((0,1))
plt.ylim((-10.0, 0.1))
plt.grid()
plt.show()
