import pandas as pd
import numpy as np
from scipy.signal import hilbert

class Efield:
    ## This class is meant to work on csv files of time domain data of volume scattering from XFdtd
    def __init__(self, fn = './VolScatNoiseData/Empty.csv', tns = np.linspace(0, 300, 10000), Eavg = None):
        self.t = tns
        df_empty = pd.read_csv('./VolScatNoiseData/Empty_20d.csv')
        self.Empty = np.interp(tns, df_empty['Time (ns)'], df_empty['Total E x (V/m)'])

        df = pd.read_csv(fn)

        ## Create a vector of the xyz compenents of the electric field
        self.E = (np.array(df['Total E x (V/m)']), np.array(df['Total E y (V/m)']), np.array(df['Total E z (V/m)']), np.array(df['Time (ns)']))
        

        ## The way I currently am running the XF sims, the main pulse is oriented in the x-direction
        ## Want to get an interpolation of the data so that from sim to sim if theres any variance in the sampled times we
        ## can be comparing data at the right time.
        if Eavg is None:
            self.Eavg_i = self.Empty
        else:
            self.Eavg_i = Eavg
        self.Ex_i = np.interp(tns, df['Time (ns)'], self.E[0])
        self.Ey_i = np.interp(tns, df['Time (ns)'], self.E[1])
        self.Ez_i = np.interp(tns, df['Time (ns)'], self.E[2])

    def getNoise(self):
        ## Currently doing the dumb thing of subtracting off original pulse. This will put a bias
        Noise = self.Ex_i - self.Eavg_i
        return Noise
        
    def getNoiseEnv(self, xpol = False):
        ## Currently doing the dumb thing of subtracting off original pulse. This will put a bias
        if (xpol == True):
            Noise = self.Ey_i
        else:
            Noise = self.Ex_i - self.Eavg_i
        env = np.abs(hilbert(Noise))
        return env
    
    def getFFT(self, xpol = False):
        Noise = self.E[1]
        FFT = np.abs(np.fft.fft(Noise))
        f = np.fft.fftfreq(len(self.E[3]), self.E[3][1] - self.E[3][0])

        return FFT, f

    
