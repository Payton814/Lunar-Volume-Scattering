import pandas as pd
import numpy as np
from scipy.signal import hilbert

class Efield:
    ## This class is meant to work on csv files of time domain data of volume scattering from XFdtd
    def __init__(self, fn = './VolScatNoiseData/Empty.csv', tns = np.linspace(0, 300, 10000), Eavg = None, matchedFilt = False):
        self.t = tns
        df_empty = pd.read_csv('./VolScatNoiseData/Empty_20d.csv')
        self.Empty = np.interp(tns, df_empty['Time (ns)'], df_empty['Total E x (V/m)'])

        df = pd.read_csv(fn)

        ## Create a vector of the xyz compenents of the electric field
        ## The E field is assumed to be traveling on the z-direction. 
        ## To be able to compare those sims with different polarizations the field vector will be split into
        ## Co-pol, X-pol, z, t
        ## The Co-pol is determined by whether x or y has the maximum value. Xpol should be only noise
        if (np.max(np.abs(np.array(df['Total E x (V/m)']))) > np.max(np.abs(np.array(df['Total E y (V/m)'])))):
            self.E = (np.array(df['Total E x (V/m)']), np.array(df['Total E y (V/m)']), np.array(df['Total E z (V/m)']), np.array(df['Time (ns)']))
        else:
            self.E = (np.array(df['Total E y (V/m)']), np.array(df['Total E x (V/m)']), np.array(df['Total E z (V/m)']), np.array(df['Time (ns)']))
        

        if Eavg is None:
            self.Eavg_i = self.Empty
        else:
            self.Eavg_i = Eavg
        self.Ecopol_i = np.interp(tns, df['Time (ns)'], self.E[0])
        self.Expol_i = np.interp(tns, df['Time (ns)'], self.E[1])
        self.Ez_i = np.interp(tns, df['Time (ns)'], self.E[2])
        if (matchedFilt == True):
            dfmf = pd.read_csv('./VolScatNoiseData/Empty_10d_Broadband.csv')
            Exmf = np.array(dfmf['Total E x (V/m)'])
            tmf = np.array(dfmf['Time (ns)'])

            self.mf = np.flip(np.interp(tns, tmf, Exmf))

            self.Ecopol_i = np.convolve(self.Ecopol_i, self.mf, mode = 'same')
            self.Expol_i = np.convolve(self.Expol_i, self.mf, mode = 'same')




    def getNoise(self):

        ## Currently doing the dumb thing of subtracting off original pulse. This will put a bias
        Noise = self.Ecopol_i - self.Eavg_i
        return Noise
        
    def getNoiseEnv(self, xpol = False):
        ## Currently doing the dumb thing of subtracting off original pulse. This will put a bias
        if (xpol == True):
            Noise = self.Expol_i
        else:
            Noise = self.Ecopol_i - self.Eavg_i
        env = np.abs(hilbert(Noise))
        return env
    
    def getFFT(self, xpol = False):
        Noise = self.E[1]
        FFT = np.abs(np.fft.fft(Noise))
        f = np.fft.fftfreq(len(self.E[3]), self.E[3][1] - self.E[3][0])

        return FFT, f

    
