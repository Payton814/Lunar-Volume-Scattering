import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample, correlate
from scipy.interpolate import UnivariateSpline

dfmatlab = np.array(pd.read_csv('./Noise/VolScatNoiseData/Mie_CE3_Cuboid.csv').iloc[:, 0])
fm = np.arange(0, 1.2, 0.001)

## Distance wave has travelled through rocks
d = 8

dfref = pd.read_csv('./Noise/VolScatNoiseData/CE3_Donut_5ro_3ri_8x0y2.5z_ref.csv')
Ey = np.array(dfref.iloc[:, 3])
t = np.array(dfref.iloc[:, 0])
fE = np.fft.fft(Ey)
freq = np.fft.fftfreq(len(t), t[1] - t[0])
print(len(freq))
plt.plot(freq)
plt.show()
fend = np.abs(1.2 - freq[:len(fE)//2]).argmin()
fend2 = np.abs(-1.2 - freq).argmin()
print(len(freq[fend2:]), len(freq[:fend]))
plt.plot(freq[:fend])
plt.plot(freq[fend2:])
plt.show()

Pmat_interp = np.interp(freq[:fend], fm, dfmatlab)
print(fend)

Pmat = np.ones(len(freq))
Pmat[:fend] = Pmat_interp
Pmat[fend2:] = np.flip(Pmat_interp)

#plt.plot(Pmat)
#plt.show()

plt.plot(freq, fE.real)
plt.plot(freq, fE.imag)
plt.plot(freq, np.abs(fE))
#plt.plot(freq, 10**(Pmat/20))
plt.plot(freq[:fend], 10**(Pmat_interp/20)*np.abs(fE[:fend]))
plt.plot(freq, 10**(Pmat/20)*np.abs(fE), linestyle = '--')
plt.show()


plt.plot(freq[:fend], fE.real[:fend])
plt.plot(freq[:fend], fE.imag[:fend])
plt.plot(freq[:fend], np.abs(fE[:fend]))
plt.plot(freq[:fend], 10**(Pmat_interp/20)*np.abs(fE[:fend]))
plt.plot(freq[:fend], 10**(Pmat_interp/20)*fE[:fend].real, linestyle = '--')
plt.plot(freq[:fend], 10**(Pmat_interp/20)*fE[:fend].imag, linestyle = '--')
plt.plot(freq[:fend], np.abs(10**(Pmat_interp/20)*fE[:fend]), linestyle = '--')

plt.show()

ifE = np.fft.ifft(fE*10**(d*Pmat/20))
print(len(ifE))
plt.plot(t, ifE.real)
plt.plot(t, np.fft.ifft(fE).real)
plt.plot(t, Ey, linestyle = '--')
plt.show()

#print(len(freq))

'''fE_rs = np.interp(fm, freq[:len(fE)//2], np.abs(fE[:len(fE)//2])/np.max(np.abs(fE[:len(fE)//2])))
#print(10**(dfmatlab/20))
print(fE_rs[:5])
print(10**(dfmatlab/20)[:5])
print((fE_rs*(10**(dfmatlab/20)))[:5])

plt.plot(freq[:len(fE)//2], np.abs(fE[:len(fE)//2])/ np.max(np.abs(fE[:len(fE)//2])))
plt.plot(fm, fE_rs, linestyle = '--')
plt.plot(fm, 10**(dfmatlab/20))
plt.plot(fm, 10**(dfmatlab/20)*fE_rs)
plt.xlim(0, 1.2)
plt.show()

print(fE[0])
fE_rs_real = np.interp(fm, freq[:len(fE)//2], fE[:len(fE)//2].real)
fE_rs_imag = np.interp(fm, freq[:len(fE)//2], fE[:len(fE)//2].imag)

##plt.plot(fE_rs_real)
#plt.plot(fE_rs_imag)
#plt.plot(np.abs(fE_rs_real + 1j*fE_rs_imag))
#plt.show()

E = np.fft.irfft(10**(dfmatlab/20)*fE_rs)
E = np.roll(E, len(E)//2)
plt.plot(np.arange(len(E))*t[-1]/len(E), E)
plt.plot(t, Ey)
plt.show()'''


