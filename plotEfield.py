import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dfice = pd.read_csv('../../Downloads/iceLayer_pointSensor45.csv')
dfbasalt = pd.read_csv('../../Downloads/basaltLayer_pointSensor45.csv')

Exice = dfice.iloc[:, 1]
Eyice = dfice.iloc[:, 2]
Ezice = dfice.iloc[:, 3]

Exbasalt = dfbasalt.iloc[:, 1]
Eybasalt = dfbasalt.iloc[:, 2]
Ezbasalt = dfbasalt.iloc[:, 3]

fig, axs = plt.subplots(3, 2)
axs[0,0].plot(dfice.iloc[:, 0], Exice)
axs[0,0].set_xlim(175, 225)
axs[1,0].plot(dfice.iloc[:, 0], Eyice)
axs[1,0].set_xlim(175, 225)
axs[2,0].plot(dfice.iloc[:, 0], Ezice)
axs[2,0].set_xlim(175, 225)


axs[0,1].plot(dfbasalt.iloc[:, 0], Exbasalt)
axs[0,1].set_xlim(175, 225)
axs[1,1].plot(dfbasalt.iloc[:, 0], Eybasalt)
axs[1,1].set_xlim(175, 225)
axs[2,1].plot(dfbasalt.iloc[:, 0], Ezbasalt)
axs[2,1].set_xlim(175, 225)

plt.show()