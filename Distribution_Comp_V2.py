from Distribution_Class import Distribution
import matplotlib.pyplot as plt
import glob
import os

import os

from tkinter import Tk
from tkinter.filedialog import askdirectory
Folder = askdirectory(title='Select Folder') # shows dialog box and return the path
Folder = Folder + '/*.csv'

#dirs = os.listdir(Folder)
for file in glob.glob(Folder):
    Dist = Distribution(fn = file)
    D, NgtD4 = Dist.getDistribution(Dmin = 0.001)
    if (Dist.name == 'SVII'):
        width = 3.0
    else:
        width = 1.0
    plt.plot(D, NgtD4, label = Dist.name, linewidth = width)
plt.legend()
plt.yscale('log')
plt.xlabel('Diameter [m]', fontsize = 20)
plt.ylabel('N per Area [number/$m^2$]', fontsize = 20)
plt.title('Cumulative Number of Rocks per Area for Known Data', fontsize = 20)
plt.grid()
plt.show()
