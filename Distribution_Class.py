import pandas as pd
import numpy as np
class Distribution:
    def __init__(Dist, fn = "Rock Distributions/ChangE 4.csv"):
        df = pd.read_csv(fn)
        Dist.name = df['Name'][0]
        Dist.type = df['Type'][0]
        Dist.k = df['k'][0]
        Dist.l = df['l'][0]

    def getDistribution(Dist, Dmin = 0.01, Dmax = 3.0, DeltaD = 0.001):
        D = np.linspace(Dmin, Dmax, int((Dmax - Dmin)/DeltaD))
        if (Dist.type == 'Exponential'):
            NgtD = Dist.k*np.exp(-Dist.l*D)/D**2
        elif (Dist.type == 'Power Law'):
            NgtD = Dist.k*D**(-Dist.l)
        else:
            print("ERROR: Not a valid distribution type.")

        return D, NgtD
