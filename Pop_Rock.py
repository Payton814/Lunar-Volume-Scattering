#############################################################################
#
#   In XF, sampling the CE4 data until a number of spheres is reached
#   means that there is a variation in the occupied volume fraction.
#   This code aims to try and see how waveguide size affect this variation
#   which should give insight into the types of variation one would expect
#   in the attenuation
#
#############################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import curve_fit


def nDpm3(D, Distribution = 'SVII', Shape = 'Sphere'):
    if (Distribution == 'CE4'):
        k = 0.0021
        qk = 0.5648 + 0.01258/k
        if (Shape == "Sphere"):
             G = 4/np.pi
        elif (Shape == "Cuboid"):
             G = 1/(0.8*0.54)
        n = G*k*qk*np.exp(-qk*D)/D**3
    elif( Distribution == 'CE3'):
         k = 0.0135
         qk = 1.734
         #D_ave = 0.11
         if (Shape == "Sphere"):
             G = 4/np.pi
         elif (Shape == "Cuboid"):
             G = 1/(0.8*0.54)
         n = G*k*qk*np.exp(-qk*D)/D**3
    elif( Distribution == 'SVII'):
        K = 7.9e3/1000**1.8 
        gamma = 1.8
        if (Shape == "Cuboid"):
            G = 1/0.54   
        else:
             G = 1  
        n = G*gamma*K*D**(-(gamma + 2))
    return n

def nDpm2(D, Shape = "Sphere"):
    k = 0.0021
    qk = 0.5648 + 0.01258/k
    if(Shape == "Sphere" or Shape == "Torus"):
          GeoConst = 4/np.pi
    if(Shape == "Cube" or Shape == "Pyramid"):
          GeoConst = 1
    if(Shape == "Cuboid"):
          GeoConst = 1/0.8
    n = GeoConst*k*qk*np.exp(-qk*D)/D**2
    return n

def expintA(x):
    A = np.log((0.56146/x + 0.65)*(1+x))
    B = (x**4)*np.exp(7.7*x)*(2+x)**3.7
    Ei = ((A)**-7.7+B)**-0.13
    return Ei

def NgtD4(D, Shape = "Shere"):
    if(Shape == "Sphere" or Shape == "Torus"):
          GeoConst = 4/np.pi
    if(Shape == "Cube" or Shape == "Pyramid"):
          GeoConst = 1
    if(Shape == "Cuboid"):
          GeoConst = 1/0.8
    x = 6.555*D
    N = GeoConst*0.0137608*(np.exp(-x)/D - 6.555*expintA(x))
    return N

def nDpm2D(D, Shape = "Sphere"):
    k = 0.0021
    qk = 0.5648 + 0.01258/k
    if(Shape == "Sphere" or Shape == "Torus"):
          GeoConst = 4/np.pi
    if(Shape == "Cube" or Shape == "Pyramid"):
          GeoConst = 1
    if(Shape == "Cuboid"):
          GeoConst = 1/(0.8*0.54)
    n = GeoConst*k*qk*np.exp(-qk*D)/D
    return n

def hpD(x):
    mean = 0.54
    std = 0.03
    expo = 0.5*((x - mean)/std)**2
    return np.exp(-expo)

def bpD(x):
    mean = 0.8
    std = 0.16
    expo = 0.5*((x - mean)/std)**2
    return np.exp(-expo)

D = 0.01 ## This is the smallest Diameter rock in the space
Dmax = 3.0
Distribution = 'SVII'
#D_ave = quad(nDpm2D, D, np.inf)[0]/quad(nDpm2, D, np.inf)[0]
#print("Average Diameter:", D_ave)

Geometries = ["Sphere", "Cuboid"]
std_list = []
mean_list = []
V = 5*5*10
Phi_arr = []

MAX_ROCK_D = 0

for j in Geometries:
    print('On Geometry:', j)
    #D_ave = quad(nDpm2D, D, np.inf, args = (j))[0]/quad(nDpm2, D, np.inf, args = (j))[0]
    #print("Average Diameter:", D_ave)
    #NSphere = NgtD4(D, Shape = j)*V/D_ave
    G = 1
    if (j == "Cuboid"):
         G = np.pi/(4*0.54*0.8)
    NSphere = quad(nDpm3, D, Dmax)[0]*V*G
    print("NRocks:", NSphere)
    phi = []

    for ii in range(1000):
        if ii%100 == 0:
            print(ii)
        Vs = 0
        for i in range(int(np.floor(NSphere))):
            '''if i%(V/1000) == 0:
                print(i)'''
            fx = -99
            xmax = Dmax
            fxmax = nDpm3(D, Distribution, j)
            W = fxmax
            while W > fx:
                W = fxmax*np.random.random()
                x = (xmax - D)*np.random.random() + D
                fx = nDpm3(x, Distribution, j)
            if (j == "Cuboid"):
                b = 0
                h = 0
                fb = -99
                fh = -99
                bxmax = x
                hxmax = x
                bymax = 1
                hymax = 1
                while(bymax > fb):
                    bymax = np.random.random(); ## Since the functions are normalized gaussian, only need to sample between 0 and 1
                    bx = np.random.random()
                    fb = bpD(bx)

                
                b = bx*x; ## If we break the loop then we have selected a width

                while(hymax > fh):
                    hymax = np.random.random(); ## Since the functions are normalized gaussian, only need to sample between 0 and 1
                    hx = np.random.random()
                    fh = hpD(hx)
                
                h = hx*x ##If we break the loop then we have selected a height
            if (x > MAX_ROCK_D):
                MAX_ROCK_D = x
            if(j == "Sphere"):
                Vs = Vs + (4/3)*np.pi*(x/2)**3
            if(j == "Cube"):
                Vs = Vs + x**3
            if(j == "Cuboid"):
                Vs = Vs + b*h*x
            if(j == "Pyramid"):
                Vs = Vs + x**3/3
            if(j == "Cylinder"):
                Vs = Vs + np.pi*x**3/4
            if(j == "Torus"):
                Vs = Vs + np.pi*np.pi*x**3/32
        phi.append(Vs/V)
        #phi_cube.append(Vs_cube/V[j])
        #phi_cuboid.append(Vs_cuboid/V[j])
    Phi_arr.append(phi)

for k in range(len(Geometries)):
    print("mean of ", Geometries[k], ": ", np.mean(Phi_arr[k]))

'''mean_sphere = np.mean(Phi_arr[0])
mean_list.append(mean_sphere)
std_sphere = np.std(Phi_arr[0])
std_list.append(std_sphere)
print("sphere mean:", mean_sphere)
print("sphere standard deviation:", std_sphere)

mean_cube = np.mean(Phi_arr[1])
mean_list.append(mean_cube)
std_cube = np.std(Phi_arr[1])
std_list.append(std_cube)
print("cube mean:", mean_cube)
print("cube standard deviation:", std_cube)

mean_cuboid = np.mean(Phi_arr[2])
mean_list.append(mean_cuboid)
std_cuboid = np.std(Phi_arr[2])
std_list.append(std_cuboid)
print("cuboid mean:", mean_cuboid)
print("cuboid standard deviation:", std_cuboid)

mean_pyramid = np.mean(Phi_arr[3])
mean_list.append(mean_pyramid)
std_pyramid = np.std(Phi_arr[3])
std_list.append(std_pyramid)
print("pyramid mean:", mean_cuboid)
print("pyramid standard deviation:", std_pyramid)'''

'''mean_cuboid = np.mean(phi_cuboid)
mean_list.append(mean_cuboid)
std_cuboid = np.std(phi_cuboid)
std_list.append(std_cuboid)
print("cube mean:", mean_cuboid)
print("cube standard deviation:", std_cuboid)'''

print("Max Rock Diameter: ", MAX_ROCK_D)

for kk in range(len(Geometries)):
    plt.hist(Phi_arr[kk], bins = 30, density = True, label = Geometries[kk])

#plt.hist(Phi_arr[0], bins = 10, density = True, label = 'Spheres')
#plt.hist(Phi_arr[1], bins = 10, density = True, label = 'Cubes')
#plt.hist(Phi_arr[2], bins = 10, density = True, label = 'Cuboid')
#plt.hist(Phi_arr[3], bins = 10, density = True, label = 'Pyramid')
#plt.hist(phi_cuboid, bins = 10, density = True, label = 'Cuboid')
plt.legend()

plt.title('Occupied volume fraction')
plt.show()

#plt.plot(V, std_list)
#plt.show()