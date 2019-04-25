import numpy as np
import sys,os

dMnu = 0.025

k, Pk0 = np.loadtxt('0.00eV.txt',  unpack=True)
k, Pk1 = np.loadtxt('0.025eV.txt', unpack=True)
k, Pk2 = np.loadtxt('0.050eV.txt', unpack=True)
k, Pk3 = np.loadtxt('0.075eV.txt', unpack=True)
k, Pk4 = np.loadtxt('0.100eV.txt', unpack=True)
k, Pk5 = np.loadtxt('0.125eV.txt', unpack=True)

der1 = (Pk1 - Pk0)/dMnu
der2 = (-Pk2 + 4.0*Pk1 -3.0*Pk0)/(2.0*dMnu)
der3 = (2.0*Pk3 - 9.0*Pk2 + 18.0*Pk1 - 11.0*Pk0)/(6.0*dMnu)
der4 = (-3.0*Pk4 + 16.0*Pk3 - 36.0*Pk2 + 48.0*Pk1 - 25.0*Pk0)/(12.0*dMnu)
der5 = (12.0*Pk5 - 75.0*Pk4 + 200.0*Pk3 - 300.0*Pk2 + 300.0*Pk1 - 137.0*Pk0)/(60.0*dMnu)

np.savetxt('der_Pkmm_dMnu.txt', np.transpose([k,der1,der2,der3,der4,der5]))
