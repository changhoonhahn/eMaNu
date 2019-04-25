import numpy as np
import CAMB_library as CL
import sys,os


################################## INPUT ######################################
Omega_m = 0.3175
Omega_b = 0.049
h       = 0.6711
ns      = 0.9624
s8      = 0.834
Mnu     = 0.125 #eV

As      = 2.13e-9
Omega_k = 0.0
tau     = None

hierarchy = 'degenerate' #'degenerate', 'normal', 'inverted'
Nnu       = 3  #number of massive neutrinos
Neff      = 3.046

pivot_scalar = 0.05
pivot_tensor = 0.05

redshifts    = [0.0] 
kmax         = 10.0
k_per_logint = 10
###############################################################################

Pk = CL.PkL(Omega_m, Omega_b, h, ns, s8, Mnu, As, Omega_k,
            pivot_scalar, pivot_tensor, Nnu, hierarchy, Neff, tau, redshifts,
            kmax, k_per_logint)

np.savetxt('0.125eV.txt', np.transpose([Pk.k, Pk.Pkmm[0]]))
