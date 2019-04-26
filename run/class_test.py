import os 
import numpy as np 
from classy import Class
# --- eMaNu --- 
from emanu import util as UT
# --- plotting --- 
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.xmargin'] = 1
mpl.rcParams['xtick.labelsize'] = 'x-large'
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.labelsize'] = 'x-large'
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['legend.frameon'] = False


def _Pm(): 
    ''' simple test of whether I can get the matter powerspectrum at z=0 
    for Mnu > 0 case. 
    '''
    fig = plt.figure()
    sub = fig.add_subplot(111)

    h = 0.6711 
    for i_nu, mnu in enumerate([0.0, 0.025, 0.075, 0.125]): 
        params = {
                'output': 'mPk', 
                'Omega_cdm': 0.3175 - 0.049 - mnu/93.14/h**2,
                'Omega_b': 0.049, 
                'Omega_k': 0.0, 
                'h': h, 
                'n_s': 0.9624, 
                #'A_s': 2.13e-9, 
                'k_pivot': 0.05*h, 
                'sigma8': 0.834,
                'N_eff': 2.0328, # don't change this.
                'N_ncdm': 1, 
                'deg_ncdm': 1.0, 
                'm_ncdm': mnu, #eV
                'P_k_max_1/Mpc':10.0, 
                'z_pk': 0. 
                }
        cosmo = Class()
        cosmo.set(params) 
        cosmo.compute() 

        # read in Paco's Pm(k) 
        if mnu in [0.025, 0.05, 0.075, 0.10, 0.125]: 
            fpaco = os.path.join(UT.dat_dir(), 'paco', '%.3feV.txt' % mnu)
        elif mnu == 0.0: 
            fpaco = os.path.join(UT.dat_dir(), 'paco', '0.00eV.txt')
        k_paco, pmk_paco = np.loadtxt(fpaco, unpack=True, usecols=[0,1])
    
        pmk = np.array([cosmo.pk_lin(k*h, 0.)*h**3 for k in k_paco]) 
        sub.plot(k_paco, (1. - i_nu * 0.25) * pmk, c='C'+str(i_nu))
        sub.plot(k_paco, (1. - i_nu * 0.25) * pmk_paco, c='k', ls='--')
        print (pmk/pmk_paco)[:5], (pmk/pmk_paco)[-5:]
    sub.set_xlabel('$k$', fontsize=25) 
    sub.set_xscale("log") 
    sub.set_xlim(1e-2, 1)
    sub.set_ylabel('$P_m(k)$', fontsize=25) 
    sub.set_yscale("log") 
    sub.set_ylim(1e2, 5e4) 
    fig.savefig(os.path.join(UT.fig_dir(), 'class_pmk_test.png'), bbox_inches='tight')  
    return None

    
if __name__=="__main__": 
    _Pm() 
