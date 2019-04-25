'''

code for linear theory predictions

'''
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


def compare_dPmdMnu(): 
    ''' compare the derivatives computed with different number of points using 
    finite differences
    '''
    # read in paco's calculations 
    k_paco, dPm_paco = np.loadtxt(os.path.join(UT.dat_dir(), 'CAMB_test', 'der_Pkmm_dMnu.txt'), unpack=True, usecols=[0,5]) 

    k = np.logspace(-3., 1, 500) 

    fig = plt.figure()
    sub = fig.add_subplot(111)
    for n in [1, 2, 3, 4, 5]: 
        dPm = dPmdMnu(k, npoints=n) 
        sub.plot(k, np.abs(dPm), lw=0.5, label='w/ %i points' % n) 
    sub.plot(k_paco, np.abs(dPm_paco), c='k', ls='--', label="Paco's") 
    sub.legend(loc='upper right', fontsize=15) 
    sub.set_xlabel('$k$', fontsize=20) 
    sub.set_xscale('log') 
    sub.set_xlim(1e-3, 10.) 
    sub.set_ylabel(r'$|dP_m/d M_\nu|$', fontsize=20) 
    sub.set_yscale('log') 
    sub.set_ylim(1e-1, 1e4) 
    fig.savefig(os.path.join(UT.fig_dir(), 'dPmdMnu.class.png'), bbox_inches='tight') 
    return None 


def dPmdMnu(k, npoints=5): 
    '''calculate derivatives of the linear theory matter power spectrum  
    at 0.0 eV using npoints different points 
    '''
    if npoints == 1: 
        mnus = [0.0, 0.025]
        coeffs = [-1., 1.]
        fdenom = 1.
    elif npoints == 2:  
        mnus = [0.0, 0.025, 0.05]
        coeffs = [-3., 4., -1.]
        fdenom = 2.
    elif npoints == 3:  
        mnus = [0.0, 0.025, 0.05, 0.075]
        coeffs = [-11., 18., -9., 2.]
        fdenom = 6.
    elif npoints == 4:  
        mnus = [0.0, 0.025, 0.05, 0.075, 0.1]
        coeffs = [-25., 48., -36., 16., -3.]
        fdenom = 12.
    elif npoints == 5:  
        mnus = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125]
        coeffs = [-137., 300., -300., 200., -75., 12.]
        fdenom = 60.
    else: 
        raise ValueError
    
    dPm = np.zeros(len(k))
    for mnu, coeff in zip(mnus, coeffs): 
        dPm += coeff * _Pm_Mnu(mnu, k) 
    dPm /= fdenom * 0.025
    return dPm 


def _Pm_Mnu(mnu, karr): 
    ''' linear theory matter power spectrum with fiducial parameters and input Mnu for k
    '''
    h = 0.6711 
    params = {
            'output': 'mPk', 
            'Omega_cdm': 0.3175 - 0.049 - mnu/93.14/h**2,
            'Omega_b': 0.049, 
            'Omega_k': 0.0, 
            'h': h, 
            'n_s': 0.9624, 
            'k_pivot': 0.05*h, 
            'sigma8': 0.834,
            'N_eff': 0.00641, 
            'N_ncdm': 1, 
            'deg_ncdm': 3.0, 
            'm_ncdm': mnu/3., #eV
            'P_k_max_1/Mpc':20.0, 
            'z_pk': 0. 
            }
    cosmo = Class()
    cosmo.set(params) 
    cosmo.compute() 
    pmk = np.array([cosmo.pk_lin(k*h, 0.)*h**3 for k in karr]) 
    return pmk 


if __name__=="__main__": 
    compare_dPmdMnu()
