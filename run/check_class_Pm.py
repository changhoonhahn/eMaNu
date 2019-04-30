'''

Compare LT Pm computed using CLASSY python 
wrapper  versus the CLASS package. The main 
difference is in the precision.

'''
import os 
import numpy as np 
from classy import Class
# --- emanu --- 
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


def Pm_classy(mnu, karr): 
    ''' linear theory matter P(k) with fiducial parameters and input Mnu for k
    computed by CLASSY python package
    '''
    h = 0.6711 
    if mnu > 0.: 
        params = {
                'output': 'mPk', 
                'Omega_cdm': 0.3175 - 0.049 - mnu/93.14/h**2,
                'Omega_b': 0.049, 
                'Omega_k': 0.0, 
                'h': h, 
                'n_s': 0.9624, 
                'k_pivot': 0.05, 
                'sigma8': 0.834,
                'N_eff': 0.00641, 
                'N_ncdm': 1, 
                'deg_ncdm': 3.0, 
                'm_ncdm': mnu/3., #eV
                'P_k_max_1/Mpc':20.0, 
                'z_pk': 0.
                }
    else: 
        params = {
                'output': 'mPk', 
                'Omega_cdm': 0.3175 - 0.049,
                'Omega_b': 0.049, 
                'Omega_k': 0.0, 
                'h': h, 
                'n_s': 0.9624, 
                'k_pivot': 0.05, 
                'sigma8': 0.834,
                'N_eff': 3.046, 
                'P_k_max_1/Mpc':20.0, 
                'z_pk': 0.
                }
    cosmo = Class()
    cosmo.set(params) 
    cosmo.compute() 
    pmk = np.array([cosmo.pk_lin(k*h, 0.)*h**3 for k in karr]) 
    return pmk 


def Pm_class(mnu): 
    ''' linear theory matter P(k) with fiducial parameters and input Mnu
    high precision calculation from CLASS package 
    '''
    if mnu == 0.0: 
        fema = os.path.join(UT.dat_dir(), 'lt', 'output', '0eV_pk.dat')
    elif mnu == 0.025: 
        fema = os.path.join(UT.dat_dir(), 'lt', 'output', '0p025eV_pk.dat')
    elif mnu == 0.05: 
        fema = os.path.join(UT.dat_dir(), 'lt', 'output', '0p05eV_pk.dat')
    elif mnu == 0.075: 
        fema = os.path.join(UT.dat_dir(), 'lt', 'output', '0p075eV_pk.dat')
    elif mnu == 0.1: 
        fema = os.path.join(UT.dat_dir(), 'lt', 'output', '0p1eV_pk.dat')
    elif mnu == 0.125: 
        fema = os.path.join(UT.dat_dir(), 'lt', 'output', '0p125eV_pk.dat')
    elif mnu == 0.2: 
        fema = os.path.join(UT.dat_dir(), 'lt', 'output', '0p2eV_pk.dat')
    elif mnu == 0.4: 
        fema = os.path.join(UT.dat_dir(), 'lt', 'output', '0p4eV_pk.dat')
    k, pmk = np.loadtxt(fema, unpack=True, skiprows=4, usecols=[0,1]) 
    return k, pmk 


if __name__=="__main__": 
    # class pm 
    _, pm_lcdm_class = Pm_class(0.0) 
    k, pm_ncdm_class = Pm_class(0.1) 
    klim = (k > 1e-3) 

    # classy Pm
    pm_lcdm_classy = Pm_classy(0.0, k[klim]) 
    pm_ncdm_classy = Pm_classy(0.1, k[klim]) 
    
    # ratio of Pm 
    fig = plt.figure(figsize=(6,6))
    sub = fig.add_subplot(111)
    sub.plot(k[klim], pm_lcdm_class[klim]/pm_lcdm_classy, c='C0', label='LCDM') 
    sub.plot(k[klim], pm_ncdm_class[klim]/pm_ncdm_classy, c='C1', label='0.1eV') 
    sub.legend(loc='upper right', fontsize=15) 
    sub.set_xlabel('$k$', fontsize=20) 
    sub.set_xscale('log') 
    sub.set_xlim(1e-3, 1.) 
    sub.set_ylabel(r'($P_m$ CLASS)/($P_m$ CLASSY)', fontsize=20) 
    sub.set_ylim(0.99, 1.01) 
    fig.savefig(os.path.join(UT.fig_dir(), '_Pm.class.classy.png'), bbox_inches='tight') 
    
    # class, classy d Pm / d Mnu  
    mnus = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125]
    coeffs = [-137., 300., -300., 200., -75., 12.]
    fdenom = 60.* 0.025
    dPm_class, dPm_classy = np.zeros(len(k)), np.zeros(np.sum(klim))
    for mnu, coeff in zip(mnus, coeffs): 
        _, pm_class = Pm_class(mnu)     # CLASS
        pm_classy = Pm_classy(mnu, k[klim])   # CLASSy 
        dPm_class += coeff * pm_class
        dPm_classy += coeff * pm_classy
    dPm_class /= fdenom 
    dPm_classy /= fdenom 

    fig = plt.figure(figsize=(6,6))
    sub = fig.add_subplot(111)
    sub.plot(k[klim], dPm_class[klim], c='k', label="CLASS") 
    sub.plot(k[klim], dPm_classy, c='C1', label="CLASSy")
    sub.legend(loc='lower left', fontsize=15) 
    sub.set_xlabel('$k$', fontsize=20) 
    sub.set_xscale('log') 
    sub.set_xlim(1e-3, 10.) 
    sub.set_ylabel(r'$dP_m/d M_\nu$', fontsize=20) 
    sub.set_yscale('symlog') 
    fig.savefig(os.path.join(UT.fig_dir(), '_dPmdMnu.class.classy.png'), bbox_inches='tight') 
