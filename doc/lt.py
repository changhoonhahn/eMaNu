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
    k = np.logspace(-3., 1, 500) 
    # read in paco's calculations 
    k_paco, dPm_paco = np.loadtxt(os.path.join(UT.dat_dir(), 'paco', 'der_Pkmm_dMnu.txt'), unpack=True, usecols=[0,5]) 
    k_ema, dPm_ema = _dPmdMnu_ema(npoints=5)
    dPm_pema = _dPmdMnu_pema(k, npoints=5) 

    fig = plt.figure(figsize=(8,8))
    sub = fig.add_subplot(111)
    for n in [3,4,5, 'quijote']: 
        dPm = dPmdMnu(k, npoints=n) 
        sub.plot(k, np.abs(dPm), lw=0.75, label='w/ %s points' % str(n)) 
    sub.plot(k_paco, np.abs(dPm_paco), c='k', lw=0.75, ls='--', label="Paco's") 
    sub.plot(k_ema, np.abs(dPm_ema), c='k', lw=0.75, ls=':', label="Ema's") 
    #sub.plot(k, np.abs(dPm_pema), c='k', lw=0.75, ls='-.', label="pseudo Ema's") 
    sub.legend(loc='upper right', fontsize=15) 
    sub.set_xlabel('$k$', fontsize=20) 
    sub.set_xscale('log') 
    sub.set_xlim(1e-3, 10.) 
    sub.set_ylabel(r'$|dP_m/d M_\nu|$', fontsize=20) 
    sub.set_yscale('log') 
    sub.set_ylim(1e-1, 1e5) 
    fig.savefig(os.path.join(UT.fig_dir(), 'dPmdMnu.class.png'), bbox_inches='tight') 
    return None 


def compare_Pm(): 
    ''' compare linear theory P_m(k) 
    '''
    k = np.logspace(-3., 1, 500) 

    fig = plt.figure(figsize=(8,8))
    sub = fig.add_subplot(111)
    Pms, Pms_ema, Pms_pema,Pms_paco = [], [], [], [] 
    for i_nu, mnu in enumerate([0.0, 0.025, 0.05, 0.075, 0.1, 0.125]): 
        k_ema, Pm_ema = _Pm_Mnu_ema(mnu) # ema's Pm
        Pm_pema = _Pm_Mnu_pema(mnu, k) 
        k_paco, Pm_paco = _Pm_Mnu_paco(mnu)
        Pm = _Pm_Mnu(mnu, k) 
        sub.plot(k, k * Pm * 0.6**i_nu, 
                lw=1, c='C%i' % i_nu, label='%.3feV' % mnu) 
        sub.plot(k_ema, k_ema * Pm_ema * 0.6**i_nu, 
                lw=1, c='k', ls=':', label=["Ema's", None][bool(i_nu)]) 
        sub.plot(k_paco, k_paco * Pm_paco * 0.6**i_nu, 
                lw=0.5, c='k', ls='--', label=["Paco's", None][bool(i_nu)]) 

        Pms.append(Pm) 
        Pms_ema.append(Pm_ema) 
        Pms_pema.append(Pm_pema) 
        Pms_paco.append(Pm_paco) 

    sub.legend(loc='upper right', fontsize=15) 
    sub.set_xlabel('$k$', fontsize=20) 
    sub.set_xscale('log') 
    sub.set_xlim(1e-2, 1.) 
    sub.set_ylabel(r'$k P_m(k)$', fontsize=20) 
    sub.set_yscale('log') 
    sub.set_ylim(1e1, 1e3) 
    fig.savefig(os.path.join(UT.fig_dir(), 'Pm_Mnu.class.png'), bbox_inches='tight') 

    fig = plt.figure(figsize=(8,8))
    sub = fig.add_subplot(211)
    for i_nu, mnu in enumerate([0.0, 0.025, 0.05, 0.075, 0.1, 0.125]): 
        Pm_ema = np.interp(k, k_ema, Pms_ema[i_nu]) # ema's Pm
        Pm_pema = Pms_pema[i_nu] # ema's Pm
        Pm = Pms[i_nu] 
        sub.plot(k, Pm_ema / Pm, lw=1, c='C%i' % i_nu, label="Ema's %.3feV" % mnu) 
        sub.plot(k, Pm_pema / Pm, lw=1, c='C%i' % i_nu, ls=':') 
    sub.plot(k, np.ones(len(k)), lw=1, c='k')
    sub.legend(loc='upper right', fontsize=15) 
    sub.set_xscale('log') 
    sub.set_xlim(1e-3, 10.) 
    sub.set_ylabel(r'$P_m(k)$ ratio', fontsize=20) 
    sub.set_ylim(0.98, 1.02) 

    sub = fig.add_subplot(212)
    for i_nu, mnu in enumerate([0.0, 0.025, 0.05, 0.075, 0.1, 0.125]): 
        Pm_paco = np.interp(k, k_paco, Pms_paco[i_nu]) # paco's Pm
        Pm = Pms[i_nu] 
        sub.plot(k, Pm_paco / Pm, 
                lw=1, c='C%i' % i_nu, label="Paco's %.3feV" % mnu) 
    sub.plot(k, np.ones(len(k)), lw=1, c='k')
    sub.legend(loc='upper right', fontsize=15) 
    sub.set_xlabel('$k$', fontsize=20) 
    sub.set_xscale('log') 
    sub.set_xlim(1e-3, 10.) 
    sub.set_ylabel(r'$P_m(k)$ ratio', fontsize=20) 
    sub.set_ylim(0.98, 1.02) 
    fig.savefig(os.path.join(UT.fig_dir(), 'Pm_Mnu.class.ratio.png'), bbox_inches='tight') 
    return None 


def dPmdMnu(k, npoints=5): 
    '''calculate derivatives of the linear theory matter power spectrum  
    at 0.0 eV using npoints different points 
    '''
    if npoints == 1: 
        mnus = [0.0, 0.025]
        coeffs = [-1., 1.]
        fdenom = 1. * 0.025
    elif npoints == 2:  
        mnus = [0.0, 0.025, 0.05]
        coeffs = [-3., 4., -1.]
        fdenom = 2. * 0.025
    elif npoints == 3:  
        mnus = [0.0, 0.025, 0.05, 0.075]
        coeffs = [-11., 18., -9., 2.]
        fdenom = 6. * 0.025
    elif npoints == 4:  
        mnus = [0.0, 0.025, 0.05, 0.075, 0.1]
        coeffs = [-25., 48., -36., 16., -3.]
        fdenom = 12.* 0.025
    elif npoints == 5:  
        mnus = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125]
        coeffs = [-137., 300., -300., 200., -75., 12.]
        fdenom = 60.* 0.025
    elif npoints == 'quijote': 
        mnus = [0.0, 0.1, 0.2, 0.4]
        coeffs = [-21., 32., -12., 1.]
        fdenom = 1.2 
    else: 
        raise ValueError
    
    dPm = np.zeros(len(k))
    for mnu, coeff in zip(mnus, coeffs): 
        dPm += coeff * _Pm_Mnu(mnu, k) 
    dPm /= fdenom 
    return dPm 


def _Pm_Mnu(mnu, karr): 
    ''' linear theory matter power spectrum with fiducial parameters and input Mnu for k
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


def _dPmdMnu_ema(npoints=5): 
    '''calculate derivatives of the linear theory matter power spectrum  
    at 0.0 eV using npoints different points with Ema's CLASS outputs 
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
    
    for i, mnu, coeff in zip(range(len(mnus)), mnus, coeffs): 
        k, pmk = _Pm_Mnu_ema(mnu) 
        if i == 0: dPm = np.zeros(len(k))
        dPm += coeff * pmk 
    dPm /= fdenom * 0.025
    return k, dPm 


def _Pm_Mnu_ema(mnu): 
    ''' read in Ema's class Pm(k, z=0) 
    '''
    if mnu == 0.0: 
        fema = os.path.join(UT.dat_dir(), 'ema', 'HADES_0p00eV_z1_pk.dat')
        fAs = 0.9818992345104124**2
    elif mnu == 0.025: 
        fema = os.path.join(UT.dat_dir(), 'ema', 'HADES_0p025eV_z1_pk.dat')
        fAs = 0.9879232989481492**2
    elif mnu == 0.05: 
        fema = os.path.join(UT.dat_dir(), 'ema', 'HADES_0p05eV_z1_pk.dat')
        fAs = 0.9961075140612549**2
    elif mnu == 0.075: 
        fema = os.path.join(UT.dat_dir(), 'ema', 'HADES_0p075eV_z1_pk.dat')
        fAs = 1.00415885618811**2
    elif mnu == 0.1: 
        fema = os.path.join(UT.dat_dir(), 'ema', 'HADES_0p10eV_z1_pk.dat')
        fAs = 1.0123494887789048**2
    elif mnu == 0.125: 
        fema = os.path.join(UT.dat_dir(), 'ema', 'HADES_0p125eV_z1_pk.dat')
        fAs = 1.0205994937246368**2
    k, pmk = np.loadtxt(fema, unpack=True, skiprows=4, usecols=[0,1]) 
    return k, fAs*pmk 


def _dPmdMnu_pema(k, npoints=5): 
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
        dPm += coeff * _Pm_Mnu_pema(mnu, k) 
    dPm /= fdenom * 0.025
    return dPm 


def _Pm_Mnu_pema(mnu, karr): 
    ''' linear theory matter power spectrum with fiducial parameters and input Mnu for k
    with ema's parameters
    '''
    h = 0.6711 
    Ocdm_dict = {0.: 0.2685, 0.025: 0.2681, 0.05: 0.2673, 0.075: 0.2667, 0.1: 0.2661, 0.125: 0.2655} 
    Neff_dict = {0.: 3.046, 0.025: 2.0328, 0.05: 2.0328, 0.075: 2.0328, 0.1: 2.0328, 0.125: 2.0328} 
    f_As_dict = {
            0.: 0.9818992345104124**2, 
            0.025: 0.9879232989481492**2, 
            0.05: 0.9961075140612549**2, 
            0.075: 1.00415885618811**2, 
            0.1: 1.0123494887789048**2,
            0.125: 1.0205994937246368**2}
    
    if mnu > 0.: 
        params = {
                'output': 'mPk', 
                'Omega_cdm': Ocdm_dict[mnu],
                'Omega_b': 0.049, 
                'Omega_k': 0.0, 
                'h': h, 
                'n_s': 0.9624, 
                'k_pivot': 0.05, 
                'A_s': 2.215e-9,
                'N_eff': Neff_dict[mnu], 
                'N_ncdm': 1, 
                'm_ncdm': mnu, #eV
                'P_k_max_1/Mpc':20.0, 
                'z_pk': 0.
                }
    else: 
        params = {
                'output': 'mPk', 
                'Omega_cdm': Ocdm_dict[mnu],
                'Omega_b': 0.049, 
                'Omega_k': 0.0, 
                'h': h, 
                'n_s': 0.9624, 
                'k_pivot': 0.05, 
                'A_s': 2.215e-9,
                'N_eff': Neff_dict[mnu], 
                'P_k_max_1/Mpc':20.0, 
                'z_pk': 0.
                }
    cosmo = Class()
    cosmo.set(params) 
    cosmo.compute() 
    pmk = np.array([cosmo.pk_lin(k*h, 0.)*h**3 for k in karr]) 
    return f_As_dict[mnu] * pmk 


def _Pm_Mnu_paco(mnu): 
    ''' read in Paco's CAMB Pm(k, z=0) 
    '''
    if mnu == 0.0: 
        fpaco = os.path.join(UT.dat_dir(), 'paco', '0.00eV.txt')
    elif mnu == 0.025: 
        fpaco = os.path.join(UT.dat_dir(), 'paco', '0.025eV.txt')
    elif mnu == 0.05: 
        fpaco = os.path.join(UT.dat_dir(), 'paco', '0.050eV.txt')
    elif mnu == 0.075: 
        fpaco = os.path.join(UT.dat_dir(), 'paco', '0.075eV.txt')
    elif mnu == 0.1: 
        fpaco = os.path.join(UT.dat_dir(), 'paco', '0.100eV.txt')
    elif mnu == 0.125: 
        fpaco = os.path.join(UT.dat_dir(), 'paco', '0.125eV.txt')
    k, pmk = np.loadtxt(fpaco, unpack=True, usecols=[0,1]) 
    return k ,pmk 


if __name__=="__main__": 
    compare_Pm()
    #compare_dPmdMnu()
