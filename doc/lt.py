'''

code for linear theory predictions

'''
import os 
import numpy as np 
from classy import Class
# --- eMaNu --- 
from emanu import util as UT
from emanu import obvs as Obvs
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
    fig = plt.figure(figsize=(8,8))
    sub = fig.add_subplot(111)
    #for n in [3,4,5, 'quijote']: 
    #    dPm = dPmdMnu(k, npoints=n) 
    #    sub.plot(k, np.abs(dPm), lw=0.75, label='w/ %s points' % str(n)) 
    for i_n, n in enumerate([3, 4, 5]): 
        k_ema, dPm_ema = _dPmdMnu_ema(npoints=n)
        sub.plot(k_ema, dPm_ema, lw=0.75, ls=':', label="Ema's %i pts" % n) 
    for i_n, n in enumerate([1, 2, 3]):#, 4]): 
        k_class, dPm_class = _dPmdMnu_class(npoints=n)
        sub.plot(k_class, dPm_class, c='C%i' % i_n, lw=1, ls='-', label="hi prec. CLASS %i pts" % n) 
    sub.plot(k_paco, dPm_paco, c='r', lw=0.75, ls='-', label="Paco's") 
    sub.legend(loc='lower left', fontsize=15) 
    sub.set_xlabel('$k$', fontsize=20) 
    sub.set_xscale('log') 
    sub.set_xlim(1e-3, 10.) 
    sub.set_ylabel(r'$dP_m/d M_\nu$', fontsize=20) 
    sub.set_yscale('symlog') 
    #sub.set_ylim(1e-1, 1e5) 
    fig.savefig(os.path.join(UT.fig_dir(), 'dPmdMnu.class.png'), bbox_inches='tight') 
    return None 


def compare_Pm(): 
    ''' compare linear theory P_m(k) 
    '''
    mnus = [0.0, 0.025, 0.05, 0.075]#, 0.1, 0.125]

    fig = plt.figure(figsize=(8,8))
    sub = fig.add_subplot(111)
    for i_nu, mnu in enumerate(mnus): 
        k_ema, Pm_ema = _Pm_Mnu_ema(mnu) 
        k_paco, Pm_paco = _Pm_Mnu_paco(mnu)
        k_class, Pm_class = _Pm_Mnu_class(mnu) 
        sub.plot(k_class, k_class * Pm_class * 0.6**i_nu, 
                lw=0.5, c='k', ls='--', label=["CLASS hi prec.", None][bool(i_nu)]) 
        sub.plot(k_ema, k_ema * Pm_ema * 0.6**i_nu, 
                lw=1, c='k', ls=':', label=["Ema's", None][bool(i_nu)]) 
        sub.plot(k_paco, k_paco * Pm_paco * 0.6**i_nu, 
                lw=0.5, c='k', ls='--', label=["Paco's", None][bool(i_nu)]) 
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
    for i_nu, mnu in enumerate(mnus): 
        k_ema, Pm_ema = _Pm_Mnu_ema(mnu) 
        k_class, Pm_class = _Pm_Mnu_class(mnu) 
        _Pm_ema = np.interp(k_class, k_ema, Pm_ema) # ema's Pm
        sub.plot(k_class, _Pm_ema / Pm_class, lw=1, c='C%i' % i_nu, label="Ema's %.3feV" % mnu) 
    sub.plot(k_class, np.ones(len(k_class)), lw=1, c='k')
    sub.legend(loc='upper right', fontsize=15) 
    sub.set_xscale('log') 
    sub.set_xlim(1e-3, 10.) 
    sub.set_ylim(0.98, 1.02) 

    sub = fig.add_subplot(212)
    for i_nu, mnu in enumerate(mnus): 
        k_paco, Pm_paco = _Pm_Mnu_paco(mnu) 
        k_class, Pm_class = _Pm_Mnu_class(mnu) 
        _Pm_paco = np.interp(k_class, k_paco, Pm_paco) # ema's Pm
        sub.plot(k_class, _Pm_paco / Pm_class, 
                lw=1, c='C%i' % i_nu, label="Paco's %.3feV" % mnu) 
    sub.plot(k_class, np.ones(len(k_class)), lw=1, c='k')
    sub.legend(loc='upper right', fontsize=15) 
    sub.set_xscale('log') 
    sub.set_xlim(1e-3, 10.) 
    sub.set_ylabel(r'$P_m(k)/P^{\rm CLASS~hi.prec.}_m$ ratio', fontsize=20) 
    sub.set_ylim(0.98, 1.02) 
    fig.savefig(os.path.join(UT.fig_dir(), 'Pm_Mnu.class.ratio.png'), bbox_inches='tight') 
    return None 


def compare_PmMnu_PmLCDM(): 
    ''' compare (P_m(k) Mnu > 0)/(P_m(k) LCDM) 
    '''
    mnus = [0.025, 0.05, 0.075, 0.1, 0.125] 
    k_fid, Pm_fid = _Pm_Mnu_class(0.) 
    k_ema_fid, Pm_ema_fid = _Pm_Mnu_ema(0.) 

    fig = plt.figure(figsize=(8,8))
    sub = fig.add_subplot(111)
    for i_nu, mnu in enumerate(mnus): 
        k_ema, Pm_ema = _Pm_Mnu_ema(mnu) 
        k_class, Pm_class = _Pm_Mnu_class(mnu) 
        sub.plot(k_class, Pm_class / Pm_fid, 
                lw=0.5, c='C%i' % i_nu, label=["CLASS hi prec.", None][bool(i_nu)]) 
        sub.plot(k_ema, Pm_ema / Pm_ema_fid, 
                lw=1, c='C%i' % i_nu, ls=':', label=["Ema's", None][bool(i_nu)]) 
    sub.legend(loc='upper right', fontsize=15) 
    sub.set_xlabel('$k$', fontsize=20) 
    sub.set_xscale('log') 
    sub.set_xlim(1e-6, 1.) 
    #sub.set_yscale('log') 
    sub.set_ylim(0.8, 1.2) 
    fig.savefig(os.path.join(UT.fig_dir(), 'Pm_Mnu_Pm_LCDM.class.png'), bbox_inches='tight') 
    return None 


def compare_dPdthetas():
    ''' Plot the derivatives of Pm w.r.t. to the different parameters 
    '''
    thetas = ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu'] 
    theta_lbls = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$']

    k = np.logspace(-3., 1, 200) 
    dpdts = []  
    for tt in thetas:  
        if tt != 'Mnu': 
            dpdts.append(dPmdtheta(tt, k)) 
        else: 
            dpdts.append(dPmdMnu(k, npoints=5)) 

    fig = plt.figure(figsize=(7,8))
    sub = fig.add_subplot(111)

    for dpdt, lbl in zip(dpdts, theta_lbls): 
        if dpdt.min() < 0: 
            sub.plot(k, np.abs(dpdt), ls=':', label=lbl) 
        else: 
            sub.plot(k, dpdt, label=lbl) 

    sub.legend(loc='upper right', ncol=2, fontsize=15) 
    sub.set_xlabel('$k$', fontsize=25) 
    sub.set_xscale('log') 
    sub.set_xlim(1e-2, 1.) 
    sub.set_ylabel(r'$|{\rm d}P_m/d\theta|$', fontsize=25) 
    sub.set_yscale('log') 
    ffig = os.path.join(UT.fig_dir(), 'dPmdthetas.class.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def _PmFij(tt_i, tt_j): 
    ''' calculate fisher matrix for Pm
    '''
    # get d Pm(k) / d thetas
    k_i, dpdtti = dPmdtheta_class(tt_i) 
    k_j, dpdttj = dPmdtheta_class(tt_i) 
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


def _dPmdMnu_class(npoints=5): 
    '''calculate derivatives of the linear theory matter power spectrum  
    at 0.0 eV using npoints different points with high precision CLASS outputs 
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
        k, pmk = _Pm_Mnu_class(mnu) 
        if i == 0: dPm = np.zeros(len(k))
        dPm += coeff * pmk 
    dPm /= fdenom * 0.025
    return k, dPm 


def _Pm_Mnu_class(mnu): 
    ''' linear theory matter power spectrum with fiducial parameters and input Mnu
    high precision calculation from CLASS 
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
    k, pmk = np.loadtxt(fema, unpack=True, skiprows=4, usecols=[0,1]) 
    return k, pmk 


def dPmdtheta(theta, k): 
    ''' derivative of Pm w.r.t. to theta 
    '''
    h_dict = {'Ob': 0.002, 'Om': 0.02, 'h': 0.04, 'ns': 0.04, 's8': 0.03} 
    h = h_dict[theta]
    Pm_m = _Pm_theta('%s_m' % theta, k) 
    Pm_p = _Pm_theta('%s_p' % theta, k) 
    return (Pm_p - Pm_m)/h 


def _Pm_theta(theta, karr): 
    ''' linear theory matter power spectrum with fiducial parameters except theta for k 
    '''
    h = 0.6711 
    assert 'Mnu' not in theta 
    # fiducial LCDM 
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
    if theta == 'fiducial': 
        pass 
    elif theta == 'Ob_m': 
        params['Omega_b'] = 0.048
    elif theta == 'Ob_p': 
        params['Omega_b'] = 0.050
    elif theta == 'Om_m': 
        params['Omega_cdm'] = 0.3075 - 0.049
    elif theta == 'Om_p': 
        params['Omega_cdm'] = 0.3275 - 0.049
    elif theta == 'h_m': 
        params['h'] = 0.6911
        h = 0.6911
    elif theta == 'h_p':
        params['h'] = 0.6511
        h = 0.6511
    elif theta == 'ns_m': 
        params['n_s'] = 0.9424
    elif theta == 'ns_p': 
        params['n_s'] = 0.9824
    elif theta == 's8_m': 
        params['sigma8'] = 0.819
    elif theta == 's8_p': 
        params['sigma8'] = 0.849
    else: 
        raise ValueError
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
    compare_PmMnu_PmLCDM()
    compare_dPmdMnu()
    #compare_dPdthetas() 
    #CovP_gauss()
