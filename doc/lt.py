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
    fig = plt.figure(figsize=(8,8))
    sub = fig.add_subplot(111)
    # ema's derivative
    for i_n, n in enumerate([3, 4, 5]): 
        k_ema, dPm_ema = _dPmdMnu_ema(npoints=n)
        sub.plot(k_ema, dPm_ema, lw=0.75, ls=':', label="Ema's %i pts" % n) 
    # class derivative
    for i_n, n in enumerate([1, 2, 3, 4, 5, 'quijote']): 
        dPm_class = dPmdMnu(k_ema, npoints=n)
        sub.plot(k_ema, dPm_class, c='C%i' % i_n, lw=1, ls='-', label="CLASS %s pts" % str(n))
    # read in paco's calculations 
    k_paco, dPm_paco = np.loadtxt(os.path.join(UT.dat_dir(), 'paco', 'der_Pkmm_dMnu.txt'), unpack=True, usecols=[0,5]) 
    sub.plot(k_paco, dPm_paco, c='r', lw=0.75, ls='--', label="Paco's") 
    sub.legend(loc='lower left', fontsize=15) 
    sub.set_xlabel('$k$', fontsize=20) 
    sub.set_xscale('log') 
    sub.set_xlim(1e-3, 10.) 
    sub.set_ylabel(r'$dP_m/d M_\nu$', fontsize=20) 
    sub.set_yscale('symlog') 
    fig.savefig(os.path.join(UT.fig_dir(), 'dPmdMnu.class.png'), bbox_inches='tight') 
    return None 


def compare_dPmdMnu_0p1eV(): 
    ''' compare the d P_m(k) / d Mnu at 0.1eV among the various calculations.
    finite differences
    '''
    fig = plt.figure(figsize=(8,8))
    sub = fig.add_subplot(111)
    # ema's derivatve
    _, Pm_ema_m =_Pm_Mnu_ema(0.075) 
    k, Pm_ema_p =_Pm_Mnu_ema(0.125) 
    dPm_ema = (Pm_ema_p - Pm_ema_m)/0.05
    sub.plot(k, dPm_ema, lw=1, ls=':', label="Ema's") 
    # paco's derivative
    _, Pm_paco_m =_Pm_Mnu_paco(0.075) 
    k, Pm_paco_p =_Pm_Mnu_paco(0.125) 
    dPm_paco = (Pm_paco_p - Pm_paco_m)/0.05
    sub.plot(k, dPm_paco, lw=1, ls='--', label="Paco's") 
    # class derivative
    Pm_class_m = _Pm_Mnu(0.075, k) 
    Pm_class_p = _Pm_Mnu(0.125, k) 
    dPm_class = (Pm_class_p - Pm_class_m)/0.05
    sub.plot(k, dPm_class, lw=1, ls='-', label="CLASS (0.075, 0.125)") 
    # class derivative
    Pm_class_m = _Pm_Mnu(0.0, k) 
    Pm_class_p = _Pm_Mnu(0.2, k) 
    dPm_class = (Pm_class_p - Pm_class_m)/0.2
    sub.plot(k, dPm_class, lw=1, ls='-', label="CLASS (0.0, 0.2)") 
    sub.legend(loc='lower left', fontsize=15) 
    sub.set_xlabel('$k$', fontsize=20) 
    sub.set_xscale('log') 
    sub.set_xlim(1e-3, 10.) 
    sub.set_ylabel(r'$dP_m/d M_\nu$ at 0.1eV', fontsize=20) 
    sub.set_yscale('symlog') 
    #sub.set_ylim(1e-1, 1e5) 
    fig.savefig(os.path.join(UT.fig_dir(), 'dPmdMnu_0.1eV.class.png'), bbox_inches='tight') 
    return None 


def compare_Pm(): 
    ''' compare linear theory P_m(k) 
    '''
    mnus = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125]

    fig = plt.figure(figsize=(8,8))
    sub = fig.add_subplot(111)
    for i_nu, mnu in enumerate(mnus): 
        k_ema, Pm_ema = _Pm_Mnu_ema(mnu) 
        k_paco, Pm_paco = _Pm_Mnu_paco(mnu)
        Pm_class = _Pm_Mnu(mnu, k_ema) 
        sub.plot(k_ema, k_ema * Pm_class * 0.6**i_nu, 
                lw=0.5, label=["CLASS hi prec.", None][bool(i_nu)]) 
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
        Pm_class = _Pm_Mnu(mnu, k_ema) 
        sub.plot(k_ema, Pm_ema / Pm_class, lw=1, c='C%i' % i_nu, label="%.3feV" % mnu) 
    sub.plot(k_ema, np.ones(len(k_ema)), lw=1, c='k')
    sub.legend(loc='upper right', ncol=2, fontsize=15) 
    sub.set_xscale('log') 
    sub.set_xlim(1e-3, 10.) 
    sub.set_ylim(0.98, 1.02) 
    sub.set_ylabel(r'$P^{\rm ema}_m(k)/P^{\rm CLASS}_m$ ratio', fontsize=20) 

    sub = fig.add_subplot(212)
    for i_nu, mnu in enumerate(mnus): 
        k_paco, Pm_paco = _Pm_Mnu_paco(mnu) 
        Pm_class = _Pm_Mnu(mnu, k_paco) 
        sub.plot(k_paco, Pm_paco / Pm_class, lw=1, c='C%i' % i_nu) 
    sub.plot(k_paco, np.ones(len(k_paco)), lw=1, c='k')
    sub.set_xscale('log') 
    sub.set_xlim(1e-3, 10.) 
    sub.set_ylabel(r'$P^{\rm ema}_m(k)/P^{\rm CLASS}_m$ ratio', fontsize=20) 
    sub.set_ylim(0.98, 1.02) 
    fig.savefig(os.path.join(UT.fig_dir(), 'Pm_Mnu.class.ratio.png'), bbox_inches='tight') 
    return None 


def compare_PmMnu_PmLCDM(): 
    ''' compare (P_m(k) Mnu > 0)/(P_m(k) LCDM) 
    '''
    mnus = [0.025, 0.05, 0.075, 0.1, 0.125] 
    k_ema, Pm_ema_fid = _Pm_Mnu_ema(0.) 
    Pm_fid = _Pm_Mnu(0., k_ema) 

    fig = plt.figure(figsize=(8,8))
    sub = fig.add_subplot(111)
    for i_nu, mnu in enumerate(mnus): 
        _, Pm_ema = _Pm_Mnu_ema(mnu) 
        Pm_class = _Pm_Mnu(mnu, k_ema) 
        sub.plot(k_ema, Pm_class / Pm_fid, 
                lw=0.5, c='C%i' % i_nu, label=["CLASS hi prec.", None][bool(i_nu)]) 
        sub.plot(k_ema, Pm_ema / Pm_ema_fid, 
                lw=1, c='C%i' % i_nu, ls=':', label=["Ema's", None][bool(i_nu)]) 
    sub.legend(loc='upper right', fontsize=15) 
    sub.set_xlabel('$k$', fontsize=20) 
    sub.set_xscale('log') 
    sub.set_xlim(1e-6, 1.) 
    sub.set_ylim(0.8, 1.2) 
    fig.savefig(os.path.join(UT.fig_dir(), 'Pm_Mnu_Pm_LCDM.class.png'), bbox_inches='tight') 
    return None 


def compare_dPdthetas():
    ''' Plot the derivatives of Pm w.r.t. to the different parameters 
    '''
    thetas = ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu'] 
    theta_lbls = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$']

    k = np.logspace(-3., 10, 500) 
    dpdts = []  
    for tt in thetas:  
        dpm = dPmdtheta(tt, k )
        dpdts.append(dpm) 

    fig = plt.figure(figsize=(6,6))
    sub = fig.add_subplot(111)

    for dpdt, lbl in zip(dpdts, theta_lbls): 
        if dpdt.min() < 0: 
            sub.plot(k, np.abs(dpdt), ls=':', label=lbl) 
        else: 
            sub.plot(k, dpdt, label=lbl) 
    sub.legend(loc='lower left', ncol=2, fontsize=15) 
    sub.set_xlabel('$k$', fontsize=25) 
    sub.set_xscale('log') 
    sub.set_xlim(1e-3, 10.) 
    sub.set_ylabel(r'$|{\rm d}P_m/d\theta|$', fontsize=25) 
    sub.set_yscale('log') 
    ffig = os.path.join(UT.fig_dir(), 'dPmdthetas.class.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def LT_sigma_kmax(npoints=5):
    ''' 
    '''
    thetas = ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
    theta_lbls = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$']

    kmaxs = np.pi/500. * 3 * np.arange(1, 28) 
    sigma_thetas = []
    for i_k, kmax in enumerate(kmaxs): 
        Fij =  _Fij_Pm(np.logspace(-5, 2, 500), kmax=kmax, npoints=npoints) 
        sigma_thetas.append(np.sqrt(np.diag(np.linalg.inv(Fij))))
        print sigma_thetas[-1]
    sigma_thetas = np.array(sigma_thetas)
    #sigma_theta_lims = [(0, 0.1), (0., 0.08), (0., 0.8), (0, 0.8), (0., 0.12), (0., 0.8)]
    #sigma_theta_lims = [(0, 0.2), (0., 0.2), (0., 2.), (0, 2.), (0., 1.), (0., 2.)]
    #sigma_theta_lims = [(0, 10.), (0., 10.), (0., 10.), (0, 10.), (0., 10.), (0., 10.)]
    sigma_theta_lims = [(1e-2, 10.), (1e-3, 10.), (1e-2, 50), (1e-2, 50.), (1e-2, 50.), (1e-2, 50.)]

    fig = plt.figure(figsize=(15,8))
    for i, theta in enumerate(thetas): 
        sub = fig.add_subplot(2,len(thetas)/2,i+1) 
        sub.plot(kmaxs, sigma_thetas[:,i], c='k', ls='-') 
        sub.set_xlim(0.005, 0.5)
        sub.text(0.9, 0.9, theta_lbls[i], ha='right', va='top', transform=sub.transAxes, fontsize=30)
        sub.set_ylim(sigma_theta_lims[i]) 
        sub.set_yscale('log') 
        if i == 0: 
            sub.text(0.5, 0.35, r"$P^{\rm lin}_m$", ha='left', va='bottom', color='k', transform=sub.transAxes, fontsize=24)
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'$k_{\rm max}$', fontsize=28) 
    bkgd.set_ylabel(r'$1\sigma$ constraint on $\theta$', labelpad=10, fontsize=28) 

    fig.subplots_adjust(wspace=0.2, hspace=0.15) 
    ffig = os.path.join(UT.doc_dir(), 'figs', 'LT_sigma_kmax.png')
    fig.savefig(ffig, bbox_inches='tight') 

    fdat = os.path.join(UT.dat_dir(), 'LT_sigma_kmax.dat')
    np.savetxt(fdat, sigma_thetas, delimiter='\t') 
    return None


def _Fij_Pm(k, kmax=0.5, npoints=5): 
    ''' calculate fisher matrix for Pm
    '''
    thetas = ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
    #thetas = ['s8', 'Mnu']
    klim = (k < kmax) 
    k = k[klim] 

    Pm_fid = _Pm_Mnu(0.0, k) 
    factor = 1.e9 / (4. * np.pi**2) 
    
    Fij = np.zeros((len(thetas), len(thetas)))
    for i, tt_i in enumerate(thetas): 
        for j, tt_j in enumerate(thetas): 
            # get d ln Pm(k) / d thetas
            #dlnPdtti = dPmdtheta(tt_i, k, log=True, npoints=npoints) 
            #dlnPdttj = dPmdtheta(tt_j, k, log=True, npoints=npoints) 
            #Fij[i,j] = np.trapz(k**3 * dlnPdtti * dlnPdttj, np.log(k))
            dPdtti = dPmdtheta(tt_i, k, log=False, npoints=npoints) 
            dPdttj = dPmdtheta(tt_j, k, log=False, npoints=npoints) 
            Fij[i,j] = np.trapz(k**2 * dPdtti * dPdttj / Pm_fid**2, k)
    return Fij * factor 


def dPmdtheta(theta, k, log=False, npoints=5): 
    ''' derivative of Pm w.r.t. to theta 
    '''
    if theta == 'Mnu': 
        return dPmdMnu(k, npoints=npoints, log=log) 
    else:
        h_dict = {'Ob': 0.002, 'Om': 0.02, 'h': 0.04, 'ns': 0.04, 's8': 0.03} 
        h = h_dict[theta]
        Pm_m = _Pm_theta('%s_m' % theta, k) 
        Pm_p = _Pm_theta('%s_p' % theta, k) 
        if not log: 
            return (Pm_p - Pm_m)/h 
        else: # dlnP/dtheta
            return (np.log(Pm_p) - np.log(Pm_m))/h


def _Pm_theta(theta, karr): 
    ''' linear theory matter power spectrum with fiducial parameters except theta for k 
    '''
    if theta not in ['fiducial', 'Ob_m', 'Ob_p', 'Om_m', 'Om_p', 'h_m', 'h_p', 'ns_m', 'ns_p', 's8_m', 's8_p']: 
        raise ValueError 
    f = os.path.join(UT.dat_dir(), 'lt', 'output', '%s_pk.dat' % theta)
    k, pmk = np.loadtxt(f, unpack=True, skiprows=4, usecols=[0,1]) 
    return np.interp(karr, k, pmk) 


def dPmdMnu(k, npoints=5, log=False): 
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
    elif npoints == '0.1eV':
        mnus = [0.075, 0.125]
        coeffs = [-1., 1.]
        fdenom = 0.05 
    elif npoints == 'paco': 
        # read in paco's calculations 
        k_paco, dPm_paco = np.loadtxt(os.path.join(UT.dat_dir(), 'paco', 'der_Pkmm_dMnu.txt'), unpack=True, usecols=[0,5]) 
        return np.interp(k, k_paco, dPm_paco) 
    elif npoints == 'ema': 
        k_ema, dPm_ema = _dPmdMnu_ema(npoints=5)
        return np.interp(k, k_ema, dPm_ema) 
    else: 
        raise ValueError
    
    dPm = np.zeros(len(k))
    for i, mnu, coeff in zip(range(len(mnus)), mnus, coeffs): 
        if not log: 
            pm = _Pm_Mnu(mnu, k) 
        else: 
            pm = np.log(_Pm_Mnu(mnu, k))
        dPm += coeff * pm 
    dPm /= fdenom 
    return dPm 


def _Pm_Mnu(mnu, karr): 
    ''' linear theory matter P(k) with fiducial parameters and input Mnu
    calculated using the high precision CLASS setting 
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
    return np.interp(karr, k, pmk) 


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
    karr = np.logspace(-6, 1, 1000) 
    return karr, fAs * np.interp(karr, k, pmk) 


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
    #compare_Pm()
    #compare_PmMnu_PmLCDM()
    #compare_dPmdMnu()
    #compare_dPmdMnu_0p1eV()
    #compare_dPdthetas() 
    #for npoints in [5, 'quijote', '0.1eV', 'paco', 'ema']: 
    #    Fij =  _Fij_Pm(np.logspace(-5, 2, 500), kmax=0.5, npoints=npoints) 
    #    print npoints, np.sqrt(np.diag(np.linalg.inv(Fij)))
    #Fij_ema = np.array([[5533.04, 73304.8], [73304.8, 1.05542e6]])
    #print np.sqrt(np.diag(np.linalg.inv(Fij_ema)))[::-1]
    LT_sigma_kmax()
