'''

code for linear theory predictions

'''
import os 
import numpy as np 
from scipy.interpolate import interp1d
# --- eMaNu --- 
from emanu import util as UT
from emanu import lineartheory as LT
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
    k_arr = np.logspace(-4, 1, 500)

    fig = plt.figure(figsize=(8,8))
    sub = fig.add_subplot(111)
    # ema's derivative
    for i_n, n in enumerate([1, 2, 3, 4, 5]): 
        dPm_ema = LT.dPmdMnu(k_arr, npoints=n, flag='ema')
        sub.plot(k_arr, dPm_ema, lw=0.75, ls=':', label="Ema's %i pts" % n) 
    # class derivative
    for i_n, n in enumerate([1, 2, 3, 4, 5, 'quijote']): 
        dPm_class = LT.dPmdMnu(k_arr, npoints=n)
        sub.plot(k_arr, dPm_class, c='C%i' % i_n, lw=1, ls='-', label="CLASS %s pts" % str(n))
    # read in paco's calculations 
    k_paco, dPm_paco = np.loadtxt(os.path.join(UT.dat_dir(), 'paco', 'der_Pkmm_dMnu.txt'), unpack=True, usecols=[0,5]) 
    sub.plot(k_paco, dPm_paco, c='r', lw=0.75, ls='--', label="Paco's") 
    sub.legend(loc='lower left', fontsize=15) 
    sub.set_xlabel('$k$', fontsize=20) 
    sub.set_xscale('log') 
    sub.set_xlim(1e-3, 10.) 
    sub.set_ylabel(r'$dP_m/d M_\nu$', fontsize=20) 
    sub.set_yscale('symlog') 
    sub.set_ylim(-1e4, 2e4) 
    fig.savefig(os.path.join(UT.fig_dir(), 'dPmdMnu.class.png'), bbox_inches='tight') 
    return None 


def compare_dPmdMnu_0p1eV(): 
    ''' compare the d P_m(k) / d Mnu at 0.1eV among the various calculations.
    finite differences
    '''
    k_arr = np.logspace(-4, 1, 500)

    fig = plt.figure(figsize=(8,8))
    sub = fig.add_subplot(111)
    # ema's derivatve
    Pm_ema_m = LT._Pm_Mnu(0.075, k_arr, flag='ema') 
    Pm_ema_p = LT._Pm_Mnu(0.125, k_arr, flag='ema') 
    dPm_ema = (Pm_ema_p - Pm_ema_m)/0.05
    sub.plot(k_arr, dPm_ema, lw=1, ls=':', label="Ema's") 
    # paco's derivative
    Pm_paco_m = LT._Pm_Mnu(0.075, k_arr, flag='paco') 
    Pm_paco_p = LT._Pm_Mnu(0.125, k_arr, flag='paco') 
    dPm_paco = (Pm_paco_p - Pm_paco_m)/0.05
    sub.plot(k_arr, dPm_paco, lw=1, ls='--', label="Paco's") 
    # class derivative
    Pm_class_m = LT._Pm_Mnu(0.075, k_arr) 
    Pm_class_p = LT._Pm_Mnu(0.125, k_arr) 
    dPm_class = (Pm_class_p - Pm_class_m)/0.05
    sub.plot(k_arr, dPm_class, lw=1, ls='-', label="CLASS (0.075, 0.125)") 
    # class derivative
    Pm_class_m = LT._Pm_Mnu(0.0, k_arr) 
    Pm_class_p = LT._Pm_Mnu(0.2, k_arr) 
    dPm_class = (Pm_class_p - Pm_class_m)/0.2
    sub.plot(k_arr, dPm_class, lw=1, ls='-', label="CLASS (0.0, 0.2)") 
    sub.legend(loc='lower left', fontsize=15) 
    sub.set_xlabel('$k$', fontsize=20) 
    sub.set_xscale('log') 
    sub.set_xlim(1e-3, 10.) 
    sub.set_ylabel(r'$dP_m/d M_\nu$ at 0.1eV', fontsize=20) 
    sub.set_yscale('symlog') 
    sub.set_ylim(-1e4, 2e4) 
    fig.savefig(os.path.join(UT.fig_dir(), 'dPmdMnu_0.1eV.class.png'), bbox_inches='tight') 
    return None 


def compare_Pm(): 
    ''' compare linear theory P_m(k) 
    '''
    mnus = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125]
    k_arr = np.logspace(-4, 1, 500)

    fig = plt.figure(figsize=(8,8))
    sub = fig.add_subplot(111)
    for i_nu, mnu in enumerate(mnus): 
        Pm_ema  = LT._Pm_Mnu(mnu, k_arr, flag='ema') 
        Pm_paco = LT._Pm_Mnu(mnu, k_arr, flag='paco') 
        Pm_class= LT._Pm_Mnu(mnu, k_arr) 
        sub.plot(k_arr, k_arr * Pm_class * 0.6**i_nu, 
                lw=0.5, label=["CLASS hi prec.", None][bool(i_nu)]) 
        sub.plot(k_arr, k_arr * Pm_ema * 0.6**i_nu, 
                lw=1, c='k', ls=':', label=["Ema's", None][bool(i_nu)]) 
        sub.plot(k_arr, k_arr * Pm_paco * 0.6**i_nu, 
                lw=0.5, c='k', ls='--', label=["Paco's", None][bool(i_nu)]) 
    sub.legend(loc='upper right', fontsize=15) 
    sub.set_xlabel('$k$', fontsize=20) 
    sub.set_xscale('log') 
    sub.set_xlim(1e-2, 1.) 
    sub.set_ylabel(r'$k P_m(k)$', fontsize=20) 
    sub.set_yscale('log') 
    sub.set_ylim(1e1, 1e3) 
    fig.savefig(os.path.join(UT.fig_dir(), 'Pm_Mnu.class.png'), bbox_inches='tight') 

    fig = plt.figure(figsize=(8,12))
    sub = fig.add_subplot(311)
    for i_nu, mnu in enumerate(mnus): 
        Pm_ema  = LT._Pm_Mnu(mnu, k_arr, flag='ema') 
        Pm_class= LT._Pm_Mnu(mnu, k_arr) 
        sub.plot(k_arr, Pm_ema / Pm_class, lw=1, c='C%i' % i_nu, label="%.3feV" % mnu) 
    sub.plot(k_arr, np.ones(len(k_arr)), lw=1, c='k')
    sub.legend(loc='upper right', ncol=2, fontsize=15) 
    sub.set_xscale('log') 
    sub.set_xlim(1e-3, 10.) 
    sub.set_ylim(0.98, 1.02) 
    sub.set_ylabel(r'$P^{\rm ema}_m(k)/P^{\rm CLASS}_m$ ratio', fontsize=20) 

    sub = fig.add_subplot(312)
    for i_nu, mnu in enumerate(mnus): 
        Pm_paco = LT._Pm_Mnu(mnu, k_arr, flag='paco') 
        Pm_class= LT._Pm_Mnu(mnu, k_arr) 
        sub.plot(k_arr, Pm_paco / Pm_class, lw=1, c='C%i' % i_nu) 
    sub.plot(k_arr, np.ones(len(k_arr)), lw=1, c='k')
    sub.set_xscale('log') 
    sub.set_xlim(1e-3, 10.) 
    sub.set_ylabel(r'$P^{\rm paco}_m(k)/P^{\rm CLASS}_m$ ratio', fontsize=20) 
    sub.set_ylim(0.98, 1.02) 
    
    sub = fig.add_subplot(313)
    for i_nu, mnu in enumerate(mnus): 
        Pm_paco = LT._Pm_Mnu(mnu, k_arr, flag='cb') 
        Pm_class= LT._Pm_Mnu(mnu, k_arr) 
        sub.plot(k_arr, Pm_paco / Pm_class, lw=1, c='C%i' % i_nu) 
        Pm_paco = LT._Pm_Mnu(mnu, k_arr, flag='ema_cb') 
        Pm_class= LT._Pm_Mnu(mnu, k_arr, flag='ema') 
        sub.plot(k_arr, Pm_paco / Pm_class, lw=1, ls='--', c='C%i' % i_nu, label=r'$P^{\rm ema}_{\rm cb}/P^{\rm ema}_m$') 
        if i_nu == 0: sub.legend(loc='lower right', fontsize=20) 
    sub.plot(k_arr, np.ones(len(k_arr)), lw=1, c='k')
    sub.set_xscale('log') 
    sub.set_xlim(1e-3, 10.) 
    sub.set_ylabel(r'$P_{\rm cb}(k)/P^{\rm CLASS}_m$ ratio', fontsize=20) 
    sub.set_ylim(0.98, 1.02) 
    fig.savefig(os.path.join(UT.fig_dir(), 'Pm_Mnu.class.ratio.png'), bbox_inches='tight') 
    return None 


def compare_Pm_ns(): 
    ''' compare linear theory P_m(k) 
    '''
    k_arr = np.logspace(-4, 1, 500)
    Pm_fid = LT._Pm_Mnu(0.0, k_arr) 
    Pm_fid_As = LT._Pm_theta('fid_As', k_arr) 

    fig = plt.figure(figsize=(8,10))
    sub = fig.add_subplot(211)
    sub.plot(k_arr, Pm_fid, c='k', lw=1, label='$n_s=0.9624$') 
    Pm = LT._Pm_theta('ns_m', k_arr) 
    sub.plot(k_arr, Pm, lw=1, label='$n_s=0.9424$') 
    Pm = LT._Pm_theta('ns_p', k_arr) 
    sub.plot(k_arr, Pm, lw=1, label='$n_s=0.9824$') 
    Pm = LT._Pm_theta('ns_m_As', k_arr) 
    sub.plot(k_arr, Pm, lw=1, label='$n_s=0.9424$; fixed $A_s$') 
    Pm = LT._Pm_theta('ns_p_As', k_arr) 
    sub.plot(k_arr, Pm, lw=1, label='$n_s=0.9824$; fixed $A_s$') 
    sub.legend(loc='upper right', fontsize=15) 
    sub.set_xlabel('$k$', fontsize=20) 
    sub.set_xscale('log') 
    sub.set_xlim(1e-3, 1.) 
    sub.set_ylabel(r'$P_m(k)$', fontsize=20) 
    sub.set_yscale('log') 
    sub.set_ylim(5e1, 1e5) 

    sub = fig.add_subplot(212)
    Pm = LT._Pm_theta('ns_m', k_arr) 
    sub.plot(k_arr, Pm/Pm_fid, lw=1, label='$n_s=0.9424$') 
    sub.plot(k_arr, k_arr**-0.02, ls=':', c='k') 

    Pm = LT._Pm_theta('ns_p', k_arr) 
    sub.plot(k_arr, Pm/Pm_fid, lw=1, label='$n_s=0.9824$') 
    sub.plot(k_arr, k_arr**0.02, ls=':', c='k') 
    
    Pm = LT._Pm_theta('ns_m_As', k_arr) 
    print (Pm/k_arr**0.9424)[:10]
    sub.plot(k_arr, Pm/Pm_fid_As, lw=1, label='$n_s=0.9424$; fixed $A_s$') 
    Pm = LT._Pm_theta('ns_p_As', k_arr) 
    print (Pm/k_arr**0.9824)[:10]
    sub.plot(k_arr, Pm/Pm_fid_As, lw=1, label='$n_s=0.9824$; fixed $A_s$') 
    sub.legend(loc='upper right', fontsize=15) 
    sub.set_xlabel('$k$', fontsize=20) 
    sub.set_xscale('log') 
    sub.set_xlim(1e-3, 1.) 
    sub.set_ylabel(r'$P_m/P^{\rm fid}_m$', fontsize=20) 
    sub.set_ylim(0.8, 1.2) 
    fig.savefig(os.path.join(UT.fig_dir(), 'Pm_ns.class.png'), bbox_inches='tight') 
    return None 


def compare_dPmdns(): 
    ''' compare linear theory P_m(k) 
    '''
    k_arr = np.logspace(-4, 1, 500)
    Pm_fid = LT._Pm_Mnu(0.0, k_arr) 

    fig = plt.figure(figsize=(8,5))
    sub = fig.add_subplot(111)
    Pm_m = LT._Pm_theta('ns_m', k_arr) 
    Pm_p = LT._Pm_theta('ns_p', k_arr) 
    sub.plot(k_arr, (Pm_p - Pm_m)/0.04/Pm_fid, label='numerical') 
    dpm = LT.dPmdtheta('ns', k_arr, log=False) 
    sub.plot(k_arr, dpm/Pm_fid, ls=':', label='no log') 
    dpm = LT.dPmdtheta('ns', k_arr, log=True) 
    sub.plot(k_arr, dpm, ls='-.', label='log') 

    Pm_fid_As = LT._Pm_theta('fid_As', k_arr) 
    Pm_m_As = LT._Pm_theta('ns_m_As', k_arr) 
    Pm_p_As = LT._Pm_theta('ns_p_As', k_arr) 
    sub.plot(k_arr, (Pm_p_As - Pm_m_As)/0.04/Pm_fid_As, label='fixed $A_s$') 

    sub.plot(k_arr, np.log(k_arr), ls='--', label='$\log(k)$')
    sub.legend(loc='upper left', fontsize=15) 
    sub.set_xlabel('$k$', fontsize=20) 
    sub.set_xscale('log') 
    sub.set_xlim(1e-4, 1.) 
    sub.set_ylabel(r'$dP_m(k)/dn_s/P_m$', fontsize=20) 
    #sub.set_yscale('log') 
    #sub.set_ylim(5e1, 1e5) 
    fig.savefig(os.path.join(UT.fig_dir(), 'dPmdns.class.png'), bbox_inches='tight') 
    return None 


def compare_dlogPmcbdMnu(): 
    ''' compare linear theory d P(k) / d theta 
    '''
    k_arr = np.logspace(-4, 1, 500)

    dPm = LT.dPmdMnu(k_arr, log=True, npoints='0.1eV') 
    dPcb = LT.dPmdMnu(k_arr, log=True, npoints='0.1eV', flag='cb') 
    
    dPm_ema = LT.dPmdMnu(k_arr, log=True, npoints='0.1eV', flag='ema') 
    dPcb_ema = LT.dPmdMnu(k_arr, log=True, npoints='0.1eV', flag='ema_cb') 

    fig = plt.figure(figsize=(8,4))
    sub = fig.add_subplot(121)
    sub.plot([1e-4, 1.], [0, 0], c='k', ls='--') 
    sub.plot(k_arr, dPm, c='C0', label='m') 
    sub.plot(k_arr, dPcb, c='C1', label='cb') 
    sub.plot(k_arr, dPm_ema, c='C0', ls=':', label='ema m') 
    sub.plot(k_arr, dPcb_ema, c='C1', ls=':', label='ema cb') 
    sub.legend(loc='best', fontsize=15) 
    sub.set_xlabel('$k$', fontsize=20) 
    sub.set_xscale('log') 
    sub.set_xlim(1e-4, 1.) 
    sub.set_ylabel(r'$d\log P/d M_\nu$ at 0.1eV', fontsize=20) 
    sub = fig.add_subplot(122)
    sub.plot([1e-4, 1.], [0, 0], c='k', ls='--') 
    sub.plot(k_arr, k_arr**2*dPm**2, c='C0', label='m') 
    sub.plot(k_arr, k_arr**2*dPcb**2, c='C1', label='cb') 
    sub.plot(k_arr, k_arr**2*dPm_ema**2, c='C0', ls=':', label='ema m') 
    sub.plot(k_arr, k_arr**2*dPcb_ema**2, c='C1', ls=':', label='ema cb') 
    sub.set_xlabel('$k$', fontsize=20) 
    sub.set_xscale('log') 
    sub.set_xlim(1e-4, 1.) 
    sub.set_ylabel(r'$k^2(d\log P/d M_\nu)^2$ at 0.1eV', fontsize=20) 
    sub.set_yscale('log') 
    fig.subplots_adjust(wspace=0.4) 
    fig.savefig(os.path.join(UT.fig_dir(), 'dlogPmcbdMnu.class.png'), bbox_inches='tight') 
    return None 


def compare_PmMnu_PmLCDM(): 
    ''' compare (P_m(k) Mnu > 0)/(P_m(k) LCDM) 
    '''
    mnus = [0.025, 0.05, 0.075, 0.1, 0.125] 
    k_arr = np.logspace(-5, 1, 700)

    #Pm_paco_fid = LT._Pm_Mnu(0., k_arr, flag='paco') 
    Pm_ema_fid = LT._Pm_Mnu(0., k_arr, flag='ema') 
    Pm_fid = LT._Pm_Mnu(0., k_arr) 

    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(121)
    for i_nu, mnu in enumerate(mnus): 
        Pm_ema      = LT._Pm_Mnu(mnu, k_arr, flag='ema') 
        #Pm_paco     = LT._Pm_Mnu(mnu, k_arr, flag='paco') 
        Pm_class    = LT._Pm_Mnu(mnu, k_arr) 

        f_ema   = np.median((Pm_ema / Pm_ema_fid)[:10])
        f_class = np.median((Pm_class / Pm_fid)[:10])
        sub.plot(k_arr, Pm_class / Pm_fid / f_class, 
                lw=0.5, c='C%i' % i_nu, label=["CLASS", None][bool(i_nu)]) 
        #sub.plot(k_arr, Pm_paco / Pm_paco_fid, 
        #        lw=1, c='C%i' % i_nu, ls='--', label=["Paco's", None][bool(i_nu)]) 
        sub.plot(k_arr, Pm_ema / Pm_ema_fid /f_ema, 
                lw=1, c='C%i' % i_nu, ls=':', label=["Ema's", None][bool(i_nu)]) 
        #if mnu == 0.125: 
        #    print mnu / 93.14 / 0.6711**2 / 0.3175 * 8.
        #    print (np.mean((Pm_class / Pm_fid)[:10]) - np.mean((Pm_class / Pm_fid)[-10:]))
        #    print (np.mean((Pm_ema / Pm_ema_fid)[:5]) - np.mean((Pm_ema / Pm_ema_fid)[-5:]))
        
        #sub.plot([1e-6, 1e1], [np.median((Pm_class / Pm_fid)[:10]), np.median((Pm_class / Pm_fid)[:10])], c='k', ls=':') 
        fnu = mnu / (93.14 * 0.6711**2) / 0.3175
        #sub.plot([1e-6, 1e1], [1. - 8.*fnu, 1. - 8.*fnu], c='k', ls=':', lw=0.5) 
        sub.plot([1e-6, 1e1], [0.915, 0.915]) 
        sub.plot([1e-6, 1e1], [0.915, 0.915]) 
        sub.plot([1e-6, 1e1], [0.915, 0.915]) 
        sub.plot([1e-6, 1e1], [0.915, 0.915]) 

    sub.legend(loc='upper right', fontsize=15) 
    sub.set_xlabel('$k$', fontsize=20) 
    sub.set_xscale('log') 
    sub.set_xlim(1e-5, 10.) 
    sub.set_ylim(0.9, 1.02) 
    
    Pm_ema_fid = LT._Pm_Mnu(0., k_arr, flag='ema_cb') 
    Pm_fid = LT._Pm_Mnu(0., k_arr, flag='cb') 
    sub = fig.add_subplot(122)
    for i_nu, mnu in enumerate(mnus): 
        Pm_ema      = LT._Pm_Mnu(mnu, k_arr, flag='ema_cb') 
        Pm_class    = LT._Pm_Mnu(mnu, k_arr, flag='cb') 
        
        f_ema   = np.median((Pm_ema / Pm_ema_fid)[:10])
        f_class = np.median((Pm_class / Pm_fid)[:10])

        sub.plot(k_arr, Pm_class / Pm_fid / f_class, lw=0.5, c='C%i' % i_nu) 
        sub.plot(k_arr, Pm_ema / Pm_ema_fid / f_ema, lw=1, c='C%i' % i_nu, ls=':') 

        #sub.plot([1e-6, 1e1], [np.median((Pm_class / Pm_fid)[:10]), np.median((Pm_class / Pm_fid)[:10])], c='k', ls=':') 
        fnu = mnu / 93.14 / 0.6711**2 / 0.3175
        sub.plot([1e-6, 1e1], [1. - 6.*fnu, 1. - 6.*fnu], c='k', ls=':', lw=0.5) 
    sub.set_xlabel('$k$', fontsize=20) 
    sub.set_xscale('log') 
    sub.set_xlim(1e-5, 10.) 
    sub.set_ylim(0.9, 1.02) 
    fig.savefig(os.path.join(UT.fig_dir(), 'Pm_Mnu_Pm_LCDM.class.png'), bbox_inches='tight') 
    return None 


def compare_dPdthetas():
    ''' Plot the derivatives of Pm w.r.t. to the different parameters 
    '''
    thetas = ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu'] 
    theta_lbls = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$']

    k = np.logspace(-4., 1, 500) 
    dpdts = []  
    for tt in thetas:  
        dpm = LT.dPmdtheta(tt, k)
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


def LT_sigma_kmax(npoints=5, flag=None):
    ''' linear theory Pm forecasts as a function of kmax 
    '''
    thetas = ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
    theta_lbls = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$']

    kmaxs = np.pi/500. * 3 * np.array([1, 5, 10, 15, 20, 28])# np.arange(1, 28) 
    sigma_thetas = []
    sigma_ii = [] 
    for i_k, kmax in enumerate(kmaxs): 
        Fij =  LT.Fij_Pm(np.logspace(-5, 2, 500), kmax=kmax, npoints=npoints, flag=flag) 
        sigma_thetas.append(np.sqrt(np.diag(np.linalg.inv(Fij))))
        sigma_ii.append(1./np.sqrt(np.diag(Fij)))
        #print sigma_thetas[-1]
    sigma_thetas = np.array(sigma_thetas)
    sigma_ii = np.array(sigma_ii)
    sigma_theta_lims = [(1e-3, 10.), (1e-3, 10.), (1e-3, 50), (1e-3, 50.), (1e-3, 50.), (1e-3, 50.)]

    fig = plt.figure(figsize=(15,8))
    for i, theta in enumerate(thetas): 
        sub = fig.add_subplot(2,len(thetas)/2,i+1) 
        sub.plot(kmaxs, sigma_thetas[:,i], c='k', ls='-', label='$P_m$ forecast') 
        sub.plot(kmaxs, sigma_ii[:,i], c='k', ls=':', label='$P_m$ unmarginalized') 
        sub.set_xlim(0.005, 0.5)
        sub.text(0.9, 0.9, theta_lbls[i], ha='right', va='top', transform=sub.transAxes, fontsize=30)
        sub.set_ylim(sigma_theta_lims[i]) 
        sub.set_yscale('log') 
        if i == 0: sub.legend(loc='lower right', fontsize=15) 
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'$k_{\rm max}$', fontsize=28) 
    bkgd.set_ylabel(r'$1\sigma$ constraint on $\theta$', labelpad=10, fontsize=28) 

    fig.subplots_adjust(wspace=0.2, hspace=0.15) 
    if flag is not None: 
        ffig = os.path.join(UT.fig_dir(), 'LT_sigma_kmax.npoints%s.%s.png' % (str(npoints), flag))
    else: 
        ffig = os.path.join(UT.fig_dir(), 'LT_sigma_kmax.npoints%s.png' % (str(npoints)))
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def LT_s8Mnu_kmax(npoints=5):
    ''' 
    '''
    thetas = ['s8', 'Mnu']
    theta_lbls = [r'$\sigma_8$', r'$M_\nu$']

    kmaxs = [0.1, 0.2, 0.3, 0.5] 
    sigma_thetas = []
    for i_k, kmax in enumerate(kmaxs): 
        Fij =  LT.Fij_Pm(np.logspace(-5, 2, 500), kmax=kmax, npoints=npoints, thetas=thetas) 
        sigma_thetas.append(np.sqrt(np.diag(np.linalg.inv(Fij))))
    sigma_thetas = np.array(sigma_thetas)
    sigma_theta_lims = [(1e-3, 50.), (1e-2, 50.)]

    fig = plt.figure(figsize=(10,5))
    for i, theta in enumerate(thetas): 
        sub = fig.add_subplot(1,2,i+1) 
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
    ffig = os.path.join(UT.doc_dir(), 'figs', 'LT_s8Mmin_kmax.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def LT_sigma_kmax_cb_m():
    ''' compare linear theory forecasts for Pm and Pcb
    '''
    thetas = ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
    theta_lbls = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$']

    kmaxs = np.pi/500. * 3 * np.array([1, 5, 10, 15, 20, 28])# np.arange(1, 28) 
    sigma_m, sigma_cb = [], [] 
    sigma_ii_m, sigma_ii_cb = [], [] 
    for i_k, kmax in enumerate(kmaxs): 
        Fij =  LT.Fij_Pm(np.logspace(-5, 2, 500), kmax=kmax, npoints=5) 
        sigma_m.append(np.sqrt(np.diag(np.linalg.inv(Fij))))
        Fij =  LT.Fij_Pm(np.logspace(-5, 2, 500), kmax=kmax, npoints=5, flag='cb') 
        sigma_cb.append(np.sqrt(np.diag(np.linalg.inv(Fij))))

        Fij =  LT.Fij_Pm(np.logspace(-5, 2, 500), kmax=kmax, npoints=5) 
        sigma_ii_m.append(1./np.sqrt(np.diag(Fij)))
        Fij =  LT.Fij_Pm(np.logspace(-5, 2, 500), kmax=kmax, npoints=5, flag='cb') 
        sigma_ii_cb.append(1./np.sqrt(np.diag(Fij)))
    sigma_m = np.array(sigma_m)
    sigma_cb = np.array(sigma_cb)
    sigma_ii_m = np.array(sigma_ii_m)
    sigma_ii_cb = np.array(sigma_ii_cb)
    sigma_theta_lims = [(1e-3, 10.), (1e-3, 10.), (1e-3, 50), (1e-3, 50.), (1e-3, 50.), (1e-3, 50.)]

    fig = plt.figure(figsize=(15,8))
    for i, theta in enumerate(thetas): 
        sub = fig.add_subplot(2,len(thetas)/2,i+1) 
        sub.plot(kmaxs, sigma_m[:,i], c='k', ls='-', label='$P_m$') 
        sub.plot(kmaxs, sigma_cb[:,i], c='k', ls=':', label='$P_{cb}$') 

        sub.plot(kmaxs, sigma_ii_m[:,i], c='C1', ls='-', label='$P_m$ unmargin.') 
        sub.plot(kmaxs, sigma_ii_cb[:,i], c='C1', ls=':', label='$P_{cb}$ unargmin.') 
        sub.set_xlim(0.005, 0.5)
        sub.text(0.9, 0.9, theta_lbls[i], ha='right', va='top', transform=sub.transAxes, fontsize=30)
        sub.set_ylim(sigma_theta_lims[i]) 
        sub.set_yscale('log') 
        if i == 0: sub.legend(loc='lower right', frameon=True, fontsize=15) 
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'$k_{\rm max}$', fontsize=28) 
    bkgd.set_ylabel(r'$1\sigma$ constraint on $\theta$', labelpad=10, fontsize=28) 

    fig.subplots_adjust(wspace=0.2, hspace=0.15) 
    ffig = os.path.join(UT.fig_dir(), 'LT_sigma_kmax_cb_m.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None


if __name__=="__main__": 
    #compare_Pm()
    #compare_Pm_ns()
    #compare_dPmdns() 
    #compare_dlogPmcbdMnu()
    #compare_PmMnu_PmLCDM()
    #compare_dPmdMnu()
    #compare_dPmdMnu_0p1eV()
    #compare_dPdthetas() 
    #for npoints in ['0.1eV', 'paco', 'ema0.1eV']: #[5, '0.1eV', 'ema0.1eV']:#, 'quijote', '0.1eV', 'paco', 'ema']: 
    #    Fij =  _Fij_Pm(np.logspace(-2, 2, 100), kmax=0.5, npoints=npoints, thetas=['s8', 'Mnu']) 
    #    print '----', npoints, '----'  
    #    print Fij
    #    print np.sqrt(np.diag(np.linalg.inv(Fij)))
    #Fij_ema = np.array([[5533.04, 73304.8], [73304.8, 1.05542e6]])
    #print np.sqrt(np.diag(np.linalg.inv(Fij_ema)))[::-1]
    #LT_sigma_kmax()
    LT_sigma_kmax_cb_m()
    #LT_sigma_kmax(npoints='0.1eV', flag='cb')
    #LT_sigma_kmax(npoints='0.1eV', flag='ema')
    #LT_sigma_kmax(npoints='0.1eV', flag='ema_cb')
    #LT_s8Mnu_kmax(npoints='0.1eV')
