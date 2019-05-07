'''

Quijote forecasts with fixed nbar. This is a particularly valuable test 
because it tries to remove the effect of bias on the derivatives -- i.e. 
the halo distribution at theta+ and theta- do *not* have the same bias. 

'''
import os 
import h5py
import numpy as np 
# --- eMaNu --- 
from emanu import util as UT
from emanu import obvs as Obvs
from emanu import plots as ePlots
from emanu import forecast as Forecast
from emanu import lineartheory as LT 
# --- plotting --- 
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.ticker import StrMethodFormatter, NullFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
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


quijote_thetas = {
        'Mnu': [0.1, 0.2, 0.4], # +, ++, +++ 
        'Ob': [0.048, 0.050], # others are - + 
        'Om': [0.3075, 0.3275],
        'h': [0.6511, 0.6911],
        'ns': [0.9424, 0.9824],
        's8': [0.819, 0.849] }
##################################################################
# qujiote fisher 
##################################################################
kf = 2.*np.pi/1000. # fundmaentla mode
thetas = ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
theta_lbls = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$']
theta_lims = [(0.25, 0.385), (0.02, 0.08), (0.3, 1.1), (0.6, 1.35), (0.8, 0.87), (-0.45, 0.45)]
theta_fid = {'Mnu': 0., 'Ob': 0.049, 'Om': 0.3175, 'h': 0.6711,  'ns': 0.9624,  's8': 0.834} # fiducial theta 
ntheta = len(thetas)


def Pk_fixednbar(): 
    ''' Comparison of the P(theta_fid), P(theta+), P(theta-) for LT Pm and halo P
    '''
    klin = np.logspace(-5, 1, 400)
    LT_fid = LT._Pm_Mnu(0., klin) 

    fig = plt.figure(figsize=(15,8))
    for _i, theta in enumerate(thetas): 
        quij_fid = Obvs.quijoteBk('fiducial', flag='.fixed_nbar') # theta_fiducial 
        pk_fid = np.average(quij_fid['p0k1'], axis=0) 
        quij_p = Obvs.quijoteBk(theta+'_p', flag='.fixed_nbar') 
        pk_p = np.average(quij_p['p0k1'], axis=0) 
        if theta != 'Mnu': 
            quij_m = Obvs.quijoteBk(theta+'_m', flag='.fixed_nbar') 
            pk_m = np.average(quij_m['p0k1'], axis=0) 
        else: 
            quij_pp = Obvs.quijoteBk(theta+'_pp', flag='.fixed_nbar') 
            pk_pp = np.average(quij_pp['p0k1'], axis=0) 
            quij_ppp = Obvs.quijoteBk(theta+'_ppp', flag='.fixed_nbar') 
            pk_ppp = np.average(quij_ppp['p0k1'], axis=0) 

        if theta != 'Mnu': 
            LT_p = LT._Pm_theta(theta+'_p', klin)
            LT_m = LT._Pm_theta(theta+'_m', klin)
        else: 
            LT_p = LT._Pm_Mnu(0.1, klin)
            LT_pp = LT._Pm_Mnu(0.2, klin)
            LT_ppp = LT._Pm_Mnu(0.4, klin)

            LT_cb_p = LT._Pm_Mnu(0.1, klin, flag='cb')
            LT_cb_pp = LT._Pm_Mnu(0.2, klin, flag='cb')
            LT_cb_ppp = LT._Pm_Mnu(0.4, klin, flag='cb')
    
        i_k = quij_fid['k1'] 
        _, _iuniq = np.unique(i_k, return_index=True)
        iuniq = np.zeros(len(i_k)).astype(bool) 
        iuniq[_iuniq] = True
        
        i_tt = thetas.index(theta)

        sub = fig.add_subplot(2,3,i_tt+1) 
        sub.plot(klin, np.ones(len(klin)), c='k', ls='--')
        if theta == 's8': 
            sub.plot(kf*i_k[iuniq], pk_p[iuniq]/pk_fid[iuniq], c='C0')
            sub.plot(kf*i_k[iuniq], pk_m[iuniq]/pk_fid[iuniq], c='C1')
            sub.plot(klin, LT_p/LT_fid, c='k', ls='-.')
            sub.plot(klin, LT_m/LT_fid, c='k', ls=':')
        elif theta == 'Mnu': 
            sub.plot(kf*i_k[iuniq], pk_p[iuniq]/pk_fid[iuniq], 
                    c='C0', label=r'$%s^+$' % theta_lbls[i_tt].strip('$'))
            sub.plot(kf*i_k[iuniq], pk_pp[iuniq]/pk_fid[iuniq], 
                    c='C1', label=r'$%s^{++}$' % theta_lbls[i_tt].strip('$'))
            sub.plot(kf*i_k[iuniq], pk_ppp[iuniq]/pk_fid[iuniq], 
                    c='C2', label=r'$%s^{+++}$' % theta_lbls[i_tt].strip('$'))

            sub.plot(klin, LT_p/LT_fid, c='C0',   lw=0.5, ls='--', label=r'LT$^+$')
            sub.plot(klin, LT_pp/LT_fid, c='C1',  lw=0.5, ls='-.', label=r'LT$^{++}$')
            sub.plot(klin, LT_ppp/LT_fid, c='C2', lw=0.5, ls=':', label=r'LT$^{+++}$')
            sub.plot(klin, LT_cb_p/LT_fid, c='k', ls='--', label=r'LT$_{cb}^+$')
            sub.plot(klin, LT_cb_pp/LT_fid, c='k', ls='-.', label=r'LT$_{cb}^{++}$')
            sub.plot(klin, LT_cb_ppp/LT_fid, c='k', ls=':', label=r'LT$_{cb}^{+++}$')
            sub.legend(loc='best', ncol=2, fontsize=10) 
            sub.set_ylim(0.95, 1.3)  
        else:  
            sub.plot(kf*i_k[iuniq], pk_p[iuniq]/pk_fid[iuniq], c='C0', label=r'$\theta^+$')
            sub.plot(kf*i_k[iuniq], pk_m[iuniq]/pk_fid[iuniq], c='C1', label=r'$\theta^-$')

            sub.plot(klin, LT_p/LT_fid, c='k', ls='-.', label=r'LT$^+$')
            sub.plot(klin, LT_m/LT_fid, c='k', ls=':', label=r'LT$^-$')

        if theta == 's8': 
            f_p =(pk_fid[0] + pk_fid[1]) / (pk_p[0] + pk_p[1]) * 0.849**2/0.834**2
            f_m =(pk_fid[0] + pk_fid[1]) / (pk_m[0] + pk_m[1]) * 0.819**2/0.834**2
            sub.plot(kf*i_k[iuniq], (f_p * pk_p[iuniq])/pk_fid[iuniq], c='C0', ls=':', label='rescaled +')
            sub.plot(kf*i_k[iuniq], (f_m * pk_m[iuniq])/pk_fid[iuniq], c='C1', ls=':', label='rescaled -')
            sub.legend(loc='best', fontsize=15)

        if _i == 0: sub.legend(loc='best', ncol=2, fontsize=15) 
        sub.set_xlim(1e-2, 1) 
        sub.set_xscale("log") 
        sub.text(0.05, 0.05, theta_lbls[_i], ha='left', va='bottom', transform=sub.transAxes, fontsize=25)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel('k [$h$/Mpc]', fontsize=25) 
    bkgd.set_ylabel(r'$P_0/P^{\rm fid}_0$', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ffig = os.path.join(UT.fig_dir(), 'Pk_fixednbar.thetas.ratio.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def quijote_dPk_fixednbar(theta, dmnu='fin', log=False, s8corr=True):
    ''' calculate d P0(k)/d theta using the paired and fixed quijote simulations
    run on perturbed theta 

    :param theta: 
        string that specifies the parameter to take the 
        derivative by. 
    '''
    c_dpk = 0.
    if theta == 'Mnu': 
        tts = ['fiducial', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp']
        if dmnu == 'p': 
            coeffs = [-1., 1., 0., 0.]
            h = 0.1
        elif dmnu == 'pp': 
            coeffs = [-1., 0., 1., 0.]
            h = 0.2
        elif dmnu == 'ppp': 
            coeffs = [-1., 0., 0., 1.]
            h = 0.4
        elif dmnu == 'fin0': 
            coeffs = [-3., 4., -1., 0.] # finite difference coefficient
            h = 0.2
        elif dmnu == 'fin': 
            coeffs = [-21., 32., -12., 1.] # finite difference coefficient
            h = 1.2
    elif theta == 's8': 
        if s8corr: 
            tts = ['fiducial'] 
            coeffs = [0.] 
            h = 1. 
            quij = Obvs.quijoteBk('fiducial', flag='.fixed_nbar') 
            pk_fid = np.average(quij['p0k1'], axis=0) 
            quij = Obvs.quijoteBk('s8_p', flag='.fixed_nbar') 
            pk_p = np.average(quij['p0k1'], axis=0) 
            quij = Obvs.quijoteBk('s8_m', flag='.fixed_nbar') 
            pk_m = np.average(quij['p0k1'], axis=0) 
            h = quijote_thetas['s8'][1] - quijote_thetas['s8'][0]
        
            f_p =(pk_fid[0] + pk_fid[1]) / (pk_p[0] + pk_p[1]) * 0.849**2/0.834**2
            f_m =(pk_fid[0] + pk_fid[1]) / (pk_m[0] + pk_m[1]) * 0.819**2/0.834**2
            if not log: c_dpk = (f_p * pk_p - f_m * pk_m)/h
            else: c_dpk = (np.log(f_p * pk_p) - np.log(f_m * pk_m))/h
        else: 
            tts = [theta+'_m', theta+'_p'] 
            coeffs = [-1., 1.] 
            h = quijote_thetas[theta][1] - quijote_thetas[theta][0]
    elif theta == 'Amp': 
        # amplitude of P(k) is a free parameter
        tts = ['fiducial'] 
        coeffs = [0.] 
        h = 1. 
        quij = Obvs.quijoteBk('fiducial', flag='.fixed_nbar') 
        if not log: c_dpk = np.average(quij['p0k1'], axis=0) 
        else: c_dpk = np.ones(quij['p0k1'].shape[1]) 
    elif theta == 'Asn': 
        # constant shot noise term is a free parameter
        tts = ['fiducial'] 
        coeffs = [0.] 
        h = 1. 
        quij = Obvs.quijoteBk('fiducial') 
        if not log: c_dpk = np.ones(quij['p0k1'].shape[1]) 
        else: c_dpk = 1./np.average(quij['p0k1'], axis=0) 
    else: 
        tts = [theta+'_m', theta+'_p'] 
        coeffs = [-1., 1.] 
        h = quijote_thetas[theta][1] - quijote_thetas[theta][0]

    for i_tt, tt, coeff in zip(range(len(tts)), tts, coeffs): 
        quij = Obvs.quijoteBk(tt, flag='.fixed_nbar') # read P0k 
        if i_tt == 0: dpk = np.zeros(quij['p0k1'].shape[1]) 
        _pk = np.average(quij['p0k1'], axis=0)  

        if log: _pk = np.log(_pk) # log 

        dpk += coeff * _pk 
    return dpk / h + c_dpk


def quijote_dBk_fixednbar(theta, dmnu='fin', log=False):
    ''' calculate d B(k)/d theta using the paired and fixed quijote simulations
    run on perturbed theta 

    :param theta: 
        string that specifies the parameter to take the 
        derivative by. 
    '''
    c_dbk = 0.
    if theta == 'Mnu': 
        tts = ['fiducial', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp']
        if dmnu == 'p': 
            coeffs = [-1., 1., 0., 0.]
            h = 0.1
        elif dmnu == 'pp': 
            coeffs = [-1., 0., 1., 0.]
            h = 0.2
        elif dmnu == 'ppp': 
            coeffs = [-1., 0., 0., 1.]
            h = 0.4
        elif dmnu == 'fin0': 
            coeffs = [-3., 4., -1., 0.] # finite difference coefficient
            h = 0.2
        elif dmnu == 'fin': 
            coeffs = [-21., 32., -12., 1.] # finite difference coefficient
            h = 1.2
    elif theta == 'Amp': 
        # amplitude scaling is a free parameter
        tts = ['fiducial'] 
        coeffs = [0.] 
        h = 1. 
        quij = Obvs.quijoteBk('fiducial', flag='.fixed_nbar')
        if not log: c_dbk = np.average(quij['b123'], axis=0) 
        else: c_dbk = np.ones(quij['b123'].shape[1])
    else: 
        tts = [theta+'_m', theta+'_p'] 
        coeffs = [-1., 1.] 
        h = quijote_thetas[theta][1] - quijote_thetas[theta][0]

    for i_tt, tt, coeff in zip(range(len(tts)), tts, coeffs): 
        quij = Obvs.quijoteBk(tt, flag='.fixed_nbar') # read P0k 
        if i_tt == 0: dbk = np.zeros(quij['b123'].shape[1]) 
        _bk = np.average(quij['b123'], axis=0)  

        if log: _bk = np.log(_bk) # log 

        dbk += coeff * _bk 
    return dbk / h + c_dbk


def quijote_dPdthetas_LT_fixednbar(dmnu='fin'):
    ''' Compare the derivatives of the powerspectrum also with linear theory
    '''
    # fiducial P0(k)  
    quij = Obvs.quijoteP0k('fiducial')
    pk_fid = np.average(quij['p0k'], axis=0) 

    klin = np.logspace(-5, 1, 400)
    pm_fid = LT._Pm_Mnu(0., klin) 
    
    quij_fn = Obvs.quijoteBk('fiducial')
    i_k = quij_fn['k1'] 
    _, _iuniq = np.unique(i_k, return_index=True)
    iuniq = np.zeros(len(i_k)).astype(bool) 
    iuniq[_iuniq] = True
    klim = (iuniq) 
    k_fn = np.pi/500. * i_k[klim]

    ylims = [(-12., 5.), (-6., 15.), (-4., 4.), (-10., 5.), (-3., 3.), (-0.2, 0.7)]

    fig = plt.figure(figsize=(12,8))
    for i_tt, tt, lbl in zip(range(len(thetas)), thetas, theta_lbls): 
        sub = fig.add_subplot(2,3,i_tt+1)
        dpdt = quijote_dPk(tt, dmnu=dmnu, log=True)
        sub.plot(quij['k'], dpdt, lw=1, c='k', label=r'$P_h$') 

        dpdt_fn = quijote_dPk_fixednbar(tt, dmnu=dmnu, log=True, s8corr=False)  # fixed nbar 
        sub.plot(k_fn, dpdt_fn[klim], lw=1, c='k', ls='--', label=r'$P_h$ fixed $\bar{n}$')  

        dpmdt = LT.dPmdtheta(tt, klin, log=True, npoints='quijote') 
        sub.plot(klin, dpmdt, c='C%i' % i_tt, lw=1, ls='-', label='$P_m$')
        if tt == 'Mnu': 
            dpmdt = LT.dPmdtheta(tt, klin, log=True, npoints='quijote', flag='cb') 
            sub.plot(klin, dpmdt, c='C%i' % i_tt, lw=2, ls='--', label='$P_{cb}$')
            sub.legend(loc='lower left', fontsize=15) 
        elif tt == 'h': 
            dpmdt = LT.dPmdtheta(tt, klin, log=True, npoints='quijote') 
            plt_mm, = sub.plot(klin, -dpmdt, c='C%i' % i_tt, lw=1, ls=':', label='$-P_m$')
            sub.legend([plt_mm], ['$-P_m$'], loc='lower left', fontsize=15) 
        elif tt == 's8': 
            dpdt_fn = quijote_dPk_fixednbar(tt, dmnu=dmnu, log=True, s8corr=True) 
            plt_fn, = sub.plot(k_fn, dpdt_fn[klim], lw=1, c='k', ls='-.')
            
            dpmdt = LT.dPmdtheta(tt, klin, log=True, npoints='quijote') 
            plt_mm, = sub.plot(klin, -dpmdt, c='C%i' % i_tt, lw=1, ls=':')
            sub.legend([plt_mm, plt_fn], ['$-P_m$', r'$P_h$ fixed $\bar{n}$ w/ $\sigma_8$ corr.'], 
                    loc='lower left', fontsize=15) 
        sub.set_xscale('log') 
        sub.set_xlim(1.e-5, 10) 
        #sub.set_yscale('symlog', linthreshy=1e-1) 
        #sub.set_ylim(-5e1, 8e1) 
        sub.set_ylim(ylims[i_tt]) 
        sub.text(0.95, 0.95, lbl, ha='right', va='top', transform=sub.transAxes, fontsize=25)
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel('$k$', fontsize=25) 
    bkgd.set_ylabel(r'${\rm d}\log P/{\rm d} \theta$', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ffig = os.path.join(UT.fig_dir(), 'quijote_dlogPdthetas_LT_fixednbar.%s.png' % dmnu)
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def quijote_dBdthetas_fixednbar(dmnu='fin'):
    ''' Compare the derivatives of the powerspectrum also with linear theory
    '''
    kf = 2.*np.pi/1000. # fundmaentla mode
    # fiducialBP(k)  
    quij = Obvs.quijoteBk('fiducial')
    i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3']

    # impose k limit on bispectrum
    klim = ((i_k*kf <= 0.5) & (j_k*kf <= 0.5) & (l_k*kf <= 0.5))
    ijl = UT.ijl_order(i_k[klim], j_k[klim], l_k[klim], typ='GM') # order of triangles 

    bk_ylim = [(-15., -0.5), (1., 40.), (-10., -0.5), (-10., -0.5), (-10., -5e-2), (1e-2, 4)]

    fig = plt.figure(figsize=(12,24))
    for i_tt, tt, lbl in zip(range(len(thetas)), thetas, theta_lbls): 
        sub = fig.add_subplot(6,1,i_tt+1)
        dbdt = quijote_dBk(tt, dmnu=dmnu, log=True)
        sub.plot(range(np.sum(klim)), dbdt[klim][ijl], lw=1, c='C%i' % i_tt, label=r'$B_h$') 

        dbdt_fn = quijote_dBk_fixednbar(tt, dmnu=dmnu, log=True)  # fixed nbar 
        sub.plot(range(np.sum(klim)), dbdt_fn[klim][ijl], lw=1, c='k', ls='--', 
            label=r'$B_h$ fixed $\bar{n}$')  
        if i_tt == 0: sub.legend(loc='upper left', fontsize=15) 
        sub.set_xlim(0, np.sum(klim)) 
        sub.set_yscale('symlog', linthreshy=1e-3) 
        sub.set_ylim(bk_ylim[i_tt])  
        sub.text(0.95, 0.95, lbl, ha='right', va='top', transform=sub.transAxes, fontsize=25)
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel('triangle configurations', fontsize=25) 
    bkgd.set_ylabel(r'${\rm d}\log B/{\rm d} \theta$', labelpad=15, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ffig = os.path.join(UT.fig_dir(), 'quijote_dlogBdthetas_fixednbar.%s.png' % dmnu)
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def quijote_Fisher_fixednbar(obs, kmax=0.5, dmnu='fin', fhartlap=True, s8corr=True, bSN=False): 
    ''' calculate fisher matrix for parameters ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
    with fixed nbar 
    '''
    quij = Obvs.quijoteBk('fiducial', flag='.fixed_nbar') # theta_fiducial 
    i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3']

    if obs == 'pk': 
        pks = quij['p0k1'] + 1e9/quij['Nhalos'][:,None] # uncorrect shot noise 
        _, _iuniq = np.unique(i_k, return_index=True)
        iuniq = np.zeros(len(i_k)).astype(bool) 
        iuniq[_iuniq] = True
        klim = (iuniq & (i_k * kf < kmax)) 

        nmock = quij['p0k1'].shape[0]
        ndata = np.sum(klim) 

        C_fid = np.cov(pks[:,klim].T) # covariance matrix 

    elif obs == 'bk': 
        # read in full covariance matrix (with shotnoise; this is the correct one) 
        bks = quij['b123'] + quij['b_sn'] # shotnoise uncorrected B(k) 
        
        klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) # k limit 

        nmock = quij['b123'].shape[0]
        ndata = np.sum(klim) 
        C_fid = np.cov(bks[:,klim].T)
    
    if fhartlap: f_hartlap = float(nmock - ndata - 2)/float(nmock - 1) 
    else: f_hartlap = 1. 
    C_inv = f_hartlap * np.linalg.inv(C_fid) # invert the covariance 
    
    dobs_dt = [] 
    if bSN: _thetas = thetas +['Amp', 'Asn']
    else: _thetas = thetas 
    for par in _thetas: # calculate the derivative of Bk along all the thetas 
        if obs == 'pk': 
            dobs_dti = quijote_dPk_fixednbar(par, dmnu=dmnu, s8corr=s8corr) 
        elif obs == 'bk': 
            dobs_dti = quijote_dBk_fixednbar(par, dmnu=dmnu)
        dobs_dt.append(dobs_dti[klim])
    Fij = Forecast.Fij(dobs_dt, C_inv) 
    return Fij 


def quijote_Forecast_fixednbar(obs, kmax=0.5, dmnu='fin'):
    ''' fisher forecast for quijote where we impose fixed nbar 
    
    :param krange: (default: [0.01, 0.5]) 
        tuple specifying the kranges of k1, k2, k3 in the bispectrum
    '''
    # fisher matrix (Fij)
    _Fij = quijote_Fisher_freeMmin(obs, kmax=kmax, dmnu=dmnu)
    Fij = quijote_Fisher_fixednbar(obs, kmax=kmax, dmnu=dmnu) 
    _Finv = np.linalg.inv(_Fij)
    Finv = np.linalg.inv(Fij) # invert fisher matrix 
    i_s8 = thetas.index('s8')
    print('original Fii', np.sqrt(np.diag(_Finv)))
    print('fixed nbar Fii', np.sqrt(np.diag(Finv)))
    print('original sigma_s8 = %f' % np.sqrt(_Finv[i_s8,i_s8]))
    print('fixed nbar sigma_s8 = %f' % np.sqrt(Finv[i_s8,i_s8]))
    i_Mnu = thetas.index('Mnu')
    print('original sigma_Mnu = %f' % np.sqrt(_Finv[i_Mnu,i_Mnu]))
    print('fixed nbar sigma_Mnu = %f' % np.sqrt(Finv[i_Mnu,i_Mnu]))

    fig = plt.figure(figsize=(17, 15))
    for i in xrange(ntheta-1): 
        for j in xrange(i+1, ntheta): 
            # sub inverse fisher matrix 
            Finv_sub = np.array([[Finv[i,i], Finv[i,j]], [Finv[j,i], Finv[j,j]]]) 
            theta_fid_i, theta_fid_j = theta_fid[thetas[i]], theta_fid[thetas[j]]
            sub = fig.add_subplot(ntheta-1, ntheta-1, (ntheta-1) * (j-1) + i + 1) 
            Forecast.plotEllipse(Finv_sub, sub, 
                    theta_fid_ij=[theta_fid_i, theta_fid_j], color='C0')
                
            _Finv_sub = np.array([[_Finv[i,i], _Finv[i,j]], [_Finv[j,i], _Finv[j,j]]]) 
            Forecast.plotEllipse(_Finv_sub, sub, 
                    theta_fid_ij=[theta_fid_i, theta_fid_j], color='C1')
            
            sub.set_xlim(theta_lims[i])
            sub.set_ylim(theta_lims[j])
            if i == 0:   
                sub.set_ylabel(theta_lbls[j], fontsize=30) 
            else: 
                sub.set_yticks([])
                sub.set_yticklabels([])
            
            if j == ntheta-1: 
                sub.set_xlabel(theta_lbls[i], labelpad=10, fontsize=30) 
            else: 
                sub.set_xticks([])
                sub.set_xticklabels([]) 

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.fill_between([],[],[], color='C0', label=r'fiducial') 
    bkgd.fill_between([],[],[], color='C1', label=r'fixed $\bar{n}$') 
    bkgd.legend(loc='upper right', bbox_to_anchor=(0.875, 0.775), fontsize=25)
    bkgd.text(0.8, 0.61, r'$k_{\rm max} = %.1f$; $z=0.$' % kmax, ha='right', va='bottom', 
            transform=bkgd.transAxes, fontsize=25)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.05, hspace=0.05) 
    ffig = os.path.join(UT.fig_dir(), 'quijote_%s_fixednbar_Fisher_kmax%.2f.png' % (obs, kmax))
    fig.savefig(ffig, bbox_inches='tight') 
    return None


# sigma_theta(kmax) tests
def quijote_Forecast_Fii_kmax_fixednbar(dmnu='fin', s8corr=False):
    ''' 1/sqrt(Fii) as a function of kmax 
    '''
    #kmaxs = np.pi/500. * 3 * np.arange(3, 28) 
    kmaxs = np.pi/500. * 3 * np.array([3, 5, 10, 15, 20, 27]) 
    
    klin = np.logspace(-5, 2, 500)
    print(', '.join([str(tt) for tt in thetas]))
    # read in fisher matrix (Fij)
    Fii_pk, Fii_bk, Fii_pm, Fii_pcb = [], [], [], [] 
    for i_k, kmax in enumerate(kmaxs): 
        print('kmax=%.3f ---' %  kmax) 
        # linear theory Pm 
        Fij = LT.Fij_Pm(klin, kmax=kmax, npoints=5) 
        Fii_pm.append(1./np.sqrt(np.diag(Fij)))
        Fij = LT.Fij_Pm(klin, kmax=kmax, npoints=5, flag='cb') 
        Fii_pcb.append(1./np.sqrt(np.diag(Fij)))
        print('pm: %s' % ', '.join(['%.2e' % fii for fii in np.diag(Fij)[:6]])) 
        Fij = quijote_Fisher_fixednbar('pk', kmax=kmax, dmnu=dmnu)
        Fii_pk.append(1./np.sqrt(np.diag(Fij)))
        print('pk: %s' % ', '.join(['%.2e' % fii for fii in np.diag(Fij)[:6]])) 
        Fij = quijote_Fisher_fixednbar('bk', kmax=kmax, dmnu=dmnu) 
        Fii_bk.append(1./np.sqrt(np.diag(Fij)))
        print('bk: %s' % ', '.join(['%.2e' % fii for fii in np.diag(Fij)[:6]])) 
    Fii_pk = np.array(Fii_pk)
    Fii_bk = np.array(Fii_bk)
    Fii_pm = np.array(Fii_pm)
    Fii_pcb = np.array(Fii_pcb)
    sigma_theta_lims = [(1e-4, 1.), (1e-4, 1.), (1e-3, 2), (1e-3, 2.), (1e-4, 1.), (5e-3, 10.)]

    fig = plt.figure(figsize=(15,8))
    for i, theta in enumerate(thetas): 
        sub = fig.add_subplot(2,len(thetas)/2,i+1) 
        plt_ph, = sub.plot(kmaxs, Fii_pk[:,i], c='C0')
        plt_bh, = sub.plot(kmaxs, Fii_bk[:,i], c='C1') 
        plt_pm, = sub.plot(kmaxs, Fii_pm[:,i], c='k', ls='-') 
        plt_pc, = sub.plot(kmaxs, Fii_pcb[:,i], c='k', ls=':') 
        if i == 0: sub.legend([plt_ph, plt_bh], [r'$P_h$ fix. $\bar{n}$', r'$B_h$ fix. $\bar{n}$'], loc='upper left', fontsize=15) 
        if theta == 'Mnu': sub.legend([plt_pm, plt_bc], [r'$P_m$', r'$P_{cb}$'], loc='upper left', fontsize=15) 
        sub.set_xlim(0.005, 0.5)
        sub.text(0.95, 0.95, theta_lbls[i], ha='right', va='top', transform=sub.transAxes, fontsize=25)
        sub.set_ylim(sigma_theta_lims[i]) 
        sub.set_yscale('log') 

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'$k_{\rm max}$', fontsize=28) 
    bkgd.set_ylabel(r'unmarginalized $\sigma_\theta$ $(1/\sqrt{F_{i,i}})$', labelpad=10, fontsize=28) 
    fig.subplots_adjust(wspace=0.2, hspace=0.15) 
    ffig = os.path.join(UT.fig_dir(), 'quijote_Fii_dmnu_%s_kmax_fixednbar.png' % dmnu)
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def quijote_Forecast_sigmatheta_kmax_fixednbar(tts=['Om', 'Ob', 'h', 'ns', 's8', 'Mnu'], dmnu='fin', 
        s8corr=False, bSN=False):
    ''' fisher forecast for quijote for different kmax values 
    '''
    kmaxs = np.pi/500. * 3 * np.array([3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 28]) 

    klin = np.logspace(-5, 2, 500)
    
    i_tts = np.zeros(len(thetas)).astype(bool) 
    for tt in tts: i_tts[thetas.index(tt)] = True
    if bSN: 
        _i_tts = np.zeros(len(thetas)+2).astype(bool) 
        for tt in tts: _i_tts[thetas.index(tt)] = True
        _i_tts[-2] = True
        _i_tts[-1] = True
    else: _i_tts = i_tts

    # read in fisher matrix (Fij)
    sig_pk, sig_pm, sig_pcb, _sig_pk = [], [], [], []
    for i_k, kmax in enumerate(kmaxs): 
        pmFij = LT.Fij_Pm(klin, kmax=kmax, npoints=5) # linear theory Pm 
        if 'Mnu' in tts: pcbFij = LT.Fij_Pm(klin, kmax=kmax, npoints=5, flag='cb') # LT Pcb 
        pkFij = quijote_Fisher_fixednbar('pk', kmax=kmax, dmnu=dmnu, s8corr=s8corr, bSN=bSN) # halo P(k)
        if s8corr or bSN: _pkFij = quijote_Fisher_fixednbar('pk', kmax=kmax, dmnu=dmnu, s8corr=False, bSN=False) # fiducial halo P

        sig_pm.append(np.sqrt(np.diag(np.linalg.inv(pmFij[:,i_tts][i_tts,:]))))
        if 'Mnu' in tts: sig_pcb.append(np.sqrt(np.diag(np.linalg.inv(pcbFij[:,i_tts][i_tts,:]))))
        sig_pk.append(np.sqrt(np.diag(np.linalg.inv(pkFij[:,_i_tts][_i_tts,:]))))
        if s8corr or bSN: _sig_pk.append(np.sqrt(np.diag(np.linalg.inv(_pkFij[:,i_tts][i_tts,:]))))

    sig_pk = np.array(sig_pk)
    sig_pm = np.array(sig_pm)
    if 'Mnu' in tts: sig_pcb = np.array(sig_pcb)
    if s8corr or bSN: _sig_pk = np.array(_sig_pk) 

    sigma_theta_lims = np.array([(5e-3, 5.), (1e-3, 5.), (1e-2, 50), (1e-2, 20.), (1e-2, 50.), (1e-2, 1e3)])[i_tts]
    _lbls = np.array(theta_lbls)[i_tts]

    fig = plt.figure(figsize=(15,8))
    for i, theta in enumerate(tts): 
        sub = fig.add_subplot(2,len(thetas)/2,i+1) 
        if s8corr or bSN: 
            plt_pk, = sub.plot(kmaxs, sig_pk[:,i], c='C1', ls='-') 
            _plt_pk, = sub.plot(kmaxs, _sig_pk[:,i], c='C0', ls='-') 
            plts = [_plt_pk, plt_pk]
            _lbl = r'$P_h$ fix. $\bar{n}$ '
            if s8corr: _lbl += 'corr. $\sigma_8$ '
            if bSN: _lbl +=  r'marg. $b_1$ $A_{\rm SN}$'
            if i == 0: sub.legend(plts, [r'$P_h$ fix. $\bar{n}$', _lbl], loc='lower left', fontsize=15) 
        else: 
            plt_pk, = sub.plot(kmaxs, sig_pk[:,i], c='C0', ls='-') 
            if i == 0: sub.legend([plt_pk], [r'$P_h$ fix. $\bar{n}$']) 
        sub.plot(kmaxs, sig_pm[:,i], c='k', ls='--', label=r"$P^{\rm lin.}_{m}$") 
        if 'Mnu' in tts: sub.plot(kmaxs, sig_pcb[:,i], c='k', ls=':', label=r"$P^{\rm lin.}_{cb}$") 
        if theta == 'Mnu': sub.legend(loc='lower left', fontsize=15) 
        sub.set_xlim(0.05, 0.5)
        sub.text(0.9, 0.9, _lbls[i], ha='right', va='top', transform=sub.transAxes, fontsize=30)
        sub.set_ylim(sigma_theta_lims[i]) 
        sub.set_yscale('log') 

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'$k_{\rm max}$', fontsize=28) 
    bkgd.set_ylabel(r'marginalized $\sigma_\theta$', labelpad=10, fontsize=28) 
    
    flag_str = ''
    if s8corr: flag_str += '.s8corr'
    if bSN: flag_str += '.bSN'

    fig.subplots_adjust(wspace=0.2, hspace=0.15) 
    if np.sum(i_tts) == len(thetas): 
        ffig = ('quijote_Fisher_dmnu_%s_sigmakmax_fixednbar%s.png' % (dmnu, flag_str))
    else: 
        ffig = ('quijote_Fisher_dmnu_%s_sigmakmax_fixednbar%s.%s.png' % (dmnu, flag_str, ''.join(tts)))
    fig.savefig(os.path.join(UT.fig_dir(), ffig), bbox_inches='tight') 
    return None


def quijote_Fij_LT_kmax_fixednbar(kmax=0.5, dmnu='fin'):
    ''' compare fisher matrix of Ph with fixed nbar w/ LT fisher matrix.  
    '''
    klin = np.logspace(-5, 2, 500)
    # read in fisher matrices (Fij)
    pmFij   = LT.Fij_Pm(klin, kmax=kmax, npoints=5)
    pcbFij  = LT.Fij_Pm(klin, kmax=kmax, npoints=5, flag='cb')
    pkFij   = quijote_Fisher_fixednbar('pk', kmax=kmax, dmnu=dmnu, s8corr=True)
    bkFij   = quijote_Fisher_fixednbar('bk', kmax=kmax, dmnu=dmnu)

    fig = plt.figure(figsize=(20,5))
    sub = fig.add_subplot(141)
    cm = sub.pcolormesh(pmFij, norm=SymLogNorm(1e3, vmin=-4e6, vmax=4e6), cmap='RdBu')
    sub.set_xticks(np.arange(pmFij.shape[0])+0.5) 
    sub.set_xticklabels(np.array(theta_lbls)) 
    sub.set_yticks(np.arange(pmFij.shape[0])+0.5) 
    sub.set_yticklabels(np.array(theta_lbls)) 
    sub.set_title('$F_{ij}$ $P_m$', fontsize=20) 
    sub = fig.add_subplot(142)
    cm = sub.pcolormesh(pcbFij, norm=SymLogNorm(1e3, vmin=-4e6, vmax=4e6), cmap='RdBu')
    sub.set_xticks(np.arange(pmFij.shape[0])+0.5) 
    sub.set_xticklabels(np.array(theta_lbls)) 
    sub.set_yticks(np.arange(pmFij.shape[0])+0.5) 
    sub.set_yticklabels([]) 
    sub.set_title('$F_{ij}$ $P_{cb}$', fontsize=20) 
    sub = fig.add_subplot(143)
    cm = sub.pcolormesh(pkFij, norm=SymLogNorm(1e3, vmin=-4e6, vmax=4e6), cmap='RdBu')
    sub.set_xticks(np.arange(pmFij.shape[0])+0.5) 
    sub.set_xticklabels(np.array(theta_lbls)) 
    sub.set_xticks(np.arange(pmFij.shape[0])+0.5) 
    sub.set_yticklabels([]) 
    sub.set_title(r'$F_{ij}$ $P_h$ fixed $\bar{n}$', fontsize=20) 
    sub = fig.add_subplot(144)
    cm = sub.pcolormesh(bkFij, norm=SymLogNorm(1e3, vmin=-4e6, vmax=4e6), cmap='RdBu')
    sub.set_xticks(np.arange(pmFij.shape[0])+0.5) 
    sub.set_xticklabels(np.array(theta_lbls)) 
    sub.set_xticks(np.arange(pmFij.shape[0])+0.5) 
    sub.set_yticklabels([]) 
    sub.set_title(r'$F_{ij}$ $B_h$ fixed $\bar{n}$', fontsize=20) 

    fig.subplots_adjust(wspace=0.05, right=0.9)
    cbar_ax = fig.add_axes([0.925, 0.15, 0.02, 0.7])
    fig.colorbar(cm, cax=cbar_ax) 
    ffig = os.path.join(UT.fig_dir(), 'quijote_Fij_LT_kmax_fixednbar.kmax%.1f.png' % kmax) 
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def quijote_PhFij_PcbFij(kmax=0.1, dmnu='fin'):
    ''' compare the ratio between fisher matrix of Ph with fixed nbar w/ Pcb LT fisher matrix
    '''
    klin = np.logspace(-5, 2, 500)
    # read in fisher matrices (Fij)
    pcbFij  = LT.Fij_Pm(klin, kmax=kmax, npoints=5, flag='cb')
    pkFij   = quijote_Fisher_fixednbar('pk', kmax=kmax, dmnu='fin', s8corr=True)
    i_h = thetas.index('h') 
    pkFij[:,i_h] *= -1.
    pkFij[i_h,:] *= -1.

    fig = plt.figure(figsize=(5,5))
    sub = fig.add_subplot(111)
    cm = sub.pcolormesh(pkFij/pcbFij, vmin=-2., vmax=2.) 
    #norm=SymLogNorm(1e3, vmin=-4e6, vmax=4e6), cmap='RdBu')
    sub.set_xticks(np.arange(pkFij.shape[0])+0.5) 
    sub.set_xticklabels(np.array(theta_lbls)) 
    sub.set_yticks(np.arange(pkFij.shape[0])+0.5) 
    sub.set_yticklabels(np.array(theta_lbls)) 
    sub.set_title('$(F_{ij}$ $P_{h})/(F_{ij}$ $P_{cb})$', fontsize=20) 
    cbar_ax = fig.add_axes([0.925, 0.15, 0.02, 0.7])
    fig.colorbar(cm, cax=cbar_ax) 
    ffig = os.path.join(UT.fig_dir(), 'quijote_PhFij_PcbFij.kmax%.1f.png' % kmax) 
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def quijote_Forecast_sigma_kmax_fixednbar_scaleFij(dmnu='fin', f_b=0.05, f_s8=0.025, f_mnu=0.10):
    ''' fisher forecast for quijote as a function of kmax 
    '''
    kmaxs = np.pi/500. * 3 * np.array([3, 5, 10, 15, 20, 27]) 
    
    i_b, i_s8, i_mnu = thetas.index('Ob'), thetas.index('s8'), thetas.index('Mnu') 
    print(', '.join([str(tt) for tt in thetas]))
    # read in fisher matrix (Fij)
    sigma_thetas_pk, sigma_thetas_bk, sigma_thetas_Pcb = [], [], []
    wellcond_pk, wellcond_bk = np.ones(len(kmaxs)).astype(bool), np.ones(len(kmaxs)).astype(bool) 
    for i_k, kmax in enumerate(kmaxs): 
        # linear theory Pcb
        Fij = LT.Fij_Pm(np.logspace(-5, 2, 500), kmax=kmax, npoints=5, flag='cb') 
        Finv = np.linalg.inv(Fij) # invert fisher matrix 
        if np.linalg.cond(Fij) > 1e16: wellcond_pk[i_k] = False 
        sigma_thetas_Pcb.append(np.sqrt(np.diag(Finv)))
        print('pcb: %s' % ', '.join(['%.2e' % fii for fii in np.diag(Fij)[:6]])) 

        Fij = quijote_Fisher_fixednbar('pk', kmax=kmax, dmnu=dmnu)
        Finv = np.linalg.inv(Fij) # invert fisher matrix 
        if np.linalg.cond(Fij) > 1e16: wellcond_pk[i_k] = False 
        sigma_thetas_pk.append(np.sqrt(np.diag(Finv)))
        print('pk: %s' % ', '.join(['%.2e' % fii for fii in np.diag(Fij)[:6]])) 

        Fij = quijote_Fisher_fixednbar('bk', kmax=kmax, dmnu=dmnu) 
        if np.linalg.cond(Fij) > 1e16: wellcond_bk[i_k] = False 
        Fij[i_b, :] /= np.sqrt(1.+f_b)
        Fij[:,i_b] /= np.sqrt(1.+f_b)
        Fij[i_s8, :] /= np.sqrt(1.+f_s8)
        Fij[:,i_s8] /= np.sqrt(1.+f_s8)
        Fij[i_mnu, :] /= np.sqrt(1.+f_mnu)
        Fij[:,i_mnu] /= np.sqrt(1.+f_mnu)
        Finv = np.linalg.inv(Fij) # invert fisher matrix 
        sigma_thetas_bk.append(np.sqrt(np.diag(Finv)))
        print('bk: %s' % ', '.join(['%.2e' % fii for fii in np.diag(Fij)[:6]])) 
        #print('kmax=%.3e' % kmax)
        #print('Pk sig_theta =', sigma_thetas_pk[-1][:]) 
        #print('Bk sig_theta =', sigma_thetas_bk[-1][:])
    sigma_thetas_pk = np.array(sigma_thetas_pk)
    sigma_thetas_bk = np.array(sigma_thetas_bk)
    sigma_thetas_Pcb = np.array(sigma_thetas_Pcb)
    #sigma_theta_lims = [(0, 0.1), (0., 0.08), (0., 0.8), (0, 0.8), (0., 0.12), (0., 0.8)]
    #sigma_theta_lims = [(0, 0.2), (0., 0.2), (0., 2.), (0, 2.), (0., 1.), (0., 2.)]
    #sigma_theta_lims = [(0, 10.), (0., 10.), (0., 10.), (0, 10.), (0., 10.), (0., 10.)]
    sigma_theta_lims = [(5e-3, 5.), (1e-3, 5.), (1e-2, 50), (1e-2, 20.), (1e-2, 50.), (1e-2, 1e3)]

    fig = plt.figure(figsize=(15,8))
    for i, theta in enumerate(thetas): 
        sub = fig.add_subplot(2,len(thetas)/2,i+1) 
        sub.plot(kmaxs, sigma_thetas_pk[:,i], c='C0', ls='-') 
        sub.plot(kmaxs, sigma_thetas_bk[:,i], c='C1', ls='-') 
        sub.plot(kmaxs, sigma_thetas_Pcb[:,i], c='k', ls=':', label=r"$P^{\rm lin.}_{cb}$") 
        if theta == 'Mnu': sub.legend(loc='lower left', fontsize=15) 
        sub.set_xlim(0.05, 0.5)
        sub.text(0.9, 0.9, theta_lbls[i], ha='right', va='top', transform=sub.transAxes, fontsize=30)
        sub.set_ylim(sigma_theta_lims[i]) 
        sub.set_yscale('log') 
        if i == 0: 
            sub.text(0.5, 0.35, r"$P$", ha='left', va='bottom', color='C0', transform=sub.transAxes, fontsize=24)
            sub.text(0.25, 0.2, r"$B$", ha='right', va='top', color='C1', transform=sub.transAxes, fontsize=24)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'$k_{\rm max}$', fontsize=28) 
    bkgd.set_ylabel(r'$1\sigma$ constraint on $\theta$', labelpad=10, fontsize=28) 

    fig.subplots_adjust(wspace=0.2, hspace=0.15) 
    ffig = os.path.join(UT.fig_dir(), 
            'quijote_Fisher_dmnu_%s_sigmakmax_fixednbar_scaledFij.png' % dmnu)
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def quijote_Forecast_sigma_kmax_fixednbar_negh(dmnu='fin'):
    ''' fisher forecast for quijote as a function of kmax. Here we flip the sign of the 
    deriatives w.r.t. to h. 
    '''
    kmaxs = np.pi/500. * 3 * np.array([3, 5, 10, 15, 20, 27]) 

    klin = np.logspace(-5, 2, 500)

    i_h = thetas.index('h') 
    # read in fisher matrix (Fij)
    sig_pk, sig_pk_nh, sig_bk, sig_Pm, sig_Pcb = [], [], [], [], [] 
    for i_k, kmax in enumerate(kmaxs): 
        # linear theory Pm 
        pmFij = LT.Fij_Pm(klin, kmax=kmax, npoints=5) 
        pcbFij = LT.Fij_Pm(klin, kmax=kmax, npoints=5, flag='cb') 
        pkFij = quijote_Fisher_fixednbar('pk', kmax=kmax, dmnu=dmnu)
        pkFij_nh = pkFij.copy() 
        pkFij_nh[i_h,:] *= -1.
        pkFij_nh[:,i_h] *= -1.
        print pkFij[i_h,:]
        print pkFij_nh[i_h,:]
        #bkFij = quijote_Fisher_fixednbar('bk', kmax=kmax, dmnu=dmnu) 

        sig_Pm.append(np.sqrt(np.diag(np.linalg.inv(pmFij))))
        sig_Pcb.append(np.sqrt(np.diag(np.linalg.inv(pcbFij))))
        sig_pk.append(np.sqrt(np.diag(np.linalg.inv(pkFij))))
        sig_pk_nh.append(np.sqrt(np.diag(np.linalg.inv(pkFij_nh))))
        #sig_bk.append(np.sqrt(np.diag(np.linalg.inv(bkFij))))
    sig_Pm = np.array(sig_Pm)
    sig_Pcb = np.array(sig_Pcb)
    sig_pk = np.array(sig_pk)
    sig_pk_nh = np.array(sig_pk_nh)
    #sig_bk = np.array(sig_bk)
    sigma_theta_lims = [(5e-3, 5.), (1e-3, 5.), (1e-2, 50), (1e-2, 20.), (1e-2, 50.), (1e-2, 1e3)]

    fig = plt.figure(figsize=(15,8))
    for i, theta in enumerate(thetas): 
        sub = fig.add_subplot(2,len(thetas)/2,i+1) 
        sub.plot(kmaxs, sig_pk[:,i], c='C0', ls='-', label='$P_h$') 
        sub.plot(kmaxs, sig_pk_nh[:,i], c='C1', ls='-', label='$P_h$ (-$h$ deriv.)') 
        #sub.plot(kmaxs, sig_bk[:,i], c='C1', ls='-') 
        sub.plot(kmaxs, sig_Pm[:,i], c='k', ls='--', label=r"$P^{\rm lin.}_{m}$") 
        sub.plot(kmaxs, sig_Pcb[:,i], c='k', ls=':', label=r"$P^{\rm lin.}_{cb}$") 
        if theta == 'Mnu': sub.legend(loc='lower left', fontsize=15) 
        sub.set_xlim(0.05, 0.5)
        sub.text(0.9, 0.9, theta_lbls[i], ha='right', va='top', 
                transform=sub.transAxes, fontsize=30)
        sub.set_ylim(sigma_theta_lims[i]) 
        sub.set_yscale('log') 

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'$k_{\rm max}$', fontsize=28) 
    bkgd.set_ylabel(r'$1\sigma$ constraint on $\theta$', labelpad=10, fontsize=28) 

    fig.subplots_adjust(wspace=0.2, hspace=0.15) 
    ffig = os.path.join(UT.doc_dir(), 'figs', 
            'quijote_Fisher_dmnu_%s_sigmakmax_fixednbar_negh.png' % dmnu)
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def quijote_dPk(theta, dmnu='fin', log=False, Nfp=None):
    ''' calculate d P0(k)/d theta using the paired and fixed quijote simulations
    run on perturbed theta 

    :param theta: 
        string that specifies the parameter to take the 
        derivative by. 
    '''
    c_dpk = 0.
    if theta == 'Mnu': 
        tts = ['fiducial', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp']
        if dmnu == 'p': 
            coeffs = [-1., 1., 0., 0.]
            h = 0.1
        elif dmnu == 'pp': 
            coeffs = [-1., 0., 1., 0.]
            h = 0.2
        elif dmnu == 'ppp': 
            coeffs = [-1., 0., 0., 1.]
            h = 0.4
        elif dmnu == 'fin0': 
            coeffs = [-3., 4., -1., 0.] # finite difference coefficient
            h = 0.2
        elif dmnu == 'fin': 
            coeffs = [-21., 32., -12., 1.] # finite difference coefficient
            h = 1.2
    elif theta == 'Mmin': # halo mass limit 
        tts = ['Mmin_m', 'Mmin_p'] 
        coeffs = [-1., 1.] 
        h = 0.2 # 3.3 - 3.1 x 10^13 Msun 
    elif theta == 'Amp': 
        # amplitude of P(k) is a free parameter
        tts = ['fiducial'] 
        coeffs = [0.] 
        h = 1. 
        quij = Obvs.quijoteP0k('fiducial') 
        if not log: c_dpk = np.average(quij['p0k'], axis=0) 
        else: c_dpk = np.ones(quij['p0k'].shape[1]) 
    elif theta == 'Asn' : 
        # constant shot noise term is a free parameter
        tts = ['fiducial'] 
        coeffs = [0.] 
        h = 1. 
        quij = Obvs.quijoteP0k('fiducial') 
        if not log: c_dpk = np.ones(quij['p0k'].shape[1]) 
        else: c_dpk = 1./np.average(quij['p0k'], axis=0) 
    else: 
        tts = [theta+'_m', theta+'_p'] 
        coeffs = [-1., 1.] 
        h = quijote_thetas[theta][1] - quijote_thetas[theta][0]

    for i_tt, tt, coeff in zip(range(len(tts)), tts, coeffs): 
        quij = Obvs.quijoteP0k(tt) # read P0k 
        if i_tt == 0: dpk = np.zeros(quij['p0k'].shape[1]) 
    
        if Nfp is not None and tt != 'fiducial': 
            _pk = np.average(quij['p0k'][:Nfp,:], axis=0)  
        else: 
            _pk = np.average(quij['p0k'], axis=0)  

        if log: _pk = np.log(_pk) # log 

        dpk += coeff * _pk 
    return dpk / h + c_dpk 


def quijote_Fisher_fixednbar_LTdtheta(obs, tt_LT, kmax=0.5, dmnu='fin', fhartlap=True, s8corr=True): 
    ''' calculate fisher matrix for parameters ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
    with fixed nbar where derivative w.r.t. to tt_LT is the linear theory Pm/Pcb derivative.
    '''
    quij = Obvs.quijoteBk('fiducial', flag='.fixed_nbar') # theta_fiducial 
    i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3']

    if obs == 'pk': 
        pks = quij['p0k1'] + 1e9/quij['Nhalos'][:,None] # uncorrect shot noise 
        _, _iuniq = np.unique(i_k, return_index=True)
        iuniq = np.zeros(len(i_k)).astype(bool) 
        iuniq[_iuniq] = True
        klim = (iuniq & (i_k * kf < kmax)) 

        nmock = quij['p0k1'].shape[0]
        ndata = np.sum(klim) 

        C_fid = np.cov(pks[:,klim].T) # covariance matrix 
    else: 
        raise ValueError

    if fhartlap: f_hartlap = float(nmock - ndata - 2)/float(nmock - 1) 
    else: f_hartlap = 1. 
    C_inv = f_hartlap * np.linalg.inv(C_fid) # invert the covariance 
    
    dobs_dt = [] 
    for par in thetas:#+['Amp']: # calculate the derivative of Bk along all the thetas 
        if par != tt_LT: 
            dobs_dti = quijote_dPk_fixednbar(par, dmnu=dmnu, s8corr=s8corr) 
            dobs_dt.append(dobs_dti[klim])
        else: 
            _p0k = np.average(quij['p0k1'], axis=0)[klim]
            dobs_dti = _p0k * LT.dPmdtheta(tt_LT, kf * i_k[klim], log=True, npoints=5, flag='cb')
            dobs_dt.append(dobs_dti)
    Fij = Forecast.Fij(dobs_dt, C_inv) 
    return Fij 


def quijote_Forecast_sigma_kmax_fixednbar_LTdtheta(tt_LT, dmnu='fin'):
    ''' fisher forecast for quijote for different kmax values 
    '''
    #kmaxs = np.pi/500. * 3 * np.arange(3, 15) 
    kmaxs = np.pi/500. * 3 * np.array([3, 5, 10, 15, 20])#, 27]) 

    klin = np.logspace(-5, 2, 500)
    print(', '.join([str(tt) for tt in thetas]))

    # read in fisher matrix (Fij)
    _sig_pk, sig_pk, sig_Pm, sig_Pcb = [], [], [], [] 
    for i_k, kmax in enumerate(kmaxs): 
        # linear theory Pm 
        Fij = LT.Fij_Pm(klin, kmax=kmax, npoints=5) 
        Finv = np.linalg.inv(Fij) # invert fisher matrix 
        sig_Pm.append(np.sqrt(np.diag(Finv)))
        print('kmax=%.3f ---' %  kmax) 
        print('pm: %s' % ', '.join(['%.2e' % fii for fii in np.diag(Fij)[:6]])) 
        # linear theory Pcb
        Fij = LT.Fij_Pm(klin, kmax=kmax, npoints=5, flag='cb') 
        Finv = np.linalg.inv(Fij) # invert fisher matrix 
        sig_Pcb.append(np.sqrt(np.diag(Finv)))
        print('pcb: %s' % ', '.join(['%.2e' % fii for fii in np.diag(Fij)[:6]])) 
        
        Fij = quijote_Fisher_fixednbar('pk', kmax=kmax, dmnu=dmnu)
        print Fij[thetas.index(tt_LT),:]
        Finv = np.linalg.inv(Fij) # invert fisher matrix 
        _sig_pk.append(np.sqrt(np.diag(Finv)))
        print('_pk: %s' % ', '.join(['%.2e' % fii for fii in np.diag(Fij)[:6]])) 

        Fij = quijote_Fisher_fixednbar_LTdtheta('pk', tt_LT, kmax=kmax, dmnu=dmnu)
        print Fij[thetas.index(tt_LT),:]
        Finv = np.linalg.inv(Fij) # invert fisher matrix 
        sig_pk.append(np.sqrt(np.diag(Finv)))
        print('pk: %s' % ', '.join(['%.2e' % fii for fii in np.diag(Fij)[:6]])) 
    sig_pk = np.array(sig_pk)
    _sig_pk = np.array(_sig_pk)
    sig_Pm = np.array(sig_Pm)
    sig_Pcb = np.array(sig_Pcb)
    sigma_theta_lims = [(1e-3, 1e6), (1e-3, 1e5), (1e-2, 1e6), (1e-2, 1e6), (5e-3, 1e6), (1e-1, 5e7)]

    fig = plt.figure(figsize=(15,8))
    for i, theta in enumerate(thetas): 
        sub = fig.add_subplot(2,len(thetas)/2,i+1) 
        sub.plot(kmaxs, _sig_pk[:,i], c='C0', ls='-', label="$P_h$") 
        sub.plot(kmaxs, sig_pk[:,i], c='C1', ls='-', label=r"$P_h$ LT ${\rm d}P/{\rm d}M_\nu$") 
        sub.plot(kmaxs, sig_Pm[:,i], c='k', ls='--', label=r"$P^{\rm lin.}_{m}$") 
        sub.plot(kmaxs, sig_Pcb[:,i], c='k', ls=':', label=r"$P^{\rm lin.}_{cb}$") 
        if theta == 'Mnu': sub.legend(loc='lower left', fontsize=15) 
        sub.set_xlim(0.05, 0.5)
        sub.text(0.9, 0.9, theta_lbls[i], ha='right', va='top', 
                transform=sub.transAxes, fontsize=30)
        sub.set_ylim(sigma_theta_lims[i]) 
        sub.set_yscale('log') 

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'$k_{\rm max}$', fontsize=28) 
    bkgd.set_ylabel(r'$1\sigma$ constraint on $\theta$', labelpad=10, fontsize=28) 

    fig.subplots_adjust(wspace=0.2, hspace=0.15) 
    ffig = os.path.join(UT.doc_dir(), 'figs', 
            'quijote_Fisher_dmnu_%s_sigmakmax_fixednbar_LTd%s.png' % (dmnu, tt_LT))
    fig.savefig(ffig, bbox_inches='tight') 
    return None


if __name__=="__main__": 
    #Pk_fixednbar()
    #quijote_dPdthetas_LT_fixednbar(dmnu='fin')
    #quijote_Fij_LT_kmax_fixednbar(kmax=0.1, dmnu='fin')
    #quijote_Forecast_Fii_kmax_fixednbar(dmnu='fin')
    quijote_Forecast_sigmatheta_kmax_fixednbar(dmnu='fin')
    quijote_Forecast_sigmatheta_kmax_fixednbar(dmnu='fin', s8corr=True)
    quijote_Forecast_sigmatheta_kmax_fixednbar(dmnu='fin', s8corr=True, bSN=True)
    #quijote_Forecast_sigmatheta_kmax_fixednbar(['Om', 'Ob', 'h', 'ns', 's8'], dmnu='fin')
    #for tt in ['Om', 'Ob', 'h', 'ns', 's8']: 
    #    quijote_Forecast_sigmatheta_kmax_fixednbar([tt, 'Mnu'], dmnu='fin')
    #quijote_Forecast_sigma_kmax_fixednbar_negh(dmnu='fin')
    #quijote_PhFij_PcbFij(kmax=0.1, dmnu='fin')
    #quijote_PhFij_PcbFij(kmax=0.2, dmnu='fin')
    #for tt in ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']: 
    #   quijote_Forecast_sigma_kmax_fixednbar_LTdtheta(tt, dmnu='fin')
    #quijote_Forecast_sigma_kmax_fixednbar_bSN(dmnu='fin')
    ''' 
        quijote_dBdthetas_fixednbar(dmnu='fin')
        quijote_Forecast_fixednbar('pk', kmax=0.2, dmnu='fin')
        quijote_Forecast_fixednbar('bk', kmax=0.2, dmnu='fin')
    '''
