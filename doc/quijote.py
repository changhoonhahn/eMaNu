'''

forecasts using Quijote simulations 

'''
import os 
import h5py
import numpy as np 
from copy import copy as copy
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


dir_bk  = os.path.join(UT.dat_dir(), 'bispectrum') # bispectrum directory
dir_fig = UT.fig_dir() # figure directory
dir_doc = os.path.join(UT.doc_dir(), 'paper1', 'figs') # figures for paper 

quijote_thetas = {
        'Mnu': [0.1, 0.2, 0.4], # +, ++, +++ 
        'Ob': [0.048, 0.050], # others are - + 
        'Ob2': [0.047, 0.051], # others are - + 
        'Om': [0.3075, 0.3275],
        'h': [0.6511, 0.6911],
        'ns': [0.9424, 0.9824],
        's8': [0.819, 0.849] }
##################################################################
# qujiote fisher 
##################################################################
kf = 2.*np.pi/1000. # fundmaentla mode
thetas = ['Om', 'Ob2', 'h', 'ns', 's8', 'Mnu']
thetas_pk = ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
theta_lbls = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$']
theta_lims = [(0.25, 0.385), (0.02, 0.08), (0.3, 1.1), (0.6, 1.35), (0.8, 0.87), (-0.45, 0.45)]
theta_fid = {'Mnu': 0., 'Ob': 0.049, 'Ob2': 0.049, 'Om': 0.3175, 'h': 0.6711,  'ns': 0.9624,  's8': 0.834} # fiducial theta 
ntheta = len(thetas)
# nuisance params.
theta_nuis_lbls = {'Mmin': r'$M_{\rm min}$', 'Amp': "$b'$", 'Asn': r"$A_{\rm SN}$", 'Bsn': r"$B_{\rm SN}$", 
        'b2': '$b_2$', 'g2': '$g_2$'}
theta_nuis_fids = {'Mmin': 3.2, 'Amp': 1., 'Asn': 1e-6, 'Bsn': 1e-3, 'b2': 1., 'g2': 1.} 
theta_nuis_lims = {'Mmin': (2.1, 4.3), 'Amp': (0.75, 1.25), 'Asn': (0., 1.), 'Bsn': (0., 1.), 
        'b2': (0.95, 1.05), 'g2': (0.95, 1.05)} 

fscale_pk = 2e5


def pkCov(rsd=2, flag='reg', silent=True): 
    ''' calculate the covariance matrix of the RSD quijote powerspectrum. 
    '''
    assert flag == 'reg', "only n-body should be used for covariance"
    assert rsd != 'all', "only one RSD direction should be used otherwise modes will be correlated" 

    fcov = os.path.join(dir_bk, 'quijote_pCov_full%s%s.hdf5' % (_rsd_str(rsd), _flag_str(flag)))
    if os.path.isfile(fcov): 
        if not silent: print('reading ... %s' % os.path.basename(fcov))
        Fcov = h5py.File(fcov, 'r') # read in fiducial covariance matrix 
        C_pk = Fcov['C_pk'][...]
        k = Fcov['k'][...]
        Nmock = Fcov['Nmock'][...]
    else:
        if not silent: print('calculating ... %s' % os.path.basename(fcov))
        # read in P(k) 
        quij = Obvs.quijotePk('fiducial', rsd=rsd, flag=flag, silent=silent) 
        pks = quij['p0k'] + quij['p_sn'][:,None] # shotnoise uncorrected P(k) 
        C_pk = np.cov(pks.T) # covariance matrix 
        k = quij['k'] 
        Nmock = quij['p0k'].shape[0] 

        f = h5py.File(fcov, 'w') # write to hdf5 file 
        f.create_dataset('Nmock', data=Nmock)
        f.create_dataset('C_pk', data=C_pk) 
        f.create_dataset('k', data=k) 
        f.close()
    return k, C_pk, Nmock 


def p02kCov(rsd=2, flag='reg', silent=True): 
    ''' calculate the covariance matrix of the RSD quijote power spectrum
    monopole *and* quadrupole. 
    '''
    assert flag == 'reg', "only n-body should be used for covariance"
    assert rsd != 'all', "only one RSD direction should be used otherwise modes will be correlated" 

    fcov = os.path.join(dir_bk, 'quijote_p02Cov_full%s%s.hdf5' % (_rsd_str(rsd), _flag_str(flag)))
    if os.path.isfile(fcov): 
        if not silent: print('reading ... %s' % os.path.basename(fcov))
        Fcov = h5py.File(fcov, 'r') # read in fiducial covariance matrix 
        C_pk = Fcov['C_pk'][...]
        k = Fcov['k'][...]
        Nmock = Fcov['Nmock'][...]
    else:
        if not silent: print('calculating ... %s' % os.path.basename(fcov))
        # read in P(k) 
        quij = Obvs.quijotePk('fiducial', rsd=rsd, flag=flag, silent=silent) 
        p0ks = quij['p0k'] + quij['p_sn'][:,None] # shotnoise uncorrected P0 
        p2ks = quij['p2k'] # P2 
        pks = np.concatenate([p0ks, p2ks], axis=1)  # concatenate P0 and P2
        C_pk = np.cov(pks.T) # covariance matrix 
        k = np.concatenate([quij['k'], quij['k']]) 
        Nmock = quij['p0k'].shape[0] 

        f = h5py.File(fcov, 'w') # write to hdf5 file 
        f.create_dataset('Nmock', data=Nmock)
        f.create_dataset('C_pk', data=C_pk) 
        f.create_dataset('k', data=k) 
        f.close()
    return k, C_pk, Nmock 


def bkCov(rsd=2, flag='reg', silent=True): 
    ''' return the full covariance matrix of the quijote bispectrum
    computed using 
    - rsd == 'all': 46500 simulations (15000 n-body x 3  + 500 ncv x 3) 
    - rsd == 0: 15500 simulations (15000 n-body + 500 ncv) 
    - rsd == 'real': 15500 simulations (15000 n-body + 500 ncv)
    
    at the fiducial parameter values. 

    :return cov:
        big ass covariance matrix of all the triangle configurations in 
        the default ordering. 
    '''
    assert flag == 'reg', "only n-body should be used for covariance"
    assert rsd != 'all', "only one RSD direction should be used otherwise modes will be correlated" 

    fcov = os.path.join(dir_bk, 'quijote_bCov_full%s%s.hdf5' % (_rsd_str(rsd), _flag_str(flag)))
    if os.path.isfile(fcov): 
        if not silent: print('reading ... %s' % os.path.basename(fcov))
        Fcov = h5py.File(fcov, 'r') # read in covariance matrix 
        cov = Fcov['C_bk'][...]
        k1, k2, k3 = Fcov['k1'][...], Fcov['k2'][...], Fcov['k3'][...]
        Nmock = Fcov['Nmock'][...] 
    else: 
        if not silent: print('calculating ... %s' % os.path.basename(fcov))
        quij = Obvs.quijoteBk('fiducial', rsd=rsd, flag=flag, silent=silent) 
        bks = quij['b123'] + quij['b_sn']
        if not silent: print('%i Bk measurements' % bks.shape[0]) 
        cov = np.cov(bks.T) # calculate the covariance
        k1, k2, k3 = quij['k1'], quij['k2'], quij['k3']
        Nmock = quij['b123'].shape[0]

        f = h5py.File(fcov, 'w') # write to hdf5 file 
        f.create_dataset('Nmock', data=Nmock)
        f.create_dataset('C_bk', data=cov) 
        f.create_dataset('k1', data=quij['k1']) 
        f.create_dataset('k2', data=quij['k2']) 
        f.create_dataset('k3', data=quij['k3']) 
        f.close()
    return k1, k2, k3, cov, Nmock

# plot covariance matrices 
def _pkCov(kmax=0.5, rsd=2, flag='reg'): 
    ''' plot the covariance matrix of the quijote fiducial powerspectrum
    '''
    k, C_pk, _ = pkCov(rsd=rsd, flag=flag) 
    klim = (k <= kmax)
    C_pk = C_pk[klim,:][:,klim]
    print('covariance matrix condition number = %.5e' % np.linalg.cond(C_pk)) 

    # plot the covariance matrix 
    fig = plt.figure(figsize=(10,8))
    sub = fig.add_subplot(111)
    cm = sub.pcolormesh(C_pk, norm=LogNorm(vmin=1e3, vmax=1e8))
    cbar = fig.colorbar(cm, ax=sub) 
    cbar.set_label(r'$\widehat{P}_0(k)$ covariance matrix, ${\bf C}_{P}$', 
            fontsize=25, labelpad=10, rotation=90)
    ffig = os.path.join(dir_doc, 
            'quijote_pkCov_kmax%s%s%s.png' % (str(kmax).replace('.', ''), _rsd_str(rsd), _flag_str(flag)))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def _p02kCov(kmax=0.5, rsd=2, flag='reg'): 
    ''' plot the covariance matrix of the quijote fiducial powerspectrum
    '''
    k, C_pk, _ = p02kCov(rsd=rsd, flag=flag) 
    klim = (k <= kmax)
    C_pk = C_pk[klim,:][:,klim]
    print('covariance matrix condition number = %.5e' % np.linalg.cond(C_pk)) 

    # plot the covariance matrix 
    fig = plt.figure(figsize=(10,8))
    sub = fig.add_subplot(111)
    cm = sub.pcolormesh(C_pk, norm=SymLogNorm(1e5, vmin=-1e5, vmax=1e8))
    print(C_pk.min(), C_pk.max()) 
    cbar = fig.colorbar(cm, ax=sub) 
    cbar.set_label(r'$\widehat{P}_{\ell = 0, 2}(k)$ covariance matrix, ${\bf C}_{P}$', 
            fontsize=25, labelpad=10, rotation=90)
    ffig = os.path.join(dir_doc, 
            'quijote_p02kCov_kmax%s%s%s.png' % (str(kmax).replace('.', ''), _rsd_str(rsd), _flag_str(flag)))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def _bkCov(kmax=0.5, rsd=2, flag='reg'): 
    ''' plot the covariance matrix of the quijote fiducial 
    bispectrum. 
    '''
    i_k, j_k, l_k, C_bk, _ = bkCov(rsd=rsd, flag=flag) 
    bklim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) # k limit
    C_bk = C_bk[bklim,:][:,bklim]

    #ijl = UT.ijl_order(i_k[bklim], j_k[bklim], l_k[bklim], typ='GM') # order of triangles 
    #C_bk = C_bk[ijl,:][:,ijl]
    print('covariance matrix condition number = %.5e' % np.linalg.cond(C_bk)) 

    # plot the covariance matrix 
    fig = plt.figure(figsize=(10,8))
    sub = fig.add_subplot(111)
    cm = sub.pcolormesh(C_bk, norm=LogNorm(vmin=1e11, vmax=1e18))
    cbar = fig.colorbar(cm, ax=sub) 
    cbar.set_label(r'$\widehat{B}_0(k_1, k_2, k_3)$ covariance matrix, ${\bf C}_{B}$', 
            fontsize=25, labelpad=10, rotation=90)
    ffig = os.path.join(dir_doc, 
            'quijote_bkCov_kmax%s%s%s.png' % (str(kmax).replace('.', ''), _rsd_str(rsd), _flag_str(flag)))
    #fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') # latex friednly
    return None 


def Pk_comparison(): 
    ''' Comparison of the P(theta_fid), P(theta+), P(theta-) for LT Pm and halo P
    '''
    klin = np.logspace(-5, 1, 400)
    LT_fid = LT._Pm_Mnu(0., klin) 
    LT_fid_fixAs = LT._Pm_Mnu(0., klin, flag='fixAs') 

    fig = plt.figure(figsize=(12,12))
    for _i, theta in enumerate(thetas+['Mmin']): 
        quij_fid = Obvs.quijoteBk('fiducial') # theta_fiducial 
        pk_fid = np.average(quij_fid['p0k1'], axis=0) 
        quij_p = Obvs.quijoteBk(theta+'_p') 
        pk_p = np.average(quij_p['p0k1'], axis=0) 
        if theta != 'Mnu': 
            quij_m = Obvs.quijoteBk(theta+'_m') 
            pk_m = np.average(quij_m['p0k1'], axis=0) 
        else: 
            quij_pp = Obvs.quijoteBk(theta+'_pp') 
            pk_pp = np.average(quij_pp['p0k1'], axis=0) 
            quij_ppp = Obvs.quijoteBk(theta+'_ppp') 
            pk_ppp = np.average(quij_ppp['p0k1'], axis=0) 

        if theta == 'Mmin': 
            pass 
        elif theta == 's8': 
            LT_p = LT._Pm_theta(theta+'_p', klin)
            LT_m = LT._Pm_theta(theta+'_m', klin)
        elif theta != 'Mnu': 
            LT_p = LT._Pm_theta(theta+'_p', klin)
            LT_m = LT._Pm_theta(theta+'_m', klin)
            LT_p_fixAs = LT._Pm_theta(theta+'_p', klin, flag='fixAs') 
            LT_m_fixAs = LT._Pm_theta(theta+'_m', klin, flag='fixAs') 
        else: 
            LT_p = LT._Pm_Mnu(0.1, klin)
            LT_pp = LT._Pm_Mnu(0.2, klin)
            LT_ppp = LT._Pm_Mnu(0.4, klin)
            
            LT_p_fixAs = LT._Pm_Mnu(0.1, klin, flag='fixAs')
            LT_pp_fixAs = LT._Pm_Mnu(0.2, klin, flag='fixAs')
            LT_ppp_fixAs = LT._Pm_Mnu(0.4, klin, flag='fixAs')

            LT_cb_p = LT._Pm_Mnu(0.1, klin, flag='cb')
            LT_cb_pp = LT._Pm_Mnu(0.2, klin, flag='cb')
            LT_cb_ppp = LT._Pm_Mnu(0.4, klin, flag='cb')
    
        i_k = quij_fid['k1'] 
        _, _iuniq = np.unique(i_k, return_index=True)
        iuniq = np.zeros(len(i_k)).astype(bool) 
        iuniq[_iuniq] = True
        
        sub = fig.add_subplot(3,3,_i+1) 
        sub.plot(klin, np.ones(len(klin)), c='k', ls='--')
        if theta == 's8': 
            sub.plot(kf*i_k[iuniq], pk_p[iuniq]/pk_fid[iuniq], c='C0')
            sub.plot(kf*i_k[iuniq], pk_m[iuniq]/pk_fid[iuniq], c='C1')
            sub.plot(klin, LT_p/LT_fid, c='k', ls='-.')
            sub.plot(klin, LT_m/LT_fid, c='k', ls=':')
        elif theta == 'Mmin': 
            sub.plot(kf*i_k[iuniq], pk_p[iuniq]/pk_fid[iuniq], c='C0')
            sub.plot(kf*i_k[iuniq], pk_m[iuniq]/pk_fid[iuniq], c='C1')
        elif theta == 'Mnu': 
            i_tt = thetas.index(theta)
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

            sub.plot(klin, LT_p_fixAs/LT_fid_fixAs, c='C0',   lw=0.5, ls='--', label=r'LT$^+$')
            sub.plot(klin, LT_pp_fixAs/LT_fid_fixAs, c='C1',  lw=0.5, ls='-.', label=r'LT$^{++}$')
            sub.plot(klin, LT_ppp_fixAs/LT_fid_fixAs, c='C2', lw=0.5, ls=':', label=r'LT$^{+++}$')

            sub.legend(loc='best', ncol=2, fontsize=10) 
            sub.set_ylim(0.95, 1.3)  
        else:  
            sub.plot(kf*i_k[iuniq], pk_p[iuniq]/pk_fid[iuniq], c='C0', label=r'$\theta^+$')
            sub.plot(kf*i_k[iuniq], pk_m[iuniq]/pk_fid[iuniq], c='C1', label=r'$\theta^-$')

            sub.plot(klin, LT_p/LT_fid, c='k', ls='-.', label=r'LT$^+$')
            sub.plot(klin, LT_m/LT_fid, c='k', ls=':', label=r'LT$^-$')
            
            sub.plot(klin, LT_p_fixAs/LT_fid_fixAs, c='k', ls='-', lw=0.5)
            sub.plot(klin, LT_m_fixAs/LT_fid_fixAs, c='k', ls='-', lw=0.5)

        #if theta == 's8': 
        #    f_p =(pk_fid[0] + pk_fid[1]) / (pk_p[0] + pk_p[1]) * 0.849**2/0.834**2
        #    f_m =(pk_fid[0] + pk_fid[1]) / (pk_m[0] + pk_m[1]) * 0.819**2/0.834**2
        #    sub.plot(kf*i_k[iuniq], (f_p * pk_p[iuniq])/pk_fid[iuniq], c='C0', ls=':', label='rescaled +')
        #    sub.plot(kf*i_k[iuniq], (f_m * pk_m[iuniq])/pk_fid[iuniq], c='C1', ls=':', label='rescaled -')
        #    sub.legend(loc='best', fontsize=15)

        if _i == 0: sub.legend(loc='best', ncol=2, fontsize=15) 
        sub.set_xlim(1e-2, 1) 
        sub.set_xscale("log") 
        if theta == 'Mmin': 
            sub.text(0.05, 0.05, r'$M_{\rm min}$', ha='left', va='bottom', transform=sub.transAxes, fontsize=25)
        else: 
            i_tt = thetas.index(theta)
            sub.text(0.05, 0.05, theta_lbls[i_tt], ha='left', va='bottom', transform=sub.transAxes, fontsize=25)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel('k [$h$/Mpc]', fontsize=25) 
    bkgd.set_ylabel(r'$P_0/P^{\rm fid}_0$', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ffig = os.path.join(UT.fig_dir(), 'Pk_comparison.thetas.ratio.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


# Derivatives
def dPkdtheta(theta, log=False, rsd='all', flag='reg', dmnu='fin', returnks=False, silent=True):
    ''' read d P(k)/d theta  

    :param theta: 
        string that specifies the parameter to take the 
        derivative by. 
    '''
    dir_dat = os.path.join(UT.doc_dir(), 'dat') 
    fdpk = os.path.join(dir_dat, 'dPdtheta.%s%s%s.dat' % (theta, _rsd_str(rsd), _flag_str(flag))) 
    if not silent: print('--- reading %s ---' % fdpk) 
    i_dpk = 1 
    if theta == 'Mnu': 
        index_dict = {'fin': 1, 'fin0': 3, 'p': 5, 'pp': 7, 'ppp': 9}  
        i_dpk = index_dict[dmnu] 
    if log: i_dpk += 1 

    k, dpdt = np.loadtxt(fdpk, skiprows=1, unpack=True, usecols=[0,i_dpk]) 
    if not returnks: return dpdt 
    else: return k, dpdt 


def dP02kdtheta(theta, log=False, rsd='all', flag='reg', dmnu='fin', returnks=False, silent=True):
    ''' read in derivatives d P(k)/d theta  

    :param theta: 
        string that specifies the parameter to take the 
        derivative by. 
    '''
    dir_dat = os.path.join(UT.doc_dir(), 'dat') 
    fdpk = os.path.join(dir_dat, 'dP02dtheta.%s%s%s.dat' % (theta, _rsd_str(rsd), _flag_str(flag))) 
    if not silent: print('--- reading %s ---' % fdpk) 
    i_dpk = 1 
    if theta == 'Mnu': 
        index_dict = {'fin': 1, 'fin0': 3, 'p': 5, 'pp': 7, 'ppp': 9}  
        i_dpk = index_dict[dmnu] 
    if log: i_dpk += 1 

    k, dpdt = np.loadtxt(fdpk, skiprows=1, unpack=True, usecols=[0,i_dpk]) 
    if not returnks: return dpdt 
    else: return k, dpdt 


def dBkdtheta(theta, log=False, rsd='all', flag=None, dmnu='fin', returnks=False):
    ''' read d B(k)/d theta  

    :param theta: 
        string that specifies the parameter to take the 
        derivative by. 
    '''
    dir_dat = os.path.join(UT.doc_dir(), 'dat') 
    fdbk = os.path.join(dir_dat, 'dBdtheta.%s%s%s.dat' % (theta, _rsd_str(rsd), _flag_str(flag))) 
    i_dbk = 3 
    if theta == 'Mnu': 
        index_dict = {'fin': 3, 'fin0': 5, 'p': 7, 'pp': 9, 'ppp': 11}  
        i_dbk = index_dict[dmnu] 
    if log: i_dbk += 1 

    i_k, j_k, l_k, dbdt = np.loadtxt(fdbk, skiprows=1, unpack=True, usecols=[0,1,2,i_dbk]) 
    if not returnks: return dbdt 
    else: return i_k, j_k, l_k, dbdt 


def P_theta(kmax=0.5, rsd='all', flag='reg'): 
    ''' compare the P0 for theta- and theta+  
    '''
    quij = Obvs.quijotePk('fiducial', rsd=rsd, flag=flag) 
    pk_fid = np.average(quij['p0k'], axis=0) 
    print('%i fiducial Pk measurements' % quij['p0k'].shape[0])  
    k = quij['k'] 
    klim = (k <= kmax) 

    quijote_thetas['Mmin'] = [3.1, 3.3]
    _thetas = thetas + ['Mmin'] 
    _theta_lbls = theta_lbls + [r'$M_{\rm min}$']  

    fig = plt.figure(figsize=(12,12))
    for i_tt, tt, lbl in zip(range(len(_thetas)), _thetas, _theta_lbls): 
        sub = fig.add_subplot(3,3,i_tt+1)
        pks = [] 
        if tt == 'Mnu': 
            for _tt in ['Mnu_p', 'Mnu_pp', 'Mnu_ppp']: 
                _quij = Obvs.quijotePk(_tt, rsd=rsd, flag=flag)
                pks.append(np.average(_quij['p0k'], axis=0)) 
                print('%i %s Pk measurements' % (_quij['p0k'].shape[0], _tt))
        else: 
            for _tt in ['_m', '_p']:
                _quij = Obvs.quijotePk(tt+_tt, rsd=rsd, flag=flag)
                pks.append(np.average(_quij['p0k'], axis=0)) 
                print('%i %s%s Pk measurements' % (_quij['p0k'].shape[0], tt, _tt))

        sub.plot([1e-3, 0.5], [1.0, 1.0], c='k', ls='--')
        for _tt, pk in zip(quijote_thetas[tt], pks): 
            sub.plot(k[klim], pk[klim]/pk_fid[klim], label='%s=%s' % (_theta_lbls[_thetas.index(tt)], str(_tt)))
        sub.legend(loc='upper right', fontsize=15) 
        sub.set_xscale('log') 
        sub.set_xlim(5e-3, 0.5) 
        if tt == 'Mnu': sub.set_ylim(0.95, 1.25) 
        else: sub.set_ylim(0.85, 1.15) 

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel('$k$', fontsize=25) 
    bkgd.set_ylabel(r'$P_h / P_h^{\rm fid}$', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    ffig = os.path.join(dir_fig, 'quijote_P_thetas%s%s.png' % (_rsd_str(rsd), _flag_str(flag))) 
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def P02_theta(kmax=0.5, rsd='all', flag='reg'): 
    ''' compare the P0 and P2 for theta- and theta+  
    '''
    quij = Obvs.quijotePk('fiducial', rsd=rsd, flag=flag) 
    pk_fid = np.average(np.concatenate([quij['p0k'], quij['p2k']], axis=1), axis=0) 
    print('%i fiducial Pk measurements' % quij['p0k'].shape[0])  
    nk = len(quij['k'])
    k = quij['k'] 
    klim = (k <= kmax) 

    quijote_thetas['Mmin'] = [3.1, 3.3]
    _thetas = thetas + ['Mmin'] 
    _theta_lbls = theta_lbls + [r'$M_{\rm min}$']  

    fig = plt.figure(figsize=(12,12))
    for i_tt, tt, lbl in zip(range(len(_thetas)), _thetas, _theta_lbls): 
        sub = fig.add_subplot(3,3,i_tt+1)
        pks = [] 
        if tt == 'Mnu': _tts = ['Mnu_p', 'Mnu_pp', 'Mnu_ppp']
        else: _tts = [tt+'_m', tt+'_p'] 
        for _tt in _tts: 
            _quij = Obvs.quijotePk(_tt, rsd=rsd, flag=flag)
            pks.append(np.average(np.concatenate([_quij['p0k'], _quij['p2k']], axis=1), axis=0)) 
            print('%i %s Pk measurements' % (_quij['p0k'].shape[0], _tt))

        sub.plot([1e-3, 0.5], [1.0, 1.0], c='k', ls='--')
        for _i, _tt, pk in zip(range(len(pks)), quijote_thetas[tt], pks): 
            sub.plot(k[klim], pk[:nk][klim]/pk_fid[:nk][klim], c='C%i' % _i, 
                    label='%s=%s' % (_theta_lbls[_thetas.index(tt)], str(_tt)))
            sub.plot(k[klim], pk[nk:][klim]/pk_fid[nk:][klim], c='C%i' % _i, ls='--')
        sub.legend(loc='upper right', fontsize=15) 
        sub.set_xscale('log') 
        sub.set_xlim(5e-3, 0.5) 
        if tt == 'Mnu': sub.set_ylim(0.95, 1.25) 
        else: sub.set_ylim(0.85, 1.15) 

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel('$k$', fontsize=25) 
    bkgd.set_ylabel(r'$P_h / P_h^{\rm fid}$', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    ffig = os.path.join(dir_fig, 'quijote_P02_thetas%s%s.png' % (_rsd_str(rsd), _flag_str(flag))) 
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def B_theta(kmax=0.5, rsd='all', flag='reg'): 
    ''' compare the B for theta- and theta+  
    '''
    quij = Obvs.quijoteBk('fiducial', rsd=rsd, flag='reg') 
    bk_fid = np.average(quij['b123'], axis=0) 
    print('%i fiducial Bk measurements' % quij['b123'].shape[0])  
    i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3']

    klim = ((i_k * kf <= kmax) & (j_k * kf <= kmax) & (l_k * kf <= kmax)) 
    ijl = UT.ijl_order(i_k[klim], j_k[klim], l_k[klim], typ='GM') # order of triangles 

    quijote_thetas['Mmin'] = [3.1, 3.3]
    _thetas = thetas + ['Mmin'] 
    _theta_lbls = theta_lbls + [r'$M_{\rm min}$']  

    fig = plt.figure(figsize=(20,20))
    for i_tt, tt, lbl in zip(range(len(_thetas)), _thetas, _theta_lbls): 
        sub = fig.add_subplot(len(_thetas),1,i_tt+1)
        bks = [] 
        if tt == 'Mnu': 
            for _tt in ['Mnu_p', 'Mnu_pp', 'Mnu_ppp']: 
                _quij = Obvs.quijoteBk(_tt, rsd=rsd, flag=flag)
                bks.append(np.average(_quij['b123'], axis=0)) 
                print('%i %s Bk measurements' % (_quij['b123'].shape[0], _tt))
        else: 
            for _tt in ['_m', '_p']:
                _quij = Obvs.quijoteBk(tt+_tt, rsd=rsd, flag=flag)
                bks.append(np.average(_quij['b123'], axis=0)) 
                print('%i %s%s Bk measurements' % (_quij['b123'].shape[0], tt, _tt))

        sub.plot([1e-3, 0.5], [1.0, 1.0], c='k', ls='--')
        for _tt, bk in zip(quijote_thetas[tt], bks): 
            sub.plot(range(np.sum(klim)), bk[klim][ijl]/bk_fid[klim][ijl], label='%s=%s' % 
                    (_theta_lbls[_thetas.index(tt)], str(_tt)))
        sub.legend(loc='upper left', ncol=3, fontsize=15) 
        sub.set_xlim(0., np.sum(klim))
        if tt == 'Mnu': sub.set_ylim(0.95, 1.25) 
        else: sub.set_ylim(0.85, 1.15) 

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel('triangle configurations', fontsize=25) 
    bkgd.set_ylabel(r'$B(k_1, k_2, k_3)/B^{\rm fid}$', labelpad=10., fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    ffig = os.path.join(dir_fig, 'quijote_B_thetas%s%s.png' % (_rsd_str(rsd), _flag_str(flag))) 
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def _dPdthetas(log=True, rsd='all', flag='reg', dmnu='fin', silent=True):
    ''' comparison of dlogP/dtheta or dP/dtheta
    '''
    klin = np.logspace(-5, 1, 400)
    _thetas = copy(thetas) + ['Amp', 'Mmin', 'Asn'] # tacked on some nuisance parameters
    _theta_lbls = copy(theta_lbls) + ["$b'$", r'$M_{\rm min}$', r'$A_{\rm SN}$']

    ylims = [(-10., 5.), (-5, 15), (-3., 3.), (-5., 7.), (-2., 2.6), (-0.1, 0.6), (0.95, 1.05), (0.05, 0.16), (1e-5, 1e-3)] 

    fig = plt.figure(figsize=(12,12))
    for i_tt, tt, lbl in zip(range(len(_thetas)), _thetas, _theta_lbls): 
        sub = fig.add_subplot(3,3,i_tt+1)

        k, dpdt = dPkdtheta(tt, log=log, rsd=rsd, flag=flag, dmnu=dmnu, returnks=True, silent=silent)
        plt_pk, = sub.plot(k, dpdt, c='C%i' % i_tt) 
        
        if tt in thetas_pk and log: 
            dpmdt = LT.dPmdtheta(tt, klin, log=True, npoints='quijote') 
            plt_pm, = sub.plot(klin, dpmdt, c='k', lw=1, ls='--')
            dpcbdt = LT.dPmdtheta(tt, klin, log=True, npoints='quijote', flag='cb') 
            plt_pcb, = sub.plot(klin, dpcbdt, c='k', lw=0.5, ls=':')
        
        if i_tt == 0: sub.legend([plt_pk], [r'$P_{\rm h}$'], loc='lower right', fontsize=15) 
        elif tt == 'Mnu': sub.legend([plt_pm, plt_pcb], [r'LT $P_{\rm m}$', r'LT $P_{\rm cb}$'], 
                loc='lower left', fontsize=15) 

        sub.set_xscale('log') 
        sub.set_xlim(5e-3, 0.5) 
        if not log: 
            sub.set_yscale('log') 
            sub.set_ylim(1e2, 1e6) 
        else: 
            sub.set_ylim(ylims[i_tt]) 
        if tt == 'Asn': sub.set_yscale('log') 
        sub.text(0.95, 0.95, lbl, ha='right', va='top', transform=sub.transAxes, fontsize=25)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel('$k$', fontsize=25) 
    if not log: 
        bkgd.set_ylabel(r'${\rm d} P/{\rm d} \theta$', labelpad=10, fontsize=25) 
    else: 
        bkgd.set_ylabel(r'${\rm d}\log P/{\rm d} \theta$', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    ffig = os.path.join(dir_fig, 'quijote_d%sPdthetas%s%s.%s.png' % (['', 'log'][log], _rsd_str(rsd), _flag_str(flag), dmnu))
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def _dP02dthetas(log=True, rsd='all', flag='reg', dmnu='fin', silent=True):
    ''' comparison of the P0 and P2 derivatives (dlogP02/dtheta or dP02/dtheta) 
    '''
    klin = np.logspace(-5, 1, 400)
    _thetas = copy(thetas) + ['Amp', 'Mmin', 'Asn'] # tacked on some nuisance parameters
    _theta_lbls = copy(theta_lbls) + ["$b'$", r'$M_{\rm min}$', r'$A_{\rm SN}$']

    ylims = [(-10., 5.), (-5, 15), (-3., 3.), (-5., 7.), (-2., 2.6), (-0.1, 0.6), (0.95, 1.05), (0.05, 0.16), (1e-5, 1e-3)] 

    fig = plt.figure(figsize=(12,12))
    for i_tt, tt, lbl in zip(range(len(_thetas)), _thetas, _theta_lbls): 
        sub = fig.add_subplot(3,3,i_tt+1)

        k, dpdt = dP02kdtheta(tt, log=log, rsd=rsd, flag=flag, dmnu=dmnu, returnks=True, silent=silent)
        nk = len(k)/2
        plt_pk, = sub.plot(k[:nk], dpdt[:nk], c='C%i' % i_tt) 
        sub.plot(k[nk:], dpdt[nk:], ls='--', c='C%i' % i_tt) 
        
        sub.set_xscale('log') 
        sub.set_xlim(5e-3, 0.5) 
        if not log: 
            sub.set_yscale('log') 
            sub.set_ylim(1e2, 1e6) 
        else: 
            sub.set_ylim(ylims[i_tt]) 
        if tt == 'Asn': sub.set_yscale('log') 
        sub.text(0.95, 0.95, lbl, ha='right', va='top', transform=sub.transAxes, fontsize=25)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel('$k$', fontsize=25) 
    if not log: 
        bkgd.set_ylabel(r'${\rm d} P_{\ell = 0,2}/{\rm d} \theta$', labelpad=10, fontsize=25) 
    else: 
        bkgd.set_ylabel(r'${\rm d}\log P_{\ell = 0, 2}/{\rm d} \theta$', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    ffig = os.path.join(dir_fig, 'quijote_d%sP02dthetas%s%s.%s.png' % (['', 'log'][log], _rsd_str(rsd), _flag_str(flag), dmnu))
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def dBdthetas(kmax=0.5, log=True, rsd='all', flag='reg', dmnu='fin'):
    ''' Compare the derivatives of the bispectrum w.r.t. all the parameters
    
    :param rsd: (default: 'all') 
        rsd kwarg that specifies rsd set up for B(k). 
        If rsd == 'all', include 3 RSD directions. 
        If rsd in [0,1,2] include one of the directions

    :param flag: (default: 'reg')  
        kwarg specifying the flag for B(k). 
        If `flag is None`, include paired-fixed and regular N-body simulation. 
        If `flag == 'ncv'` only include paired-fixed. 
        If `flag == 'reg'` only include regular N-body

    :param dmnu: 
        derivative finite differences setup
    '''
    _thetas = copy(thetas)
    _theta_lbls = copy(theta_lbls) 
    _theta_lims = [(-15., 5.), (-2., 24.), (-4., 0.5), (-5., -0.5), (-6., 0), (-1.5, 2.)]

    i_k, j_k, l_k, _ = dpdt = dBkdtheta('Om', log=log, rsd=rsd, flag=flag, dmnu=dmnu, returnks=True)
    klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) # get k limit
    ijl = UT.ijl_order(i_k[klim], j_k[klim], l_k[klim], typ='GM') # order of triangles 
    print i_k[:10], j_k[:10], l_k[:10]
    print i_k[ijl][:10], j_k[ijl][:10], l_k[ijl][:10]

    fig = plt.figure(figsize=(15,15))
    for i_tt, tt, lbl in zip(range(len(_thetas)), _thetas, _theta_lbls): 
        sub = fig.add_subplot(len(_thetas),1,i_tt+1)
        dpdt = dBkdtheta(tt, log=log, rsd=rsd, flag=flag, dmnu=dmnu)
        plt_bk, = sub.plot(range(np.sum(klim)), dpdt[klim])
        #plt_bk, = sub.plot(range(np.sum(klim)), dpdt[klim][ijl])
        
        sub.set_xlim(0, np.sum(klim)) 
        if not log: sub.set_yscale('log') 
        else: sub.set_ylim(_theta_lims[i_tt]) 
        #if i_tt == 0: sub.legend([plt_bk], [r'$B_{\rm h}$'], loc='upper left', fontsize=15)
        if tt != _thetas[-1]: sub.set_xticklabels([]) 
        sub.text(0.975, 0.925, lbl, ha='right', va='top', transform=sub.transAxes, fontsize=25)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel('triangle configurations', fontsize=25) 
    if not log: bkgd.set_ylabel(r'$|{\rm d}B/{\rm d} \theta|$', labelpad=10, fontsize=25) 
    else: bkgd.set_ylabel(r'${\rm d}\log B/{\rm d} \theta$', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    fig.subplots_adjust(hspace=0.075) 
    ffig = os.path.join(dir_doc, 'quijote_d%sBdthetas%s%s.%s.png' % (['', 'log'][log], _rsd_str(rsd), _flag_str(flag), dmnu))
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') # latex friednly
    return None


def _dBdthetas(kmax=0.5, log=True, rsd='all', flag='reg', dmnu='fin'):
    ''' Compare the derivatives of the bispectrum w.r.t. all the parameters
    
    :param rsd: (default: 'all') 
        rsd kwarg that specifies rsd set up for B(k). 
        If rsd == 'all', include 3 RSD directions. 
        If rsd in [0,1,2] include one of the directions

    :param flag: (default: 'reg')  
        kwarg specifying the flag for B(k). 
        If `flag is None`, include paired-fixed and regular N-body simulation. 
        If `flag == 'ncv'` only include paired-fixed. 
        If `flag == 'reg'` only include regular N-body

    :param dmnu: 
        derivative finite differences setup
    '''
    _thetas = copy(thetas) + ['Amp', 'Mmin', 'Asn', 'Bsn', 'b2', 'g2'] 
    _theta_lbls = copy(theta_lbls) + ["$b'$", r'$M_{\rm min}$', r"$A_{\rm SN}$", r"$B_{\rm SN}$", "$b_2$", "$g_2$"]
    _theta_lims = [(-17., 6.), (-2., 24.), (-5., -0.5), (-5., -0.5), (-6., 0), 
            (-1.5, 2.), (0.95, 1.05), (0.4, -0.4), None, None, None, None] 

    i_k, j_k, l_k, _ = dpdt = dBkdtheta('Om', log=log, rsd=rsd, flag=flag, dmnu=dmnu, returnks=True)
    klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) # get k limit

    fig = plt.figure(figsize=(20, 40))
    for i_tt, tt, lbl in zip(range(len(_thetas)), _thetas, _theta_lbls): 
        sub = fig.add_subplot(len(_thetas),1,i_tt+1)
        dpdt = dBkdtheta(tt, log=log, rsd=rsd, flag=flag, dmnu=dmnu)
        plt_bk, = sub.plot(range(np.sum(klim)), dpdt[klim])
        
        sub.set_xlim(0, np.sum(klim)) 
        if not log: sub.set_yscale('log') 
        else: sub.set_ylim(_theta_lims[i_tt]) 
        if tt in ['Asn', 'Bsn']: sub.set_yscale('log') 
        if i_tt == 0: sub.legend([plt_bk], [r'$B_{\rm h}$'], loc='upper left', fontsize=15)
        sub.text(0.975, 0.95, lbl, ha='right', va='top', transform=sub.transAxes, fontsize=25)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel('triangle configuration', fontsize=25) 
    if not log: bkgd.set_ylabel(r'$|{\rm d}B/{\rm d} \theta|$', labelpad=10, fontsize=25) 
    else: bkgd.set_ylabel(r'${\rm d}\log B/{\rm d} \theta$', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ffig = os.path.join(dir_doc, 'quijote_d%sBdthetas%s%s.%s.png' % (['', 'log'][log], _rsd_str(rsd), _flag_str(flag), dmnu))
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def _dBdthetas_ncv(kmax=0.5, log=True, rsd='all', dmnu='fin'):
    ''' Compare the derivatives of the paired-fixed bispectrum to N-body bispectrum
    
    :param rsd: 
        rsd kwarg that specifies rsd set up for B(k). 
        If rsd == True, include 3 RSD directions. 
        If rsd in [0,1,2] include one of the directions
    :param dmnu: 
        derivative finite differences setup
    '''
    _thetas = copy(thetas)
    _theta_lbls = copy(theta_lbls)
    _theta_lims = [(-17., 6.), (-2., 24.), (-5., -0.5), (-5., -0.5), (-6., 0), (-1.5, 2.)] 

    i_k, j_k, l_k, _ = dBkdtheta('Om', log=log, rsd=rsd, flag='reg', dmnu=dmnu, returnks=True)
    klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) # get k limit

    fig = plt.figure(figsize=(20,20))
    for i_tt, tt, lbl in zip(range(len(_thetas)), _thetas, _theta_lbls): 
        sub = fig.add_subplot(len(_thetas),1,i_tt+1)
        for flag in ['ncv', 'reg']: 
            dpdt = dBkdtheta(tt, log=log, rsd=rsd, flag=flag, dmnu=dmnu)
            sub.plot(range(np.sum(klim)), dpdt[klim], label=flag)
        
        sub.set_xlim(0, np.sum(klim)) 
        if not log: sub.set_yscale('log') 
        else: sub.set_ylim(_theta_lims[i_tt]) 
        if tt in ['Asn', 'Bsn']: sub.set_yscale('log') 
        if i_tt == 0: sub.legend(loc='upper left', fontsize=15)
        sub.text(0.975, 0.95, lbl, ha='right', va='top', transform=sub.transAxes, fontsize=25)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel('triangle configuration', fontsize=25) 
    if not log: bkgd.set_ylabel(r'$|{\rm d}B/{\rm d} \theta|$', labelpad=10, fontsize=25) 
    else: bkgd.set_ylabel(r'${\rm d}\log B/{\rm d} \theta$', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ffig = os.path.join(dir_fig, 'quijote_d%sBdthetas%s.ncv_reg.%s.png' % (['', 'log'][log], _rsd_str(rsd), dmnu))
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def dlogBdMnu(kmax=0.5, flag='reg'):
    ''' Compare the dlogB/dMnu for different methods 
    '''
    fig = plt.figure(figsize=(20, 5))
    sub = fig.add_subplot(111)
    for dmnu in ['fin', 'fin0', 'p', 'pp', 'ppp']: 
        i_k, j_k, l_k, dpdt = dBkdtheta('Mnu', log=True, rsd='all', flag=flag, dmnu=dmnu, returnks=True) 
        klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 
        i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim]

        sub.plot(np.arange(np.sum(klim)), dpdt[klim], label=dmnu)
    sub.legend(loc='upper left', markerscale=10, fontsize=20, ncol=2) 
    sub.set_xlim(0, np.sum(klim)) 
    #sub.set_yscale('log') 
    sub.set_ylim(-1, 2) 
    sub.set_xlabel('triangle configuration', fontsize=25) 
    sub.set_ylabel(r'${\rm d}\log B/{\rm d} M_\nu$', labelpad=10, fontsize=25) 
    ffig = os.path.join(dir_fig, 'quijote_dlogBdMnu.kmax%s%s.png' % (str(kmax), _flag_str(flag))) 
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def dlogBdMnu_IC(kmax=0.5): 
    ''' compare dlogB/dMnu estimate at 0.2eV: one calculate 
    only with ZA sims (Mnu = 0.1, 0.2, 0.4eV), the other 
    calculated with 2LPT + ZA sims (Mnu = 0, 0.1, 0.2, 0.4eV) 
    to see if the difference in IC makes a significant impact.
    '''
    i_k, j_k, l_k, dbdmnu_za = Forecast.quijote_dBkdtheta('Mnu', log=True, rsd='all', flag='reg', dmnu='0.2eV_ZA', silent=False)
    _, _, _, dbdmnu_2lpt = Forecast.quijote_dBkdtheta('Mnu', log=True, rsd='all', flag='reg', dmnu='0.2eV_2LPTZA', silent=False)
    
    klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 

    fig = plt.figure(figsize=(20, 5))
    sub = fig.add_subplot(111)
    sub.plot(np.arange(np.sum(klim)), dbdmnu_za[klim], 
            label=r'Only ZA IC sims ($M_\nu = 0.1, 0.2, 0.4$eV)')
    sub.plot(np.arange(np.sum(klim)), dbdmnu_2lpt[klim], 
            label=r'ZA + 2LPT IC sims ($M_\nu = 0.0, 0.1, 0.2, 0.4$eV)')
    sub.legend(loc='upper left', markerscale=10, fontsize=20, ncol=2) 
    sub.set_xlim(0, np.sum(klim)) 
    #sub.set_yscale('log') 
    sub.set_ylim(-1, 2) 
    sub.set_xlabel('triangle configuration', fontsize=25) 
    sub.set_ylabel(r'${\rm d}\log B/{\rm d} M_\nu$', labelpad=10, fontsize=25) 
    ffig = os.path.join(dir_fig, 'quijote_dlogBdMnu.kmax%s.IC.png' % (str(kmax))) 
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def dlogPBdMnu(rsd='all', flag='reg'):
    ''' Compare the dlogP/dMnu and dlogB/dMnu for different numerical derivative step size 

    :param rsd: 
        rsd kwarg that specifies rsd set up for B(k). 
        If rsd == True, include 3 RSD directions. 
        If rsd in [0,1,2] include one of the directions
    :param flag: 
        kwarg specifying the flag for B(k). 
        If `flag is None`, include paired-fixed and regular N-body simulation. 
        If `flag == 'ncv'` only include paired-fixed. 
        If `flag == 'reg'` only include regular N-body
    '''
    dmnus = ['fin', 'fin0', 'p']
    #dmnus_lbls = ['0.0, 0.1, 0.2, 0.4 eV', '0.0, 0.1, 0.2 eV', '0.0, 0.1eV']
    dmnus_lbls = [r'$M^{\rm fid}_\nu, M^{+}_\nu, M^{++}_\nu, M^{+++}_\nu$ (Eq.12)', 
                  r'$M^{\rm fid}_\nu, M^{+}_\nu, M^{++}_\nu$', 
                  r'$M^{\rm fid}_\nu, M^{+}_\nu$']
    colors = ['C0', 'C1', 'C2']
    kmax=0.5

    quij = Obvs.quijoteBk('Mnu_p', rsd=0, flag='ncv') 
    i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3']
    bklim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) # bk limit 
    ijl = UT.ijl_order(i_k[bklim], j_k[bklim], l_k[bklim], typ='GM') # order of triangles 

    fig = plt.figure(figsize=(20, 5))
    gs = mpl.gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1,3], wspace=0.2) 
    gs0 = mpl.gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0])
    gs1 = mpl.gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[1])
    
    # dlogP/dMnu
    sub0 = plt.subplot(gs0[0,0])
    for i, dmnu, lbl, c in zip(range(len(dmnus)), dmnus, dmnus_lbls, colors): 
        k_pk, dpk = dP02kdtheta('Mnu', log=True, rsd=rsd, flag=flag, dmnu=dmnu, returnks=True) 
        nk = len(k_pk) 
        k_p0k, dp0k = k_pk[:nk/2], dpk[:nk/2]
        k_p2k, dp2k = k_pk[nk/2:], dpk[nk/2:]
        p0klim = (k_p0k <= 0.5) 
        p2klim = (k_p2k <= 0.5) 
        sub0.plot(k_p0k[p0klim], dp0k[p0klim], c=c, label='$\ell = 0$') 
        sub0.plot(k_p2k[p2klim], 0.5+dp2k[p2klim], c=c, ls=':', label='$\ell=2$') 
        if dmnu == 'fin': _dpk = dpk 
        else: 
            ratio = (dpk/_dpk)
            print('(%s)/(fin) = %f' % (dmnu, np.median(ratio[np.isfinite(ratio)]))) 
        if i == 0: sub0.legend(loc='lower left', handletextpad=0.2, fontsize=15) 
    sub0.set_xlabel('$k$', fontsize=25) 
    sub0.set_xscale('log') 
    sub0.set_xlim(5e-3, 1.0) 
    sub0.set_xticks([0.02, 0.1, 0.5])
    sub0.set_xticklabels([0.02, 0.1, 0.5])
    sub0.set_ylabel(r'${\rm d}\log P_\ell(k)/{\rm d} M_\nu + \ell/4$', fontsize=25) 
    sub0.set_ylim(0., 1.) 
    # dlogB/dMnu
    sub1 = plt.subplot(gs1[0,0])
    splts = [] 
    for dmnu, lbl in zip(dmnus, dmnus_lbls): 
        dbk = dBkdtheta('Mnu', log=True, rsd=rsd, flag=flag, dmnu=dmnu) 
        splt, = sub1.plot(range(np.sum(bklim)), dbk[bklim][ijl]) 
        splts.append(splt) 
        if dmnu == 'fin': _dbk = dbk 
        else: 
            ratio = (dbk/_dbk)
            print('(%s)/(fin) = %f' % (dmnu, np.median(ratio[np.isfinite(ratio)]))) 
    sub1.legend(splts[::-1], dmnus_lbls[::-1], loc='lower left', fontsize=18) 
    sub1.set_xlabel('triangle configurations', fontsize=25) 
    sub1.set_xlim(0, np.sum(bklim)) 
    sub1.set_ylabel(r'${\rm d}\log B(k_1, k_2, k_3)/{\rm d} M_\nu$', labelpad=-5, fontsize=25) 
    sub1.set_ylim(-0.65, 1.5) 
    sub1.set_yticks([-0.5, 0., 0.5, 1., 1.5]) 
    ffig = os.path.join(dir_doc, 'quijote_dPBdMnu_dmnu%s%s.png' % (_rsd_str(rsd), _flag_str(flag)))
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') # latex friednly
    return None

# Fisher Matrix
def FisherMatrix(obs, kmax=0.5, rsd=True, flag=None, dmnu='fin', theta_nuis=None): 
    ''' calculate fisher matrix for parameters ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu'] 
    and specified nuisance parameters
    
    :param obs: 
        observable ('pk' or 'bk') 
    :param kmax: (default: 0.5) 
        kmax of the analysis 
    :param rsd: (default: True) 
        rsd kwarg that specifies rsd set up for B(k). 
        If rsd == True, include 3 RSD directions. 
        If rsd in [0,1,2] include one of the directions
    :param flag: (default: None) 
        kwarg specifying the flag for B(k). 
        If `flag is None`, include paired-fixed and regular N-body simulation. 
        If `flag == 'ncv'` only include paired-fixed. 
        If `flag == 'reg'` only include regular N-body
    :param dmnu: (default: 'fin') 
        derivative finite differences setup
    :param theta_nuis: (default: None) 
        list of nuisance parameters to include in the forecast. 
    '''
    # calculate covariance matrix (with shotnoise; this is the correct one) 
    # *** rsd and flag kwargs are ignored for the covariace matirx ***
    if obs == 'pk': 
        k, _Cfid, nmock = pkCov(rsd=2, flag='reg') # only N-body and 1 RSD direction 
        klim = (k <= kmax) # determine k limit 
        C_fid = _Cfid[klim,:][:,klim]
    elif obs == 'p02k':  # monopole and quadrupole 
        k, _Cfid, nmock = p02kCov(rsd=2, flag='reg') # only N-body and 1 RSD direction 
        klim = (k <= kmax) # determine k limit 
        C_fid = _Cfid[klim,:][:,klim]
    elif obs == 'bk':
        i_k, j_k, l_k, _Cfid, nmock = bkCov(rsd=2, flag='reg') # only N-body for covariance
        klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) # k limit 
        C_fid = _Cfid[:,klim][klim,:]
    else: 
        raise NotImplementedError

    ndata = np.sum(klim) 
    f_hartlap = float(nmock - ndata - 2)/float(nmock - 1) 
    C_inv = f_hartlap * np.linalg.inv(C_fid) # invert the covariance 
    
    _thetas = copy(thetas) 
    if theta_nuis is not None: _thetas += theta_nuis 
    
    # calculate the derivatives along all the thetas 
    dobs_dt = [] 
    for par in _thetas: 
        if obs == 'pk': 
            dobs_dti = dPkdtheta(par, rsd=rsd, flag=flag, dmnu=dmnu)
            dobs_dt.append(dobs_dti[klim])
        elif obs == 'p02k': 
            dobs_dti = dP02kdtheta(par, rsd=rsd, flag=flag, dmnu=dmnu)
            dobs_dt.append(dobs_dti[klim])
        elif obs == 'bk': 
            # rsd and flag kwargs are passed to the derivatives
            dobs_dti = dBkdtheta(par, rsd=rsd, flag=flag, dmnu=dmnu)
            dobs_dt.append(dobs_dti[klim])
    Fij = Forecast.Fij(dobs_dt, C_inv) 
    #assert ndata > Fij.shape[0] 
    return Fij 

##################################################################
# Fisher forecasts 
##################################################################
def forecast(obs, kmax=0.5, rsd=True, flag=None, dmnu='fin', theta_nuis=None, planck=False):
    ''' fisher forecast for quijote observables where we include theta_nuis as free parameters. 
    
    :param obs: 
        observable ('pk' or 'bk') 
    :param kmax: (default: 0.5) 
        kmax of the analysis 
    :param rsd: (default: True) 
        rsd kwarg that specifies rsd set up for B(k). 
        If rsd == True, include 3 RSD directions. 
        If rsd in [0,1,2] include one of the directions
    :param flag: (default: None) 
        kwarg specifying the flag for B(k). 
        If `flag is None`, include paired-fixed and regular N-body simulation. 
        If `flag == 'ncv'` only include paired-fixed. 
        If `flag == 'reg'` only include regular N-body
    :param dmnu: (default: 'fin') 
        derivative finite differences setup
    :param theta_nuis: (default: None) 
        list of nuisance parameters to include in the forecast. 
    :param planck: (default: False)
        If True add Planck prior 
    '''
    # fisher matrix (Fij)
    _Fij    = FisherMatrix(obs, kmax=kmax, rsd=rsd, flag=flag, dmnu=dmnu, theta_nuis=None) # no nuisance param. 
    Fij     = FisherMatrix(obs, kmax=kmax, rsd=rsd, flag=flag, dmnu=dmnu, theta_nuis=theta_nuis) # marg. over nuisance param. 
    cond = np.linalg.cond(Fij)
    if cond > 1e16: print('Fij is ill-conditioned %.5e' % cond)
    _Finv   = np.linalg.inv(_Fij) # invert fisher matrix 
    Finv    = np.linalg.inv(Fij) # invert fisher matrix 
    
    if planck: # add planck prior 
        _Finv_noplanck = Finv.copy() 
        _Fij_planck = np.load(os.path.join(UT.dat_dir(), 'Planck_2018_s8.npy')) # read in planck prior fisher (order is Om, Ob, h, ns, s8 and Mnu) 
        Fij_planck = Fij.copy() 
        Fij_planck[:6,:6] += _Fij_planck
        Finv = np.linalg.inv(Fij_planck)

    i_Mnu = thetas.index('Mnu')
    if not planck: 
        print('--- i = Mnu ---')
        print('n nuis. Fii=%f, sigma_i = %f' % (_Fij[i_Mnu,i_Mnu], np.sqrt(_Finv[i_Mnu,i_Mnu])))
        print("y nuis. Fii=%f, sigma_i = %f" % (Fij[i_Mnu,i_Mnu], np.sqrt(Finv[i_Mnu,i_Mnu])))
        print('--- thetas ---')
        print('n nuis. sigmas %s' % ', '.join(['%.2e' % sii for sii in np.sqrt(np.diag(_Finv))]))
        print('y nuis. sigmas %s' % ', '.join(['%.2e' % sii for sii in np.sqrt(np.diag(Finv))]))
    else: 
        print('n nuis. Fii=%f, sigma_i = %f' % (_Fij[i_Mnu,i_Mnu], np.sqrt(_Finv[i_Mnu,i_Mnu])))
        print("y nuis. Fii=%f, sigma_i = %f" % (Fij[i_Mnu,i_Mnu], np.sqrt(_Finv_noplanck[i_Mnu,i_Mnu])))
        print("w/ Planck2018")
        print("y nuis. Fii=%f, sigma_i = %f" % (Fij_planck[i_Mnu,i_Mnu], np.sqrt(Finv[i_Mnu,i_Mnu])))
        print('--- thetas ---')
        print('n nuis. sigmas %s' % ', '.join(['%.2e' % sii for sii in np.sqrt(np.diag(_Finv))]))
        print('y nuis. sigmas %s' % ', '.join(['%.2e' % sii for sii in np.sqrt(np.diag(_Finv_noplanck))]))
        print("w/ Planck2018")
        print('y nuis. sigmas %s' % ', '.join(['%.2e' % sii for sii in np.sqrt(np.diag(Finv))]))
    
    if obs in ['pk', 'p02k']: _thetas = copy(thetas_pk) + theta_nuis 
    else: _thetas = copy(thetas) + theta_nuis 
    _theta_lbls = copy(theta_lbls) + [theta_nuis_lbls[tt] for tt in theta_nuis]

    _theta_fid = theta_fid.copy() # fiducial thetas
    for tt in theta_nuis: _theta_fid[tt] = theta_nuis_fids[tt]
    # x, y ranges
    _theta_lims = [(0.25, 0.385), (0.02, 0.08), (0.3, 1.1), (0.6, 1.3), (0.8, 0.88), (-0.4, 0.4)]
    if obs == 'pk' and kmax <= 0.2:
        _theta_lims = [(0.1, 0.5), (0.0, 0.15), (0.0, 1.6), (0., 2.), (0.5, 1.3), (0, 2.)]
    _theta_lims += [theta_nuis_lims[tt] for tt in theta_nuis]

    #print(_thetas) 
    #print('%s' % '$ & $\pm '.join(['%.5f' % np.sqrt(Finv[i_Mnu,i_Mnu])]+['%.5f' % sii for sii in np.sqrt(np.diag(Finv))]))
    
    fig = plt.figure(figsize=(17, 15))
    for i in range(len(_thetas)+1): 
        for j in range(i+1, len(_thetas)): 
            # sub inverse fisher matrix 
            Finv_sub = np.array([[Finv[i,i], Finv[i,j]], [Finv[j,i], Finv[j,j]]]) 
            # plot the ellipse
            sub = fig.add_subplot(len(_thetas)-1, len(_thetas)-1, (len(_thetas)-1) * (j-1) + i + 1) 
            Forecast.plotEllipse(Finv_sub, sub, 
                    theta_fid_ij=[_theta_fid[_thetas[i]], _theta_fid[_thetas[j]]], color='C0')
            sub.set_xlim(_theta_lims[i])
            sub.set_ylim(_theta_lims[j])
            if i == 0:   
                sub.set_ylabel(_theta_lbls[j], fontsize=30) 
            else: 
                sub.set_yticks([])
                sub.set_yticklabels([])
            
            if j == len(_thetas)-1: 
                sub.set_xlabel(_theta_lbls[i], labelpad=10, fontsize=30) 
            else: 
                sub.set_xticks([])
                sub.set_xticklabels([]) 
    fig.subplots_adjust(wspace=0.05, hspace=0.05) 
    
    nuis_str = ''
    if 'Amp' in theta_nuis: nuis_str += 'b'
    if 'Mmin' in theta_nuis: nuis_str += 'Mmin'
    if ('Asn' in theta_nuis) or ('Bsn' in theta_nuis): nuis_str += 'SN'
    if 'b2' in theta_nuis: nuis_str += 'b2'
    if 'g2' in theta_nuis: nuis_str += 'g2'

    planck_str = ''
    if planck: planck_str = '.planck'

    ffig = os.path.join(dir_doc, 
            'quijote.%sFisher.%s.dmnu_%s.kmax%.2f%s%s%s.png' % 
            (obs, nuis_str, dmnu, kmax, _rsd_str(rsd), _flag_str(flag), planck_str))
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') # latex version 
    return None 


def forecast_kmax_table(rsd='all', flag='reg', theta_nuis=None, dmnu='fin'): 
    '''print out the constraints so that it can be directly copied and pasted to 
    the latex file 
    
    :param rsd: (default: True) 
        rsd kwarg that specifies rsd set up for B(k). 
        If rsd == True, include 3 RSD directions. 
        If rsd in [0,1,2] include one of the directions
    :param flag: (default: None) 
        kwarg specifying the flag for B(k). 
        If `flag is None`, include paired-fixed and regular N-body simulation. 
        If `flag == 'ncv'` only include paired-fixed. 
        If `flag == 'reg'` only include regular N-body
    :param theta_nuis: (default: None) 
        list of nuisance parameters to include in the forecast. 
    :param dmnu: (default: 'fin') 
        derivative finite differences setup
    '''
    if theta_nuis is None: theta_nuis = [] 
    for obs in ['pk', 'bk']:  
        for kmax in [0.2, 0.3, 0.4, 0.5]: 
            Fij = FisherMatrix(obs, kmax=kmax, rsd=rsd, flag=flag, dmnu=dmnu, theta_nuis=theta_nuis) # marg. over nuisance param. 
            cond = np.linalg.cond(Fij)
            if cond > 1e16: print('Fij is ill-conditioned %.5e' % cond)
            Finv = np.linalg.inv(Fij) # invert fisher matrix 
    
            if obs == 'pk': 
                i_thetas = [thetas_pk.index('Mnu'), thetas_pk.index('Om'), thetas_pk.index('Ob'), thetas_pk.index('h'), 
                        thetas_pk.index('ns'), thetas_pk.index('s8')]+range(6,6+len(theta_nuis))
            elif obs == 'bk': 
                i_thetas = [thetas.index('Mnu'), thetas.index('Om'), thetas.index('Ob2'), thetas.index('h'), 
                        thetas.index('ns'), thetas.index('s8')]+range(6,6+len(theta_nuis))
            print(r'& %.1f & $\pm%s$\\' % (kmax, '$ & $\pm'.join(['%.3f' % np.sqrt(Finv[ii,ii]) for ii in i_thetas])))
    return None 

# P, B forecast comparison
def pbForecast(kmax=0.5, rsd=True, flag=None, theta_nuis=None, dmnu='fin', planck=False):
    ''' fisher forecast for quijote P and B where theta_nuis are added as free parameters. 
    
    :param kmax: (default: 0.5) 
        kmax of the analysis 
    :param rsd: (default: True) 
        rsd kwarg that specifies rsd set up for B(k). 
        If rsd == True, include 3 RSD directions. 
        If rsd in [0,1,2] include one of the directions
    :param flag: (default: None) 
        kwarg specifying the flag for B(k). 
        If `flag is None`, include paired-fixed and regular N-body simulation. 
        If `flag == 'ncv'` only include paired-fixed. 
        If `flag == 'reg'` only include regular N-body
    :param dmnu: (default: 'fin') 
        derivative finite differences setup
    :param theta_nuis: (default: None) 
        list of nuisance parameters to include in the forecast. 
    :param planck: (default: False)
        If True add Planck prior 
    '''
    pk_theta_nuis = list(np.array(theta_nuis).copy())
    bk_theta_nuis = list(np.array(theta_nuis).copy())
    if 'Bsn' in theta_nuis: pk_theta_nuis.remove('Bsn') 
    if 'b2' in theta_nuis: pk_theta_nuis.remove('b2') 
    if 'g2' in theta_nuis: pk_theta_nuis.remove('g2') 

    # fisher matrix (Fij)
    pkFij   = FisherMatrix('pk', kmax=kmax, rsd=rsd, flag=flag, dmnu=dmnu, theta_nuis=pk_theta_nuis)  
    bkFij   = FisherMatrix('bk', kmax=kmax, rsd=rsd, flag=flag, dmnu=dmnu, theta_nuis=bk_theta_nuis)  
    #pbkFij  = quijote_FisherMatrix('pbk', kmax=kmax, rsd=rsd, dmnu=dmnu, theta_nuis=theta_nuis)  

    if planck: # add planck prior 
        _Fij_planck = np.load(os.path.join(UT.dat_dir(), 'Planck_2018_s8.npy')) # read in planck prior fisher (order is Om, Ob, h, ns, s8 and Mnu) 
        print('Planck Fii', np.diag(_Fij_planck)) 
        pkFij[:6,:6] += _Fij_planck
        bkFij[:6,:6] += _Fij_planck
        #pbkFij[:6,:6] += _Fij_planck
    pkFinv  = np.linalg.inv(pkFij) 
    bkFinv  = np.linalg.inv(bkFij) 
    #pbkFinv = np.linalg.inv(pbkFij) 

    i_Mnu = thetas.index('Mnu')
    print('--- i = Mnu ---') 
    print('P Fii=%f, sigma_i = %f' % (pkFij[i_Mnu, i_Mnu], np.sqrt(pkFinv[i_Mnu,i_Mnu])))
    print("B Fii=%f, sigma_i = %f" % (bkFij[i_Mnu, i_Mnu], np.sqrt(bkFinv[i_Mnu,i_Mnu])))
    #print("P+B Fii=%f, sigma_i = %f" % (pbkFij[i_Mnu, i_Mnu], np.sqrt(pbkFinv[i_Mnu,i_Mnu])))
    print('--- thetas ---') 
    print('P sigmas %s' % ', '.join(['%.4f' % sii for sii in np.sqrt(np.diag(pkFinv))]))
    print('B sigmas %s' % ', '.join(['%.4f' % sii for sii in np.sqrt(np.diag(bkFinv))]))
    print('improvement %s' % ', '.join(['%.1f' % (pii/bii) for pii, bii in zip(np.sqrt(np.diag(pkFinv)), np.sqrt(np.diag(bkFinv)))]))
    #print('P+B sigmas %s' % ', '.join(['%.2e' % sii for sii in np.sqrt(np.diag(pbkFinv))]))
    print(np.diag(pkFij))
    print(np.diag(bkFij))

    _thetas = copy(thetas) + theta_nuis 
    _theta_lbls = copy(theta_lbls) + [theta_nuis_lbls[tt] for tt in theta_nuis]
    _theta_fid = theta_fid.copy() # fiducial thetas
    for tt in theta_nuis: _theta_fid[tt] = theta_nuis_fids[tt]
    _theta_lims = [(0.25, 0.385), (0.02, 0.08), (0.3, 1.1), (0.6, 1.3), (0.77, 0.9), (-0.4, 0.4)]
    if kmax < 0.2: _theta_lims = [(0.1, 0.5), (0.0, 0.15), (0.0, 1.6), (0., 2.), (0.5, 1.3), (0, 2.)]
    if planck: 
        _theta_lims = [(_theta_fid[tt] - dtt, _theta_fid[tt] + dtt) for tt, dtt in zip(thetas, [0.03, 0.003, 0.02, 0.01, 0.05, 0.2])] 
    _theta_lims += [theta_nuis_lims[tt] for tt in theta_nuis]

    fig = plt.figure(figsize=(17, 15))
    gs = mpl.gridspec.GridSpec(2, 1, figure=fig, height_ratios=[5,2], hspace=0.15) 
    gs0 = mpl.gridspec.GridSpecFromSubplotSpec(5, 8, subplot_spec=gs[0])
    gs1 = mpl.gridspec.GridSpecFromSubplotSpec(2, 8, subplot_spec=gs[1])
    for i in range(ntheta+1): 
        for j in range(i+1, ntheta+2): 
            theta_fid_i, theta_fid_j = _theta_fid[_thetas[i]], _theta_fid[_thetas[j]] # fiducial parameter 
            
            if j < ntheta: sub = plt.subplot(gs0[j-1,i])
            else: sub = plt.subplot(gs1[j-6,i])

            for _i, Finv in enumerate([pkFinv, bkFinv]):#, pbkFinv]):
                Finv_sub = np.array([[Finv[i,i], Finv[i,j]], [Finv[j,i], Finv[j,j]]]) # sub inverse fisher matrix 
                Forecast.plotEllipse(Finv_sub, sub, theta_fid_ij=[theta_fid_i, theta_fid_j], color='C%i' % _i)

            sub.set_xlim(_theta_lims[i])
            sub.set_ylim(_theta_lims[j])
            if i == 0:   
                sub.set_ylabel(_theta_lbls[j], labelpad=5, fontsize=28) 
                sub.get_yaxis().set_label_coords(-0.35,0.5)
            else: 
                sub.set_yticks([])
                sub.set_yticklabels([])
            
            if j == ntheta+1: 
                sub.set_xlabel(_theta_lbls[i], labelpad=7, fontsize=28) 
            elif j == ntheta-1: 
                sub.set_xlabel(_theta_lbls[i], fontsize=26) 
            else: 
                sub.set_xticks([])
                sub.set_xticklabels([]) 

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.fill_between([],[],[], color='C0', label=r'$P^{\rm halo}_0(k)$') 
    bkgd.fill_between([],[],[], color='C1', label=r'$B^{\rm halo}(k_1, k_2, k_3)$') 
    bkgd.legend(loc='upper right', bbox_to_anchor=(0.875, 0.775), fontsize=25)
    bkgd.text(0.8, 0.61, r'$k_{\rm max} = %.1f$; $z=0.$' % kmax, ha='right', va='bottom', 
            transform=bkgd.transAxes, fontsize=25)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.05, hspace=0.05) 
    
    if theta_nuis is None: nuis_str = ''
    else: nuis_str = '.'
    if 'Amp' in theta_nuis: nuis_str += 'b'
    if 'Mmin' in theta_nuis: nuis_str += 'Mmin'
    if ('Asn' in theta_nuis) or ('Bsn' in theta_nuis): nuis_str += 'SN'
    if 'b2' in theta_nuis: nuis_str += 'b2'
    if 'g2' in theta_nuis: nuis_str += 'g2'

    planck_str = ''
    if planck: planck_str = '.planck'

    ffig = os.path.join(dir_doc,
            'quijote.pbkFisher%s.dmnu_%s.kmax%.2f%s%s%s.png' % 
            (nuis_str, dmnu, kmax, _rsd_str(rsd), _flag_str(flag), planck_str))
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') # latex 
    return None 


def p02bForecast(kmax=0.5, rsd=True, flag=None, theta_nuis=None, dmnu='fin', planck=False):
    ''' fisher forecast for quijote P0+P2 and B where theta_nuis are added as free parameters. 
    
    :param kmax: (default: 0.5) 
        kmax of the analysis 
    :param rsd: (default: True) 
        rsd kwarg that specifies rsd set up for B(k). 
        If rsd == True, include 3 RSD directions. 
        If rsd in [0,1,2] include one of the directions
    :param flag: (default: None) 
        kwarg specifying the flag for B(k). 
        If `flag is None`, include paired-fixed and regular N-body simulation. 
        If `flag == 'ncv'` only include paired-fixed. 
        If `flag == 'reg'` only include regular N-body
    :param dmnu: (default: 'fin') 
        derivative finite differences setup
    :param theta_nuis: (default: None) 
        list of nuisance parameters to include in the forecast. 
    :param planck: (default: False)
        If True add Planck prior 
    '''
    pk_theta_nuis = list(np.array(theta_nuis).copy())
    bk_theta_nuis = list(np.array(theta_nuis).copy())
    if 'Bsn' in theta_nuis: pk_theta_nuis.remove('Bsn') 
    if 'b2' in theta_nuis: pk_theta_nuis.remove('b2') 
    if 'g2' in theta_nuis: pk_theta_nuis.remove('g2') 

    # fisher matrix (Fij)
    pkFij   = FisherMatrix('p02k', kmax=kmax, rsd=rsd, flag=flag, dmnu=dmnu, theta_nuis=pk_theta_nuis)  
    bkFij   = FisherMatrix('bk', kmax=kmax, rsd=rsd, flag=flag, dmnu=dmnu, theta_nuis=bk_theta_nuis)  
    #pbkFij  = quijote_FisherMatrix('pbk', kmax=kmax, rsd=rsd, dmnu=dmnu, theta_nuis=theta_nuis)  

    if planck: # add planck prior 
        _Fij_planck = np.load(os.path.join(UT.dat_dir(), 'Planck_2018_s8.npy')) # read in planck prior fisher (order is Om, Ob, h, ns, s8 and Mnu) 
        print('Planck Fii', np.diag(_Fij_planck)) 
        pkFij[:6,:6] += _Fij_planck
        bkFij[:6,:6] += _Fij_planck
        #pbkFij[:6,:6] += _Fij_planck
    pkFinv  = np.linalg.inv(pkFij) 
    bkFinv  = np.linalg.inv(bkFij) 
    #pbkFinv = np.linalg.inv(pbkFij) 

    i_Mnu = thetas.index('Mnu')
    print('--- i = Mnu ---') 
    print('P Fii=%f, sigma_i = %f' % (pkFij[i_Mnu, i_Mnu], np.sqrt(pkFinv[i_Mnu,i_Mnu])))
    print("B Fii=%f, sigma_i = %f" % (bkFij[i_Mnu, i_Mnu], np.sqrt(bkFinv[i_Mnu,i_Mnu])))
    #print("P+B Fii=%f, sigma_i = %f" % (pbkFij[i_Mnu, i_Mnu], np.sqrt(pbkFinv[i_Mnu,i_Mnu])))
    print('--- thetas ---') 
    print('P sigmas %s' % ', '.join(['%.4f' % sii for sii in np.sqrt(np.diag(pkFinv))]))
    print('B sigmas %s' % ', '.join(['%.4f' % sii for sii in np.sqrt(np.diag(bkFinv))]))
    print('improvement %s' % ', '.join(['%.1f' % (pii/bii) for pii, bii in zip(np.sqrt(np.diag(pkFinv)), np.sqrt(np.diag(bkFinv)))]))
    #print('P+B sigmas %s' % ', '.join(['%.2e' % sii for sii in np.sqrt(np.diag(pbkFinv))]))
    print(np.diag(pkFij))
    print(np.diag(bkFij))

    _thetas = copy(thetas) + theta_nuis 
    _theta_lbls = copy(theta_lbls) + [theta_nuis_lbls[tt] for tt in theta_nuis]
    _theta_fid = theta_fid.copy() # fiducial thetas
    for tt in theta_nuis: _theta_fid[tt] = theta_nuis_fids[tt]
    # theta_fid = { 's8': 0.834} # fiducial theta 
    _theta_lims = [(0.28, 0.355), (0.028, 0.07), (0.4422, 0.9), (0.7248, 1.2), (0.788, 0.88), (-0.25, 0.25)]
    if kmax < 0.2: _theta_lims = [(0.1, 0.5), (0.0, 0.15), (0.0, 1.6), (0., 2.), (0.5, 1.3), (0, 2.)]
    if planck: 
        _theta_lims = [(_theta_fid[tt] - dtt, _theta_fid[tt] + dtt) for tt, dtt in zip(thetas, [0.03, 0.003, 0.02, 0.01, 0.05, 0.2])] 
    _theta_lims += [theta_nuis_lims[tt] for tt in theta_nuis]

    fig = plt.figure(figsize=(17, 15))
    gs = mpl.gridspec.GridSpec(2, 1, figure=fig, height_ratios=[5,2], hspace=0.15) 
    gs0 = mpl.gridspec.GridSpecFromSubplotSpec(5, 8, subplot_spec=gs[0])
    gs1 = mpl.gridspec.GridSpecFromSubplotSpec(2, 8, subplot_spec=gs[1])
    for i in range(ntheta+1): 
        for j in range(i+1, ntheta+2): 
            theta_fid_i, theta_fid_j = _theta_fid[_thetas[i]], _theta_fid[_thetas[j]] # fiducial parameter 
            
            if j < ntheta: sub = plt.subplot(gs0[j-1,i])
            else: sub = plt.subplot(gs1[j-6,i])

            for _i, Finv in enumerate([pkFinv, bkFinv]):#, pbkFinv]):
                Finv_sub = np.array([[Finv[i,i], Finv[i,j]], [Finv[j,i], Finv[j,j]]]) # sub inverse fisher matrix 
                Forecast.plotEllipse(Finv_sub, sub, theta_fid_ij=[theta_fid_i, theta_fid_j], color='C%i' % _i)

            sub.set_xlim(_theta_lims[i])
            sub.set_ylim(_theta_lims[j])
            if i == 0:   
                sub.set_ylabel(_theta_lbls[j], labelpad=5, fontsize=28) 
                sub.get_yaxis().set_label_coords(-0.35,0.5)
            else: 
                sub.set_yticks([])
                sub.set_yticklabels([])
            
            if j == ntheta+1: 
                sub.set_xlabel(_theta_lbls[i], labelpad=7, fontsize=28) 
            elif j == ntheta-1: 
                sub.set_xlabel(_theta_lbls[i], fontsize=26) 
            else: 
                sub.set_xticks([])
                sub.set_xticklabels([]) 

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.fill_between([],[],[], color='C0', label=r'$P^{\rm halo}_0(k) + P^{\rm halo}_2(k)$') 
    bkgd.fill_between([],[],[], color='C1', label=r'$B^{\rm halo}(k_1, k_2, k_3)$') 
    bkgd.legend(loc='upper right', bbox_to_anchor=(0.875, 0.775), fontsize=25)
    bkgd.text(0.8, 0.61, r'$k_{\rm max} = %.1f$; $z=0.$' % kmax, ha='right', va='bottom', 
            transform=bkgd.transAxes, fontsize=25)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.05, hspace=0.05) 
    
    if theta_nuis is None: nuis_str = ''
    else: nuis_str = '.'
    if 'Amp' in theta_nuis: nuis_str += 'b'
    if 'Mmin' in theta_nuis: nuis_str += 'Mmin'
    if ('Asn' in theta_nuis) or ('Bsn' in theta_nuis): nuis_str += 'SN'
    if 'b2' in theta_nuis: nuis_str += 'b2'
    if 'g2' in theta_nuis: nuis_str += 'g2'

    planck_str = ''
    if planck: planck_str = '.planck'

    ffig = os.path.join(dir_doc,
            'quijote.p02bkFisher%s.dmnu_%s.kmax%.2f%s%s%s.png' % 
            (nuis_str, dmnu, kmax, _rsd_str(rsd), _flag_str(flag), planck_str))
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') # latex 
    return None 


def Fii_kmax(rsd=True, flag=None, dmnu='fin'):
    ''' 1/sqrt(Fii) as a function of kmax 
    
    :param rsd: (default: True) 
        rsd kwarg that specifies rsd set up for B(k). 
        If rsd == True, include 3 RSD directions. 
        If rsd in [0,1,2] include one of the directions
    :param flag: (default: None) 
        kwarg specifying the flag for B(k). 
        If `flag is None`, include paired-fixed and regular N-body simulation. 
        If `flag == 'ncv'` only include paired-fixed. 
        If `flag == 'reg'` only include regular N-body
    :param dmnu: (default: 'fin') 
        derivative finite differences setup
    '''
    kmaxs = kf * 3 * np.arange(1, 28) 
    
    # read in fisher matrix (Fij)
    Fii_pk, Fii_bk, Fii_pm, Fii_pcb = [], [], [], [] 
    for i_k, kmax in enumerate(kmaxs): 
        pmFij   = LT.Fij_Pm(np.logspace(-5, 2, 500), kmax=kmax, npoints=5, flag='cb') # linear theory Pm 
        pkFij   = FisherMatrix('pk', kmax=kmax, rsd=rsd, flag=flag, dmnu=dmnu, theta_nuis=None) 
        bkFij   = FisherMatrix('bk', kmax=kmax, rsd=rsd, flag=flag, dmnu=dmnu, theta_nuis=None) 

        Fii_pm.append(1./np.sqrt(np.diag(pmFij)))
        Fii_pk.append(1./np.sqrt(np.diag(pkFij)))
        Fii_bk.append(1./np.sqrt(np.diag(bkFij)))
        print('kmax=%.3f ---' %  kmax) 
        print('pm: %s' % ', '.join(['%.2e' % fii for fii in np.diag(pmFij)])) 
        print('pk: %s' % ', '.join(['%.2e' % fii for fii in np.diag(pkFij)])) 
        print('bk: %s' % ', '.join(['%.2e' % fii for fii in np.diag(bkFij)])) 
    Fii_pm = np.array(Fii_pm)
    Fii_pk = np.array(Fii_pk)
    Fii_bk = np.array(Fii_bk)
    sigma_theta_lims = [(1e-4, 1.), (1e-4, 1.), (1e-3, 2), (1e-3, 2.), (1e-4, 1.), (5e-3, 10.)]

    fig = plt.figure(figsize=(15,8))
    for i, theta in enumerate(thetas): 
        sub = fig.add_subplot(2,len(thetas)/2,i+1) 
        plt_pk, = sub.plot(kmaxs, Fii_pk[:,i], c='C0', ls='-') 
        plt_bk, = sub.plot(kmaxs, Fii_bk[:,i], c='C1', ls='-') 
        if i == 0: sub.legend([plt_pk, plt_bk], ['$P_h$', '$B_h$'], loc='lower left', fontsize=15) 

        plt_pm, = sub.plot(kmaxs, Fii_pm[:,i], c='k', ls='-') 
        if theta == 'Mnu': sub.legend([plt_pm], [r'$P^{\rm lin.}_{cb}$'], loc='lower left', fontsize=15) 

        sub.set_xlim(0.005, 0.5)
        sub.text(0.95, 0.95, theta_lbls[i], ha='right', va='top', transform=sub.transAxes, fontsize=25)
        sub.set_ylim(sigma_theta_lims[i]) 
        sub.set_yscale('log') 

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'$k_{\rm max}$', fontsize=28) 
    bkgd.set_ylabel(r'unmarginalized $\sigma_\theta$ $(1/\sqrt{F_{i,i}})$', labelpad=10, fontsize=28) 
    
    fig.subplots_adjust(wspace=0.2, hspace=0.15) 
    ffig = os.path.join(UT.fig_dir(), 'quijote.Fii_kmax%s%s.dmnu_%s.png' % (_rsd_str(rsd), _flag_str(flag), dmnu)) 
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def forecast_kmax(rsd=True, flag=None, theta_nuis=None, dmnu='fin', LT=False, planck=False):
    ''' fisher forecast for quijote P and B where theta_nuis are added as free parameters 
    as a function of kmax.
    
    :param rsd: (default: True) 
        rsd kwarg that specifies rsd set up for B(k). 
        If rsd == True, include 3 RSD directions. 
        If rsd in [0,1,2] include one of the directions
    :param flag: (default: None) 
        kwarg specifying the flag for B(k). 
        If `flag is None`, include paired-fixed and regular N-body simulation. 
        If `flag == 'ncv'` only include paired-fixed. 
        If `flag == 'reg'` only include regular N-body
    :param theta_nuis: (default: None) 
        list of nuisance parameters to include in the forecast. 
    :param dmnu: (default: 'fin') 
        derivative finite differences setup
    :param LT: (default: False)
        If True include linear theory matter powerspectrum forecasts
    :param planck: (default: False)
        If True add Planck prior 
    '''
    #kmaxs = kf * 3 * np.arange(2, 28) 
    #kmaxs = kf * 3 * np.array([1, 3, 5, 10, 15, 20, 27]) 
    kmaxs = kf * 3 * np.arange(1, 28) 
    
    pk_theta_nuis = list(np.array(theta_nuis).copy())
    bk_theta_nuis = list(np.array(theta_nuis).copy())
    if 'Bsn' in theta_nuis: pk_theta_nuis.remove('Bsn') 
    if 'b2' in theta_nuis: pk_theta_nuis.remove('b2') 
    if 'g2' in theta_nuis: pk_theta_nuis.remove('g2') 

    if planck: # read in planck prior fisher (order is Om, Ob, h, ns, s8 and Mnu) 
        _Fij_planck = np.load(os.path.join(UT.dat_dir(), 'Planck_2018_s8.npy')) 
    
    # read in fisher matrix (Fij)
    sig_pk, sig_bk, sig_pk_planck, sig_bk_planck, sig_pm, sig_pcb = [], [], [], [], [], [] 
    for i_k, kmax in enumerate(kmaxs): 
        if LT: 
            pmFij   = LT.Fij_Pm(np.logspace(-5, 2, 500), kmax=kmax, npoints=5) # linear theory Pm 
            pcbFij  = LT.Fij_Pm(np.logspace(-5, 2, 500), kmax=kmax, npoints=5, flag='cb') 
        pkFij   = FisherMatrix('pk', kmax=kmax, rsd=rsd, flag=flag, dmnu=dmnu, theta_nuis=pk_theta_nuis) 
        bkFij   = FisherMatrix('bk', kmax=kmax, rsd=rsd, flag=flag, dmnu=dmnu, theta_nuis=bk_theta_nuis) 
    
        if planck: # add planck prior 
            pkFij_planck = pkFij.copy() 
            bkFij_planck = bkFij.copy() 
            pkFij_planck[:6,:6] += _Fij_planck
            bkFij_planck[:6,:6] += _Fij_planck

        if np.linalg.cond(pkFij) > 1e16: 
            print('P Fij ill-conditioned; cond # = %.2e' % np.linalg.cond(pkFij)) 
        if np.linalg.cond(bkFij) > 1e16: 
            print('B Fij ill-conditioned; cond # = %.2e' % np.linalg.cond(bkFij)) 
        if LT: 
            sig_pm.append(np.sqrt(np.diag(np.linalg.inv(pmFij))))
            sig_pcb.append(np.sqrt(np.diag(np.linalg.inv(pcbFij))))
        sig_pk.append(np.sqrt(np.diag(np.linalg.inv(pkFij))))
        sig_bk.append(np.sqrt(np.diag(np.linalg.inv(bkFij))))
        if planck: 
            sig_pk_planck.append(np.sqrt(np.diag(np.linalg.inv(pkFij_planck))))
            sig_bk_planck.append(np.sqrt(np.diag(np.linalg.inv(bkFij_planck))))
        print('kmax=%.3f ---' %  kmax) 
        print('pk: %s' % ', '.join(['%.2e' % fii for fii in sig_pk[-1]])) 
        print('bk: %s' % ', '.join(['%.2e' % fii for fii in sig_bk[-1]])) 
        if planck: 
            print('pk w/ planck: %s' % ', '.join(['%.2e' % fii for fii in sig_pk_planck[-1]])) 
            print('bk w/ planck: %s' % ', '.join(['%.2e' % fii for fii in sig_bk_planck[-1]])) 
        if LT: 
            print('pm: %s' % ', '.join(['%.2e' % fii for fii in sig_pm[-1]])) 
            print('pcb: %s' % ', '.join(['%.2e' % fii for fii in sig_pcb[-1]])) 

    sig_pk  = np.array(sig_pk)
    sig_bk  = np.array(sig_bk)
    if planck: 
        sig_pk_planck = np.array(sig_pk_planck)
        sig_bk_planck = np.array(sig_bk_planck)
    if LT: 
        sig_pm  = np.array(sig_pm)
        sig_pcb = np.array(sig_pcb)
    
    cond_pk = (kmaxs > len(sig_pk[-1]) * kf)
    cond_bk = (kmaxs > 12. * kf)

    print('improvement of B over P') 
    for i_k in range(len(kmaxs)): 
        print('%.3f %s' % (kmaxs[i_k], ', '.join(['%.3f' % fpb for fpb in sig_pk[i_k]/sig_bk[i_k]])))

    # write out to table
    #kmax_preset = np.zeros(len(kmaxs)).astype(bool) 
    #for kmax in [0.2, 0.3, 0.4, 0.5]: 
    #    kmax_preset[(np.abs(kmaxs - kmax)).argmin()] = True
    #pk_dat = np.vstack((np.atleast_2d(kmaxs[kmax_preset]), sig_pk[kmax_preset,:].T)).T 
    #bk_dat = np.vstack((np.atleast_2d(kmaxs[kmax_preset]), sig_bk[kmax_preset,:].T)).T 
    #fpk = os.path.join(UT.doc_dir(), 'dat', 'pk_forecast_kmax.dat') 
    #np.savetxt(fpk, pk_dat, delimiter=', ', fmt='%.5f')
    #fbk = os.path.join(UT.doc_dir(), 'dat', 'bk_forecast_kmax.dat') 
    #np.savetxt(fbk, bk_dat, delimiter=', ', fmt='%.5f')

    sigma_theta_lims = [(5e-3, 1.), (1e-3, 1.), (1e-2, 10), (1e-2, 10.), (1e-2, 10.), (1e-2, 1e1)]
    if planck: sigma_theta_lims = [(5e-3, 1.), (5e-4, 1.), (5e-3, 10), (3e-3, 10), (5e-3, 10), (1e-2, 10.)]

    fig = plt.figure(figsize=(15,8))
    for i, theta in enumerate(thetas): 
        sub = fig.add_subplot(2,len(thetas)/2,i+1) 
        sub.plot(kmaxs[cond_pk], sig_pk[:,i][cond_pk], c='C0', ls='-') 
        sub.plot(kmaxs[cond_bk], sig_bk[:,i][cond_bk], c='C1', ls='-') 
        if planck: 
            sub.plot(kmaxs[cond_pk], sig_pk_planck[:,i][cond_pk], c='C0', ls=':', lw=0.75) 
            _plt, =sub.plot(kmaxs[cond_bk], sig_bk_planck[:,i][cond_bk], c='C1', ls=':', lw=0.75) 
            if theta == 'Mnu': 
                sub.legend([_plt], ['w/ Planck'], loc='lower left', handletextpad=0.2, fontsize=15) 
        if LT: 
            if theta == 'Mnu': 
                sub.plot(kmaxs, sig_pm[:,i], c='k', ls='--', label=r"$P^{\rm lin.}_{m}$") 
                sub.plot(kmaxs, sig_pcb[:,i], c='k', ls=':', label=r"$P^{\rm lin.}_{cb}$") 
                sub.legend(loc='lower left', fontsize=15) 
            else: 
                sub.plot(kmaxs, sig_pm[:,i], c='k', ls='--') 
        sub.set_xlim(0.05, 0.5)
        sub.text(0.9, 0.9, theta_lbls[i], ha='right', va='top', transform=sub.transAxes, fontsize=30)
        sub.set_ylim(sigma_theta_lims[i]) 
        sub.set_yscale('log') 
        if i == 0: 
            sub.text(0.5, 0.45, r"$P$", ha='left', va='bottom', color='C0', transform=sub.transAxes, fontsize=24)
            sub.text(0.3, 0.35, r"$B$", ha='right', va='top', color='C1', transform=sub.transAxes, fontsize=24)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'$k_{\rm max}$', fontsize=28) 
    bkgd.set_ylabel(r'{\fontsize{28pt}{3em}\selectfont{}$\sigma_\theta~/$}{\fontsize{20pt}{3em}\selectfont{}$\sqrt{\frac{V}{1 ({\rm Gpc}/h)^3}}$}', labelpad=15)#, fontsize=28) 
    fig.subplots_adjust(wspace=0.2, hspace=0.15) 

    if theta_nuis is None: nuis_str = ''
    else: nuis_str = '.'
    if 'Amp' in theta_nuis: nuis_str += 'b'
    if 'Mmin' in theta_nuis: nuis_str += 'Mmin'
    if ('Asn' in theta_nuis) or ('Bsn' in theta_nuis): nuis_str += 'SN'
    if 'b2' in theta_nuis: nuis_str += 'b2'
    if 'g2' in theta_nuis: nuis_str += 'g2'

    planck_str = ''
    if planck: planck_str = '.planck'

    ffig = os.path.join(dir_doc, 
            'quijote.Fisher_kmax%s.dmnu_%s%s%s%s%s.png' % 
            (nuis_str, dmnu, _rsd_str(rsd), _flag_str(flag), planck_str, ['', '.LT'][LT]))
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') # latex 
    return None


def forecastP02B_kmax(rsd=True, flag=None, theta_nuis=None, dmnu='fin', LT=False, planck=False):
    ''' fisher forecast for quijote P0+P2 and B where theta_nuis are added as free parameters 
    as a function of kmax.
    
    :param rsd: (default: True) 
        rsd kwarg that specifies rsd set up for B(k). 
        If rsd == True, include 3 RSD directions. 
        If rsd in [0,1,2] include one of the directions
    :param flag: (default: None) 
        kwarg specifying the flag for B(k). 
        If `flag is None`, include paired-fixed and regular N-body simulation. 
        If `flag == 'ncv'` only include paired-fixed. 
        If `flag == 'reg'` only include regular N-body
    :param theta_nuis: (default: None) 
        list of nuisance parameters to include in the forecast. 
    :param dmnu: (default: 'fin') 
        derivative finite differences setup
    :param LT: (default: False)
        If True include linear theory matter powerspectrum forecasts
    :param planck: (default: False)
        If True add Planck prior 
    '''
    #kmaxs = kf * 3 * np.arange(2, 28) 
    #kmaxs = kf * 3 * np.array([1, 3, 5, 10, 15, 20, 27]) 
    kmaxs = kf * 3 * np.arange(1, 28) 
    
    pk_theta_nuis = list(np.array(theta_nuis).copy())
    bk_theta_nuis = list(np.array(theta_nuis).copy())
    if 'Bsn' in theta_nuis: pk_theta_nuis.remove('Bsn') 
    if 'b2' in theta_nuis: pk_theta_nuis.remove('b2') 
    if 'g2' in theta_nuis: pk_theta_nuis.remove('g2') 

    if planck: # read in planck prior fisher (order is Om, Ob, h, ns, s8 and Mnu) 
        _Fij_planck = np.load(os.path.join(UT.dat_dir(), 'Planck_2018_s8.npy')) 
    
    # read in fisher matrix (Fij)
    sig_pk, sig_bk, sig_pk_planck, sig_bk_planck, sig_pm, sig_pcb = [], [], [], [], [], [] 
    for i_k, kmax in enumerate(kmaxs): 
        if LT: 
            pmFij   = LT.Fij_Pm(np.logspace(-5, 2, 500), kmax=kmax, npoints=5) # linear theory Pm 
            pcbFij  = LT.Fij_Pm(np.logspace(-5, 2, 500), kmax=kmax, npoints=5, flag='cb') 
        pkFij   = FisherMatrix('p02k', kmax=kmax, rsd=rsd, flag=flag, dmnu=dmnu, theta_nuis=pk_theta_nuis) 
        bkFij   = FisherMatrix('bk', kmax=kmax, rsd=rsd, flag=flag, dmnu=dmnu, theta_nuis=bk_theta_nuis) 
    
        if planck: # add planck prior 
            pkFij_planck = pkFij.copy() 
            bkFij_planck = bkFij.copy() 
            pkFij_planck[:6,:6] += _Fij_planck
            bkFij_planck[:6,:6] += _Fij_planck

        if np.linalg.cond(pkFij) > 1e16: 
            print('P02 Fij ill-conditioned; cond # = %.2e' % np.linalg.cond(pkFij)) 
        if np.linalg.cond(bkFij) > 1e16: 
            print('B Fij ill-conditioned; cond # = %.2e' % np.linalg.cond(bkFij)) 
        if LT: 
            sig_pm.append(np.sqrt(np.diag(np.linalg.inv(pmFij))))
            sig_pcb.append(np.sqrt(np.diag(np.linalg.inv(pcbFij))))
        sig_pk.append(np.sqrt(np.diag(np.linalg.inv(pkFij))))
        sig_bk.append(np.sqrt(np.diag(np.linalg.inv(bkFij))))
        if planck: 
            sig_pk_planck.append(np.sqrt(np.diag(np.linalg.inv(pkFij_planck))))
            sig_bk_planck.append(np.sqrt(np.diag(np.linalg.inv(bkFij_planck))))
        print('kmax=%.3f ---' %  kmax) 
        print('pk: %s' % ', '.join(['%.2e' % fii for fii in sig_pk[-1]])) 
        print('bk: %s' % ', '.join(['%.2e' % fii for fii in sig_bk[-1]])) 
        if planck: 
            print('pk w/ planck: %s' % ', '.join(['%.2e' % fii for fii in sig_pk_planck[-1]])) 
            print('bk w/ planck: %s' % ', '.join(['%.2e' % fii for fii in sig_bk_planck[-1]])) 
        if LT: 
            print('pm: %s' % ', '.join(['%.2e' % fii for fii in sig_pm[-1]])) 
            print('pcb: %s' % ', '.join(['%.2e' % fii for fii in sig_pcb[-1]])) 

    sig_pk  = np.array(sig_pk)
    sig_bk  = np.array(sig_bk)
    if planck: 
        sig_pk_planck = np.array(sig_pk_planck)
        sig_bk_planck = np.array(sig_bk_planck)
    if LT: 
        sig_pm  = np.array(sig_pm)
        sig_pcb = np.array(sig_pcb)
     
    #cond_pk = (kmaxs > float(len(sig_pk[-1])) * kf)
    cond_pk = (kmaxs > 5. * kf)
    cond_bk = (kmaxs > 12. * kf)
    print('PK kmax > %f' % (5. * kf)) 
    print('BK kmax > %f' %  (12. * kf))

    #print('improvement of B over P') 
    #for i_k in range(len(kmaxs)): 
    #    print('%.3f %s' % (kmaxs[i_k], ', '.join(['%.3f' % fpb for fpb in sig_pk[i_k]/sig_bk[i_k]])))

    # write out to table
    kmax_preset = np.zeros(len(kmaxs)).astype(bool) 
    for kmax in [0.2, 0.3, 0.4, 0.5]: 
        kmax_preset[(np.abs(kmaxs - kmax)).argmin()] = True
    pk_dat = np.vstack((np.atleast_2d(kmaxs[kmax_preset]), sig_pk[kmax_preset,:].T)).T 
    bk_dat = np.vstack((np.atleast_2d(kmaxs[kmax_preset]), sig_bk[kmax_preset,:].T)).T 
    fpk = os.path.join(UT.doc_dir(), 'dat', 'p02k_forecast_kmax.dat') 
    np.savetxt(fpk, pk_dat, delimiter=', ', fmt='%.5f')
    fbk = os.path.join(UT.doc_dir(), 'dat', 'bk_forecast_kmax.dat') 
    np.savetxt(fbk, bk_dat, delimiter=', ', fmt='%.5f')

    sigma_theta_lims = [(5e-3, 1.), (1e-3, 1.), (1e-2, 10), (1e-2, 10.), (1e-2, 10.), (1e-2, 1e1)]
    if planck: sigma_theta_lims = [(5e-3, 0.8), (5e-4, 1.), (5e-3, 10), (3e-3, 10), (5e-3, 10), (1e-2, 10.)]

    fig = plt.figure(figsize=(15,8))
    for i, theta in enumerate(thetas): 
        sub = fig.add_subplot(2,len(thetas)/2,i+1) 
        sub.plot(kmaxs[cond_pk], sig_pk[:,i][cond_pk], c='C0', ls='-') 
        sub.plot(kmaxs[cond_bk], sig_bk[:,i][cond_bk], c='C1', ls='-') 
        if planck: 
            sub.plot(kmaxs[cond_pk], sig_pk_planck[:,i][cond_pk], c='C0', ls='--', lw=1) 
            _plt, =sub.plot(kmaxs[cond_bk], sig_bk_planck[:,i][cond_bk], c='C1', ls='--', lw=1) 
            if theta == 'Mnu': 
                sub.legend([_plt], ['w/ Planck'], loc='lower left', handletextpad=0.2, fontsize=15) 
        if LT: 
            if theta == 'Mnu': 
                sub.plot(kmaxs, sig_pm[:,i], c='k', ls='--', label=r"$P^{\rm lin.}_{m}$") 
                sub.plot(kmaxs, sig_pcb[:,i], c='k', ls=':', label=r"$P^{\rm lin.}_{cb}$") 
                sub.legend(loc='lower left', fontsize=15) 
            else: 
                sub.plot(kmaxs, sig_pm[:,i], c='k', ls='--') 
        sub.set_xlim(0.05, 0.5)
        sub.text(0.9, 0.9, theta_lbls[i], ha='right', va='top', transform=sub.transAxes, fontsize=30)
        sub.set_ylim(sigma_theta_lims[i]) 
        sub.set_yscale('log') 
        if i == 0: 
            sub.text(0.45, 0.4, r"$P_0,~P_2$", ha='left', va='bottom', color='C0', transform=sub.transAxes, fontsize=24)
            sub.text(0.3, 0.35, r"$B$", ha='right', va='top', color='C1', transform=sub.transAxes, fontsize=24)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'$k_{\rm max}$', fontsize=28) 
    bkgd.set_ylabel(r'{\fontsize{28pt}{3em}\selectfont{}$\sigma_\theta~/$}{\fontsize{20pt}{3em}\selectfont{}$\sqrt{\frac{V}{1 ({\rm Gpc}/h)^3}}$}', labelpad=15)#, fontsize=28) 
    fig.subplots_adjust(wspace=0.2, hspace=0.15) 

    if theta_nuis is None: nuis_str = ''
    else: nuis_str = '.'
    if 'Amp' in theta_nuis: nuis_str += 'b'
    if 'Mmin' in theta_nuis: nuis_str += 'Mmin'
    if ('Asn' in theta_nuis) or ('Bsn' in theta_nuis): nuis_str += 'SN'
    if 'b2' in theta_nuis: nuis_str += 'b2'
    if 'g2' in theta_nuis: nuis_str += 'g2'

    planck_str = ''
    if planck: planck_str = '.planck'

    ffig = os.path.join(dir_doc, 
            'quijote.p02bFisher_kmax%s.dmnu_%s%s%s%s%s.png' % 
            (nuis_str, dmnu, _rsd_str(rsd), _flag_str(flag), planck_str, ['', '.LT'][LT]))
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') # latex 
    return None


def forecast_thetas_kmax(tts=['Om', 'Ob2', 'h', 'ns', 's8'], rsd='all', flag='reg', dmnu='fin', theta_nuis=None):
    ''' fisher forecast for quijote P and B where theta_nuis are added as free parameters 
    as a function of kmax.
    '''
    if tts == 'lcdm': tts = ['Om', 'Ob2', 'h', 'ns', 's8']
    #kmaxs = np.pi/500. * 3 * np.arange(1, 28) 
    kmaxs = kf * 3 * np.array([2, 5, 10, 15, 20, 27]) 
    
    pk_theta_nuis = list(np.array(theta_nuis).copy())
    bk_theta_nuis = list(np.array(theta_nuis).copy())
    if 'Bsn' in theta_nuis: pk_theta_nuis.remove('Bsn') 
    if 'b2' in theta_nuis: pk_theta_nuis.remove('b2') 
    if 'g2' in theta_nuis: pk_theta_nuis.remove('g2') 

    i_tt_lt = np.zeros(len(thetas)).astype(bool) 
    for tt in tts: i_tt_lt[thetas.index(tt)] = True
    i_tt_pk = np.zeros(len(thetas)+len(pk_theta_nuis)).astype(bool) 
    for tt in tts: i_tt_pk[thetas.index(tt)] = True
    i_tt_pk[len(thetas):] = True
    i_tt_bk = np.zeros(len(thetas)+len(bk_theta_nuis)).astype(bool) 
    for tt in tts: i_tt_bk[thetas.index(tt)] = True
    i_tt_bk[len(thetas):] = True

    # read in fisher matrix (Fij)
    sig_pk, sig_bk, sig_pm, sig_pc = [], [], [], [] 
    cond_pk, cond_bk = np.ones(len(kmaxs)).astype(bool), np.ones(len(kmaxs)).astype(bool) 
    for i_k, kmax in enumerate(kmaxs): 
        pmFij   = LT.Fij_Pm(np.logspace(-5, 2, 500), kmax=kmax, npoints=5) # linear theory Pm 
        pcFij  = LT.Fij_Pm(np.logspace(-5, 2, 500), kmax=kmax, npoints=5, flag='cb') 
        pkFij   = FisherMatrix('pk', kmax=kmax, rsd=rsd, flag=flag, dmnu=dmnu, theta_nuis=pk_theta_nuis) 
        bkFij   = FisherMatrix('bk', kmax=kmax, rsd=rsd, flag=flag, dmnu=dmnu, theta_nuis=bk_theta_nuis) 

        pmFij = pmFij[:,i_tt_lt][i_tt_lt,:]
        pcFij = pcFij[:,i_tt_lt][i_tt_lt,:]
        pkFij = pkFij[:,i_tt_pk][i_tt_pk,:]
        bkFij = bkFij[:,i_tt_bk][i_tt_bk,:]

        if np.linalg.cond(pkFij) > 1e16: cond_pk[i_k] = False 
        if np.linalg.cond(bkFij) > 1e16: cond_bk[i_k] = False 
        sig_pm.append(np.sqrt(np.diag(np.linalg.inv(pmFij))))
        sig_pc.append(np.sqrt(np.diag(np.linalg.inv(pcFij))))
        sig_pk.append(np.sqrt(np.diag(np.linalg.inv(pkFij))))
        sig_bk.append(np.sqrt(np.diag(np.linalg.inv(bkFij))))
        print('kmax=%.3f ---' %  kmax) 
        print('pm: %s' % ', '.join(['%.2e' % fii for fii in sig_pm[-1]])) 
        print('pc: %s' % ', '.join(['%.2e' % fii for fii in sig_pc[-1]])) 
        print('pk: %s' % ', '.join(['%.2e' % fii for fii in sig_pk[-1]])) 
        print('bk: %s' % ', '.join(['%.2e' % fii for fii in sig_bk[-1]]))

    sig_pm  = np.array(sig_pm)
    sig_pc  = np.array(sig_pc)
    sig_pk  = np.array(sig_pk)
    sig_bk  = np.array(sig_bk)

    sigma_theta_lims = [(5e-3, 5.), (1e-3, 5.), (1e-2, 50), (1e-2, 20.), (1e-2, 50.), (1e-2, 1e3)]

    fig = plt.figure(figsize=(15,8))
    for i, theta in enumerate(thetas): 
        if not i_tt_bk[i]: continue 
        sub = fig.add_subplot(2,len(thetas)/2,i+1) 

        sub.plot(kmaxs, sig_pk[:,i], c='C0', ls='-') 
        sub.plot(kmaxs, sig_bk[:,i], c='C1', ls='-') 
        if theta == 'Mnu': 
            sub.plot(kmaxs, sig_pm[:,i], c='k', ls='--', label=r"$P^{\rm lin.}_{m}$") 
            sub.plot(kmaxs, sig_pc[:,i], c='k', ls=':', label=r"$P^{\rm lin.}_{cb}$") 
            sub.legend(loc='lower left', fontsize=15) 
        else: 
            sub.plot(kmaxs, sig_pm[:,i], c='k', ls='--') 
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
    bkgd.set_ylabel(r'marginalized $\sigma_\theta$', labelpad=10, fontsize=28) 
    fig.subplots_adjust(wspace=0.2, hspace=0.15) 

    if theta_nuis is None: nuis_str = ''
    else: nuis_str = '.'
    if 'Amp' in theta_nuis: nuis_str += 'b'
    if 'Mmin' in theta_nuis: nuis_str += 'Mmin'
    if ('Asn' in theta_nuis) or ('Bsn' in theta_nuis): nuis_str += 'SN'
    if 'b2' in theta_nuis: nuis_str += 'b2'
    if 'g2' in theta_nuis: nuis_str += 'g2'
    
    ffig = os.path.join(dir_doc, 
            'quijote.Fisher.%s%s.dmnu_%s%s%s.png' % (''.join(tts), nuis_str, dmnu, _rsd_str(rsd), _flag_str(flag)))
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') # latex version 
    return None


def quijote_Forecast_kmax_scaleBFii(rsd=True, dmnu='fin', theta_nuis=None, f_Mnu=0.9, f_Ob=0.95, f_s8=0.975):
    ''' fisher forecast for quijote P and B where theta_nuis are added as free parameters 
    as a function of kmax. Convergence tests show that Fii elements of Mnu, Ob, and s8 are 
    sensitive to Nfp. The convergence relationship is also monotonic (i.e. higher Nfp, 
    lower Fii) so we test this by scaling down the Fii terms. Off-diagonal terms are 
    surprisingly immune to this. 
    '''
    #kmaxs = [np.pi/500.*6, np.pi/500.*9, 0.075, 0.1, 0.15, 0.16, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5] 
    #kmaxs = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5] #np.arange(2, 15) * 2.*np.pi/1000. * 6 #np.linspace(0.1, 0.5, 20) 
    #kmaxs = np.pi/500. * 3 * np.arange(2, 28) 
    kmaxs = np.pi/500. * 3 * np.array([2, 5, 10, 15, 20, 27]) 
    
    pk_theta_nuis = list(np.array(theta_nuis).copy())
    bk_theta_nuis = list(np.array(theta_nuis).copy())
    if 'Bsn' in theta_nuis: pk_theta_nuis.remove('Bsn') 
    if 'b2' in theta_nuis: pk_theta_nuis.remove('b2') 
    if 'g2' in theta_nuis: pk_theta_nuis.remove('g2') 
    i_ob = thetas.index('Ob') 
    i_s8 = thetas.index('s8') 
    i_mnu = thetas.index('Mnu') 
    
    # read in fisher matrix (Fij)
    sig_pk, sig_bk, sig_pm, sig_pcb = [], [], [], [] 
    cond_pk, cond_bk = np.ones(len(kmaxs)).astype(bool), np.ones(len(kmaxs)).astype(bool) 
    for i_k, kmax in enumerate(kmaxs): 
        pmFij   = LT.Fij_Pm(np.logspace(-5, 2, 500), kmax=kmax, npoints=5) # linear theory Pm 
        pcbFij  = LT.Fij_Pm(np.logspace(-5, 2, 500), kmax=kmax, npoints=5, flag='cb') 
        pkFij   = quijote_FisherMatrix('pk', kmax=kmax, rsd=rsd, dmnu=dmnu, theta_nuis=pk_theta_nuis)
        bkFij   = quijote_FisherMatrix('bk', kmax=kmax, rsd=rsd, dmnu=dmnu, theta_nuis=bk_theta_nuis) 
        if f_Ob is not None: bkFij[i_ob,i_ob] *= f_Ob
        if f_s8 is not None: bkFij[i_s8,i_s8] *= f_s8
        if f_Mnu is not None: bkFij[i_mnu,i_mnu] *= f_Mnu

        if np.linalg.cond(pkFij) > 1e16: cond_pk[i_k] = False 
        if np.linalg.cond(bkFij) > 1e16: cond_bk[i_k] = False 
        sig_pm.append(np.sqrt(np.diag(np.linalg.inv(pmFij))))
        sig_pcb.append(np.sqrt(np.diag(np.linalg.inv(pcbFij))))
        sig_pk.append(np.sqrt(np.diag(np.linalg.inv(pkFij))))
        sig_bk.append(np.sqrt(np.diag(np.linalg.inv(bkFij))))
        print('kmax=%.3f ---' %  kmax) 
        print('pm: %s' % ', '.join(['%.2e' % fii for fii in sig_pm[-1]])) 
        print('pcb: %s' % ', '.join(['%.2e' % fii for fii in sig_pcb[-1]])) 
        print('pk: %s' % ', '.join(['%.2e' % fii for fii in sig_pk[-1]])) 
        print('bk: %s' % ', '.join(['%.2e' % fii for fii in sig_bk[-1]])) 

    sig_pk  = np.array(sig_pk)
    sig_bk  = np.array(sig_bk)
    sig_pm  = np.array(sig_pm)
    sig_pcb = np.array(sig_pcb)

    #sigma_theta_lims = [(0, 0.1), (0., 0.08), (0., 0.8), (0, 0.8), (0., 0.12), (0., 0.8)]
    #sigma_theta_lims = [(0, 0.2), (0., 0.2), (0., 2.), (0, 2.), (0., 1.), (0., 2.)]
    #sigma_theta_lims = [(0, 10.), (0., 10.), (0., 10.), (0, 10.), (0., 10.), (0., 10.)]
    sigma_theta_lims = [(5e-3, 5.), (1e-3, 5.), (1e-2, 50), (1e-2, 20.), (1e-2, 50.), (1e-2, 1e3)]

    fig = plt.figure(figsize=(15,8))
    for i, theta in enumerate(thetas): 
        sub = fig.add_subplot(2,len(thetas)/2,i+1) 
        sub.plot(kmaxs, sig_pk[:,i], c='C0', ls='-') 
        sub.plot(kmaxs, sig_bk[:,i], c='C1', ls='-') 
        if theta == 'Mnu': 
            sub.plot(kmaxs, sig_pm[:,i], c='k', ls='--', label=r"$P^{\rm lin.}_{m}$") 
            sub.plot(kmaxs, sig_pcb[:,i], c='k', ls=':', label=r"$P^{\rm lin.}_{cb}$") 
            sub.legend(loc='lower left', fontsize=15) 
        else: 
            sub.plot(kmaxs, sig_pm[:,i], c='k', ls='--') 
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
    bkgd.set_ylabel(r'marginalized $\sigma_\theta$', labelpad=10, fontsize=28) 
    fig.subplots_adjust(wspace=0.2, hspace=0.15) 

    if theta_nuis is None: nuis_str = ''
    else: nuis_str = '.'
    if 'Amp' in theta_nuis: nuis_str += 'b'
    if 'Mmin' in theta_nuis: nuis_str += 'Mmin'
    if ('Asn' in theta_nuis) or ('Bsn' in theta_nuis): nuis_str += 'SN'
    if 'b2' in theta_nuis: nuis_str += 'b2'
    if 'g2' in theta_nuis: nuis_str += 'g2'

    fscale_str = ''
    if f_Mnu: fscale_str += ('.%s' % str(f_Mnu))
    if f_Ob: fscale_str += ('.%s' % str(f_Ob))
    if f_s8: fscale_str += ('.%s' % str(f_s8))

    ffig = ('quijote.Fisher_kmax.dmnu_%s%s.scaleBderiv%s%s.png' % (dmnu, nuis_str, fscale_str, ['.real', ''][rsd]))
    fig.savefig(os.path.join(UT.fig_dir(), ffig), bbox_inches='tight') 
    return None

##################################################################
# convergence tests
##################################################################
def dlogPBdtheta_Nfixedpair(rsd='all', flag='reg', dmnu='fin'):
    ''' d log P/B / d theta for different values of Nfixedpair 
    '''
    kmax = 0.5 
    if flag is None: nderivs = np.arange(1, 16) * 200 
    elif flag == 'ncv': nderivs = np.arange(1, 16) * 100 
    elif flag == 'reg': nderivs = np.arange(1, 16) * 100 

    b_ylims = [(-20, 5), (-50., 50.), (-7.5, 3), (-7., 1.), (-8., 3.), (-5., 5.)]
    fig1 = plt.figure(figsize=(12,8)) 
    fig2 = plt.figure(figsize=(12,24)) 
    for i, par in enumerate(thetas): # calculate the derivative of Bk along all the thetas 
        sub1 = fig1.add_subplot(2,3,i+1)
        sub2 = fig2.add_subplot(6,1,i+1)

        for nderiv in nderivs: 
            k, dpk_dti = Forecast.quijote_dP02kdtheta(par, log=True, rsd=rsd, flag=flag, dmnu=dmnu, Nderiv=nderiv)
            i_k, j_k, l_k, dbk_dti = Forecast.quijote_dBkdtheta(par, log=True, rsd=rsd, flag=flag, dmnu=dmnu, Nderiv=nderiv)
            if i == 0: 
                pklim = (k < kmax)
                bklim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 
                ijl = UT.ijl_order(i_k[bklim], j_k[bklim], l_k[bklim], typ='GM') # order of triangles 
            sub1.plot(k[pklim], dpk_dti[pklim]) 
            sub2.plot(range(np.sum(bklim)), dbk_dti[bklim][ijl]) 
        sub1.plot(k[pklim], dpk_dti[pklim], c='k') 
        sub2.plot(range(np.sum(bklim)), dbk_dti[bklim][ijl], c='k') 
        sub1.text(0.95, 0.95, theta_lbls[i], ha='right', va='top', transform=sub1.transAxes, fontsize=25)
        sub2.text(0.95, 0.95, theta_lbls[i], ha='right', va='top', transform=sub2.transAxes, fontsize=25)

        if i == 5: sub1.legend(loc='upper left', fontsize=15) 
        sub1.set_xscale('log') 
        sub1.set_xlim(6e-3, 0.5) 
        #sub1.set_ylim(0., 0.6) 

        if i == 5: sub2.legend(loc='upper left', fontsize=15) 
        sub2.set_xlim(0, np.sum(bklim)) 
        sub2.set_ylim(b_ylims[i]) 

    bkgd1 = fig1.add_subplot(111, frameon=False)
    bkgd1.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd1.set_xlabel(r'$k$', labelpad=10, fontsize=25) 
    bkgd1.set_ylabel(r'${\rm d}\log P_0(k)/{\rm d} \theta$', labelpad=10, fontsize=25) 
    bkgd2 = fig2.add_subplot(111, frameon=False)
    bkgd2.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd2.set_xlabel(r'triangle configurations', fontsize=25) 
    bkgd2.set_ylabel(r'${\rm d}\log B(k_1, k_2, k_3)/{\rm d} \theta$', labelpad=10, fontsize=25) 

    ffig = os.path.join(UT.fig_dir(), 'quijote_dlogPdtheta_Nderiv%s%s.dmnu_%s.png' % (_rsd_str(rsd), _flag_str(flag), dmnu))
    fig1.savefig(ffig, bbox_inches='tight') 
    ffig = os.path.join(UT.fig_dir(), 'quijote_dlogBdtheta_Nderiv%s%s.dmnu_%s.png'% (_rsd_str(rsd), _flag_str(flag), dmnu))
    fig2.savefig(ffig, bbox_inches='tight') 
    return None


def FisherMatrix_convergence(obs, kmax=0.5, rsd='all', flag=None, dmnu='fin', theta_nuis=None, Ncov=None, Nderiv=None, silent=True): 
    ''' calculate fisher matrix for parameters ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu'] 
    and specified nuisance parameters with extra kwargs for convergence testing. This is 
    slower 
    
    :param obs: 
        observable ('pk' or 'bk') 

    :param kmax: (default: 0.5) 
        kmax of the analysis 

    :param rsd: (default: True) 
        rsd kwarg that specifies rsd set up for B(k). 
        If rsd == True, include 3 RSD directions. 
        If rsd in [0,1,2] include one of the directions

    :param flag: (default: None) 
        kwarg specifying the flag for B(k). 
        If `flag is None`, include paired-fixed and regular N-body simulation. 
        If `flag == 'ncv'` only include paired-fixed. 
        If `flag == 'reg'` only include regular N-body

    :param dmnu: (default: 'fin') 
        derivative finite differences setup

    :param theta_nuis: (default: None) 
        list of nuisance parameters to include in the forecast. 
    '''
    # calculate covariance matrix (with shotnoise; this is the correct one) 
    # *** rsd and flag kwargs are ignored for the covariace matirx ***
    if obs == 'pk': 
        # read in P(k) 
        quij = Obvs.quijotePk('fiducial', rsd=2, flag='reg', silent=silent) # theta_fiducial 
        pks = quij['p0k'] + quij['p_sn'][:,None] # shotnoise uncorrected P(k) 
        if Ncov is None: 
            _Cfid = np.cov(pks.T) # covariance matrix 
            nmock = pks.shape[0]
        else: 
            _Cfid = np.cov(pks[:Ncov,:].T) # covariance matrix 
            nmock = Ncov
        klim = (quij['k'] <= kmax) # determine k limit 
        C_fid = _Cfid[:,klim][klim,:]
    elif obs == 'p02k': # monopole and quadrupole 
        # read in P(k) 
        quij = Obvs.quijotePk('fiducial', rsd=2, flag='reg', silent=silent) 
        p0ks = quij['p0k'] + quij['p_sn'][:,None] # shotnoise uncorrected P0 
        p2ks = quij['p2k'] # P2 
        pks = np.concatenate([p0ks, p2ks], axis=1)  # concatenate P0 and P2

        if Ncov is None: 
            _Cfid = np.cov(pks.T) # covariance matrix 
            nmock = pks.shape[0]
        else: 
            _Cfid = np.cov(pks[:Ncov,:].T) # covariance matrix 
            nmock = Ncov
        k = np.concatenate([quij['k'], quij['k']]) 
        klim = (k <= kmax) # determine k limit 
        C_fid = _Cfid[:,klim][klim,:]
    elif obs == 'bk':
        quij = Obvs.quijoteBk('fiducial', rsd=2, flag='reg', silent=silent) # this is hardcoded
        bks = quij['b123'] + quij['b_sn']
        if Ncov is None: 
            _Cfid = np.cov(bks.T) # calculate the covariance
            nmock = bks.shape[0]
        else: 
            if not silent: print('%i of %i mocks' % (Ncov, bks.shape[0]))
            _Cfid = np.cov(bks[:Ncov,:].T) # calculate the covariance
            nmock = Ncov
        i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3']

        klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) # k limit 
        C_fid = _Cfid[:,klim][klim,:]
    else: 
        raise NotImplementedError

    ndata = np.sum(klim) 
    f_hartlap = float(nmock - ndata - 2)/float(nmock - 1) 
    C_inv = f_hartlap * np.linalg.inv(C_fid) # invert the covariance 
    
    _thetas = copy(thetas) 
    if theta_nuis is not None: _thetas += theta_nuis 
    
    # calculate the derivatives along all the thetas 
    dobs_dt = [] 
    for par in _thetas: 
        if obs == 'pk': 
            _, dobs_dti = Forecast.quijote_dPkdtheta(par, rsd=rsd, flag=flag, dmnu=dmnu, Nderiv=Nderiv, silent=silent)
            dobs_dt.append(dobs_dti[klim])
        elif obs == 'p02k': 
            _, dobs_dti = Forecast.quijote_dP02kdtheta(par, rsd=rsd, flag=flag, dmnu=dmnu, Nderiv=Nderiv, silent=silent)
            dobs_dt.append(dobs_dti[klim])
        elif obs == 'bk': 
            # rsd and flag kwargs are passed to the derivatives
            _, _, _, dobs_dti = Forecast.quijote_dBkdtheta(par, rsd=rsd, flag=flag, dmnu=dmnu, Nderiv=Nderiv, silent=silent)
            dobs_dt.append(dobs_dti[klim])
    Fij = Forecast.Fij(dobs_dt, C_inv) 
    #assert ndata > Fij.shape[0] 
    return Fij 


def _Fij_Nderiv_convergence(obs, kmax=0.5, rsd=True, flag=None, dmnu='fin'): 
    ''' check the convergence of Fisher matrix terms when we calculate the covariance 
    matrix or derivatives using different number of mocks. 
    '''
    ij_pairs = [] 
    ij_pairs_str = [] 
    nthetas = len(thetas) 
    for i in xrange(nthetas): 
        for j in xrange(i, nthetas): 
            ij_pairs.append((i,j))
            ij_pairs_str.append(','.join([theta_lbls[i], theta_lbls[j]]))
    ij_pairs = np.array(ij_pairs) 

    fig = plt.figure(figsize=(6,15))
    print('--- Nderiv. test ---' ) 
    # convergence of derivatives 
    if obs == 'pk': nderivs = [100, 200, 300, 350, 400, 425, 450, 475, 500]
    elif obs == 'bk': 
        if flag is None: nderivs = [100, 500, 1000, 1500, 2000, 2500, 3000]
        else: 
            if (type(rsd) == bool) and rsd: nderivs = [100, 250, 500, 750, 1000, 1250, 1500]
            else: nderivs = [100, 250, 500]
    # read in fisher matrix (Fij)
    Fijs = []
    for nderiv in nderivs: 
        Fijs.append(FisherMatrix_convergence(obs, kmax=kmax, rsd=rsd, flag=flag, dmnu=dmnu, Nderiv=nderiv)) 
    Fijs = np.array(Fijs) 

    sub = fig.add_subplot(111)
    for _i, ij in enumerate(ij_pairs): 
        sub.fill_between([100, 3000], [1.-_i*0.3-0.05, 1.-_i*0.3-0.05], [1.-_i*0.3+0.05, 1.-_i*0.3+0.05],
                color='k', linewidth=0, alpha=0.25) 
        sub.plot([100., 3000.], [1.-_i*0.3, 1.-_i*0.3], c='k', ls='--', lw=1) 
        _Fij = np.array([Fijs[ik][ij[0],ij[1]] for ik in range(len(nderivs))]) 
        sub.plot(nderivs, _Fij/_Fij[-1] - _i*0.3)
    sub.set_xlabel(r"$N_{\rm deriv.}$ Quijote simulations", labelpad=10, fontsize=25) 
    if obs == 'pk': 
        sub.set_xlim([100, 500]) 
        sub.set_ylabel(r'$F_{ij}(N_{\rm deriv.})/F_{ij}(N_{\rm deriv.}=500)$', fontsize=25)
    elif obs == 'bk':
        if flag is None: sub.set_xlim(100, 3000) 
        else: 
            if (type(rsd) == bool) and rsd: sub.set_xlim(100, 1500) 
            else: sub.set_xlim(100, 500)
        sub.set_ylabel(r'$F_{ij}(N_{\rm deriv.})/F_{ij}(N_{\rm deriv.}=%i)$' % sub.get_xlim()[1], fontsize=25)
    sub.set_ylim([1.-0.3*len(ij_pairs), 1.3]) 
    sub.set_yticks([1. - 0.3 * ii for ii in range(len(ij_pairs))])
    sub.set_yticklabels(ij_pairs_str) 
    fig.subplots_adjust(wspace=0.4) 
    if obs == 'pk': 
        ffig = os.path.join(UT.fig_dir(), 'quijote.%sFij_Nderiv_convergence.dmnu_%s.kmax%.1f.png' % (obs, dmnu, kmax))
    else: 
        ffig = os.path.join(UT.fig_dir(), 'quijote.%sFij_Nderiv_convergence%s%s.dmnu_%s.kmax%.1f.png' % 
                (obs, _rsd_str(rsd), _flag_str(flag), dmnu, kmax))
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def Fij_convergence(obs, kmax=0.5, rsd=True, flag=None, dmnu='fin', silent=True): 
    ''' check the convergence of Fisher matrix terms when we calculate the covariance 
    matrix or derivatives using different number of mocks. 
    '''
    ij_pairs = [] 
    ij_pairs_str = [] 
    nthetas = len(thetas) 
    for i in xrange(nthetas): 
        for j in xrange(i, nthetas): 
            ij_pairs.append((i,j))
            ij_pairs_str.append(','.join([theta_lbls[i], theta_lbls[j]]))
    ij_pairs = np.array(ij_pairs) 

    print('--- Ncov test ---' ) 
    # convegence of covariance matrix 
    ncovs = [3000, 4000, 5000, 7500, 10000, 12000, 14000, 15000]
    # read in fisher matrix (Fij)
    Fijs = []
    for ncov in ncovs: 
        Fijs.append(FisherMatrix_convergence(obs, kmax=kmax, rsd=rsd, flag=flag, dmnu=dmnu, Ncov=ncov, silent=silent)) 
    Fijs = np.array(Fijs) 

    print(ncovs)
    fig = plt.figure(figsize=(12,15))
    sub = fig.add_subplot(121) 
    for _i, ij in enumerate(ij_pairs): 
        sub.fill_between([1000, 15000], [1.-_i*0.3-0.05, 1.-_i*0.3-0.05], [1.-_i*0.3+0.05, 1.-_i*0.3+0.05],
                color='k', linewidth=0, alpha=0.25) 
        sub.plot([1000, 15000], [1.-_i*0.3, 1.-_i*0.3], c='k', ls='--', lw=1) 
        _Fij = Fijs[:,ij[0],ij[1]] 
        sub.plot(ncovs, _Fij/_Fij[-1] - _i*0.3) 
        print('--- %s ---' % ij_pairs_str[_i]) 
        print(_Fij/_Fij[-1]) 
    sub.set_xlabel(r"$N_{\rm cov}$ Quijote simulations", labelpad=10, fontsize=25) 
    sub.set_xlim(3000, 15000) 
    sub.set_ylabel(r'$F_{ij}(N_{\rm cov})/F_{ij}(N_{\rm cov}=%i)$' % sub.get_xlim()[1], fontsize=25)
    sub.set_ylim([1. - 0.3*len(ij_pairs), 1.3]) 
    sub.set_yticks([1. - 0.3 * ii for ii in range(len(ij_pairs))])
    sub.set_yticklabels(ij_pairs_str) 

    print('--- Nderiv. test ---' ) 
    # convergence of derivatives 
    if rsd == 'all': nderivs = [100, 200, 400, 600, 800, 1000, 1200, 1400, 1500]
    else: nderivs = [100, 200, 300, 350, 400, 450, 475, 500]
    # read in fisher matrix (Fij)
    Fijs = []
    for nderiv in nderivs: 
        Fijs.append(FisherMatrix_convergence(obs, kmax=kmax, rsd=rsd, flag=flag, dmnu=dmnu, Nderiv=nderiv, silent=silent)) 
    Fijs = np.array(Fijs) 
    print(nderivs) 
    sub = fig.add_subplot(122)
    for _i, ij in enumerate(ij_pairs): 
        sub.fill_between([100, 3000], [1.-_i*0.3-0.05, 1.-_i*0.3-0.05], [1.-_i*0.3+0.05, 1.-_i*0.3+0.05],
                color='k', linewidth=0, alpha=0.25) 
        sub.plot([100., 3000.], [1.-_i*0.3, 1.-_i*0.3], c='k', ls='--', lw=1) 
        _Fij = Fijs[:,ij[0],ij[1]]
        sub.plot(nderivs, _Fij/_Fij[-1] - _i*0.3)
        print('--- %s ---' % ij_pairs_str[_i]) 
        print(_Fij/_Fij[-1]) 
    sub.set_xlabel(r"$N_{\rm deriv.}$ Quijote simulations", labelpad=10, fontsize=25) 
    sub.set_xlim(100, nderivs[-1]) 
    sub.set_ylabel(r'$F_{ij}(N_{\rm deriv.})/F_{ij}(N_{\rm deriv.}=%i)$' % sub.get_xlim()[1], fontsize=25)
    sub.set_ylim([1.-0.3*len(ij_pairs), 1.3]) 
    sub.set_yticks([1. - 0.3 * ii for ii in range(len(ij_pairs))])
    sub.set_yticklabels(ij_pairs_str) 
    fig.subplots_adjust(wspace=0.4) 
    ffig = os.path.join(dir_doc, 'quijote.%sFij_convergence%s%s.dmnu_%s.kmax%.1f.png' % 
                        (obs, _rsd_str(rsd), _flag_str(flag), dmnu, kmax))
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') # latex version 
    return None


def forecast_convergence(obs, kmax=0.5, rsd=True, flag=None, dmnu='fin', theta_nuis=None, planck=False): 
    ''' fisher forecast where we compute the covariance matrix or derivatives using different 
    number of mocks. 
    
    :param kmax: (default: 0.5) 
        k1, k2, k3 <= kmax
    '''
    if planck: # read in planck prior fisher (order is Om, Ob, h, ns, s8 and Mnu) 
        _Fij_planck = np.load(os.path.join(UT.dat_dir(), 'Planck_2018_s8.npy')) 
    # convegence of covariance matrix 
    ncovs = [3000, 4000, 5000, 7500, 10000, 12000, 14000, 15000]
    # read in fisher matrix (Fij)
    Finvs, Fiis = [], [] 
    for ncov in ncovs: 
        Fij = FisherMatrix_convergence(obs, kmax=kmax, rsd=rsd, flag=flag, dmnu=dmnu, theta_nuis=theta_nuis, Ncov=ncov)
        if planck: Fij[:6,:6] += _Fij_planck
        Finvs.append(np.linalg.inv(Fij)) # invert fisher matrix 
        Fiis.append(np.diag(Fij)) 
    Finvs = np.array(Finvs) 
    Fiis = np.array(Fiis) 

    fig = plt.figure(figsize=(12,6))
    sub = fig.add_subplot(121) 
    sub.plot([3000, 15000], [1., 1.], c='k', ls='--', lw=1) 
    sub.plot([3000, 15000], [0.9, 0.9], c='k', ls=':', lw=1) 
    print(ncovs) 
    for i in range(ntheta): 
        sig_theta = np.sqrt(Finvs[:,i,i]) 
        sigii_theta = 1./np.sqrt(Fiis[:,i]) 
        print('--- %s ---' % theta_lbls[i]) 
        print(sig_theta/sig_theta[-1]) 
        #print(sigii_theta/sigii_theta[-1]) 
        sub.plot(ncovs, sig_theta/sig_theta[-1], label=r'$%s$' % theta_lbls[i]) 
    sub.set_xlabel(r"$N_{\rm cov}$ Quijote simulations", labelpad=10, fontsize=25) 
    sub.set_xlim(3000, 15000) 
    sub.set_ylabel(r'$\sigma_\theta(N_{\rm cov})/\sigma_\theta(N_{\rm cov}=15000)$', fontsize=25)
    sub.set_ylim(0.5, 1.1) 

    print('--- Nfp test ---' ) 
    # convergence of derivatives 
    if rsd == 'all': nderivs = [100, 200, 400, 600, 800, 1000, 1200, 1400, 1500]
    else: nderivs = [100, 200, 300, 350, 400, 450, 475, 500]
    # read in fisher matrix (Fij)
    Finvs, Fiis = [], [] 
    for nderiv in nderivs: 
        Fij = FisherMatrix_convergence(obs, kmax=kmax, rsd=rsd, flag=flag, dmnu=dmnu, theta_nuis=theta_nuis, Nderiv=nderiv)
        if planck: Fij[:6,:6] += _Fij_planck
        Finvs.append(np.linalg.inv(Fij)) # invert fisher matrix 
        Fiis.append(Fij) 
    Finvs = np.array(Finvs) 
    Fiis = np.array(Fiis) 

    sub = fig.add_subplot(122)
    sub.plot([100., 3000.], [1., 1.], c='k', ls='--', lw=1) 
    sub.plot([100., 3000.], [0.9, 0.9], c='k', ls=':', lw=1) 
    for i in range(ntheta): 
        sig_theta = np.sqrt(Finvs[:,i,i]) 
        sigii_theta = 1./np.sqrt(Fiis[:,i,i]) 
        print('--- %s ---' % theta_lbls[i]) 
        print(sig_theta/sig_theta[-1]) 
        #print(sigii_theta/sigii_theta[-1]) 
        sub.plot(nderivs, sig_theta/sig_theta[-1], label=(r'$%s$' % theta_lbls[i]))
    sub.legend(loc='lower right', fontsize=20) 
    sub.set_xlabel(r"$N_{\rm deriv.}$ Quijote simulations", labelpad=10, fontsize=25) 
    sub.set_xlim(200, nderivs[-1]) 
    sub.set_ylabel(r'$\sigma_\theta(N_{\rm deriv.})/\sigma_\theta(N_{\rm deriv.}=%i)$' % nderivs[-1], fontsize=25)
    sub.set_ylim([0.5, 1.1]) 
    fig.subplots_adjust(wspace=0.25) 

    planck_str = ''
    if planck: planck_str = '.planck'

    ffig = os.path.join(dir_doc, 'quijote.%sFisher_convergence%s%s.dmnu_%s.kmax%.1f%s.png' % 
            (obs, _rsd_str(rsd), _flag_str(flag), dmnu, kmax, planck_str))
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') # latex friednly
    return None

############################################################
def quijote_nbars(): 
    '''some calculations involving nbar''' 
    # fiducial 
    quij = Obvs.quijotePk('fiducial')
    nbar_fid = (np.average(quij['Nhalos'])/1.e9)
    p0k = np.average(quij['p0k'], axis=0) 
    p0k_0p1 = p0k[np.abs(quij['k'] - 0.1).argmin()]
    print('At fiducial cosmology...')
    print('Nhalo = %f' % np.average(quij['Nhalos']))
    print('nbar = %f' % nbar_fid) 
    print('nbar x P0(k=%f) = %f' % (quij['k'][np.abs(quij['k'] - 0.1).argmin()], p0k_0p1 * nbar_fid)) 
    raise ValueError
    thetas = ['fiducial', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp', 'Ob_m', 'Ob_p', 'Om_m', 'Om_p', 'h_m', 'h_p', 'ns_m', 'ns_p', 's8_m', 's8_p']
    nbars = [] 
    for sub in thetas:
        quij = Obvs.quijoteBk(sub)
        nbars.append(np.average(quij['Nhalos'])/1.e9)
        print('%s, %.5e' % (sub, np.average(quij['Nhalos'])/1.e9))
    print(np.min(nbars))
    print(np.min(nbars) * 1e9 )
    print(thetas[np.argmin(nbars)])
    return None

##################################################################
# forecasts without free Mmin and b' (defunct) 
##################################################################
def quijote_pbkForecast_Mnu_s8(kmax=0.5):
    ''' fisher forecast for Mnu and sigma 8 using quijote from P(k) and B(k) 
    overlayed on them. Then overlay Hades parameter points ontop to see if
    things make somewhat sense.
    
    :param krange: (default: [0.01, 0.5]) 
        tuple specifying the kranges of k1, k2, k3 in the bispectrum
    '''
    # fisher matrices for powerspectrum and bispectrum
    pkFij = quijote_Fisher('pk', kmax=kmax, rsd=True, dmnu='fin', validate=False)
    bkFij = quijote_Fisher('bk', kmax=kmax, rsd=True, dmnu='fin', validate=False)

    pkFinv = np.linalg.inv(pkFij) # invert fisher matrix 
    bkFinv = np.linalg.inv(bkFij) # invert fisher matrix 

    print('P(k) marginalized constraint on Mnu = %f' % np.sqrt(pkFinv[-1,-1]))
    print('B(k1,k2,k3) marginalized constraint on Mnu = %f' % np.sqrt(bkFinv[-1,-1]))
    
    fig = plt.figure(figsize=(7,7))
    sub = fig.add_subplot(111) 
    i, j = 4, 5
    for _i, Finv in enumerate([pkFinv, bkFinv]):
        # sub inverse fisher matrix 
        Finv_sub = np.array([[Finv[i,i], Finv[i,j]], [Finv[j,i], Finv[j,j]]]) 
        theta_fid_i, theta_fid_j = theta_fid[thetas[j]], theta_fid[thetas[i]]
                
        plotEllipse(Finv_sub, sub, theta_fid_ij=[theta_fid_i, theta_fid_j], color='C%i' % _i)
        
    sub.set_xlabel(theta_lbls[i], labelpad=10, fontsize=30) 
    sub.set_xlim(theta_lims[i])
    sub.set_ylabel(theta_lbls[j], fontsize=30) 
    sub.set_ylim(theta_lims[j])
    sub.fill_between([],[],[], color='C0', label=r'$P^{\rm halo}_0(k)$') 
    sub.fill_between([],[],[], color='C1', label=r'$B^{\rm halo}(k_1, k_2, k_3)$') 
    # overlay hades
    hades_sig8 = [0.833, 0.822, 0.815, 0.806]
    hades_Mnus = [0., 0.06, 0.10, 0.15]
    sub.scatter(hades_sig8, hades_Mnus, c='k', marker='+', zorder=10, label='HADES sims')
    
    _hades_sig8 = [0.822, 0.818, 0.807, 0.798]
    sub.scatter(_hades_sig8, np.zeros(len(_hades_sig8)), c='k', marker='x', zorder=10)

    sub.legend(loc='upper right', handletextpad=0.2, markerscale=5, fontsize=20)
    ffig = os.path.join(dir_doc, 'quijote_pbkFisher_Mnu_sig8_kmax%s.png' % str(kmax).replace('.', ''))
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def quijote_Forecast_dmnu(obs, rsd=True):
    ''' fisher forecast for quijote for different methods of calculating the 
    derivative along Mnu 
    '''
    dmnus = ['fin', 'p', 'pp'][::-1]
    colrs = ['C2', 'C1', 'C0']
    alphas = [0.7, 0.8, 0.9] 
    
    # read in fisher matrix (Fij)
    Finvs = [] 
    print('%s-space' % ['real', 'redshift'][rsd])
    for dmnu in dmnus:
        Fij = quijote_Fisher(obs, kmax=0.5, rsd=rsd, dmnu=dmnu, validate=False) 
        Finv = np.linalg.inv(Fij) # invert fisher matrix 
        Finvs.append(Finv)
        print('d%s/dMnu_%s, sigma_Mnu=%f' % (obs, dmnu, np.sqrt(Finv[-1,-1])))

    fig = plt.figure(figsize=(17, 15))
    for i in xrange(ntheta-1): 
        for j in xrange(i+1, ntheta): 
            for i_k, Finv in enumerate(Finvs): 
                # sub inverse fisher matrix 
                Finv_sub = np.array([[Finv[i,i], Finv[i,j]], [Finv[j,i], Finv[j,j]]]) 

                theta_fid_i = theta_fid[thetas[i]]
                theta_fid_j = theta_fid[thetas[j]]
                
                # get ellipse parameters 
                a = np.sqrt(0.5*(Finv_sub[0,0] + Finv_sub[1,1]) + np.sqrt(0.25*(Finv_sub[0,0]-Finv_sub[1,1])**2 + Finv_sub[0,1]**2))
                b = np.sqrt(0.5*(Finv_sub[0,0] + Finv_sub[1,1]) - np.sqrt(0.25*(Finv_sub[0,0]-Finv_sub[1,1])**2 + Finv_sub[0,1]**2))
                theta = 0.5 * np.arctan2(2.0 * Finv_sub[0,1], (Finv_sub[0,0] - Finv_sub[1,1]))
        
                # plot the ellipse
                sub = fig.add_subplot(ntheta-1, ntheta-1, (ntheta-1) * (j-1) + i + 1) 

                e = Ellipse(xy=(theta_fid_i, theta_fid_j), width=2.48 * a, height=2.48 * b, angle=theta * 360./(2.*np.pi))
                sub.add_artist(e)
                e.set_alpha(alphas[i_k])
                e.set_facecolor(colrs[i_k])
                
                e = Ellipse(xy=(theta_fid_i, theta_fid_j), width=1.52 * a, height=1.52 * b, angle=theta * 360./(2.*np.pi),
                        fill=False, edgecolor='k', linestyle='--')
                sub.add_artist(e)

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
    if 'bk' in obs: 
        bkgd.text(0.82, 0.77, r'$B^{\rm halo}(k_1, k_2, k_3)$', ha='right', va='bottom', 
                    transform=bkgd.transAxes, fontsize=25)
    elif 'pk' in obs: 
        bkgd.text(0.82, 0.77, r'$P^{\rm halo}(k)$', ha='right', va='bottom', transform=bkgd.transAxes, fontsize=25)
    for colr, alpha, dmnu in zip(colrs, alphas, dmnus): 
        bkgd.fill_between([],[],[], color=colr, alpha=alpha, label=r'${\rm d}B(k)/{\rm d}M_\nu$ %s' % dmnu) 
    bkgd.legend(loc='upper right', handletextpad=0.3, bbox_to_anchor=(0.925, 0.775), fontsize=25)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.05, hspace=0.05) 
    ffig = os.path.join(dir_doc, 'quijote_%sFisher_dmnu%s.pdf' % (obs, ['_real', ''][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    ffig = os.path.join(dir_doc, 'quijote_%sFisher_dmnu%s.png' % (obs, ['_real', ''][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def hades_dchi2(krange=[0.01, 0.5]):
    ''' calculate delta chi-squared for the hades simulation using the quijote 
    simulation covariance matrix
    
    :param krange: (default: [0.01, 0.5]) 
        tuple specifying the kranges of k1, k2, k3 in the bispectrum
    '''
    # read in B(k) of fiducial quijote simulation for covariance matrix
    quij = quijoteBk('fiducial') # theta_fiducial 
    bks = quij['b123'] + quij['b_sn']                   # shotnoise uncorrected B(k) 
    i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3']

    # impose k limit on bispectrum
    kmin, kmax = krange 
    kf = 2.*np.pi/1000. # fundmaentla mode
    bklim = ((i_k*kf <= kmax) & (i_k*kf >= kmin) &
            (j_k*kf <= kmax) & (j_k*kf >= kmin) & 
            (l_k*kf <= kmax) & (l_k*kf >= kmin)) 
    ijl = UT.ijl_order(i_k[bklim], j_k[bklim], l_k[bklim], typ='GM') # order of triangles 
    bks = bks[:,bklim][:,ijl] 

    C_bk = np.cov(bks.T) # covariance matrix 
    C_inv = np.linalg.inv(C_bk) # invert the covariance 
    
    # fiducial hades B(k) 
    hades_fid = hadesBk(0.0, nzbin=4, rsd=True) # fiducial bispectrum
    Bk_fid = np.average(hades_fid['b123'], axis=0)

    mnus = [0.0, 0.06, 0.1, 0.15]
    sig8s = [0.822, 0.818, 0.807, 0.798]
    for mnu in mnus: # bispectrum for different Mnu
        hades_i = hadesBk(mnu, nzbin=4, rsd=True)
        _bk = np.average(hades_i['b123'], axis=0)
        dbk = (_bk - Bk_fid)[bklim][ijl]
        chi2 = np.sum(np.dot(dbk.T, np.dot(C_inv, dbk)))
        print('Mnu=%.2f, delta chi-squared %.2f' % (mnu, chi2))

    for sig8 in sig8s: # bispectrum for different sigma8 
        hades_i = hadesBk_s8(sig8, nzbin=4, rsd=True)
        _bk = np.average(hades_i['b123'], axis=0)
        dbk = (_bk - Bk_fid)[bklim][ijl]
        chi2 = np.sum(np.dot(dbk.T, np.dot(C_inv, dbk)))
        print('sig8=%.3f, delta chi-squared %.2f' % (sig8, chi2))
    return None


def quijote_FisherInfo(obs, kmax=0.5): 
    ''' plot the fisher information contribution of data vectors
    '''
    kf = 2.*np.pi/1000. # fundmaentla mode
    # calculate covariance matrix (with shotnoise; this is the correct one) 
    quij = Obvs.quijoteBk('fiducial', rsd=True, flag=None) # theta_fiducial 
    if obs == 'pk': 
        pks = quij['p0k1'] + 1e9/quij['Nhalos'][:,None] # uncorrect shotn oise 
        i_k = quij['k1']

        # impose k limit 
        _, _iuniq = np.unique(i_k, return_index=True)
        iuniq = np.zeros(len(i_k)).astype(bool) 
        iuniq[_iuniq] = True
        klim = (iuniq & (i_k*kf <= kmax)) 
        i_k = i_k[klim]

        C_fid = np.cov(pks[:,klim].T) 
    elif obs in ['bk', 'bk_equ', 'bk_squ']: 
        # read in full covariance matrix (with shotnoise; this is the correct one) 
        i_k, j_k, l_k, C_fid = quijoteCov(rsd=True, flag=None)
        # impose k limit 
        klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 

        i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 
        C_fid = C_fid[:,klim][klim,:]
        ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # order of triangles 

    C_inv = np.linalg.inv(C_fid) # invert the covariance 
    
    dobs_dt = [] 
    for par in thetas: # calculate the derivative of Bk along all the thetas 
        if obs == 'pk': 
            dobs_dti = quijote_dPk(par, rsd=True, dmnu='fin', flag=None)
            dobs_dt.append(dobs_dti[klim])
        elif obs == 'bk': 
            dobs_dti = quijote_dBk(par, rsd=True, dmnu='fin', flag=None)
            dobs_dt.append(dobs_dti[klim])

    ntheta = len(thetas) 
    Fij_i = np.zeros((ntheta, ntheta, C_inv.shape[0]))
    for i in range(ntheta): 
        for j in range(ntheta): 
            dmu_dt_i, dmu_dt_j = dobs_dt[i], dobs_dt[j]
            Mij = np.dot(dmu_dt_i[:,None], dmu_dt_j[None,:]) + np.dot(dmu_dt_j[:,None], dmu_dt_i[None,:])
            Fij_i[i,j,:] = 0.5 * np.diag(np.dot(C_inv, Mij))

    if obs == 'pk': 
        fig = plt.figure(figsize=(10,20))
        sub = fig.add_subplot(len(thetas)+1,1,1)

        sub.plot(i_k*kf, np.average(quij['p0k1'], axis=0)[klim], c='k')
        sub.set_xscale('log') 
        sub.set_xlim(0.018, 0.5)
        sub.set_ylabel(r'$P(k)$', fontsize=25) 
        sub.set_yscale('log') 

        for i in range(len(thetas)): 
            sub = fig.add_subplot(len(thetas)+1,1,i+2)
            sub.plot(i_k*kf, np.ones(np.sum(klim)), c='k', ls='--') 
            sub.plot(i_k*kf, Fij_i[i,i,:], c='C0', zorder=7)
            sub.text(0.95, 0.9, theta_lbls[i], ha='right', va='top', transform=sub.transAxes, fontsize=30)
            sub.set_xlim(0.018, 0.5)
            sub.set_xscale('log') 
        bkgd = fig.add_subplot(111, frameon=False)
        bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        bkgd.set_xlabel('$k_j$', fontsize=25) 
        bkgd.set_ylabel(r'$F_{i,i}^j$', fontsize=30) 

        ffig = os.path.join(UT.fig_dir(), 'quijote_Pk_FisherInfo.png')
        fig.savefig(ffig, bbox_inches='tight') 

    elif obs == 'bk': 
        fig = plt.figure(figsize=(20,20))
        sub = fig.add_subplot(len(thetas)+1,1,1)

        i_k, j_k, l_k = i_k[ijl], j_k[ijl], l_k[ijl] 
        equ = (i_k == j_k) & (j_k == l_k) 
        fld = (i_k == j_k + l_k) 
        squ = (i_k > 5 * l_k) & (j_k > 5 * l_k)
        kmin02 = (i_k*kf > 0.2) | (j_k*kf > 0.2) | (l_k*kf > 0.2) 
        kmin03 = (i_k*kf > 0.3) | (j_k*kf > 0.3) | (l_k*kf > 0.3) 
        kmin04 = (i_k*kf > 0.4) | (j_k*kf > 0.4) | (l_k*kf > 0.4) 

        sub.scatter(np.arange(np.sum(klim)), np.average(quij['b123'], axis=0)[klim][ijl], c='k', s=2, zorder=7)
        sub.scatter(np.arange(np.sum(klim))[kmin02], np.average(quij['b123'], axis=0)[klim][ijl][kmin02], c='C0', s=2, zorder=8, label=r'$k_{\rm min} > 0.2$')
        sub.scatter(np.arange(np.sum(klim))[kmin03], np.average(quij['b123'], axis=0)[klim][ijl][kmin03], c='C1', s=2, zorder=9, label=r'$k_{\rm min} > 0.3$')
        sub.scatter(np.arange(np.sum(klim))[kmin04], np.average(quij['b123'], axis=0)[klim][ijl][kmin04], c='C2', s=2, zorder=10, label=r'$k_{\rm min} > 0.4$')
        #sub.scatter(np.arange(np.sum(klim))[equ], np.average(quij['b123'], axis=0)[klim][ijl][equ], c='C0', s=10, zorder=8, label='Equ.')
        #sub.scatter(np.arange(np.sum(klim))[fld], np.average(quij['b123'], axis=0)[klim][ijl][fld], c='C1', s=10, zorder=9, label='Fold.')
        #sub.scatter(np.arange(np.sum(klim))[squ], np.average(quij['b123'], axis=0)[klim][ijl][squ], c='C3', s=10, zorder=10, label='Squ.')
        sub.legend(loc='upper right', ncol=3, handletextpad=0.1, markerscale=5, fontsize=20)
        sub.set_xlim(0, np.sum(klim))
        sub.set_ylabel(r'$B(k_1, k_2, k_3)$', fontsize=25) 
        sub.set_yscale('log') 

        for i in range(len(thetas)): 
            sub = fig.add_subplot(len(thetas)+1,1,i+2)
            sub.plot(range(np.sum(klim)), np.ones(np.sum(klim)), c='k', ls='--') 
            sub.scatter(np.arange(np.sum(klim)), Fij_i[i,i,:][ijl], c='k', s=2, zorder=7)
            sub.scatter(np.arange(np.sum(klim))[kmin02], Fij_i[i,i,:][ijl][kmin02], c='C0', s=2, zorder=8)
            sub.scatter(np.arange(np.sum(klim))[kmin03], Fij_i[i,i,:][ijl][kmin03], c='C1', s=2, zorder=9)
            sub.scatter(np.arange(np.sum(klim))[kmin04], Fij_i[i,i,:][ijl][kmin04], c='C2', s=2, zorder=10)
            #sub.scatter(np.arange(np.sum(klim))[equ], Fij_i[i,i,:][ijl][equ], c='C0', s=10, zorder=8)
            #sub.scatter(np.arange(np.sum(klim))[fld], Fij_i[i,i,:][ijl][fld], c='C1', s=10, zorder=9)
            #sub.scatter(np.arange(np.sum(klim))[squ], Fij_i[i,i,:][ijl][squ], c='C3', s=10, zorder=10)
            sub.text(0.95, 0.9, theta_lbls[i], ha='right', va='top', transform=sub.transAxes, fontsize=30)
            sub.set_xlim(0, np.sum(klim))
        #blah = ((Fij_i[i,i,:][ijl] > 5) & (np.arange(np.sum(klim)) > 1250))
        #sub.scatter(np.arange(np.sum(klim))[blah], Fij_i[i,i,:][ijl][blah], c='C2', s=2, zorder=10)

        bkgd = fig.add_subplot(111, frameon=False)
        bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        bkgd.set_xlabel('triangle configurations $j$', fontsize=25) 
        bkgd.set_ylabel(r'$F_{i,i}^j$', fontsize=30) 

        ffig = os.path.join(UT.fig_dir(), 'quijote_Bk_FisherInfo.png')
        fig.savefig(ffig, bbox_inches='tight') 

        # examine triangle configurations that contribute most to Mnu 
        highFii = (Fij_i[5,5,:][ijl] > 15) 
        print(np.sum(highFii))
        x_bins = np.linspace(0., 1., 32)
        y_bins = np.linspace(0.5, 1., 16)

        fig = plt.figure(figsize=(6,4))
        sub = fig.add_subplot(111)
        print(l_k[highFii]/i_k[highFii])
        print(j_k[highFii]/i_k[highFii])
        bplot = sub.scatter(l_k[highFii]/i_k[highFii], j_k[highFii]/i_k[highFii], c=quij['counts'][ijl][highFii]) 
        sub.set_xlim([0., 1.]) 
        sub.set_ylim([0.5, 1.]) 
        cbar_ax = fig.add_axes([0.95, 0.15, 0.0125, 0.7])
        cbar = fig.colorbar(bplot, cax=cbar_ax)
        cbar.set_label('triangle configurations', labelpad=10, rotation=90, fontsize=20)
        ffig = os.path.join(UT.fig_dir(), 'quijote_Bk_FisherInfo.Mnu_shape.png')
        fig.savefig(ffig, bbox_inches='tight') 
    return None 

##################################################################
# forecasts at Mnu = 0.1 eV 
##################################################################
def quijote_dPdthetas_non0eV():
    ''' Compare the derivatives of the powerspectrum 
    '''
    # fiducial P0(k)  
    quij = Obvs.quijoteBk('fiducial', rsd=True, flag=None)
    pk_fid = np.average(quij['p0k1'], axis=0) 
    i_k = quij['k1']
    kf = 2.*np.pi/1000.
    klim = (kf * i_k < 0.5)
    
    # fiducial derivative at Mnu = 0.0 eV
    dpdt_fid = quijote_dPk('Mnu', rsd=True, dmnu='fin', flag=None)
    dpdt_fid0 = quijote_dPk('Mnu', rsd=True, dmnu='p', flag=None)

    # derivative at Mnu = 0.1eV
    quij = Obvs.quijoteBk('Mnu_p', rsd=True, flag=None)
    pk_mnu_p = np.average(quij['p0k1'], axis=0)     # 0.1 eV
    quij = Obvs.quijoteBk('Mnu_pp', rsd=True, flag=None)
    pk_mnu_pp = np.average(quij['p0k1'], axis=0)    # 0.2 eV
    quij = Obvs.quijoteBk('Mnu_ppp', rsd=True, flag=None)
    pk_mnu_ppp = np.average(quij['p0k1'], axis=0)   # 0.4 eV
    #dpdt_0p1eV = (-9. * pk_fid - 8. * pk_mnu_p + 18. * pk_mnu_pp - pk_mnu_ppp)/2.4
    dpdt_0p1eV = (-45. * pk_fid - 40. * pk_mnu_p + 90. * pk_mnu_pp - 5. * pk_mnu_ppp)/12
    dpdt_0p1eV0 = (pk_mnu_pp- pk_fid)/0.2

    fig = plt.figure(figsize=(7,8))
    sub = fig.add_subplot(111)
    sub.plot(i_k[klim] * kf, dpdt_fid[klim], c='C0', label=r'at 0.0 eV (fin. diff. 0.0, 0.1, 0.2, 0.4 eV)') 
    sub.plot(i_k[klim] * kf, dpdt_fid0[klim], c='C0', ls='--', label=r'at 0.05 eV (w. 0.0, 0.1eV)') 
    sub.plot(i_k[klim] * kf, dpdt_0p1eV[klim], c='C1', label=r'at 0.1 eV (fin. diff. w. 0.0, 0.1, 0.2, 0.4 eV)') 
    sub.plot(i_k[klim] * kf, dpdt_0p1eV0[klim], c='C1', ls='--', label=r'at 0.1eV (w. 0.0, 0.2 eV)') 
    sub.legend(loc='upper right', fontsize=15) 
    sub.set_xlabel('$k$', fontsize=25) 
    sub.set_xscale('log') 
    sub.set_xlim(1.8e-2, 0.5) 
    sub.set_ylabel(r'$\partial P/\partial M_\nu$', fontsize=25) 
    sub.set_yscale('log') 
    sub.set_ylim(1e2, 1e5) 
    ffig = os.path.join(UT.fig_dir(), 'quijote_dPdMnu.fidtest.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def quijote_dBdthetas_non0eV(kmax=0.5):
    ''' Compare the derivatives of the bispectrum 
    '''
    # fiducial P0(k)  
    quij = Obvs.quijoteBk('fiducial', rsd=True, flag=None)
    bk_fid = np.average(quij['b123'], axis=0)     # 0.0 eV
    i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3']
    kf = 2.*np.pi/1000.
    
    klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 
    ijl = UT.ijl_order(i_k[klim], j_k[klim], l_k[klim], typ='GM') # order of triangles 

    dbdt_fid = quijote_dBk('Mnu', rsd=True, dmnu='fin', flag=None)  # derivative at 0.0eV
    dbdt_fid0 = quijote_dBk('Mnu', rsd=True, dmnu='p', flag=None)   
    dbdt_fid1 = quijote_dBk('Mnu', rsd=True, dmnu='pp', flag=None)
    
    # derivative at 0.1eV
    quij = Obvs.quijoteBk('Mnu_p', rsd=True, flag=None)
    bk_mnu_p = np.average(quij['b123'], axis=0)     # 0.1 eV
    quij = Obvs.quijoteBk('Mnu_pp', rsd=True, flag=None)
    bk_mnu_pp = np.average(quij['b123'], axis=0)    # 0.2 eV
    quij = Obvs.quijoteBk('Mnu_ppp', rsd=True, flag=None)
    bk_mnu_ppp = np.average(quij['b123'], axis=0)   # 0.4 eV
    #dbdt_0p1eV = (-9. * bk_fid - 8. * bk_mnu_p + 18. * bk_mnu_pp - bk_mnu_ppp)/2.4
    dbdt_0p1eV = (-45. * bk_fid - 40. * bk_mnu_p + 90. * bk_mnu_pp - 5. * bk_mnu_ppp)/12
    dbdt_0p1eV0 = (bk_mnu_pp - bk_fid)/0.2

    fig = plt.figure(figsize=(20, 5))
    sub = fig.add_subplot(111)
    sub.plot(range(np.sum(klim)), dbdt_fid[klim], c='C0', label=r'at 0.0 eV (fin. diff. 0.0, 0.1, 0.2, 0.4 eV)')
    sub.plot(range(np.sum(klim)), dbdt_fid0[klim], c='C0', ls='--', label=r'at 0.05 eV (w. 0.0, 0.1eV)') 
    sub.plot(range(np.sum(klim)), dbdt_0p1eV[klim], c='C1', label=r'at 0.1 eV (fin. diff. 0.0, 0.1, 0.2, 0.4 eV)')
    sub.plot(range(np.sum(klim)), dbdt_0p1eV0[klim], c='C1', ls='--', label=r'at 0.1eV (w. 0.0, 0.2 eV)') 
    sub.legend(loc='upper right', ncol=2, fontsize=15, frameon=True) 
    sub.set_xlabel('triangle configuration', fontsize=25) 
    sub.set_xlim(0, np.sum(klim)) 
    sub.set_ylabel(r'$\partial B/\partial M_\nu$', fontsize=25) 
    sub.set_yscale('log') 
    ffig = os.path.join(UT.fig_dir(), 'quijote_dBdMnu.fidtest.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def Cov_gauss(Mnu=0.0, validate=False): 
    ''' Get the Gaussian part of the bispectrum covariance matrix 
    from the power spectrum using Eq. 19 of Sefusatti et al (2006) 
    '''
    if Mnu == 0.0:  
        quij = Obvs.quijoteBk('fiducial', rsd=2, flag='reg', silent=False)  
    elif Mnu == 0.1:  
        quij = Obvs.quijoteBk('Mnu_p', rsd=2, flag='reg', silent=False)  

    i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3'] 
    p0k1 = np.average(quij['p0k1'] + 1e9/quij['Nhalos'][:,None], axis=0)
    p0k2 = np.average(quij['p0k2'] + 1e9/quij['Nhalos'][:,None], axis=0)
    p0k3 = np.average(quij['p0k3'] + 1e9/quij['Nhalos'][:,None], axis=0)
    counts = quij['counts'] 
    C_B = np.identity(len(i_k)) / counts * (2.*np.pi)**3 / kf**3 * p0k1 * p0k2 * p0k3 

    if validate:
        bks = quij['b123'] + quij['b_sn']
        _, _, _, C_fid, _ = bkCov(rsd=2, flag='reg') # only N-body for covariance
        # impose k limit 
        kmax = 0.15
        klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 
        ijl = UT.ijl_order(i_k[klim], j_k[klim], l_k[klim], typ='GM') # order of triangles 

        C_fid = C_fid[:,klim][klim,:]
        _C_B = C_B[:,klim][klim,:] 
        
        fig = plt.figure(figsize=(10,5))
        sub = fig.add_subplot(111)
        sub.plot(range(np.sum(klim)), np.diag(_C_B)/np.diag(C_fid), c='C0')
        sub.plot(range(np.sum(klim)), np.ones(np.sum(klim)), c='k', ls='--') 
        sub.set_xlabel('configurations', fontsize=25) 
        sub.set_xlim(0., np.sum(klim))
        sub.set_ylabel(r'($C_{i,i}^{\rm gauss}$ %.1f eV)/($C_{i,i}^{\rm full}$ 0.0 eV)' % Mnu, fontsize=25) 
        sub.set_ylim(0., 1.2) 

        ffig = os.path.join(UT.fig_dir(), 'Cov_Bk_gauss.%.1feV.png' % Mnu)
        fig.savefig(ffig, bbox_inches='tight') 
    return i_k, j_k, l_k, C_B
     

def quijote_bkFisher_gaussCov(kmax=0.5, Mnu_fid=0.0, dmnu='fin'): 
    ''' calculate fisher matrix for parameters ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu', 'Mmin']
    with Gaussian B covariance  
    '''
    # read in full GAUSSIAN covariance matrix 
    i_k, j_k, l_k, C_fid = Cov_gauss(Mnu=Mnu_fid)
        
    klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 

    ndata = np.sum(klim) 
    C_fid = C_fid[:,klim][klim,:]
    C_inv = np.linalg.inv(C_fid) # invert the covariance 
    
    dobs_dt = [] 
    for par in thetas: # calculate the derivative of Bk along all the thetas 
        dobs_dti = quijote_dBk(par, rsd=True, dmnu=dmnu)
        dobs_dt.append(dobs_dti[klim])
    Fij = Forecast.Fij(dobs_dt, C_inv) 
    return Fij 


def quijote_bkForecast_gaussCov(kmax=0.5, Mnu_fid=0.0, dmnu='fin'):
    ''' fisher forecast for quijote observables where a scaling factor of the
    amplitude and halo Mmin are added as free parameters. This is the default 
    set-up!
    
    :param kmax: (default: 0.5) 
        kmax of the analysis 
    '''
    # fisher matrix (Fij)
    Fij     = quijote_bkFisher_gaussCov(kmax=kmax, Mnu_fid=Mnu_fid, dmnu=dmnu)   # w/ free Mmin 
    Finv    = np.linalg.inv(Fij[4:6,4:6]) # invert fisher matrix 
    if Mnu_fid == 0.0: 
        _Fij     = quijote_Fisher_freeMmin('bk', kmax=kmax, rsd=True, dmnu=dmnu)   # w/ free Mmin 
        _Finv   = np.linalg.inv(_Fij[4:6,4:6]) # invert fisher matrix 
        print('fiducial', np.sqrt(np.diag(_Finv)))
        print('fiducial', np.sqrt(_Fij[5,5]**(-1)))
    print('gauss C', np.sqrt(np.diag(Finv)))
    print('gauss C', np.sqrt(Fij[5,5]**(-1))) 
    return None


def quijote_Forecast_s8mnu_kmax_gaussCov(dmnu='fin'):
    ''' Gaussian covariance fisher forecast for quijote for different kmax values 
    '''
    kmaxs = np.pi/500. * 6 * np.arange(1, 15) 
    
    i_s8 = thetas.index('s8')
    i_Mnu = thetas.index('Mnu')
    # get forecasts 
    sig_tt_pk_fid, sig_tt_bk_fid, sig_tt_bk_gC0, sig_tt_bk_gC1 = [], [], [], []
    for i_k, kmax in enumerate(kmaxs): 
        # fiducial P
        Fij = quijote_Fisher_freeMmin('pk', kmax=kmax, rsd=True, dmnu=dmnu) 
        Finv = np.linalg.inv(Fij[i_s8:i_Mnu+1,i_s8:i_Mnu+1]) # invert fisher matrix 
        sig_tt_pk_fid.append(np.sqrt(np.diag(Finv)))
        # fiducial B
        Fij = quijote_Fisher_freeMmin('bk', kmax=kmax, rsd=True, dmnu=dmnu) 
        Finv = np.linalg.inv(Fij[i_s8:i_Mnu+1,i_s8:i_Mnu+1]) # invert fisher matrix 
        sig_tt_bk_fid.append(np.sqrt(np.diag(Finv)))
        # gaussian Covariance at Mnu=0eV
        Fij = quijote_bkFisher_gaussCov(kmax=kmax, Mnu_fid=0.0, dmnu='fin') 
        Finv = np.linalg.inv(Fij[i_s8:i_Mnu+1,i_s8:i_Mnu+1]) # invert fisher matrix 
        sig_tt_bk_gC0.append(np.sqrt(np.diag(Finv)))
        # gaussian Covariance and derivative at Mnu=0.1eV
        Fij = quijote_bkFisher_gaussCov(kmax=kmax, Mnu_fid=0.1, dmnu='pp') 
        Finv = np.linalg.inv(Fij[i_s8:i_Mnu+1,i_s8:i_Mnu+1]) # invert fisher matrix 
        sig_tt_bk_gC1.append(np.sqrt(np.diag(Finv)))

        print('kmax=%.3e' % kmax)
        print('fiducial P sig_theta =', sig_tt_pk_fid[-1][:]) 
        print('fiducial B sig_theta =', sig_tt_bk_fid[-1][:]) 
        print('gauss 0eV sig_theta =', sig_tt_bk_gC0[-1][:]) 
        print('gauss 0.1eV sig_theta =', sig_tt_bk_gC1[-1][:]) 

    sig_tt_pk_fid = np.array(sig_tt_pk_fid)
    sig_tt_bk_fid = np.array(sig_tt_bk_fid)
    sig_tt_bk_gC0 = np.array(sig_tt_bk_gC0)
    sig_tt_bk_gC1 = np.array(sig_tt_bk_gC1)
    sigma_theta_lims = [(0., 0.11), (0., 0.1)]

    Fij_lt = np.array([[1.05542e6, 73304.8], [73304.8, 5533.04]]) 
    Finv_lt = np.linalg.inv(Fij_lt) 

    fig = plt.figure(figsize=(10,5))
    for i, theta in enumerate(thetas[i_s8:i_Mnu+1]): 
        sub = fig.add_subplot(1,2,i+1) 
        sub.plot(kmaxs, sig_tt_pk_fid[:,i], c='k', ls='--', label='fiducial P') 
        sub.plot(kmaxs, sig_tt_bk_fid[:,i], c='k', label='fiducial B') 
        sub.plot(kmaxs, sig_tt_bk_gC0[:,i], c='C0', label=r'$C^G$ 0eV') 
        sub.plot(kmaxs, sig_tt_bk_gC1[:,i], c='C1', label=r'$C^G$ + deriv. at 0.1eV') 
        sub.plot([0., 1.], [np.sqrt(np.diag(Finv_lt))[i], np.sqrt(np.diag(Finv_lt))[i]], c='k', ls=':', 
                label='lin. theory at 0.1eV') 
        sub.set_xlim(0.05, 0.5)
        sub.text(0.9, 0.9, theta_lbls[[i_s8,i_Mnu][i]], ha='right', va='top', 
                transform=sub.transAxes, fontsize=30)
        sub.set_ylim(sigma_theta_lims[i]) 
        if i == 0: 
            sub.legend(loc='upper left', frameon=True, fontsize=15) 

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'$k_{\rm max}$', fontsize=28) 
    bkgd.set_ylabel(r'$1\sigma$ constraint on $\theta$', labelpad=10, fontsize=28) 

    fig.subplots_adjust(wspace=0.2, hspace=0.15) 
    ffig = os.path.join(UT.fig_dir(), 
            'quijote_Fisher_dmnu_%s_s8mnu_kmax_gaussCov.png' % dmnu)
    fig.savefig(ffig, bbox_inches='tight') 
    return None

# fisher matrix test
def quijote_FisherTest(kmax=0.5, rsd=True, dmnu='fin', flag=None): 
    ''' comparison of the computed fisher matrix from our method verses
    the implementation in pydelfi
    '''
    for obs in ['pk', 'bk']:
        # calculate covariance matrix (with shotnoise; this is the correct one) 
        if obs == 'pk': 
            quij = Obvs.quijoteBk('fiducial', rsd=rsd, flag=flag) # theta_fiducial 
            pks = quij['p0k1'] + 1e9/quij['Nhalos'][:,None] # uncorrect shotn oise 
            i_k = quij['k1']

            # impose k limit 
            _, _iuniq = np.unique(i_k, return_index=True)
            iuniq = np.zeros(len(i_k)).astype(bool) 
            iuniq[_iuniq] = True
            klim = (iuniq & (i_k*kf <= kmax)) 
            i_k = i_k[klim]

            C_fid = np.cov(pks[:,klim].T) 
        elif 'bk' in obs: 
            # read in full covariance matrix (with shotnoise; this is the correct one) 
            i_k, j_k, l_k, C_fid = quijotebkCov(rsd=rsd, flag=flag)
            
            # impose k limit 
            if obs == 'bk': 
                klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 
            elif obs == 'bk_equ': 
                tri = (i_k == j_k) & (j_k == l_k) 
                klim = (tri & (i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 
            elif obs == 'bk_squ': 
                tri = (i_k == j_k) & (l_k == 3)
                klim = (tri & (i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 

            i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 
            C_fid = C_fid[:,klim][klim,:]

        C_inv = np.linalg.inv(C_fid) # invert the covariance 
        
        dobs_dt = [] 
        for par in thetas: # calculate the derivative of Bk along all the thetas 
            if obs == 'pk': 
                dobs_dti = quijote_dPk(par, rsd=rsd, dmnu=dmnu, flag=flag)
            elif obs == 'bk': 
                dobs_dti = quijote_dBk(par, rsd=rsd, dmnu=dmnu, flag=flag)
            dobs_dt.append(dobs_dti[klim]) 
        Fij = Forecast.Fij(dobs_dt, C_inv) # how we calculate Fij 
    
        npar = len(thetas)
        Fij_delfi = np.zeros((npar, npar)) 
        for a in range(npar):
            for b in range(npar):
                Fij_delfi[a,b] += 0.5*(np.dot(dobs_dt[a], np.dot(C_inv, dobs_dt[b])) + 
                        np.dot(dobs_dt[b], np.dot(C_inv, dobs_dt[a])))
        print('--- %s ---' % obs) 
        print(np.max(Fij-Fij_delfi))
    return None 

############################################################
# real vs rsd
############################################################
def B_detail(rsd='all', flag='reg'): 
    ''' compare the B 
    '''
    # read in fiducial bispectrum
    quij = Obvs.quijoteBk('fiducial', rsd=rsd, flag='reg') 
    bk_fid = np.average(quij['b123'], axis=0) 
    i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3']

    quij_real = Obvs.quijoteBk('fiducial', rsd='real', flag='reg') 
    bk_real = np.average(quij_real['b123'], axis=0) 
    
    klim05 = ((i_k * kf <= 0.5) & (j_k * kf <= 0.5) & (l_k * kf <= 0.5)) 

    #ijl = UT.ijl_order(i_k[klim], j_k[klim], l_k[klim], typ='GM') # order of triangles 

    fig = plt.figure(figsize=(20,10))
    # illustrate kmax 
    sub = fig.add_subplot(311)
    for kmax, clr in zip([0.5, 0.4, 0.3, 0.2, 0.1], ['C0', 'C1', 'C2', 'C3', 'C4']):  
        klim = ((i_k * kf <= kmax) & (j_k * kf <= kmax) & (l_k * kf <= kmax)) 
        print('kmax=%.1f, %i triangles' % (kmax, np.sum(klim)))
        if kmax == 0.5: sub.plot(range(np.sum(klim)), bk_fid[klim], c=clr, label=r'$k_{\rm max}=%.1f$' % kmax)   
        else: sub.plot(range(np.sum(klim)), bk_fid[klim], c=clr, label=r'%.1f' % kmax)    
    sub.legend(loc='upper right', ncol=5, fontsize=18) 
    sub.set_xlim(0., np.sum(klim05))
    sub.set_xticklabels([]) 
    sub.set_yscale('log') 
    sub.set_ylim(3e5, 2e10) 
    
    sub = fig.add_subplot(312)
    sub.plot(range(np.sum(klim05)), bk_fid[klim05], c='C0')    

    equ = ((i_k[klim05] == j_k[klim05]) & (l_k[klim05] == i_k[klim05]))
    sub.scatter(np.arange(np.sum(klim05))[equ], bk_fid[klim05][equ], marker='^', 
            facecolors='none', edgecolors='k', zorder=10, label='equilateral ($k_1 = k_2 = k_3$)') 
    
    fld = (i_k[klim05] == j_k[klim05] + l_k[klim05]) 
    sub.scatter(np.arange(np.sum(klim05))[fld], bk_fid[klim05][fld], marker='*', 
            facecolors='none', edgecolors='darkgreen', zorder=10, label='folded ($k_1 = k_2 + k_3$)') 

    #squ = ((1.8/i_k[klim05] <= 1./j_k[klim05]) & (1.8/i_k[klim05] <= 1./l_k[klim05]))
    squ = ((j_k[klim05]/i_k[klim05] > 0.8) & (l_k[klim05]/i_k[klim05] < 0.33) & ~fld) 
    sub.scatter(np.arange(np.sum(klim05))[squ], bk_fid[klim05][squ], marker='X', 
            facecolors='none', edgecolors='darkred', zorder=10, label='squeezed ($k_1, k_2 \gg k_3$)') 

    #for i in range(508, 548)[::10]: 
    #    sub.scatter([i], [bk_fid[i]], c='k') 
    #    sub.text(i, 1e9, '(%i, %i, %i)' % (i_k[i], j_k[i], l_k[i]), 
    #            ha='left', va='bottom', fontsize=5, rotation=60)
    #colin = ((1./i_k[klim05] + 1./j_k[klim05]) == 1./l_k[klim05])
    #print np.sum(colin) 
    #print i_k[klim05][colin][:10]
    #print j_k[klim05][colin][:10]
    #print l_k[klim05][colin][:10]
    #sub.scatter(np.arange(np.sum(klim05))[colin], bk_fid[klim05][colin], marker='D', 
    #        facecolors='none', edgecolors='k', zorder=10, label='co-linear triangle') 

    sub.legend(loc='upper right', handletextpad=0., ncol=5, fontsize=18) 
    sub.set_xlim(0., np.sum(klim05))
    sub.set_xticklabels([]) 
    sub.set_yscale('log') 
    sub.set_ylim(3e5, 2e10) 
    
    sub = fig.add_subplot(313)
    sub.plot(range(np.sum(klim05)), bk_fid[klim05], c='C0', label='redshift-space')    
    sub.plot(range(np.sum(klim05)), bk_real[klim05], c='C1', label='real-space')    
    sub.legend(loc='upper right', fontsize=18) 
    sub.set_xlim(0., np.sum(klim05))
    sub.set_yscale('log') 
    sub.set_ylim(3e5, 2e10) 

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel('triangle configurations', fontsize=25) 
    bkgd.set_ylabel(r'$B(k_1, k_2, k_3)$', labelpad=10., fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    fig.subplots_adjust(hspace=0.1) 
    ffig = os.path.join(dir_doc, 'quijote_B_details.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') # latex friednly
    return None


def compare_Pk_rsd(krange=[0.01, 0.5]):  
    ''' Compare the amplitude of the fiducial HADES and quijote bispectrum 
    in redshift space versus real space. 

    :param krange: (default: [0.01, 0.5]) 
        k-range of k1, k2, k3 
    '''
    kmin, kmax = krange # impose k range 
    kf = 2.*np.pi/1000. 

    fig = plt.figure(figsize=(5,5))
    sub = fig.add_subplot(111)

    hades_fid = hadesBk(0.0, nzbin=4, rsd=False) # fiducial bispectrum
    i_k = hades_fid['k1']
    Pk_fid = np.average(hades_fid['p0k1'], axis=0)
    hades_fid = hadesBk(0.0, nzbin=4, rsd=True) # fiducial bispectrum
    Pk_fid_rsd = np.average(hades_fid['p0k1'], axis=0)

    isort = np.argsort(i_k) 
    sub.plot(kf * i_k[isort], Pk_fid[isort], c='C0', label='real-space') 
    sub.plot(kf * i_k[isort], Pk_fid_rsd[isort], c='C1', label='redshift-space') 
    print('HADES', Pk_fid_rsd / Pk_fid) 

    quij_fid = quijoteBk('fiducial', rsd=False) # fiducial bispectrum
    i_k = quij_fid['k1']
    Pk_fid = np.average(quij_fid['p0k1'], axis=0)
    quij_fid = quijoteBk('fiducial', rsd=True) # fiducial bispectrum
    Pk_fid_rsd = np.average(quij_fid['p0k1'], axis=0)

    isort = np.argsort(i_k) 
    sub.plot(kf * i_k[isort], Pk_fid[isort], c='C0', ls=':', label='Quijote') 
    sub.plot(kf * i_k[isort], Pk_fid_rsd[isort], c='C1', ls=':') 
    print('Quijote', Pk_fid_rsd / Pk_fid)

    sub.legend(loc='lower left', fontsize=25) 
    sub.set_xlabel('$k$', labelpad=15, fontsize=25) 
    sub.set_xscale('log') 
    sub.set_xlim([1e-2, 1])
    sub.set_yscale('log') 
    sub.set_ylabel('$P(k_1)$', labelpad=10, fontsize=25) 
    krange_str = str(kmin).replace('.', '')+'_'+str(kmax).replace('.', '') 
    ffig = os.path.join(dir_doc, 
            'haloPk_amp_%s_%s_rsd_comparison.pdf' % (str(kmin).replace('.', ''), str(kmax).replace('.', '')))
    fig.savefig(ffig, bbox_inches='tight') 
    ffig = os.path.join(dir_doc, 
            'haloPk_amp_%s_%s_rsd_comparison.png' % (str(kmin).replace('.', ''), str(kmax).replace('.', '')))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def compare_Bk_rsd(kmax=0.5):  
    ''' Compare the amplitude of the fiducial HADES bispectrum in
    redshift space versus real space. 

    :param kmax: (default: 0.5) 
        k_max of k1, k2, k3 
    '''
    fig = plt.figure(figsize=(25,5))
    sub = fig.add_subplot(111)

    hades_fid = hadesBk(0.0, nzbin=4, rsd=False) # fiducial bispectrum
    Bk_fid = np.average(hades_fid['b123'], axis=0)
    hades_fid = hadesBk(0.0, nzbin=4, rsd=True) # fiducial bispectrum
    Bk_fid_rsd = np.average(hades_fid['b123'], axis=0) 

    i_k, j_k, l_k = hades_fid['k1'], hades_fid['k2'], hades_fid['k3']
    kf = 2.*np.pi/1000. 
    klim = ((i_k * kf <= kmax) & (j_k * kf <= kmax) & (l_k * kf <= kmax)) 

    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 
    ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # order of triangles 

    tri = np.arange(np.sum(klim))
    sub.plot(tri, Bk_fid[klim][ijl], c='C0', label='real-space') 
    sub.plot(tri, Bk_fid_rsd[klim][ijl], c='C1', label='redshift-space') 

    equ = ((i_k[ijl] == j_k[ijl]) & (l_k[ijl] == i_k[ijl]))
    sub.scatter(tri[equ], Bk_fid_rsd[klim][ijl][equ], marker='^', 
            facecolors='none', edgecolors='k', zorder=10, label='equilateral\ntriangles') 
    for iii, ii_k, x, y in zip(range(10000), i_k[ijl][equ][1::2], tri[equ][1::2], Bk_fid_rsd[klim][ijl][equ][1::2]):  
        if iii < 1: 
            sub.text(x, 1.2*y, '$k = %.2f$' % (ii_k * kf), ha='center', va='bottom', fontsize=15)
        elif iii < 6: 
            sub.text(x, 2.5*y, '$k = %.2f$' % (ii_k * kf), ha='center', va='bottom', fontsize=15)
        elif iii < 10: 
            sub.text(x, 1.5*y, '$k = %.2f$' % (ii_k * kf), ha='center', va='bottom', fontsize=15)

    quij_fid = quijoteBk('fiducial', rsd=False) # fiducial bispectrum
    _Bk_fid = np.average(quij_fid['b123'], axis=0)
    quij_fid = quijoteBk('fiducial', rsd=True) # fiducial bispectrum
    _Bk_fid_rsd = np.average(quij_fid['b123'], axis=0) 

    #sub.plot(tri, _Bk_fid[klim][ijl], c='C0', ls=':', label='Quijote') 
    #sub.plot(tri, _Bk_fid_rsd[klim][ijl], c='C1') 

    sub.legend(loc='upper right', handletextpad=0.5, markerscale=2, fontsize=22) 
    sub.set_yscale('log') 
    sub.set_xlim([0, np.sum(klim)])
    sub.set_ylim([1e5, 1e10]) 
    sub.set_xlabel('triangle configurations', labelpad=15, fontsize=25) 
    sub.set_ylabel('$B(k_1, k_2, k_3)$', labelpad=10, fontsize=25) 
    ffig = os.path.join(dir_doc,
            'haloBk_amp_kmax%s_rsd_comparison.pdf' % str(kmax).replace('.', ''))
    fig.savefig(ffig, bbox_inches='tight') 
    ffig = os.path.join(dir_doc, 
            'haloBk_amp_kmax%s_rsd_comparison.png' % str(kmax).replace('.', ''))
    fig.savefig(ffig, bbox_inches='tight') 

    fig = plt.figure(figsize=(25,5))
    sub = fig.add_subplot(111)
    tri = np.arange(np.sum(klim))
    sub.plot(tri, Bk_fid_rsd[klim][ijl]/Bk_fid[klim][ijl], c='C0', label='HADES')
    sub.plot(tri, _Bk_fid_rsd[klim][ijl]/_Bk_fid[klim][ijl], c='C1', ls=':', label='Quijote')
    sub.set_xlim([0, np.sum(klim)])
    sub.set_ylim([1., 5.]) 
    sub.set_xlabel('triangle configurations', labelpad=15, fontsize=25) 
    sub.set_ylabel('$B^{(s)}/B$', labelpad=10, fontsize=25) 
    ffig = os.path.join(dir_doc, 
            'haloBk_amp_kmax%s_rsd_ratio.png' % str(kmax).replace('.', ''))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def compare_Qk_rsd(krange=[0.01, 0.5]):  
    ''' Compare the amplitude of the fiducial HADES bispectrum in
    redshift space versus real space. 

    :param krange: (default: [0.01, 0.5]) 
        k-range of k1, k2, k3 
    '''
    hades_fid = hadesBk(0.0, nzbin=4, rsd=False) # fiducial bispectrum
    Qk_fid = np.average(hades_fid['q123'], axis=0)
    hades_fid = hadesBk(0.0, nzbin=4, rsd=True) # fiducial bispectrum
    Qk_fid_rsd = np.average(hades_fid['q123'], axis=0) 

    i_k, j_k, l_k = hades_fid['k1'], hades_fid['k2'], hades_fid['k3']
    kf = 2.*np.pi/1000. 
    kmin, kmax = krange # impose k range 
    klim = ((i_k * kf <= kmax) & (i_k * kf >= kmin) &
            (j_k * kf <= kmax) & (j_k * kf >= kmin) & 
            (l_k * kf <= kmax) & (l_k * kf >= kmin)) 

    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 

    ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # order of triangles 

    fig = plt.figure(figsize=(25,5))
    sub = fig.add_subplot(111)
    tri = np.arange(np.sum(klim))
    sub.plot(tri, Qk_fid[klim][ijl], c='C0', label='real-space') 
    sub.plot(tri, Qk_fid_rsd[klim][ijl], c='C1', label='redshift-space') 
    # mark the equilateral triangles 
    equ = ((i_k[ijl] == j_k[ijl]) & (l_k[ijl] == i_k[ijl]))
    sub.scatter(tri[equ], Qk_fid_rsd[klim][ijl][equ], c='k', zorder=10, label='equilateral') 
    for ii_k, x, y in zip(i_k[ijl][equ][::2], tri[equ][::2], Qk_fid_rsd[klim][ijl][equ][::2]):  
        sub.text(x, 0.9*y, '$k = %.2f$' % (ii_k * kf), ha='left', va='top', fontsize=15)

    quijote_fid = quijoteBk('fiducial', rsd=False) # fiducial bispectrum
    _Qk_fid = np.average(quijote_fid['q123'], axis=0)
    quijote_fid = quijoteBk('fiducial', rsd=True) # fiducial bispectrum
    _Qk_fid_rsd = np.average(quijote_fid['q123'], axis=0) 
    sub.plot(tri, _Qk_fid[klim][ijl], c='C0', ls=':', label='Quijote') 
    sub.plot(tri, _Qk_fid_rsd[klim][ijl], c='C1', ls=':') 
    
    sub.legend(loc='upper right', fontsize=25) 
    sub.set_xlim([0, np.sum(klim)])
    sub.set_ylim(0, 1) 
    sub.set_xlabel('triangle configurations', labelpad=15, fontsize=25) 
    sub.set_ylabel('$Q(k_1, k_2, k_3)$', labelpad=10, fontsize=25) 
    krange_str = str(kmin).replace('.', '')+'_'+str(kmax).replace('.', '') 
    ffig = os.path.join(dir_doc,
            'haloQk_amp_%s_%s_rsd_comparison.pdf' % (str(kmin).replace('.', ''), str(kmax).replace('.', '')))
    fig.savefig(ffig, bbox_inches='tight') 
    ffig = os.path.join(dir_doc, 
            'haloQk_amp_%s_%s_rsd_comparison.png' % (str(kmin).replace('.', ''), str(kmax).replace('.', '')))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def compare_Bk_rsd_SNuncorr(krange=[0.01, 0.5]):  
    ''' Compare the amplitude of the fiducial HADES bispectrum in
    redshift space versus real space. 

    :param krange: (default: [0.01, 0.5]) 
        k-range of k1, k2, k3 
    '''
    hades_fid = hadesBk(0.0, nzbin=4, rsd=False) # fiducial bispectrum
    Bk_fid = np.average(hades_fid['b123'] + hades_fid['b_sn'], axis=0)
    hades_fid = hadesBk(0.0, nzbin=4, rsd=True) # fiducial bispectrum
    Bk_fid_rsd = np.average(hades_fid['b123'] + hades_fid['b_sn'], axis=0) 

    i_k, j_k, l_k = hades_fid['k1'], hades_fid['k2'], hades_fid['k3']
    kf = 2.*np.pi/1000. 
    kmin, kmax = krange # impose k range 
    klim = ((i_k * kf <= kmax) & (i_k * kf >= kmin) &
            (j_k * kf <= kmax) & (j_k * kf >= kmin) & 
            (l_k * kf <= kmax) & (l_k * kf >= kmin)) 

    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 

    ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # order of triangles 

    fig = plt.figure(figsize=(25,5))
    sub = fig.add_subplot(111)
    tri = np.arange(np.sum(klim))
    sub.plot(tri, Bk_fid[klim][ijl], c='C0', label='real-space') 
    sub.plot(tri, Bk_fid_rsd[klim][ijl], c='C1', label='redshift-space') 

    equ = ((i_k[ijl] == j_k[ijl]) & (l_k[ijl] == i_k[ijl]))
    sub.scatter(tri[equ], Bk_fid_rsd[klim][ijl][equ], c='k', zorder=10, label='equilateral') 
    for ii_k, x, y in zip(i_k[ijl][equ][::2], tri[equ][::2], Bk_fid_rsd[klim][ijl][equ][::2]):  
        sub.text(x, 1.1*y, '$k = %.2f$' % (ii_k * kf), ha='left', va='bottom', fontsize=15)

    sub.legend(loc='upper right', fontsize=25) 
    sub.set_yscale('log') 
    sub.set_xlim([0, np.sum(klim)])
    sub.set_ylim([5e7, 1e10]) 
    sub.set_xlabel('triangle configurations', labelpad=15, fontsize=25) 
    sub.set_ylabel('$B(k_1, k_2, k_3)$ shot noise uncorrected', labelpad=10, fontsize=25) 
    krange_str = str(kmin).replace('.', '')+'_'+str(kmax).replace('.', '') 
    ffig = os.path.join(UT.fig_dir(), 
            'haloBkSNuncorr_amp_%s_%s_rsd_comparison.png' % (str(kmin).replace('.', ''), str(kmax).replace('.', '')))
    fig.savefig(ffig, bbox_inches='tight') 

    fig = plt.figure(figsize=(25,5))
    sub = fig.add_subplot(111)
    tri = np.arange(np.sum(klim))
    sub.plot(tri, Bk_fid_rsd[klim][ijl]/Bk_fid[klim][ijl], c='C0')
    sub.set_xlim([0, np.sum(klim)])
    sub.set_ylim([1., 2.]) 
    sub.set_xlabel('triangle configurations', labelpad=15, fontsize=25) 
    sub.set_ylabel('$B^{(s)}/B$', labelpad=10, fontsize=25) 
    krange_str = str(kmin).replace('.', '')+'_'+str(kmax).replace('.', '') 
    ffig = os.path.join(UT.fig_dir(), 
            'haloBkSNuncorr_amp_%s_%s_rsd_ratio.png' % (str(kmin).replace('.', ''), str(kmax).replace('.', '')))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def compare_Qk_rsd_SNuncorr(krange=[0.01, 0.5]):  
    ''' Compare the amplitude of the fiducial HADES bispectrum in
    redshift space versus real space. 

    :param krange: (default: [0.01, 0.5]) 
        k-range of k1, k2, k3 
    '''
    hades_fid = hadesBk(0.0, nzbin=4, rsd=False) # fiducial bispectrum
    bk = np.average((hades_fid['b123'] + hades_fid['b_sn']), axis=0)
    pk1 = np.average(hades_fid['p0k1'] + 1e9/hades_fid['Nhalos'][:,None], axis=0)
    pk2 = np.average(hades_fid['p0k2'] + 1e9/hades_fid['Nhalos'][:,None], axis=0)
    pk3 = np.average(hades_fid['p0k3'] + 1e9/hades_fid['Nhalos'][:,None], axis=0)
    Qk_fid = bk / (pk1 * pk2 + pk1 * pk3 + pk2 * pk3)
    hades_fid = hadesBk(0.0, nzbin=4, rsd=True) # fiducial bispectrum
    bk = np.average((hades_fid['b123'] + hades_fid['b_sn']), axis=0)
    pk1 = np.average(hades_fid['p0k1'] + 1e9/hades_fid['Nhalos'][:,None], axis=0)
    pk2 = np.average(hades_fid['p0k2'] + 1e9/hades_fid['Nhalos'][:,None], axis=0)
    pk3 = np.average(hades_fid['p0k3'] + 1e9/hades_fid['Nhalos'][:,None], axis=0)
    Qk_fid_rsd = bk / (pk1 * pk2 + pk1 * pk3 + pk2 * pk3)

    i_k, j_k, l_k = hades_fid['k1'], hades_fid['k2'], hades_fid['k3']
    kf = 2.*np.pi/1000. 
    kmin, kmax = krange # impose k range 
    klim = ((i_k * kf <= kmax) & (i_k * kf >= kmin) &
            (j_k * kf <= kmax) & (j_k * kf >= kmin) & 
            (l_k * kf <= kmax) & (l_k * kf >= kmin)) 

    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 

    ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # order of triangles 

    fig = plt.figure(figsize=(25,5))
    sub = fig.add_subplot(111)
    tri = np.arange(np.sum(klim))
    sub.plot(tri, Qk_fid[klim][ijl], c='C0', label='real-space') 
    sub.plot(tri, Qk_fid_rsd[klim][ijl], c='C1', label='redshift-space') 
    
    # mark the equilateral triangles 
    equ = ((i_k[ijl] == j_k[ijl]) & (l_k[ijl] == i_k[ijl]))
    sub.scatter(tri[equ], Qk_fid_rsd[klim][ijl][equ], c='k', s=2, zorder=10, label='equilateral') 
    for ii_k, x, y in zip(i_k[ijl][equ][::2], tri[equ][::2], Qk_fid_rsd[klim][ijl][equ][::2]):  
        sub.text(x, 0.9*y, '$k = %.2f$' % (ii_k * kf), ha='left', va='top', fontsize=15)

    sub.legend(loc='upper right', fontsize=25) 
    sub.set_xlim([0, np.sum(klim)])
    sub.set_ylim(0, 1) 
    sub.set_xlabel('triangle configurations', labelpad=15, fontsize=25) 
    sub.set_ylabel('$Q(k_1, k_2, k_3)$ shot noise uncorrected', labelpad=10, fontsize=25) 
    krange_str = str(kmin).replace('.', '')+'_'+str(kmax).replace('.', '') 
    ffig = os.path.join(UT.fig_dir(), 
            'haloQkSNuncorr_amp_%s_%s_rsd_comparison.png' % (str(kmin).replace('.', ''), str(kmax).replace('.', '')))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


############################################################
# P + B joint forecast 
############################################################
def quijotePk_scalefactor(rsd=True, validate=False): 
    ''' we have to scale the amplitude of P(k) so that we can invert the joint
    P(k) + B(k) covariance matrix. We pick a scale factor that roughly minimizes
    the conditional number
    '''
    scalefactor = 2e5
    if validate: 
        # lets start with something stupid 
        quij = Obvs.quijoteBk('fiducial', rsd=rsd)  
        i_k = quij['k1'] 
        mu_pk = np.average(quij['p0k1'], axis=0)    # get average quijote pk 
        mu_bk = np.average(quij['b123'], axis=0)    # get average quijote bk 

        # only keep 
        _, iuniq = np.unique(i_k, return_index=True) 
        med_pkamp = np.median(mu_pk[iuniq]) # median pk amplitude
        med_bkamp = np.median(mu_bk) # median bk amplitude 

        fig = plt.figure(figsize=(20,5))
        sub = fig.add_subplot(111)
        sub.scatter(range(len(iuniq)), mu_pk[iuniq], c='C1', s=1, label='unscale $P_0(k)$') 
        sub.scatter(range(len(iuniq)), scalefactor * mu_pk[iuniq], c='C0', s=1, label='scaled $P_0(k)$') 
        sub.scatter(range(len(iuniq), len(iuniq)+len(mu_bk)), mu_bk, c='k', s=1, label='$B_0(k)$')
        sub.legend(loc='upper right', markerscale=10, handletextpad=0.25, fontsize=20) 
        sub.set_xlim(0, len(iuniq)+len(mu_bk)) 
        sub.set_yscale('log') 
        ffig = os.path.join(UT.fig_dir(), 'quijotePk_scalefactor%s.png' % ['_real', ''][rsd])
        fig.savefig(ffig, bbox_inches='tight') 
    return scalefactor  


def quijote_scaledpkbkCov(kmax=0.5, rsd=True): 
    ''' plot the covariance matrix of the quijote fiducial bispectrum and scaled
    fiducial powerspectrum. 
    '''
    # read in P(k) and B(k) 
    quij = Obvs.quijoteBk('fiducial', rsd=rsd) # theta_fiducial 
    bks = quij['b123'] + quij['b_sn']                   # shotnoise uncorrected B(k) 
    pks = quij['p0k1'] + 1e9 / quij['Nhalos'][:,None]   # shotnoise uncorrected P(k) 
    i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3']
     
    # impose k limit on bispectrum
    kf = 2.*np.pi/1000. # fundmaentla mode
    bklim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 
    ijl = UT.ijl_order(i_k[bklim], j_k[bklim], l_k[bklim], typ='GM') # order of triangles 
    bks = bks[:,bklim][:,ijl] 
    # impose k limit on powerspectrum 
    _, _iuniq = np.unique(i_k, return_index=True)
    iuniq = np.zeros(len(i_k)).astype(bool) 
    iuniq[_iuniq] = True
    pklim = (iuniq & (i_k*kf <= kmax)) 
    pks = pks[:,pklim]

    # powerspectrum amplitude has to be scaled 
    f_pks = quijotePk_scalefactor(rsd=rsd, validate=False)
    pbks = np.concatenate([f_pks * pks, bks], axis=1) # joint data vector

    C_pbk = np.cov(pbks.T) # covariance matrix 
    print('P(k) scaled by %f' % f_pks) 
    print('covariance matrix condition number = %.5e' % np.linalg.cond(C_pbk)) 

    # plot the covariance matrix 
    fig = plt.figure(figsize=(10,8))
    sub = fig.add_subplot(111)
    cm = sub.pcolormesh(C_pbk, norm=LogNorm(vmin=1e5, vmax=1e18))
    cbar = fig.colorbar(cm, ax=sub) 
    sub.set_title(r'Quijote $P(k)$ and $B(k_1, k_2, k_3)$ Covariance', fontsize=25)
    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_scaledpbkCov_kmax%s%s.png' % 
            (str(kmax).replace('.', ''), ['_real', ''][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def _quijote_pkbkCov_triangle(typ, krange=[0.01, 0.5]): 
    ''' plot the covariance matrix of the quijote fiducial bispectrum. 
    '''
    # read in P(k) and B(k) 
    quij = quijoteBk('fiducial') # theta_fiducial 
    bks = quij['b123'] + quij['b_sn']                   # shotnoise uncorrected B(k) 
    pks = quij['p0k1'] + 1e9 / quij['Nhalos'][:,None]   # shotnoise uncorrected P(k) 
    i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3']
     
    # impose k limit on bispectrum
    kmin, kmax = krange 
    kf = 2.*np.pi/1000. # fundmaentla mode
    # impose k limit 
    if typ == 'equ': # only equilateral 
        tri = (i_k == j_k) & (j_k == l_k) 
    elif typ == 'squ': # i_k >= j_k >= l_k (= 3*kf~0.01) 
        tri = (i_k == j_k) & (l_k == 3)
    bklim = (tri & 
            (i_k*kf <= kmax) & (i_k*kf >= kmin) &
            (j_k*kf <= kmax) & (j_k*kf >= kmin) & 
            (l_k*kf <= kmax) & (l_k*kf >= kmin)) 
    ijl = UT.ijl_order(i_k[bklim], j_k[bklim], l_k[bklim], typ='GM') # order of triangles 
    bks = bks[:,bklim][:,ijl] 
    # impose k limit on powerspectrum 
    _, _iuniq = np.unique(i_k, return_index=True)
    iuniq = np.zeros(len(i_k)).astype(bool) 
    iuniq[_iuniq] = True
    pklim = (iuniq & (i_k*kf <= kmax) & (i_k*kf >= kmin)) 
    pks = pks[:,pklim]
    
    pbks = np.concatenate([pks, bks], axis=1) # joint data vector

    C_pbk = np.cov(pbks.T) # covariance matrix 
    print('covariance matrix condition number = %.5e' % np.linalg.cond(C_pbk)) 

    # plot the covariance matrix 
    fig = plt.figure(figsize=(10,8))
    sub = fig.add_subplot(111)
    cm = sub.pcolormesh(C_pbk, norm=LogNorm(vmin=1e5, vmax=1e18))
    cbar = fig.colorbar(cm, ax=sub) 
    sub.set_title(r'Quijote $P(k)$ and $B(k_1, k_2, k_3)$ Covariance', fontsize=25)
    ffig = os.path.join(UT.fig_dir(), 'quijote_pbk_Cov_%s_%s_%s.png' % 
            (typ, str(kmin).replace('.', ''), str(kmax).replace('.', '')))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def _quijote_Fishertest(kmax):
    quij_fid = Obvs.quijoteBk('fiducial') # theta_fiducial 
    kf = 2.*np.pi/1000.
    k1 = quij_fid['k1'] 

    _, _iuniq = np.unique(k1, return_index=True)
    iuniq = np.zeros(len(k1)).astype(bool) 
    iuniq[_iuniq] = True
    klim = iuniq & (kf * k1 <= kmax) 

    # read in P(k) 
    pks = quij_fid['p0k1'][:,klim] + 1e9 / quij_fid['Nhalos'][:,None]   # shotnoise uncorrected P(k) 
    C_pk = np.cov(pks.T) # covariance matrix 
    Cinv = np.linalg.inv(C_pk)#np.identity(np.sum(klim)) * np.diag(C_pk)**(-1)

    p0k_fid = np.average(quij_fid['p0k1'], axis=0)[klim]

    quij_0p1 = Obvs.quijoteBk('Mnu_p')
    p0k_0p1 = np.average(quij_0p1['p0k1'], axis=0)[klim]

    dpk = ((p0k_0p1 - p0k_fid)/0.1)#/p0k_0p1
    dpk_dmnu = quijote_dPk('Mnu', rsd=True, dmnu='p')[klim]
    dpk_dsig = quijote_dPk('s8', rsd=True)[klim]
    
    M00 = np.dot(dpk_dsig[:,None], dpk_dsig[None,:]) + np.dot(dpk_dsig[:,None], dpk_dsig[None,:])
    M01 = np.dot(dpk_dsig[:,None], dpk_dmnu[None,:]) + np.dot(dpk_dmnu[:,None], dpk_dsig[None,:])
    M10 = np.dot(dpk_dmnu[:,None], dpk_dsig[None,:]) + np.dot(dpk_dsig[:,None], dpk_dmnu[None,:])
    M11 = np.dot(dpk_dmnu[:,None], dpk_dmnu[None,:]) + np.dot(dpk_dmnu[:,None], dpk_dmnu[None,:])

    F00 = 0.5 * np.trace(np.dot(Cinv, M00)) 
    F01 = 0.5 * np.trace(np.dot(Cinv, M01)) 
    F10 = 0.5 * np.trace(np.dot(Cinv, M10)) 
    F11 = 0.5 * np.trace(np.dot(Cinv, M11)) 
    Fij = np.array([[F00, F01], [F10, F11]]) 
    print("Licia's way")
    print(Fij)
    print(np.sqrt(np.diag(np.linalg.inv(Fij))))

    F00 = np.dot(dpk_dsig, np.dot(Cinv, dpk_dsig)) 
    F01 = np.dot(dpk_dsig, np.dot(Cinv, dpk_dmnu)) 
    F10 = np.dot(dpk_dmnu, np.dot(Cinv, dpk_dsig)) 
    F11 = np.dot(dpk_dmnu, np.dot(Cinv, dpk_dmnu)) 
    Fij = np.array([[F00, F01], [F10, F11]]) 
    print("not Licia's way")
    print(Fij)
    print(np.sqrt(np.diag(np.linalg.inv(Fij))))
    print(np.sqrt(np.diag(np.linalg.inv(Forecast.Fij([dpk_dsig, dpk_dmnu], Cinv)))))
    print("full pipeline p")
    _Fij = quijote_Fisher_freeMmin('pk', kmax=kmax, rsd=True, dmnu='p')
    print(_Fij[4:6,4:6])
    print( np.sqrt(np.diag(np.linalg.inv(_Fij[4:6,4:6]))))
    print(np.sqrt(np.diag(np.linalg.inv(_Fij)[4:6,4:6])))
    _Fij = quijote_Fisher_freeMmin('pk', kmax=kmax, rsd=True, dmnu='fin')
    print("full pipeline fin")
    print(_Fij[4:6,4:6])
    print(np.sqrt(np.diag(np.linalg.inv(_Fij[4:6,4:6]))))
    print(np.sqrt(np.diag(np.linalg.inv(_Fij)[4:6,4:6])))
    return None 


def _ChanBlot_SN(): 
    ''' calculate SN as a function of kmax  using Eq. (47) in 
    Chan & Blot (2017). 
    '''
    # read in P
    quij = Obvs.quijotePk('fiducial', rsd='all', flag='reg', silent=False) 
    k_p = quij['k'] 
    pk = np.average(quij['p0k'], axis=0) 
    pksn = quij['p0k'] + quij['p_sn'][:,None] # shotnoise uncorrected P(k) 

    # read in B 
    quij = Obvs.quijoteBk('fiducial', rsd='all', flag='reg', silent=False)  
    bk = np.average(quij['b123'], axis=0) 
    bksn = quij['b123'] + quij['b_sn'] # uncorrected for shot noise  
    i_k, l_k, j_k = quij['k1'], quij['k2'], quij['k3'] 

    kmaxs = np.linspace(0.04, 0.75, 10) 
    SN_B, SN_P = [], [] 
    for kmax in kmaxs: 
        # k limit 
        pklim = (k_p <= kmax) 
        bklim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 

        C_bk = np.cov(bksn[:,bklim].T) # calculate the B covariance
        Ci_B = np.linalg.inv(C_bk) 
        C_pk = np.cov(pksn[:,pklim].T) # calculate the P covariance  
        Ci_P = np.linalg.inv(C_pk) 
        sn_b_i = np.sqrt(np.matmul(bk[bklim].T, np.matmul(Ci_B, bk[bklim])))
        sn_p_i = np.sqrt(np.matmul(pk[pklim].T, np.matmul(Ci_P, pk[pklim])))

        print('kmax=%.2f: P:%f, B:%f' % (kmax, sn_p_i, sn_b_i)) 
        SN_B.append(sn_b_i)
        SN_P.append(sn_p_i)

    cb_p_kmax = np.array([0.04869675251658636, 0.06760829753919823, 0.07673614893618194, 0.08609937521846012, 0.10592537251772895, 0.11415633046188466, 0.1341219935405372, 0.14288939585111032, 0.1612501027337743, 0.17080477200597086, 0.18728370830175495, 0.19838096568365063, 0.21134890398366477, 0.22908676527677735, 0.25703957827688645, 0.3037386091946106, 0.3823842536581126, 0.44668359215096326, 0.5339492735741767, 0.6760829753919819, 0.8511380382023763, 0.9828788730000325, 1.1481536214968824]) 
    cb_p_sn = np.array([9.554281212853747, 14.56687309222178, 16.512840354510395, 18.506604023110235, 21.462641346777612, 22.20928889966336, 24.054044983627143, 24.329804200374014, 25.176195307362626, 25.756751569612643, 25.756751569612643, 26.052030872682657, 25.756751569612643, 26.35069530242852, 26.35069530242852, 26.35069530242852, 26.35069530242852, 26.65278366645541, 26.35069530242852, 26.65278366645541, 26.65278366645541, 26.65278366645541, 27.2673896573547]) 

    cb_b_kmax = np.array([0.03845917820453539, 0.05754399373371572, 0.07629568911615335, 0.09495109992021988, 0.11415633046188466, 0.1333521432163325, 0.15310874616820308, 0.17179083871575893, 0.19164610627353873, 0.21134890398366477, 0.2277718241857323, 0.24688801049062103, 0.26607250597988114, 0.2834653633489668, 0.3054921113215514, 0.325461783498046,  0.3487385841352186]) 
    cb_b_sn = np.array([2.027359157379195,3.5440917014545286,5.1041190104348715,6.338408101544683,7.023193813563101,7.8711756484503095,8.33282150847736,8.922674486302233,9.233078642191177,9.445990941294742,9.886657856152153,10,10.230597298425085,10.347882416158368,10.707867049863955,10.707867049863955,10.830623660351296]) 

    fig = plt.figure(figsize=(6,5)) 
    sub = fig.add_subplot(111)
    sub.plot(kmaxs, SN_B, c='k', label='B') 
    sub.plot(cb_b_kmax, cb_b_sn, c='k', ls='--') 
    sub.plot(kmaxs, SN_P, c='C0', label='P') 
    sub.plot(cb_p_kmax, cb_p_sn, c='C0', ls='--', label='Chan and Blot 2017') 
    sub.legend(loc='lower right', fontsize=15) 
    sub.set_xlabel(r'$k_{\rm max}$ [$h$/Mpc]', fontsize=20) 
    sub.set_xscale('log')
    sub.set_xlim(3e-2, 1.) 
    sub.set_ylabel(r'S/N', fontsize=20) 
    sub.set_yscale('log') 
    sub.set_ylim(0.5, 3e2) 
    ffig = os.path.join(UT.fig_dir(), '_ChanBlot_SN.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


############################################################
# etc 
############################################################
def _rsd_str(rsd): 
    # assign string based on rsd kwarg 
    if rsd == 'all': return ''
    elif rsd in [0, 1, 2]: return '.rsd%i' % rsd
    elif rsd == 'real': return '.real'
    else: raise NotImplementedError


def _flag_str(flag): 
    # assign string based on flag kwarg
    return ['.%s' % flag, ''][flag is None]


if __name__=="__main__": 
    # covariance matrices
    '''
        for rsd in [0, 1, 2]: pkCov(rsd=rsd, flag='reg', silent=False) 
        for rsd in [0, 1, 2]: p02kCov(rsd=rsd, flag='reg', silent=False) 
        for rsd in [0, 1, 2]: bkCov(rsd=rsd, flag='reg', silent=False) 
        _pkCov(kmax=0.5, rsd=0, flag='reg') # condition number 3.39136+05 
        _pkCov(kmax=0.5, rsd=1, flag='reg') # condition number 3.38878+05 
        _pkCov(kmax=0.5, rsd=2, flag='reg') # condition number 3.40810+05 
        _p02kCov(kmax=0.5, rsd=0, flag='reg') # condition number 1.46109e+06
        _p02kCov(kmax=0.5, rsd=1, flag='reg') # condition number 1.44529e+06
        _p02kCov(kmax=0.5, rsd=2, flag='reg') # condition number 1.49064e+06
        _bkCov(kmax=0.5, rsd=0, flag='reg') # condition number 1.73685+08
        _bkCov(kmax=0.5, rsd=1, flag='reg') # condition number 1.71010+08
        _bkCov(kmax=0.5, rsd=2, flag='reg') # condition number 1.73518+08
        _pkbkCov(kmax=0.5)      # condition number 1.74845e+08
    '''
    # deriatives 
    '''
        for flag in ['ncv', 'reg']: P_theta(rsd=0, flag=flag)
        for flag in ['ncv', 'reg']: P02_theta(rsd=0, flag=flag)
        for flag in ['ncv', 'reg']: B_theta(rsd=0, flag=flag)
        for flag in ['ncv', 'reg']: P_theta(rsd='all', flag=flag)
        for flag in ['ncv', 'reg']: P02_theta(rsd='all', flag=flag)
        for flag in ['ncv', 'reg']: B_theta(rsd='all', flag=flag)
        for flag in ['ncv', 'reg']: _dPdthetas(log=True, rsd=0, flag=flag, dmnu='fin')
        for flag in ['ncv', 'reg']: _dP02dthetas(log=True, rsd=0, flag=flag, dmnu='fin')
        for flag in ['ncv', 'reg']: _dBdthetas(log=True, rsd=0, flag=flag, dmnu='fin')
        dBdthetas(kmax=0.5, log=True, rsd='all', flag='reg', dmnu='fin')
        dlogBdMnu(kmax=0.5, flag='reg')
        for flag in ['ncv', 'reg']: dlogPBdMnu(rsd='all', flag=flag)
        _dBdthetas_ncv(kmax=0.5, log=True, rsd=True, dmnu='fin')
    ''' 
    # fisher forecasts with different nuisance parameters 
    '''
        for flag in ['ncv', 'reg']: 
            forecast('pk', kmax=0.5, rsd='all', flag=flag, dmnu='fin', theta_nuis=['Amp', 'Mmin'])
            forecast('pk', kmax=0.5, rsd='all', flag=flag, dmnu='fin', theta_nuis=['Amp', 'Mmin', 'Asn'])
            forecast('p02k', kmax=0.5, rsd='all', flag=flag, dmnu='fin', theta_nuis=['Amp', 'Mmin'])
            forecast('p02k', kmax=0.5, rsd='all', flag=flag, dmnu='fin', theta_nuis=['Amp', 'Mmin', 'Asn'])
            forecast('bk', kmax=0.5, rsd='all', flag=flag, dmnu='fin', theta_nuis=['Amp', 'Mmin'])
            forecast('bk', kmax=0.5, rsd='all', flag=flag, dmnu='fin', theta_nuis=['Amp', 'Mmin', 'Asn', 'Bsn'])
            forecast('bk', kmax=0.5, rsd='all', flag=flag, dmnu='fin', theta_nuis=['Amp', 'Mmin', 'Asn', 'Bsn', 'b2', 'g2'])
        for flag in ['ncv', 'reg']: 
            forecast('pk', kmax=0.5, rsd='all', flag=flag, dmnu='fin', theta_nuis=['Amp', 'Mmin'], planck=True)
            forecast('pk', kmax=0.5, rsd='all', flag=flag, dmnu='fin', theta_nuis=['Amp', 'Mmin', 'Asn'], planck=True)
            forecast('bk', kmax=0.5, rsd='all', flag=flag, dmnu='fin', theta_nuis=['Amp', 'Mmin'], planck=True)
            forecast('bk', kmax=0.5, rsd='all', flag=flag, dmnu='fin', theta_nuis=['Amp', 'Mmin', 'Asn', 'Bsn'], planck=True)
            forecast('bk', kmax=0.5, rsd='all', flag=flag, dmnu='fin', theta_nuis=['Amp', 'Mmin', 'Asn', 'Bsn', 'b2', 'g2'], planck=True)
        forecast_kmax_table(dmnu='fin', theta_nuis=None)
    '''
    # P+B fisher forecasts with different nuisance parameters 
    p02bForecast(kmax=0.5, rsd='all', flag='reg', theta_nuis=['Amp', 'Mmin'], dmnu='fin', planck=True)
    '''
        for flag in ['ncv', 'reg']: 
            pbForecast(kmax=0.5, rsd='all', flag=flag, theta_nuis=['Amp', 'Mmin'], dmnu='fin')
            pbForecast(kmax=0.5, rsd='all', flag=flag, theta_nuis=['Amp', 'Mmin', 'Asn', 'Bsn'], dmnu='fin')
            pbForecast(kmax=0.5, rsd='all', flag=flag, theta_nuis=['Amp', 'Mmin', 'Asn', 'Bsn', 'b2', 'g2'], dmnu='fin')
        for flag in ['ncv', 'reg']: 
            p02bForecast(kmax=0.5, rsd='all', flag=flag, theta_nuis=['Amp', 'Mmin'], dmnu='fin')
            p02bForecast(kmax=0.5, rsd='all', flag=flag, theta_nuis=['Amp', 'Mmin', 'Asn', 'Bsn'], dmnu='fin')
            p02bForecast(kmax=0.5, rsd='all', flag=flag, theta_nuis=['Amp', 'Mmin', 'Asn', 'Bsn', 'b2', 'g2'], dmnu='fin')
        for flag in ['ncv', 'reg']: 
            pbForecast(kmax=0.5, rsd='all', flag=flag, theta_nuis=['Amp', 'Mmin'], dmnu='fin', planck=True)
            pbForecast(kmax=0.5, rsd='all', flag=flag, theta_nuis=['Amp', 'Mmin', 'Asn', 'Bsn'], dmnu='fin', planck=True)
            pbForecast(kmax=0.5, rsd='all', flag=flag, theta_nuis=['Amp', 'Mmin', 'Asn', 'Bsn', 'b2', 'g2'], dmnu='fin', planck=True)
    '''
    # fisher forecasts as a function of kmax with different nuisance parameters 
    '''
        for flag in ['ncv', 'reg']: 
            Fii_kmax(rsd='all', flag=flag, dmnu='fin')
            forecast_kmax(rsd='all', flag=flag, dmnu='fin', theta_nuis=['Amp', 'Mmin'])
            forecast_kmax(rsd='all', flag=flag, dmnu='fin', theta_nuis=['Amp', 'Mmin'], planck=True)
            forecast_kmax(rsd='all', flag=flag, dmnu='fin', theta_nuis=['Amp', 'Mmin', 'Asn', 'Bsn'])
            forecast_kmax(rsd='all', flag=flag, dmnu='fin', theta_nuis=['Amp', 'Mmin', 'Asn', 'Bsn', 'b2', 'g2'])
            forecastP02B_kmax(rsd='all', flag=flag, dmnu='fin', theta_nuis=['Amp', 'Mmin'])
            forecastP02B_kmax(rsd='all', flag=flag, dmnu='fin', theta_nuis=['Amp', 'Mmin'], planck=True)
    '''
    # fisher forecasts as a function of kmax excluding some cosmo. parameters with different nuisance parameters 
    '''
        for flag in ['ncv', 'reg']: 
            forecast_thetas_kmax(tts='lcdm', rsd='all', flag=flag, dmnu='fin', theta_nuis=['Amp', 'Mmin'])
            forecast_thetas_kmax(tts='lcdm', rsd='all', flag=flag, dmnu='fin', theta_nuis=['Amp', 'Mmin', 'Asn', 'Bsn'])
            forecast_thetas_kmax(tts='lcdm', rsd='all', flag=flag, dmnu='fin', theta_nuis=['Amp', 'Mmin', 'Asn', 'Bsn', 'b2', 'g2'])
    '''
    # --- convergence tests ---  
    '''
        for flag in ['ncv', 'reg']: 
            dlogPBdtheta_Nfixedpair(rsd=True, flag=flag, dmnu='fin')
            Fij_convergence('pk', kmax=0.5, rsd='all', flag=flag, dmnu='fin')
            Fij_convergence('p02k', kmax=0.5, rsd='all', flag=flag, dmnu='fin')
            Fij_convergence('bk', kmax=0.5, rsd='all', flag=flag, dmnu='fin')
        for flag in ['ncv', 'reg']: 
            forecast_convergence('pk', kmax=0.5, rsd='all', flag=flag, dmnu='fin')
            forecast_convergence('p02k', kmax=0.5, rsd='all', flag=flag, dmnu='fin')
            forecast_convergence('bk', kmax=0.5, rsd='all', flag=flag, dmnu='fin')
            forecast_convergence('pk', kmax=0.5, rsd='all', flag=flag, dmnu='fin', planck=True)
            forecast_convergence('bk', kmax=0.5, rsd='all', flag=flag, dmnu='fin', planck=True)
    ''' 
    # calculations 
    '''
        quijote_nbars()
        _ChanBlot_SN()
        hades_dchi2(krange=[0.01, 0.5])
        Cov_gauss(Mnu=0.0, validate=False)
        Cov_gauss(Mnu=0.1, validate=False)
    '''
    # --- fisher matrix test --- 
    '''
        for kmax in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]: 
            quijote_FisherTest(kmax=kmax, rsd=True, dmnu='fin')
    '''
    # rsd 
    #B_detail(rsd='all', flag='reg')
