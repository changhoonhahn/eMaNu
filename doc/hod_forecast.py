'''

cosmological parameter forecasts including HOD parameters
using halo catalogs from the Quijote simulations. 

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
from matplotlib.colors import LogNorm


kf = 2.*np.pi/1000. # fundmaentla mode

dir_doc = os.path.join(UT.doc_dir(), 'paper2', 'figs') # figures for paper 
dir_hod = os.path.join(UT.dat_dir(), 'Galaxies') 

fscale_pk = 1e5 # see fscale_Pk() for details

thetas = ['Om', 'Ob2', 'h', 'ns', 's8', 'Mnu', 'logMmin', 'sigma_logM', 'logM0', 'alpha', 'logM1'] 
theta_lbls = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$', 
        r'$\log M_{\rm min}$', r'$\sigma_{\log M}$', r'$\log M_0$', r'$\alpha$', r'$\log M_1$']
theta_fid = {'Mnu': 0., 'Ob': 0.049, 'Ob2': 0.049, 'Om': 0.3175, 'h': 0.6711,  'ns': 0.9624,  's8': 0.834, 
        'logMmin': 13.65, 'sigma_logM': 0.2, 'logM0': 14.0, 'alpha': 1.1, 'logM1': 14.0} # fiducial theta 
# nuisance parameters
theta_nuis_lbls = {'Amp': "$b'$", 'b1': r'$b_1$', 'Asn': r"$A_{\rm SN}$", 'Bsn': r"$B_{\rm SN}$", 'b2': '$b_2$', 'g2': '$g_2$'}
theta_nuis_fids = {'Amp': 1., 'b1': 1., 'Asn': 1e-6, 'Bsn': 1e-3, 'b2': 1., 'g2': 1.} 
theta_nuis_lims = {'Amp': (0.75, 1.25), 'b1': (0.9, 1.1), 'Asn': (0., 1.), 'Bsn': (0., 1.), 'b2': (0.95, 1.05), 'g2': (0.95, 1.05)} 

# --- covariance matrices --- 
def p02kCov(rsd=2, flag='reg', silent=True): 
    ''' return the covariance matrix of the galaxy power spectrum monopole *and* quadrupole. 
    '''
    assert flag == 'reg', "only n-body should be used for covariance"
    assert rsd != 'all', "only one RSD direction should be used otherwise modes will be correlated" 

    fcov = os.path.join(dir_hod, 'quijhod_p02Cov%s%s.hdf5' % (_rsd_str(rsd), _flag_str(flag)))

    if os.path.isfile(fcov): 
        if not silent: print('reading ... %s' % os.path.basename(fcov))
        Fcov = h5py.File(fcov, 'r') # read in fiducial covariance matrix 
        C_pk = Fcov['Cov'][...]
        k = Fcov['k'][...]
        Nmock = Fcov['Nmock'][...]
    else:
        if not silent: print('calculating ... %s' % os.path.basename(fcov))
        # read in P(k) 
        quij = Obvs.quijhod_Pk('fiducial', rsd=rsd, flag=flag, silent=silent) 
        p0ks = quij['p0k'] + quij['p_sn'][:,None] # shotnoise uncorrected P0 
        p2ks = quij['p2k'] # P2 
        pks = np.concatenate([p0ks, p2ks], axis=1)  # concatenate P0 and P2
        C_pk = np.cov(pks.T) # covariance matrix 
        k = np.concatenate([quij['k'], quij['k']]) 
        Nmock = quij['p0k'].shape[0] 

        f = h5py.File(fcov, 'w') # write to hdf5 file 
        f.create_dataset('Nmock', data=Nmock)
        f.create_dataset('Cov', data=C_pk) 
        f.create_dataset('k', data=k) 
        f.close()
    return k, C_pk, Nmock 


def bkCov(rsd=2, flag='reg', silent=True): 
    ''' return the full covariance matrix of the quijote bispectrum
    computed using 

    :return cov:
        big ass covariance matrix of all the triangle configurations in 
        the default ordering. 
    '''
    assert flag == 'reg', "only n-body should be used for covariance"
    assert rsd != 'all', "only one RSD direction should be used otherwise modes will be correlated" 

    fcov = os.path.join(dir_hod, 'quijhod_bCov%s%s.hdf5' % (_rsd_str(rsd), _flag_str(flag)))

    if os.path.isfile(fcov): 
        if not silent: print('reading ... %s' % os.path.basename(fcov))

        Fcov = h5py.File(fcov, 'r') # read in covariance matrix 
        cov = Fcov['Cov'][...]
        k1, k2, k3 = Fcov['k1'][...], Fcov['k2'][...], Fcov['k3'][...]
        Nmock = Fcov['Nmock'][...] 
    else: 
        if not silent: print('calculating ... %s' % os.path.basename(fcov))

        quij = Obvs.quijhod_Bk('fiducial', rsd=rsd, flag=flag, silent=silent) 
        bks = quij['b123'] + quij['b_sn']
        if not silent: print('%i Bk measurements' % bks.shape[0]) 

        cov = np.cov(bks.T) # calculate the covariance
        k1, k2, k3 = quij['k1'], quij['k2'], quij['k3']
        Nmock = quij['b123'].shape[0]

        f = h5py.File(fcov, 'w') # write to hdf5 file 
        f.create_dataset('Nmock', data=Nmock)
        f.create_dataset('Cov', data=cov) 
        f.create_dataset('k1', data=quij['k1']) 
        f.create_dataset('k2', data=quij['k2']) 
        f.create_dataset('k3', data=quij['k3']) 
        f.close()
    return k1, k2, k3, cov, Nmock


def fscale_Pk(rsd=2, flag='reg', kmax=0.5, silent=True): 
    ''' determine amplitude scaling factor for the power spectrum
    to reduce the range of the Pk and Bk amplitudes 
    '''
    # read in Pl(k)
    quij = Obvs.quijhod_Pk('fiducial', rsd=rsd, flag=flag, silent=silent) 
    p0ks = quij['p0k'] + quij['p_sn'][:,None] # shotnoise uncorrected P0 
    p2ks = quij['p2k'] # P2 

    k = quij['k']
    pklim = (k <= kmax) 
    
    # read in B(k) 
    quij = Obvs.quijhod_Bk('fiducial', rsd=rsd, flag=flag, silent=silent) 
    bks = quij['b123'] + quij['b_sn']

    i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3']
    bklim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) # k limit 

    pbks = np.concatenate([p0ks[:,pklim], p2ks[:,pklim], bks[:,bklim]], axis=1)
    pbk = np.mean(pbks, axis=0) 

    fig = plt.figure(figsize=(20,5))
    sub = fig.add_subplot(111)
    sub.scatter(range(len(pbk)), pbk, c='k', s=2)
    for i, fscale in enumerate([1e4, 1e5, 1e6]): 
        _pbks = np.concatenate([fscale*p0ks[:,pklim], fscale*p2ks[:,pklim], bks[:,bklim]], axis=1)
        _pbk = np.mean(_pbks, axis=0) 
        cov = np.cov(_pbks.T) # calculate the covariance
        print('%i, %.2e' % (fscale, np.linalg.cond(cov))) 
        sub.plot(range(len(_pbk)), _pbk, c='C%i' % i, lw=1, label=r'$f_{\rm scale} = %i$' % fscale)
    sub.set_xlim(0, len(pbk))
    sub.set_ylabel('$P_0, P_2, B_0$', fontsize=20) 
    sub.set_yscale('log') 
    sub.set_ylim(2e7, 1e11)
    sub.legend(loc='upper right', fontsize=15) 
    fig.savefig(os.path.join(dir_hod, 'figs', 'fscale_Pk.png'), bbox_inches='tight')
    return None 


def p02bkCov(rsd=2, flag='reg', silent=True):
    ''' calculate the covariance matrix of the P0, P2, B
    '''
    assert flag == 'reg', "only n-body should be used for covariance"
    assert rsd != 'all', "only one RSD direction should be used otherwise modes will be correlated" 

    fcov = os.path.join(dir_hod, 'quijhod_p02bCov%s%s.hdf5' % (_rsd_str(rsd), _flag_str(flag)))

    if os.path.isfile(fcov): 
        if not silent: print('reading ... %s' % os.path.basename(fcov))
        Fcov = h5py.File(fcov, 'r') # read in fiducial covariance matrix 
        cov = Fcov['Cov'][...]
        k = Fcov['k'][...]
        k1 = Fcov['k1'][...]
        k2 = Fcov['k2'][...]
        k3 = Fcov['k3'][...]
        Nmock = Fcov['Nmock'][...]
    else:
        if not silent: print('calculating ... %s' % os.path.basename(fcov))

        # read in P(k) 
        quij = Obvs.quijhod_Pk('fiducial', rsd=rsd, flag=flag, silent=silent) 
        p0ks = quij['p0k'] + quij['p_sn'][:,None] # shotnoise uncorrected P0 
        p2ks = quij['p2k'] # P2 
        pks = np.concatenate([p0ks, p2ks], axis=1)  # concatenate P0 and P2
        k = np.concatenate([quij['k'], quij['k']]) 

        # read in B(k) 
        quij = Obvs.quijhod_Bk('fiducial', rsd=rsd, flag=flag, silent=silent) 
        k1, k2, k3 = quij['k1'], quij['k2'], quij['k3']
        bks = quij['b123'] + quij['b_sn']
        pbks = np.concatenate([fscale_pk * pks, bks], axis=1) 
        
        cov = np.cov(pbks.T) # calculate the covariance
        Nmock = pbks.shape[0]

        f = h5py.File(fcov, 'w') # write to hdf5 file 
        f.create_dataset('Nmock', data=Nmock)
        f.create_dataset('Cov', data=cov) 
        f.create_dataset('k', data=k) 
        f.create_dataset('k1', data=k1) 
        f.create_dataset('k2', data=k2) 
        f.create_dataset('k3', data=k3) 
        f.close()
    return [k, k1, k2, k3], cov, Nmock 


def plot_p02bkCov(kmax=0.5, rsd=2, flag='reg'): 
    ''' plot the covariance and correlation matrices of the quijote fiducial bispectrum. 
    '''
    ks, Cov, _ = p02bkCov(rsd=rsd, flag=flag) 
    k, i_k, j_k, l_k = ks 

    klim = np.zeros(Cov.shape[0]).astype(bool) 
    klim[:len(k)] = (k <= kmax)
    klim[len(k):] = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) # k limit
    Cov = Cov[klim,:][:,klim]
    print(Cov[:5,:5])
    print('covariance matrix condition number = %.5e' % np.linalg.cond(Cov)) 

    # correlation matrix 
    Ccorr = ((Cov / np.sqrt(np.diag(Cov))).T / np.sqrt(np.diag(Cov))).T
    print(Ccorr[:5,:5])

    # plot the covariance matrix 
    fig = plt.figure(figsize=(20,8))
    sub = fig.add_subplot(121)
    cm = sub.pcolormesh(Cov, norm=LogNorm(vmin=1e11, vmax=1e18))
    cbar = fig.colorbar(cm, ax=sub) 
    cbar.set_label(r'$\widehat{P}_0(k), \widehat{P}_2(k), \widehat{B}_0(k_1, k_2, k_3)$ covariance matrix, ${\bf C}_{P,B}$', 
            fontsize=25, labelpad=10, rotation=90)

    sub = fig.add_subplot(122)
    cm = sub.pcolormesh(Ccorr, vmin=-0.025, vmax=1.) 
    cbar = fig.colorbar(cm, ax=sub) 
    cbar.set_label(r'$\widehat{P}_0(k), \widehat{P}_2(k), \widehat{B}_0(k_1, k_2, k_3)$ correlation matrix', 
            fontsize=25, labelpad=10, rotation=90)
    ffig = os.path.join(dir_hod, 'figs', 
            'quijote_p02bCov_kmax%s%s%s.png' % (str(kmax).replace('.', ''), _rsd_str(rsd), _flag_str(flag)))
    fig.savefig(ffig, bbox_inches='tight') 
    #fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') # latex friednly
    return None 

# --- derivatives --- 
def dP02k(theta, seed=range(5), rsd='all', dmnu='fin', Nderiv=None, silent=True, overwrite=False):
    ''' d P_l,g(k)/d theta. Derivative of the HOD power spectrum multipole. 

    :param theta: 
        string that specifies the parameter to take the 
        derivative by. 

    :param seed:
        If seed == 'all', use all the HOD seeds, which as of 06/03/2020 is
        seeds 0 to 5. For seed = 0, 1 there are 500 realizations. For seed =
        2-4 there are 100 but more are being run. (default: 'all') 

    :param rsd: 
        RSD direction (0, 1, 2, or 'all'). If rsd == 'all', combines the
        derivatives along all three RSD directions  

    :param dmnu: 
        finite different step choices for calculating derivative along Mnu.
        fin = {fid_za, +, ++, +++}
        p = {fid_za, p} 
        (default: 'fin') 

    :param Nderiv: 
        reduced number of realizations for derivatives. This is used for
        convergence tests only. (default: None) 

    :param silent: 
        If False, prints out info 

    :param overwrite: 
        If True, overwrites file stores in ${doc_dir}/dat/hod/

    :return: 
        [k, dp02, dlogp02]
    '''
    # --- hardcoded settings --- 
    flag = 'reg'
    z = 0 
    # --- 
    dir_dat = os.path.join(UT.doc_dir(), 'dat', 'hod') 
    fdpk = os.path.join(dir_dat, 
            'hod_dP02dtheta.%s%s%s%s%s.dat' % 
            (theta, _rsd_str(rsd), _seed_str(seed), _flag_str(flag),
                _Nderiv_str(Nderiv))) 

    if not os.path.isfile(fdpk): raise ValueError('%s does not exist' %
            os.path.basename(fdpk))

    if not silent: print('--- reading %s ---' % fdpk) 
    icols = range(3) 
    if theta == 'Mnu': 
        idmnu = {'fin': 1, 'fin0': 3, 'p': 5, 'pp': 7, 'ppp': 9}[dmnu]
        icols = [0, idmnu, idmnu+1]

    k, dpdt, dlogpdt = np.loadtxt(fdpk, skiprows=1, unpack=True, usecols=icols)
    return k, dpdt, dlogpdt


def dBk(theta, seed=range(5), rsd='all', dmnu='fin', Nderiv=None, silent=True, overwrite=False):
    ''' d Bg(k)/d theta. Derivative of the HOD bispectrum. 

    :param theta: 
        string that specifies the parameter to take the 
        derivative by. 

    :param seed:
        If seed == 'all', use all the HOD seeds, which as of 06/03/2020 is
        seeds 0 to 5. For seed = 0, 1 there are 500 realizations. For seed =
        2-4 there are 100 but more are being run. (default: 'all') 

    :param rsd: 
        RSD direction (0, 1, 2, or 'all'). If rsd == 'all', combines the
        derivatives along all three RSD directions  

    :param dmnu: 
        finite different step choices for calculating derivative along Mnu.
        fin = {fid_za, +, ++, +++}
        p = {fid_za, p} 
        (default: 'fin') 

    :param Nderiv: 
        reduced number of realizations for derivatives. This is used for
        convergence tests only. (default: None) 

    :param silent: 
        If False, prints out info 

    :param overwrite: 
        If True, overwrites file stores in ${doc_dir}/dat/hod/

    :return: 
        i_k, j_k, l_k, db/dt, dlogb/dt
    '''
    # --- hardcoded settings --- 
    flag = 'reg'
    z = 0
    # --- 
    dir_dat = os.path.join(UT.doc_dir(), 'dat', 'hod') 
    fdbk = os.path.join(dir_dat, 
            'hod_dBdtheta.%s%s%s%s%s.dat' % 
            (theta, _rsd_str(rsd), _seed_str(seed), _flag_str(flag),
                _Nderiv_str(Nderiv))) 

    assert os.path.isfile(fdbk)

    if not silent: print('--- reading %s ---' % fdbk) 
    icols = range(5) 
    if theta == 'Mnu': 
        idmnu = {'fin': 3, 'fin0': 5, 'p': 7, 'pp': 9, 'ppp': 11}[dmnu]
        icols = [0, 1, 2, idmnu, idmnu+1]
    
    i_k, j_k, l_k, dbdt, dlogbdt = np.loadtxt(fdbk, skiprows=1,
            unpack=True, usecols=icols) 

    return i_k, j_k, l_k, dbdt, dlogbdt


def dP02Bk(theta, seed=range(5), rsd='all', dmnu='fin', fscale_pk=1., Nderiv=None, silent=True):
    ''' combined derivative dP0/dtt, dP2/dtt, dB/dtt
    '''
    # dP0/dtheta, dP2/dtheta
    k, dp, dlogp = dP02k(theta, seed=seed, rsd=rsd, dmnu=dmnu, Nderiv=Nderiv, silent=silent) 
    # dB0/dtheta
    i_k, j_k, l_k, db, dlogb = dBk(theta, seed=seed, rsd=rsd, dmnu=dmnu, Nderiv=Nderiv, silent=silent) 
    return k, i_k, j_k, l_k, np.concatenate([fscale_pk * dp, db]), np.concatenate([dlogp, dlogb])


def plot_dP02B(kmax=0.5, seed=range(5), rsd='all', dmnu='fin', log=True): 
    ''' compare derivatives w.r.t. the different thetas
    '''
    # y range of P0, P2 plots 
    logplims = [(-10., 5.), (-2, 15), (-3., 1.), (-4., 0.), (-2., 2.), (-0.5, 0.5), (-2., 4), (-2., 1.), None, (-1., 2.), (-5., 1.)] 
    logblims = [(-13., -6.), (-2., 24.), (-5., -0.5), (-5., -0.5), (-2., 2.), (0.2, 0.7), (3., 6.), None, None, (2, 6), None] 

    plims = [(-5e6, 1e5), (-1e5, 5e6), (-1e6, 5e4), (-1e6, 6e4), (-1e5, 5e5),
            (None, 5e4), (-1e5, 5e5), (-1e5, 1.1e4), (-2e4, 5e3), (-5e4, 2e5),
            (-5e5, 1e5)]

    fig = plt.figure(figsize=(30,3*len(thetas)))
    gs = mpl.gridspec.GridSpec(len(thetas), 2, figure=fig, width_ratios=[1,7], hspace=0.1, wspace=0.15) 

    for i, tt in enumerate(thetas): 
        _k, _ik, _jk, _lk, dpb, dlogpb = dP02Bk(tt, seed=seed, rsd=rsd, dmnu=dmnu, silent=False)
        if log: dpb = dlogpb 
        
        nk0 = int(len(_k)/2) 
        kp  = _k[:nk0]
        dp0 = dpb[:nk0]
        dp2 = dpb[nk0:len(_k)]
        db  = dpb[len(_k):] 

        # k limits 
        pklim = (kp < kmax) 
        bklim = ((_ik*kf <= kmax) & (_jk*kf <= kmax) & (_lk*kf <= kmax))
        
        # plot dP0/dtheta and dP2/theta
        sub = plt.subplot(gs[2*i])
        sub.plot(kp[pklim], dp0[pklim], c='C0', label=r'$\ell = 0$')
        sub.plot(kp[pklim], dp2[pklim], c='C1', label=r'$\ell = 2$')

        sub.set_xscale('log') 
        sub.set_xlim(5e-3, kmax) 
        if i != len(thetas)-1: sub.set_xticklabels([]) 
        else: 
            sub.set_xlabel('k [$h$/Mpc]', fontsize=30) 
            if log: 
                sub.legend(handletextpad=0.2, loc='lower left', fontsize=15) 
            else: 
                sub.legend(handletextpad=0.2, loc='upper left', fontsize=15) 

        if not log: 
            sub.set_yscale('symlog', linthreshy=1e4) 
            sub.set_ylim(plims[i]) 
        else: 
            sub.set_ylim(logplims[i]) 
        if i == 5: 
            sub.set_ylabel(r'${\rm d} %s P_{g, \ell}/{\rm d}\theta$' % 
                    (['', '\log'][log]), fontsize=30) 

        # plot dB/dtheta
        sub = plt.subplot(gs[2*i+1])
        sub.plot(range(np.sum(bklim)), db[bklim])
        sub.set_xlim(0, np.sum(bklim)) 
        if not log: 
            #sub.set_yscale('symlog') 
            sub.set_yscale('symlog', linthreshy=1e8) 
        else: 
            sub.set_ylim(logblims[i]) 
        sub.text(0.99, 0.9, theta_lbls[i], ha='right', va='top', 
                transform=sub.transAxes, fontsize=25, bbox=dict(facecolor='white', alpha=0.75, edgecolor='None'))
        if i != len(thetas)-1: sub.set_xticklabels([]) 
        else: sub.set_xlabel('triangles', fontsize=30) 

        if i == 5: 
            sub.set_ylabel(r'${\rm d} %s B_{g, 0}/{\rm d}\theta$' %
                (['', '\log'][log]), fontsize=30) 

    ffig = os.path.join(dir_doc, 
            'quijote_d%sP02Bdtheta%s%s.%s.png' % 
            (['', 'log'][log], _seed_str(seed), _rsd_str(rsd), dmnu))
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') # latex friednly
    return None 


def plot_dP02B_Mnu_degen(kmax=0.5, seed='all', rsd='all', dmnu='fin'): 
    ''' compare derivatives that appear to be degenerate with the deriv. w.r.t. Mnu 
    '''
    fig = plt.figure(figsize=(30,8))
    gs = mpl.gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1,1,8], wspace=0.2) 
    sub0 = plt.subplot(gs[0]) 
    sub1 = plt.subplot(gs[1]) 
    sub2 = plt.subplot(gs[2]) 
    sub3 = plt.subplot(gs[3]) 
    sub4 = plt.subplot(gs[4]) 
    sub5 = plt.subplot(gs[5]) 
    
    # factor to scale bk deriv.s
    #fbks = [1., 0.1, 0.1]
    fbks = [1., 1., 1.]

    for i, tt in enumerate(['Mnu', 'logMmin', 'alpha']): 
        _k, _ik, _jk, _lk, dpb, _ = dP02Bk(tt, seed=seed, rsd=rsd, dmnu=dmnu, silent=False)
        
        nk0 = int(len(_k)/2) 
        kp  = _k[:nk0]
        dp0 = dpb[:nk0]
        dp2 = dpb[nk0:len(_k)]
        db  = dpb[len(_k):] 

        # k limits 
        pklim = (kp < kmax) 
        bklim = ((_ik*kf <= kmax) & (_jk*kf <= kmax) & (_lk*kf <= kmax))
        
        # plot dP0/dtheta and dP2/theta
        sub0.plot(kp[pklim], dp0[pklim], c='C%i' % i)
        sub1.plot(kp[pklim], dp2[pklim], c='C%i' % i)
        
        # plot dB/dtheta
        sub2.plot(range(np.sum(bklim)), fbks[i] * db[bklim], c='C%i' % i, lw=1, label=theta_lbls[thetas.index(tt)])
        if tt == 'Mnu': 
            dp0_mnu = dp0
            dp2_mnu = dp2
            db_mnu = db
        else: 
            #print('--- (dB/d%s)/(dB/dMnu) ---' % tt) 
            #print((db/db_mnu)[bklim][::10]) 
            sub3.plot(kp[pklim], (dp0/dp0_mnu)[pklim], c='C%i' % i) 
            sub4.plot(kp[pklim], (dp2/dp2_mnu)[pklim], c='C%i' % i) 
            sub5.plot(range(np.sum(bklim)), (db/db_mnu)[bklim], c='C%i' % i) 
    
    for i, sub in enumerate([sub0, sub1]):
        sub.set_xscale('log') 
        sub.set_xlim(5e-3, kmax) 
        sub.set_xticklabels([]) 
        sub.set_yscale('symlog', linthreshy=1e3) 
        sub.set_ylabel(r'${\rm d} P_%i/{\rm d}\theta$' % (2*i), fontsize=25) 

    for i, sub in enumerate([sub3, sub4]):
        sub.set_xlim(5e-3, kmax) 
        sub.set_xlabel('k [$h$/Mpc]', fontsize=25) 
        sub.set_ylabel(r'$({\rm d} P_%i/{\rm d}\theta)/({\rm d} P_%i/{\rm d}M_\nu)$' % (2*i, 2*i), fontsize=25) 

    sub2.set_xlim(0, np.sum(bklim)) 
    sub5.set_xlim(0, np.sum(bklim)) 
    sub2.set_ylim(-1e2, 5e11) 
    sub5.set_ylim(0, 50) 
    sub2.set_yscale('symlog', linthreshy=1e8) 
    sub2.set_xticklabels([]) 
    sub5.set_xlabel('triangles', fontsize=25) 
    sub2.set_ylabel(r'${\rm d} B_0/{\rm d}\theta$', fontsize=25) 
    sub5.set_ylabel(r'$({\rm d} B_0/{\rm d}\theta)/({\rm d} B_0/{\rm d}M_\nu)$', fontsize=25) 
    sub2.legend(handletextpad=0.1, loc='upper right', fontsize=20)

    ffig = os.path.join(dir_hod, 'figs', 
            'quijote_dP02Bdtheta_Mnu_degen%s%s.%s.png' % (_seed_str(seed), _rsd_str(rsd), dmnu))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def plot_dP02B_Mnu(kmax=0.5, seed='all', rsd='all'):
    ''' compare the derivative w.r.t. Mnu for different methods 
    '''
    fig = plt.figure(figsize=(30,4))
    gs = mpl.gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1,1,8], wspace=0.2) 
    sub0 = plt.subplot(gs[0]) 
    sub1 = plt.subplot(gs[1]) 
    sub2 = plt.subplot(gs[2]) 

    lstyles = ['-', '--'] 
    
    for i, dmnu in enumerate(['fin', 'p']):#, 'pp', 'fin_2lpt']): 
        _k, _ik, _jk, _lk, dpb, _ = dP02Bk('Mnu', seed=seed, rsd=rsd, dmnu=dmnu, silent=False)
        
        nk0 = int(len(_k)/2) 
        kp  = _k[:nk0]
        dp0 = dpb[:nk0]
        dp2 = dpb[nk0:len(_k)]
        db  = dpb[len(_k):] 

        # k limits 
        pklim = (kp < kmax) 
        bklim = ((_ik*kf <= kmax) & (_jk*kf <= kmax) & (_lk*kf <= kmax))
        
        # plot dP0/dtheta and dP2/theta
        sub0.plot(kp[pklim], dp0[pklim], c='C%i' % i, ls=lstyles[i])
        sub1.plot(kp[pklim], dp2[pklim], c='C%i' % i, ls=lstyles[i])
        
        # plot dB/dtheta
        sub2.plot(range(np.sum(bklim)), db[bklim], c='C%i' % i, lw=1,
                ls=lstyles[i], label=dmnu.replace('_', ' '))
    
    for i, sub in enumerate([sub0, sub1]):
        sub.set_xlabel('k [$h$/Mpc]', fontsize=25) 
        sub.set_xscale('log') 
        sub.set_xlim(5e-3, kmax) 
        sub.set_xticklabels([]) 
        sub.set_yscale('symlog', linthreshy=1e3) 
        sub.set_ylabel(r'${\rm d} P_%i/{\rm d}\theta$' % (2*i), fontsize=25) 

    sub2.set_xlim(0, np.sum(bklim)) 
    sub2.set_yscale('symlog', linthreshy=1e8) 
    sub2.set_xticklabels([]) 
    sub2.set_xlabel('triangles', fontsize=25) 
    sub2.set_ylabel(r'${\rm d} B_0/{\rm d}\theta$', fontsize=25) 
    sub2.legend(handletextpad=0.1, loc='upper right', fontsize=20, ncol=5)

    ffig = os.path.join(dir_hod, 'figs', 'quijote_dP02BdMnu%s%s.png' %
            (_seed_str(seed), _rsd_str(rsd)))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def dP02Bh(theta, log=False, rsd='all', flag='reg', dmnu='fin', fscale_pk=1., returnks=False, silent=True):
    ''' read in halo dP0/dtt, dP2/dtt, dB/dtt derivatives
    '''
    # dP0/dtheta, dP2/dtheta
    dir_dat = os.path.join(UT.doc_dir(), 'dat') 
    fdpk = os.path.join(dir_dat, 'dP02dtheta.%s%s%s.dat' % (theta, _rsd_str(rsd), _flag_str(flag))) 
    if not silent: print('--- reading %s ---' % fdpk) 
    i_dpk = 1 
    if theta == 'Mnu': 
        index_dict = {'fin': 1, 'fin0': 3, 'p': 5, 'pp': 7, 'ppp': 9, 'fin_2lpt': 11}  
        i_dpk = index_dict[dmnu] 
    if log: i_dpk += 1 

    _k, _dp = np.loadtxt(fdpk, skiprows=1, unpack=True, usecols=[0,i_dpk]) 

    # dB0/dtheta
    fdbk = os.path.join(dir_dat, 'dBdtheta.%s%s%s.dat' % (theta, _rsd_str(rsd), _flag_str(flag))) 
    if not silent: print('--- reading %s ---' % fdbk) 
    i_dbk = 3 
    if theta == 'Mnu': 
        index_dict = {'fin': 3, 'fin0': 5, 'p': 7, 'pp': 9, 'ppp': 11,  'fin_2lpt': 13}  
        i_dbk = index_dict[dmnu] 
    if log: i_dbk += 1 
    
    _ik, _jk, _lk, _db = np.loadtxt(fdbk, skiprows=1, unpack=True, usecols=[0,1,2,i_dbk]) 

    if returnks: 
        return _k, _ik, _jk, _lk, np.concatenate([fscale_pk * _dp, _db]) 
    else: 
        return np.concatenate([fscale_pk * _dp, _db]) 


def plot_dPBg_dPBh(theta, seed='all', rsd='all', dmnu='fin', log=False): 
    ''' compare derivatives w.r.t. the different thetas
    '''
    # y range of log dP02,B plots 
    logplims = [(-10., 10.), (-2, 15), (-3., 1.), (-4., 0.), (-2., 2.), (-0.5, 0.5)] 
    logblims = [(-13., 0.), (-2., 24.), (-5., -0.5), (-5., -0.5), (-5., 2.), (-0.5, 0.7)] 

    k, ik, jk, lk, _dpbg, _dlogpbg= dP02Bk(theta, seed=seed, rsd=rsd, dmnu=dmnu, silent=False)
    if log: dpbg = _dlogpbg
    else: dpbg = _dpbg
    #k, ik, jk, lk, dpbg = dP02Bk(theta, log=False, rsd=rsd, flag=flag, dmnu=dmnu, returnks=True, silent=False)
    #if log: 
    #    qhod_p = Obvs.quijhod_Pk('fiducial', flag=flag, rsd=rsd, silent=False) 
    #    p0g, p2g  = np.average(qhod_p['p0k'], axis=0), np.average(qhod_p['p2k'], axis=0)

    #    qhod_b = Obvs.quijhod_Bk('fiducial', flag=flag, rsd=rsd, silent=False) 
    #    b0g = np.average(qhod_b['b123'], axis=0) 
    #    pbg_fid = np.concatenate([p0g, p2g, b0g]) 
    #    dpbg /= pbg_fid

    dpbh = dP02Bh(theta, log=log, rsd=rsd, flag='reg', dmnu=dmnu, silent=False)
    k0 = k[:int(len(k)/2)]
    k2 = k[int(len(k)/2):]
    dp0g = dpbg[:int(len(k)/2)]
    dp2g = dpbg[int(len(k)/2):len(k)]
    db0g = dpbg[len(k):]
    dp0h = dpbh[:int(len(k)/2)]
    dp2h = dpbh[int(len(k)/2):len(k)]
    db0h = dpbh[len(k):]

    p0klim = (k0 < 0.5) 
    p2klim = (k2 < 0.5) 
    b0klim = ((ik*kf <= 0.5) & (jk*kf <= 0.5) & (lk*kf <= 0.5)) # k limit 

    fig = plt.figure(figsize=(30,3))
    gs = mpl.gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1,1,6], wspace=0.2) 
    sub0 = plt.subplot(gs[0]) 
    sub1 = plt.subplot(gs[1]) 
    sub2 = plt.subplot(gs[2]) 
    
    sub0.plot(k0[p0klim], dp0g[p0klim], c='k') 
    sub0.plot(k0[p0klim], dp0h[p0klim], c='C1') 
    sub0.set_xlabel('k', fontsize=25) 
    sub0.set_xscale('log') 
    sub0.set_xlim(5e-3, 0.5) 
    sub0.set_ylabel(r'${\rm d} %s P_0/{\rm d}\theta$' % (['', '\log'][log]), fontsize=20)
    if not log: sub0.set_yscale('symlog', linthreshy=1e3) 
    else: sub0.set_ylim(logplims[thetas.index(theta)]) 
    
    sub1.plot(k2[p2klim], dp2g[p2klim], c='k') 
    sub1.plot(k2[p2klim], dp2h[p2klim], c='C1') 
    sub1.set_xlabel('k', fontsize=25) 
    sub1.set_xscale('log') 
    sub1.set_xlim(5e-3, 0.5) 
    sub1.set_ylabel(r'${\rm d} %s P_2/{\rm d}\theta$' % (['', '\log'][log]), fontsize=20)
    if not log: sub1.set_yscale('symlog', linthreshy=1e3) 
    else: sub1.set_ylim(logplims[thetas.index(theta)]) 

    sub2.plot(range(np.sum(b0klim)), db0g[b0klim], c='k', label='galaxy') 
    sub2.plot(range(np.sum(b0klim)), db0h[b0klim], c='C1', label='halo') 
    sub2.legend(loc='upper left', fontsize=20, ncol=2) 
    sub2.set_xlabel('triangles', fontsize=25) 
    sub2.set_xlim(0, np.sum(b0klim)) 
    sub2.set_ylabel(r'${\rm d} %s B_0/{\rm d}\theta$' % (['', '\log'][log]), fontsize=20)
    if not log: 
        if theta == 'Mnu': 
            sub2.set_ylim(1e5, 5e10)
            sub2.set_yscale('log')
        else: sub2.set_yscale('symlog', linthreshy=1e8) 
    else: sub2.set_ylim(logblims[thetas.index(theta)]) 
    sub2.text(0.99, 0.9, theta_lbls[thetas.index(theta)], ha='right', va='top', 
            transform=sub2.transAxes, fontsize=25, bbox=dict(facecolor='white', alpha=0.75, edgecolor='None'))
    
    ffig = os.path.join(dir_hod, 'figs', 
            'd%sPBg_d%sPBh.%s%s%s.png' % (['', 'log'][log], ['', 'log'][log],
                theta, _seed_str(seed), _rsd_str(rsd)))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def _plot_dP02B_Mnu_Mmin(kmax=0.5, rsd='all', flag='reg', dmnu='fin'):
    ''' compare the derivative w.r.t. Mnu and Mmin 
    '''
    fig = plt.figure(figsize=(30,8))
    gs = mpl.gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1,1,8],
            height_ratios=[1,1], wspace=0.2) 
    sub0 = plt.subplot(gs[0]) 
    sub1 = plt.subplot(gs[1]) 
    sub2 = plt.subplot(gs[2]) 
    sub3 = plt.subplot(gs[3]) 
    sub4 = plt.subplot(gs[4]) 
    sub5 = plt.subplot(gs[5]) 
    
    _k, _ik, _jk, _lk, dpb_mnu = dP02Bk('Mnu', log=False, rsd=rsd, flag=flag, dmnu=dmnu, returnks=True, silent=False)
    _, _, _, _, dpb_mmin = dP02Bk('logMmin', log=False, rsd=rsd, flag=flag, dmnu=dmnu, returnks=True, silent=False)
        
    for i, dpb in enumerate([dpb_mnu, dpb_mmin]): 
        nk0 = int(len(_k)/2) 
        kp  = _k[:nk0]
        dp0 = dpb[:nk0]
        dp2 = dpb[nk0:len(_k)]
        db  = dpb[len(_k):] 

        # k limits 
        pklim = (kp < kmax) 
        bklim = ((_ik*kf <= kmax) & (_jk*kf <= kmax) & (_lk*kf <= kmax))
        
        # plot dP0/dtheta and dP2/theta
        sub0.plot(kp[pklim], dp0[pklim], c='C%i' % i)
        sub1.plot(kp[pklim], dp2[pklim], c='C%i' % i)
        
        # plot dB/dtheta
        sub2.plot(range(np.sum(bklim)), db[bklim], c='C%i' % i, lw=1,
                label=[r'$M_\nu$', r'$\log M_{\rm min}$'][i])

    sub3.plot(kp[pklim], (dpb_mmin/dpb_mnu)[:nk0][pklim])
    sub4.plot(kp[pklim], (dpb_mmin/dpb_mnu)[nk0:len(_k)][pklim])
    sub5.plot(range(np.sum(bklim)), (dpb_mmin/dpb_mnu)[len(_k):][bklim])
        
    for i, sub in enumerate([sub0, sub1, sub3, sub4]):
        sub.set_xscale('log') 
        sub.set_xlim(5e-3, kmax) 
        sub.set_xticklabels([]) 
        if i > 1: 
            sub.set_xlabel('k [$h$/Mpc]', fontsize=25) 
            sub.set_ylabel(
                    r'$({\rm d} P_%i/{\rm d}\log M_{\rm min})/({\rm d} P_%i/{\rm d}M_\nu)$' % 
                    (2*(i % 2), 2*(i % 2)), fontsize=25) 
        else: 
            sub.set_ylabel(r'${\rm d} P_%i/{\rm d}\theta$' % (2*(i % 2)), fontsize=25) 
            sub.set_yscale('symlog', linthreshy=1e3) 
    sub3.set_ylim(5, 36) 
    sub4.set_ylim(0, 30) 

    sub2.set_xlim(0, np.sum(bklim)) 
    sub2.set_yscale('symlog', linthreshy=1e8) 
    sub2.set_xticklabels([]) 
    sub2.set_ylabel(r'${\rm d} B_0/{\rm d}\theta$', fontsize=25) 
    sub2.legend(handletextpad=0.1, loc='upper right', fontsize=20, ncol=5)

    sub5.set_xlim(0, np.sum(bklim)) 
    sub5.set_xticklabels([]) 
    sub5.set_ylim(5, 35)
    sub5.set_xlabel('triangles', fontsize=25) 
    sub5.set_ylabel(r'$({\rm d} B_0/{\rm d}\log M_{\rm min})/({\rm d} B_0/{\rm d}M_\nu)$', fontsize=25) 

    ffig = os.path.join(dir_doc, '_dP02Bg_Mnu_%s_logMmin%s%s.png' % (dmnu, _rsd_str(rsd), _flag_str(flag)))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 

# --- power/bispectrum --- 
def nbar(): 
    ''' galaxy number density for the fiducial HOD parameters at fiducial
    cosmology. Also calculate number density of halos. 
    '''
    hod = Obvs.quijhod_Pk('fiducial', seed=0, flag='reg', rsd='all',
            silent=True) 
    nbar_g = 1./hod['p_sn']
    print('median galaxy number density = %.5e' % np.median(nbar_g)) 

    halo = Obvs.quijotePk('fiducial', flag='reg', rsd='all', silent=True)
    nbar_h = 1./halo['p_sn']
    print('median halo number density = %.5e' % np.median(nbar_h)) 
    return None 


def plot_PBg(rsd='all'): 
    ''' plot Pg0, Pg2, Bg0 with Ph0, Ph2, Bh0 included for reference. 
    '''
    # read in P0g, P2g, and Bg
    qhod_p = Obvs.quijhod_Pk('fiducial', seed=0, rsd=rsd, flag='reg', silent=False) 
    k, p0g, p2g  = qhod_p['k'], np.average(qhod_p['p0k'], axis=0), np.average(qhod_p['p2k'], axis=0)
    print('N_galaxies = %i' % np.average(qhod_p['Ngalaxies']))

    qhod_b = Obvs.quijhod_Bk('fiducial', seed=0, rsd=rsd, flag='reg', silent=False) 
    i_k, j_k, l_k, b0g = qhod_b['k1'], qhod_b['k2'], qhod_b['k3'], np.average(qhod_b['b123'], axis=0) 

    # read in P0h, P2h, and Bh
    quij_p = Obvs.quijotePk('fiducial', rsd=rsd, flag='reg', silent=False) 
    p0h, p2h = np.average(quij_p['p0k'], axis=0), np.average(quij_p['p2k'], axis=0)
    print('N_halos = %i' % np.average(quij_p['Nhalos']))

    quij_b = Obvs.quijoteBk('fiducial', rsd=rsd, flag='reg', silent=False) 
    b0h = np.average(quij_b['b123'], axis=0) 

    pklim = (k < 0.5) 
    bklim = ((i_k*kf <= 0.5) & (j_k*kf <= 0.5) & (l_k*kf <= 0.5)) # k limit 

    fig = plt.figure(figsize=(25,5))
    gs = mpl.gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1,5], wspace=0.15) 
    sub0 = plt.subplot(gs[0]) 
    sub1 = plt.subplot(gs[1]) 

    sub0.plot(k[pklim], p0h[pklim], c='k', lw=0.5) 
    sub0.plot(k[pklim], p2h[pklim], c='k', lw=0.5, ls='--') 

    sub0.plot(k[pklim], p0g[pklim], c='C0', label=r'$\ell = 0$')
    sub0.plot(k[pklim], np.abs(p2g[pklim]), c='C0', ls='--', label=r'$\ell = 2$')
    
    sub0.legend(loc='lower left', handletextpad=0.2, fontsize=20) 
    sub0.set_xlabel('k [$h$/Mpc]', fontsize=25) 
    sub0.set_xscale('log') 
    sub0.set_xlim(5e-3, 0.5) 
    sub0.set_ylabel('$|P_{g, \ell}(k)|$', fontsize=25) 
    sub0.set_yscale('log') 
    
    _bhplt, = sub1.plot(range(np.sum(bklim)), b0h[bklim], c='k', ls=':') 
    _bgplt, = sub1.plot(range(np.sum(bklim)), b0g[bklim], c='C0') 
    sub1.legend([_bgplt, _bhplt], [r'galaxy', r'halo'], loc='upper right', 
            handletextpad=0.3, fontsize=20) 
    sub1.set_xlabel('triangle configurations', fontsize=25) 
    sub1.set_xlim(0, np.sum(bklim)) 
    sub1.set_ylabel('$B_{g, 0}(k_1, k_2, k_3)$', fontsize=25) 
    sub1.set_yscale('log') 
    sub1.set_ylim(1e6, 5e10)
    
    ffig = os.path.join(dir_doc, 'PBg%s.png' % (_rsd_str(rsd)))
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') # latex version 
    return None 


def plot_bias(rsd='all'): 
    '''plot bias by comparing the ratio of the galaxy and halo power spectrum
    monopole (P0g, P0h) to the matter power spectrum (Pm) 
    '''
    from emanu import lineartheory as LT 
    # read in P0g
    qhod_p = Obvs.quijhod_Pk('fiducial', seed=0, rsd=rsd, silent=False) 
    k, p0g = qhod_p['k'], np.average(qhod_p['p0k'], axis=0)
    # read in P0h
    quij_p = Obvs.quijotePk('fiducial', rsd=rsd, flag='reg', silent=False) 
    p0h = np.average(quij_p['p0k'], axis=0)
    # read in Pm 
    plin = LT._Pm_Mnu(0., k) 

    # k limits 
    pklim = (k < 0.5) 
    # --- plotting ---
    fig = plt.figure(figsize=(6,6))
    sub = fig.add_subplot(111)
    
    sub.plot(k[pklim], np.sqrt(p0g[pklim]/plin[pklim]), c='C0', ls='-') 
    sub.plot(k[pklim], np.sqrt(p0h[pklim]/plin[pklim]), c='C1', ls='-') 
    sub.plot([5e-3, 0.5], [2.55, 2.55], c='k', ls='--', label=r'$b_g = 2.55$') 
    sub.plot([5e-3, 0.5], [1.85, 1.85], c='k', ls=':', label=r'$b_h = 1.85$') 
    sub.set_xlabel('k', fontsize=25) 
    sub.set_xscale('log') 
    sub.set_xlim(5e-3, 0.5) 
    sub.set_ylabel(r'$\sqrt{\frac{P_{g,0}}{P_m}(k)}$', fontsize=20) 
    sub.set_ylim(1.5, 3.5)
    sub.legend(loc='upper left', fontsize=25) 

    ffig = os.path.join(dir_doc, 'galaxy_bias%s.png' % (_rsd_str(rsd)))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def _PBg_rsd_comparison(flag='reg'): 
    ''' compare Pg0, Pg2, Bg0 for different RSD 
    '''
    for theta in ['Mnu_p', 'Mnu_pp', 'Mnu_ppp', 'Om_m', 'Om_p', 'Ob2_m',
            'Ob2_p', 'h_m', 'h_p', 'ns_m', 'ns_p', 's8_m', 's8_p', 'alpha_m',
            'alpha_p', 'logM0_m', 'logM0_p', 'logM1_m', 'logM1_p', 'logMmin_m',
            'logMmin_p', 'sigma_logM_m', 'sigma_logM_p', 'fiducial']:
        fig = plt.figure(figsize=(30,3))
        gs = mpl.gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1,6], wspace=0.15) 
        sub0 = plt.subplot(gs[0]) 
        sub1 = plt.subplot(gs[1]) 

        for i, rsd, clr in zip(range(4), ['all', 0, 1, 2], ['k', 'C0', 'C1', 'C2']): 
            # read in P0g, P2g, and Bg
            qhod_p = Obvs.quijhod_Pk(theta, flag=flag, rsd=rsd, silent=False) 
            k, p0g, p2g  = qhod_p['k'], np.average(qhod_p['p0k'], axis=0), np.average(qhod_p['p2k'], axis=0)

            qhod_b = Obvs.quijhod_Bk(theta, flag=flag, rsd=rsd, silent=False) 
            i_k, j_k, l_k, b0g = qhod_b['k1'], qhod_b['k2'], qhod_b['k3'], np.average(qhod_b['b123'], axis=0) 

            pklim = (k < 0.5) 
            bklim = ((i_k*kf <= 0.5) & (j_k*kf <= 0.5) & (l_k*kf <= 0.5)) # k limit 

            sub0.plot(k[pklim], p0g[pklim], c=clr, label=r'$\ell = 0$')
            sub0.plot(k[pklim], np.abs(p2g[pklim]), c=clr, ls='--', label=r'$\ell = 2$')
            
            if i == 0: 
                bk_ref = b0g[bklim]
            _bgplt, = sub1.plot(range(np.sum(bklim)), b0g[bklim]/bk_ref, c=clr) 
            
            if i == 0: 
                sub0.legend(loc='lower left', handletextpad=0.1, fontsize=15) 
        sub0.set_xlabel('k [$h$/Mpc]', fontsize=25) 
        sub0.set_xscale('log') 
        sub0.set_xlim(5e-3, 0.5) 
        sub0.set_ylabel('$|P^g_\ell(k)|$', fontsize=25) 
        sub0.set_yscale('log') 
        
        #sub1.legend([_bgplt], [r'galaxy $P_\ell, B_0$'], loc='upper right', handletextpad=0.2, fontsize=15) 
        sub1.set_xlabel('triangle configurations', fontsize=25) 
        sub1.set_xlim(0, np.sum(bklim)) 
        sub1.set_ylabel(r'$B^g_{\rm RSD}(k_1, k_2, k_3)/B^g$', fontsize=25) 
        sub1.set_ylim(0.9, 1.1) 
        
        ffig = os.path.join(dir_hod, 'figs', 
                'PBg.%s%s.rsd_comparison.png' % (theta, _flag_str(flag)))
        fig.savefig(ffig, bbox_inches='tight') 
        plt.close()  
    return None 


def plot_PBg_PBh(theta, rsd='all'): 
    ''' comparison between the galaxy power/bispectrum to the halo power/bispectrum
    '''
    seed = 0 
    if theta in ['fiducial', 'fiducial_za']: 
        _thetas = [theta] 
    elif theta == 'Mnu': 
        _thetas = ['fiducial_za', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp'] 
    else: 
        _thetas = [theta+'_m', theta+'_p']

    p0gs, p2gs, b0gs = [], [], [] 
    p0hs, p2hs, b0hs = [], [], [] 

    for tt in _thetas: 
        # read in P0g, P2g, and Bg
        qhod_p = Obvs.quijhod_Pk(tt, seed=seed, rsd=rsd, silent=False) 
        k, p0g, p2g  = qhod_p['k'], np.average(qhod_p['p0k'], axis=0), np.average(qhod_p['p2k'], axis=0)

        qhod_b = Obvs.quijhod_Bk(tt, seed=seed, rsd=rsd, silent=False) 
        i_k, j_k, l_k, b0g = qhod_b['k1'], qhod_b['k2'], qhod_b['k3'], np.average(qhod_b['b123'], axis=0) 

        # read in P0h, P2h, and Bh
        quij_p = Obvs.quijotePk(tt, rsd=rsd, flag='reg', silent=False) 
        p0h, p2h = np.average(quij_p['p0k'], axis=0), np.average(quij_p['p2k'], axis=0)

        quij_b = Obvs.quijoteBk(tt, rsd=rsd, flag='reg', silent=False) 
        b0h = np.average(quij_b['b123'], axis=0) 

        p0gs.append(p0g)
        p2gs.append(p2g)
        b0gs.append(b0g)
        p0hs.append(p0h)
        p2hs.append(p2h)
        b0hs.append(b0h)

    pklim = (k < 0.5) 
    bklim = ((i_k*kf <= 0.5) & (j_k*kf <= 0.5) & (l_k*kf <= 0.5)) # k limit 

    fig = plt.figure(figsize=(30,3))
    gs = mpl.gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1,1,6], wspace=0.2) 
    sub0 = plt.subplot(gs[0]) 
    sub1 = plt.subplot(gs[1]) 
    sub2 = plt.subplot(gs[2]) 
    
    _ls = ['-', ':', '-.', '--'] 
    for i in range(len(p0gs)): 
        sub0.plot(k[pklim], p0gs[i][pklim], c='k', ls=_ls[i]) 
        sub0.plot(k[pklim], p0hs[i][pklim], c='C1', ls=_ls[i]) 
    sub0.set_xlabel('k', fontsize=25) 
    sub0.set_xscale('log') 
    sub0.set_xlim(5e-3, 0.5) 
    sub0.set_ylabel('$P_0(k)$', fontsize=20) 
    sub0.set_yscale('log') 
    
    for i in range(len(p0gs)): 
        sub1.plot(k[pklim], p2gs[i][pklim], c='k', ls=_ls[i]) 
        sub1.plot(k[pklim], p2hs[i][pklim], c='C1', ls=_ls[i]) 
    sub1.set_xlabel('k', fontsize=25) 
    sub1.set_xscale('log') 
    sub1.set_xlim(5e-3, 0.5) 
    sub1.set_ylabel('$P_2(k)$', fontsize=20) 
    sub1.set_yscale('log') 
    
    for i in range(len(p0gs)): 
        sub2.plot(range(np.sum(bklim)), b0gs[i][bklim], c='k', ls=_ls[i], label='galaxy') 
        sub2.plot(range(np.sum(bklim)), b0hs[i][bklim], c='C1', ls=_ls[i], label='halo') 
        if i == 0: sub2.legend(loc='upper right', fontsize=20, ncol=2) 
    sub2.set_xlabel('triangles', fontsize=25) 
    sub2.set_xlim(0, np.sum(bklim)) 
    sub2.set_ylabel('$B(k_1, k_2, k_3)$', fontsize=20) 
    sub2.set_yscale('log') 
    
    ffig = os.path.join(dir_hod, 'PBg_PBh.%s%s%s.png' % (theta, _seed_str(seed), _rsd_str(rsd)))
    fig.savefig(ffig, bbox_inches='tight') 
    if theta == 'fiducial':  return None 

    fig = plt.figure(figsize=(30,3))
    gs = mpl.gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1,1,6], wspace=0.2) 
    sub0 = plt.subplot(gs[0]) 
    sub1 = plt.subplot(gs[1]) 
    sub2 = plt.subplot(gs[2]) 

    for i in range(1, len(p0gs)): 
        sub0.plot(k[pklim], p0gs[i][pklim] - p0gs[0][pklim], c='k', ls=_ls[i]) 
        sub0.plot(k[pklim], p0hs[i][pklim] - p0hs[0][pklim], c='C1', ls=_ls[i]) 
    sub0.set_xlabel('k', fontsize=25) 
    sub0.set_xscale('log') 
    sub0.set_xlim(5e-3, 0.5) 
    sub0.set_ylabel('$\Delta P_0(k)$', fontsize=20) 
    sub0.set_yscale('symlog') 
    
    for i in range(1, len(p0gs)): 
        sub1.plot(k[pklim], p2gs[i][pklim] - p2gs[0][pklim], c='k', ls=_ls[i]) 
        sub1.plot(k[pklim], p2hs[i][pklim] - p2hs[0][pklim], c='C1', ls=_ls[i]) 
    sub1.set_xlabel('k', fontsize=25) 
    sub1.set_xscale('log') 
    sub1.set_xlim(5e-3, 0.5) 
    sub1.set_ylabel('$\Delta P_2(k)$', fontsize=20) 
    sub1.set_yscale('symlog') 
    
    for i in range(1, len(p0gs)): 
        sub2.plot(range(np.sum(bklim)), b0gs[i][bklim] - b0gs[0][bklim], c='k', ls=_ls[i], label='galaxy') 
        sub2.plot(range(np.sum(bklim)), b0hs[i][bklim] - b0hs[0][bklim], c='C1', ls=_ls[i], label='halo') 
        if i == 1: sub2.legend(loc='upper right', fontsize=20, ncol=2) 
    sub2.set_xlabel('triangles', fontsize=25) 
    sub2.set_xlim(0, np.sum(bklim)) 
    sub2.set_ylabel('$\Delta B(k_1, k_2, k_3)$', fontsize=20) 
    if theta == 'Mnu': sub2.set_yscale('log') 
    else: sub2.set_yscale('symlog', linthreshy=1e6) 
    
    ffig = os.path.join(dir_hod, 'figs', 
            'dPBg_PBh.%s%s%s.png' % (theta, _seed_str(seed), _rsd_str(rsd)))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def _plot_PBg_PBh_SN(rsd='all'): 
    ''' comparison between the Pg, Bg, Pg_sn, Bg_sn and Ph, Bh, Ph_sn, Bh_sn
    '''
    # read in P0g, P2g, and Bg
    qhod_p = Obvs.quijhod_Pk('fiducial', seed=0, rsd=rsd, flag='reg', silent=False) 
    k, p0g, p2g = qhod_p['k'], np.average(qhod_p['p0k'], axis=0), np.average(qhod_p['p2k'], axis=0)
    pg_uncorr = np.average(qhod_p['p0k'] + qhod_p['p_sn'][:,np.newaxis], axis=0)
    pg_sn = np.average(qhod_p['p_sn'], axis=0)
    
    print('N_galaxies = %i' % np.average(qhod_p['Ngalaxies']))

    qhod_b = Obvs.quijhod_Bk('fiducial', seed=0, rsd=rsd, flag='reg', silent=False) 
    i_k, j_k, l_k, b0g = qhod_b['k1'], qhod_b['k2'], qhod_b['k3'], np.average(qhod_b['b123'], axis=0) 
    bg_uncorr = np.average(qhod_b['b123'] + qhod_b['b_sn'], axis=0) 
    bg_sn = np.average(qhod_b['b_sn'], axis=0) 

    # read in P0h, P2h, and Bh
    quij_p = Obvs.quijotePk('fiducial', rsd=rsd, flag='reg', silent=False) 
    p0h, p2h = np.average(quij_p['p0k'], axis=0), np.average(quij_p['p2k'], axis=0)
    ph_uncorr = np.average(quij_p['p0k'] + quij_p['p_sn'][:,np.newaxis], axis=0) 
    ph_sn = np.average(quij_p['p_sn'], axis=0) 
    
    print('N_halos = %i' % np.average(quij_p['Nhalos']))

    quij_b = Obvs.quijoteBk('fiducial', rsd=rsd, flag='reg', silent=False) 
    b0h = np.average(quij_b['b123'], axis=0) 
    bh_uncorr = np.average(quij_b['b123'] + quij_b['b_sn'], axis=0) 
    bh_sn = np.average(quij_b['b_sn'], axis=0) 
    
    # k limits 
    pklim = (k < 0.5) 
    bklim = ((i_k*kf <= 0.5) & (j_k*kf <= 0.5) & (l_k*kf <= 0.5)) # k limit 
    
    # --- plotting ---
    fig = plt.figure(figsize=(30,7))
    gs = mpl.gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1,1,6], wspace=0.2) 
    sub0 = plt.subplot(gs[0]) 
    sub1 = plt.subplot(gs[1]) 
    sub2 = plt.subplot(gs[2]) 
    sub3 = plt.subplot(gs[3]) 
    sub4 = plt.subplot(gs[4]) 
    sub5 = plt.subplot(gs[5]) 
    
    sub0.plot(k[pklim], pg_uncorr[pklim], c='k', ls=':') 
    sub0.plot(k[pklim], p0g[pklim], c='k', ls='-') 
    sub0.plot(k[pklim], np.repeat(pg_sn, np.sum(pklim)), c='C0', ls='--') 
    sub0.set_xscale('log') 
    sub0.set_xlim(5e-3, 0.5) 
    sub0.set_xticklabels([]) 
    sub0.set_ylabel('$P_{g,0}(k)$', fontsize=20) 
    sub0.set_yscale('log') 
    sub0.set_ylim(1e3, 2e5)

    sub3.plot(k[pklim], ph_uncorr[pklim], c='k', ls=':') 
    sub3.plot(k[pklim], p0h[pklim], c='k', ls='-') 
    sub3.plot(k[pklim], np.repeat(ph_sn, np.sum(pklim)), c='C0', ls='--') 
    sub3.set_xlabel('k', fontsize=25) 
    sub3.set_xscale('log') 
    sub3.set_xlim(5e-3, 0.5) 
    sub3.set_ylabel('$P_{h,0}(k)$', fontsize=20) 
    sub3.set_yscale('log') 
    sub3.set_ylim(1e3, 2e5)
    
    sub1.plot(k[pklim], p2g[pklim], c='k') 
    sub1.set_xscale('log') 
    sub1.set_xlim(5e-3, 0.5) 
    sub1.set_xticklabels([]) 
    sub1.set_ylabel('$P_{g,2}(k)$', fontsize=20) 
    sub1.set_yscale('log') 
    sub1.set_ylim(5e2, 6e4)

    sub4.plot(k[pklim], p2h[pklim], c='k') 
    sub4.set_xlabel('k', fontsize=25) 
    sub4.set_xscale('log') 
    sub4.set_xlim(5e-3, 0.5) 
    sub4.set_ylabel('$P_{h,2}(k)$', fontsize=20) 
    sub4.set_yscale('log') 
    sub4.set_ylim(5e2, 6e4)
    
    sub2.plot(range(np.sum(bklim)), bg_uncorr[bklim], c='k', ls=':', 
            label='SN uncorr.') 
    sub2.plot(range(np.sum(bklim)), b0g[bklim], c='k', ls='-') 
    sub2.plot(range(np.sum(bklim)), bg_sn[bklim], c='C0', ls='--', label='shot noise') 
    sub2.plot(range(np.sum(bklim)), np.repeat(pg_sn**2, np.sum(bklim)), c='C1',
            ls=':', lw=0.5, label=r'$1/\bar{n}_g^2$') 
    sub2.legend(loc='upper right', fontsize=20, ncol=2) 
    sub2.set_xlim(0, np.sum(bklim)) 
    sub2.set_xticklabels([]) 
    sub2.set_ylabel('$B_g(k_1, k_2, k_3)$', fontsize=20) 
    sub2.set_yscale('log') 
    sub2.set_ylim(1e7, 7e10)

    sub5.plot(range(np.sum(bklim)), bh_uncorr[bklim], c='k', ls=':') 
    sub5.plot(range(np.sum(bklim)), b0h[bklim], c='k', ls='-') 
    sub5.plot(range(np.sum(bklim)), bh_sn[bklim], c='C0', ls='--') 
    sub5.plot(range(np.sum(bklim)), np.repeat(ph_sn**2, np.sum(bklim)), c='C1',
            ls=':', lw=0.5) 
    sub5.set_xlabel('triangles', fontsize=25) 
    sub5.set_xlim(0, np.sum(bklim)) 
    sub5.set_ylabel('$B_h(k_1, k_2, k_3)$', fontsize=20) 
    sub5.set_yscale('log') 
    sub5.set_ylim(1e7, 7e10)
    
    ffig = os.path.join(dir_hod, 'figs', '_PBg_PBh_SN%s%s.png' % (_rsd_str(rsd), _flag_str(flag)))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 

# --- forecasts --- 
def FisherMatrix(obs, kmax=0.5, seed=range(5), rsd='all', dmnu='fin',
        params='default', theta_nuis=None, cross_covariance=True,
        diag_covariance=False, silent=True): 
    ''' calculate fisher matrix for parameters ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu'] 
    and specified nuisance parameters
    
    :param obs: 
        observable ('p02k', 'bk', 'p02bk') 
    :param kmax: (default: 0.5) 
        kmax of the analysis 
    :param rsd: (default: True) 
        rsd kwarg that specifies rsd set up for B(k) deriv.. 
        If rsd == 'all', include 3 RSD directions. 
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

    if obs == 'p02k':  # monopole and quadrupole 
        k, _Cov, nmock = p02kCov(rsd=2, flag='reg') # only reg. N-body and 1 RSD direction (z) 
        klim = (k <= kmax) # determine k limit 
        Cov = _Cov[klim,:][:,klim]
    elif obs == 'bk':
        i_k, j_k, l_k, _Cov, nmock = bkCov(rsd=2, flag='reg') # only N-body for covariance
        klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) # k limit 
        Cov = _Cov[:,klim][klim,:]
    elif obs == 'p02bk': 
        ks, _Cov, nmock = p02bkCov(rsd=2, flag='reg')
        k, i_k, j_k, l_k = ks 

        pklim = (k <= kmax) 
        bklim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) # k limit 
        klim = np.concatenate([pklim, bklim]) 
        Cov = _Cov[:,klim][klim,:]
        if not cross_covariance: 
            # neglect cross covariance between Pell and B 
            _Cov_ = np.zeros((np.sum(klim), np.sum(klim)))
            _Cov_[:np.sum(pklim),:np.sum(pklim)] = Cov[:np.sum(pklim),:np.sum(pklim)]
            _Cov_[np.sum(pklim):,np.sum(pklim):] = Cov[np.sum(pklim):,np.sum(pklim):]
            Cov = _Cov_.copy() 
            print('no cross covariance') 
            print(Cov[np.sum(pklim)-2:np.sum(pklim)+2,np.sum(pklim)-2:np.sum(pklim)+2])
            print('covariance matrix condition number = %.5e' % np.linalg.cond(Cov)) 
        if diag_covariance: 
            Cov = np.identity(np.sum(klim)) * np.diag(Cov)
            print('only diagonal') 
            print(Cov[np.sum(pklim)-2:np.sum(pklim)+2,np.sum(pklim)-2:np.sum(pklim)+2])
            print('covariance matrix condition number = %.5e' % np.linalg.cond(Cov)) 
    else: 
        raise NotImplementedError
    #print('%iD data vector' % Cov.shape[0])
    ndata = np.sum(klim) 
    f_hartlap = float(nmock - ndata - 2)/float(nmock - 1) 
    C_inv = f_hartlap * np.linalg.inv(Cov) # invert the covariance 
    
    if params == 'default': # default set of parameters 
        _thetas = ['Om', 'Ob2', 'h', 'ns', 's8', 'Mnu', 'logMmin', 'sigma_logM', 'logM0', 'alpha', 'logM1'] 
    elif params == 'default_hr': # default set of parameters 
        _thetas = ['Om', 'Ob2', 'h', 'ns', 's8', 'Mnu', 'logMmin', 'sigma_logM_HR', 'logM0', 'alpha', 'logM1'] 
    elif params == 'lcdm': 
        _thetas = ['Om', 'Ob2', 'h', 'ns', 's8', 'logMmin', 'sigma_logM', 'logM0', 'alpha', 'logM1'] 
    elif params == 'halo':  # same parameters as halo case
        _thetas = ['Om', 'Ob2', 'h', 'ns', 's8', 'Mnu', 'logMmin'] 
    else: 
        raise NotImplementedError
    if theta_nuis is not None: _thetas += theta_nuis 
    
    # calculate the derivatives along all the thetas 
    dobs_dt = [] 
    for par in _thetas: 
        if obs == 'p02k': 
            _, dobs_dti, _ = dP02k(par, seed=seed, rsd=rsd, dmnu=dmnu,
                    silent=silent)
        elif obs == 'bk': 
            # rsd and flag kwargs are passed to the derivatives
            _, _, _, dobs_dti, _ = dBk(par, seed=seed, rsd=rsd, dmnu=dmnu, silent=silent)
        elif obs == 'p02bk': 
            _, _, _, _, dobs_dti, _ = dP02Bk(par, seed=seed, rsd=rsd,
                    dmnu=dmnu, fscale_pk=fscale_pk, silent=silent) 
        dobs_dt.append(dobs_dti[klim])
            
    Fij = Forecast.Fij(dobs_dt, C_inv) 
    return Fij 


def forecast(obs, kmax=0.5, rsd='all', flag='reg', dmnu='fin',
        params='default', theta_nuis=None, planck=False):
    ''' fisher forecast for quijote observables where we marginalize over theta_nuis parameters. 
    
    :param obs: 
        observable ('p02k', 'bk', 'p02bk') 
    :param kmax: (default: 0.5) 
        kmax of the analysis 
    :param rsd: (default: True) 
        specifies the rsd set up for the B(k) derivative. 
        If rsd == 'all', include 3 RSD directions. 
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
    if params == 'default': 
        _thetas = copy(thetas) 
        _theta_lbls = copy(theta_lbls) 
        _theta_dlims = [0.07, 0.03, 0.4, 0.35, 0.04, 0.4, 0.35, 0.15, 1., 0.6, 1.]
        if obs == 'bk': 
            _theta_dlims = [0.02, 0.005, 0.06, 0.05, 0.04, 0.05, 0.125, 0.15, 0.4, 0.12, 0.2]
    elif params == 'lcdm': 
        _thetas = ['Om', 'Ob2', 'h', 'ns', 's8', 'logMmin', 'sigma_logM', 'logM0', 'alpha', 'logM1'] 
        _theta_lbls = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', 
                r'$\log M_{\rm min}$', r'$\sigma_{\log M}$', r'$\log M_0$', r'$\alpha$', r'$\log M_1$']
        #_theta_lims = [(0.25, 0.385), (0.02, 0.08), (0.3, 1.1), (0.6, 1.3), (0.8, 0.88), (-0.4, 0.4), (13.3, 14.), [0.1, 0.3], [13., 15.], [0.5, 1.7], [13., 15.]]
        _theta_dlims = [0.07, 0.03, 0.4, 0.35, 0.04, 0.35, 0.15, 1., 0.6, 1.]
    else: 
        raise NotImplementedError

    _theta_fid = theta_fid.copy() # fiducial thetas

    # fisher matrix (Fij)
    _Fij    = FisherMatrix(obs, kmax=kmax, rsd=rsd, flag=flag, dmnu=dmnu,
            params=params, theta_nuis=None) # no nuisance param. 
    Fij     = FisherMatrix(obs, kmax=kmax, rsd=rsd, flag=flag, dmnu=dmnu,
            params=params, theta_nuis=theta_nuis) # marg. over nuisance param. 
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
    
    if params == 'default': 
        i_Mnu = thetas.index('Mnu')
        print('--- i = Mnu ---')
        print('n nuis. Fii=%f, sigma_i = %f' % (_Fij[i_Mnu,i_Mnu], np.sqrt(_Finv[i_Mnu,i_Mnu])))
        print("y nuis. Fii=%f, sigma_i = %f" % (Fij[i_Mnu,i_Mnu], np.sqrt(Finv[i_Mnu,i_Mnu])))
        if planck: 
            print("w/ Planck2018")
            print("y nuis. Fii=%f, sigma_i = %f" % (Fij_planck[i_Mnu,i_Mnu], np.sqrt(Finv[i_Mnu,i_Mnu])))
    print('--- thetas ---')
    print('               %s' % ',    \t'.join(_thetas))
    print('n nuis. sigmas %s' % ', '.join(['%.2e' % sii for sii in np.sqrt(np.diag(_Finv))]))
    print('y nuis. sigmas %s' % ', '.join(['%.2e' % sii for sii in np.sqrt(np.diag(Finv))]))
    if planck: 
        print("w/ Planck2018")
        print('y nuis. sigmas %s' % ', '.join(['%.2e' % sii for sii in np.sqrt(np.diag(Finv))]))
     

    if theta_nuis is not None: 
        _thetas += theta_nuis 
        _theta_lbls += [theta_nuis_lbls[tt] for tt in theta_nuis]
        for tt in theta_nuis: _theta_fid[tt] = theta_nuis_fids[tt]
        _theta_lims += [theta_nuis_lims[tt] for tt in theta_nuis]

    fig = plt.figure(figsize=(17, 15))
    for i in range(len(_thetas)+1): 
        for j in range(i+1, len(_thetas)): 
            # sub inverse fisher matrix 
            Finv_sub = np.array([[Finv[i,i], Finv[i,j]], [Finv[j,i], Finv[j,j]]]) 
            # plot the ellipse
            sub = fig.add_subplot(len(_thetas)-1, len(_thetas)-1, (len(_thetas)-1) * (j-1) + i + 1) 
            Forecast.plotEllipse(Finv_sub, sub, 
                    theta_fid_ij=[_theta_fid[_thetas[i]], _theta_fid[_thetas[j]]], color='C0')
            sub.set_xlim(_theta_fid[_thetas[i]]-_theta_dlims[i], _theta_fid[_thetas[i]]+_theta_dlims[i])
            sub.set_ylim(_theta_fid[_thetas[j]]-_theta_dlims[j], _theta_fid[_thetas[j]]+_theta_dlims[j])
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
    if theta_nuis is not None: 
        if 'Amp' in theta_nuis: nuis_str += 'b'
        if 'Mmin' in theta_nuis: nuis_str += 'Mmin'
        if ('Asn' in theta_nuis) or ('Bsn' in theta_nuis): nuis_str += 'SN'
        if 'b2' in theta_nuis: nuis_str += 'b2'
        if 'g2' in theta_nuis: nuis_str += 'g2'

    planck_str = ''
    if planck: planck_str = '.planck'

    ffig = os.path.join(dir_doc, 
            'quijote.%sFisher%s%s.dmnu_%s.kmax%.2f%s%s%s.png' % 
            (obs, _params_str(params), nuis_str, dmnu, kmax, _rsd_str(rsd), _flag_str(flag), planck_str))
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') # latex version 
    return None 


def P02B_Forecast(kmax=0.5, seed=range(5), rsd='all', dmnu='fin', theta_nuis=None,
        planck=False, silent=True):
    ''' fisher forecast comparison for P0+P2, B, and P0+P2+B for quijote observables where we 
    marginalize over theta_nuis parameters. 
    
    :param kmax: (default: 0.5) 
        kmax of the analysis 
    :param rsd: (default: True) 
        specifies the rsd set up for the B(k) derivative. 
        If rsd == 'all', include 3 RSD directions. 
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
    # fisher matrices (Fij)
    pk_theta_nuis = copy(theta_nuis)
    bk_theta_nuis = copy(theta_nuis)
    if theta_nuis is not None: 
        if 'Bsn' in theta_nuis: pk_theta_nuis.remove('Bsn') 
        if 'b2' in theta_nuis: pk_theta_nuis.remove('b2') 
        if 'g2' in theta_nuis: pk_theta_nuis.remove('g2') 

    pkFij   = FisherMatrix('p02k', kmax=kmax, seed=seed, rsd=rsd, dmnu=dmnu,
            theta_nuis=pk_theta_nuis, silent=silent)  
    bkFij   = FisherMatrix('bk', kmax=kmax, seed=seed, rsd=rsd, dmnu=dmnu,
            theta_nuis=bk_theta_nuis, silent=silent)  
    pbkFij  = FisherMatrix('p02bk', kmax=kmax, seed=seed, rsd=rsd, dmnu=dmnu,
            theta_nuis=theta_nuis, silent=silent)
    cond = np.linalg.cond(pbkFij)
    if cond > 1e16: print('Fij is ill-conditioned %.5e' % cond)
    
    if planck: # add planck prior 
        _Fij_planck = np.load(os.path.join(UT.dat_dir(), 'Planck_2018_s8.npy')) # read in planck prior fisher (order is Om, Ob, h, ns, s8 and Mnu) 
        print('Planck Fii', np.diag(_Fij_planck)) 
        pkFij[:6,:6] += _Fij_planck
        bkFij[:6,:6] += _Fij_planck
        pbkFij[:6,:6] += _Fij_planck
    pkFinv  = np.linalg.inv(pkFij) 
    bkFinv  = np.linalg.inv(bkFij) 
    pbkFinv = np.linalg.inv(pbkFij) 

    i_Mnu = thetas.index('Mnu')
    print('--- i = Mnu ---')
    print('P Fii=%f, sigma_i = %f' % (pkFij[i_Mnu,i_Mnu], np.sqrt(pkFinv[i_Mnu,i_Mnu])))
    print("B Fii=%f, sigma_i = %f" % (bkFij[i_Mnu,i_Mnu], np.sqrt(bkFinv[i_Mnu,i_Mnu])))
    print("P+B Fii=%f, sigma_i = %f" % (pbkFij[i_Mnu,i_Mnu], np.sqrt(pbkFinv[i_Mnu,i_Mnu])))
    print('--- thetas ---')
    print('P sigmas %s' % ', '.join(['%.3f' % sii for sii in np.sqrt(np.diag(pkFinv))]))
    print('B sigmas %s' % ', '.join(['%.3f' % sii for sii in np.sqrt(np.diag(bkFinv))]))
    print('P+B sigmas %s' % ', '.join(['%.3f' % sii for sii in np.sqrt(np.diag(pbkFinv))]))
    if theta_nuis is not None and 'Bsn' in theta_nuis: 
        pass 
    else: 
        print('improve. over P %s' % ', '.join(['%.1f' % sii for sii in np.sqrt(np.diag(pkFinv)/np.diag(pbkFinv))]))
    print('improve. over B %s' % ', '.join(['%.1f' % sii for sii in np.sqrt(np.diag(bkFinv)/np.diag(pbkFinv))]))
        
    ntheta = len(thetas) 
    _thetas = copy(thetas) 
    _theta_lbls = copy(theta_lbls) 
    _theta_fid = theta_fid.copy() # fiducial thetas
    #_theta_dlims = [0.0425, 0.016, 0.16, 0.15, 0.09, 0.25, 0.4, 0.75, 0.4, 0.2, 0.25]
    _theta_dlims = [0.05, 0.02, 0.25, 0.3, 0.125, 0.45, 0.6, 1.25, 0.75, 0.375, 0.275]
    if planck: _theta_dlims = [0.024, 0.002, 0.016, 0.0075, 0.032, 0.1, 0.16, 0.2, 0.25, 0.2, 0.2]
    
    n_nuis = 0  
    if theta_nuis is not None: 
        _thetas += theta_nuis 
        _theta_lbls += [theta_nuis_lbls[tt] for tt in theta_nuis]
        for tt in theta_nuis: _theta_fid[tt] = theta_nuis_fids[tt]
        #_theta_lims += [theta_nuis_lims[tt] for tt in theta_nuis]
        _theta_dlims += [0.5 * (theta_nuis_lims[tt][1] - theta_nuis_lims[tt][0]) for tt in theta_nuis]
        n_nuis = len(theta_nuis) 

    Finvs   = [pkFinv, bkFinv, pbkFinv]
    colors  = ['C0', 'C2', 'C1']
    sigmas  = [[1, 2], [2], [1, 2]]
    alphas  = [[0.8, 0.6], [0.9], [1., 0.7]]

    fig = plt.figure(figsize=(20, 20))
    gs = mpl.gridspec.GridSpec(2, 1, figure=fig, height_ratios=[5,5+n_nuis], hspace=0.1) 
    gs0 = mpl.gridspec.GridSpecFromSubplotSpec(5, ntheta+n_nuis-1, subplot_spec=gs[0])
    gs1 = mpl.gridspec.GridSpecFromSubplotSpec(5+n_nuis, ntheta+n_nuis-1, subplot_spec=gs[1])
    
    for i in range(ntheta+n_nuis-1): 
        for j in range(i+1, ntheta+n_nuis): 
            theta_fid_i, theta_fid_j = _theta_fid[_thetas[i]], _theta_fid[_thetas[j]] # fiducial parameter 
            
            if j < 6: sub = plt.subplot(gs0[j-1,i]) # cosmo. params
            else: sub = plt.subplot(gs1[j-6,i]) # the rest

            for _i, Finv in enumerate(Finvs):
                try:
                    Finv_sub = np.array([[Finv[i,i], Finv[i,j]], [Finv[j,i], Finv[j,j]]]) # sub inverse fisher matrix 
                except IndexError: 
                    continue 
                Forecast.plotEllipse(Finv_sub, sub, theta_fid_ij=[theta_fid_i, theta_fid_j], color=colors[_i], 
                        sigmas=sigmas[_i], alphas=alphas[_i])

            sub.set_xlim(theta_fid_i - _theta_dlims[i], theta_fid_i + _theta_dlims[i])
            sub.set_ylim(theta_fid_j - _theta_dlims[j], theta_fid_j + _theta_dlims[j])
            if i == 0:   
                sub.set_ylabel(_theta_lbls[j], labelpad=5, fontsize=24) 
                sub.get_yaxis().set_label_coords(-0.35,0.5)
            else: 
                sub.set_yticks([])
                sub.set_yticklabels([])
            
            if j == ntheta+n_nuis-1: 
                sub.set_xlabel(_theta_lbls[i], labelpad=7, fontsize=24) 
            elif j == 5: 
                sub.set_xlabel(_theta_lbls[i], fontsize=24) 
            else: 
                sub.set_xticks([])
                sub.set_xticklabels([]) 

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.fill_between([],[],[], color=colors[0], label=r'$P^g_{0}(k) + P^g_{2}(k)$') 
    bkgd.fill_between([],[],[], color=colors[1], label=r'$B^g_0(k_1, k_2, k_3)$') 
    bkgd.fill_between([],[],[], color=colors[2], label=r'$P^g_{0} + P^g_{2} + B^g_0$') 
    bkgd.legend(loc='upper right', bbox_to_anchor=(0.875, 0.775), handletextpad=0.2, fontsize=25)
    bkgd.text(0.85, 0.62, r'$k_{\rm max} = %.1f$; $z=0.$' % kmax, ha='right', va='bottom', 
            transform=bkgd.transAxes, fontsize=25)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.05, hspace=0.05) 
    
    nuis_str = ''
    if theta_nuis is not None: 
        nuis_str += '.'
        if 'b1' in theta_nuis: nuis_str += 'b1'
        if ('Asn' in theta_nuis) or ('Bsn' in theta_nuis): nuis_str += 'SN'
        if 'b2' in theta_nuis: nuis_str += 'b2'
        if 'g2' in theta_nuis: nuis_str += 'g2'

    planck_str = ''
    if planck: planck_str = '.planck'

    ffig = os.path.join(dir_doc, 
            'Fisher.p02bk%s.dmnu_%s.kmax%.2f%s%s%s.png' % 
            (nuis_str, dmnu, kmax, _seed_str(seed), _rsd_str(rsd), planck_str))
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') # latex version 
    return None 


def P02B_Forecast_kmax(seed='all', rsd='all', dmnu='fin', theta_nuis=None):
    ''' fisher forecast for quijote P0+P2 and B where theta_nuis are added as free parameters 
    as a function of kmax.
    
    :param rsd: (default: True) 
        rsd kwarg that specifies rsd set up for B(k). 
        If rsd == True, include 3 RSD directions. 
        If rsd in [0,1,2] include one of the directions
    :param theta_nuis: (default: None) 
        list of nuisance parameters to include in the forecast. 
    :param dmnu: (default: 'fin') 
        derivative finite differences setup
    :param LT: (default: False)
        If True include linear theory matter powerspectrum forecasts
    :param planck: (default: False)
        If True add Planck prior 
    '''
    kmaxs = kf * 3 * np.arange(1, 28) 

    pk_theta_nuis = copy(theta_nuis)
    bk_theta_nuis = copy(theta_nuis)
    if theta_nuis is not None: 
        if 'Bsn' in theta_nuis: pk_theta_nuis.remove('Bsn') 
        if 'b2' in theta_nuis: pk_theta_nuis.remove('b2') 
        if 'g2' in theta_nuis: pk_theta_nuis.remove('g2') 
    
    _Fij_planck = np.load(os.path.join(UT.dat_dir(), 'Planck_2018_s8.npy')) # read in planck prior fisher (order is Om, Ob, h, ns, s8 and Mnu) 

    # read in fisher matrix (Fij)
    sig_pk, sig_pbk, sig_pk_planck, sig_pbk_planck = [], [], [], []
    for i_k, kmax in enumerate(kmaxs): 
        pkFij   = FisherMatrix('p02k', kmax=kmax, seed=seed, rsd=rsd, dmnu=dmnu, theta_nuis=pk_theta_nuis)  
        pbkFij  = FisherMatrix('p02bk', kmax=kmax, seed=seed, rsd=rsd, dmnu=dmnu, theta_nuis=theta_nuis)  
        
        pkFij_planck = pkFij.copy() 
        pbkFij_planck = pbkFij.copy() 
        pkFij_planck[:6,:6] += _Fij_planck
        pbkFij_planck[:6,:6] += _Fij_planck
        
        sig_pk.append(np.sqrt(np.diag(np.linalg.inv(pkFij))))
        sig_pbk.append(np.sqrt(np.diag(np.linalg.inv(pbkFij))))

        sig_pk_planck.append(np.sqrt(np.diag(np.linalg.inv(pkFij_planck))))
        sig_pbk_planck.append(np.sqrt(np.diag(np.linalg.inv(pbkFij_planck))))

        print('kmax=%.3f, %i---' %  (kmax, int(kmax/kf)))
        if np.linalg.cond(pkFij) > 1e16: 
            print('  P02 Fij ill-conditioned; cond # = %.2e' % np.linalg.cond(pkFij)) 
        #if np.linalg.cond(bkFij) > 1e16: 
        #    print('B Fij ill-conditioned; cond # = %.2e' % np.linalg.cond(bkFij)) 
        if np.linalg.cond(pbkFij) > 1e16: 
            print('  P02+B Fij ill-conditioned; cond # = %.2e' % np.linalg.cond(pbkFij)) 

        print('  pk: %s' % ', '.join(['%.2e' % fii for fii in sig_pk[-1]])) 
        print('  pbk: %s' % ', '.join(['%.2e' % fii for fii in sig_pbk[-1]])) 
        print('  pk w/ planck: %s' % ', '.join(['%.2e' % fii for fii in sig_pk_planck[-1]])) 
        print('  pbk w/ planck: %s' % ', '.join(['%.2e' % fii for fii in sig_pbk_planck[-1]])) 

    sig_pk  = np.array(sig_pk)
    sig_pbk = np.array(sig_pbk)
    sig_pk_planck = np.array(sig_pk_planck)
    sig_pbk_planck= np.array(sig_pbk_planck)
     
    cond_pk = (kmaxs > 6. * kf)
    cond_pbk = (kmaxs > 3. * kf)
    print('PK kmax > %f' % (6. * kf)) 
    print('PBK kmax > %f' %  (3. * kf))

    # write out to table
    kmax_preset = np.zeros(len(kmaxs)).astype(bool) 
    for kmax in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]: 
        kmax_preset[(np.abs(kmaxs - kmax)).argmin()] = True

    pk_dat = np.vstack((np.atleast_2d(kmaxs[kmax_preset]), sig_pk[kmax_preset,:].T)).T 
    pbk_dat = np.vstack((np.atleast_2d(kmaxs[kmax_preset]), sig_pbk[kmax_preset,:].T)).T 

    fpk = os.path.join(UT.doc_dir(), 'dat', 
            'P02g_forecast_kmax%s.dmnu_%s.dat' % (_nuis_str(pk_theta_nuis), dmnu)) 
    np.savetxt(fpk, pk_dat, delimiter=', ', fmt='%.5f')
    fpbk = os.path.join(UT.doc_dir(), 'dat', 
            'P02Bg_forecast_kmax%s.dmnu_%s.dat' % (_nuis_str(bk_theta_nuis), dmnu))
    np.savetxt(fpbk, pbk_dat, delimiter=', ', fmt='%.5f')

    #sigma_theta_lims = [(5e-3, 0.8), (5e-4, 1.), (1e-3, 10), (3e-3, 10), (5e-3, 10), (6e-3, 10.)]
    sigma_theta_lims = [(5e-3, 1.), (5e-4, 1.), (5e-3, 10), (3e-3, 10), (5e-3, 10), (1e-2, 10.)]
    
    colors  = ['C0', 'C2', 'C1']

    fig = plt.figure(figsize=(15,8))
    for i, theta in enumerate(thetas[:6]): 
        sub = fig.add_subplot(2,3,i+1) 
        _plt_pk, = sub.plot(kmaxs[cond_pk], sig_pk[:,i][cond_pk], c=colors[0], ls='-') 
        _plt_pbk, = sub.plot(kmaxs[cond_pbk], sig_pbk[:,i][cond_pbk], c=colors[2], ls='-') 

        sub.plot(kmaxs[cond_pk], sig_pk_planck[:,i][cond_pk], c=colors[0], ls='--', lw=1) 
        _plt, = sub.plot(kmaxs[cond_pbk], sig_pbk_planck[:,i][cond_pbk], c=colors[2], ls='--', lw=1) 
        if theta == 'Mnu': 
            sub.legend([_plt_pbk, _plt], ['LSS only', 'w/ Planck priors'], loc='lower left', 
                    bbox_to_anchor=(0.0, -0.05), handletextpad=0.25, fontsize=18) 

        sub.set_xlim(0.05, 0.5)
        sub.text(0.9, 0.9, theta_lbls[i], ha='right', va='top', transform=sub.transAxes, fontsize=30)
        sub.set_ylim(sigma_theta_lims[i]) 
        sub.set_yscale('log') 
        if i == 1: 
            sub.legend([_plt_pk, _plt_pbk], [r"$P^g_\ell$", r"$P^g_\ell+B^g_0$"], 
                    loc='lower center', bbox_to_anchor=(0.5, 1.0), handletextpad=0.2, fontsize=20, ncol=20)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'$k_{\rm max}$', fontsize=28) 
    bkgd.set_ylabel(r'{\fontsize{28pt}{3em}\selectfont{}$\sigma_\theta~/$}{\fontsize{20pt}{3em}\selectfont{}$\sqrt{\frac{V}{1 ({\rm Gpc}/h)^3}}$}', labelpad=15)#, fontsize=28) 
    fig.subplots_adjust(wspace=0.2, hspace=0.15) 

    ffig = os.path.join(dir_doc, 
            'Fisher_kmax.p02bk%s.dmnu_%s%s%s.png' % (_nuis_str(bk_theta_nuis), dmnu, _seed_str(seed), _rsd_str(rsd)))
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') # latex 
    return None


def _P02B_P02Bh_Forecast_kmax(rsd='all', flag='reg', dmnu='fin', theta_nuis=None):
    ''' fisher forecast for quijote P0+P2 and B where theta_nuis are added as free parameters 
    as a function of kmax.
    '''
    assert rsd == 'all'
    assert flag == 'reg'

    pk_theta_nuis = copy(theta_nuis)
    bk_theta_nuis = copy(theta_nuis)
    if theta_nuis is not None: 
        if 'Bsn' in theta_nuis: pk_theta_nuis.remove('Bsn') 
        if 'b2' in theta_nuis: pk_theta_nuis.remove('b2') 
        if 'g2' in theta_nuis: pk_theta_nuis.remove('g2') 
    
    # read the galaxy constraints
    # Pg constraint with {theta_cosmo}
    fpg = os.path.join(UT.doc_dir(), 'dat', 
            'P02g_forecast_kmax%s.dmnu_%s.dat' % (_nuis_str(pk_theta_nuis), dmnu)) 
    sigmas_pg = np.loadtxt(fpg, delimiter=',', unpack=True, usecols=range(7))
    # Pg+Bg constraint with {theta_cosmo}
    fpbg = os.path.join(UT.doc_dir(), 'dat', 
            'P02Bg_forecast_kmax%s.dmnu_%s.dat' % (_nuis_str(bk_theta_nuis), dmnu))
    sigmas_pbg = np.loadtxt(fpbg, delimiter=',', unpack=True, usecols=range(7))

    # read the halo constraints
    # Ph constraint with {theta_cosmo}
    fph = os.path.join(UT.doc_dir(), 'dat', 'P02h_forecast_kmax.dat') 
    sigmas_ph = np.loadtxt(fph, delimiter=',', unpack=True, usecols=range(7))
    # Ph+Bh constraint with {theta_cosmo}
    fpbh = os.path.join(UT.doc_dir(), 'dat', 'P02Bh_forecast_kmax.dat') 
    sigmas_pbh = np.loadtxt(fpbh, delimiter=',', unpack=True, usecols=range(7))

    # read the halo constraints with nuisance parameters
    # Ph constraint with {theta_cosmo}
    fphn = os.path.join(UT.doc_dir(), 'dat', 'P02h_forecast_kmax.b1Mmin.dat') 
    sigmas_phn = np.loadtxt(fphn, delimiter=',', unpack=True, usecols=range(7))
    # Ph+Bh constraint with {theta_cosmo}
    fpbhn = os.path.join(UT.doc_dir(), 'dat', 'P02Bh_forecast_kmax.b1Mmin.dat') 
    sigmas_pbhn = np.loadtxt(fpbhn, delimiter=',', unpack=True, usecols=range(7))

    #sigma_theta_lims = [(5e-3, 0.8), (5e-4, 1.), (1e-3, 10), (3e-3, 10), (5e-3, 10), (6e-3, 10.)]
    sigma_theta_lims = [(1e-3, 1.), (5e-4, 1.), (5e-3, 10), (3e-3, 10), (1e-3, 10), (1e-2, 10.)]
    
    colors  = ['C0', 'C2', 'C1']
    fig = plt.figure(figsize=(15,8))
    for i, theta in enumerate(thetas[:6]): 
        sub = fig.add_subplot(2,3,i+1) 

        _plt_pg, = sub.plot(sigmas_pg[0], sigmas_pg[i+1], c=colors[0], ls='-') 
        _plt_ph, = sub.plot(sigmas_ph[0], sigmas_ph[i+1], c=colors[0], ls='--') 
        _plt_phn, = sub.plot(sigmas_phn[0], sigmas_phn[i+1], c=colors[0], ls=':') 
        _plt_pbg, = sub.plot(sigmas_pbg[0], sigmas_pbg[i+1], c=colors[2], ls='-') 
        _plt_pbh, = sub.plot(sigmas_pbh[0], sigmas_pbh[i+1], c=colors[2], ls='--') 
        _plt_pbhn, = sub.plot(sigmas_pbhn[0], sigmas_pbhn[i+1], c=colors[2], ls=':') 

        if theta == 'Mnu': 
            sub.legend(
                    [_plt_pbg, _plt_pbh, _plt_pbhn], 
                    ['galaxy', 'halo', r'halo + $b_1, M_{\rm min}$'], loc='lower left', 
                    bbox_to_anchor=(0.0, -0.05), handletextpad=0.25, fontsize=18) 

        sub.set_xlim(0.05, 0.5)
        sub.text(0.9, 0.9, theta_lbls[i], ha='right', va='top', transform=sub.transAxes, fontsize=30)
        sub.set_ylim(sigma_theta_lims[i]) 
        sub.set_yscale('log') 
        if i == 1: 
            sub.legend([_plt_pg, _plt_pbg], [r"$P_\ell$", r"$P_\ell+B_0$"], 
                    loc='lower center', bbox_to_anchor=(0.5, 1.0), handletextpad=0.2, fontsize=20, ncol=20)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'$k_{\rm max}$', fontsize=28) 
    bkgd.set_ylabel(r'{\fontsize{28pt}{3em}\selectfont{}$\sigma_\theta~/$}{\fontsize{20pt}{3em}\selectfont{}$\sqrt{\frac{V}{1 ({\rm Gpc}/h)^3}}$}', labelpad=15)#, fontsize=28) 
    fig.subplots_adjust(wspace=0.2, hspace=0.15) 

    ffig = os.path.join(dir_doc, 
            'Fisher_kmax.galaxy_v_halos%s.dmnu_%s.png' %
            (_nuis_str(bk_theta_nuis), dmnu))
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def _P02B_Forecast_kmax_flex(rsd='all', flag='reg', dmnu='fin',
        params='default', theta_nuis=None):
    ''' flexible version of fisher forecast for quijote P0+P2 and B where theta_nuis are added as free parameters 
    as a function of kmax.
    '''
    kmaxs = kf * 3 * np.arange(3, 28) 

    pk_theta_nuis = copy(theta_nuis)
    bk_theta_nuis = copy(theta_nuis)
    if theta_nuis is not None: 
        if 'Bsn' in theta_nuis: pk_theta_nuis.remove('Bsn') 
        if 'b2' in theta_nuis: pk_theta_nuis.remove('b2') 
        if 'g2' in theta_nuis: pk_theta_nuis.remove('g2') 
    
    _Fij_planck = np.load(os.path.join(UT.dat_dir(), 'Planck_2018_s8.npy')) # read in planck prior fisher (order is Om, Ob, h, ns, s8 and Mnu) 

    # read in fisher matrix (Fij)
    sig_pk, sig_pbk, sig_pk_planck, sig_pbk_planck = [], [], [], []
    for i_k, kmax in enumerate(kmaxs): 
        pkFij   = FisherMatrix('p02k', kmax=kmax, rsd=rsd, flag=flag,
                dmnu=dmnu, params=params, theta_nuis=pk_theta_nuis)  
        pbkFij  = FisherMatrix('p02bk', kmax=kmax, rsd=rsd,  flag=flag,
                dmnu=dmnu, params=params, theta_nuis=theta_nuis) 
        
        pkFij_planck = pkFij.copy() 
        pbkFij_planck = pbkFij.copy() 
        pkFij_planck[:6,:6] += _Fij_planck
        pbkFij_planck[:6,:6] += _Fij_planck
        
        sig_pk.append(np.sqrt(np.diag(np.linalg.inv(pkFij))))
        sig_pbk.append(np.sqrt(np.diag(np.linalg.inv(pbkFij))))

        sig_pk_planck.append(np.sqrt(np.diag(np.linalg.inv(pkFij_planck))))
        sig_pbk_planck.append(np.sqrt(np.diag(np.linalg.inv(pbkFij_planck))))

        print('kmax=%.3f, %i---' %  (kmax, int(kmax/kf)))
        if np.linalg.cond(pkFij) > 1e16: 
            print('  P02 Fij ill-conditioned; cond # = %.2e' % np.linalg.cond(pkFij)) 
        #if np.linalg.cond(bkFij) > 1e16: 
        #    print('B Fij ill-conditioned; cond # = %.2e' % np.linalg.cond(bkFij)) 
        if np.linalg.cond(pbkFij) > 1e16: 
            print('  P02+B Fij ill-conditioned; cond # = %.2e' % np.linalg.cond(bkFij)) 

        print('  pk: %s' % ', '.join(['%.2e' % fii for fii in sig_pk[-1]])) 
        print('  pbk: %s' % ', '.join(['%.2e' % fii for fii in sig_pbk[-1]])) 
        print('  pk w/ planck: %s' % ', '.join(['%.2e' % fii for fii in sig_pk_planck[-1]])) 
        print('  pbk w/ planck: %s' % ', '.join(['%.2e' % fii for fii in sig_pbk_planck[-1]])) 

    sig_pk  = np.array(sig_pk)
    sig_pbk = np.array(sig_pbk)
    sig_pk_planck = np.array(sig_pk_planck)
    sig_pbk_planck= np.array(sig_pbk_planck)
     
    cond_pk = (kmaxs > 6. * kf)
    cond_pbk = (kmaxs > 12. * kf)
    print('PK kmax > %f' % (5. * kf)) 
    print('PBK kmax > %f' %  (12. * kf))

    # write out to table
    kmax_preset = np.zeros(len(kmaxs)).astype(bool) 
    for kmax in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]: 
        kmax_preset[(np.abs(kmaxs - kmax)).argmin()] = True

    pk_dat = np.vstack((np.atleast_2d(kmaxs[kmax_preset]), sig_pk[kmax_preset,:].T)).T 
    pbk_dat = np.vstack((np.atleast_2d(kmaxs[kmax_preset]), sig_pbk[kmax_preset,:].T)).T 

    fpk = os.path.join(UT.doc_dir(), 'dat', 
            'P02g_forecast_kmax%s%s.dmnu_%s.dat' % 
            (_params_str(params), _nuis_str(pk_theta_nuis), dmnu)) 
    np.savetxt(fpk, pk_dat, delimiter=', ', fmt='%.5f')
    fpbk = os.path.join(UT.doc_dir(), 'dat', 
            'P02Bg_forecast_kmax%s%s.dmnu_%s.dat' % 
            (_params_str(params), _nuis_str(bk_theta_nuis), dmnu)) 
    np.savetxt(fpbk, pbk_dat, delimiter=', ', fmt='%.5f')

    sigma_theta_lims = [(5e-3, 1.), (5e-4, 1.), (5e-3, 10), (3e-3, 10), (5e-3, 10), (1e-2, 10.)]
    
    colors  = ['C0', 'C2', 'C1']

    fig = plt.figure(figsize=(15,8))
    for i, theta in enumerate(thetas[:6]): 
        sub = fig.add_subplot(2,3,i+1) 
        _plt_pk, = sub.plot(kmaxs[cond_pk], sig_pk[:,i][cond_pk], c=colors[0], ls='-') 
        _plt_pbk, = sub.plot(kmaxs[cond_pbk], sig_pbk[:,i][cond_pbk], c=colors[2], ls='-') 

        sub.plot(kmaxs[cond_pk], sig_pk_planck[:,i][cond_pk], c=colors[0], ls='--', lw=1) 
        _plt, = sub.plot(kmaxs[cond_pbk], sig_pbk_planck[:,i][cond_pbk], c=colors[2], ls='--', lw=1) 
        if theta == 'Mnu': 
            sub.legend([_plt_pbk, _plt], ['LSS only', 'w/ Planck priors'], loc='lower left', 
                    bbox_to_anchor=(0.0, -0.05), handletextpad=0.25, fontsize=18) 

        sub.set_xlim(0.05, 0.5)
        sub.text(0.9, 0.9, theta_lbls[i], ha='right', va='top', transform=sub.transAxes, fontsize=30)
        sub.set_ylim(sigma_theta_lims[i]) 
        sub.set_yscale('log') 
        if i == 1: 
            sub.legend([_plt_pk, _plt_pbk], [r"$P^g_\ell$", r"$P^g_\ell+B^g_0$"], 
                    loc='lower center', bbox_to_anchor=(0.5, 1.0), handletextpad=0.2, fontsize=20, ncol=20)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'$k_{\rm max}$', fontsize=28) 
    bkgd.set_ylabel(r'{\fontsize{28pt}{3em}\selectfont{}$\sigma_\theta~/$}{\fontsize{20pt}{3em}\selectfont{}$\sqrt{\frac{V}{1 ({\rm Gpc}/h)^3}}$}', labelpad=15)#, fontsize=28) 
    fig.subplots_adjust(wspace=0.2, hspace=0.15) 

    ffig = os.path.join(dir_doc, 
            'Fisher_kmax.p02bk%s%s.dmnu_%s%s%s.png' % (_params_str(params),
                _nuis_str(theta_nuis), dmnu, _rsd_str(rsd), _flag_str(flag)))
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def _PgPh_Forecast_kmax(): 
    ''' quick comparison of the Pg and Ph forecasts as a function of kmax 
    '''
    # Pg constraint with {theta_cosmo, theta_hod} 
    fpg0 = os.path.join(UT.doc_dir(), 'dat', 'P02g_forecast_kmax.dat') 
    sigmas_pg0 = np.loadtxt(fpg0, delimiter=',', unpack=True, usecols=range(7))
    # Pg constraint with {theta_cosmo, b1, Mmin} 
    fpg1 = os.path.join(UT.doc_dir(), 'dat', 'P02g_forecast_kmax.halo_param.b1.dat')
    sigmas_pg1 = np.loadtxt(fpg1, delimiter=',', unpack=True, usecols=range(7))
    
    # Ph constraint with {theta_cosmo, b1, Mmin}
    fph = os.path.join(UT.doc_dir(), 'dat', 'p02k_forecast_kmax.b1Mmin.dat') 
    sigmas_ph = np.loadtxt(fph, delimiter=',', unpack=True, usecols=range(7))
    
    sigma_theta_lims = [(5e-3, 1.), (5e-4, 1.), (5e-3, 10), (3e-3, 10), (5e-3, 10), (1e-2, 10.)]
    #pg_lcdm = [2.46e-02, 8.82e-03, 8.99e-02, 1.09e-01, 5.99e-02]
    #ph_lcdm = [1.33e-02, 1.09e-02, 9.76e-02, 5.79e-02, 1.38e-02]
    
    fig = plt.figure(figsize=(15,8))
    for i, theta in enumerate(thetas[:6]): 
        sub = fig.add_subplot(2,3,i+1) 
        _plt_ph, = sub.plot(sigmas_ph[0], sigmas_ph[i+1], c='k', ls='-') 
        _plt_pg0, = sub.plot(sigmas_pg0[0], sigmas_pg0[i+1], c='C0', ls='-') 
        _plt_pg1, = sub.plot(sigmas_pg1[0], sigmas_pg1[i+1], c='C1', ls='-') 

        sub.set_xlim(0.05, 0.5)
        sub.text(0.9, 0.9, theta_lbls[i], ha='right', va='top', transform=sub.transAxes, fontsize=30)
        sub.set_ylim(sigma_theta_lims[i]) 
        sub.set_yscale('log') 
        if i == 1: 
            sub.legend([_plt_ph, _plt_pg0, _plt_pg1], 
            [
                r"$P_{h}$ $\{\theta_{\rm cosmo}, b_1, M_{\rm min}\}$", 
                r"$P_{g}$ $\{\theta_{\rm cosmo}, \theta_{\rm HOD}\}$", 
                r"$P_{g}$ $\{\theta_{\rm cosmo}, b_1, M_{\rm min}\}$"], 
                    loc='lower center', bbox_to_anchor=(0.5, 1.0), handletextpad=0.2, fontsize=20, ncol=20)
        #if i < 5: 
        #    sub.plot(sigma_theta_lims[i], [pg_lcdm[i], pg_lcdm[i]], ls='--', c='C0') 
        #    sub.plot(sigma_theta_lims[i], [ph_lcdm[i], ph_lcdm[i]], ls='--', c='k') 

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'$k_{\rm max}$', fontsize=28) 
    bkgd.set_ylabel(r'{\fontsize{28pt}{3em}\selectfont{}$\sigma_\theta~/$}{\fontsize{20pt}{3em}\selectfont{}$\sqrt{\frac{V}{1 ({\rm Gpc}/h)^3}}$}', labelpad=15)#, fontsize=28) 
    fig.subplots_adjust(wspace=0.2, hspace=0.15) 
    
    ffig = os.path.join(dir_doc, '_PgPh_Forecast_kmax.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def _FisherMatrix_diffkmax(obs, pkmax=0.2, bkmax=0.1, seed='all', rsd='all', dmnu='fin',
        params='default', theta_nuis=None, cross_covariance=True, Cgauss=False,
        silent=True): 
    ''' calculate fisher matrix for parameters ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu'] 
    and specified nuisance parameters
    
    :param obs: 
        observable ('p02k', 'bk', 'p02bk') 
    :param kmax: (default: 0.5) 
        kmax of the analysis 
    :param rsd: (default: True) 
        rsd kwarg that specifies rsd set up for B(k) deriv.. 
        If rsd == 'all', include 3 RSD directions. 
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
    if obs == 'p02bk': 
        ks, _Cov, nmock = p02bkCov(rsd=2, flag='reg')
        k, i_k, j_k, l_k = ks 

        pklim = (k <= pkmax) 
        bklim = ((i_k*kf <= bkmax) & (j_k*kf <= bkmax) & (l_k*kf <= bkmax)) # k limit 
        klim = np.concatenate([pklim, bklim]) 
        Cov = _Cov[:,klim][klim,:]
    else: 
        raise NotImplementedError
    #print('%iD data vector' % Cov.shape[0])
    ndata = np.sum(klim) 
    f_hartlap = float(nmock - ndata - 2)/float(nmock - 1) 
    C_inv = f_hartlap * np.linalg.inv(Cov) # invert the covariance 
    
    if params == 'default': # default set of parameters 
        _thetas = ['Om', 'Ob2', 'h', 'ns', 's8', 'Mnu', 'logMmin', 'sigma_logM', 'logM0', 'alpha', 'logM1'] 
    elif params == 'lcdm': 
        _thetas = ['Om', 'Ob2', 'h', 'ns', 's8', 'logMmin', 'sigma_logM', 'logM0', 'alpha', 'logM1'] 
    elif params == 'halo':  # same parameters as halo case
        _thetas = ['Om', 'Ob2', 'h', 'ns', 's8', 'Mnu', 'logMmin'] 
    else: 
        raise NotImplementedError
    if theta_nuis is not None: _thetas += theta_nuis 
    
    # calculate the derivatives along all the thetas 
    dobs_dt = [] 
    for par in _thetas: 
        if obs == 'p02k': 
            _, dobs_dti, _ = dP02k(par, seed=seed, rsd=rsd, dmnu=dmnu,
                    silent=silent)
        elif obs == 'bk': 
            # rsd and flag kwargs are passed to the derivatives
            _, _, _, dobs_dti, _ = dBk(par, seed=seed, rsd=rsd, dmnu=dmnu, silent=silent)
        elif obs == 'p02bk': 
            _, _, _, _, dobs_dti, _ = dP02Bk(par, seed=seed, rsd=rsd,
                    dmnu=dmnu, fscale_pk=fscale_pk, silent=silent) 
        dobs_dt.append(dobs_dti[klim])
            
    Fij = Forecast.Fij(dobs_dt, C_inv) 
    return Fij 


def _P02B_Forecast_diffkmax(pkmax=0.2, bkmax=0.1, seed='all', rsd='all', dmnu='fin', theta_nuis=None,
        planck=False, silent=True):
    ''' fisher forecast comparison for P0+P2, B, and P0+P2+B where we set
    different kmax values for P and B for quijote observables where we 
    marginalize over theta_nuis parameters. 
    
    :param kmax: (default: 0.5) 
        kmax of the analysis 
    :param rsd: (default: True) 
        specifies the rsd set up for the B(k) derivative. 
        If rsd == 'all', include 3 RSD directions. 
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
    # fisher matrices (Fij)
    pk_theta_nuis = copy(theta_nuis)
    bk_theta_nuis = copy(theta_nuis)
    if theta_nuis is not None: 
        if 'Bsn' in theta_nuis: pk_theta_nuis.remove('Bsn') 
        if 'b2' in theta_nuis: pk_theta_nuis.remove('b2') 
        if 'g2' in theta_nuis: pk_theta_nuis.remove('g2') 

    pkFij   = FisherMatrix('p02k', kmax=pkmax, seed=seed, rsd=rsd, dmnu=dmnu,
            theta_nuis=pk_theta_nuis, silent=silent)  
    bkFij   = FisherMatrix('bk', kmax=bkmax, seed=seed, rsd=rsd, dmnu=dmnu,
            theta_nuis=bk_theta_nuis, silent=silent)  
    pbkFij  = _FisherMatrix_diffkmax('p02bk', pkmax=pkmax, bkmax=bkmax,
            seed=seed, rsd=rsd, dmnu=dmnu, theta_nuis=theta_nuis, silent=silent)
    cond = np.linalg.cond(pbkFij)
    if cond > 1e16: print('Fij is ill-conditioned %.5e' % cond)
    
    if planck: # add planck prior 
        _Fij_planck = np.load(os.path.join(UT.dat_dir(), 'Planck_2018_s8.npy')) # read in planck prior fisher (order is Om, Ob, h, ns, s8 and Mnu) 
        print('Planck Fii', np.diag(_Fij_planck)) 
        pkFij[:6,:6] += _Fij_planck
        bkFij[:6,:6] += _Fij_planck
        pbkFij[:6,:6] += _Fij_planck
    pkFinv  = np.linalg.inv(pkFij) 
    bkFinv  = np.linalg.inv(bkFij) 
    pbkFinv = np.linalg.inv(pbkFij) 

    i_Mnu = thetas.index('Mnu')
    print('P kmax = %.2f' % pkmax)
    print('B kmax = %.2f' % bkmax)
    print('--- thetas ---')
    print('P sigmas %s' % ', '.join(['%.5f' % sii for sii in np.sqrt(np.diag(pkFinv))]))
    print('B sigmas %s' % ', '.join(['%.5f' % sii for sii in np.sqrt(np.diag(bkFinv))]))
    print('P+B sigmas %s' % ', '.join(['%.5f' % sii for sii in np.sqrt(np.diag(pbkFinv))]))
    if theta_nuis is not None and 'Bsn' in theta_nuis: 
        pass 
    else: 
        print('improve. over P %s' % ', '.join(['%.2f' % sii for sii in np.sqrt(np.diag(pkFinv)/np.diag(pbkFinv))]))
    print('improve. over B %s' % ', '.join(['%.1f' % sii for sii in np.sqrt(np.diag(bkFinv)/np.diag(pbkFinv))]))
    return None 


def _forecast_cross_covariance(kmax=0.5, seed=range(5), rsd='all', dmnu='fin'):
    ''' examining the impact of ignoring cross covariance between Pl and B
    '''
    # reference fisher matrix no nuisance param. 
    Fij_ref     = FisherMatrix('p02bk', kmax=kmax, seed=seed, rsd=rsd, 
            dmnu=dmnu, params='default', theta_nuis=None) 
    # fisher matrix without cross-covariance
    Fij_nocross = FisherMatrix('p02bk', kmax=kmax, seed=seed, rsd=rsd,
            dmnu=dmnu, params='default', cross_covariance=False) 
    # fisher matrix without off-diagonal 
    Fij_diag    = FisherMatrix('p02bk', kmax=kmax, seed=seed, rsd=rsd,
            dmnu=dmnu, params='default', diag_covariance=True) 
    
    # invert fisher matrix
    Finv_ref        = np.linalg.inv(Fij_ref) 
    Finv_nocross    = np.linalg.inv(Fij_nocross) 
    Finv_diag       = np.linalg.inv(Fij_diag) 
    print('Finv_ref condition number = %.5e' % np.linalg.cond(Finv_ref)) 
    print('Finv_nocross condition number = %.5e' % np.linalg.cond(Finv_nocross)) 
    print('Finv_diag condition number = %.5e' % np.linalg.cond(Finv_diag)) 
    
    onesig_ref      = np.sqrt(np.diag(Finv_ref))
    onesig_nocross  = np.sqrt(np.diag(Finv_nocross))
    onesig_diag     = np.sqrt(np.diag(Finv_diag))

    print('reference sigmas %s' % ', '.join(['%.3f' % sii for sii in
        onesig_ref[:6]]))
    print('no cross  sigmas %s' % ', '.join(['%.3f' % sii for sii in
        onesig_nocross[:6]]))
    print('only diag sigmas %s' % ', '.join(['%.3f' % sii for sii in
        onesig_diag[:6]]))
    print('no cross improve. over ref. %s' % 
            ', '.join(['%.3f' % sii for sii in (onesig_ref/onesig_nocross)[:6]]))
    print('diag only improve. over ref. %s' % 
            ', '.join(['%.3f' % sii for sii in (onesig_ref/onesig_diag)[:6]]))
    return None 


def _P02B_dmnu(kmax=0.5, seed=range(5), rsd='all', silent=True): 
    '''
    '''
    # fisher matrices (Fij)
    pbkFij_fin  = FisherMatrix('p02bk', kmax=kmax, seed=seed, rsd=rsd, dmnu='fin', theta_nuis=None, silent=silent)
    pbkFij_p    = FisherMatrix('p02bk', kmax=kmax, seed=seed, rsd=rsd, dmnu='p', theta_nuis=None, silent=silent)
    pbkFij_pp   = FisherMatrix('p02bk', kmax=kmax, seed=seed, rsd=rsd, dmnu='pp', theta_nuis=None, silent=silent)

    pbkFinv_fin = np.linalg.inv(pbkFij_fin) 
    pbkFinv_p   = np.linalg.inv(pbkFij_p) 
    pbkFinv_pp  = np.linalg.inv(pbkFij_pp) 

    print('P+B sigmas fin %s' % ', '.join(['%.3f' % sii for sii in
        np.sqrt(np.diag(pbkFinv_fin))]))
    print('P+B sigmas p   %s' % ', '.join(['%.3f' % sii for sii in
        np.sqrt(np.diag(pbkFinv_p))]))
    print('P+B sigmas pp  %s' % ', '.join(['%.3f' % sii for sii in
        np.sqrt(np.diag(pbkFinv_pp))]))
    print('(+)/(fin) %s' % ', '.join(['%.1f' % sii for sii in
        np.sqrt(np.diag(pbkFinv_p)/np.diag(pbkFinv_fin))]))
    print('(++)/(fin) %s' % ', '.join(['%.1f' % sii for sii in
        np.sqrt(np.diag(pbkFinv_pp)/np.diag(pbkFinv_fin))]))
    return None 
        

# --- convergence tests --- 
def _converge_FisherMatrix(obs, kmax=0.5, seed=range(5), rsd='all', flag='reg', dmnu='fin',
        params='default', theta_nuis=None, Ncov=None, Nderiv=None): 
    ''' calculate fisher matrix for parameters ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu'] 
    and specified nuisance parameters
    
    :param obs: 
        observable ('p02k', 'bk', 'p02bk') 
    :param kmax: (default: 0.5) 
        kmax of the analysis 
    :param rsd: (default: True) 
        rsd kwarg that specifies rsd set up for B(k) deriv.. 
        If rsd == 'all', include 3 RSD directions. 
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

    if obs == 'p02bk':  # monopole and quadrupole 
        if Ncov is None: 
            ks, _Cov, nmock = p02bkCov(rsd=2, flag='reg')
            k, i_k, j_k, l_k = ks 
        else: 
            # read in P(k) 
            quij = Obvs.quijhod_Pk('fiducial', rsd=2, flag='reg', silent=True) 
            p0ks = quij['p0k'] + quij['p_sn'][:,None] # shotnoise uncorrected P0 
            p2ks = quij['p2k'] # P2 
            k = np.concatenate([quij['k'], quij['k']]) 

            # read in B(k) 
            quij = Obvs.quijhod_Bk('fiducial', rsd=2, flag='reg', silent=True) 
            i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3']
            bks = quij['b123'] + quij['b_sn']

            pbks = np.concatenate([fscale_pk * p0ks, fscale_pk * p2ks, bks], axis=1)[:Ncov]
            
            _Cov = np.cov(pbks.T) # calculate the covariance
            nmock = pbks.shape[0]
            print('Ncov = %i' % Ncov) 

        pklim = (k <= kmax) 
        bklim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) # k limit 
        klim = np.concatenate([pklim, bklim]) 
        Cov = _Cov[:,klim][klim,:]
    elif obs == 'p02k': 
        if Ncov is None: 
            k, _Cov, nmock = p02kCov(rsd=2, flag='reg')
        else: 
            # read in P(k) 
            quij = Obvs.quijhod_Pk('fiducial', rsd=2, flag='reg', silent=True) 
            p0ks = quij['p0k'] + quij['p_sn'][:,None] # shotnoise uncorrected P0 
            p2ks = quij['p2k'] # P2 
            k = np.concatenate([quij['k'], quij['k']]) 

            pks = np.concatenate([p0ks, p2ks], axis=1)[:Ncov]
            
            _Cov = np.cov(pks.T) # calculate the covariance
            nmock = pks.shape[0]
            print('Ncov = %i' % Ncov) 

        klim = (k <= kmax) 
        Cov = _Cov[:,klim][klim,:]
    else: 
        raise NotImplementedError

    ndata = np.sum(klim) 
    print('Ndata = %i' % ndata) 
    f_hartlap = float(nmock - ndata - 2)/float(nmock - 1) 
    f_tdist = float((nmock-ndata)*nmock)/float((nmock-1) * (nmock-2))
    print('f_hartlap = %.3f, f_tdist = %.3f' % (f_hartlap, f_tdist))
    C_inv = f_tdist * np.linalg.inv(Cov) # invert the covariance 
    

    if params == 'default': # default set of parameters 
        _thetas = ['Om', 'Ob2', 'h', 'ns', 's8', 'Mnu', 'logMmin', 'sigma_logM', 'logM0', 'alpha', 'logM1'] 
    elif params == 'lcdm': 
        _thetas = ['Om', 'Ob2', 'h', 'ns', 's8', 'logMmin', 'sigma_logM', 'logM0', 'alpha', 'logM1'] 
    else: 
        raise NotImplementedError
    if theta_nuis is not None: _thetas += theta_nuis 
    
    # calculate the derivatives along all the thetas 
    dobs_dt = [] 
    for par in _thetas: 
        if obs == 'p02k': 
            _, dobs_dti, _ = dP02k(par, seed=seed, rsd=rsd, dmnu=dmnu,
                    Nderiv=Nderiv, silent=False)
        elif obs == 'bk': 
            # rsd and flag kwargs are passed to the derivatives
            _, _, _, dobs_dti, _ = dBk(par, seed=seed, rsd=rsd, dmnu=dmnu, Nderiv=Nderiv, silent=False)
        elif obs == 'p02bk': 
            _, _, _, _, dobs_dti, _ = dP02Bk(par, seed=seed, rsd=rsd,
                    dmnu=dmnu, fscale_pk=fscale_pk, Nderiv=Nderiv, silent=False) 
        dobs_dt.append(dobs_dti[klim])
            
    Fij = Forecast.Fij(dobs_dt, C_inv) 
    return Fij 


def converge_Fij(obs, kmax=0.5, seed=range(5), rsd='all', flag='reg', dmnu='fin',
        params='default', silent=True): 
    ''' convergence test of Fisher matrix elements  when we calculate the covariance 
    matrix or derivatives using different number of mocks. 
    '''
    ij_pairs = [] 
    ij_pairs_str = [] 
    if params == 'default': 
        theta_cosmo = ['Om', 'Ob2', 'h', 'ns', 's8', 'Mnu']
        theta_cosmo_lbls = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$'] 
    elif params == 'lcdm': 
        theta_cosmo = ['Om', 'Ob2', 'h', 'ns', 's8']
        theta_cosmo_lbls = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$'] 
    nthetas = len(theta_cosmo)  # only the cosmological parameters 

    for i in range(nthetas): 
        for j in range(i, nthetas): 
            ij_pairs.append((i,j))
            ij_pairs_str.append(','.join([theta_cosmo_lbls[i], theta_cosmo_lbls[j]]))
    ij_pairs = np.array(ij_pairs) 

    print('--- Ncov test ---' ) 
    # convegence of covariance matrix 
    ncovs = [5000, 7500, 10000, 12000, 14000, 15000]
    # read in fisher matrix (Fij)
    Fijs = []
    for ncov in ncovs: 
        Fijs.append(
                _converge_FisherMatrix(obs, 
                    kmax=kmax, 
                    seed=seed, 
                    rsd=rsd, 
                    flag=flag,
                    params=params, 
                    dmnu=dmnu,
                    Ncov=ncov, 
                    Nderiv=None)) 
    Fijs = np.array(Fijs) 

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
    sub.set_xlabel(r"$N_{\rm cov}$ simulations", labelpad=10, fontsize=25) 
    sub.set_xlim(5000, 15000) 
    sub.set_ylabel(r'$F_{ij}(N_{\rm cov})/F_{ij}(N_{\rm cov}=%i)$' % sub.get_xlim()[1], fontsize=25)
    sub.set_ylim([1. - 0.3*len(ij_pairs), 1.3]) 
    sub.set_yticks([1. - 0.3 * ii for ii in range(len(ij_pairs))])
    sub.set_yticklabels(ij_pairs_str) 

    print('--- Nderiv. test ---' ) 
    # convergence of derivatives 
    assert rsd == 'all'
    nderivs = [500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 7500]

    # read in fisher matrix (Fij)
    Fijs = []
    for nderiv in nderivs: 
        Fijs.append(
                _converge_FisherMatrix(obs,
                    kmax=kmax,
                    seed=seed, 
                    rsd=rsd, 
                    flag=flag, 
                    params=params,
                    dmnu=dmnu,
                    Ncov=None, 
                    Nderiv=nderiv)) 
    Fijs = np.array(Fijs) 
    sub = fig.add_subplot(122)
    for _i, ij in enumerate(ij_pairs): 
        sub.fill_between([100, 8000], [1.-_i*0.3-0.05, 1.-_i*0.3-0.05], [1.-_i*0.3+0.05, 1.-_i*0.3+0.05],
                color='k', linewidth=0, alpha=0.25) 
        sub.plot([100., 8000.], [1.-_i*0.3, 1.-_i*0.3], c='k', ls='--', lw=1) 
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

    ffig = os.path.join(dir_doc, 'converge.%sFij%s%s%s%s.dmnu_%s.kmax%.1f.png' %
            (obs, _params_str(params), _rsd_str(rsd), _seed_str(seed), _flag_str(flag), dmnu, kmax))
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') # latex version 
    return None


def converge_P02B_Forecast(kmax=0.5, seed=range(5), rsd='all', flag='reg', dmnu='fin',
        params='default'):
    ''' convergence test of cosmological parameter constraints 
    '''
    if params == 'default': 
        theta_cosmo = ['Om', 'Ob2', 'h', 'ns', 's8', 'Mnu']
        theta_cosmo_lbls = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$'] 
    elif params == 'lcdm': 
        theta_cosmo = ['Om', 'Ob2', 'h', 'ns', 's8']
        theta_cosmo_lbls = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$'] 
    else: 
        raise ValueError

    # convegence of covariance matrix 
    ncovs = [3000, 4000, 5000, 7500, 10000, 12000, 14000, 15000]
    # read in fisher matrix (Fij)
    Finvs, Fiis = [], [] 
    for ncov in ncovs: 
        Fij = _converge_FisherMatrix('p02bk', 
                seed=seed, 
                kmax=kmax, 
                rsd=rsd, 
                flag=flag,
                dmnu=dmnu, 
                params=params, 
                Ncov=ncov, 
                Nderiv=None)
        Finvs.append(np.linalg.inv(Fij)) # invert fisher matrix 
        Fiis.append(np.diag(Fij)) 
    Finvs = np.array(Finvs) 
    Fiis = np.array(Fiis) 

    fig = plt.figure(figsize=(12,6))
    sub = fig.add_subplot(121) 
    sub.plot([3000, 15000], [1., 1.], c='k', ls='--', lw=1) 
    sub.plot([3000, 15000], [0.9, 0.9], c='k', ls=':', lw=1) 

    for i in range(len(theta_cosmo)): 
        sig_theta = np.sqrt(Finvs[:,i,i]) 
        sigii_theta = 1./np.sqrt(Fiis[:,i]) 
        sub.plot(ncovs, sig_theta/sig_theta[-1], label=r'$%s$' % theta_cosmo_lbls[i]) 

        print('--- %s ---' % theta_lbls[i]) 
        print(sig_theta/sig_theta[-1]) 

    sub.set_xlabel(r"$N_{\rm cov}$", labelpad=10, fontsize=25) 
    sub.set_xlim(3000, 15000) 
    sub.set_ylabel(r'$\sigma_\theta(N_{\rm cov})/\sigma_\theta(N_{\rm cov}=15000)$', fontsize=25)
    sub.set_ylim(0.5, 1.1) 

    print('--- Nderiv test ---' ) 
    # convergence of derivatives 
    assert rsd == 'all'
    nderivs = [500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 7500]
    
    # read in fisher matrix (Fij)
    Finvs, Fiis = [], [] 
    for nderiv in nderivs: 
        Fij = _converge_FisherMatrix('p02bk', 
                seed=seed,
                kmax=kmax, 
                rsd=rsd, 
                flag=flag,
                dmnu=dmnu, 
                params=params, 
                Ncov=None, 
                Nderiv=nderiv)
        Finvs.append(np.linalg.inv(Fij)) # invert fisher matrix 
        Fiis.append(Fij) 
    Finvs = np.array(Finvs) 
    Fiis = np.array(Fiis) 

    sub = fig.add_subplot(122)
    sub.plot([100., 8000.], [1., 1.], c='k', ls='--', lw=1) 
    sub.plot([100., 8000.], [0.9, 0.9], c='k', ls=':', lw=1) 
    for i in range(len(theta_cosmo)): 
        sig_theta = np.sqrt(Finvs[:,i,i]) 
        sigii_theta = 1./np.sqrt(Fiis[:,i,i]) 
        sub.plot(nderivs, sig_theta/sig_theta[-1], label=(r'$%s$' % theta_cosmo_lbls[i]))
        print('--- %s ---' % theta_lbls[i]) 
        print(sig_theta/sig_theta[-1]) 
    sub.legend(loc='lower right', fontsize=20) 
    sub.set_xlabel(r"$N_{\rm deriv.}$", labelpad=10, fontsize=25) 
    sub.set_xlim(200, nderivs[-1]) 
    sub.set_ylabel(r'$\sigma_\theta(N_{\rm deriv.})/\sigma_\theta(N_{\rm deriv.}=%i)$' % nderivs[-1], fontsize=25)
    sub.set_ylim([0.5, 1.1]) 
    fig.subplots_adjust(wspace=0.25) 

    ffig = os.path.join(dir_doc, 
            'converge.p02bkFisher%s%s%s%s.dmnu_%s.kmax%.1f.png' %
            (_params_str(params), _rsd_str(rsd), _seed_str(seed), _flag_str(flag), dmnu, kmax))
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') # latex friednly
    return None


def plot_converge_dP02B(theta, kmax=0.5, seed=range(5), rsd='all', flag='reg', dmnu='fin', log=True): 
    ''' compare derivatives w.r.t. the different thetas
    '''
    # y range of P0, P2 plots 
    logplims = [(-10., 5.), (-2, 15), (-3., 1.), (-4., 0.), (-2., 2.), (-0.5, 0.5), (-2., 4), (-2., 1.), None, (-1., 2.), (-5., 1.)] 
    logblims = [(-13., -6.), (-2., 24.), (-5., -0.5), (-5., -0.5), (-2., 2.), (0.0, 0.7), (3., 6.), None, None, (2, 6), None] 

    fid = 'fiducial' 
    if theta == 'Mnu': fid = 'fiducial_za'
    qhod_p = Obvs.quijhod_Pk(fid, flag=flag, rsd=rsd, silent=False) 
    _k, p0g, p2g  = qhod_p['k'], np.average(qhod_p['p0k'], axis=0), np.average(qhod_p['p2k'], axis=0)
    k = np.concatenate([_k, _k]) 

    qhod_b = Obvs.quijhod_Bk(fid, flag=flag, rsd=rsd, silent=False) 
    _ik, _jk, _lk, b0g = qhod_b['k1'], qhod_b['k2'], qhod_b['k3'], np.average(qhod_b['b123'], axis=0) 
    pbg_fid = np.concatenate([p0g, p2g, b0g]) 

    fig = plt.figure(figsize=(30,3))
    gs = mpl.gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1,1,5], hspace=0.1, wspace=0.15) 
    sub0 = plt.subplot(gs[0])
    sub1 = plt.subplot(gs[1])
    sub2 = plt.subplot(gs[2])

    for nderiv, clr, lw in zip([500, 3000, 7500], ['C1', 'C0', 'k'], [2, 1, 0.5]): 
        _, _, _, _, dpb, _ = dP02Bk(theta, seed=seed, rsd=rsd, dmnu=dmnu,
                fscale_pk=fscale_pk, Nderiv=nderiv, silent=False) 
        if log: dpb /= pbg_fid
        
        nk0 = int(len(k)/2) 
        kp  = k[:nk0]
        dp0 = dpb[:nk0]
        dp2 = dpb[nk0:len(k)]
        db  = dpb[len(k):] 

        # k limits 
        pklim = (kp < kmax) 
        bklim = ((_ik*kf <= kmax) & (_jk*kf <= kmax) & (_lk*kf <= kmax))
        
        # plot dP0/dtheta and dP2/theta
        sub0.plot(kp[pklim], dp0[pklim], c=clr)
        sub1.plot(kp[pklim], dp2[pklim], c=clr)
        sub2.plot(range(np.sum(bklim)), db[bklim], c=clr, lw=lw,
                label=r'$N_{\rm deriv} = %i$' % nderiv)

    sub0.set_xscale('log') 
    sub0.set_xlim(5e-3, kmax) 
    sub0.set_xlabel('k [$h$/Mpc]', fontsize=25) 
    if log: sub0.set_ylim(logplims[thetas.index(theta)])
    else: sub0.set_yscale('symlog', linthreshy=1e3) 
    sub0.set_ylabel(r'${\rm d} %s X/{\rm d}\theta (N_{\rm deriv})$' % (['', '\log'][log]), fontsize=25) 
    sub0.text(0.01, 0.9, '$P_0$', ha='left', va='top', 
            transform=sub0.transAxes, fontsize=25)

    sub1.set_xscale('log') 
    sub1.set_xlim(5e-3, kmax) 
    sub1.set_xlabel('k [$h$/Mpc]', fontsize=25) 
    if log: sub1.set_ylim(logplims[thetas.index(theta)])
    else: sub1.set_yscale('symlog', linthreshy=1e3) 
    sub1.text(0.01, 0.9, '$P_2$', ha='left', va='top', 
            transform=sub1.transAxes, fontsize=25)
    #sub1.set_ylabel(r'${\rm d} %s P_2/{\rm d}\theta (N_{\rm deriv})$' % (['', '\log'][log]), fontsize=25) 
    
    sub2.legend(loc='upper right', ncol=3, handletextpad=0.1, fontsize=20) 
    sub2.set_xlim(0, np.sum(bklim)) 
    if not log: sub2.set_yscale('symlog', linthreshy=1e8) 
    else: sub2.set_ylim(logblims[thetas.index(theta)]) 
    sub2.text(0.01, 0.9, '$B_0$', ha='left', va='top', 
            transform=sub2.transAxes, fontsize=25)
    sub2.text(0.99, 0.9, theta_lbls[thetas.index(theta)], ha='right', va='top', 
            transform=sub2.transAxes, fontsize=25, bbox=dict(facecolor='white', alpha=0.75, edgecolor='None'))
    sub2.set_xlabel('triangles', fontsize=25) 
    #sub2.set_ylabel(r'${\rm d} %s B_0/{\rm d}\theta (N_{\rm deriv})$' % (['', '\log'][log]), fontsize=25) 
    ffig = os.path.join(dir_hod, 'figs', 
            'converge.d%sP02Bd%s%s%s%s.%s.png' % (['', 'log'][log], theta,
                _rsd_str(rsd), _seed_str(seed), _flag_str(flag), dmnu))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def _converge_P02_Forecast(kmax=0.5, rsd='all', flag='reg', dmnu='fin',
        params='default'):
    ''' convergence test of cosmological parameter constraints for P02 only
    '''
    # convegence of covariance matrix 
    ncovs = [3000, 4000, 5000, 7500, 10000, 12000, 14000, 15000]
    # read in fisher matrix (Fij)
    Finvs, Fiis = [], [] 
    for ncov in ncovs: 
        Fij = _converge_FisherMatrix('p02k', kmax=kmax, rsd=rsd, flag=flag,
                dmnu=dmnu, params=params, Ncov=ncov, Nderiv=None)
        Finvs.append(np.linalg.inv(Fij)) # invert fisher matrix 
        Fiis.append(np.diag(Fij)) 
    Finvs = np.array(Finvs) 
    Fiis = np.array(Fiis) 

    fig = plt.figure(figsize=(12,6))
    sub = fig.add_subplot(121) 
    sub.plot([3000, 15000], [1., 1.], c='k', ls='--', lw=1) 
    sub.plot([3000, 15000], [0.9, 0.9], c='k', ls=':', lw=1) 

    for i in range(6): 
        sig_theta = np.sqrt(Finvs[:,i,i]) 
        sigii_theta = 1./np.sqrt(Fiis[:,i]) 
        sub.plot(ncovs, sig_theta/sig_theta[-1], label=r'$%s$' % theta_lbls[i]) 

        print('--- %s ---' % theta_lbls[i]) 
        print(sig_theta/sig_theta[-1]) 

    sub.set_xlabel(r"$N_{\rm cov}$", labelpad=10, fontsize=25) 
    sub.set_xlim(3000, 15000) 
    sub.set_ylabel(r'$\sigma_\theta(N_{\rm cov})/\sigma_\theta(N_{\rm cov}=15000)$', fontsize=25)
    sub.set_ylim(0.5, 1.1) 

    print('--- Nderiv test ---' ) 
    # convergence of derivatives 
    #if rsd == 'all': nderivs = [100, 200, 400, 600, 800, 1000, 1200, 1400, 1500]
    #else: nderivs = [100, 200, 300, 350, 400, 450, 475, 500]
    if rsd == 'all': nderivs = [100, 500, 1000, 1400, 1500]
    else: nderivs = [100, 300, 450, 475, 500]
    # read in fisher matrix (Fij)
    Finvs, Fiis = [], [] 
    for nderiv in nderivs: 
        Fij = _converge_FisherMatrix('p02k', kmax=kmax, rsd=rsd, flag=flag, dmnu=dmnu, 
                params=params, Ncov=None, Nderiv=nderiv)
        Finvs.append(np.linalg.inv(Fij)) # invert fisher matrix 
        Fiis.append(Fij) 
    Finvs = np.array(Finvs) 
    Fiis = np.array(Fiis) 

    if params == 'default': 
        _theta_lbls = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$', 
                r'$\log M_{\rm min}$', r'$\sigma_{\log M}$', r'$\log M_0$', r'$\alpha$', r'$\log M_1$']
    elif params == 'lcdm': 
        _theta_lbls = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', 
                r'$\log M_{\rm min}$', r'$\sigma_{\log M}$', r'$\log M_0$', r'$\alpha$', r'$\log M_1$']

    sub = fig.add_subplot(122)
    sub.plot([100., 3000.], [1., 1.], c='k', ls='--', lw=1) 
    sub.plot([100., 3000.], [0.9, 0.9], c='k', ls=':', lw=1) 
    for i in range(Fij.shape[0]): 
        sig_theta = np.sqrt(Finvs[:,i,i]) 
        sigii_theta = 1./np.sqrt(Fiis[:,i,i]) 
        if i < 6: 
            sub.plot(nderivs, sig_theta/sig_theta[-1], 
                    label=(r'$%s$' % _theta_lbls[i]))
        else: 
            sub.plot(nderivs, sig_theta/sig_theta[-1], ls='--', 
                    label=(r'$%s$' % _theta_lbls[i]))
        print('--- %s ---' % theta_lbls[i]) 
        print(sig_theta/sig_theta[-1]) 
    sub.legend(loc='lower right', fontsize=20) 
    sub.set_xlabel(r"$N_{\rm deriv.}$", labelpad=10, fontsize=25) 
    sub.set_xlim(200, nderivs[-1]) 
    sub.set_ylabel(r'$\sigma_\theta(N_{\rm deriv.})/\sigma_\theta(N_{\rm deriv.}=%i)$' % nderivs[-1], fontsize=25)
    sub.set_ylim([0.5, 1.1]) 
    fig.subplots_adjust(wspace=0.25) 

    ffig = os.path.join(dir_hod, 'figs',
            'converge.p02kFisher%s%s%s.dmnu_%s.kmax%.1f.png' %
            (_params_str(params), _rsd_str(rsd), _flag_str(flag), dmnu, kmax))
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') # latex friednly
    return None


def _signal_to_noise(): 
    ''' calculate SN as a function of kmax  using Eq. (47) in Chan & Blot (2017). 
    '''
    # galaxy P  
    quij = Obvs.quijhod_Pk('fiducial', rsd='all', flag='reg', silent=False) 
    k_p = np.concatenate([quij['k'], quij['k']]) 
    pg = np.concatenate([np.average(quij['p0k'], axis=0),
        np.average(quij['p2k'], axis=0)]) 
    # shotnoise uncorrected Pg
    pgsn = np.concatenate([quij['p0k'] + quij['p_sn'][:,None], quij['p2k']], axis=1) 
    
    # galaxy B 
    quij = Obvs.quijhod_Bk('fiducial', rsd='all', flag='reg', silent=False)  
    bg = np.average(quij['b123'], axis=0) 
    bgsn = quij['b123'] + quij['b_sn'] # uncorrected for shot noise  
    i_k, l_k, j_k = quij['k1'], quij['k2'], quij['k3'] 

    # halo P 
    quij = Obvs.quijotePk('fiducial', rsd='all', flag='reg', silent=False) 
    ph = np.concatenate([np.average(quij['p0k'], axis=0),
        np.average(quij['p2k'], axis=0)]) 
    phsn = np.concatenate([quij['p0k'] + quij['p_sn'][:,None], quij['p2k']],
            axis=1) 
    # halo B 
    quij = Obvs.quijoteBk('fiducial', rsd='all', flag='reg', silent=False)  
    bh = np.average(quij['b123'], axis=0) 
    bhsn = quij['b123'] + quij['b_sn'] # uncorrected for shot noise  

    kmaxs = np.linspace(0.04, 0.75, 10) 
    SN_Bg, SN_Pg = [], [] 
    SN_Bh, SN_Ph = [], [] 
    for kmax in kmaxs: 
        # k limit 
        pklim = (k_p <= kmax) 
        bklim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 
        
        C_bk = np.cov(bgsn[:,bklim].T) # calculate the Bg covariance
        Ci_B = np.linalg.inv(C_bk) 
        C_pk = np.cov(pgsn[:,pklim].T) # calculate the P covariance  
        Ci_P = np.linalg.inv(C_pk) 
        sn_b_i = np.sqrt(np.matmul(bg[bklim].T, np.matmul(Ci_B, bg[bklim])))
        sn_p_i = np.sqrt(np.matmul(pg[pklim].T, np.matmul(Ci_P, pg[pklim])))
        SN_Bg.append(sn_b_i)
        SN_Pg.append(sn_p_i)

        C_bk = np.cov(bhsn[:,bklim].T) # calculate the B covariance
        Ci_B = np.linalg.inv(C_bk) 
        C_pk = np.cov(phsn[:,pklim].T) # calculate the P covariance  
        Ci_P = np.linalg.inv(C_pk) 
        sn_b_i = np.sqrt(np.matmul(bh[bklim].T, np.matmul(Ci_B, bh[bklim])))
        sn_p_i = np.sqrt(np.matmul(ph[pklim].T, np.matmul(Ci_P, ph[pklim])))
        SN_Bh.append(sn_b_i)
        SN_Ph.append(sn_p_i)
        print('kmax=%.2f: Pg:%f, Bg:%f, Ph:%f, Bh:%f' % (kmax, SN_Pg[-1], SN_Bg[-1], SN_Ph[-1], SN_Bh[-1])) 
    
    # Chan & Blot numbers 
    #cb_p_kmax = np.array([0.04869675251658636, 0.06760829753919823, 0.07673614893618194, 0.08609937521846012, 0.10592537251772895, 0.11415633046188466, 0.1341219935405372, 0.14288939585111032, 0.1612501027337743, 0.17080477200597086, 0.18728370830175495, 0.19838096568365063, 0.21134890398366477, 0.22908676527677735, 0.25703957827688645, 0.3037386091946106, 0.3823842536581126, 0.44668359215096326, 0.5339492735741767, 0.6760829753919819, 0.8511380382023763, 0.9828788730000325, 1.1481536214968824]) 
    #cb_p_sn = np.array([9.554281212853747, 14.56687309222178, 16.512840354510395, 18.506604023110235, 21.462641346777612, 22.20928889966336, 24.054044983627143, 24.329804200374014, 25.176195307362626, 25.756751569612643, 25.756751569612643, 26.052030872682657, 25.756751569612643, 26.35069530242852, 26.35069530242852, 26.35069530242852, 26.35069530242852, 26.65278366645541, 26.35069530242852, 26.65278366645541, 26.65278366645541, 26.65278366645541, 27.2673896573547]) 

    #cb_b_kmax = np.array([0.03845917820453539, 0.05754399373371572, 0.07629568911615335, 0.09495109992021988, 0.11415633046188466, 0.1333521432163325, 0.15310874616820308, 0.17179083871575893, 0.19164610627353873, 0.21134890398366477, 0.2277718241857323, 0.24688801049062103, 0.26607250597988114, 0.2834653633489668, 0.3054921113215514, 0.325461783498046,  0.3487385841352186]) 
    #cb_b_sn = np.array([2.027359157379195,3.5440917014545286,5.1041190104348715,6.338408101544683,7.023193813563101,7.8711756484503095,8.33282150847736,8.922674486302233,9.233078642191177,9.445990941294742,9.886657856152153,10,10.230597298425085,10.347882416158368,10.707867049863955,10.707867049863955,10.830623660351296]) 

    fig = plt.figure(figsize=(6,5)) 
    sub = fig.add_subplot(111)
    sub.plot(kmaxs, SN_Bg, c='k', label='$B_g$') 
    sub.plot(kmaxs, SN_Bh, c='k', ls=':', label='$B_h$') 
    #sub.plot(cb_b_kmax, cb_b_sn, c='k', ls='--') 
    sub.plot(kmaxs, SN_Pg, c='C0', label='$P_g$') 
    sub.plot(kmaxs, SN_Ph, c='C0', ls=':', label='$P_h$') 
    sub.plot([0.2, 0.2], [0.5, 3e2]) 
    sub.plot([0.5, 0.5], [0.5, 3e2]) 
    #sub.plot(cb_p_kmax, cb_p_sn, c='C0', ls='--', label='Chan and Blot 2017') 
    sub.legend(loc='lower right', fontsize=15) 
    sub.set_xlabel(r'$k_{\rm max}$ [$h$/Mpc]', fontsize=20) 
    sub.set_xscale('log')
    sub.set_xlim(3e-2, 1.) 
    sub.set_ylabel(r'S/N', fontsize=20) 
    sub.set_yscale('log') 
    sub.set_ylim(0.5, 3e2) 
    ffig = os.path.join(dir_doc, '_signal_to_noise.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def _P02B_sigmalogM(kmax):
    ''' fisher forecast comparison for  P0+P2+B for quijote observables where we 
    marginalize over theta_nuis parameters. 
    
    :param kmax: (default: 0.5) 
        kmax of the analysis 
    :param rsd: (default: True) 
        specifies the rsd set up for the B(k) derivative. 
        If rsd == 'all', include 3 RSD directions. 
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
    # fisher matrices (Fij)
    pbkFij_fid = FisherMatrix('p02bk', kmax=kmax, seed=range(5), rsd='all',
            dmnu='fin', params='default', silent=False)
    pbkFij_hir = FisherMatrix('p02bk', kmax=kmax, seed=range(5), rsd='all',
            dmnu='fin', params='default_hr', silent=False)

    cond = np.linalg.cond(pbkFij_fid)
    if cond > 1e16: print('Fij is ill-conditioned %.5e' % cond)
    pbkFinv_fid = np.linalg.inv(pbkFij_fid) 
    pbkFinv_hir = np.linalg.inv(pbkFij_hir) 

    print('--- thetas ---')
    print('P+B fid sigmas %s' % ', '.join(['%.4f' % sii for sii in
        np.sqrt(np.diag(pbkFinv_fid))]))
    print('P+B HR  sigmas %s' % ', '.join(['%.4f' % sii for sii in
        np.sqrt(np.diag(pbkFinv_hir))]))
        
    ntheta = len(thetas) 
    _thetas = copy(thetas) 
    _theta_lbls = copy(theta_lbls) 
    _theta_fid = theta_fid.copy() # fiducial thetas
    #_theta_dlims = [0.0425, 0.016, 0.16, 0.15, 0.09, 0.25, 0.4, 0.75, 0.4, 0.2, 0.25]
    _theta_dlims = [0.05, 0.02, 0.25, 0.3, 0.125, 0.45, 0.6, 1.25, 0.75, 0.375, 0.275]
    
    n_nuis = 0  

    Finvs   = [pbkFinv_fid, pbkFinv_hir]
    colors  = ['C0', 'C1']
    sigmas  = [[2], [1, 2]]
    alphas  = [[0.9], [1., 0.7]]

    fig = plt.figure(figsize=(20, 20))
    gs = mpl.gridspec.GridSpec(2, 1, figure=fig, height_ratios=[5,5+n_nuis], hspace=0.1) 
    gs0 = mpl.gridspec.GridSpecFromSubplotSpec(5, ntheta+n_nuis-1, subplot_spec=gs[0])
    gs1 = mpl.gridspec.GridSpecFromSubplotSpec(5+n_nuis, ntheta+n_nuis-1, subplot_spec=gs[1])
    
    for i in range(ntheta+n_nuis-1): 
        for j in range(i+1, ntheta+n_nuis): 
            theta_fid_i, theta_fid_j = _theta_fid[_thetas[i]], _theta_fid[_thetas[j]] # fiducial parameter 
            
            if j < 6: sub = plt.subplot(gs0[j-1,i]) # cosmo. params
            else: sub = plt.subplot(gs1[j-6,i]) # the rest

            for _i, Finv in enumerate(Finvs):
                try:
                    Finv_sub = np.array([[Finv[i,i], Finv[i,j]], [Finv[j,i], Finv[j,j]]]) # sub inverse fisher matrix 
                except IndexError: 
                    continue 
                Forecast.plotEllipse(Finv_sub, sub, theta_fid_ij=[theta_fid_i, theta_fid_j], color=colors[_i], 
                        sigmas=sigmas[_i], alphas=alphas[_i])

            sub.set_xlim(theta_fid_i - _theta_dlims[i], theta_fid_i + _theta_dlims[i])
            sub.set_ylim(theta_fid_j - _theta_dlims[j], theta_fid_j + _theta_dlims[j])
            if i == 0:   
                sub.set_ylabel(_theta_lbls[j], labelpad=5, fontsize=24) 
                sub.get_yaxis().set_label_coords(-0.35,0.5)
            else: 
                sub.set_yticks([])
                sub.set_yticklabels([])
            
            if j == ntheta+n_nuis-1: 
                sub.set_xlabel(_theta_lbls[i], labelpad=7, fontsize=24) 
            elif j == 5: 
                sub.set_xlabel(_theta_lbls[i], fontsize=24) 
            else: 
                sub.set_xticks([])
                sub.set_xticklabels([]) 

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.fill_between([],[],[], color=colors[0], label=r'fiducial ($\sigma_{\log M} = 0.2~{\rm dex}$)') 
    bkgd.fill_between([],[],[], color=colors[1], label=r'high res. ($\sigma_{\log M} = 0.55~{\rm dex}$)')
    bkgd.legend(loc='upper right', bbox_to_anchor=(0.875, 0.775), handletextpad=0.2, fontsize=25)
    bkgd.text(0.85, 0.62, r'$k_{\rm max} = %.1f$; $z=0.$' % kmax, ha='right', va='bottom', 
            transform=bkgd.transAxes, fontsize=25)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.05, hspace=0.05) 
    
    ffig = os.path.join(dir_doc, '_Fisher.p02bk.sigmalogM.kmax%.2f.png' % kmax)
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def _write_FisherMatrix(obs='p02bk', kmax=0.5, seed=range(5), rsd='all', dmnu='fin',
        theta_nuis=None, planck=False, silent=True): 
    ''' write out Fisher matrix for Ema 
    '''
    # fisher matrices (Fij)
    pk_theta_nuis = copy(theta_nuis)
    bk_theta_nuis = copy(theta_nuis)
    if theta_nuis is not None: 
        if 'Bsn' in theta_nuis: pk_theta_nuis.remove('Bsn') 
        if 'b2' in theta_nuis: pk_theta_nuis.remove('b2') 
        if 'g2' in theta_nuis: pk_theta_nuis.remove('g2') 

    pbkFij  = FisherMatrix(obs, kmax=kmax, seed=seed, rsd=rsd, dmnu=dmnu,
            theta_nuis=theta_nuis, silent=silent)
    cond = np.linalg.cond(pbkFij)
    if cond > 1e16: raise ValueError('Fij is ill-conditioned %.5e' % cond)
    
    if planck: # add planck prior 
        # read in planck prior fisher (order is Om, Ob, h, ns, s8 and Mnu) 
        _Fij_planck = np.load(os.path.join(UT.dat_dir(), 'Planck_2018_s8.npy')) 
        pbkFij[:6,:6] += _Fij_planck


    # writeout Fisher matrix
    hdr = '# %s' % (', '.join(thetas)) 
    print(hdr) 

    nuis_str = ''
    if theta_nuis is not None: 
        if 'Amp' in theta_nuis: nuis_str += 'b'
        if 'Mmin' in theta_nuis: nuis_str += 'Mmin'
        if ('Asn' in theta_nuis) or ('Bsn' in theta_nuis): nuis_str += 'SN'
        if 'b2' in theta_nuis: nuis_str += 'b2'
        if 'g2' in theta_nuis: nuis_str += 'g2'
    planck_str = ''
    if planck: planck_str = '.planck'
    fout = os.path.join(dir_doc, 
            'FisherMatrix.%s%s.dmnu_%s.kmax%.2f%s%s%s.dat' % 
            (obs, nuis_str, dmnu, kmax, _seed_str(seed), _rsd_str(rsd), planck_str))

    np.savetxt(fout, pbkFij, header=hdr) 
    return None 


def _forecast_latex_table(seed=range(5), rsd='all', dmnu='fin'): 
    ''' latex table of Pell, B, Pell + B 
    '''
    # planck prior 
    _Fij_planck = np.load(os.path.join(UT.dat_dir(), 'Planck_2018_s8.npy')) 

    Finv_kmax0p2, Finv_kmax0p5 = [], []
    Finv_planck_kmax0p2, Finv_planck_kmax0p5 = [], []

    for obs in ['p02k', 'bk', 'p02bk']: 
        _Fij_kmax0p2 = FisherMatrix(obs, kmax=0.2, seed=seed, rsd=rsd, dmnu=dmnu, theta_nuis=None, silent=True)
        _Fij_kmax0p5 = FisherMatrix(obs, kmax=0.5, seed=seed, rsd=rsd, dmnu=dmnu, theta_nuis=None, silent=True)

        _Fij_planck_kmax0p2 = _Fij_kmax0p2.copy() 
        _Fij_planck_kmax0p5 = _Fij_kmax0p5.copy() 
        _Fij_planck_kmax0p2[:6,:6] += _Fij_planck
        _Fij_planck_kmax0p5[:6,:6] += _Fij_planck

        Finv_kmax0p2.append(np.linalg.inv(_Fij_kmax0p2))
        Finv_kmax0p5.append(np.linalg.inv(_Fij_kmax0p5))
        Finv_planck_kmax0p2.append(np.linalg.inv(_Fij_planck_kmax0p2))
        Finv_planck_kmax0p5.append(np.linalg.inv(_Fij_planck_kmax0p5))

    i_Mnu = thetas.index('Mnu')
    i_order = np.concatenate([[i_Mnu], range(i_Mnu), range(i_Mnu+1, len(thetas))])

    for i in i_order: 
        constraints = '&'.join([
            ' %.3f (%.3f) ' % (np.sqrt(np.diag(Finv_kmax0p2[0]))[i], np.sqrt(np.diag(Finv_planck_kmax0p2[0]))[i]),
            ' %.3f (%.3f) ' % (np.sqrt(np.diag(Finv_kmax0p2[1]))[i], np.sqrt(np.diag(Finv_planck_kmax0p2[1]))[i]),
            ' %.3f (%.3f) ' % (np.sqrt(np.diag(Finv_kmax0p2[2]))[i], np.sqrt(np.diag(Finv_planck_kmax0p2[2]))[i]),
            ' %.3f (%.3f) ' % (np.sqrt(np.diag(Finv_kmax0p5[0]))[i], np.sqrt(np.diag(Finv_planck_kmax0p5[0]))[i]),
            ' %.3f (%.3f) ' % (np.sqrt(np.diag(Finv_kmax0p5[1]))[i], np.sqrt(np.diag(Finv_planck_kmax0p5[1]))[i]),
            ' %.3f (%.3f) ' % (np.sqrt(np.diag(Finv_kmax0p5[2]))[i], np.sqrt(np.diag(Finv_planck_kmax0p5[2]))[i])]) 
        if thetas[i] == 's8': 
            print(r'%s & %s \\ [3pt] \hline' % (theta_lbls[i], constraints)) 
        else: 
            print(r'%s & %s \\' % (theta_lbls[i], constraints)) 
    return None
    
# --- etc ---
def _nuis_str(theta_nuis): 
    if theta_nuis is None: nuis_str = ''
    else: 
        nuis_str = '.'
        if 'b1' in theta_nuis: nuis_str += 'b1'
        if 'Amp' in theta_nuis: nuis_str += 'b'
        if 'Mmin' in theta_nuis: nuis_str += 'Mmin'
        if ('Asn' in theta_nuis) or ('Bsn' in theta_nuis): nuis_str += 'SN'
        if 'b2' in theta_nuis: nuis_str += 'b2'
        if 'g2' in theta_nuis: nuis_str += 'g2'
    return nuis_str


def _dmnu_str(theta, dmnu): 
    if theta != 'Mnu': return ''
    return '.%s' % dmnu


def _seed_str(seed): 
    if isinstance(seed, int): 
        return '.seed%i' % seed
    else: 
        if np.array_equal(np.array(seed), np.arange(np.min(seed), np.max(seed)+1)): 
            return '.seed%ito%i' % (np.min(seed), np.max(seed))
        else: 
            return '.seed%s' % '_'.join([str(s) for s in seed])


def _rsd_str(rsd): 
    # assign string based on rsd kwarg 
    if rsd == 'all': return ''
    elif rsd in [0, 1, 2]: return '.rsd%i' % rsd
    elif rsd == 'real': return '.real'
    else: raise NotImplementedError


def _params_str(params): 
    # assign string based on rsd kwarg 
    if params == 'default': return ''
    elif params == 'lcdm': return '.lcdm'
    elif params == 'halo': return '.halo_param'
    else: raise NotImplementedError


def _flag_str(flag): 
    # assign string based on flag kwarg
    return ['.%s' % flag, ''][flag is None]


def _Nderiv_str(Nderiv): 
    if Nderiv is None: return ''
    else: return '.Nderiv%i' % Nderiv


if __name__=="__main__": 
    # covariance matrices 
    '''
        # calculate the covariance matrices 
        p02kCov(rsd=2, flag='reg', silent=False)
        bkCov(rsd=2, flag='reg', silent=False)
        fscale_Pk(rsd=2, flag='reg', silent=False)
        p02bkCov(rsd=2, flag='reg', silent=False)

        # plot the covariance matrices 
        plot_p02bkCov(kmax=0.5, rsd=2, flag='reg')
    '''
    # derivatives 
    '''
        # compare derivatives w.r.t. the cosmology + HOD parameters 
        plot_dP02B(kmax=0.5, seed=range(5), rsd='all', dmnu='fin', log=False)
        # compare derivatives for a subset of parameters to look into 
        # dgenerates with deriv. w.r.t. Mnu
        plot_dP02B_Mnu_degen(kmax=0.5, seed=range(5), rsd='all', dmnu='fin')
        # deriv. w.r.t. Mnu for different methods 
        plot_dP02B_Mnu(kmax=0.5, seed=range(5), rsd='all')
    '''
    # comparisons between galaxy and halo P and B 
    '''
        plot_PBg(rsd='all')
        _plot_PBg_PBh_SN(rsd='all')
        for theta in ['fiducial', 'Om', 'Ob2', 'h', 'ns', 's8', 'Mnu']: 
            plot_PBg_PBh(theta, rsd='all')
        for theta in ['Om', 'Ob2', 'h', 'ns', 's8', 'Mnu']: 
            plot_dPBg_dPBh(theta, rsd='all', flag='reg', dmnu='fin', log=False)
            plot_dPBg_dPBh(theta, rsd='all', flag='reg', dmnu='fin', log=True)
        _PBg_rsd_comparison(flag='reg')
        plot_bias(rsd='all', flag='reg')
        nbar()
    '''
    # convergence tests
    '''
        # convergence of derivatives
        for theta in ['Om', 'Ob2', 'h', 'ns', 's8', 'Mnu']: 
            plot_converge_dP02B(theta, kmax=0.5, seed=range(5), rsd='all', flag='reg', dmnu='fin', log=True)
        for theta in ['logMmin', 'sigma_logM', 'logM0', 'alpha', 'logM1']:
            plot_converge_dP02B(theta, kmax=0.5, seed=range(5), rsd='all', flag='reg', dmnu='fin', log=True)

        # convergence of Fisher matrix  
        converge_Fij('p02k', kmax=0.5, seed=range(5), rsd='all', flag='reg', dmnu='fin',
                params='default', silent=True)
        converge_Fij('p02bk', kmax=0.5, seed=range(5), rsd='all', flag='reg', dmnu='fin',
                params='default', silent=True)

        # convergence of forecast 
        _converge_P02_Forecast(kmax=0.5, seed=range(5), rsd='all', flag='reg', dmnu='fin',
                params='default')
        converge_P02B_Forecast(kmax=0.5, seed=range(5), rsd='all', flag='reg', dmnu='fin',
                params='default')

        # LCDM convergence tests 
        converge_Fij('p02k', kmax=0.5, rsd='all', flag='reg', dmnu='fin',
                params='lcdm', silent=True)
        converge_Fij('p02bk', kmax=0.5, rsd='all', flag='reg', dmnu='fin',
                params='lcdm', silent=True)
        _converge_P02_Forecast(kmax=0.5, rsd='all', flag='reg', dmnu='fin',
                params='lcdm')
        converge_P02B_Forecast(kmax=0.5, rsd='all', flag='reg', dmnu='fin', 
                params='lcdm')
    '''
    # forecasts 
    _P02B_Forecast_diffkmax(pkmax=0.15, bkmax=0.05, seed=range(5), rsd='all', dmnu='fin', theta_nuis=None, planck=False)
    '''
        P02B_Forecast(kmax=0.5, seed=range(5), rsd='all', dmnu='fin', theta_nuis=None, planck=False)
        P02B_Forecast(kmax=0.5, seed=range(5), rsd='all', dmnu='fin', theta_nuis=None, planck=True)
        P02B_Forecast_kmax(seed='all', rsd='all', dmnu='fin', theta_nuis=None)
        for theta_nuis in [None, ['Asn', 'Bsn']]:
            # P0, P2, B
            P02B_Forecast(kmax=0.5, rsd='all', flag='reg', dmnu='fin', theta_nuis=theta_nuis, planck=False)
            # Forecast(kmax) 
            P02B_Forecast_kmax(rsd='all', seed=range(5), dmnu='fin', theta_nuis=None)
        _PgPh_Forecast_kmax() # comparison of Pg forecast to Ph 
        _P02B_P02Bh_Forecast_kmax(rsd='all', flag='reg', dmnu='fin', theta_nuis=None)
        _P02B_Forecast_diffkmax(pkmax=0.25, bkmax=0.1, seed=range(5), rsd='all', dmnu='fin', theta_nuis=None, planck=False)
        _write_FisherMatrix(kmax=0.5, seed=range(5), rsd='all', dmnu='fin', theta_nuis=None, planck=False)
        _write_FisherMatrix(kmax=0.5, seed=range(5), rsd='all', dmnu='fin', theta_nuis=None, planck=True)
        _write_FisherMatrix(kmax=0.5, seed=range(5), rsd='all', dmnu='p', theta_nuis=None, planck=False)
        _write_FisherMatrix(kmax=0.5, seed=range(5), rsd='all', dmnu='p', theta_nuis=None, planck=True)
        _forecast_latex_table(seed=range(5), rsd='all', dmnu='fin')
        _forecast_cross_covariance(kmax=0.5, seed=range(5), rsd='all', flag='reg', dmnu='fin')
        _P02B_dmnu(kmax=0.5, seed=range(5), rsd='all', dmnu='fin', silent=True)
    '''
    # etc
    '''
        _signal_to_noise()
    '''
    #_P02B_sigmalogM(0.5)
