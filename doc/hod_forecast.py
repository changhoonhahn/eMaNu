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
dir_hod = os.path.join(UT.dat_dir(), 'hod_forecast') 

fscale_pk = 1e5 # see fscale_Pk() for details

thetas = ['Om', 'Ob2', 'h', 'ns', 's8', 'Mnu', 'logMmin', 'sigma_logM', 'logM0', 'alpha', 'logM1'] 
theta_lbls = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$', 
        r'$\log M_{\rm min}$', r'$\sigma_{\log M}$', r'$\log M_0$', r'$\alpha$', r'$\log M_1$']
theta_fid = {'Mnu': 0., 'Ob': 0.049, 'Ob2': 0.049, 'Om': 0.3175, 'h': 0.6711,  'ns': 0.9624,  's8': 0.834, 
        'logMmin': 13.65, 'sigma_logM': 0.2, 'logM0': 14.0, 'alpha': 1.1, 'logM1': 14.0} # fiducial theta 
# nuisance parameters
theta_nuis_lbls = {'Amp': "$b'$", 'Asn': r"$A_{\rm SN}$", 'Bsn': r"$B_{\rm SN}$", 'b2': '$b_2$', 'g2': '$g_2$'}
theta_nuis_fids = {'Amp': 1., 'Asn': 1e-6, 'Bsn': 1e-3, 'b2': 1., 'g2': 1.} 
theta_nuis_lims = {'Amp': (0.75, 1.25), 'Asn': (0., 1.), 'Bsn': (0., 1.), 'b2': (0.95, 1.05), 'g2': (0.95, 1.05)} 

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
    fig.savefig(os.path.join(dir_hod, 'fscale_Pk.png'), bbox_inches='tight')
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
    ffig = os.path.join(dir_hod, 
            'quijote_p02bCov_kmax%s%s%s.png' % (str(kmax).replace('.', ''), _rsd_str(rsd), _flag_str(flag)))
    fig.savefig(ffig, bbox_inches='tight') 
    #fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') # latex friednly
    return None 

# --- derivatives --- 
def dP02k(theta, log=False, rsd='all', flag='reg', dmnu='fin', returnks=False, silent=True):
    ''' read in derivatives d P_l(k) / d theta  

    :param theta: 
        string that specifies the parameter to take the 
        derivative by. 
    '''
    dir_dat = os.path.join(UT.doc_dir(), 'dat') 
    fdpk = os.path.join(dir_dat, 'hod_dP02dtheta.%s%s%s.dat' % (theta, _rsd_str(rsd), _flag_str(flag))) 
    if not silent: print('--- reading %s ---' % fdpk) 
    i_dpk = 1 
    if theta == 'Mnu': 
        index_dict = {'fin': 1, 'fin0': 3, 'p': 5, 'pp': 7, 'ppp': 9, 'fin_2lpt': 11}  
        i_dpk = index_dict[dmnu] 
    if log: i_dpk += 1 
    
    if theta not in ['Bsn']: 
        k, dpdt = np.loadtxt(fdpk, skiprows=1, unpack=True, usecols=[0,i_dpk]) 
    else: 
        fdpk = os.path.join(dir_dat, 'hod_dP02dtheta.%s%s%s.dat' % ('Mnu', _rsd_str(rsd), _flag_str(flag))) 
        k, _ = np.loadtxt(fdpk, skiprows=1, unpack=True, usecols=[0,i_dpk]) 
        dpdt = np.zeros(len(k))
    if not returnks: return dpdt 
    else: return k, dpdt 


def dBk(theta, log=False, rsd='all', flag='reg', dmnu='fin', returnks=False, silent=True):
    ''' d B(k)/d theta  

    :param theta: 
        string that specifies the parameter to take the 
        derivative by. 
    '''
    dir_dat = os.path.join(UT.doc_dir(), 'dat') 
    fdbk = os.path.join(dir_dat, 'hod_dBdtheta.%s%s%s.dat' % (theta, _rsd_str(rsd), _flag_str(flag))) 
    if not silent: print('--- reading %s ---' % fdbk) 
    i_dbk = 3 
    if theta == 'Mnu': 
        index_dict = {'fin': 3, 'fin0': 5, 'p': 7, 'pp': 9, 'ppp': 11,  'fin_2lpt': 13}  
        i_dbk = index_dict[dmnu] 
    if log: i_dbk += 1 

    i_k, j_k, l_k, dbdt = np.loadtxt(fdbk, skiprows=1, unpack=True, usecols=[0,1,2,i_dbk]) 
    if not returnks: return dbdt 
    else: return i_k, j_k, l_k, dbdt 


def dP02Bk(theta, log=False, rsd='all', flag='reg', dmnu='fin', fscale_pk=1., returnks=False, silent=True):
    ''' dP0/dtt, dP2/dtt, dB/dtt
    '''
    # dP0/dtheta, dP2/dtheta
    dp = dP02k(theta, log=log, rsd=rsd, flag=flag, dmnu=dmnu, returnks=returnks, silent=silent) 
    # dB0/dtheta
    db = dBk(theta, log=log, rsd=rsd, flag=flag, dmnu=dmnu, returnks=returnks, silent=silent) 

    if returnks: 
        _k, _dp = dp 
        _ik, _jk, _lk, _db = db
        return _k, _ik, _jk, _lk, np.concatenate([fscale_pk * _dp, _db]) 
    else: 
        return np.concatenate([fscale_pk * dp, db]) 


def plot_dP02B(kmax=0.5, rsd='all', flag='reg', dmnu='fin', log=True): 
    ''' compare derivatives w.r.t. the different thetas
    '''
    # y range of P0, P2 plots 
    logplims = [(-10., 5.), (-2, 15), (-3., 1.), (-4., 0.), (-2., 2.), (-0.5, 0.5), (-2., 4), (-2., 1.), None, (-1., 2.), (-5., 1.)] 
    logblims = [(-13., -6.), (-2., 24.), (-5., -0.5), (-5., -0.5), (-2., 2.), (0.2, 0.7), (3., 6.), None, None, (2, 6), None] 


    if log: 
        qhod_p = Obvs.quijhod_Pk('fiducial', flag=flag, rsd=rsd, silent=False) 
        p0g, p2g  = np.average(qhod_p['p0k'], axis=0), np.average(qhod_p['p2k'], axis=0)

        qhod_b = Obvs.quijhod_Bk('fiducial', flag=flag, rsd=rsd, silent=False) 
        b0g = np.average(qhod_b['b123'], axis=0) 
        pbg_fid = np.concatenate([p0g, p2g, b0g]) 


    fig = plt.figure(figsize=(30,3*len(thetas)))
    gs = mpl.gridspec.GridSpec(len(thetas), 2, figure=fig, width_ratios=[1,7], hspace=0.1, wspace=0.15) 

    for i, tt in enumerate(thetas): 
        #_k, _ik, _jk, _lk, dpb = dP02Bk(tt, log=log, rsd=rsd, flag=flag, dmnu=dmnu, returnks=True, silent=False)
        _k, _ik, _jk, _lk, dpb = dP02Bk(tt, log=False, rsd=rsd, flag=flag, dmnu=dmnu, returnks=True, silent=False)
        if log: dpb /= pbg_fid
        
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
            sub.set_xlabel('k [$h$/Mpc]', fontsize=25) 
            if log: 
                sub.legend(handletextpad=0.1, loc='lower left', fontsize=15) 
            else: 
                sub.legend(handletextpad=0.1, loc='upper left', fontsize=15) 

        if not log: 
            sub.set_yscale('symlog', linthreshy=1e3) 
        else: 
            sub.set_ylim(logplims[i]) 
        if i == 5: sub.set_ylabel(r'${\rm d} %s P_\ell/{\rm d}\theta$' % (['', '\log'][log]), fontsize=25) 

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
        else: sub.set_xlabel('triangles', fontsize=25) 
        if i == 5: sub.set_ylabel(r'${\rm d} %s B_0/{\rm d}\theta$' % (['', '\log'][log]), fontsize=25) 

    ffig = os.path.join(dir_doc, 'quijote_d%sP02Bdtheta%s%s.%s.png' % (['', 'log'][log], _rsd_str(rsd), _flag_str(flag), dmnu))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def plot_dP02B_Mnu_degen(kmax=0.5, rsd='all', flag='reg', dmnu='fin'): 
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
        _k, _ik, _jk, _lk, dpb = dP02Bk(tt, log=False, rsd=rsd, flag=flag, dmnu=dmnu, returnks=True, silent=False)
        
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
            print('--- (dB/d%s)/(dB/dMnu) ---' % tt) 
            print((db/db_mnu)[bklim][::10]) 
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
    sub2.set_yscale('symlog', linthreshy=1e8) 
    sub2.set_xticklabels([]) 
    sub5.set_xlabel('triangles', fontsize=25) 
    sub2.set_ylabel(r'${\rm d} B_0/{\rm d}\theta$', fontsize=25) 
    sub5.set_ylabel(r'$({\rm d} B_0/{\rm d}\theta)/({\rm d} B_0/{\rm d}M_\nu)$', fontsize=25) 
    sub2.legend(handletextpad=0.1, loc='upper right', fontsize=20)

    ffig = os.path.join(dir_hod, 'quijote_dP02Bdtheta_Mnu_degen%s%s.%s.png' % (_rsd_str(rsd), _flag_str(flag), dmnu))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def plot_dP02B_Mnu(kmax=0.5, rsd='all', flag='reg', log=True):
    ''' compare the derivative w.r.t. Mnu for different methods 
    '''
    fig = plt.figure(figsize=(30,4))
    gs = mpl.gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1,1,8], wspace=0.2) 
    sub0 = plt.subplot(gs[0]) 
    sub1 = plt.subplot(gs[1]) 
    sub2 = plt.subplot(gs[2]) 
    
    for i, dmnu in enumerate(['fin', 'p', 'pp', 'fin_2lpt']): 
        _k, _ik, _jk, _lk, dpb = dP02B('Mnu', log=False, rsd=rsd, flag=flag, dmnu=dmnu, returnks=True, silent=False)
        
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
        sub2.plot(range(np.sum(bklim)), db[bklim], c='C%i' % i, lw=1, label=dmnu.replace('_', ' '))
    
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

    ffig = os.path.join(dir_hod, 'quijote_dP02BdMnu%s%s.png' % (_rsd_str(rsd), _flag_str(flag)))
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


def plot_dPBg_dPBh(theta, rsd='all', flag='reg', dmnu='fin', log=False): 
    ''' compare derivatives w.r.t. the different thetas
    '''
    # y range of log dP02,B plots 
    logplims = [(-10., 10.), (-2, 15), (-3., 1.), (-4., 0.), (-2., 2.), (-0.5, 0.5)] 
    logblims = [(-13., 0.), (-2., 24.), (-5., -0.5), (-5., -0.5), (-5., 2.), (-0.5, 0.7)] 

    k, ik, jk, lk, dpbg = dP02Bk(theta, log=log, rsd=rsd, flag=flag, dmnu=dmnu, returnks=True, silent=False)
    #k, ik, jk, lk, dpbg = dP02Bk(theta, log=False, rsd=rsd, flag=flag, dmnu=dmnu, returnks=True, silent=False)
    #if log: 
    #    qhod_p = Obvs.quijhod_Pk('fiducial', flag=flag, rsd=rsd, silent=False) 
    #    p0g, p2g  = np.average(qhod_p['p0k'], axis=0), np.average(qhod_p['p2k'], axis=0)

    #    qhod_b = Obvs.quijhod_Bk('fiducial', flag=flag, rsd=rsd, silent=False) 
    #    b0g = np.average(qhod_b['b123'], axis=0) 
    #    pbg_fid = np.concatenate([p0g, p2g, b0g]) 
    #    dpbg /= pbg_fid

    dpbh = dP02Bh(theta, log=log, rsd=rsd, flag=flag, dmnu=dmnu, returnks=False, silent=False)
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
    
    ffig = os.path.join(dir_hod, 
            'd%sPBg_d%sPBh.%s%s%s.png' % (['', 'log'][log], ['', 'log'][log], theta, _rsd_str(rsd), _flag_str(flag)))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


# --- power/bispectrum --- 
def plot_PBg(rsd='all', flag='reg'): 
    ''' plot Pg0, Pg2, Bg0 with Ph0, Ph2, Bh0 included for reference. 
    '''
    # read in P0g, P2g, and Bg
    qhod_p = Obvs.quijhod_Pk('fiducial', flag=flag, rsd=rsd, silent=False) 
    k, p0g, p2g  = qhod_p['k'], np.average(qhod_p['p0k'], axis=0), np.average(qhod_p['p2k'], axis=0)

    qhod_b = Obvs.quijhod_Bk('fiducial', flag=flag, rsd=rsd, silent=False) 
    i_k, j_k, l_k, b0g = qhod_b['k1'], qhod_b['k2'], qhod_b['k3'], np.average(qhod_b['b123'], axis=0) 

    # read in P0h, P2h, and Bh
    quij_p = Obvs.quijotePk('fiducial', rsd=rsd, flag=flag, silent=False) 
    p0h, p2h = np.average(quij_p['p0k'], axis=0), np.average(quij_p['p2k'], axis=0)

    quij_b = Obvs.quijoteBk('fiducial', flag=flag, rsd=rsd, silent=False) 
    b0h = np.average(quij_b['b123'], axis=0) 

    pklim = (k < 0.5) 
    bklim = ((i_k*kf <= 0.5) & (j_k*kf <= 0.5) & (l_k*kf <= 0.5)) # k limit 

    fig = plt.figure(figsize=(30,3))
    gs = mpl.gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1,6], wspace=0.15) 
    sub0 = plt.subplot(gs[0]) 
    sub1 = plt.subplot(gs[1]) 

    sub0.plot(k[pklim], p0h[pklim], c='k', lw=0.5) 
    sub0.plot(k[pklim], p2h[pklim], c='k', lw=0.5, ls='--') 

    sub0.plot(k[pklim], p0g[pklim], c='C0', label=r'$\ell = 0$')
    sub0.plot(k[pklim], np.abs(p2g[pklim]), c='C0', ls='--', label=r'$\ell = 2$')
    
    sub0.legend(loc='lower left', handletextpad=0.1, fontsize=15) 
    sub0.set_xlabel('k [$h$/Mpc]', fontsize=25) 
    sub0.set_xscale('log') 
    sub0.set_xlim(5e-3, 0.5) 
    sub0.set_ylabel('$|P^g_\ell(k)|$', fontsize=25) 
    sub0.set_yscale('log') 
    
    _bhplt, = sub1.plot(range(np.sum(bklim)), b0h[bklim], c='k', lw=0.25) 
    _bgplt, = sub1.plot(range(np.sum(bklim)), b0g[bklim], c='C0') 
    sub1.legend([_bgplt, _bhplt], [r'galaxy $P_\ell, B_0$', r'halo $P_\ell, B_0$'], loc='upper right', 
            handletextpad=0.2, fontsize=15) 
    sub1.set_xlabel('triangle configurations', fontsize=25) 
    sub1.set_xlim(0, np.sum(bklim)) 
    sub1.set_ylabel('$B^g_0(k_1, k_2, k_3)$', fontsize=25) 
    sub1.set_yscale('log') 
    sub1.set_ylim(1e7, 5e10)
    
    ffig = os.path.join(dir_doc, 'PBg%s%s.png' % (_rsd_str(rsd), _flag_str(flag)))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def plot_PBg_PBh(theta, rsd='all', flag='reg'): 
    ''' comparison between the galaxy power/bispectrum to the halo power/bispectrum
    '''
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
        qhod_p = Obvs.quijhod_Pk(tt, flag=flag, rsd=rsd, silent=False) 
        k, p0g, p2g  = qhod_p['k'], np.average(qhod_p['p0k'], axis=0), np.average(qhod_p['p2k'], axis=0)

        qhod_b = Obvs.quijhod_Bk(tt, flag=flag, rsd=rsd, silent=False) 
        i_k, j_k, l_k, b0g = qhod_b['k1'], qhod_b['k2'], qhod_b['k3'], np.average(qhod_b['b123'], axis=0) 

        # read in P0h, P2h, and Bh
        quij_p = Obvs.quijotePk(tt, rsd=rsd, flag=flag, silent=False) 
        p0h, p2h = np.average(quij_p['p0k'], axis=0), np.average(quij_p['p2k'], axis=0)

        quij_b = Obvs.quijoteBk(tt, flag=flag, rsd=rsd, silent=False) 
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
    
    ffig = os.path.join(dir_hod, 'PBg_PBh.%s%s%s.png' % (theta, _rsd_str(rsd), _flag_str(flag)))
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
    
    ffig = os.path.join(dir_hod, 'dPBg_PBh.%s%s%s.png' % (theta, _rsd_str(rsd), _flag_str(flag)))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


# --- forecasts --- 
def FisherMatrix(obs, kmax=0.5, rsd='all', flag='reg', dmnu='fin', theta_nuis=None, cross_covariance=True, Cgauss=False): 
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
    else: 
        raise NotImplementedError

    ndata = np.sum(klim) 
    f_hartlap = float(nmock - ndata - 2)/float(nmock - 1) 
    C_inv = f_hartlap * np.linalg.inv(Cov) # invert the covariance 
    
    _thetas = copy(thetas) 
    if theta_nuis is not None: _thetas += theta_nuis 
    
    # calculate the derivatives along all the thetas 
    dobs_dt = [] 
    for par in _thetas: 
        if obs == 'p02k': 
            dobs_dti = dP02k(par, rsd=rsd, flag=flag, dmnu=dmnu)
        elif obs == 'bk': 
            # rsd and flag kwargs are passed to the derivatives
            dobs_dti = dBk(par, rsd=rsd, flag=flag, dmnu=dmnu)
        elif obs == 'p02bk': 
            dobs_dti = dP02Bk(par, rsd=rsd, flag=flag, dmnu=dmnu, fscale_pk=fscale_pk) 
        dobs_dt.append(dobs_dti[klim])
            
    Fij = Forecast.Fij(dobs_dt, C_inv) 
    return Fij 


def forecast(obs, kmax=0.5, rsd='all', flag='reg', dmnu='fin', theta_nuis=None, planck=False):
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
     
    _thetas = copy(thetas) 
    _theta_lbls = copy(theta_lbls) 
    _theta_fid = theta_fid.copy() # fiducial thetas
    _theta_lims = [(0.25, 0.385), (0.02, 0.08), (0.3, 1.1), (0.6, 1.3), (0.8, 0.88), (-0.4, 0.4), (13.3, 14.), [0.1, 0.3], [13., 15.], [0.5, 1.7], [13., 15.]]

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
    if theta_nuis is not None: 
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


def P02B_Forecast(kmax=0.5, rsd='all', flag='reg', dmnu='fin', theta_nuis=None, planck=False):
    ''' fisher forecast comparison for P0+P2, B, and P0+P2+B for quijote observables where we marginalize over theta_nuis parameters. 
    
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

    pkFij   = FisherMatrix('p02k', kmax=kmax, rsd=rsd, flag=flag, dmnu=dmnu, theta_nuis=pk_theta_nuis)  
    bkFij   = FisherMatrix('bk', kmax=kmax, rsd=rsd, flag=flag, dmnu=dmnu, theta_nuis=bk_theta_nuis)  
    pbkFij  = FisherMatrix('p02bk', kmax=kmax, rsd=rsd,  flag=flag, dmnu=dmnu, theta_nuis=theta_nuis)  
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
    print('P sigmas %s' % ', '.join(['%.2e' % sii for sii in np.sqrt(np.diag(pkFinv))]))
    print('B sigmas %s' % ', '.join(['%.2e' % sii for sii in np.sqrt(np.diag(bkFinv))]))
    print('P+B sigmas %s' % ', '.join(['%.2e' % sii for sii in np.sqrt(np.diag(pbkFinv))]))
    print('improve. over P %s' % ', '.join(['%.1f' % sii for sii in np.sqrt(np.diag(pkFinv)/np.diag(pbkFinv))]))
    print('improve. over B %s' % ', '.join(['%.1f' % sii for sii in np.sqrt(np.diag(bkFinv)/np.diag(pbkFinv))]))
        
    ntheta = len(thetas) 
    _thetas = copy(thetas) 
    _theta_lbls = copy(theta_lbls) 
    _theta_fid = theta_fid.copy() # fiducial thetas
    _theta_lims = [(0.275, 0.36), (0.03, 0.07), (0.5, 0.9), (0.8124, 1.1124), (0.75, 0.93), (-0.25, 0.25), (13.4, 13.9), [-0.1, 0.5], [13.5, 14.5], [0.9, 1.3], [13.7, 14.3]]
    
    n_nuis = 0  
    if theta_nuis is not None: 
        _thetas += theta_nuis 
        _theta_lbls += [theta_nuis_lbls[tt] for tt in theta_nuis]
        for tt in theta_nuis: _theta_fid[tt] = theta_nuis_fids[tt]
        _theta_lims += [theta_nuis_lims[tt] for tt in theta_nuis]
        n_nuis = len(theta_nuis) 

    fig = plt.figure(figsize=(20, 20))
    gs = mpl.gridspec.GridSpec(2, 1, figure=fig, height_ratios=[5,5+n_nuis], hspace=0.1) 
    gs0 = mpl.gridspec.GridSpecFromSubplotSpec(5, ntheta+n_nuis-1, subplot_spec=gs[0])
    gs1 = mpl.gridspec.GridSpecFromSubplotSpec(5+n_nuis, ntheta+n_nuis-1, subplot_spec=gs[1])
    
    for i in range(ntheta+n_nuis-1): 
        for j in range(i+1, ntheta+n_nuis): 
            theta_fid_i, theta_fid_j = _theta_fid[_thetas[i]], _theta_fid[_thetas[j]] # fiducial parameter 
            
            if j < 6: sub = plt.subplot(gs0[j-1,i]) # cosmo. params
            else: sub = plt.subplot(gs1[j-6,i]) # the rest

            for _i, Finv in enumerate([pkFinv, bkFinv, pbkFinv]):
                Finv_sub = np.array([[Finv[i,i], Finv[i,j]], [Finv[j,i], Finv[j,j]]]) # sub inverse fisher matrix 
                Forecast.plotEllipse(Finv_sub, sub, theta_fid_ij=[theta_fid_i, theta_fid_j], color='C%i' % _i)

            sub.set_xlim(_theta_lims[i])
            sub.set_ylim(_theta_lims[j])
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
    bkgd.fill_between([],[],[], color='C0', label=r'$P_{g,0}(k) + P_{g,2}(k)$') 
    bkgd.fill_between([],[],[], color='C1', label=r'$B_g(k_1, k_2, k_3)$') 
    bkgd.fill_between([],[],[], color='C2', label=r'$P_{g,0} + P_{g,2} + B_g$') 
    bkgd.legend(loc='upper right', bbox_to_anchor=(0.875, 0.775), fontsize=25)
    bkgd.text(0.8, 0.61, r'$k_{\rm max} = %.1f$; $z=0.$' % kmax, ha='right', va='bottom', 
            transform=bkgd.transAxes, fontsize=25)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
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
            'quijote.p02bkFisher.%s.dmnu_%s.kmax%.2f%s%s%s.png' % 
            (nuis_str, dmnu, kmax, _rsd_str(rsd), _flag_str(flag), planck_str))
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') # latex version 
    return None 


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
        plot_dP02B(kmax=0.5, rsd='all', flag='reg', dmnu='fin', log=False)
        # compare derivatives for a subset of parameters to look into 
        # dgenerates with deriv. w.r.t. Mnu
        plot_dP02B_Mnu_degen(kmax=0.5, rsd='all', flag='reg', dmnu='fin')
        # deriv. w.r.t. Mnu for different methods 
        plot_dP02B_Mnu(kmax=0.5, rsd='all', flag='reg', log=True)
    '''
    # comparisons between galaxy and halo P and B 
    plot_PBg(rsd='all', flag='reg')
    '''
        for theta in ['fiducial', 'Om', 'Ob2', 'h', 'ns', 's8', 'Mnu']: 
            plot_PBg_PBh(theta, rsd='all', flag='reg')
        for theta in ['Om', 'Ob2', 'h', 'ns', 's8', 'Mnu']: 
            plot_dPBg_dPBh(theta, rsd='all', flag='reg', dmnu='fin', log=False)
            plot_dPBg_dPBh(theta, rsd='all', flag='reg', dmnu='fin', log=True)
    '''
    # forecasts 
    #P02B_Forecast(kmax=0.5, rsd='all', flag='reg', dmnu='fin', theta_nuis=None, planck=False)
    #P02B_Forecast(kmax=0.5, rsd='all', flag='reg', dmnu='fin', theta_nuis=None, planck=True)
    '''
        for theta_nuis in [None, ['Asn', 'Bsn']]:
            # P0,P2 only 
            forecast('p02k', kmax=0.5, rsd='all', flag='reg', dmnu='fin', theta_nuis=theta_nuis, planck=False)
            forecast('p02k', kmax=0.5, rsd='all', flag='reg', dmnu='fin', theta_nuis=theta_nuis, planck=True)
            # B only 
            forecast('bk', kmax=0.5, rsd='all', flag='reg', dmnu='fin', theta_nuis=theta_nuis, planck=False)
            forecast('bk', kmax=0.5, rsd='all', flag='reg', dmnu='fin', theta_nuis=theta_nuis, planck=True)
            # P0, P2, B
            P02B_Forecast(kmax=0.5, rsd='all', flag='reg', dmnu='fin', theta_nuis=theta_nuis, planck=False)
    '''
