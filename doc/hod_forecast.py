'''

cosmological parameter forecasts including HOD parameters
using halo catalogs from the Quijote simulations. 

'''
import os 
import h5py
import numpy as np 
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

dir_hod = os.path.join(UT.dat_dir(), 'hod_forecast') 

fscale_pk = 1e5 # see fscale_Pk() for details

thetas = ['Om', 'Ob2', 'h', 'ns', 's8', 'Mnu']

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

    k, dpdt = np.loadtxt(fdpk, skiprows=1, unpack=True, usecols=[0,i_dpk]) 
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


def dP02B(theta, log=False, rsd='all', flag='reg', dmnu='fin', returnks=False, silent=True):
    ''' dP0/dtt, dP2/dtt, dB/dtt
    '''
    # dP0/dtheta, dP2/dtheta
    dp = dP02k(theta, log=log, rsd=rsd, flag=flag, dmnu=dmnu, returnks=returnks, silent=silent) 
    # dB0/dtheta
    db = dBk(theta, log=log, rsd=rsd, flag=flag, dmnu=dmnu, returnks=returnks, silent=silent) 

    if returnks: 
        _k, _dp = dp 
        _ik, _jk, _lk, _db = db
        return _k, _ik, _jk, _lk, np.concatenate([_dp, _db]) 
    else: 
        return np.concatenate([dp, db]) 


def plot_dP02B(kmax=0.5, rsd='all', flag='reg', log=True): 
    ''' compare derivatives 
    '''
    
    fig = plt.figure(figsize=(12,12))
    # continue here
    # continue here
    # continue here
    # continue here
    # continue here
    # continue here
    # continue here
    # continue here


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
    '''
        # calculate the covariance matrices 
        p02kCov(rsd=2, flag='reg', silent=False)
        bkCov(rsd=2, flag='reg', silent=False)
        fscale_Pk(rsd=2, flag='reg', silent=False)
        p02bkCov(rsd=2, flag='reg', silent=False)

        # plot the covariance matrices 
        plot_p02bkCov(kmax=0.5, rsd=2, flag='reg')
    '''
    plot_p02bkCov(kmax=0.5, rsd=2, flag='reg')

