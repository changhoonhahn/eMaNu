'''


data compression methods for the bispectrum analysis 



'''
import os 
import h5py 
import numpy as np 
import GPy
from sklearn.decomposition import PCA
# --- eMaNu --- 
from emanu import util as UT
from emanu import obvs as Obvs
from emanu import plots as ePlots
from emanu import compressor as Comp
from emanu import forecast as Forecast
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


dir_doc = os.path.join(UT.doc_dir(), 'paper3', 'figs') # figures for paper 
dir_fig = os.path.join(UT.fig_dir(), 'compress')  # figure directory
kf = 2.*np.pi/1000. # fundmaentla mode

thetas = ['Om', 'Ob2', 'h', 'ns', 's8']#, 'Mnu']
theta_lbls = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$']#, r'$M_\nu$']
theta_fid = [0.3175, 0.049, 0.6711, 0.9624, 0.834]#, 0.] # fiducial theta 


def compressedFisher(obvs='pk', method='KL', kmax=0.5, n_components=20):
    ''' Comparison of the Fisher forecast of compressed P0 versus full P0 
    '''
    # read in data, covariance matrix, inverse covariance matrix, and derivatives of observable X. 
    X, Cov, Cinv, dXdt = load_X(obvs=obvs, kmax=kmax)
    if method == 'PCA': print('%i-dimensional data being PCA compressed to %i-dimensions' % (X.shape[1], n_components))
    # "true" Fisher 
    Fij_true = Forecast.Fij(dXdt, Cinv) 
    # invert "true" Fisher 
    Finv_true = np.linalg.inv(Fij_true) 
    print('FULL; 1-sigmas %s' % 
            (', '.join(['%s: %.2e' % (tt, sii) for tt, sii in zip(thetas, np.sqrt(np.diag(Finv_true)))])))
    
    # now lets calculate the Fisher matrices for compressions derived
    # from different number of simulations 
    Finvs = []
    if obvs == 'pk': Nmocks = [15000, 10000, 5000, 1000, 500, 100]
    elif obvs == 'bk': Nmocks = [15000, 10000, 5000, 1000]

    for Nmock in Nmocks: 
        # fit the compressor
        if method == 'KL':
            cmpsr = Comp.Compressor(method='KL')   # KL compression 
            cmpsr.fit(X[:Nmock], dXdt) # fit compression matrix
        elif method == 'PCA': # PCA compression
            cmpsr = Comp.Compressor(method='PCA')      
            cmpsr.fit(X[:Nmock], n_components=n_components, whiten=False)   
        
        # transform X and dXdt 
        cX      = cmpsr.transform(X[:Nmock])
        dcXdt   = cmpsr.transform(dXdt)

        # get covariance and precision matrices for compressed data 
        cCov    = np.cov(cX.T) 
        cCinv   = np.linalg.inv(cCov) 
        _nmock, _ndata = cX.shape
        f_hartlap   = float(_nmock - _ndata - 2)/float(_nmock - 1) 
        cCinv       *= f_hartlap
        
        # get compressed Fisher 
        cFij    = Forecast.Fij(dcXdt, cCinv) # fisher 
        cFinv   = np.linalg.inv(cFij) # invert fisher matrix 

        print('Nmock=%i; 1-sigmas %s' % 
                (Nmock, ', '.join(['%s: %.2e' % (tt, sii) for tt, sii in zip(thetas, np.sqrt(np.diag(cFinv)))])))
        Finvs.append(cFinv) 

    theta_lims = [(0.25, 0.385), (0.02, 0.08), (0.3, 1.1), (0.6, 1.3), (0.77, 0.9)]#, (-0.4, 0.4)]

    # marginalized constraints
    onesigma_true = np.sqrt(np.diag(Finv_true)) # the "true" marginalized constraints
    onesigma = []
    for _Finv in Finvs: 
        onesigma.append(np.sqrt(np.diag(_Finv)))
    onesigma = np.array(onesigma) 

    fig = plt.figure(figsize=(6,6))
    sub = fig.add_subplot(111) 
    for i in range(len(thetas)): 
        sub.plot(Nmocks, onesigma[:,i]/onesigma_true[i], label=theta_lbls[i]) 
    sub.legend(loc='upper right', ncol=2, handletextpad=0.2, fontsize=15)
    sub.set_xlabel(r'$N_{\rm mock}$', fontsize=20) 
    sub.set_xlim(np.min(Nmocks), 10000) 
    sub.plot(sub.get_xlim(), [1., 1.], c='k', ls='--') 
    sub.set_ylabel(r'$\sigma^{\rm c}_\theta(N_{\rm mock})/\sigma_\theta$', fontsize=20) 
    if method == 'PCA': 
        sub.set_ylim(0.9, 5)
    elif method == 'KL': 
        sub.set_ylim(0.8, 1.2)
    
    str_method = '%s' % method
    if method == 'PCA': str_method += '_ncomp%i' % n_components

    ffig = os.path.join(dir_fig, '%s.compress.%s.1sigma.png' % (obvs, str_method))
    fig.savefig(ffig, bbox_inches='tight') 

    fig = Forecast.plotFisher(Finvs[::-1]+[Finv_true], theta_fid, ranges=theta_lims, labels=theta_lbls, 
            colors=['C5', 'C4', 'C3', 'C2', 'C1', 'C0', 'k']) 
    ffig = os.path.join(dir_fig, '%s.compress.%s.contours.png' % (obvs, str_method)) 
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def load_X(obvs='pk', kmax=0.5):
    ''' read in data X, covariance matrix, and derivatives w.r.t thetas (which in our 
    case are ['Om', 'Ob2', 'h', 'ns', 's8', 'Mnu']. This is an attempt to make everything
    more modular and agnostic to the observables
    
    :param obvs:
        string specifying the cosmological observable. (default: 'pk') 

    :param kmax:
        kmax value specifying the k range. (default: 0.5)  

    :return X:
        (Nsim x Ndata) array of the data vectors of the simulation

    :return Cov:
        (Ndata x Ndata) covariance matrix

    :return Cinv:
        (Ndata x Ndata) inverse covariance matrix 
        **hartlap factor is included here**

    :return dXdt: 
        derivatives of X w.r.t. the thetas. 
    '''
    if obvs == 'pk': 
        # read in Quijote P0
        quij    = Obvs.quijotePk('fiducial', rsd=0, flag='reg') 
        klim    = (quij['k'] <= kmax) # k limit 
        X       = quij['p0k'][:,klim] + quij['p_sn'][:,None] # shotnoise uncorrected P(k)

        # calculate covariance
        Cov     = np.cov(X.T) 
        if np.linalg.cond(Cov) >= 1e16: print('Covariance matrix is ill-conditioned') 

        # calculate inverse covariance
        Cinv    = np.linalg.inv(Cov) 
        nmock, ndata = X.shape
        f_hartlap = float(nmock - ndata - 2)/float(nmock - 1) # hartlap factor
        Cinv    *= f_hartlap
    
        dXdt = [] 
        for par in thetas: 
            dXdt_i = dPkdtheta(par, rsd='all', flag='reg', dmnu='fin')
            dXdt.append(dXdt_i[klim])
        dXdt = np.array(dXdt) 

    elif obvs == 'bk': 
        # read in Quijote B 
        quij    = Obvs.quijoteBk('fiducial', rsd=0, flag='reg') 

        # k limit 
        i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3']
        klim    = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) # k limit
        X       = quij['b123'][:,klim] + quij['b_sn'][:,klim]

        # calculate covariance
        Cov     = np.cov(X.T) 
        if np.linalg.cond(Cov) >= 1e16: print('Covariance matrix is ill-conditioned') 

        # calculate inverse covariance
        Cinv    = np.linalg.inv(Cov) 
        nmock, ndata = X.shape 
        f_hartlap = float(nmock - ndata - 2)/float(nmock - 1) 
        Cinv    *= f_hartlap
    
        # derivative of B w.r.t theta
        dXdt = [] 
        for par in thetas: 
            dXdt_i = dBkdtheta(par, rsd='all', flag='reg', dmnu='fin')
            dXdt.append(dXdt_i[klim])
        dXdt = np.array(dXdt) 
    else: 
        raise NotImplementedError 
    return X, Cov, Cinv, dXdt 


def traceCy(method='KL', kmax=0.5):
    '''
    '''
    # read in Quijote P0
    quij    = Obvs.quijotePk('fiducial', rsd=0, flag='reg') 
    klim = (quij['k'] <= kmax) # k limit 
    # calculate P0 covariance
    pks = quij['p0k'][:,klim] + quij['p_sn'][:,None] # shotnoise uncorrected P(k)
    # derivative of P w.r.t theta
    dpkdt = [] 
    for par in ['Om', 'Ob2', 'h', 'ns', 's8', 'Mnu']: 
        dpkdt_i = dPkdtheta(par, rsd='all', flag='reg', dmnu='fin')
        dpkdt.append(dpkdt_i[klim])
    dpkdt = np.array(dpkdt) 
    
    f_hartlap = lambda nmock: float(nmock - pks.shape[1] - 2)/float(nmock - 1) 
    
    pkNmocks = range(100, 15100, 100) 
    pk_trace_ratio = [] 
    for Nmock in pkNmocks: 
        if method == 'KL':
            cmpsr = Comp.Compressor(method='KL')   # KL compression 
            cmpsr.fit(pks[:Nmock], dpkdt) # fit compression matrix
        elif method == 'PCA': # PCA compression
            cmpsr = Comp.Compressor(method='PCA')      
            cmpsr.fit(pks[:Nmock], n_components=20, whiten=False)   
        cpks = cmpsr.transform(pks[:Nmock])
        _cpks = cmpsr.transform(pks) 

        cCpk    = np.cov(cpks.T) 
        cCinv   = f_hartlap(Nmock) * np.linalg.inv(cCpk) 
        cCpk_true   = np.cov(_cpks.T) 
        cCinv_true  = f_hartlap(15000) * np.linalg.inv(cCpk_true) 
        pk_trace_ratio.append(np.trace(cCinv_true)/np.trace(cCinv)) 

    # read in Quijote B 
    quij    = Obvs.quijoteBk('fiducial', rsd=0, flag='reg') 
    i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3']
    klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) # k limit
    # calculate B covariance
    bks = quij['b123'][:,klim] + quij['b_sn'][:,klim]
    
    f_hartlap = lambda nmock: float(nmock - bks.shape[1] - 2)/float(nmock - 1) 

    # derivative of B w.r.t theta
    dbkdt = [] 
    for par in ['Om', 'Ob2', 'h', 'ns', 's8', 'Mnu']: 
        dbkdt_i = dBkdtheta(par, rsd='all', flag='reg', dmnu='fin')
        dbkdt.append(dbkdt_i[klim])
    dbkdt = np.array(dbkdt) 

    bkNmocks = range(2000, 16000, 1000) 
    bk_trace_ratio = [] 
    for Nmock in bkNmocks: 
        if method == 'KL':
            cmpsr = Comp.Compressor(method='KL')   # KL compression 
            cmpsr.fit(bks[:Nmock], dbkdt) # fit compression matrix
        elif method == 'PCA': # PCA compression
            cmpsr = Comp.Compressor(method='PCA')      
            cmpsr.fit(bks[:Nmock], n_components=100, whiten=False)   
        cbks    = cmpsr.transform(bks[:Nmock])     # compressed B 
        _cbks   = cmpsr.transform(bks)

        cCbk    = np.cov(cbks.T) 
        cCinv   = f_hartlap(Nmock) * np.linalg.inv(cCbk) 
        cCbk_true   = np.cov(_cbks.T) 
        cCinv_true  = f_hartlap(15000) * np.linalg.inv(cCbk_true) 
        bk_trace_ratio.append(np.trace(cCinv_true)/np.trace(cCinv)) 

    ############################################
    fig = plt.figure(figsize=(6,6)) 
    sub = fig.add_subplot(111)
    sub.plot(pkNmocks, np.array(pk_trace_ratio))
    sub.plot(bkNmocks, np.array(bk_trace_ratio))
    sub.plot(pkNmocks, np.ones(len(pkNmocks)), c='k', ls='--')
    sub.plot([1000., 1000.], [0., 2.], c='k', ls=':') 
    sub.set_xlabel(r'$N_{\rm mocks}$', fontsize=20)
    sub.set_xlim(0, 15000) 
    sub.set_ylabel(r'Tr($C_Y^{-1}$)/Tr($\hat{C}_Y^{-1})$', fontsize=20) 
    sub.set_ylim(0.5, 1.3) 
    ffig = os.path.join(dir_fig, 'traceCy.%s.png' % method) 
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def compare_GP_deriv(kmax=0.5): 
    ''' I suspect the unexpected compression results are caused by the noise in the 
    derivatives. Perhaps taking derivatives w.r.t. theta using GP will help(?)
    '''
    thetas = ['Om', 'Ob2', 'h', 'ns', 's8']
    theta_steps = [0.02, 0.004, 0.04, 0.04, 0.03]
    thetas_m    = [0.3075, 0.047, 0.6511, 0.9424, 0.819]
    thetas_p    = [0.3275, 0.051, 0.6911, 0.9824, 0.849]
    thetas_fid  = [0.3175, 0.049, 0.6711, 0.9624, 0.834]

    quij = Obvs.quijotePk('fiducial', z=0, rsd=0, flag='reg', silent=False) 
    k = quij['k']
    klim = (k < kmax) 
    p0k1 = quij['p0k'][:10,:]
    p0k_fid = np.average(p0k1[:,klim], axis=0) 

    m_gps, dPdts = [], [] 
    for i, theta in enumerate(thetas): 
        quij = Obvs.quijotePk('%s_m' % theta, z=0, rsd=0, flag='reg', silent=False) 
        p0k0 = quij['p0k']
        quij = Obvs.quijotePk('%s_p' % theta, z=0, rsd=0, flag='reg', silent=False) 
        p0k2 = quij['p0k']
        
        kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        X = np.concatenate([
            np.repeat(thetas_m[i], p0k0.shape[0]), 
            np.repeat(thetas_fid[i], p0k1.shape[0]), 
            np.repeat(thetas_p[i], p0k2.shape[0])])
        Y = np.concatenate([p0k0[:,klim], p0k1[:,klim], p0k2[:,klim]])

        m_gp = GPy.models.GPRegression(np.atleast_2d(X).T, np.atleast_2d(Y), kernel, normalizer=True)
        #m_gp.optimize_restarts(num_restarts=10)
        m_gp.optimize()
        m_gps.append(m_gp)

        dPdts.append((np.average(p0k2[:,klim], axis=0) - np.average(p0k0[:,klim], axis=0))/theta_steps[i]/p0k_fid)

    fig = plt.figure(figsize=(30,5))
    for i in range(len(thetas)): 
        sub = fig.add_subplot(1,5,i+1)
        sub.plot(k[klim], dPdts[i], c='k', label='Numerical Deriv.')
        
        X_pred = np.atleast_2d(np.array([thetas_fid[i]]))
        Y_pred, var_Y_pred = m_gps[i].predict(X_pred)
        dY_pred, var_dY_pred = m_gps[i].predictive_gradients(X_pred)

        sub.plot(k[klim], (m_gp.normalizer.inverse_mean(dY_pred)).flatten()/Y_pred.flatten(), label='GP Deriv.')
        sub.set_xlim(5e-3, None) 
        sub.set_xscale('log') 
        if i == len(thetas)-1: sub.legend(loc='upper right', handletextpad=0.2, fontsize=15) 
        if i == 0: sub.set_ylabel(r'$\partial \log P_0/\partial \theta$', fontsize=25) 

    ffig = os.path.join(dir_fig, 'compare_GPderiv.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    return None

# --- support functions ---
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


def _rsd_str(rsd): 
    # assign string based on rsd kwarg 
    if rsd == 'all': return ''
    elif rsd in [0, 1, 2]: return '.rsd%i' % rsd
    elif rsd == 'real': return '.real'
    else: raise NotImplementedError


def _flag_str(flag): 
    # assign string based on flag kwarg
    return ['.%s' % flag, ''][flag is None]


if __name__=='__main__': 
    #compare_GP_deriv()
    compressedFisher(obvs='pk', method='KL', kmax=0.5)
    #compressedFisher(obvs='bk', method='KL', kmax=0.5)
    #for ncomp in [20, 40, 60, 70]: 
    #    compressedFisher(obvs='pk', method='PCA', kmax=0.5, n_components=ncomp)
    #for ncomp in [50, 100, 200, 300, 500]: 
    #    compressedFisher(obvs='bk', method='PCA', kmax=0.5, n_components=ncomp)
