'''


data compression methods for the bispectrum analysis 



'''
import os 
import h5py 
import numpy as np 
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


def Bk_compression(method='KL', kmax=0.5):
    ''' Comparison of compressed B0 versus full B0
    '''
    # read in Quijote B 
    quij    = Obvs.quijoteBk('fiducial', rsd=0, flag='reg') 
    # k limit 
    i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3']
    klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) # k limit
    # calculate B covariance
    bks     = quij['b123'][:,klim] + quij['b_sn'][:,klim]
    Cbk     = np.cov(bks.T) 
    if np.linalg.cond(Cbk) >= 1e16: print('Covariance matrix is ill-conditioned') 
    Cinv    = np.linalg.inv(Cbk) 
    ndata       = Cbk.shape[0]
    nmock       = bks.shape[0]
    f_hartlap   = float(nmock - ndata - 2)/float(nmock - 1) 
    Cinv        *= f_hartlap
    
    # derivative of B w.r.t theta
    dbkdt = [] 
    for par in ['Om', 'Ob2', 'h', 'ns', 's8', 'Mnu']: 
        dbkdt_i = dBkdtheta(par, rsd='all', flag='reg', dmnu='fin')
        dbkdt.append(dbkdt_i[klim])
    dbkdt = np.array(dbkdt) 
    
    # true fisher 
    Fij     = Forecast.Fij(dbkdt, Cinv) 
    # invert fisher matrix 
    Finv    = np.linalg.inv(Fij) 
    print('B0 sigmas %s' % ', '.join(['%.2e' % sii for sii in np.sqrt(np.diag(Finv))]))

    Finvs = []
    Nmocks = [2000, 3000, 5000, 10000, 15000][::-1]
    for Nmock in Nmocks: 
        if method == 'KL':
            cmpsr = Comp.Compressor(method='KL')   # KL compression 
            cmpsr.fit(bks[:Nmock], dbkdt) # fit compression matrix
        elif method == 'PCA': # PCA compression
            cmpsr = Comp.Compressor(method='PCA')      
            cmpsr.fit(bks[:Nmock], n_components=100, whiten=False)   
        cbks    = cmpsr.transform(bks[:Nmock])     # compressed P 
        dcbkdt  = cmpsr.transform(dbkdt)         # compressed dP/dtheta

        cCbk    = np.cov(cbks.T) 
        cCinv   = np.linalg.inv(cCbk) 
        
        _cbks   = cmpsr.transform(bks)
        cCbk_true   = np.cov(_cbks.T) 
        cCinv_true  = np.linalg.inv(cCbk_true) 

        ndata       = bks.shape[1]
        f_hartlap   = float(Nmock - ndata - 2)/float(Nmock - 1) 
        cCinv       *= f_hartlap
        print('trace ratio %.4f; hartlap factor %.4f' % (np.trace(cCinv_true)/np.trace(cCinv), f_hartlap))
        
        cFij    = Forecast.Fij(dcbkdt, cCinv) # fisher 
        cFinv   = np.linalg.inv(cFij) # invert fisher matrix 

        print('Nmock=%i; cB0 sigmas %s' % (Nmock, ', '.join(['%.2e' % sii for sii in np.sqrt(np.diag(cFinv))]))) 
        Finvs.append(cFinv) 

    theta_lbls = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$']
    theta_fid = [0.3175, 0.049, 0.6711, 0.9624, 0.834, 0.] # fiducial theta 
    theta_lims = [(0.25, 0.385), (0.02, 0.08), (0.3, 1.1), (0.6, 1.3), (0.77, 0.9), (-0.4, 0.4)]
    ntheta = len(theta_fid)

    fig = plt.figure(figsize=(10, 10))
    for i in range(ntheta): 
        for j in range(i+1, ntheta): 
            sub = fig.add_subplot(ntheta-1,ntheta-1,(ntheta-1)*(j-1)+i+1)
            
            theta_fid_i, theta_fid_j = theta_fid[i], theta_fid[j] # fiducial parameter 
            for _i, _Finv in enumerate(Finvs[::-1]):
                Finv_sub = np.array([[_Finv[i,i], _Finv[i,j]], [_Finv[j,i], _Finv[j,j]]]) # sub inverse fisher matrix 
                Forecast.plotEllipse(Finv_sub, sub, theta_fid_ij=[theta_fid_i, theta_fid_j], color='C%i' % (_i+1))
            Finv_sub = np.array([[Finv[i,i], Finv[i,j]], [Finv[j,i], Finv[j,j]]]) # sub inverse fisher matrix 
            Forecast.plotEllipse(Finv_sub, sub, theta_fid_ij=[theta_fid_i, theta_fid_j], color='C0')

            sub.set_xlim(theta_lims[i])
            sub.set_ylim(theta_lims[j])
            if i == 0:   
                sub.set_ylabel(theta_lbls[j], labelpad=5, fontsize=28) 
                sub.get_yaxis().set_label_coords(-0.35,0.5)
            else: 
                sub.set_yticks([])
                sub.set_yticklabels([])
            
            if j == ntheta-1: 
                sub.set_xlabel(theta_lbls[i], fontsize=26) 
            else: 
                sub.set_xticks([])
                sub.set_xticklabels([]) 

    bkgd = fig.add_subplot(111, frameon=False)

    for i, Nmock in enumerate(Nmocks[::-1]):
        bkgd.fill_between([],[],[], color='C%i' % (i+1), 
                label=r'%s comp. w/ $N_{\rm mock} = %i$' % (method.replace('_', ' '), Nmock))
    bkgd.fill_between([],[],[], color='C0', label=r'$B_0(k_1, k_2, k_3)$') 
    bkgd.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95), fontsize=15)
    bkgd.text(0.8, 0.61, r'$k_{\rm max} = %.1f$; $z=0.$' % kmax, ha='right', va='bottom', transform=bkgd.transAxes, fontsize=15)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.05, hspace=0.05) 
    ffig = os.path.join(dir_fig, 'bkcompress.%s.png' % method) 
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def Pk_compression(method='KL', kmax=0.5):
    ''' Comparison of compressed P0 versus full P0
    '''
    # read in Quijote P0
    quij    = Obvs.quijotePk('fiducial', rsd=0, flag='reg') 
    # k limit 
    klim = (quij['k'] <= kmax) 
    # calculate P0 covariance
    pks     = quij['p0k'][:,klim] + quij['p_sn'][:,None] # shotnoise uncorrected P(k)
    Cpk     = np.cov(pks.T) 
    Cinv    = np.linalg.inv(Cpk) 
    if np.linalg.cond(Cpk) >= 1e16: print('Covariance matrix is ill-conditioned') 
    ndata       = Cpk.shape[0]
    nmock       = pks.shape[0]
    f_hartlap   = float(nmock - ndata - 2)/float(nmock - 1) 
    Cinv        *= f_hartlap
    
    # derivative of P w.r.t theta
    dpkdt = [] 
    for par in ['Om', 'Ob2', 'h', 'ns', 's8', 'Mnu']: 
        dpkdt_i = dPkdtheta(par, rsd='all', flag='reg', dmnu='fin')
        dpkdt.append(dpkdt_i[klim])
    dpkdt = np.array(dpkdt) 
    
    # true fisher 
    Fij     = Forecast.Fij(dpkdt, Cinv) 
    # invert fisher matrix 
    Finv    = np.linalg.inv(Fij) 
    print('P0 sigmas %s' % ', '.join(['%.2e' % sii for sii in np.sqrt(np.diag(Finv))]))

    Finvs = []
    Nmocks = [1000, 3000, 5000][::-1]
    for Nmock in Nmocks: 
        if method == 'KL':
            cmpsr = Comp.Compressor(method='KL')   # KL compression 
            cmpsr.fit(pks[:Nmock], dpkdt) # fit compression matrix
        elif method == 'PCA': # PCA compression
            cmpsr = Comp.Compressor(method='PCA')      
            cmpsr.fit(pks[:Nmock], n_components=20, whiten=False)   
        cpks = cmpsr.transform(pks[:Nmock])
        _cpks = cmpsr.transform(pks)               # compressed P 
        dcpkdt = cmpsr.transform(dpkdt)           # compressed dP/dtheta

        cCpk    = np.cov(cpks.T) 
        cCinv   = np.linalg.inv(cCpk) 
        _cCpk   = np.cov(_cpks.T) 
        _cCinv  = np.linalg.inv(_cCpk) 
        #print('trace ratio %f' % (np.trace(_cCinv)/np.trace(cCinv)))

        nmock, ndata = cpks.shape
        f_hartlap   = float(nmock - ndata - 2)/float(nmock - 1) 
        cCinv       *= f_hartlap
        cCinv       = _cCinv

        cFij    = Forecast.Fij(dcpkdt, cCinv) # fisher 
        cFinv   = np.linalg.inv(cFij) # invert fisher matrix 

        print('Nmock=%i; cP0 1/sqrt(Fii) %s' % (Nmock, ', '.join(['%.2e' % sii for sii in 1./np.sqrt(np.diag(cFij))]))) 
        print('Nmock=%i; cP0 sigmas %s' % (Nmock, ', '.join(['%.2e' % sii for sii in np.sqrt(np.diag(cFinv))]))) 
        Finvs.append(cFinv) 

    theta_lbls = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$']
    theta_fid = [0.3175, 0.049, 0.6711, 0.9624, 0.834, 0.] # fiducial theta 
    theta_lims = [(0.25, 0.385), (0.02, 0.08), (0.3, 1.1), (0.6, 1.3), (0.77, 0.9), (-0.4, 0.4)]
    ntheta = len(theta_fid)

    fig = plt.figure(figsize=(10, 10))
    for i in range(ntheta): 
        for j in range(i+1, ntheta): 
            sub = fig.add_subplot(ntheta-1,ntheta-1,(ntheta-1)*(j-1)+i+1)
            
            theta_fid_i, theta_fid_j = theta_fid[i], theta_fid[j] # fiducial parameter 
            for _i, _Finv in enumerate(Finvs[::-1]):
                Finv_sub = np.array([[_Finv[i,i], _Finv[i,j]], [_Finv[j,i], _Finv[j,j]]]) # sub inverse fisher matrix 
                Forecast.plotEllipse(Finv_sub, sub, theta_fid_ij=[theta_fid_i, theta_fid_j], color='C%i' % (_i+1))
            Finv_sub = np.array([[Finv[i,i], Finv[i,j]], [Finv[j,i], Finv[j,j]]]) # sub inverse fisher matrix 
            Forecast.plotEllipse(Finv_sub, sub, theta_fid_ij=[theta_fid_i, theta_fid_j], color='C0')

            sub.set_xlim(theta_lims[i])
            sub.set_ylim(theta_lims[j])
            if i == 0:   
                sub.set_ylabel(theta_lbls[j], labelpad=5, fontsize=28) 
                sub.get_yaxis().set_label_coords(-0.35,0.5)
            else: 
                sub.set_yticks([])
                sub.set_yticklabels([])
            
            if j == ntheta-1: 
                sub.set_xlabel(theta_lbls[i], fontsize=26) 
            else: 
                sub.set_xticks([])
                sub.set_xticklabels([]) 

    bkgd = fig.add_subplot(111, frameon=False)

    for i, Nmock in enumerate(Nmocks[::-1]):
        bkgd.fill_between([],[],[], color='C%i' % (i+1), 
                label=r'%s comp. w/ $N_{\rm mock} = %i$' % (method.replace('_', ' '), Nmock))
    bkgd.fill_between([],[],[], color='C0', label=r'$P_0(k)$') 
    bkgd.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95), fontsize=15)
    bkgd.text(0.8, 0.61, r'$k_{\rm max} = %.1f$; $z=0.$' % kmax, ha='right', va='bottom', transform=bkgd.transAxes, fontsize=15)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.05, hspace=0.05) 
    ffig = os.path.join(dir_fig, 'pkcompress.%s.png' % method) 
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


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
    #Pk_compression(method='KL')
    Pk_compression(method='PCA')
    #Bk_compression(method='KL')
    #Bk_compression(method='PCA')
    #traceCy(method='KL')
