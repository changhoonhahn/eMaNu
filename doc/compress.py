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
    Nmocks = [100, 250, 500, 1000, 5000][::-1]
    for Nmock in Nmocks: 
        if method == 'KL':
            cmpsr = Comp.Compressor(method='KL')   # KL compression 
            cmpsr.fit(pks[:Nmock], dpkdt) # fit compression matrix
        elif method == 'PCA': # PCA compression
            cmpsr = Comp.Compressor(method='PCA')      
            cmpsr.fit(pks[:Nmock], n_components=20, whiten=False)   
        _cpks = cmpsr.transform(pks[:Nmock])
        cpks = cmpsr.transform(pks)               # compressed P 
        dcpkdt = cmpsr.transform(dpkdt)           # compressed dP/dtheta

        cCpk    = np.cov(cpks.T) 
        cCinv   = np.linalg.inv(cCpk) 

        ndata       = cCpk.shape[0]
        nmock       = cpks.shape[0]
        f_hartlap   = float(nmock - ndata - 2)/float(nmock - 1) 
        cCinv       *= f_hartlap
        
        cFij    = Forecast.Fij(dcpkdt, cCinv) # fisher 
        cFinv   = np.linalg.inv(cFij) # invert fisher matrix 

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
    Nmocks = [100, 250, 500, 1000, 5000][::-1]
    for Nmock in Nmocks: 
        if method == 'KL':
            cmpsr = Comp.Compressor(method='KL')   # KL compression 
            cmpsr.fit(pks[:Nmock], dpkdt) # fit compression matrix
        elif method == 'PCA': # PCA compression
            cmpsr = Comp.Compressor(method='PCA')      
            cmpsr.fit(pks[:Nmock], n_components=20, whiten=False)   
        _cpks = cmpsr.transform(pks[:Nmock])
        cpks = cmpsr.transform(pks)               # compressed P 
        dcpkdt = cmpsr.transform(dpkdt)           # compressed dP/dtheta

        cCpk    = np.cov(cpks.T) 
        cCinv   = np.linalg.inv(cCpk) 

        ndata       = cCpk.shape[0]
        nmock       = cpks.shape[0]
        f_hartlap   = float(nmock - ndata - 2)/float(nmock - 1) 
        cCinv       *= f_hartlap
        
        cFij    = Forecast.Fij(dcpkdt, cCinv) # fisher 
        cFinv   = np.linalg.inv(cFij) # invert fisher matrix 

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
