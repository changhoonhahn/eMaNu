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
import matplotlib.cm as cm
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

thetas = ['Om', 'Ob2', 'h', 'ns', 's8', 'Mnu']
theta_lbls = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$']
theta_fid = [0.3175, 0.049, 0.6711, 0.9624, 0.834, 0.] # fiducial theta 


def compressedFisher_1sigma_PCA(kmax=0.3, correction='tdist'): 
    ''' ratio of 1 sigma parameter Fisher constraints of compressed over "true" (full)  
    for PCA compression 
    '''
    fig = plt.figure(figsize=(10, 5))
    sub0 = fig.add_subplot(121) # pk panel 
    sub1 = fig.add_subplot(122) # bk panel 

    for obvs, sub in zip(['pk', 'bk'], [sub0, sub1]): 
        # read in data, covariance matrix, inverse covariance matrix, and derivatives of observable X. 
        X, Cov, Cinv, dXdt = load_X(obvs=obvs, kmax=kmax)
        Finv_true = Finv_full(X, Cov, dXdt, correction=correction)
        sig1_true = np.sqrt(np.diag(Finv_true)) # the "true" marginalized constraints

        print('FULL; 1-sigmas %s' % 
                (', '.join(['%s: %.2e' % (tt, sii) for tt, sii in zip(thetas, sig1_true)])))

        if obvs == 'pk': 
            Ncomps = [10, 15, 20, 25, 30, 35, 40, 45, 50]
            Nmocks = [10000, 5000, 4000, 3000, 2000, 1000][::-1]
        elif obvs == 'bk': 
            Ncomps = [20, 30, 40, 50, 60, 70, 80, 90, 100]
            Nmocks = [10000, 5000, 4000, 3000, 2000, 1000][::-1]
        Nmocks = np.array(Nmocks)[np.array(Nmocks) > 2*X.shape[1]] 

        # Fisher matrices for compressions derived from different number of simulations 
        csig1s, cFinvs = [], [] 
        for Nmock in Nmocks: 
            _csig1s, _cFinvs = [], [] 
            for Ncomp in Ncomps: 
                cFinv = Finv_PCA(X, Cov, dXdt, Nmock, Ncomp, correction=correction) 
                csig1 = np.sqrt(np.diag(cFinv)) 
                print('Nmock=%i; 1-sigmas %s' % 
                        (Nmock, ', '.join(['%s: %.2e' % (tt, sii) for tt, sii in zip(thetas, np.sqrt(np.diag(cFinv)))])))
                _csig1s.append(csig1) 
                _cFinvs.append(cFinv) 
            csig1s.append(np.array(_csig1s))
            cFinvs.append(np.array(_cFinvs))
        csig1s = np.array(csig1s)
        cFinvs = np.array(cFinvs)

        X, Y = np.meshgrid(Nmocks, Ncomps) 
        #_CS = sub.contourf(X, Y, csig1s[:,:,0].T/sig1_true[0], vmin=1., vmax=5., cmap=cm.coolwarm)
        _CS = sub.pcolormesh(X, Y, csig1s[:,:,0].T/sig1_true[0], vmin=1., vmax=5., cmap=cm.coolwarm)
        sub.set_xlim(1000, 10000) 
        sub.set_ylim(Ncomps[0], Ncomps[-1]) 
    sub0.text(0.95, 0.95, r'$P_\ell$', color='w', ha='right', va='top', transform=sub0.transAxes, fontsize=25)
    sub1.text(0.95, 0.95, r'$B_0$', color='w', ha='right', va='top', transform=sub1.transAxes, fontsize=25)
    m = cm.ScalarMappable(cmap=cm.coolwarm)
    m.set_array(csig1s[:,:,0].T/sig1_true[0])
    m.set_clim(1., 5.) 

    cbar_ax = fig.add_axes([0.925, 0.15, 0.0125, 0.7])
    cbar = fig.colorbar(m, cax=cbar_ax)
    cbar.set_label(r'$\sigma^{\rm PCA}_{\Omega_m}/\sigma^{\rm full}_{\Omega_m}$', 
            fontsize=25, labelpad=20, rotation=90)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'$N_{\rm mock}$', labelpad=10, fontsize=25)
    bkgd.set_ylabel(r'$N_{\rm PCA}$', labelpad=10, fontsize=25)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    ffig = os.path.join(dir_doc, 'compressedFisher.PCA.kmax%.1f.%s_corr.1sigma.png' % (kmax, correction))
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') # latex friednly
    return None 


def compressedFisher_1sigma_KL(kmax=0.3, correction='tdist'):
    ''' ratio of 1 sigma parameter Fisher constraints of compressed over "true" (full)  
    for KL compression 
    '''
    fig = plt.figure(figsize=(10, 5))
    sub0 = fig.add_subplot(121) # pk panel 
    sub1 = fig.add_subplot(122) # bk panel 

    for obvs, sub in zip(['pk', 'bk'], [sub0, sub1]): 
        # read in data, covariance matrix, inverse covariance matrix, and derivatives of observable X. 
        X, Cov, Cinv, dXdt = load_X(obvs=obvs, kmax=kmax)
        Finv_true = Finv_full(X, Cov, dXdt, correction=correction)
        sig1_true = np.sqrt(np.diag(Finv_true)) # the "true" marginalized constraints

        print('FULL; 1-sigmas %s' % 
                (', '.join(['%s: %.2e' % (tt, sii) for tt, sii in zip(thetas, sig1_true)])))

        Nmocks = [10000, 5000, 4000, 3000, 2000, 1000][::-1]
        Nmocks = np.array(Nmocks)[np.array(Nmocks) > 2*X.shape[1]] 

        # Fisher matrices for compressions derived from different number of simulations 
        csig1s, cFinvs = [], [] 
        for Nmock in Nmocks: 
            cFinv = Finv_KL(X, Cov, dXdt, Nmock, correction=correction) 
            csig1 = np.sqrt(np.diag(cFinv)) 
            print('Nmock=%i; 1-sigmas %s' % 
                    (Nmock, ', '.join(['%s: %.2e' % (tt, sii) for tt, sii in zip(thetas, np.sqrt(np.diag(cFinv)))])))
            csig1s.append(csig1) 
            cFinvs.append(cFinv) 
        csig1s = np.array(csig1s)
        cFinvs = np.array(cFinvs)

        for i in range(len(thetas)): 
            sub.plot(Nmocks, csig1s[:,i]/sig1_true[i], label=theta_lbls[i], c='C%i' % i) 
        sub.set_xlim(1000, 10000) 
        sub.plot(sub.get_xlim(), [1., 1.], c='k', ls='--') 
        sub.set_ylim(0.95, 1.15) 
    
    sub0.set_yticks([0.95, 1., 1.05, 1.1, 1.15])
    sub1.set_yticks([0.95, 1., 1.05, 1.1, 1.15])
    sub1.set_yticklabels([])
    sub1.legend(loc='lower left', ncol=3, columnspacing=0.8, handletextpad=0.25, bbox_to_anchor=(0.0, 0.0), fontsize=15)

    sub0.text(0.95, 0.95, r'$P_\ell$', ha='right', va='top', transform=sub0.transAxes, fontsize=25)
    sub1.text(0.95, 0.95, r'$B_0$', ha='right', va='top', transform=sub1.transAxes, fontsize=25)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'$N_{\rm mock}$', labelpad=10, fontsize=25)
    bkgd.set_ylabel(r'$\sigma^{\rm KL}_{\Omega_m}/\sigma^{\rm full}_{\Omega_m}$', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    fig.subplots_adjust(wspace=0.1) 
    ffig = os.path.join(dir_doc, 'compressedFisher.KL.kmax%.1f.%s_corr.1sigma.png' % (kmax, correction))
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') # latex friednly
    return None 


def compressedFisher_contour_PCA_KL(kmax=0.3, correction='tdist'):
    ''' Comparison of the Fisher forecast contours for 
    - PCA compressed B0 w/ 4000 mocks
    - KL compressed B0 w/ 4000 mocks 
    - full B0
    '''
    # read in data, covariance matrix, inverse covariance matrix, and derivatives of observable X. 
    X, Cov, Cinv, dXdt = load_X(obvs='bk', kmax=kmax)
    Finv_true = Finv_full(X, Cov, dXdt, correction=correction)
    sig1_true = np.sqrt(np.diag(Finv_true)) # the "true" marginalized constraints
    
    Nmock = 4000 # sensible and reasonable number of mocks to expect for in DESI 
    Npca = 100
            
    cFinv_PCA = Finv_PCA(X, Cov, dXdt, Nmock, Npca, correction=correction) 
    csig1_PCA = np.sqrt(np.diag(cFinv_PCA)) # the "true" marginalized constraints
    cFinv_KL = Finv_KL(X, Cov, dXdt, Nmock, correction=correction) 
    csig1_KL = np.sqrt(np.diag(cFinv_KL)) # the "true" marginalized constraints
    
    theta_lims = [(0.25, 0.385), (0.02, 0.08), (0.3, 1.1), (0.6, 1.3), (0.77, 0.9), (-0.5, 0.5)]

    titles = ['\n'.join([r'$\sigma^{\rm KL}_\theta = %.2f ~\sigma^{\rm full}_\theta$' % (csig_kl/sig_full),
        r'$\sigma^{\rm PCA}_\theta = %.1f ~\sigma^{\rm full}_\theta$' % (csig_pca/sig_full)]) 
        for csig_kl, csig_pca, sig_full in zip(csig1_KL, csig1_PCA, sig1_true)]

    fig = Forecast.plotFisher([cFinv_PCA, cFinv_KL, Finv_true], theta_fid, ranges=theta_lims, 
            labels=theta_lbls, titles=titles, title_kwargs={'fontsize': 15},
            linestyles=['-', '-', '--'], colors=['C1', 'r', 'C0'], sigmas=[1, 2], alphas=[1., 0.7]) 
    
    # legend 
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.fill_between([],[],[], color='C0', label=r'Full $B_0$; 15,000 mocks') 
    bkgd.fill_between([],[],[], color='r', label=r'KL $B_0$; 4000 mocks') 
    bkgd.fill_between([],[],[], color='C1', label=r'PCA $B_0$; 4000 mocks') 
    bkgd.legend(loc='upper right', bbox_to_anchor=(0.95, 0.85), fontsize=25)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    
    ffig = os.path.join(dir_doc, 'compressedFisher.PCA_KL.kmax%.1f.%s_corr.contour.png' % (kmax, correction))
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') # latex friednly
    return None 


def Finv_full(X, Cov, dXdt, correction='tdist'):
    '''inverse Fisher matrix for full data. 
    No data compression 
    '''
    N, p = X.shape 
    
    Cinv = np.linalg.inv(Cov)

    Fij = Forecast.Fij(dXdt, Cinv) 
    # apply correction factor since the likelihood is not exactly a Gaussian
    if correction == 'tdist': Fij *= F_tdist(p, N)
    elif correction == 'hartlap': Fij *= F_hartlap(p, N)
    # invert Fisher matrix
    Finv = np.linalg.inv(Fij) 
    return Finv 
  

def Finv_PCA(X, Cov, dXdt, Nmock, n_components, correction='tdist'):  
    ''' inverse Fisher matrix for PCA compressed data. 
    '''
    N, p = X.shape 
    # fit compression 
    cmpsr = Comp.Compressor(method='PCA')      
    cmpsr.fit(X[:Nmock], n_components=n_components, whiten=False)   

    # transform X and dXdt 
    cX      = cmpsr.transform(X[:Nmock])
    dcXdt   = cmpsr.transform(dXdt)

    # get covariance and precision matrices for compressed data 
    cCov    = np.cov(cX.T) 
    cCinv   = np.linalg.inv(cCov) 
    
    # get compressed Fisher 
    cFij    = Forecast.Fij(dcXdt, cCinv) # fisher 
    # correct cFij for the t-distribution 
    if correction == 'tdist': cFij *= F_tdist(p, Nmock)
    elif correction == 'hartlap': cFij *= F_hartlap(p, Nmock)
    cFinv   = np.linalg.inv(cFij) # invert fisher matrix 
    return cFinv 


def Finv_KL(X, Cov, dXdt, Nmock, correction='tdist'):  
    ''' inverse Fisher matrix for PCA compressed data. 
    '''
    N, p = X.shape 
    cmpsr = Comp.Compressor(method='KL')   # KL compression 
    cmpsr.fit(X[:Nmock], dXdt) # fit compression matrix

    # transform X and dXdt 
    cX      = cmpsr.transform(X[:Nmock])
    dcXdt   = cmpsr.transform(dXdt)

    # get covariance and precision matrices for compressed data 
    cCov    = np.cov(cX.T) 
    cCinv   = np.linalg.inv(cCov) 
    
    # get compressed Fisher 
    cFij    = Forecast.Fij(dcXdt, cCinv) # fisher 
    # correct cFij for the t-distribution 
    if correction == 'tdist': cFij *= F_tdist(p, Nmock)
    elif correction == 'hartlap': cFij *= F_hartlap(p, Nmock)
    cFinv   = np.linalg.inv(cFij) # invert fisher matrix 

    return cFinv 


def compressedFisher_1sigma(obvs='pk', method='KL', kmax=0.5, n_components=20, correction='tdist'):
    ''' Comparison of the Fisher forecast of compressed P0 versus full P0 
    '''
    # read in data, covariance matrix, inverse covariance matrix, and derivatives of observable X. 
    X, Cov, Cinv, dXdt = load_X(obvs=obvs, kmax=kmax)
    if method == 'KL':  print('%i-dimensional data being KL compressed to %i-dimensions' % (X.shape[1], len(dXdt)))
    elif method == 'PCA': print('%i-dimensional data being PCA compressed to %i-dimensions' % (X.shape[1], n_components))
    # "true" Fisher 
    Fij_true = Forecast.Fij(dXdt, Cinv) 
    # apply t-distribution correction factor since the likelihood is not exactly a Gaussian
    n_data = X.shape[1] 
    if correction == 'tdist': 
        Fij_true *= F_tdist(n_data, X.shape[0])
    elif correction == 'hartlap': 
        Fij_true *= F_hartlap(n_data, X.shape[0])
    # invert "true" Fisher 
    Finv_true = np.linalg.inv(Fij_true) 
    print('FULL; 1-sigmas %s' % 
            (', '.join(['%s: %.2e' % (tt, sii) for tt, sii in zip(thetas, np.sqrt(np.diag(Finv_true)))])))
    
    # now lets calculate the Fisher matrices for compressions derived
    # from different number of simulations 
    Finvs = []
    if obvs == 'pk': 
        Nmocks = [15000, 10000, 5000, 4000, 2000, 1000, 500]
    elif obvs == 'bk': 
        Nmocks = [15000, 10000, 5000, 4000, 3000, 2000, 1000]
    Nmocks = np.array(Nmocks)[np.array(Nmocks) > 2*n_data] 
    clrs = ['C%i' % i for i in range(len(Nmocks))[::-1]] + ['k']

    for Nmock in Nmocks: 
        # fit the compressor
        if method == 'KL':
            cmpsr = Comp.Compressor(method='KL')   # KL compression 
            cmpsr.fit(X[:Nmock], dXdt) # fit compression matrix
        elif method == 'PCA': # PCA compression
            cmpsr = Comp.Compressor(method='PCA')      
            cmpsr.fit(X[:Nmock], n_components=n_components, whiten=False)   
        #print('condition # of covariance = %.2e' % np.linalg.cond(np.cov(X[:Nmock].T)))
        # transform X and dXdt 
        cX      = cmpsr.transform(X[:Nmock])
        dcXdt   = cmpsr.transform(dXdt)

        # get covariance and precision matrices for compressed data 
        cCov    = np.cov(cX.T) 
        cCinv   = np.linalg.inv(cCov) 
        
        # get compressed Fisher 
        cFij    = Forecast.Fij(dcXdt, cCinv) # fisher 
        # correct cFij for the t-distribution 
        if correction == 'tdist': 
            cFij    *= F_tdist(n_data, Nmock)
        elif correction == 'hartlap': 
            cFij    *= F_hartlap(n_data, Nmock)
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
        sub.plot(Nmocks, onesigma[:,i]/onesigma_true[i], label=theta_lbls[i], c='C%i' % i) 
    sub.legend(loc='upper right', handletextpad=0.25, fontsize=20)
    sub.set_xlabel(r'$N_{\rm mock}$', fontsize=20) 
    sub.set_xlim(500, 15000) 
    sub.plot(sub.get_xlim(), [1., 1.], c='k', ls='--') 
    sub.set_ylabel(r'$\sigma^{\rm c}_\theta(N_{\rm mock})/\sigma^{\rm ``true"}_\theta$', fontsize=20) 
    if method == 'PCA': 
        sub.set_ylim(0.9, 5)
    elif method == 'KL': 
        sub.set_ylim(0.95, 1.15)
        sub.set_yticks([0.95, 1., 1.05, 1.1, 1.15]) 
    
    str_method = '%s' % method
    if method == 'PCA': str_method += '_ncomp%i' % n_components

    ffig = os.path.join(dir_fig, '%s.compress.%s.kmax%.1f.%s_corr.1sigma.png' % (obvs, str_method, kmax, correction))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def compressedFisher_contour(obvs='pk', method='KL', kmax=0.5, n_components=20, correction='tdist'):
    ''' Comparison of the Fisher forecast of compressed P0 versus full P0 
    '''
    # read in data, covariance matrix, inverse covariance matrix, and derivatives of observable X. 
    X, Cov, Cinv, dXdt = load_X(obvs=obvs, kmax=kmax)
    if method == 'KL':  print('%i-dimensional data being KL compressed to %i-dimensions' % (X.shape[1], len(dXdt)))
    elif method == 'PCA': print('%i-dimensional data being PCA compressed to %i-dimensions' % (X.shape[1], n_components))
    # "true" Fisher 
    Fij_true = Forecast.Fij(dXdt, Cinv) 
    # apply t-distribution correction factor since the likelihood is not exactly a Gaussian
    n_data = X.shape[1] 
    if correction == 'tdist': 
        Fij_true *= F_tdist(n_data, X.shape[0])
    elif correction == 'hartlap': 
        Fij_true *= F_hartlap(n_data, X.shape[0])
    # invert "true" Fisher 
    Finv_true = np.linalg.inv(Fij_true) 
    print('FULL; 1-sigmas %s' % 
            (', '.join(['%s: %.2e' % (tt, sii) for tt, sii in zip(thetas, np.sqrt(np.diag(Finv_true)))])))
    
    # now lets calculate the Fisher matrices for compressions derived
    # from different number of simulations 
    Finvs = []
    if obvs == 'pk': 
        Nmocks = [10000, 5000, 2000, 500]
    elif obvs == 'bk': 
        Nmocks = [10000, 5000, 2500, 1000]
    Nmocks = np.array(Nmocks)[np.array(Nmocks) > 2*n_data] 
    clrs = ['C%i' % i for i in range(len(Nmocks))[::-1]] + ['k']

    for Nmock in Nmocks: 
        # fit the compressor
        if method == 'KL':
            cmpsr = Comp.Compressor(method='KL')   # KL compression 
            cmpsr.fit(X[:Nmock], dXdt) # fit compression matrix
        elif method == 'PCA': # PCA compression
            cmpsr = Comp.Compressor(method='PCA')      
            cmpsr.fit(X[:Nmock], n_components=n_components, whiten=False)   
        #print('condition # of covariance = %.2e' % np.linalg.cond(np.cov(X[:Nmock].T)))
        # transform X and dXdt 
        cX      = cmpsr.transform(X[:Nmock])
        dcXdt   = cmpsr.transform(dXdt)

        # get covariance and precision matrices for compressed data 
        cCov    = np.cov(cX.T) 
        cCinv   = np.linalg.inv(cCov) 
        
        # get compressed Fisher 
        cFij    = Forecast.Fij(dcXdt, cCinv) # fisher 
        # correct cFij for the t-distribution 
        if correction == 'tdist': 
            cFij    *= F_tdist(n_data, Nmock)
        elif correction == 'hartlap': 
            cFij    *= F_hartlap(n_data, Nmock)
        cFinv   = np.linalg.inv(cFij) # invert fisher matrix 

        print('Nmock=%i; 1-sigmas %s' % 
                (Nmock, ', '.join(['%s: %.2e' % (tt, sii) for tt, sii in zip(thetas, np.sqrt(np.diag(cFinv)))])))
        Finvs.append(cFinv) 

    theta_lims = [(0.25, 0.385), (0.02, 0.08), (0.3, 1.1), (0.6, 1.3), (0.77, 0.9)]#, (-0.4, 0.4)]

    fig = Forecast.plotFisher(Finvs[::-1]+[Finv_true], theta_fid, ranges=theta_lims, labels=theta_lbls, 
            colors=clrs, sigmas=[2], alphas=[0.75]) 
    
    # legend 
    bkgd = fig.add_subplot(111, frameon=False)
    for clr, Nmock in zip(clrs, Nmocks[::-1]): 
        bkgd.fill_between([],[],[], color=clr, label=r'%i mocks' % Nmock) 
    bkgd.legend(loc='upper right', bbox_to_anchor=(0.95, 0.925), fontsize=25)
    if obvs == 'bk': 
        bkgd.text(0.95, 0.95, r'$B_0(k_1, k_2, k_3)$; $k_{\rm max} = %.1f$' % kmax, ha='right', va='top', 
                transform=bkgd.transAxes, fontsize=25)
    elif obvs == 'pk': 
        bkgd.text(0.95, 0.95, r'$P_\ell(k)$; $k_{\rm max} = %.1f$' % kmax, ha='right', va='top', 
                transform=bkgd.transAxes, fontsize=25)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    
    str_method = '%s' % method
    if method == 'PCA': str_method += '_ncomp%i' % n_components

    ffig = os.path.join(dir_fig, '%s.compress.%s.kmax%.1f.%s_corr.contours.png' % (obvs, str_method, kmax, correction))
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
        #X       = quij['p0k'][:,klim]
        X       = quij['p0k'][:,klim] + quij['p_sn'][:,None] # shotnoise uncorrected P(k)

        # calculate covariance
        Cov     = np.cov(X.T) 
        if np.linalg.cond(Cov) >= 1e16: print('Covariance matrix is ill-conditioned') 

        # calculate inverse covariance
        Cinv    = np.linalg.inv(Cov) 

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
        #X       = quij['b123'][:,klim]
        X       = quij['b123'][:,klim] + quij['b_sn'][:,klim]

        # calculate covariance
        Cov     = np.cov(X.T) 
        if np.linalg.cond(Cov) >= 1e16: print('Covariance matrix is ill-conditioned') 

        # calculate inverse covariance
        Cinv    = np.linalg.inv(Cov) 
    
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


def F_tdist(p, N): 
    ''' correction factor for the Fisher information matrix derived from a
    modified t-distribution likelihood
    '''
    #return float(N**2)/float((N + p + 1) * (N + p - 1))
    return float((N-p)*N)/float((N-1) * (N-2))


def F_hartlap(p, N): 
    ''' Hartlap correction factor for the fact that the mean precision matrix
    estimated with N mocks is biased
    '''
    return  float(N - p - 2)/float(N - 1) 


def tdist_factor(obvs='pk', kmax=0.5): 
    ''' compare the t-distribution factor 
    '''
    # read in data, covariance matrix, inverse covariance matrix, and derivatives of observable X. 
    X, Cov, Cinv, dXdt = load_X(obvs=obvs, kmax=kmax)
    print('%i dimensional likelihood' % X.shape[1])

    nmocks = np.arange(150)*100
    
    fig = plt.figure(figsize=(6,6)) 
    sub = fig.add_subplot(111)
    sub.plot(nmocks, [F_hartlap(X.shape[1], _nmock) for _nmock in nmocks], label='Hartlap')
    sub.plot(nmocks, [F_tdist(X.shape[1], _nmock) for _nmock in nmocks], label='$t$-dist.')
    sub.plot(nmocks, np.ones(len(nmocks)), c='k', ls=':') 
    sub.legend(loc='lower right', fontsize=15)
    sub.set_xlabel(r'$N_{\rm mocks}$', fontsize=20)
    sub.set_xlim(100, 15000) 
    sub.set_ylabel(r'Corr. factor', fontsize=20) 
    sub.set_ylim(0.1, 1.1) 
    ffig = os.path.join(dir_fig, 'tdist_factor.%s.kmax%.1f.png' % (obvs, kmax))
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
    #compressedFisher_1sigma_PCA(kmax=0.3, correction='tdist')
    #compressedFisher_1sigma_KL(kmax=0.3, correction='tdist')
    #compressedFisher_contour_PCA_KL(kmax=0.3, correction='tdist')

    #compressedFisher_1sigma(obvs='pk', method='KL', kmax=0.3, correction='tdist')
    #compressedFisher_1sigma(obvs='bk', method='KL', kmax=0.3, correction='tdist')
    #compressedFisher_1sigma(obvs='pk', method='KL', kmax=0.3, correction='hartlap')
    #compressedFisher_1sigma(obvs='bk', method='KL', kmax=0.3, correction='hartlap')

    #compressedFisher_contour(obvs='pk', method='KL', kmax=0.3, correction='tdist')
    #compressedFisher_contour(obvs='bk', method='KL', kmax=0.3, correction='tdist')
    #compressedFisher_contour(obvs='pk', method='KL', kmax=0.3, correction='hartlap')
    #compressedFisher_contour(obvs='bk', method='KL', kmax=0.3, correction='hartlap')
    #for ncomp in [20, 40, 60, 70]: 
    #    compressedFisher(obvs='pk', method='PCA', kmax=0.5, n_components=ncomp)
    #for ncomp in [50, 100, 200, 300, 500]: 
    #    compressedFisher_contour(obvs='bk', method='PCA', kmax=0.3, n_components=ncomp)
    tdist_factor(obvs='pk', kmax=0.5)
    tdist_factor(obvs='bk', kmax=0.5)
