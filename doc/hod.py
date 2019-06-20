'''
'''
import os 
import scipy
import numpy as np 
# --- eMaNu --- 
from emanu import util as UT
from emanu import forwardmodel as FM
from emanu.hades import data as hadesData
# --- corrfunc -- 
from Corrfunc.theory import wp as wpCF
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

dir_hod = os.path.join(UT.dat_dir(), 'hod') 
dir_fig = os.path.join(UT.fig_dir(), 'hod')  # figure directory


def hod_mcmc(Mr=-21.5, nwalkers=100, burn_in_chain=200, main_chain=1000): 
    '''
    '''
    import emcee
    import corner as DFM
    # read in data 
    _, _wp_sdss, cov = wp_sdss(Mr=Mr) 
    f_hart  = (400. - len(_wp_sdss) -2.)/(400. - 1.) # hartlap factor 400 is the number of jackknife fields 
    f_vol   = fvolume(Mr=Mr)
    print('hartlap factor = %f' % f_hart) 
    print('volume factor = %f' % f_vol) 
    C_inv   = f_hart * f_vol * np.linalg.inv(cov)  # apply volume and hartlap factors here.  
    # read rbin
    rbins = np.loadtxt(os.path.join(dir_hod, 'rbins.dat'), unpack=True, usecols=[0]) 
    # read in halo catalog 
    ihalo = 1
    halos = hadesData.hadesMnuHalos(0., ihalo, 4, 
            mh_min=3200., dir='/Users/ChangHoon/data/emanu/halos/hades/0.0eV/%i' % ihalo)

    # prior range
    prior_lim = np.array([[13.5, 14.5], [0.5, 1.5], [13.5, 14.5], [0.85, 1.45], [13.5, 15.]]) 
    def lnprior(tt): 
        logMmin, sigma_logM, logM0, alpha, logM1 = tt # unpack array 
        if ((prior_lim[0][0] < logMmin < prior_lim[0][1]) and 
                (prior_lim[1][0] < sigma_logM < prior_lim[1][1]) and 
                (prior_lim[2][0] < logM0 < prior_lim[2][1]) and 
                (prior_lim[3][0] < alpha < prior_lim[3][1]) and 
                (prior_lim[4][0] < logM1 < prior_lim[4][1])): 
            return 0.0
        return -np.inf 

    def lnlike(tt): 
        ''' calculate the likelihood for wp 
        '''
        # population halos 
        hod = FM.hodGalaxies(halos, {'logMmin': tt[0], 'sigma_logM': tt[1], 'logM0': tt[2], 'alpha': tt[3], 'logM1': tt[4]}) 
        # apply RSD 
        xyz = FM.RSD(hod) 
        # calculate wp 
        _wp = wpCF(1000., 40., 1, rbins, xyz[:,0], xyz[:,1], xyz[:,2], verbose=False, output_rpavg=False) # calculate wp 
        # calculate chi-squared 
        dwp = _wp_sdss - _wp['wp'] 
        chisq = np.sum(np.dot(dwp.T, np.dot(C_inv, dwp))) 
        print(chisq)
        return -0.5 * chisq

    def lnpost(tt):
        lp = lnprior(tt)
        if not np.isfinite(lp): 
            return -np.inf
        return lp + lnlike(tt) 

    ndim = 5   
    pos = [np.random.uniform(prior_lim[:,0], prior_lim[:,1]) for i in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost)
    print('running burn-in chain')
    pos, prob, state = sampler.run_mcmc(pos, burn_in_chain)
    sampler.reset()
    print('running main chain')
    sampler.run_mcmc(pos, main_chain)

    # save chain
    post = sampler.flatchain.copy()
    fig = DFM.corner(post, labels=[r'$\log M_{\rm min}$', r'$\sigma_{\log M}$', r'$\log M_0$', r'$\alpha$', r'$\log M_1$'],
            quantiles=[0.16, 0.5, 0.84], bins=20,
            range=theta_lims, truths=theta_fid, truth_color='C1',
            smooth=True, show_titles=True, label_kwargs={'fontsize': 20})
    ffig = os.path.join(dir_fig, 'hod_mcmc.png')
    fig.savefig(ffig, bbox_inch='tight')
    return None 


def wp_sdss(Mr=-21.5): 
    ''' read in projected wp(r_p) and covariance matrix 
    '''
    # read in wp(r_p) 
    if Mr == -21.5: 
        fwp = os.path.join(dir_hod, 'wpxi_dr72_bright0_mr21.5_z0.198_nj400') 
        rbin, wp = np.loadtxt(fwp, unpack=True, usecols=[0, 1]) 
    else: 
        raise NotImplementedError 

    # read in covariance 
    if Mr == -21.5: 
        fcov = os.path.join(dir_hod, 'wpxicov_dr72_bright0_mr21.5_z0.198_nj400') 
        Cwp = np.loadtxt(fcov)[:12,:12]
    else: 
        raise NotImplementedError 
    return rbin, wp,  Cwp


def _plot_wp_sdss(Mr=-21.5): 
    ''' quick plot to check SDSS w_p
    '''
    rp, wp, cov = wp_sdss(Mr=Mr) 

    fig = plt.figure(figsize=(5,5))
    sub = fig.add_subplot(111)
    sub.errorbar(rp, wp, yerr=np.sqrt(np.diag(cov)), fmt='.k') 
    sub.set_xlabel(r'$r_p$ [$h^{-1}{\rm Mpc}$]', fontsize=25) 
    sub.set_xscale('log') 
    sub.set_xlim(1e-1, 25) 
    sub.set_ylabel(r'$w_p(r_p)$', fontsize=25) 
    sub.set_yscale('log') 
    sub.set_title('$w_p$ of SDSS $M_r < -21.5$', fontsize=25) 
    ffig = os.path.join(dir_fig, 'wp_sdss.png')
    fig.savefig(ffig, bbox_inches='tight') 
    plt.close() 
    return None 


def fvolume(Mr=-21.5):
    ''' volume correction factor of the covariance matrix
    '''
    Vsim = 1000.**3 # simulation volume 
    
    if Mr == -21.5:
        Vd = 134.65 * 10**6.
    else: 
        raise NotImplementedError 
    return (1. + Vd / Vsim)


if __name__=='__main__': 
    #_plot_wp_sdss(Mr=-21.5)
    hod_mcmc(Mr=-21.5)
