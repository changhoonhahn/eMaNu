'''
'''
import os 
import scipy as sp 
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


def hod_fit(Mr=-21.5, nwalkers=100, burn_in_chain=200, main_chain=1000): 
    '''
    '''
    import emcee
    import corner as DFM
    # read in data 
    rp, _wp_sdss, cov = wp_sdss(Mr=Mr) 
    f_hart  = (400. - len(_wp_sdss) -2.)/(400. - 1.) # hartlap factor 400 is the number of jackknife fields 
    f_vol   = fvolume(Mr=Mr)
    print('hartlap factor = %f' % f_hart) 
    print('volume factor = %f' % f_vol) 
    C_inv   = f_hart * f_vol * np.linalg.inv(cov)  # apply volume and hartlap factors here.  
    # read rbin
    rbins = np.loadtxt(os.path.join(dir_hod, 'rbins.dat'), unpack=True, usecols=[0]) 
    # read in halo catalog 
    ihalo = 1
    halos = hadesData.hadesMnuHalos(0., ihalo, 4, mh_min=None, dir='/Users/ChangHoon/data/emanu/halos/hades/0.0eV/%i' % ihalo)
    print np.log10(np.array(halos['Mass']).min()), np.log10(np.array(halos['Mass']).max()) 
    #halos = hadesData.hadesMnuHalos(0., ihalo, 4, 
    #halos = hadesData.hadesMnuHalos(0., ihalo, 4, 
    #        mh_min=3200., dir='/Users/ChangHoon/data/emanu/halos/hades/0.0eV/%i' % ihalo)

    # prior range
    prior_lim = np.array([[13.118305, 14.5], [0.5, 1.5], [13.118305, 14.5], [0.85, 1.45], [13.118305, 15.]]) 
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
        _wp = wp_model(halos, tt, rsd=True, rbins=rbins) 
        # calculate chi-squared 
        dwp = _wp_sdss - _wp 
        chisq = np.sum(np.dot(dwp.T, np.dot(C_inv, dwp))) 
        print(tt) 
        print('chi2=%f' % chisq) 
        return -0.5 * chisq

    def lnpost(tt):
        lp = lnprior(tt)
        if not np.isfinite(lp): 
            return -np.inf
        return lp + lnlike(tt) 

    pos0 = np.array([13.39, 0.56, 12.87, 1.26, 14.51])   
    objfn = lambda tt: -1.*lnpost(tt)
    #res = sp.optimize.minimize(objfn, pos0, method="Nelder-Mead", options={"maxfev": 1e2})#, "maxiter":1e3})

    fig = plt.figure(figsize=(5,5))
    sub = fig.add_subplot(111)
    _wp = wp_model(halos, np.array([13.39, 0.56, 12.87, 1.26, 14.51]), rsd=True) 
    sub.plot(rp, _wp, c='C0', label='Vakili \& Hahn (2019) best-fit') 

    model.param_dict['logMmin'] = 13.39
    model.param_dict['sigma_logM'] = 0.56
    model.param_dict['logM0'] = 12.87
    model.param_dict['alpha'] = 1.26
    model.param_dict['logM1'] = 14.51
    model.populate_mock(halocat) 
    _wp = wpCF(250., 40., 1, rbins, 
            model.mock.galaxy_table['x'], model.mock.galaxy_table['y'], model.mock.galaxy_table['z'], 
            verbose=False, output_rpavg=False) 
    sub.plot(rp, _wp['wp'], c='k', ls=':') 
    
    #_wp = wp_model(halos, res['x'], rsd=True) 
    #sub.plot(rp, _wp, c='C1', label='best-fit') 
    sub.errorbar(rp, _wp_sdss, yerr=np.sqrt(np.diag(cov)), fmt='.k') 
    sub.set_xlabel(r'$r_p$ [$h^{-1}{\rm Mpc}$]', fontsize=25) 
    sub.set_xscale('log') 
    sub.set_xlim(1e-1, 25) 
    sub.set_ylabel(r'$w_p(r_p)$', fontsize=25) 
    sub.set_yscale('log') 
    #sub.set_ylim(2, 1e3) 
    ffig = os.path.join(dir_fig, 'hod_fit.wp.png')
    fig.savefig(ffig, bbox_inches='tight') 
    plt.close() 
    return None 


def wp_model(halos, tt, rsd=True, rbins=None): 
    ''' wrapper for populating halos and calculating wp 
    '''
    if rbins is None: 
        rbins = np.array([0.1, 0.15848932, 0.25118864, 0.39810717, 0.63095734, 1., 1.58489319, 2.51188643, 3.98107171, 6.30957344, 10., 15.84893192, 25.11886432]) 
    # population halos 
    hod = FM.hodGalaxies(halos, {'logMmin': tt[0], 'sigma_logM': tt[1], 'logM0': tt[2], 'alpha': tt[3], 'logM1': tt[4]}) 
    #print('%f w/ Mh < 1.32x10^13' % (float(np.sum(np.array(hod['halo_mvir']) < 1.32*10**13))/float(len(np.array(hod['halo_mvir'])))))
    # apply RSD 
    if rsd: xyz = FM.RSD(hod) 
    else: xyz = np.array(hod['Position']) 
    # calculate wp 
    _wp = wpCF(1000., 40., 1, rbins, xyz[:,0], xyz[:,1], xyz[:,2], verbose=False, output_rpavg=False) 
    return _wp['wp']


def _plot_wp_model(theta, Mr=-21.5, low_sigma=False): 
    ''' plot wp of models for some arbitrary parameters and compare with SDSS wp
    '''
    # sdss wp 
    rp, _wp_sdss, cov = wp_sdss(Mr=Mr) 
    # r bin  
    rbins = np.loadtxt(os.path.join(dir_hod, 'rbins.dat'), unpack=True, usecols=[0]) 
    # halo catalogs 
    halos = hadesData.hadesMnuHalos(0., 1, 4, 
            mh_min=None, dir='/Users/ChangHoon/data/emanu/halos/hades/0.0eV/1')
    Mh_min = np.log10(np.array(halos['Mass']).min()) 
    # hi res halo catalog
    halos_hires = hadesData.hadesMnuHalos(0., '1_hires', 4, 
            mh_min=None, dir='/Users/ChangHoon/data/emanu/halos/hades/0.0eV/1_hires')
    if theta == 'zheng2007': 
        lbl = 'Zheng+(2007)'
        tt = np.array([13.38, 0.51, 13.94, 1.04, 13.91])
    elif theta == 'guo2015': 
        lbl = 'Guo+(2015)'
        tt = np.array([13.53, 0.72, 13.13, 1.14, 14.52])
    elif theta == 'vakili2019': 
        lbl = 'Vakili+(2019)'
        tt = np.array([13.39, 0.56, 12.87, 1.26, 14.51])
    elif theta == 'zhengMr22':
        lbl = 'Zheng+(2007) $M_r = -22$'
        tt = np.array([14.22, 0.77, 14.00, 0.87, 14.69])
    else: 
        raise NotImplementedError
    if low_sigma: tt[1] = 0.2 
    
    tt_dict = {'logMmin': tt[0], 'sigma_logM': tt[1], 'logM0': tt[2], 'alpha': tt[3], 'logM1': tt[4]}
    hod = FM.hodGalaxies(halos, tt_dict)
    fmlim = float(np.sum(np.array(hod['halo_mvir']) < 10**Mh_min))/float(len(np.array(hod['halo_mvir'])))
    print fmlim 

    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(121)
    _Mhbins = np.logspace(11, 15, 200)
    Ncen = Ncen_Mh(tt, np.log10(_Mhbins)) 
    Nsat = Nsat_Mh(tt, np.log10(_Mhbins)) 
    sub.plot(_Mhbins, Ncen, c='C0')
    sub.plot(_Mhbins, Nsat, c='C1')
    sub.plot(_Mhbins, Ncen+Nsat, c='k', ls=':', lw=1) 
    sub.plot([10**Mh_min, 10**Mh_min], [1e-2, 1e1], c='k', ls='--')
    sub.text(0.2, 0.95, r'%.1e of halos below $M_{\rm min}$' % fmlim, 
            ha='left', va='top', transform=sub.transAxes, fontsize=15)
    sub.set_xlabel('$M_h$', fontsize=25) 
    sub.set_xscale('log') 
    sub.set_xlim(5e12, 1e15) 
    sub.set_ylabel(r'$<N_{\rm gal}>$', fontsize=25) 
    sub.set_yscale('log') 
    sub.set_ylim(1e-1, 1e1) 

    sub = fig.add_subplot(122)
    _wp = wp_model(halos, tt, rsd=True) 
    sub.plot(rp, _wp, c='C1', label=lbl) 
    _wp = wp_model(halos_hires, tt, rsd=True) 
    sub.plot(rp, _wp, c='C1', ls=':', label='High Res.') 
    sub.errorbar(rp, _wp_sdss, yerr=np.sqrt(np.diag(cov)), fmt='.k') 
    sub.legend(loc='lower left', fontsize=15) 
    sub.set_xlabel(r'$r_p$ [$h^{-1}{\rm Mpc}$]', fontsize=25) 
    sub.set_xscale('log') 
    sub.set_xlim(1e-1, 25) 
    sub.set_ylabel(r'$w_p(r_p)$', fontsize=25) 
    sub.set_yscale('log') 
    #sub.set_ylim(2, 1e3) 
    fig.subplots_adjust(wspace=0.3) 
    if not low_sigma: ffig = os.path.join(dir_fig, 'wp_model.%s.png' % theta)
    else: ffig = os.path.join(dir_fig, 'wp_model.%s.lowsigma.png' % theta)
    fig.savefig(ffig, bbox_inches='tight') 
    plt.close() 
    return None 


def Ncen_Mh(tt, logMh): 
    ''' expected Ncen at Mh 
    '''
    logMmin, sig_logM, _, _, _ = tt
    return 0.5 * (1. + sp.special.erf((logMh - logMmin)/sig_logM)) 


def Nsat_Mh(tt, logMh): 
    ''' expected Nsat at Mh
    '''
    _Ncen = Ncen_Mh(tt, logMh) 
    _, _, logM0, alpha, logM1 = tt
    return _Ncen * ((10**logMh - 10**logM0)/10**logM1)**alpha


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
    for tt in ['zheng2007', 'guo2015', 'vakili2019', 'zhengMr22']: 
        _plot_wp_model(tt, Mr=-21.5, low_sigma=False)
        _plot_wp_model(tt, Mr=-21.5, low_sigma=True)
    #hod_fit(Mr=-21.5)
