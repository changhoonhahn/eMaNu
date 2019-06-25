'''
'''
import os 
import scipy as sp 
import numpy as np 
# --- eMaNu --- 
from emanu import util as UT
from emanu import forwardmodel as FM
from emanu.hades import data as hadesData
# -- pyspectrum -- 
from pyspectrum import pyspectrum as pySpec
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
    if low_sigma: tt[1] = low_sigma 
    
    tt_dict = {'logMmin': tt[0], 'sigma_logM': tt[1], 'logM0': tt[2], 'alpha': tt[3], 'logM1': tt[4]}
    hod = FM.hodGalaxies(halos, tt_dict)
    fmlim = float(np.sum(np.array(hod['halo_mvir']) < 10**Mh_min))/float(len(np.array(hod['halo_mvir'])))
    Nmlim = np.sum(np.array(hod['halo_mvir']) < 10**Mh_min)
    print fmlim, Nmlim 

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
    sub.text(0.2, 0.85, r'%i halos' % Nmlim, 
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
    sub.set_ylim(2, 5e3) 
    fig.subplots_adjust(wspace=0.3) 
    if not low_sigma: ffig = os.path.join(dir_fig, 'wp_model.%s.png' % theta)
    else: ffig = os.path.join(dir_fig, 'wp_model.%s.sig%.1f.png' % (theta, low_sigma))
    fig.savefig(ffig, bbox_inches='tight') 
    plt.close() 
    return None 


def _plot_wp_model_pm(theta, Mr=-21.5, low_sigma=False): 
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

    dtt = np.array([0.1, 0.1, 0.3, 0.3, 0.1]) 
    if theta == 'zheng2007': 
        lbl = 'Zheng+(2007)'
        tt  = np.array([13.38, 0.51, 13.94, 1.04, 13.91])
    elif theta == 'guo2015': 
        lbl = 'Guo+(2015)'
        tt  = np.array([13.53, 0.72, 13.13, 1.14, 14.52])
    elif theta == 'vakili2019': 
        lbl = 'Vakili+(2019)'
        tt  = np.array([13.39, 0.56, 12.87, 1.26, 14.51])
    elif theta == 'zhengMr22':
        lbl = 'Zheng+(2007) $M_r = -22$'
        tt  = np.array([14.22, 0.77, 14.00, 0.87, 14.69])
    else: 
        raise NotImplementedError
    ttm = tt - dtt
    ttp = tt + dtt 
    if low_sigma: 
        tt[1]  = low_sigma 
        ttm[1] = low_sigma - dtt[1]
        ttp[1] = low_sigma + dtt[1]
    lbls = [r'$\log M_{\rm min}$', r'$\sigma_{\log M}$',r'$\log M_0$',r'$\alpha$',r'$\log M_1$']
    names = ['logMmin', 'sig_logM', 'logM0', 'alpha', 'logM1']
    
    tt_fid = tt.copy()
    tt_dict = {'logMmin': tt_fid[0], 'sigma_logM': tt_fid[1], 'logM0': tt_fid[2], 'alpha': tt_fid[3], 'logM1': tt_fid[4]}
    hod = FM.hodGalaxies(halos, tt_dict)
    fmlim = float(np.sum(np.array(hod['halo_mvir']) < 10**Mh_min))/float(len(np.array(hod['halo_mvir'])))
    Nmlim = np.sum(np.array(hod['halo_mvir']) < 10**Mh_min)
    print tt_fid
    print('fiducial, %.3f, %i' % (fmlim, Nmlim))

    _Mhbins = np.logspace(11, 15, 300)
    Ncen_fid = Ncen_Mh(tt_fid, np.log10(_Mhbins)) 
    Nsat_fid = Nsat_Mh(tt_fid, np.log10(_Mhbins)) 
    wp_fid = wp_model(halos, tt_fid, rsd=True) 
    wp_fid_hires = wp_model(halos_hires, tt_fid, rsd=True) 
    
    fig = plt.figure(figsize=(12,25))
    for i in range(5): 
        _ttm = tt.copy() 
        _ttm[i] = ttm[i] 
        sub = fig.add_subplot(5,2,2*i+1)
        tt_dict = {'logMmin': _ttm[0], 'sigma_logM': _ttm[1], 
                'logM0': _ttm[2], 'alpha': _ttm[3], 'logM1': _ttm[4]}
        hod = FM.hodGalaxies(halos, tt_dict)
        fmlim = float(np.sum(np.array(hod['halo_mvir']) < 10**Mh_min))/float(len(np.array(hod['halo_mvir'])))
        Nmlim = np.sum(np.array(hod['halo_mvir']) < 10**Mh_min)
        print('%s- %.3f, %i' % (names[i], fmlim, Nmlim))
        _ttp = tt.copy() 
        _ttp[i] = ttp[i] 
        tt_dict = {'logMmin': _ttp[0], 'sigma_logM': _ttp[1], 
                'logM0': _ttp[2], 'alpha': _ttp[3], 'logM1': _ttp[4]}
        hod = FM.hodGalaxies(halos, tt_dict)
        fmlim = float(np.sum(np.array(hod['halo_mvir']) < 10**Mh_min))/float(len(np.array(hod['halo_mvir'])))
        Nmlim = np.sum(np.array(hod['halo_mvir']) < 10**Mh_min)
        print('%s+ %.3f, %i' % (names[i], fmlim, Nmlim))

        sub.plot(_Mhbins, Ncen_fid, c='k', ls='--', lw=1) 
        sub.plot(_Mhbins, Nsat_fid, c='k', ls=':', lw=1) 
        Ncen = Ncen_Mh(_ttm, np.log10(_Mhbins)) 
        Nsat = Nsat_Mh(_ttm, np.log10(_Mhbins)) 
        sub.plot(_Mhbins, Ncen, c='C1', lw=1, ls='--') 
        sub.plot(_Mhbins, Nsat, c='C1', lw=1, ls=':') 
        Ncen = Ncen_Mh(_ttp, np.log10(_Mhbins)) 
        Nsat = Nsat_Mh(_ttp, np.log10(_Mhbins)) 
        sub.plot(_Mhbins, Ncen, c='C0', lw=1, ls='--') 
        sub.plot(_Mhbins, Nsat, c='C0', lw=1, ls=':') 
        sub.plot([10**Mh_min, 10**Mh_min], [1e-2, 1e1], c='k', ls='--')
        #sub.text(0.2, 0.95, r'%.1e of halos below $M_{\rm min}$' % fmlim, 
        #        ha='left', va='top', transform=sub.transAxes, fontsize=15)
        #sub.text(0.2, 0.85, r'%i halos' % Nmlim, 
        #        ha='left', va='top', transform=sub.transAxes, fontsize=15)
        if i == 4: sub.set_xlabel('$M_h$', fontsize=25) 
        sub.set_xscale('log') 
        sub.set_xlim(5e12, 1e15) 
        sub.set_ylabel(r'$<N_{\rm gal}>$', fontsize=25) 
        sub.set_yscale('log') 
        sub.set_ylim(1e-1, 1e1) 

        sub = fig.add_subplot(5,2,2*i+2)
        sub.plot(rp, wp_fid, c='k') 
        sub.plot(rp, wp_fid_hires, c='k', ls=':') 
        _wp = wp_model(halos, _ttm, rsd=True) 
        sub.plot(rp, _wp, c='C1', label=r'$\theta_-$') 
        _wp = wp_model(halos, _ttp, rsd=True) 
        sub.plot(rp, _wp, c='C0', label=r'$\theta_+$') 
        _wp = wp_model(halos_hires, _ttm, rsd=True) 
        sub.plot(rp, _wp, c='C1', ls=':') 
        _wp = wp_model(halos_hires, _ttp, rsd=True) 
        sub.plot(rp, _wp, c='C0', ls=':', label='High Res.') 
        if i == 0: sub.legend(loc='lower left', fontsize=15) 
        sub.errorbar(rp, _wp_sdss, yerr=np.sqrt(np.diag(cov)), fmt='.k') 
        if i == 4: sub.set_xlabel(r'$r_p$ [$h^{-1}{\rm Mpc}$]', fontsize=25) 
        sub.text(0.95, 0.95, lbls[i], ha='right', va='top', transform=sub.transAxes, fontsize=20)
        sub.set_xscale('log') 
        sub.set_xlim(1e-1, 25) 
        sub.set_ylabel(r'$w_p(r_p)$', fontsize=25) 
        sub.set_yscale('log') 
        sub.set_ylim(2, 5e3) 
    fig.subplots_adjust(wspace=0.3) 
    if not low_sigma: ffig = os.path.join(dir_fig, 'wp_model_pm.%s.png' % theta)
    else: ffig = os.path.join(dir_fig, 'wp_model_pm.%s.sig%.1f.png' % (theta, low_sigma))
    fig.savefig(ffig, bbox_inches='tight') 
    plt.close() 
    return None 


def _plot_wp_model_pm_fiducial(): 
    ''' plot wp of models for fiducial parameter and the derivatives 
    '''
    rp, _wp_sdss, cov = wp_sdss(Mr=-21.5) 
    # r bin  
    rbins = np.loadtxt(os.path.join(dir_hod, 'rbins.dat'), unpack=True, usecols=[0]) 
    # halo catalogs 
    halos = hadesData.hadesMnuHalos(0., 1, 4, 
            mh_min=None, dir='/Users/ChangHoon/data/emanu/halos/hades/0.0eV/1')
    Mh_min = np.log10(np.array(halos['Mass']).min()) 
    # hi res halo catalog
    halos_hires = hadesData.hadesMnuHalos(0., '1_hires', 4, 
            mh_min=None, dir='/Users/ChangHoon/data/emanu/halos/hades/0.0eV/1_hires')

    dtt = np.array([0.1, 0.1, 0.3, 0.3, 0.1]) 
    tt  = np.array([13.85, 0.175, 13.94, 0.95, 14.3]) 
    ttm = tt - dtt
    ttp = tt + dtt 

    lbls = [r'$\log M_{\rm min}$', r'$\sigma_{\log M}$',r'$\log M_0$',r'$\alpha$',r'$\log M_1$']
    names = ['logMmin', 'sig_logM', 'logM0', 'alpha', 'logM1']
    
    _Mhbins = np.logspace(11, 16, 200)
    Ncen_fid = Ncen_Mh(tt, np.log10(_Mhbins)) 
    Nsat_fid = Nsat_Mh(tt, np.log10(_Mhbins)) 
    wp_fid = wp_model(halos, tt, rsd=True) 
    wp_fid_hires = wp_model(halos_hires, tt, rsd=True) 

    tt_z07_21p5 = np.array([13.38, 0.51, 13.94, 1.04, 13.91])
    tt_z07_22p0 = np.array([14.22, 0.77, 14.00, 0.87, 14.69])
    Ncen_z07_21p5 = Ncen_Mh(tt_z07_21p5, np.log10(_Mhbins))
    Nsat_z07_21p5 = Nsat_Mh(tt_z07_21p5, np.log10(_Mhbins))
    Ncen_z07_22p0 = Ncen_Mh(tt_z07_22p0, np.log10(_Mhbins))
    Nsat_z07_22p0 = Nsat_Mh(tt_z07_22p0, np.log10(_Mhbins))
    
    fig = plt.figure(figsize=(40,8))
    for i in range(5): 
        _ttm = tt.copy() 
        _ttm[i] = ttm[i] 
        sub = fig.add_subplot(2,5,i+1)
        tt_dict = {'logMmin': _ttm[0], 'sigma_logM': _ttm[1], 
                'logM0': _ttm[2], 'alpha': _ttm[3], 'logM1': _ttm[4]}
        hod = FM.hodGalaxies(halos_hires, tt_dict)
        fmlim = float(np.sum(np.array(hod['halo_mvir']) < 10**Mh_min))/float(len(np.array(hod['halo_mvir'])))
        Nmlim = np.sum(np.array(hod['halo_mvir']) < 10**Mh_min)
        print('%s- %.4f, %i' % (names[i], fmlim, Nmlim))
        _ttp = tt.copy() 
        _ttp[i] = ttp[i] 
        tt_dict = {'logMmin': _ttp[0], 'sigma_logM': _ttp[1], 
                'logM0': _ttp[2], 'alpha': _ttp[3], 'logM1': _ttp[4]}
        hod = FM.hodGalaxies(halos_hires, tt_dict)
        fmlim = float(np.sum(np.array(hod['halo_mvir']) < 10**Mh_min))/float(len(np.array(hod['halo_mvir'])))
        Nmlim = np.sum(np.array(hod['halo_mvir']) < 10**Mh_min)
        print('%s+ %.4f, %i' % (names[i], fmlim, Nmlim))

        sub.plot(_Mhbins, Ncen_fid + Nsat_fid, c='k', ls='-', lw=1) 
        Ncen = Ncen_Mh(_ttm, np.log10(_Mhbins)) 
        Nsat = Nsat_Mh(_ttm, np.log10(_Mhbins)) 
        sub.plot(_Mhbins, Ncen + Nsat, c='C1', lw=1, ls='-') 
        Ncen = Ncen_Mh(_ttp, np.log10(_Mhbins)) 
        Nsat = Nsat_Mh(_ttp, np.log10(_Mhbins)) 
        sub.plot(_Mhbins, Ncen + Nsat, c='C0', lw=1, ls='-') 

        plt_21p5, = sub.plot(_Mhbins, Ncen_z07_21p5 + Nsat_z07_21p5, c='k', ls=':', lw=1) 
        plt_22p0, = sub.plot(_Mhbins, Ncen_z07_22p0 + Nsat_z07_22p0, c='k', ls='--', lw=1) 

        sub.fill_between([10**12, 10**Mh_min], [1e-2, 1e-2], [1e2, 1e2], color='k', linewidth=0, alpha=0.5)
        #if i == 4: sub.set_xlabel('$M_h$', fontsize=25) 
        sub.set_xscale('log') 
        sub.set_xlim(5e12, 5e15) 
        sub.set_yscale('log') 
        sub.set_ylim(1e-1, 1e2) 
        if i == 0: sub.set_ylabel(r'$<N_{\rm gal}>$', fontsize=25) 
        else: sub.set_yticklabels([]) 
        if i == 4: sub.legend([plt_21p5, plt_22p0], ['SDSS $M_r < -21.5$', 'SDSS $M_r < -22$'],
                loc='upper left', fontsize=12) 

        sub = fig.add_subplot(2,5,i+6)
        sub.plot(rp, wp_fid, c='k') 
        sub.plot(rp, wp_fid_hires, c='k', ls=':') 
        _wp = wp_model(halos, _ttm, rsd=True) 
        sub.plot(rp, _wp, c='C1', label=r'$\theta_-$') 
        _wp = wp_model(halos, _ttp, rsd=True) 
        sub.plot(rp, _wp, c='C0', label=r'$\theta_+$') 
        _wp = wp_model(halos_hires, _ttm, rsd=True) 
        sub.plot(rp, _wp, c='C1', ls=':') 
        _wp = wp_model(halos_hires, _ttp, rsd=True) 
        sub.plot(rp, _wp, c='C0', ls=':', label='High Res.') 
        if i == 4: sub.legend(loc='lower left', fontsize=15) 
        
        sub.text(0.95, 0.95, '%s=$%s\pm %.1f$' % (lbls[i], str(tt[i]), dtt[i]), 
                ha='right', va='top', transform=sub.transAxes, fontsize=15)
        sub.set_xscale('log') 
        sub.set_xlim(1e-1, 25) 
        sub.set_yscale('log') 
        sub.set_ylim(1e1, 3e4) 
        if i == 0: sub.set_ylabel(r'$w_p(r_p)$', fontsize=25) 
        else: sub.set_yticklabels([]) 

    bkgd = fig.add_subplot(211, frameon=False)
    bkgd.set_xlabel('$M_h$', fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd = fig.add_subplot(212, frameon=False)
    bkgd.set_xlabel(r'$r_p$ [$h^{-1}{\rm Mpc}$]', fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    fig.subplots_adjust(wspace=0.05, hspace=0.3) 
    ffig = os.path.join(dir_fig, 'wp_model_pm.fiducial.png')
    fig.savefig(ffig, bbox_inches='tight') 
    plt.close() 
    return None 


def _plot_dPdHOD_fiducials(): 
    ''' plot the P derivatives w.r.t. HOD parameters for multiple fiducial 
    parameter choices. The main goal is to see how much the derivatives 
    change for higher sigmas 
    '''
    # r bin  
    rbins = np.loadtxt(os.path.join(dir_hod, 'rbins.dat'), unpack=True, usecols=[0]) 
    # hi res halo catalog
    halos = hadesData.hadesMnuHalos(0., '1_hires', 4, 
            mh_min=None, dir='/Users/ChangHoon/data/emanu/halos/hades/0.0eV/1_hires')
    
    dtt = np.array([0.1, 0.1, 0.3, 0.3, 0.1]) 
    fid0 = np.array([13.85, 0.2, 13.94, 0.95, 14.3]) 
    fid1 = np.array([13.85, 0.4, 13.94, 0.95, 14.3]) 
    fid2 = np.array([13.85, 0.5, 13.94, 0.95, 14.3]) 
    fids = [fid0, fid1, fid2]
    
    lbls = [r'$\log M_{\rm min}$', r'$\sigma_{\log M}$',r'$\log M_0$',r'$\alpha$',r'$\log M_1$']
    names = ['logMmin', 'sig_logM', 'logM0', 'alpha', 'logM1']
    fig = plt.figure(figsize=(30,8))
    for i in range(5): 
        sub = fig.add_subplot(2,3,i+1)
        for i_tt, tt in enumerate(fids):  # loop through different fiducial values 
            # theta + and theta - 
            _ttp = tt.copy() 
            _ttp[i] = tt[i] + dtt[i]
            ttp_dict = {'logMmin': _ttp[0], 'sigma_logM': _ttp[1], 
                    'logM0': _ttp[2], 'alpha': _ttp[3], 'logM1': _ttp[4]}
            _ttm = tt.copy() 
            _ttm[i] = tt[i] - dtt[i] 
            ttm_dict = {'logMmin': _ttm[0], 'sigma_logM': _ttm[1], 
                    'logM0': _ttm[2], 'alpha': _ttm[3], 'logM1': _ttm[4]}
            print('%s+' % names[i], _ttp)
            print('%s-' % names[i], _ttm)
            
            pkps, pkms = [], [] 
            for _iii in range(1): # average over 10 
                print('realization %i' % _iii) 
                print('... populating theta+') 
                hod = FM.hodGalaxies(halos, ttp_dict)
                pos_p = np.array(hod['Position'])
                print('... populating theta-') 
                hod = FM.hodGalaxies(halos, ttm_dict)
                pos_m = np.array(hod['Position'])
                print('... calculating Pk') 
                _pkpi = pySpec.Pk_periodic(pos_p.T, Lbox=1000, Ngrid=360, fft='pyfftw', silent=True)
                _pkmi = pySpec.Pk_periodic(pos_m.T, Lbox=1000, Ngrid=360, fft='pyfftw', silent=True)

                k = _pkpi['k'] 
                pkps.append(_pkpi['p0k']) 
                pkms.append(_pkmi['p0k']) 

            pkp = np.average(pkps, axis=0) # average P 
            pkm = np.average(pkms, axis=0) 
            #dpdhod = 0.5*(pkp - pkm)/dtt[i] # derivative 
            dlogpdhod = 0.5*(np.log(pkp) - np.log(pkm))/dtt[i] # log derivative 
            sub.plot(k, dlogpdhod, c='C%i' % i_tt, ls='-', lw=1, label=r'$\sigma_{\log M}=%.2f$' % tt[1]) 
        
        sub.set_xscale('log') 
        sub.set_xlim(5e-3, 0.5) 
        #sub.set_ylim(ylims[i_tt]) 
        sub.text(0.05, 0.95, r'%s' % lbls[i], ha='left', va='top', transform=sub.transAxes, fontsize=20)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel('$k$', fontsize=25) 
    bkgd.set_ylabel(r'${\rm d}\log P/{\rm d} \theta$', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    fig.subplots_adjust(wspace=0.2) 
    ffig = os.path.join(dir_fig, 'dPdHOD_fiducials.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    plt.close() 
    return None 


def Ncen_Mh(tt, logMh): 
    ''' expected Ncen at Mh 
    '''
    logMmin, sig_logM, _, _, _ = tt
    Ncen = 0.5 * (1. + sp.special.erf((logMh - logMmin)/sig_logM)) 
    Ncen[~np.isfinite(Ncen)] = 0. 
    return Ncen


def Nsat_Mh(tt, logMh): 
    ''' expected Nsat at Mh
    '''
    _Ncen = Ncen_Mh(tt, logMh) 
    _, _, logM0, alpha, logM1 = tt
    Nsat = _Ncen * ((10**logMh - 10**logM0)/10**logM1)**alpha
    Nsat[~np.isfinite(Nsat)] = 0. 
    return Nsat 


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
    #for tt in ['guo2015']:#, 'vakili2019', 'zhengMr22']: #'zheng2007', 
    #    print tt 
    #    _plot_wp_model_pm(tt, Mr=-21.5, low_sigma=False)
    #    _plot_wp_model_pm(tt, Mr=-21.5, low_sigma=0.2)
    #    _plot_wp_model_pm(tt, Mr=-21.5, low_sigma=0.1)
    #_plot_wp_model_pm_fiducial()
    _plot_dPdHOD_fiducials()

    #hod_fit(Mr=-21.5)
