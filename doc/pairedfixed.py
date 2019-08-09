'''

investigate paired-fixed simulations for the bispectrum

'''
import os 
import h5py
import numpy as np 
from scipy.stats import skew, skewtest
from copy import copy as copy
# --- eMaNu --- 
from emanu import util as UT
from emanu import obvs as Obvs
from emanu import plots as ePlots
from emanu import forecast as Forecast
# --- plotting --- 
import matplotlib as mpl
import matplotlib.pyplot as plt
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


dir_bk  = os.path.join(UT.dat_dir(), 'bispectrum') # bispectrum directory
dir_fig = os.path.join(UT.fig_dir(), 'pairedfixed') 
dir_doc = os.path.join(UT.doc_dir(), 'paper4', 'figs') # figures for paper 
kf = 2.*np.pi/1000. 


def pf_Pk(): 
    ''' comparison of the paired-fixed P_l to standard N-body replicating 
    Fig. 2 in Angulo & Pontzen (2016). 

    3 panel plot: 
    - top: P_ell(k) comparison
    - center: Delta P0 / sigma P0
    - bottom: Delta P2 / sigma P2 
    '''
    fig = plt.figure(figsize=(17, 9))
    gs = mpl.gridspec.GridSpec(2, 3, figure=fig, height_ratios=[2,1], hspace=0.05) 
    sub0 = plt.subplot(gs[0])
    sub1 = plt.subplot(gs[1]) 
    sub2 = plt.subplot(gs[2]) 
    sub3 = plt.subplot(gs[3])
    sub4 = plt.subplot(gs[4]) 
    sub5 = plt.subplot(gs[5]) 

    # read in real-space P (standard)
    k_real, p0k_real_std = X_std('pk', 'fiducial', rsd='real', silent=False) 
    # read in real-space P (paired-fixed)
    _, p0k_real_pfd1 = X_pfd_1('pk', 'fiducial', rsd='real', silent=False) 
    _, p0k_real_pfd2 = X_pfd_2('pk', 'fiducial', rsd='real', silent=False) 
    p0k_real_pfd = 0.5 * (p0k_real_pfd1 + p0k_real_pfd2) 
    
    # read in power spectrum (standard)
    k_rsd, p0k_rsd_std, p2k_rsd_std = X_std('pk', 'fiducial', rsd=0, silent=False) 
    # read in power spectrum (pairedfixed)
    _, p0k_rsd_pfd1, p2k_rsd_pfd1 = X_pfd_1('pk', 'fiducial', rsd=0, silent=False) 
    _, p0k_rsd_pfd2, p2k_rsd_pfd2 = X_pfd_2('pk', 'fiducial', rsd=0, silent=False) 
    p0k_rsd_pfd = 0.5 * (p0k_rsd_pfd1 + p0k_rsd_pfd2) 
    p2k_rsd_pfd = 0.5 * (p2k_rsd_pfd1 + p2k_rsd_pfd2) 

    # --- Pell comparison --- 
    # average real-space P0
    sub0.plot(k_real, np.average(p0k_real_std, axis=0), c='k', label='%i standard $N$-body' % p0k_real_std.shape[0])
    sub0.scatter(k_real, np.average(p0k_real_pfd, axis=0), color='C1', s=5, zorder=10, 
            label='%i paired-fixed pairs' % p0k_real_pfd.shape[0])
    sub0.text(0.95, 0.95, r'real-space $P_0$', ha='right', va='top', transform=sub0.transAxes, fontsize=20)
    sub0.legend(loc='lower left', markerscale=5, handletextpad=0.2, fontsize=20) 
    sub0.set_xlim(9e-3, 0.5) 
    sub0.set_xscale('log') 
    sub0.set_xticklabels([]) 
    sub0.set_ylabel('$P_\ell(k)$', fontsize=25) 
    sub0.set_yscale('log') 
    sub0.set_ylim(1e3, 2e5) 
    # average redshift-space P0
    sub1.plot(k_rsd, np.average(p0k_rsd_std, axis=0), c='k')
    sub1.scatter(k_rsd, np.average(p0k_rsd_pfd, axis=0), color='C1', s=5, zorder=10) 
    sub1.text(0.95, 0.95, r'redshift-space $P_0$', ha='right', va='top', transform=sub1.transAxes, fontsize=20)
    sub1.set_xlim(9e-3, 0.5) 
    sub1.set_xscale('log') 
    sub1.set_xticklabels([]) 
    sub1.set_yscale('log') 
    sub1.set_ylim(1e3, 2e5) 
    sub1.set_yticklabels([]) 
    # average redshift-space P2
    sub2.plot(k_rsd, np.average(p2k_rsd_std, axis=0), c='k')
    sub2.scatter(k_rsd, np.average(p2k_rsd_pfd, axis=0), color='C1', s=5, zorder=10) 
    sub2.text(0.95, 0.95, r'redshift-space $P_2$', ha='right', va='top', transform=sub2.transAxes, fontsize=20)
    sub2.set_xlim(9e-3, 0.5) 
    sub2.set_xscale('log') 
    sub2.set_xticklabels([]) 
    sub2.set_yscale('log') 
    sub2.set_ylim(1e3, 2e5) 
    sub2.set_yticklabels([]) 
    
    # --- bias comparison --- 
    thetas = ['fiducial', 'Om_m', 'Om_p', 'Ob2_m', 'Ob2_p', 'h_m', 'h_p', 'ns_m', 'ns_p', 's8_m', 's8_p']
    lbls = ['fid.', r'$\Omega_m^{-}$', r'$\Omega_m^{+}$', r'$\Omega_b^{-}$', r'$\Omega_b^{+}$', r'$h^{-}$', r'$h^{+}$', 
            r'$n_s^{-}$', r'$n_s^{+}$', r'$\sigma_8^{-}$', r'$\sigma_8^{+}$']
    
    for _i, theta in enumerate(thetas): 
        # read in real-space P (standard)
        k_real, p0k_real_std = X_std('pk', theta, rsd='real', silent=False) 
        # read in real-space P (paired-fixed)
        _, p0k_real_pfd1 = X_pfd_1('pk', theta, rsd='real', silent=False) 
        _, p0k_real_pfd2 = X_pfd_2('pk', theta, rsd='real', silent=False) 
        p0k_real_pfd = 0.5 * (p0k_real_pfd1 + p0k_real_pfd2) 
        
        # read in power spectrum (standard)
        k_rsd, p0k_rsd_std, p2k_rsd_std = X_std('pk', theta, rsd=0, silent=False) 
        # read in power spectrum (pairedfixed)
        _, p0k_rsd_pfd1, p2k_rsd_pfd1 = X_pfd_1('pk', theta, rsd=0, silent=False) 
        _, p0k_rsd_pfd2, p2k_rsd_pfd2 = X_pfd_2('pk', theta, rsd=0, silent=False) 
        p0k_rsd_pfd = 0.5 * (p0k_rsd_pfd1 + p0k_rsd_pfd2) 
        p2k_rsd_pfd = 0.5 * (p2k_rsd_pfd1 + p2k_rsd_pfd2) 

        bias_p0k_real   = pf_bias(p0k_real_std, p0k_real_pfd)
        bias_p0k_rsd    = pf_bias(p0k_rsd_std, p0k_rsd_pfd)
        bias_p2k_rsd    = pf_bias(p2k_rsd_std, p2k_rsd_pfd)
    
        clr = 'k'
        if _i > 0: clr = 'C%i' % (_i-1) 
        # bias real-space P0
        _plt, = sub3.plot(k_real, bias_p0k_real, c=clr)
        sub3.plot([9e-3, 0.5], [0.0, 0.0], c='k', ls='--') 
        sub3.set_xlim(9e-3, 0.5) 
        sub3.set_xscale('log') 
        sub3.set_ylabel(r'bias ($\beta$)', fontsize=25) 
        sub3.set_ylim(-3.5, 3.5) 
        if _i == 0: plts = [] 
        plts.append(_plt) 

        sub4.plot(k_rsd, bias_p0k_rsd, c=clr)
        sub4.plot([9e-3, 0.5], [0.0, 0.0], c='k', ls='--') 
        sub4.set_xlim(9e-3, 0.5) 
        sub4.set_xscale('log') 
        sub4.set_ylim(-3.5, 3.5) 
        sub4.set_yticklabels([]) 

        sub5.plot(k_rsd, bias_p2k_rsd, c=clr)
        sub5.plot([9e-3, 0.5], [0.0, 0.0], c='k', ls='--') 
        sub5.set_xlim(9e-3, 0.5) 
        sub5.set_xscale('log') 
        sub5.set_ylim(-3.5, 3.5) 
        sub5.set_yticklabels([]) 

    sub3.legend(plts[:4], lbls[:4], loc='lower left', ncol=6, handletextpad=0.4, fontsize=15)
    sub4.legend(plts[4:8], lbls[4:8], loc='lower left', ncol=6, handletextpad=0.4, fontsize=15)
    sub5.legend(plts[8:], lbls[8:], loc='lower left', ncol=6, handletextpad=0.4, fontsize=15)

    fig.subplots_adjust(wspace=0.1) 
    ffig = os.path.join(dir_doc, 'pf_Pk.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') 
    return None 


def pf_Bk(rsd=0, kmax=0.5): 
    ''' comparison of the paired-fixed B0 to standard N-body replicating 
    Fig. 2 in Angulo & Pontzen (2016). 

    3 panel plot: 
    - top: B0(k) comparison
    - center: Delta B0 / sigma P0
    - bottom: Delta P2 / sigma P2 
    '''
    fig = plt.figure(figsize=(15, 8))
    gs = mpl.gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3,2], hspace=0.1) 
    sub0 = plt.subplot(gs[0])
    sub1 = plt.subplot(gs[1]) 
    
    # read in bispectrum (standard)
    quij = Obvs.quijoteBk('fiducial', rsd=rsd, flag='reg', silent=False) 
    i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3']
    klim = ((i_k * kf <= kmax) & (j_k * kf <= kmax) & (l_k * kf <= kmax)) 
    bks_std = quij['b123'][:,klim] # P0 
    # read in bispectrum (pairedfixed)
    quij = Obvs.quijoteBk('fiducial', rsd=rsd, flag='ncv', silent=False) 
    bks_pfd = quij['b123'][:,klim] # P0 
    # read in covariance matrix 
    if rsd == 'real': 
        _, _, _, Cov_bk, _ = bkCov(rsd='real', flag='reg', silent=False) 
    else: 
        _, _, _, Cov_bk, _ = bkCov(rsd=2, flag='reg', silent=False) 
    sig_bk = np.sqrt(np.diag(Cov_bk))[klim]

    # average B0
    sub0.plot(range(np.sum(klim)), np.average(bks_std, axis=0), c='C0', label='%i standard $N$-body' % bks_std.shape[0])
    sub0.scatter(range(np.sum(klim)),np.average(bks_pfd, axis=0), color='C1', s=5, label='%i suppressed variance' % bks_pfd.shape[0], zorder=10) 
    #sub0.text(0.95, 0.95, r'redshift-space $B_0^{\rm halo}$', ha='right', va='top', transform=sub0.transAxes, fontsize=20)
    sub0.legend(loc='upper right', markerscale=5, handletextpad=0.2, fontsize=20) 
    sub0.set_xlim(0, np.sum(klim))
    sub0.set_xticklabels([]) 
    if rsd == 'real': 
        sub0.set_ylabel('real-space halo $B_0(k_1, k_2, k_3)$', fontsize=20)
    else: 
        sub0.set_ylabel('redshift-space halo $B_0(k_1, k_2, k_3)$', fontsize=20)
    sub0.set_yscale('log') 
    if rsd == 'real': sub0.set_ylim(1e4, 1e10)
    else: sub0.set_ylim(2e5, 1e10)

    # Delta0/sigma comparison
    thetas = ['fiducial', 'Om_m', 'Om_p', 'Ob2_m', 'Ob2_p', 'h_m', 'h_p', 'ns_m', 'ns_p', 's8_m', 's8_p']
    lbls = ['fid.', r'$\Omega_m^{-}$', r'$\Omega_m^{+}$', r'$\Omega_b^{-}$', r'$\Omega_b^{+}$', r'$h^{-}$', r'$h^{+}$', 
            r'$n_s^{-}$', r'$n_s^{+}$', r'$\sigma_8^{-}$', r'$\sigma_8^{+}$']
    plts = []
    for theta in thetas: 
        # read in power spectrum (standard)
        quij = Obvs.quijoteBk(theta, rsd=rsd, flag='reg', silent=False) 
        bks_std = quij['b123'][:,klim] # P0 
        # read in bispectrum (pairedfixed)
        quij = Obvs.quijoteBk(theta, rsd=rsd, flag='ncv', silent=False) 
        bks_pfd = quij['b123'][:,klim] # P0 

        delB0_sig = (np.average(bks_pfd, axis=0) - np.average(bks_std, axis=0))/sig_bk
        if theta == 'fiducial': 
            _plt, = sub1.plot(range(np.sum(klim)), delB0_sig, lw=1, zorder=10, color='k')  
        else: 
            _plt, = sub1.plot(range(np.sum(klim)), delB0_sig, lw=1, zorder=10)  
        plts.append(_plt) 
    sub1.plot([0, np.sum(klim)], [0, 0], c='k', ls=':', zorder=10) 
    
    inv_V = 1./np.sqrt(float(bks_pfd.shape[0]))
    sub1.fill_between([0, np.sum(klim)], [-1.*inv_V, -1.*inv_V], [inv_V, inv_V], facecolor='k', linewidth=0, alpha=0.2, zorder=100) 

    sub1.legend(plts, lbls, loc='lower left', ncol=6, handletextpad=0.4, fontsize=12) 
    sub1.set_xlabel('triangle configurations', fontsize=25)
    sub1.set_xlim(0, np.sum(klim))
    sub1.set_ylabel('$\Delta_B/\sigma_B$', fontsize=20) 
    sub1.set_ylim(-0.3, 0.3) 
    ffig = os.path.join(dir_doc, 'pf_Bk%s.png' % (_rsd_str(rsd))) 
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') 
    return None 


def pf_DelB_sigB(rsd=0, kmax=0.5):
    ''' comparison of the Delta B0/sigma_B0 distribution for the various cosmologies.
    '''
    fig = plt.figure(figsize=(8, 6))
    sub = fig.add_subplot(111)
    
    # read in covariance matrix 
    i_k, j_k, l_k, Cov_bk, _ = bkCov(rsd=2, flag='reg', silent=False) 
    klim = ((i_k * kf <= kmax) & (j_k * kf <= kmax) & (l_k * kf <= kmax)) # klim 
    sig_bk = np.sqrt(np.diag(Cov_bk))[klim]

    # Delta0/sigma comparison
    thetas = ['fiducial', 'Om_m', 'Om_p', 'Ob2_m', 'Ob2_p', 'h_m', 'h_p', 'ns_m', 'ns_p', 's8_m', 's8_p']
    lbls = ['fid.', r'$\Omega_m^{-}$', r'$\Omega_m^{+}$', r'$\Omega_b^{-}$', r'$\Omega_b^{+}$', r'$h^{-}$', r'$h^{+}$', 
            r'$n_s^{-}$', r'$n_s^{+}$', r'$\sigma_8^{-}$', r'$\sigma_8^{+}$']
    for i, theta, lbl in zip(range(len(thetas)), thetas, lbls): 
        # read in power spectrum (standard)
        quij = Obvs.quijoteBk(theta, rsd=rsd, flag='reg', silent=False) 
        bks_std = quij['b123'][:,klim] # P0 
        # read in bispectrum (pairedfixed)
        quij = Obvs.quijoteBk(theta, rsd=rsd, flag='ncv', silent=False) 
        bks_pfd = quij['b123'][:,klim] # P0 

        delB0_sig = ((bks_pfd - np.average(bks_std, axis=0))/sig_bk).flatten() 
        
        if theta == 'fiducial':  
            sub.hist(delB0_sig, density=True, range=(-3., 3.), bins=30, histtype='step', color='k', label=lbl)
        else: 
            sub.hist(delB0_sig, density=True, range=(-3., 3.), bins=30, histtype='step', color='C%i' % (i-1), label=lbl)
            #sub.plot([np.mean(delB0_sig), np.mean(delB0_sig)], [0., 0.6], c='C%i' % (i-1))
            print(theta, skew(delB0_sig), skewtest(delB0_sig)) 
    sub.plot([0., 0.], [0., 0.6], c='k', ls='--') 
    sub.legend(loc='upper left', ncol=6, handletextpad=0.4, fontsize=12) 
    sub.set_xlabel('$\Delta_{B_0}/\sigma_{B_0}$', fontsize=25) 
    sub.set_xlim(-3, 3)
    sub.set_ylim(0., 0.6) 
    ffig = os.path.join(dir_fig, 'pf_DelB_sigB%s.png' % (_rsd_str(rsd))) 
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def pf_dlogPdtheta(rsd=0): 
    ''' Comparison of dlogP/dtheta from paired-fixed vs standard N-body
    '''
    thetas  = ['Om', 'Ob2', 'h', 'ns', 's8']#, 'Mnu']
    lbls    = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$']#, r'$M_\nu$']
    ylims0  = [(-10., 0.), (0., 15.), (-3., 0.), (-3.5, -0.5), (-1.6, -0.2)]
    ylims1  = [(-10., 0.), (0., 15.), (-3., 0.), (-3.5, -0.5), (-2.5, 1.5)]
    ylims2  = [(-0.75, 0.75), (-1.5, 1.5), (-0.4, 0.4), (-0.3, 0.3), (-0.3, 0.3)]
    
    
    fig = plt.figure(figsize=(30,8))
    gs = mpl.gridspec.GridSpec(3, 1, figure=fig, height_ratios=[2,2,1], hspace=0.05) 
    gs0 = mpl.gridspec.GridSpecFromSubplotSpec(1, len(thetas), subplot_spec=gs[0])
    gs1 = mpl.gridspec.GridSpecFromSubplotSpec(1, len(thetas), subplot_spec=gs[1])
    gs2 = mpl.gridspec.GridSpecFromSubplotSpec(1, len(thetas), subplot_spec=gs[2])

    plt.rc('ytick', labelsize=8)
    for i_tt, tt, lbl in zip(range(len(thetas)), thetas, lbls): 
        sub = plt.subplot(gs0[0, i_tt]) 

        k, dpdt_std = Forecast.quijote_dP02kdtheta(tt, log=True, rsd=rsd, flag='reg', dmnu='fin') 
        _, dpdt_pfd = Forecast.quijote_dP02kdtheta(tt, log=True, rsd=rsd, flag='ncv', dmnu='fin') 
        
        # get std. dev. 
        _, _dpdt_std = Forecast.quijote_dP02kdtheta(tt, log=True, rsd=2, flag='reg', dmnu='fin', average=False) 
        sig_dpdt = np.std(_dpdt_std, axis=0) 
        if i_tt == 0: 
            # get k and klim 
            k0lim = (np.arange(len(k)) < len(k)/2)
            k2lim = (np.arange(len(k)) >= len(k)/2)
        
        sub.plot(k[k0lim], dpdt_std[k0lim], c='C0', label='standard $N$-body')
        sub.plot(k[k0lim], dpdt_pfd[k0lim], c='C1', label='paired-fixed')
        sub.text(0.975, 0.975, lbl, ha='right', va='top', transform=sub.transAxes, fontsize=25)

        sub.set_xlim(9e-3, 0.7) 
        sub.set_xscale('log') 
        sub.set_ylim(ylims0[i_tt]) 
        sub.set_xticklabels([]) 
        if i_tt == 0: sub.set_ylabel(r'${\rm d}\log P_0/{\rm d} \theta$', fontsize=20) 
        if i_tt == len(thetas)-1: sub.legend(loc='lower left', handletextpad=0.2, fontsize=15)
        #sub.text(0.975, 0.925, lbl, ha='right', va='top', transform=sub.transAxes, fontsize=25)
        
        sub = plt.subplot(gs1[0, i_tt]) 

        sub.plot(k[k2lim], dpdt_std[k2lim], c='C0')
        sub.plot(k[k2lim], dpdt_pfd[k2lim], c='C1')

        sub.set_xlim(9e-3, 0.7) 
        sub.set_xscale('log') 
        sub.set_ylim(ylims1[i_tt]) 
        sub.set_xticklabels([]) 
        if i_tt == 0: sub.set_ylabel(r'${\rm d}\log P_2/{\rm d} \theta$', fontsize=20) 
        if i_tt == 0: sub.legend(loc='upper left', fontsize=20)
        #sub.text(0.975, 0.925, lbl, ha='right', va='top', transform=sub.transAxes, fontsize=25)

        sub = plt.subplot(gs2[0, i_tt]) 
        del_dpdt = dpdt_std - dpdt_pfd
        sub.plot(k[k0lim], del_dpdt[k0lim]/sig_dpdt[k0lim], c='C3', label='$\ell=0$')
        sub.plot(k[k2lim], del_dpdt[k2lim]/sig_dpdt[k2lim], c='C7', label='$\ell=2$')
        sub.plot(k[k0lim], np.zeros(np.sum(k0lim)), c='k', ls='--')
        sub.set_xlim(9e-3, 0.7) 
        sub.set_xscale('log') 
        sub.set_ylim(-0.3, 0.3) 
        if i_tt == 0: sub.legend(loc='lower left', ncol=2, handletextpad=0.2, fontsize=15)
        if i_tt == 0: sub.set_ylabel(r'$\Delta_{(\log P_\ell)_{,\theta}}/\sigma_{(\log P_\ell)_{,\theta}}$', fontsize=15) 

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel('$k$', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(hspace=0.075, wspace=0.15) 
    ffig = os.path.join(dir_fig, 'pf_dlogPdtheta%s.png' % (_rsd_str(rsd)))
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def pf_dlogPdtheta_real(): 
    ''' Comparison of dlogP/dtheta from paired-fixed vs standard N-body 
    **in real-space**
    '''
    thetas  = ['Om', 'Ob2', 'h', 'ns', 's8']#, 'Mnu']
    lbls    = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$']#, r'$M_\nu$']
    ylims0  = [(-10., 0.), (0., 15.), (-3., 0.), (-3.5, -0.5), (-1.6, -0.2)]
    ylims1  = [(-0.75, 0.75), (-1.5, 1.5), (-0.1, 0.1), (-0.3, 0.3), (-0.3, 0.3)]
    
    fig = plt.figure(figsize=(30,5))
    gs = mpl.gridspec.GridSpec(2, 1, figure=fig, height_ratios=[2,1], hspace=0.05) 
    gs0 = mpl.gridspec.GridSpecFromSubplotSpec(1, len(thetas), subplot_spec=gs[0])
    gs1 = mpl.gridspec.GridSpecFromSubplotSpec(1, len(thetas), subplot_spec=gs[1])

    plt.rc('ytick', labelsize=8)
    for i_tt, tt, lbl in zip(range(len(thetas)), thetas, lbls): 
        sub = plt.subplot(gs0[0,i_tt])

        k, dpdt_std = Forecast.quijote_dPkdtheta(tt, log=True, rsd='real', flag='reg', dmnu='fin') 
        _, dpdt_pfd = Forecast.quijote_dPkdtheta(tt, log=True, rsd='real', flag='ncv', dmnu='fin') 
        
        _, _dpdt_std = Forecast.quijote_dPkdtheta(tt, log=True, rsd='real', flag='reg', dmnu='fin', average=False) 
        sig_dpdt = np.std(_dpdt_std, axis=0) 

        sub.plot(k, dpdt_std, c='C0', label='standard $N$-body')
        sub.plot(k, dpdt_pfd, c='C1', label='paired-fixed')

        sub.set_xlim(9e-3, 0.7) 
        sub.set_xscale('log') 
        sub.set_ylim(ylims0[i_tt]) 
        sub.set_xticklabels([]) 
        sub.text(0.975, 0.975, lbl, ha='right', va='top', transform=sub.transAxes, fontsize=25)
        if i_tt == 0: sub.set_ylabel(r'${\rm d}\log P/{\rm d} \theta$', fontsize=20) 
        if i_tt == len(thetas)-1: sub.legend(loc='lower left', handletextpad=0.2, fontsize=15)

        sub = plt.subplot(gs1[0,i_tt])
        sub.plot(k, (dpdt_std - dpdt_pfd)/sig_dpdt, c='C3', label='$\ell=0$')
        sub.plot(k, np.zeros(len(k)), c='k', ls='--')
        sub.set_xlim(9e-3, 0.7) 
        sub.set_xscale('log') 
        sub.set_ylim(-0.3, 0.3)# ylims1[i_tt]) 
        if i_tt == 0: sub.set_ylabel(r'$\Delta_{(\log P_\ell)_{,\theta}}/\sigma_{(\log P_\ell)_{,\theta}}$', fontsize=15) 
        #if i_tt == 0: sub.set_ylabel(r'$\Delta {\rm d}\log P/{\rm d} \theta$', fontsize=20) 
    
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel('$k$', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(hspace=0.075, wspace=0.15) 
    ffig = os.path.join(dir_fig, 'pf_dlogPdtheta.real.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def pf_dlogBdtheta(rsd=0, kmax=0.5): 
    ''' Comparison of dlogB/dtheta from paired-fixed vs standard N-body
    '''
    thetas  = ['Om', 'Ob2', 'h', 'ns', 's8']#, 'Mnu']
    lbls    = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$']#, r'$M_\nu$']
    ylims0  = [(-15., 5.), (-2., 24.), (-4., 0.5), (-5., -0.5), (-6., 0)]
    ylims1  = [(-7., 7.), (-25., 25.), (-4., 4), (-3., 3), (-5., 5.)]
    
    # get k1, k2, k3 and klim 
    i_k, j_k, l_k, _ = dpdt = dBkdtheta('Om', log=True, rsd=rsd, flag='reg', dmnu='fin', returnks=True)
    klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) # get k limit
    
    fig = plt.figure(figsize=(30,15))
    for i_tt, tt, lbl in zip(range(len(thetas)), thetas, lbls): 
        sub = fig.add_subplot(len(thetas),2,2*i_tt+1)

        i_k, j_k, l_k, dbdt_std = Forecast.quijote_dBkdtheta(tt, log=True, rsd=rsd, flag='reg', dmnu='fin')
        _, _, _, dbdt_pfd = Forecast.quijote_dBkdtheta(tt, log=True, rsd=rsd, flag='ncv', dmnu='fin')
        
        # get std. dev. 
        _, _, _, _dbdt_std = Forecast.quijote_dBkdtheta(tt, log=True, rsd=2, flag='reg', dmnu='fin', average=False)
        sig_dbdt = np.std(_dbdt_std, axis=0) 

        # get k limit
        if i_tt == 0: klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 

        sub.plot(range(np.sum(klim)), dbdt_std[klim], label='standard $N$-body')
        sub.plot(range(np.sum(klim)), dbdt_pfd[klim], lw=0.95, label='paired-fixed')
        
        sub.set_xlim(0, np.sum(klim)) 
        sub.set_ylim(ylims0[i_tt]) 
        if tt != thetas[-1]: sub.set_xticklabels([]) 
        if i_tt == 2: sub.set_ylabel(r'${\rm d}\log B/{\rm d} \theta$', fontsize=25) 
        if i_tt == 4: sub.set_xlabel('triangle configurations', fontsize=25)
        if i_tt == 0: sub.legend(loc='upper left', fontsize=20)
        #sub.text(0.975, 0.925, lbl, ha='right', va='top', transform=sub.transAxes, fontsize=25)

        sub = fig.add_subplot(len(thetas),2,2*i_tt+2) 
        del_dbdt = dbdt_pfd - dbdt_std
        sub.plot(range(np.sum(klim)), del_dbdt[klim]/sig_dbdt[klim])
        sub.plot(range(np.sum(klim)), np.zeros(np.sum(klim)), c='k', ls='--')
        sub.set_xlim(0, np.sum(klim)) 
        sub.set_ylim(-0.5, 0.5) 
        if tt != thetas[-1]: sub.set_xticklabels([]) 
        if i_tt == 2: sub.set_ylabel(r'$\Delta_{(\log B)_{,\theta}}/\sigma_{(\log B)_{,\theta}}$', fontsize=25) 
        if i_tt == 4: sub.set_xlabel('triangle configurations', fontsize=25)
        sub.text(0.025, 0.925, lbl, ha='left', va='top', transform=sub.transAxes, fontsize=25)

    fig.subplots_adjust(hspace=0.075) 
    ffig = os.path.join(dir_fig, 'pf_dlogBdtheta%s.kmax%.1f.png' % (_rsd_str(rsd), kmax))
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def pf_P_Fij(rsd=0, kmax=0.5): 
    ''' compare the fisher matrix derived from paired-fixed derivatives vs 
    standard n-body derivatives
    '''
    # read in covariance matrix 
    if rsd != 'real': 
        k, Cov_pk, nmock = p02kCov(rsd=2, flag='reg', silent=False) 
    else: 
        k, Cov_pk, nmock = pkCov(rsd='real', flag='reg', silent=False) 
    klim = (k < kmax) # klim 
    Cov_pk = Cov_pk[:,klim][klim,:]
    
    # get precision matrix 
    ndata = np.sum(klim) 
    f_hartlap = float(nmock - ndata - 2)/float(nmock - 1) 
    C_inv = f_hartlap * np.linalg.inv(Cov_pk) # invert the covariance 
    
    thetas      = ['Om', 'Ob2', 'h', 'ns', 's8', 'Mnu']
    theta_fid   = [0.3175, 0.049, 0.6711, 0.9624, 0.834, 0.0] # fiducial theta 
    theta_lims  = [(0.28, 0.355), (0.028, 0.07), (0.4922, 0.85), (0.7748, 1.15), (0.808, 0.86), (-0.25, 0.25)]
    lbls        = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$']

    dpdts_std, dpdts_pfd = [], [] 
    for theta in thetas:
        if rsd != 'real': 
            dpdt_std = dP02kdtheta(theta, log=False, rsd=rsd, flag='reg', dmnu='fin')
            dpdt_pfd = dP02kdtheta(theta, log=False, rsd=rsd, flag='ncv', dmnu='fin')
        else: 
            dpdt_std = dPkdtheta(theta, log=False, rsd=rsd, flag='reg', dmnu='fin')
            dpdt_pfd = dPkdtheta(theta, log=False, rsd=rsd, flag='ncv', dmnu='fin')
        dpdts_std.append(dpdt_std[klim]) 
        dpdts_pfd.append(dpdt_pfd[klim]) 

    Fij_std = Forecast.Fij(dpdts_std, C_inv) 
    Fij_pfd = Forecast.Fij(dpdts_pfd, C_inv) 

    fig = plt.figure(figsize=(5,5))
    sub = fig.add_subplot(111) 
    cm = sub.pcolormesh(Fij_pfd/Fij_std - 1, vmin=-0.1, vmax=0.1, cmap='RdBu')
    cbar_ax = fig.add_axes([0.95, 0.15, 0.0125, 0.7])
    cbar = fig.colorbar(cm, cax=cbar_ax)
    sub.set_xticks(np.arange(len(thetas))+0.5)
    sub.set_xticklabels(lbls) 
    sub.set_yticks(np.arange(len(thetas))+0.5)
    sub.set_yticklabels(lbls) 
    sub.set_title(r'$(F^{\rm PF}_{ij} - F_{ij})/ F_{ij}$', fontsize=20) 
    ffig = os.path.join(dir_fig, 'pf_P_Fij%s.kmax%.1f.png' % (_rsd_str(rsd), kmax))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def pf_B_Fij(rsd=0, kmax=0.5): 
    ''' compare the fisher matrix derived from paired-fixed derivatives vs 
    standard n-body derivatives
    '''
    # read in covariance matrix 
    if rsd != 'real':
        i_k, j_k, l_k, Cov_bk, nmock = bkCov(rsd=0, flag='reg', silent=False) 
    else: 
        i_k, j_k, l_k, Cov_bk, nmock = bkCov(rsd='real', flag='reg', silent=False) 
    klim = ((i_k * kf <= kmax) & (j_k * kf <= kmax) & (l_k * kf <= kmax)) # klim 
    Cov_bk = Cov_bk[:,klim][klim,:]
    
    # get precision matrix 
    ndata = np.sum(klim) 
    f_hartlap = float(nmock - ndata - 2)/float(nmock - 1) 
    C_inv = f_hartlap * np.linalg.inv(Cov_bk) # invert the covariance 
    
    thetas  = ['Om', 'Ob2', 'h', 'ns', 's8']
    lbls    = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$']
    dbdts_std, dbdts_pfd = [], [] 
    for theta in thetas:
        dbdt_std = dBkdtheta(theta, log=False, rsd=rsd, flag='reg', dmnu='fin')
        dbdt_pfd = dBkdtheta(theta, log=False, rsd=rsd, flag='ncv', dmnu='fin')
        dbdts_std.append(dbdt_std[klim]) 
        dbdts_pfd.append(dbdt_pfd[klim]) 

    Fij_std = Forecast.Fij(dbdts_std, C_inv) 
    Fij_pfd = Forecast.Fij(dbdts_pfd, C_inv) 

    fig = plt.figure(figsize=(5,5))
    sub = fig.add_subplot(111) 
    cm = sub.pcolormesh(Fij_pfd/Fij_std - 1, vmin=-0.1, vmax=0.1, cmap='RdBu')
    cbar_ax = fig.add_axes([0.95, 0.15, 0.0125, 0.7])
    cbar = fig.colorbar(cm, cax=cbar_ax)
    sub.set_xticks(np.arange(len(thetas))+0.5)
    sub.set_xticklabels(lbls) 
    sub.set_yticks(np.arange(len(thetas))+0.5)
    sub.set_yticklabels(lbls) 
    sub.set_title(r'$(F^{\rm PF}_{ij} - F_{ij})/ F_{ij}$', fontsize=20) 
    ffig = os.path.join(dir_fig, 'pf_B_Fij%s.kmax%.1f.png' % (_rsd_str(rsd), kmax))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def pf_P_posterior(rsd=0, kmax=0.5): 
    ''' compare the fisher matrix derived from paired-fixed derivatives vs 
    standard n-body derivatives
    '''
    # read in covariance matrix 
    if rsd != 'real': 
        k, Cov_pk, nmock = p02kCov(rsd=2, flag='reg', silent=False) 
    else: 
        k, Cov_pk, nmock = pkCov(rsd='real', flag='reg', silent=False) 
    klim = (k < kmax) # klim 
    Cov_pk = Cov_pk[:,klim][klim,:]
    
    # get precision matrix 
    ndata = np.sum(klim) 
    f_hartlap = float(nmock - ndata - 2)/float(nmock - 1) 
    C_inv = f_hartlap * np.linalg.inv(Cov_pk) # invert the covariance 
    
    thetas      = ['Om', 'Ob2', 'h', 'ns', 's8', 'Mnu']
    theta_fid   = [0.3175, 0.049, 0.6711, 0.9624, 0.834, 0.0] # fiducial theta 
    theta_lims  = [(0.28, 0.355), (0.028, 0.07), (0.4922, 0.85), (0.7748, 1.15), (0.808, 0.86), (-0.25, 0.25)]
    lbls        = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$']

    dpdts_std, dpdts_pfd = [], [] 
    for theta in thetas:
        if rsd != 'real': 
            dpdt_std = dP02kdtheta(theta, log=False, rsd=rsd, flag='reg', dmnu='fin')
            dpdt_pfd = dP02kdtheta(theta, log=False, rsd=rsd, flag='ncv', dmnu='fin')
        else: 
            dpdt_std = dPkdtheta(theta, log=False, rsd=rsd, flag='reg', dmnu='fin')
            dpdt_pfd = dPkdtheta(theta, log=False, rsd=rsd, flag='ncv', dmnu='fin')
        dpdts_std.append(dpdt_std[klim]) 
        dpdts_pfd.append(dpdt_pfd[klim]) 

    Fij_std = Forecast.Fij(dpdts_std, C_inv) 
    Fij_pfd = Forecast.Fij(dpdts_pfd, C_inv) 
    
    Finv_std = np.linalg.inv(Fij_std) # invert fisher matrix 
    Finv_pfd = np.linalg.inv(Fij_pfd) # invert fisher matrix 
    
    fig = Forecast.plotFisher([Finv_pfd, Finv_std], theta_fid, ranges=theta_lims, colors=['C1', 'C0'], labels=lbls)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.fill_between([],[],[], color='C0', label=r'standard $N$-body') 
    bkgd.fill_between([],[],[], color='C1', label=r'paired fixed') 
    bkgd.legend(loc='upper right', bbox_to_anchor=(0.8, 0.8), handletextpad=0.5, fontsize=20)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    fig.subplots_adjust(wspace=0.05, hspace=0.05) 
    ffig = os.path.join(dir_fig, 'pf_P_posterior%s.kmax%.1f.png' % (_rsd_str(rsd), kmax))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def pf_B_posterior(rsd=0, kmax=0.5): 
    ''' compare the fisher matrix derived from paired-fixed derivatives vs 
    standard n-body derivatives
    '''
    # read in covariance matrix 
    if rsd != 'real': 
        i_k, j_k, l_k, Cov_bk, nmock = bkCov(rsd=0, flag='reg', silent=False) 
    else: 
        i_k, j_k, l_k, Cov_bk, nmock = bkCov(rsd='real', flag='reg', silent=False) 
    klim = ((i_k * kf <= kmax) & (j_k * kf <= kmax) & (l_k * kf <= kmax)) # klim 
    Cov_bk = Cov_bk[:,klim][klim,:]
    
    # get precision matrix 
    ndata = np.sum(klim) 
    f_hartlap = float(nmock - ndata - 2)/float(nmock - 1) 
    C_inv = f_hartlap * np.linalg.inv(Cov_bk) # invert the covariance 
    
    thetas      = ['Om', 'Ob2', 'h', 'ns', 's8', 'Mnu']
    theta_fid   = [0.3175, 0.049, 0.6711, 0.9624, 0.834, 0.0] # fiducial theta 
    theta_lims  = [(0.3, 0.335), (0.038, 0.06), (0.5922, 0.75), (0.8748, 1.05), (0.808, 0.86), (-0.125, 0.125)]

    lbls        = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$']
    dbdts_std, dbdts_pfd = [], [] 
    for theta in thetas:
        dbdt_std = dBkdtheta(theta, log=False, rsd=rsd, flag='reg', dmnu='fin')
        dbdt_pfd = dBkdtheta(theta, log=False, rsd=rsd, flag='ncv', dmnu='fin')
        dbdts_std.append(dbdt_std[klim]) 
        dbdts_pfd.append(dbdt_pfd[klim]) 

    Fij_std = Forecast.Fij(dbdts_std, C_inv) 
    Fij_pfd = Forecast.Fij(dbdts_pfd, C_inv) 
    
    Finv_std = np.linalg.inv(Fij_std) # invert fisher matrix 
    Finv_pfd = np.linalg.inv(Fij_pfd) # invert fisher matrix 
    
    fig = Forecast.plotFisher([Finv_pfd, Finv_std], theta_fid, ranges=theta_lims, colors=['C1', 'C0'], labels=lbls)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.fill_between([],[],[], color='C0', label=r'standard $N$-body') 
    bkgd.fill_between([],[],[], color='C1', label=r'paired fixed') 
    bkgd.legend(loc='upper right', bbox_to_anchor=(0.8, 0.8), handletextpad=0.5, fontsize=20)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    fig.subplots_adjust(wspace=0.05, hspace=0.05) 
    ffig = os.path.join(dir_fig, 'pf_B_posterior%s.kmax%.1f.png' % (_rsd_str(rsd), kmax))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def PF_PkBk(rsd=0, kmax=0.3): 
    '''
    '''
    # read in fiducial P regular N-body 
    quij_fid = Obvs.quijotePk('fiducial', rsd=rsd, flag='reg') 
    k = quij_fid['k'] 
    # average P 
    pk_fid = np.average(quij_fid['p0k'], axis=0) 
    # standard deviation 
    _, Cov_pk, _ = pkCov(rsd=rsd, flag='reg', silent=True) 
    sig_pk = np.sqrt(np.diag(Cov_pk))
    # read in fiducial P paired fixed (1500 mocks) 
    quij_ncv = Obvs.quijotePk('fiducial', rsd=rsd, flag='ncv', silent=False) 
    pks_ncv  = quij_ncv['p0k']

    # read in fiducial B regular N-body (15000 mocks) 
    quij_fid = Obvs.quijoteBk('fiducial', rsd=rsd, flag='reg', silent=False) 
    i_k, j_k, l_k = quij_fid['k1'], quij_fid['k2'], quij_fid['k3']
    klim = ((i_k * kf <= kmax) & (j_k * kf <= kmax) & (l_k * kf <= kmax)) 
    # average B
    bk_fid  = np.average(quij_fid['b123'], axis=0) 
    # standard deviation 
    _, _, _, Cov_bk, _ = bkCov(rsd=rsd, flag='reg', silent=True)
    sig_fid = np.sqrt(np.diag(Cov_bk)) 

    # read in fiducial B paired fixed (1500 mocks) 
    quij_ncv = Obvs.quijoteBk('fiducial', rsd=rsd, flag='ncv', silent=False) 
    bks_ncv  = quij_ncv['b123']
    
    fig = plt.figure(figsize=(50,8)) 
    gs = mpl.gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1,4], wspace=0.2) 
    gs0 = mpl.gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0])
    gs1 = mpl.gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1])
    sub = plt.subplot(gs0[0,0])
    sub.plot(k, pk_fid, c='k', ls='--')
    sub.fill_between(k, pk_fid - sig_pk, pk_fid + sig_pk, color='k', linewidth=0, alpha=0.2) 
    sub.plot(k, pks_ncv[0,:], lw=0.5, c='C0') 
    sub.plot(k, pks_ncv[1,:], lw=0.5, c='C1') 
    sub.scatter(k, 0.5*(pks_ncv[0,:] + pks_ncv[1,:]), c='r', s=15, lw=0, zorder=10) 
    sub.plot(k, 0.5*(pks_ncv[0,:] + pks_ncv[1,:]), c='r', lw=0.5) 
    sub.set_xlim(9e-3, kmax) 
    sub.set_xscale('log') 
    sub.set_xticklabels([]) 
    sub.set_ylabel('$P_0(k)$', fontsize=25) 
    sub.set_yscale('log') 
    sub.set_ylim(1e3, 2e5) 

    sub = plt.subplot(gs1[0,0])
    sub.plot(range(np.sum(klim)), bk_fid[klim], c='k', ls='--', label='$N$-body') 
    sub.fill_between(range(np.sum(klim)), bk_fid[klim] - sig_fid[klim], bk_fid[klim] + sig_fid[klim], 
            color='k', linewidth=0, alpha=0.2) 
    sub.plot(range(np.sum(klim)), bks_ncv[0,klim], lw=1, c='C0', label='paired-fixed $i^+$') 
    sub.plot(range(np.sum(klim)), bks_ncv[1,klim], lw=1, c='C1', label='paired-fixed $i^-$') 
    sub.scatter(range(np.sum(klim)), 0.5*(bks_ncv[0,klim] + bks_ncv[1,klim]), c='r', s=15, lw=0, zorder=10) 
    sub.plot(range(np.sum(klim)), 0.5*(bks_ncv[0,klim] + bks_ncv[1,klim]), c='r', lw=1)
    sub.legend(loc='upper right', fontsize=20) 
    sub.set_xlim(0, np.sum(klim))
    sub.set_xticklabels([]) 
    sub.set_ylabel('$B(k_0, k_2, k_3)$', fontsize=25)
    sub.set_yscale('log') 
    if kmax == 0.2: 
        sub.set_ylim(2e7, 1e10)
    elif kmax == 0.3: 
        sub.set_ylim(1e7, 1e10)
    elif kmax == 0.5: 
        sub.set_ylim(1e5, 1e10)
    
    sub = plt.subplot(gs0[1,0])
    #sub.fill_between(k, 1. - sig_pk/pk_fid, 1. + sig_pk/pk_fid, color='k', linewidth=0, alpha=0.33) 
    sub.fill_between([k[0], k[-1]], [-1., -1.], [1., 1.], color='k', linewidth=0, alpha=0.2) 
    sub.plot(k, (pks_ncv[0,:]-pk_fid)/sig_pk, lw=1, c='C0') 
    sub.plot(k, (pks_ncv[1,:]-pk_fid)/sig_pk, lw=1, c='C1') 
    sub.plot(k, (0.5*(pks_ncv[0,:] + pks_ncv[1,:]) - pk_fid)/sig_pk, c='r', lw=0.5) 
    sub.scatter(k, (0.5*(pks_ncv[0,:] + pks_ncv[1,:]) - pk_fid)/sig_pk, c='r', s=10, lw=0, zorder=10) 
    sub.plot([k[0], k[-1]], [0., 0.], c='k', ls=':')
    sub.set_xlabel('k', fontsize=25) 
    sub.set_xlim(9e-3, kmax) 
    sub.set_xscale('log') 
    sub.set_ylabel(r'$\Delta_{P_0}/\sigma_{P_0}$', fontsize=25) 
    sub.set_ylim(-2.5, 2.5) 

    sub = plt.subplot(gs1[1,0])
    #sub.fill_between(range(np.sum(klim)), ((bk_fid - sig_fid)/bk_fid)[klim], ((bk_fid + sig_fid)/bk_fid)[klim], 
    #        color='k', linewidth=0, alpha=0.33) 
    sub.fill_between([0., np.sum(klim)], [-1., -1.], [1., 1.], color='k', linewidth=0, alpha=0.2) 
    sub.plot(range(np.sum(klim)), ((bks_ncv[0,:]-bk_fid)/sig_fid)[klim], lw=1, c='C0') 
    sub.plot(range(np.sum(klim)), ((bks_ncv[1,:]-bk_fid)/sig_fid)[klim], lw=1, c='C1')
    sub.plot([0., np.sum(klim)], [0., 0.], c='k', ls=':') 
    sub.plot(range(np.sum(klim)), ((0.5*(bks_ncv[0] + bks_ncv[1]) - bk_fid)/sig_fid)[klim], c='r', lw=0.5) 
    sub.scatter(range(np.sum(klim)), ((0.5*(bks_ncv[0] + bks_ncv[1]) - bk_fid)/sig_fid)[klim], c='r', s=10, lw=0, zorder=10) 
    sub.set_xlabel('triangle configuration', fontsize=25) 
    sub.set_xlim(0, np.sum(klim))
    sub.set_ylabel(r'$\Delta_{B_0}/\sigma_{B_0}$', fontsize=25) 
    sub.set_ylim(-2.5, 2.5)
    fig.subplots_adjust(hspace=0.1) 
    ffig = os.path.join(dir_fig, 'PkBk%s.kmax%.1f.png' % (_rsd_str(rsd), kmax)) 
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def PF_deltaB_sigmaB(rsd=0, kmax=0.3, nbin=31): 
    ''' examine the triangles where the average bispectrum of the paired fixed sims
    deviates by more than 1sigma from the Nbody sim
    '''
    # read in fiducial B regular N-body (15000 mocks) 
    quij_fid = Obvs.quijoteBk('fiducial', rsd=rsd, flag='reg', silent=False) 
    i_k, j_k, l_k = quij_fid['k1'], quij_fid['k2'], quij_fid['k3']
    klim = ((i_k * kf <= kmax) & (j_k * kf <= kmax) & (l_k * kf <= kmax)) 
    # average B
    bk_fid = np.average(quij_fid['b123'], axis=0) 
    # standard deviation 
    _, _, _, Cov_bk, _ = bkCov(rsd=rsd, flag='reg', silent=True)
    sig_fid = np.sqrt(np.diag(Cov_bk)) 

    # read in fiducial B paired fixed (1500 mocks) 
    quij_ncv = Obvs.quijoteBk('fiducial', rsd=rsd, flag='ncv', silent=False) 
    bk_ncv = np.average(quij_ncv['b123'], axis=0) 
    
    delB_sig = (bk_ncv[klim] - bk_fid[klim])/sig_fid[klim]

    fig = plt.figure(figsize=(30,4)) 
    gs = mpl.gridspec.GridSpec(1, 2, figure=fig, width_ratios=[3,1], wspace=0.15) 
    gs0 = mpl.gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0])
    gs1 = mpl.gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[1])
    sub = plt.subplot(gs0[0,0]) 
    sub.plot(range(np.sum(klim)), delB_sig, c='C0', ls='-')
    sub.plot([0., np.sum(klim)], [0, 0], c='k', ls=':')
    sub.legend(loc='upper right', fontsize=20) 
    sub.set_xlabel('triangle configuration', fontsize=25) 
    sub.set_xlim(0, np.sum(klim))
    sub.set_ylabel('$\Delta_{B_0}/\sigma_{B_0}$', fontsize=25)
    sub.set_ylim(-0.2, 0.2) 
    # shape 
    sub = plt.subplot(gs1[0,0]) 
    x_bins = np.linspace(0., 1., int(nbin)+1)
    y_bins = np.linspace(0.5, 1., int(0.5*nbin)+1)
    k3k1 = l_k.astype(float)/i_k.astype(float)
    k2k1 = j_k.astype(float)/i_k.astype(float)
    dBQgrid = ePlots._BorQgrid(k3k1[klim], k2k1[klim], delB_sig, np.ones(np.sum(klim)), x_bins, y_bins)
    bplot = sub.pcolormesh(x_bins, y_bins, dBQgrid.T, vmin=-0.05, vmax=0.05, cmap='RdBu')
    sub.set_xlabel('$k_3/k_1$', fontsize=25)
    sub.set_ylabel('$k_2/k_1$', fontsize=25)
    fig.subplots_adjust(wspace=0.15, hspace=0.2, right=0.99)
    cbar_ax = fig.add_axes([0.995, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(bplot, cax=cbar_ax)
    cbar.set_label('$\Delta_{B_0}/\sigma_{B_0}$', labelpad=10, rotation=90, fontsize=20)
    ffig = os.path.join(dir_fig, 'deltaB_sigmaB%s.kmax%.1f.png' % (_rsd_str(rsd), kmax)) 
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def pairedfixed_theta(theta, kmax=0.5): 
    ''' compare the bispectrum of the fiducial and fiducial pair-fixed 
    '''
    # read in fiducial B regular N-body (45000 mocks) 
    quij_fid = Obvs.quijoteBk(theta, rsd='all', flag='reg', silent=False) 
    print(quij_fid['b123'].shape) 
    bk_fid = np.average(quij_fid['b123'], axis=0) 
    # read in fiducial B paired fixed (1500 mocks) 
    quij_fid_ncv = Obvs.quijoteBk(theta, rsd='all', flag='ncv', silent=False) 
    print(quij_fid_ncv['b123'].shape) 
    bk_fid_ncv = np.average(quij_fid_ncv['b123'], axis=0) 
    
    # klimit  
    i_k, j_k, l_k = quij_fid['k1'], quij_fid['k2'], quij_fid['k3']
    klim = ((i_k * kf <= kmax) & (j_k * kf <= kmax) & (l_k * kf <= kmax)) 
    ijl = UT.ijl_order(i_k[klim], j_k[klim], l_k[klim], typ='GM') # order of triangles 

    fig = plt.figure(figsize=(20,10)) 
    sub = fig.add_subplot(211)
    #sub.plot(range(np.sum(klim)), bk_fid[klim][ijl], c='k', label='fiducial') 
    #sub.plot(range(np.sum(klim)), bk_fid_ncv[klim][ijl], c='C1', ls="--", label='fiducial NCV') 
    sub.plot(range(np.sum(klim)), bk_fid[klim], c='k', label='fiducial') 
    sub.plot(range(np.sum(klim)), bk_fid_ncv[klim], c='C1', ls="--", label='fiducial NCV') 
    sub.legend(loc='upper right', fontsize=20) 
    sub.set_xlim(0, np.sum(klim))
    sub.set_ylabel('$B(k)$', fontsize=25)
    sub.set_yscale('log') 
    sub.set_ylim(1e5, 1e10)

    sub = fig.add_subplot(212)
    #sub.plot(range(np.sum(klim)), bk_fid_ncv[klim][ijl]/bk_fid[klim][ijl], c='C0') 
    sub.plot(range(np.sum(klim)), bk_fid_ncv[klim]/bk_fid[klim], c='C0') 
    sub.plot(range(np.sum(klim)), np.ones(np.sum(klim)), c='k', ls=':') 
    sub.legend(loc='upper right', fontsize=20) 
    sub.set_xlabel('triangle configuration', fontsize=25) 
    sub.set_xlim(0, np.sum(klim))
    sub.set_ylabel(r'$B^{\rm fid;NCV}/B^{\rm fid}$', fontsize=25)
    sub.set_ylim(0.9, 1.1)

    ffig = os.path.join(dir_fig, 'quijote_%s_pairedfixed.kmax%.2f.png' % (theta, kmax)) 
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def pairedfixed_allthetas(): 
    ''' compare the bispectrum of the fiducial and fiducial pair-fixed 
    '''
    fig = plt.figure(figsize=(20,15)) 
    _sub = fig.add_subplot(311)
    sub = fig.add_subplot(312)

    ratios = [] 
    for theta in ['fiducial', 'Om_m', 'Om_p', 'Ob2_m', 'Ob2_p', 'h_m', 'h_p', 'ns_m', 'ns_p', 's8_m', 's8_p', 
            'Mnu_p', 'Mnu_pp', 'Mnu_ppp']: 
        # read in fiducial B regular N-body (45000 mocks) 
        quij_fid = Obvs.quijoteBk(theta, rsd='all', flag='reg', silent=False) 
        bk_fid = np.average(quij_fid['b123'], axis=0) 
        # read in fiducial B paired fixed (1500 mocks) 
        quij_fid_ncv = Obvs.quijoteBk(theta, rsd='all', flag='ncv', silent=False) 
        bk_fid_ncv = np.average(quij_fid_ncv['b123'], axis=0) 
        ratios.append(bk_fid_ncv/bk_fid) 
        # klimit  
        i_k, j_k, l_k = quij_fid['k1'], quij_fid['k2'], quij_fid['k3']
        klim = ((i_k * kf <= 0.5) & (j_k * kf <= 0.5) & (l_k * kf <= 0.5)) 
        _sub.plot(range(np.sum(klim)), bk_fid[klim], c='k', lw=0.1) 
        sub.plot(range(np.sum(klim)), (bk_fid_ncv/bk_fid)[klim], lw=0.1) 
    ratios = np.array(ratios) 

    klim = ((i_k * kf <= 0.5) & (j_k * kf <= 0.5) & (l_k * kf <= 0.5)) 
    _sub.set_xlim(0, np.sum(klim))
    _sub.set_xticklabels([]) 
    _sub.set_ylabel('$B(k)$', fontsize=25)
    _sub.set_yscale('log') 
    _sub.set_ylim(1e5, 1e10)

    sub.plot(range(np.sum(klim)), np.ones(np.sum(klim)), c='k', ls=':', zorder=10) 
    sub.text(0.025, 0.95, 'all cosmologies', ha='left', va='top', transform=sub.transAxes, fontsize=25)
    sub.set_xlim(0, np.sum(klim))
    sub.set_xticklabels([]) 
    sub.set_ylabel(r'$B^{\rm fid;NCV}/B^{\rm fid}$', fontsize=25)
    sub.set_ylim(0.9, 1.1)

    sub = fig.add_subplot(313)
    #for kmax in [0.5, 0.4, 0.3, 0.2, 0.1]: 
    kmax = 0.5
    klim = ((i_k * kf <= kmax) & (j_k * kf <= kmax) & (l_k * kf <= kmax)) 
    sub.plot(range(np.sum(klim)), np.average(ratios, axis=0)[klim])#, label=r'$k_{\rm max} = %.1f$' % kmax) 
    klim = ((i_k * kf <= 0.5) & (j_k * kf <= 0.5) & (l_k * kf <= 0.5)) 
    sub.plot(range(np.sum(klim)), np.ones(np.sum(klim)), c='k', ls=':', zorder=10) 
    sub.legend(loc='lower left', ncol=3, fontsize=20) 
    sub.text(0.025, 0.95, 'average', ha='left', va='top', transform=sub.transAxes, fontsize=25)
    sub.set_xlabel('triangle configuration', fontsize=25) 
    sub.set_xlim(0, np.sum(klim))
    sub.set_ylabel(r'$B^{\rm fid;NCV}/B^{\rm fid}$', fontsize=25)
    sub.set_ylim(0.9, 1.1)
    fig.subplots_adjust(hspace=0.075) 
    ffig = os.path.join(dir_fig, 'quijote_allthetas_pairedfixed.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


############################################################
# etc 
############################################################
def X_std(obs, theta, rsd=2, silent=True): 
    ''' read in standard N-body observables 
    '''
    if obs == 'pk': 
        quij = Obvs.quijotePk(theta, rsd=rsd, flag='reg', silent=silent) 
        if rsd != 'real': # redshift-space 
            return quij['k'], quij['p0k'], quij['p2k']
        else: # real-space
            return quij['k'], quij['p0k']
    elif obs == 'bk': 
        pass
    else: raise ValueError


def X_pfd_1(obs, theta, rsd=2, silent=True): 
    ''' read in observables of the first of the 
    paired-fixed pairs
    '''
    if obs == 'pk': 
        quij = Obvs.quijotePk(theta, rsd=rsd, flag='ncv', silent=silent) 
        if rsd != 'real': # redshift-space 
            return quij['k'], quij['p0k'][::2,:], quij['p2k'][::2,:]
        else: # real-space
            return quij['k'], quij['p0k'][::2,:]
    elif obs == 'bk': 
        pass
    else: raise ValueError


def X_pfd_2(obs, theta, rsd=2, silent=True): 
    ''' read in observables of the first of the 
    paired-fixed pairs
    '''
    if obs == 'pk': 
        quij = Obvs.quijotePk(theta, rsd=rsd, flag='ncv', silent=silent) 
        if rsd != 'real': # redshift-space 
            return quij['k'], quij['p0k'][1::2,:], quij['p2k'][1::2,:]
        else: # real-space
            return quij['k'], quij['p0k'][1::2,:]
    elif obs == 'bk': 
        pass
    else: raise ValueError


def pf_bias(X_std, X_pfd): 
    ''' calculate bias using Eq.(13) of Villaesucsa et al. (2018) 
    '''
    avgX_std = np.average(X_std, axis=0) 
    avgX_pfd = np.average(X_pfd, axis=0) 
    
    N_std = float(X_std.shape[0])  
    N_pfd = float(X_pfd.shape[0]) 

    sig_std = np.std(X_std, axis=0) 
    sig_pfd = np.std(X_pfd, axis=0) 

    sig_spf = np.sqrt(sig_std**2/N_std + sig_pfd**2/N_pfd)
    
    return (avgX_std - avgX_pfd) / sig_spf


def pkCov(rsd=2, flag='reg', silent=True): 
    ''' calculate the covariance matrix of the RSD quijote powerspectrum. 
    '''
    assert flag == 'reg', "only n-body should be used for covariance"
    assert rsd != 'all', "only one RSD direction should be used otherwise modes will be correlated" 

    fcov = os.path.join(dir_bk, 'quijote_pCov_full%s%s.hdf5' % (_rsd_str(rsd), _flag_str(flag)))
    if os.path.isfile(fcov): 
        if not silent: print('reading ... %s' % os.path.basename(fcov))
        Fcov = h5py.File(fcov, 'r') # read in fiducial covariance matrix 
        C_pk = Fcov['C_pk'][...]
        k = Fcov['k'][...]
        Nmock = Fcov['Nmock'][...]
    else:
        if not silent: print('calculating ... %s' % os.path.basename(fcov))
        # read in P(k) 
        quij = Obvs.quijotePk('fiducial', rsd=rsd, flag=flag, silent=silent) 
        pks = quij['p0k'] + quij['p_sn'][:,None] # shotnoise uncorrected P(k) 
        C_pk = np.cov(pks.T) # covariance matrix 
        k = quij['k'] 
        Nmock = quij['p0k'].shape[0] 

        f = h5py.File(fcov, 'w') # write to hdf5 file 
        f.create_dataset('Nmock', data=Nmock)
        f.create_dataset('C_pk', data=C_pk) 
        f.create_dataset('k', data=k) 
        f.close()
    return k, C_pk, Nmock 


def p02kCov(rsd=2, flag='reg', silent=True): 
    ''' calculate the covariance matrix of the RSD quijote power spectrum
    monopole *and* quadrupole. 
    '''
    assert flag == 'reg', "only n-body should be used for covariance"
    assert rsd != 'all', "only one RSD direction should be used otherwise modes will be correlated" 

    fcov = os.path.join(dir_bk, 'quijote_p02Cov_full%s%s.hdf5' % (_rsd_str(rsd), _flag_str(flag)))
    if os.path.isfile(fcov): 
        if not silent: print('reading ... %s' % os.path.basename(fcov))
        Fcov = h5py.File(fcov, 'r') # read in fiducial covariance matrix 
        C_pk = Fcov['C_pk'][...]
        k = Fcov['k'][...]
        Nmock = Fcov['Nmock'][...]
    else:
        if not silent: print('calculating ... %s' % os.path.basename(fcov))
        # read in P(k) 
        quij = Obvs.quijotePk('fiducial', rsd=rsd, flag=flag, silent=silent) 
        p0ks = quij['p0k'] + quij['p_sn'][:,None] # shotnoise uncorrected P0 
        p2ks = quij['p2k'] # P2 
        pks = np.concatenate([p0ks, p2ks], axis=1)  # concatenate P0 and P2
        C_pk = np.cov(pks.T) # covariance matrix 
        k = np.concatenate([quij['k'], quij['k']]) 
        Nmock = quij['p0k'].shape[0] 

        f = h5py.File(fcov, 'w') # write to hdf5 file 
        f.create_dataset('Nmock', data=Nmock)
        f.create_dataset('C_pk', data=C_pk) 
        f.create_dataset('k', data=k) 
        f.close()
    return k, C_pk, Nmock 


def bkCov(rsd=0, flag='reg', silent=True): 
    ''' return the full covariance matrix of the quijote bispectrum
    computed using 
    - rsd == 'all': 46500 simulations (15000 n-body x 3  + 500 ncv x 3) 
    - rsd == 0: 15500 simulations (15000 n-body + 500 ncv) 
    - rsd == 'real': 15500 simulations (15000 n-body + 500 ncv)
    
    at the fiducial parameter values. 

    :return cov:
        big ass covariance matrix of all the triangle configurations in 
        the default ordering. 
    '''
    assert flag == 'reg', "only n-body should be used for covariance"
    assert rsd != 'all', "only one RSD direction should be used otherwise modes will be correlated" 

    fcov = os.path.join(dir_bk, 'quijote_bCov_full%s%s.hdf5' % (_rsd_str(rsd), _flag_str(flag)))
    if os.path.isfile(fcov): 
        if not silent: print('reading ... %s' % os.path.basename(fcov))
        Fcov = h5py.File(fcov, 'r') # read in covariance matrix 
        cov = Fcov['C_bk'][...]
        k1, k2, k3 = Fcov['k1'][...], Fcov['k2'][...], Fcov['k3'][...]
        Nmock = Fcov['Nmock'][...] 
    else: 
        if not silent: print('calculating ... %s' % os.path.basename(fcov))
        quij = Obvs.quijoteBk('fiducial', rsd=rsd, flag=flag, silent=silent) 
        bks = quij['b123'] + quij['b_sn']
        if not silent: print('%i Bk measurements' % bks.shape[0]) 
        cov = np.cov(bks.T) # calculate the covariance
        k1, k2, k3 = quij['k1'], quij['k2'], quij['k3']
        Nmock = quij['b123'].shape[0]

        f = h5py.File(fcov, 'w') # write to hdf5 file 
        f.create_dataset('Nmock', data=Nmock)
        f.create_dataset('C_bk', data=cov) 
        f.create_dataset('k1', data=quij['k1']) 
        f.create_dataset('k2', data=quij['k2']) 
        f.create_dataset('k3', data=quij['k3']) 
        f.close()
    return k1, k2, k3, cov, Nmock


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


def dP02kdtheta(theta, log=False, rsd='all', flag='reg', dmnu='fin', returnks=False, silent=True):
    ''' read in derivatives d P(k)/d theta  

    :param theta: 
        string that specifies the parameter to take the 
        derivative by. 
    '''
    dir_dat = os.path.join(UT.doc_dir(), 'dat') 
    fdpk = os.path.join(dir_dat, 'dP02dtheta.%s%s%s.dat' % (theta, _rsd_str(rsd), _flag_str(flag))) 
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



if __name__=="__main__": 
    pf_Pk()
    #pf_Pk_real()
    #pf_Bk(rsd='real', kmax=0.5)
    #pf_Bk(rsd='all', kmax=0.5)
    #pf_DelB_sigB(rsd='all', kmax=0.5)
    pf_dlogPdtheta(rsd='all')
    pf_dlogPdtheta_real()
    pf_dlogBdtheta(rsd='real', kmax=0.5)
    pf_dlogBdtheta(rsd='all', kmax=0.5)
    #pf_P_Fij(rsd='real', kmax=0.5)
    #pf_P_Fij(rsd='all', kmax=0.5)
    #pf_B_Fij(rsd='real', kmax=0.5)
    #pf_B_Fij(rsd='all', kmax=0.5)
    #pf_P_posterior(rsd='real', kmax=0.5)
    #pf_P_posterior(rsd='all', kmax=0.5)
    #pf_B_posterior(rsd='real', kmax=0.5)
    #pf_B_posterior(rsd='all', kmax=0.5)
