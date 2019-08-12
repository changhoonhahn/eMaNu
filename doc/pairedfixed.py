'''

investigate paired-fixed simulations for the bispectrum

'''
import os 
import h5py
import numpy as np 
import scipy.stats 
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
    ''' Comparison between the standard N-body P_ell to the paired-fixed P_ell. 
    There's also a comparison of the bias.  
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
    sub0.set_ylim(1e3, 1e5) 
    # average redshift-space P0
    sub1.plot(k_rsd, np.average(p0k_rsd_std, axis=0), c='k')
    sub1.scatter(k_rsd, np.average(p0k_rsd_pfd, axis=0), color='C1', s=5, zorder=10) 
    sub1.text(0.95, 0.95, r'redshift-space $P_0$', ha='right', va='top', transform=sub1.transAxes, fontsize=20)
    sub1.set_xlim(9e-3, 0.5) 
    sub1.set_xscale('log') 
    sub1.set_xticklabels([]) 
    sub1.set_yscale('log') 
    sub1.set_ylim(1e3, 1e5) 
    sub1.set_yticklabels([]) 
    # average redshift-space P2
    sub2.plot(k_rsd, np.average(p2k_rsd_std, axis=0), c='k')
    sub2.scatter(k_rsd, np.average(p2k_rsd_pfd, axis=0), color='C1', s=5, zorder=10) 
    sub2.text(0.95, 0.95, r'redshift-space $P_2$', ha='right', va='top', transform=sub2.transAxes, fontsize=20)
    sub2.set_xlim(9e-3, 0.5) 
    sub2.set_xscale('log') 
    sub2.set_xticklabels([]) 
    sub2.set_yscale('log') 
    sub2.set_ylim(1e3, 1e5) 
    sub2.set_yticklabels([]) 
    
    # --- bias comparison --- 
    thetas = ['fiducial', 'Om_m', 'Om_p', 'Ob2_m', 'Ob2_p', 'h_m', 'h_p', 'ns_m', 'ns_p', 's8_m', 's8_p', 
            'Mnu_p', 'Mnu_pp', 'Mnu_ppp']
    lbls = ['fid.', r'$\Omega_m^{-}$', r'$\Omega_m^{+}$', r'$\Omega_b^{-}$', r'$\Omega_b^{+}$', r'$h^{-}$', r'$h^{+}$', 
            r'$n_s^{-}$', r'$n_s^{+}$', r'$\sigma_8^{-}$', r'$\sigma_8^{+}$', r'$M_\nu^{+}$', r'$M_\nu^{++}$', r'$M_\nu^{+++}$']
    
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
    
        lsty = '-' 
        clr = 'k'
        if (_i > 0) and (_i < 10): 
            clr = 'C%i' % (_i-1) 
        elif _i >= 10: 
            lsty = '-.'
            clr = 'C%i' % ((_i-1) % 10) 
        # bias real-space P0
        _plt, = sub3.plot(k_real, bias_p0k_real, c=clr, ls=lsty)
        sub3.plot([9e-3, 0.5], [0.0, 0.0], c='k', ls='--') 
        sub3.set_xlim(9e-3, 0.5) 
        sub3.set_xscale('log') 
        sub3.set_ylabel(r'bias $\beta$ ($\sigma$)', fontsize=25) 
        sub3.set_ylim(-4., 4.) 
        if _i == 0: plts = [] 
        plts.append(_plt) 

        sub4.plot(k_rsd, bias_p0k_rsd, c=clr, ls=lsty)
        sub4.plot([9e-3, 0.5], [0.0, 0.0], c='k', ls='--') 
        sub4.set_xlim(9e-3, 0.5) 
        sub4.set_xscale('log') 
        sub4.set_ylim(-4., 4.) 
        sub4.set_yticklabels([]) 

        sub5.plot(k_rsd, bias_p2k_rsd, c=clr, ls=lsty)
        sub5.plot([9e-3, 0.5], [0.0, 0.0], c='k', ls='--') 
        sub5.set_xlim(9e-3, 0.5) 
        sub5.set_xscale('log') 
        sub5.set_ylim(-4., 4.) 
        sub5.set_yticklabels([]) 

    sub3.fill_between([9e-3, 0.7], [-2, -2], [2, 2], color='k', alpha=0.2, linewidth=0.) 
    sub4.fill_between([9e-3, 0.7], [-2, -2], [2, 2], color='k', alpha=0.2, linewidth=0.) 
    sub5.fill_between([9e-3, 0.7], [-2, -2], [2, 2], color='k', alpha=0.2, linewidth=0.) 

    sub3.legend(plts[:5], lbls[:5], 
            loc='lower left', ncol=6, handletextpad=0.3, columnspacing=0.8, fontsize=12, 
            bbox_to_anchor=(0.,-0.05))
    sub4.legend(plts[5:10], lbls[5:10], 
            loc='lower left', ncol=6, handletextpad=0.3, columnspacing=0.8, fontsize=12, 
            bbox_to_anchor=(0.,-0.05))
    sub5.legend(plts[10:], lbls[10:], 
            loc='lower left', ncol=6, handletextpad=0.3, columnspacing=0.8, fontsize=12, 
            bbox_to_anchor=(0.,-0.05))

    fig.subplots_adjust(wspace=0.1) 
    ffig = os.path.join(dir_doc, 'pf_Pk.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') 
    return None 


def pf_Bk(kmax=0.5): 
    ''' Comparison between the standard N-body B0 to the paired-fixed B0. 
    There's also a comparison of the bias.  
    '''
    fig0 = plt.figure(figsize=(17, 9))
    gs0 = mpl.gridspec.GridSpec(2, 1, figure=fig0, height_ratios=[1,1], hspace=0.1) 
    sub0 = plt.subplot(gs0[0])
    sub2 = plt.subplot(gs0[1])

    fig1 = plt.figure(figsize=(17, 9))
    gs1 = mpl.gridspec.GridSpec(2, 1, figure=fig1, height_ratios=[1,1], hspace=0.1) 
    sub1 = plt.subplot(gs1[0]) 
    sub3 = plt.subplot(gs1[1]) 
    
    # read in real-space bispectrum (standard)
    i_k, j_k, l_k, bk_real_std = X_std('bk', 'fiducial', rsd='real', silent=False) 
    # read in real-space bispectrum (paired-fixed)
    _, _, _, bk_real_pfd1 = X_pfd_1('bk', 'fiducial', rsd='real', silent=False) 
    _, _, _, bk_real_pfd2 = X_pfd_2('bk', 'fiducial', rsd='real', silent=False) 
    bk_real_pfd = 0.5 * (bk_real_pfd1 + bk_real_pfd2)
    # read in redshift-space bispectrum (standard)
    _, _, _, bk_rsd_std = X_std('bk', 'fiducial', rsd=0, silent=False) 
    # read in redshift-space bispectrum (paired-fixed)
    _, _, _, bk_rsd_pfd1 = X_pfd_1('bk', 'fiducial', rsd=0, silent=False) 
    _, _, _, bk_rsd_pfd2 = X_pfd_2('bk', 'fiducial', rsd=0, silent=False) 
    bk_rsd_pfd = 0.5 * (bk_rsd_pfd1 + bk_rsd_pfd2)
    # k limit 
    klim = ((i_k * kf <= kmax) & (j_k * kf <= kmax) & (l_k * kf <= kmax)) 

    # average real-space B0
    sub0.plot(range(np.sum(klim)), np.average(bk_real_std, axis=0)[klim], c='k',
            label='%i standard $N$-body' % bk_real_std.shape[0])
    sub0.scatter(range(np.sum(klim)), np.average(bk_real_pfd, axis=0)[klim], color='C1', s=5, zorder=10, 
            label='%i paired-fixed pairs' % bk_real_pfd.shape[0])
    sub0.text(0.95, 0.95, r'real-space', ha='right', va='top', transform=sub0.transAxes, fontsize=25)
    sub0.legend(loc='lower left', markerscale=5, handletextpad=0.2, frameon=True, fontsize=20) 
    sub0.set_xlim(0, np.sum(klim))
    sub0.set_xticklabels([]) 
    sub0.set_ylabel('$B_0(k_1, k_2, k_3)$', fontsize=25)
    sub0.set_yscale('log') 
    sub0.set_ylim(1e5, 1e10)
    # average redshift-space B0
    sub1.plot(range(np.sum(klim)), np.average(bk_rsd_std, axis=0)[klim], c='k', 
            label='%i standard $N$-body' % bk_real_std.shape[0])
    sub1.scatter(range(np.sum(klim)), np.average(bk_rsd_pfd, axis=0)[klim], color='C1', s=5, zorder=10,
            label='%i paired-fixed pairs' % bk_real_pfd.shape[0])
    sub1.text(0.95, 0.95, r'redshift-space', ha='right', va='top', transform=sub1.transAxes, fontsize=25)
    sub1.legend(loc='lower left', markerscale=5, handletextpad=0.2, frameon=False, fontsize=20) 
    sub1.set_xlim(0, np.sum(klim))
    sub1.set_xticklabels([]) 
    sub1.set_ylabel('$B_0(k_1, k_2, k_3)$', fontsize=25)
    sub1.set_yscale('log') 
    sub1.set_ylim(1e5, 1e10)

    # --- bias comparison
    thetas = ['fiducial', 'Om_m', 'Om_p', 'Ob2_m', 'Ob2_p', 'h_m', 'h_p', 'ns_m', 'ns_p', 's8_m', 's8_p', 
            'Mnu_p', 'Mnu_pp', 'Mnu_ppp']
    lbls = ['fid.', r'$\Omega_m^{-}$', r'$\Omega_m^{+}$', r'$\Omega_b^{-}$', r'$\Omega_b^{+}$', r'$h^{-}$', r'$h^{+}$', 
            r'$n_s^{-}$', r'$n_s^{+}$', r'$\sigma_8^{-}$', r'$\sigma_8^{+}$', r'$M_\nu^{+}$', r'$M_\nu^{++}$', r'$M_\nu^{+++}$']

    for _i, theta in enumerate(thetas): 
        # read in real-space bispectrum (standard)
        _, _, _, bk_real_std = X_std('bk', theta, rsd='real', silent=False) 
        # read in real-space bispectrum (paired-fixed)
        _, _, _, bk_real_pfd1 = X_pfd_1('bk', theta, rsd='real', silent=False) 
        _, _, _, bk_real_pfd2 = X_pfd_2('bk', theta, rsd='real', silent=False) 
        bk_real_pfd = 0.5 * (bk_real_pfd1 + bk_real_pfd2)
        # read in redshift-space bispectrum (standard)
        _, _, _, bk_rsd_std = X_std('bk', theta, rsd=0, silent=False) 
        # read in redshift-space bispectrum (paired-fixed)
        _, _, _, bk_rsd_pfd1 = X_pfd_1('bk', theta, rsd=0, silent=False) 
        _, _, _, bk_rsd_pfd2 = X_pfd_2('bk', theta, rsd=0, silent=False) 
        bk_rsd_pfd = 0.5 * (bk_rsd_pfd1 + bk_rsd_pfd2)

        bias_bk_real   = pf_bias(bk_real_std, bk_real_pfd)
        bias_bk_rsd    = pf_bias(bk_rsd_std, bk_rsd_pfd)
        
        lsty = '-' 
        clr = 'k'
        if (_i > 0) and (_i < 10): 
            clr = 'C%i' % (_i-1) 
        elif _i >= 10: 
            lsty = '-.'
            clr = 'C%i' % ((_i-1) % 10) 
    
        # bias real-space P0
        _plt, = sub2.plot(range(np.sum(klim)), bias_bk_real[klim], c=clr, ls=lsty, lw=(1.-0.02*float(_i)))
        sub2.plot([0.0, np.sum(klim)], [0.0, 0.0], c='k', ls='--') 
        sub2.set_xlim(0.0, np.sum(klim)) 
        sub2.set_ylabel(r'bias $\beta$ ($\sigma$)', fontsize=25) 
        sub2.set_ylim(-4.5, 4.5) 
        if _i == 0: plts = []
        plts.append(_plt) 
        
        sub3.plot(range(np.sum(klim)), bias_bk_rsd[klim], c=clr, ls=lsty, lw=(1.-0.02*float(_i)))
        sub3.plot([0.0, np.sum(klim)], [0.0, 0.0], c='k', ls='--') 
        sub3.set_xlim(0.0, np.sum(klim)) 
        sub3.set_ylabel(r'bias $\beta$ ($\sigma$)', fontsize=25) 
        sub3.set_ylim(-4.5, 4.5)  
    
    sub2.fill_between([0.0, np.sum(klim)], [-2, -2], [2, 2], color='k', alpha=0.2, linewidth=0) 
    sub3.fill_between([0.0, np.sum(klim)], [-2, -2], [2, 2], color='k', alpha=0.2, linewidth=0) 

    sub2.legend(plts, lbls, loc='lower left', ncol=15, handletextpad=0.4, fontsize=13, 
            columnspacing=0.8, bbox_to_anchor=(0.,-0.04))
    sub3.legend(plts, lbls, loc='lower left', ncol=15, handletextpad=0.4, fontsize=13,
            columnspacing=0.8, bbox_to_anchor=(0.,-0.04))

    ffig = os.path.join(dir_doc, 'pf_Bk.real.png') 
    fig0.savefig(ffig, bbox_inches='tight') 
    fig0.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') 
    ffig = os.path.join(dir_doc, 'pf_Bk.rsd.png') 
    fig1.savefig(ffig, bbox_inches='tight') 
    fig1.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') 
    return None 


def Pk_bias(kmax=0.5): 
    ''' Compare the P_ell bias distributions for the different cosmologies
    '''
    fig = plt.figure(figsize=(15,5))
    sub0 = fig.add_subplot(131) 
    sub1 = fig.add_subplot(132) 
    sub2 = fig.add_subplot(133) 
    
    thetas = ['fiducial', 'Om_m', 'Om_p', 'Ob2_m', 'Ob2_p', 'h_m', 'h_p', 'ns_m', 'ns_p', 's8_m', 's8_p', 
            'Mnu_p', 'Mnu_pp', 'Mnu_ppp']
    lbls = ['fid.', r'$\Omega_m^{-}$', r'$\Omega_m^{+}$', r'$\Omega_b^{-}$', r'$\Omega_b^{+}$', r'$h^{-}$', r'$h^{+}$', 
            r'$n_s^{-}$', r'$n_s^{+}$', r'$\sigma_8^{-}$', r'$\sigma_8^{+}$', r'$M_\nu^{+}$', r'$M_\nu^{++}$', r'$M_\nu^{+++}$']
    
    plts = [] 
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

        if _i == 0: klim = (k_real < kmax) 

        bias_p0k_real   = pf_bias(p0k_real_std, p0k_real_pfd)[klim]
        bias_p0k_rsd    = pf_bias(p0k_rsd_std, p0k_rsd_pfd)[klim]
        bias_p2k_rsd    = pf_bias(p2k_rsd_std, p2k_rsd_pfd)[klim]
    
        lsty = '-' 
        clr = 'k'
        if (_i > 0) and (_i < 10): 
            clr = 'C%i' % (_i-1) 
        elif _i >= 10: 
            lsty = '-.'
            clr = 'C%i' % ((_i-1) % 10) 

        h0, b0 = np.histogram(bias_p0k_real, density=True, range=(-5., 5.), bins=30)
        h1, b1 = np.histogram(bias_p0k_rsd, density=True, range=(-5., 5.), bins=30)
        h2, b2 = np.histogram(bias_p2k_rsd, density=True, range=(-5., 5.), bins=30)

        _plt, = sub0.plot(b0[1:]+0.001*(_i - 0.5*float(len(thetas))), h0, drawstyle='steps-pre', color=clr, ls=lsty) 
        sub1.plot(b1[1:]+0.001*(_i - 0.5*float(len(thetas))), h1, drawstyle='steps-pre', color=clr, ls=lsty) 
        sub2.plot(b2[1:]+0.001*(_i - 0.5*float(len(thetas))), h2, drawstyle='steps-pre', color=clr, ls=lsty) 
        plts.append(_plt) 

    x = np.linspace(-5., 5., 100)
    y = scipy.stats.norm.pdf(x, 0., 1.)
    sub0.plot(x, y, c='k', ls='--') 
    sub1.plot(x, y, c='k', ls='--') 
    sub2.plot(x, y, c='k', ls='--') 
    sub0.set_xlim(-5., 5.) 
    sub1.set_xlim(-5., 5.) 
    sub2.set_xlim(-5., 5.) 
    
    sub0.set_ylim(0., 0.75) 
    sub1.set_ylim(0., 0.75) 
    sub2.set_ylim(0., 0.75) 

    sub0.text(0.95, 0.95, r'real-space $P_0$', ha='right', va='top', transform=sub0.transAxes, fontsize=20)
    sub1.text(0.95, 0.95, r'redshift-space $P_0$', ha='right', va='top', transform=sub1.transAxes, fontsize=20)
    sub2.text(0.95, 0.95, r'redshift-space $P_2$', ha='right', va='top', transform=sub2.transAxes, fontsize=20)

    sub0.legend(plts[:5], lbls[:5], loc='upper left', ncol=1, handletextpad=0.4, fontsize=15)
    sub1.legend(plts[5:10], lbls[5:10], loc='upper left', ncol=1, handletextpad=0.4, fontsize=15)
    sub2.legend(plts[10:], lbls[10:], loc='upper left', ncol=1, handletextpad=0.4, fontsize=15)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'bias $\beta$ ($\sigma$)', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    fig.subplots_adjust(wspace=0.2) 
    ffig = os.path.join(dir_fig, 'pf_Pk_bias.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def Bk_bias(kmax=0.5): 
    ''' Compare the P_ell bias distributions for the different cosmologies
    '''
    fig = plt.figure(figsize=(10,5))
    sub0 = fig.add_subplot(121) 
    sub1 = fig.add_subplot(122) 
    
    thetas = ['fiducial', 'Om_m', 'Om_p', 'Ob2_m', 'Ob2_p', 'h_m', 'h_p', 'ns_m', 'ns_p', 's8_m', 's8_p', 
            'Mnu_p', 'Mnu_pp', 'Mnu_ppp']
    lbls = ['fid.', r'$\Omega_m^{-}$', r'$\Omega_m^{+}$', r'$\Omega_b^{-}$', r'$\Omega_b^{+}$', r'$h^{-}$', r'$h^{+}$', 
            r'$n_s^{-}$', r'$n_s^{+}$', r'$\sigma_8^{-}$', r'$\sigma_8^{+}$', r'$M_\nu^{+}$', r'$M_\nu^{++}$', r'$M_\nu^{+++}$']
    
    plts = [] 
    for _i, theta in enumerate(thetas): 
        # read in real-space bispectrum (standard)
        i_k, j_k, l_k, bk_real_std = X_std('bk', theta, rsd='real', silent=False) 
        # read in real-space bispectrum (paired-fixed)
        _, _, _, bk_real_pfd1 = X_pfd_1('bk', theta, rsd='real', silent=False) 
        _, _, _, bk_real_pfd2 = X_pfd_2('bk', theta, rsd='real', silent=False) 
        bk_real_pfd = 0.5 * (bk_real_pfd1 + bk_real_pfd2)
        # read in redshift-space bispectrum (standard)
        _, _, _, bk_rsd_std = X_std('bk', theta, rsd=0, silent=False) 
        # read in redshift-space bispectrum (paired-fixed)
        _, _, _, bk_rsd_pfd1 = X_pfd_1('bk', theta, rsd=0, silent=False) 
        _, _, _, bk_rsd_pfd2 = X_pfd_2('bk', theta, rsd=0, silent=False) 
        bk_rsd_pfd = 0.5 * (bk_rsd_pfd1 + bk_rsd_pfd2)

        if _i == 0: klim = (i_k * kf <= kmax) & (j_k * kf <= kmax) & (l_k * kf <= kmax) 

        bias_bk_real   = pf_bias(bk_real_std, bk_real_pfd)[klim]
        bias_bk_rsd    = pf_bias(bk_rsd_std, bk_rsd_pfd)[klim]

        lsty = '-' 
        clr = 'k'
        if (_i > 0) and (_i < 10): 
            clr = 'C%i' % (_i-1) 
        elif _i >= 10: 
            lsty = '-.'
            clr = 'C%i' % ((_i-1) % 10) 
        
        h0, b0 = np.histogram(bias_bk_real, density=True, range=(-5., 5.), bins=30)
        h1, b1 = np.histogram(bias_bk_rsd, density=True, range=(-5., 5.), bins=30)

        _plt, = sub0.plot(b0[1:]+0.001*(_i - 0.5*float(len(thetas))), h0, drawstyle='steps-pre', color=clr, ls=lsty) 
        sub1.plot(b1[1:]+0.001*(_i - 0.5*float(len(thetas))), h1, drawstyle='steps-pre', color=clr, ls=lsty) 
        plts.append(_plt) 

    x = np.linspace(-5., 5., 100)
    y = scipy.stats.norm.pdf(x, 0., 1.)
    sub0.plot(x, y, c='k', ls='--') 
    sub1.plot(x, y, c='k', ls='--') 
    sub0.set_xlim(-5., 5.) 
    sub1.set_xlim(-5., 5.) 
    
    sub0.set_ylim(0., 0.6) 
    sub1.set_ylim(0., 0.6) 

    sub0.text(0.95, 0.95, r'real-space $B_0$', ha='right', va='top', transform=sub0.transAxes, fontsize=20)
    sub1.text(0.95, 0.95, r'redshift-space $B_0$', ha='right', va='top', transform=sub1.transAxes, fontsize=20)

    sub0.legend(plts[:7], lbls[:7], loc='upper left', ncol=1, handletextpad=0.4, fontsize=15)
    sub1.legend(plts[7:], lbls[7:], loc='upper left', ncol=1, handletextpad=0.4, fontsize=15)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'bias $\beta$ ($\sigma$)', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    fig.subplots_adjust(wspace=0.2) 
    ffig = os.path.join(dir_fig, 'pf_Bk_bias.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def pf_dPdtheta(): 
    ''' Comparison of dP/dtheta from paired-fixed vs standard N-body
    '''
    thetas  = ['Om', 'Ob2', 'h', 'ns', 's8', 'Mnu']
    lbls    = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$']
    ylims0  = [(-1e6, 1e3), (1e3, 2e6), (-1e6, -1e1), (-1e6, -1e2), (-1e6, -1e2), (1e1, 1e5)]
    ylims1  = [(-1e6, -1e3), (5e2, 1e6), (-5e5, -5e1), (-5e5, -5e2), (-1e4, 1e5), (1e2, 2e4)]
    
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    fig0 = plt.figure(figsize=(24,5))
    _gs0 = mpl.gridspec.GridSpec(2, len(thetas), figure=fig0, height_ratios=[3,2], hspace=0.075, wspace=0.2) 
    gs0 = [plt.subplot(_gs0[i]) for i in range(len(thetas))]
    gs3 = [plt.subplot(_gs0[i+len(thetas)]) for i in range(len(thetas))]
    
    fig1 = plt.figure(figsize=(24,8))
    _gs1 = mpl.gridspec.GridSpec(3, len(thetas), figure=fig1, height_ratios=[3,3,2], hspace=0.075, wspace=0.2) 
    gs1 = [plt.subplot(_gs1[i]) for i in range(len(thetas))]
    gs2 = [plt.subplot(_gs1[i+len(thetas)]) for i in range(len(thetas))]
    gs4 = [plt.subplot(_gs1[i+2*len(thetas)]) for i in range(len(thetas))]

    for i_tt, theta, lbl in zip(range(len(thetas)), thetas, lbls): 
        # real-space d logP / d theta (standard)
        k_real, dpdt_real_std = X_std('dpk', theta, rsd='real', dmnu='fin_2lpt', silent=False) 
        # real-space d logP / d theta (paired-fixed)
        _, dpdt_real_pfd0 = X_pfd_1('dpk', theta, rsd='real', dmnu='fin_2lpt', silent=False) 
        _, dpdt_real_pfd1 = X_pfd_2('dpk', theta, rsd='real', dmnu='fin_2lpt', silent=False) 
        dpdt_real_pfd = 0.5 * (dpdt_real_pfd0 + dpdt_real_pfd1) 

        # redshift-space d logP0 / d theta (standard)
        k_rsd, dp0dt_rsd_std, dp2dt_rsd_std = X_std('dpk', theta, rsd=0, dmnu='fin_2lpt', silent=False) 
        # redshift-space d logP0 / d theta (paired-fixed)
        _, dp0dt_rsd_pfd0, dp2dt_rsd_pfd0 = X_pfd_1('dpk', theta, rsd=0, dmnu='fin_2lpt', silent=False) 
        _, dp0dt_rsd_pfd1, dp2dt_rsd_pfd1 = X_pfd_2('dpk', theta, rsd=0, dmnu='fin_2lpt', silent=False) 
        dp0dt_rsd_pfd = 0.5 * (dp0dt_rsd_pfd0 + dp0dt_rsd_pfd1) 
        dp2dt_rsd_pfd = 0.5 * (dp2dt_rsd_pfd0 + dp2dt_rsd_pfd1) 
        
        bias_dpdt_real = pf_bias(dpdt_real_std, dpdt_real_pfd)
        bias_dp0dt_rsd = pf_bias(dp0dt_rsd_std, dp0dt_rsd_pfd)
        bias_dp2dt_rsd = pf_bias(dp2dt_rsd_std, dp2dt_rsd_pfd)
        
        sub0 = gs0[i_tt] 
        _plt00, = sub0.plot(k_real, np.average(dpdt_real_std, axis=0), c='k', label='standard')
        _plt01, = sub0.plot(k_real, np.average(dpdt_real_pfd, axis=0), c='C1', label='paired-fixed')
        sub0.set_xlim(9e-3, 0.7) 
        sub0.set_xscale('log') 
        sub0.set_xticklabels([]) 
        if i_tt == 0: sub0.set_yscale('symlog', linthreshy=1e4) 
        else: sub0.set_yscale('symlog') 
        sub0.set_ylim(ylims0[i_tt]) 
        #sub0.set_yticks(yticks0[i_tt])
        sub0.text(0.975, 0.95, lbl, ha='right', va='top', transform=sub0.transAxes, fontsize=25)

        sub1 = gs1[i_tt] 
        _plt10, = sub1.plot(k_rsd, np.average(dp0dt_rsd_std, axis=0), c='k', label='standard')
        _plt11, = sub1.plot(k_rsd, np.average(dp0dt_rsd_pfd, axis=0), c='C1', label='paired-fixed')
        sub1.set_xlim(9e-3, 0.7) 
        sub1.set_xscale('log') 
        sub1.set_xticklabels([]) 
        if i_tt == 0: sub1.set_yscale('symlog', linthreshy=1e4) 
        else: sub1.set_yscale('symlog') 
        sub1.set_ylim(ylims0[i_tt])
        #sub1.set_yticks(yticks0[i_tt]) 
        sub1.text(0.975, 0.95, lbl, ha='right', va='top', transform=sub1.transAxes, fontsize=25)

        sub2 = gs2[i_tt] 
        sub2.plot(k_rsd, np.average(dp2dt_rsd_std, axis=0), c='k', label='standard')
        sub2.plot(k_rsd, np.average(dp2dt_rsd_pfd, axis=0), c='C1', label='paired-fixed')
        sub2.set_xlim(9e-3, 0.7) 
        sub2.set_xscale('log') 
        sub2.set_xticklabels([]) 
        sub2.set_yscale('symlog') 
        sub2.set_ylim(ylims1[i_tt]) 
        if i_tt == 4: sub2.set_yscale('symlog', linthreshy=1e2) 
        else: sub2.set_yscale('symlog') 
        #sub2.set_yticks(yticks1[i_tt]) 

        if i_tt == 0: 
            sub0.set_ylabel(r'${\rm d} P/{\rm d} \theta$', fontsize=20) 
            sub1.set_ylabel(r'${\rm d} P_0/{\rm d} \theta$', fontsize=20) 
            sub2.set_ylabel(r'${\rm d} P_2/{\rm d} \theta$', fontsize=20) 
        if i_tt == len(thetas)-2: 
            sub0.legend([_plt00], ['%i standard\n $N$-body' % dpdt_real_std.shape[0]], 
                    loc='lower right', handletextpad=0.3, fontsize=15, bbox_to_anchor=(1.05,-0.05))
            sub1.legend([_plt10], ['%i standard\n $N$-body' % dp0dt_rsd_std.shape[0]], 
                    loc='lower right', handletextpad=0.3, fontsize=15, bbox_to_anchor=(1.05,-0.05))
        if i_tt == len(thetas)-1: 
            sub0.legend([_plt01], ['%i paired\n fixed pairs' % dpdt_real_pfd.shape[0]], 
                    loc='lower right', handletextpad=0.3, fontsize=15, bbox_to_anchor=(1.05,-0.05))
            sub1.legend([_plt11], ['%i paired\n fixed pairs' % dp0dt_rsd_pfd.shape[0]], 
                    loc='lower right', handletextpad=0.3, fontsize=15, bbox_to_anchor=(1.05,-0.05))
        
        sub3 = gs3[i_tt] 
        sub3.plot(k_real, bias_dpdt_real, c='C0', label='real $P_0$')
        sub3.fill_between([9e-3, 0.7], [-2, -2], [2, 2], color='k', alpha=0.2, linewidth=0.0) 
        sub3.plot([9e-3, 0.7], [0.0, 0.0], c='k', ls='--', zorder=10)
        sub3.set_xlim(9e-3, 0.7) 
        sub3.set_xscale('log') 
        sub3.set_ylim(-4.5, 4.5) 
        sub3.set_yticks([-4., -2., 0., 2., 4.]) 

        sub4 = gs4[i_tt] 
        _plt1, = sub4.plot(k_rsd, bias_dp0dt_rsd, c='C0')
        _plt2, = sub4.plot(k_rsd, bias_dp2dt_rsd, c='C2')
        sub4.fill_between([9e-3, 0.7], [-2, -2], [2, 2], color='k', alpha=0.2, linewidth=0.0) 
        sub4.plot([9e-3, 0.7], [0.0, 0.0], c='k', ls='--', zorder=10)
        sub4.set_xlim(9e-3, 0.7) 
        sub4.set_xscale('log') 
        sub4.set_ylim(-4.5, 4.5) 
        sub4.set_yticks([-4., -2., 0., 2., 4.]) 
        if i_tt == 0: 
            sub3.set_ylabel(r'bias $\beta$ ($\sigma$)', fontsize=20) 
            sub4.set_ylabel(r'bias $\beta$ ($\sigma$)', fontsize=20) 
            sub4.legend([_plt1], ['$\ell = 0$'], loc='lower left', bbox_to_anchor=(0.0, -0.05),
                    handletextpad=0.2, fontsize=15)
        if i_tt == 1: 
            sub4.legend([_plt2], ['$\ell = 2$'], loc='lower left', bbox_to_anchor=(0.0, -0.05),
                    handletextpad=0.2, fontsize=15)

    bkgd0 = fig0.add_subplot(111, frameon=False)
    bkgd0.set_xlabel('$k$', labelpad=10, fontsize=25) 
    bkgd0.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    
    bkgd1 = fig1.add_subplot(111, frameon=False)
    bkgd1.set_xlabel('$k$', labelpad=10, fontsize=25) 
    bkgd1.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    ffig = os.path.join(dir_doc, 'pf_dPdtheta.real.png')
    fig0.savefig(ffig, bbox_inches='tight') 
    fig0.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') 

    ffig = os.path.join(dir_doc, 'pf_dPdtheta.rsd.png')
    fig1.savefig(ffig, bbox_inches='tight') 
    fig1.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') 
    return None


def pf_dBdtheta(kmax=0.5): 
    ''' Comparison of dB/dtheta from paired-fixed vs standard N-body
    '''
    thetas  = ['Om', 'Ob2', 'h', 'ns', 's8', 'Mnu']
    lbls    = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$']
    ylims0  = [(-2e11, -1e5), (1e6, 2e11), (-2e10, -5e5), (-2e10, -5e5), (-1e10, -1e6), (5e4, 5e9)]
    
    fig1 = plt.figure(figsize=(24,12))
    fig2 = plt.figure(figsize=(24,12))
    for i_tt, theta, lbl in zip(range(len(thetas)), thetas, lbls): 
        # real-space d logB / d theta (standard)
        i_k, j_k, l_k, dbdt_real_std = X_std('dbk', theta, rsd='real', dmnu='fin_2lpt', silent=False) 
        # real-space d logP / d theta (paired-fixed)
        _, _, _, dbdt_real_pfd0 = X_pfd_1('dbk', theta, rsd='real', dmnu='fin_2lpt', silent=False) 
        _, _, _, dbdt_real_pfd1 = X_pfd_2('dbk', theta, rsd='real', dmnu='fin_2lpt', silent=False) 
        dbdt_real_pfd = 0.5 * (dbdt_real_pfd0 + dbdt_real_pfd1) 

        # redshift-space d logP0 / d theta (standard)
        _, _, _, dbdt_rsd_std = X_std('dbk', theta, rsd=0, dmnu='fin_2lpt', silent=False) 
        # redshift-space d logP0 / d theta (paired-fixed)
        _, _, _, dbdt_rsd_pfd0 = X_pfd_1('dbk', theta, rsd=0, dmnu='fin_2lpt', silent=False) 
        _, _, _, dbdt_rsd_pfd1 = X_pfd_2('dbk', theta, rsd=0, dmnu='fin_2lpt', silent=False) 
        dbdt_rsd_pfd = 0.5 * (dbdt_rsd_pfd0 + dbdt_rsd_pfd1) 
        
        bias_dbdt_real = pf_bias(dbdt_real_std, dbdt_real_pfd)
        bias_dbdt_rsd = pf_bias(dbdt_rsd_std, dbdt_rsd_pfd)

        # get k limit
        if i_tt == 0: klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 

        sub1 = fig1.add_subplot(len(thetas),2,2*i_tt+1)
        sub2 = fig2.add_subplot(len(thetas),2,2*i_tt+1)

        sub1.plot(range(np.sum(klim)), np.average(dbdt_real_std, axis=0)[klim], 
                label='%i standard $N$-body' % dbdt_real_std.shape[0])
        sub1.plot(range(np.sum(klim)), np.average(dbdt_real_pfd, axis=0)[klim], lw=0.95, 
                label='%i paired-fixed pairs' % dbdt_real_pfd.shape[0])
        sub1.set_xlim(0, np.sum(klim)) 
        sub1.set_ylim(ylims0[i_tt]) 
        sub1.set_yscale('symlog') 

        sub2.plot(range(np.sum(klim)), np.average(dbdt_rsd_std, axis=0)[klim], 
                label='%i standard $N$-body' % dbdt_rsd_std.shape[0])
        sub2.plot(range(np.sum(klim)), np.average(dbdt_rsd_pfd, axis=0)[klim], lw=0.95, 
                label='%i paired-fixed pairs' % dbdt_rsd_pfd.shape[0])
        sub2.set_xlim(0, np.sum(klim)) 
        sub2.set_ylim(ylims0[i_tt]) 
        sub2.set_yscale('symlog') 

        if theta != thetas[-1]: 
            sub1.set_xticklabels([]) 
            sub2.set_xticklabels([]) 
        else: 
            sub1.set_xlabel('triangle configurations', fontsize=20)
            sub2.set_xlabel('triangle configurations', fontsize=20)
        if i_tt == 0: 
            sub1.legend(loc='lower right', handletextpad=0.3, ncol=2, fontsize=15, 
                    columnspacing=0.8, bbox_to_anchor=(1., -0.1))
            sub2.legend(loc='lower right', handletextpad=0.3, ncol=2, fontsize=15, 
                    columnspacing=0.8, bbox_to_anchor=(1., -0.1))
        #sub.text(0.975, 0.925, lbl, ha='right', va='top', transform=sub.transAxes, fontsize=25)

        sub1 = fig1.add_subplot(len(thetas),2,2*i_tt+2) 
        sub2 = fig2.add_subplot(len(thetas),2,2*i_tt+2) 

        sub1.plot(range(np.sum(klim)), bias_dbdt_real[klim])
        sub1.plot(range(np.sum(klim)), np.zeros(np.sum(klim)), c='k', ls='--')
        sub1.fill_between(range(np.sum(klim)), np.zeros(np.sum(klim))-2, np.zeros(np.sum(klim))+2, 
                color='k', linewidth=0, alpha=0.2)
        sub1.set_xlim(0, np.sum(klim)) 
        sub1.set_yticks([-4., -2., 0., 2., 4.]) 
        sub1.set_ylim(-4.5, 4.5) 

        sub2.plot(range(np.sum(klim)), bias_dbdt_rsd[klim])
        sub2.plot(range(np.sum(klim)), np.zeros(np.sum(klim)), c='k', ls='--')
        sub2.fill_between(range(np.sum(klim)), np.zeros(np.sum(klim))-2, np.zeros(np.sum(klim))+2, 
                color='k', linewidth=0, alpha=0.2)
        sub2.set_xlim(0, np.sum(klim)) 
        sub2.set_yticks([-4., -2., 0., 2., 4.]) 
        sub2.set_ylim(-4.5, 4.5) 

        if theta != thetas[-1]: 
            sub1.set_xticklabels([]) 
            sub2.set_xticklabels([]) 
        if i_tt == 4: 
            sub1.set_xlabel('triangle configurations', fontsize=20)
            sub2.set_xlabel('triangle configurations', fontsize=20)
        sub1.text(0.98, 0.95, lbl, ha='right', va='top', transform=sub1.transAxes, fontsize=18)
        sub2.text(0.98, 0.95, lbl, ha='right', va='top', transform=sub2.transAxes, fontsize=18)

    bkgd = fig1.add_subplot(121, frameon=False)
    bkgd.set_ylabel(r'real-space $~~{\rm d}B/{\rm d} \theta$', labelpad=15, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd = fig1.add_subplot(122, frameon=False)
    bkgd.set_ylabel(r'bias $\beta$ ($\sigma$)', labelpad=5, fontsize=24) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig1.subplots_adjust(hspace=0.075, wspace=0.175) 
    ffig = os.path.join(dir_doc, 'pf_dBdtheta.real.kmax%.1f.png' % (kmax))
    fig1.savefig(ffig, bbox_inches='tight') 
    fig1.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') 

    bkgd = fig2.add_subplot(121, frameon=False)
    bkgd.set_ylabel(r'redshift-space $~~{\rm d}B_0/{\rm d} \theta$', labelpad=15, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd = fig2.add_subplot(122, frameon=False)
    bkgd.set_ylabel(r'bias $\beta$ ($\sigma$)', labelpad=5, fontsize=24) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig2.subplots_adjust(hspace=0.075, wspace=0.175) 
    ffig = os.path.join(dir_doc, 'pf_dBdtheta.rsd.kmax%.1f.png' % (kmax))
    fig2.savefig(ffig, bbox_inches='tight') 
    fig2.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') 
    return None


def dPdtheta_bias(kmax=0.5): 
    ''' Compare the d P_ell/d theta bias distributions for the different cosmologies
    '''
    from matplotlib.patches import Rectangle
    fig = plt.figure(figsize=(15,5))
    sub0 = fig.add_subplot(131) 
    sub1 = fig.add_subplot(132) 
    sub2 = fig.add_subplot(133) 

    thetas  = ['Om', 'Ob2', 'h', 'ns', 's8', 'Mnu']
    lbls    = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$']
    
    for _i, theta in enumerate(thetas): 
        # real-space d logP / d theta (standard)
        k_real, dpdt_real_std = X_std('dpk', theta, rsd='real', dmnu='fin_2lpt', silent=False) 
        # real-space d logP / d theta (paired-fixed)
        _, dpdt_real_pfd0 = X_pfd_1('dpk', theta, rsd='real', dmnu='fin_2lpt', silent=False) 
        _, dpdt_real_pfd1 = X_pfd_2('dpk', theta, rsd='real', dmnu='fin_2lpt', silent=False) 
        dpdt_real_pfd = 0.5 * (dpdt_real_pfd0 + dpdt_real_pfd1) 

        # redshift-space d logP0 / d theta (standard)
        k_rsd, dp0dt_rsd_std, dp2dt_rsd_std = X_std('dpk', theta, rsd=0, dmnu='fin_2lpt', silent=False) 
        # redshift-space d logP0 / d theta (paired-fixed)
        _, dp0dt_rsd_pfd0, dp2dt_rsd_pfd0 = X_pfd_1('dpk', theta, rsd=0, dmnu='fin_2lpt', silent=False) 
        _, dp0dt_rsd_pfd1, dp2dt_rsd_pfd1 = X_pfd_2('dpk', theta, rsd=0, dmnu='fin_2lpt', silent=False) 
        dp0dt_rsd_pfd = 0.5 * (dp0dt_rsd_pfd0 + dp0dt_rsd_pfd1) 
        dp2dt_rsd_pfd = 0.5 * (dp2dt_rsd_pfd0 + dp2dt_rsd_pfd1) 

        if _i == 0: klim = (k_real < kmax) 
        
        bias_dpdt_real = pf_bias(dpdt_real_std, dpdt_real_pfd)[klim]
        bias_dp0dt_rsd = pf_bias(dp0dt_rsd_std, dp0dt_rsd_pfd)[klim]
        bias_dp2dt_rsd = pf_bias(dp2dt_rsd_std, dp2dt_rsd_pfd)[klim]

        clr = 'k'
        if _i > 0: clr = 'C%i' % (_i-1) 

        h0, b0 = np.histogram(bias_dpdt_real, density=True, range=(-5., 5.), bins=30)
        h1, b1 = np.histogram(bias_dp0dt_rsd, density=True, range=(-5., 5.), bins=30)
        h2, b2 = np.histogram(bias_dp2dt_rsd, density=True, range=(-5., 5.), bins=30)

        sub0.plot(b0[1:]+0.001*(_i - 0.5*float(len(thetas))), h0, drawstyle='steps-pre', color=clr) 
        sub1.plot(b1[1:]+0.001*(_i - 0.5*float(len(thetas))), h1, drawstyle='steps-pre', color=clr) 
        sub2.plot(b2[1:]+0.001*(_i - 0.5*float(len(thetas))), h2, drawstyle='steps-pre', color=clr) 

        if _i == 0: handles = []
        handles.append(Rectangle((0,0), 1, 1, color='none', ec=clr))
    
    x = np.linspace(-5., 5., 100)
    y = scipy.stats.norm.pdf(x, 0., 1.)
    sub0.plot(x, y, c='k', ls='--') 
    sub1.plot(x, y, c='k', ls='--') 
    sub2.plot(x, y, c='k', ls='--') 

    sub0.set_xlim(-5., 5.) 
    sub1.set_xlim(-5., 5.) 
    sub2.set_xlim(-5., 5.) 
    
    sub0.set_ylim(0., 0.75) 
    sub1.set_ylim(0., 0.75) 
    sub2.set_ylim(0., 0.75) 

    sub0.text(0.95, 0.95, r'real-space ${\rm d} P_0/{\rm d} \theta$', 
            ha='right', va='top', transform=sub0.transAxes, fontsize=20)
    sub1.text(0.95, 0.95, r'redshift-space ${\rm d} P_0/{\rm d} \theta$',
            ha='right', va='top', transform=sub1.transAxes, fontsize=20)
    sub2.text(0.95, 0.95, r'redshift-space ${\rm d} P_2/{\rm d} \theta$',
            ha='right', va='top', transform=sub2.transAxes, fontsize=20)

    sub0.legend(handles, lbls, loc='upper left', ncol=1, handletextpad=0.4, fontsize=15)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'bias $\beta$ ($\sigma$)', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    fig.subplots_adjust(wspace=0.2) 
    ffig = os.path.join(dir_fig, 'pf_dpdt_bias.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def dBdtheta_bias(kmax=0.5): 
    ''' Compare the d B / d theta bias distributions for the different cosmologies
    '''
    from matplotlib.patches import Rectangle
    fig = plt.figure(figsize=(10,5))
    sub0 = fig.add_subplot(121) 
    sub1 = fig.add_subplot(122) 

    thetas  = ['Om', 'Ob2', 'h', 'ns', 's8', 'Mnu']
    lbls    = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$']
    
    for _i, theta in enumerate(thetas): 
        # real-space d logB / d theta (standard)
        i_k, j_k, l_k, dbdt_real_std = X_std('dbk', theta, rsd='real', dmnu='fin_2lpt', silent=False) 
        # real-space d logP / d theta (paired-fixed)
        _, _, _, dbdt_real_pfd0 = X_pfd_1('dbk', theta, rsd='real', dmnu='fin_2lpt', silent=False) 
        _, _, _, dbdt_real_pfd1 = X_pfd_2('dbk', theta, rsd='real', dmnu='fin_2lpt', silent=False) 
        dbdt_real_pfd = 0.5 * (dbdt_real_pfd0 + dbdt_real_pfd1) 

        # redshift-space d logP0 / d theta (standard)
        _, _, _, dbdt_rsd_std = X_std('dbk', theta, rsd=0, dmnu='fin_2lpt', silent=False) 
        # redshift-space d logP0 / d theta (paired-fixed)
        _, _, _, dbdt_rsd_pfd0 = X_pfd_1('dbk', theta, rsd=0, dmnu='fin_2lpt', silent=False) 
        _, _, _, dbdt_rsd_pfd1 = X_pfd_2('dbk', theta, rsd=0, dmnu='fin_2lpt', silent=False) 
        dbdt_rsd_pfd = 0.5 * (dbdt_rsd_pfd0 + dbdt_rsd_pfd1) 

        if _i == 0: klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 
        
        bias_dbdt_real = pf_bias(dbdt_real_std, dbdt_real_pfd)[klim]
        bias_dbdt_rsd = pf_bias(dbdt_rsd_std, dbdt_rsd_pfd)[klim]

        clr = 'k'
        if _i > 0: clr = 'C%i' % (_i-1) 

        h0, b0 = np.histogram(bias_dbdt_real, density=True, range=(-5., 5.), bins=30)
        h1, b1 = np.histogram(bias_dbdt_rsd, density=True, range=(-5., 5.), bins=30)

        sub0.plot(b0[1:]+0.001*(_i - 0.5*float(len(thetas))), h0, drawstyle='steps-pre', color=clr) 
        sub1.plot(b1[1:]+0.001*(_i - 0.5*float(len(thetas))), h1, drawstyle='steps-pre', color=clr) 

        if _i == 0: handles = []
        handles.append(Rectangle((0,0), 1, 1, color='none', ec=clr))
    
    x = np.linspace(-5., 5., 100)
    y = scipy.stats.norm.pdf(x, 0., 1.)
    sub0.plot(x, y, c='k', ls='--') 
    sub1.plot(x, y, c='k', ls='--') 

    sub0.set_xlim(-5., 5.) 
    sub1.set_xlim(-5., 5.) 
    
    sub0.set_ylim(0., 1.5) 
    sub1.set_ylim(0., 0.65) 

    sub0.text(0.95, 0.95, r'real-space ${\rm d} B_0/{\rm d} \theta$', 
            ha='right', va='top', transform=sub0.transAxes, fontsize=20)
    sub1.text(0.95, 0.95, r'redshift-space ${\rm d} B_0/{\rm d} \theta$',
            ha='right', va='top', transform=sub1.transAxes, fontsize=20)

    sub0.legend(handles, lbls, loc='upper left', ncol=1, handletextpad=0.4, fontsize=15)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'bias $\beta$ ($\sigma$)', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    fig.subplots_adjust(wspace=0.2) 
    ffig = os.path.join(dir_fig, 'pf_dbdt_bias.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def dBdtheta_chi2(kmax=0.2): 
    ''' Compare the d B / d theta bias distributions for the different cosmologies
    '''
    thetas  = ['Om', 'Ob2', 'h', 'ns', 's8', 'Mnu']
    lbls    = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$']
    
    for _i, theta in enumerate(thetas): 
        # real-space d logB / d theta (standard)
        i_k, j_k, l_k, dbdt_real_std = X_std('dbk', theta, rsd='real', dmnu='fin_2lpt', silent=False) 
        # real-space d logP / d theta (paired-fixed)
        _, _, _, dbdt_real_pfd0 = X_pfd_1('dbk', theta, rsd='real', dmnu='fin_2lpt', silent=False) 
        _, _, _, dbdt_real_pfd1 = X_pfd_2('dbk', theta, rsd='real', dmnu='fin_2lpt', silent=False) 
        dbdt_real_pfd = 0.5 * (dbdt_real_pfd0 + dbdt_real_pfd1) 

        # redshift-space d logP0 / d theta (standard)
        _, _, _, dbdt_rsd_std = X_std('dbk', theta, rsd=0, dmnu='fin_2lpt', silent=False) 
        # redshift-space d logP0 / d theta (paired-fixed)
        _, _, _, dbdt_rsd_pfd0 = X_pfd_1('dbk', theta, rsd=0, dmnu='fin_2lpt', silent=False) 
        _, _, _, dbdt_rsd_pfd1 = X_pfd_2('dbk', theta, rsd=0, dmnu='fin_2lpt', silent=False) 
        dbdt_rsd_pfd = 0.5 * (dbdt_rsd_pfd0 + dbdt_rsd_pfd1) 

        if _i == 0: klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 

        del_dbdt_real = np.average(dbdt_real_std, axis=0)[klim] - np.average(dbdt_real_pfd, axis=0)[klim]
        del_dbdt_rsd = np.average(dbdt_rsd_std, axis=0)[klim] - np.average(dbdt_rsd_pfd, axis=0)[klim]

        cov_real = np.cov(dbdt_real_std[:,klim].T)
        cov_rsd = np.cov(dbdt_rsd_std[:,klim].T)

        cinv_real = np.linalg.inv(cov_real) 
        cinv_rsd = np.linalg.inv(cov_rsd) 

        print('--- %s ---' % theta) 
        print('real-space : %.4f' % np.dot(np.dot(del_dbdt_real, cinv_real), del_dbdt_real.T))
        print('redshift-space : %.4f' % np.dot(np.dot(del_dbdt_rsd, cinv_rsd), del_dbdt_rsd.T))
    return None 


def pf_dlogPdtheta(): 
    ''' Comparison of dlogP/dtheta from paired-fixed vs standard N-body
    '''
    thetas  = ['Om', 'Ob2', 'h', 'ns', 's8']#, 'Mnu']
    lbls    = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$']#, r'$M_\nu$']
    ylims0  = [(-10., 1.), (-1., 15.), (-3.2, 0.), (-3.5, -0.4), (-1.6, -0.2)]
    yticks0 = [[-8, -4, 0], [0, 4, 8, 12], [-3., -2.,  -1., 0.], [-3., -2., -1.], [-1.4, -1., -0.6, -0.2]]
    ylims1  = [(-7., 0.), (-1., 11.5), (-2.5, 0.), (-2.6, -0.4), (-2.5, 1.5)]
    yticks1 = [[-6, -4, -2, 0], [0, 4, 8], [-2.,  -1., 0.], [-2., -1.], [-2., -1., 0., 1.]]
    
    plt.rc('ytick', labelsize=10)
    fig0 = plt.figure(figsize=(24,5))
    _gs0 = mpl.gridspec.GridSpec(2, len(thetas), figure=fig0, height_ratios=[3,2], hspace=0.075, wspace=0.2) 
    gs0 = [plt.subplot(_gs0[i]) for i in range(len(thetas))]
    gs3 = [plt.subplot(_gs0[i+len(thetas)]) for i in range(len(thetas))]
    
    fig1 = plt.figure(figsize=(24,8))
    _gs1 = mpl.gridspec.GridSpec(3, len(thetas), figure=fig1, height_ratios=[3,3,2], hspace=0.075, wspace=0.2) 
    gs1 = [plt.subplot(_gs1[i]) for i in range(len(thetas))]
    gs2 = [plt.subplot(_gs1[i+len(thetas)]) for i in range(len(thetas))]
    gs4 = [plt.subplot(_gs1[i+2*len(thetas)]) for i in range(len(thetas))]

    for i_tt, theta, lbl in zip(range(len(thetas)), thetas, lbls): 
        # real-space d logP / d theta (standard)
        k_real, dpdt_real_std = X_std('dlogpk', theta, rsd='real', silent=False) 
        # real-space d logP / d theta (paired-fixed)
        _, dpdt_real_pfd0 = X_pfd_1('dlogpk', theta, rsd='real', silent=False) 
        _, dpdt_real_pfd1 = X_pfd_2('dlogpk', theta, rsd='real', silent=False) 
        dpdt_real_pfd = 0.5 * (dpdt_real_pfd0 + dpdt_real_pfd1) 

        # redshift-space d logP0 / d theta (standard)
        k_rsd, dp0dt_rsd_std, dp2dt_rsd_std = X_std('dlogpk', theta, rsd=0, silent=False) 
        # redshift-space d logP0 / d theta (paired-fixed)
        _, dp0dt_rsd_pfd0, dp2dt_rsd_pfd0 = X_pfd_1('dlogpk', theta, rsd=0, silent=False) 
        _, dp0dt_rsd_pfd1, dp2dt_rsd_pfd1 = X_pfd_2('dlogpk', theta, rsd=0, silent=False) 
        dp0dt_rsd_pfd = 0.5 * (dp0dt_rsd_pfd0 + dp0dt_rsd_pfd1) 
        dp2dt_rsd_pfd = 0.5 * (dp2dt_rsd_pfd0 + dp2dt_rsd_pfd1) 
        
        bias_dpdt_real = pf_bias(dpdt_real_std, dpdt_real_pfd)
        bias_dp0dt_rsd = pf_bias(dp0dt_rsd_std, dp0dt_rsd_pfd)
        bias_dp2dt_rsd = pf_bias(dp2dt_rsd_std, dp2dt_rsd_pfd)
    
        sub0 = gs0[i_tt] 
        sub0.plot(k_real, np.average(dpdt_real_std, axis=0), c='k', label='standard')
        sub0.plot(k_real, np.average(dpdt_real_pfd, axis=0), c='C1', label='paired-fixed')
        sub0.set_xlim(9e-3, 0.7) 
        sub0.set_xscale('log') 
        sub0.set_xticklabels([]) 
        sub0.set_ylim(ylims0[i_tt]) 
        sub0.set_yticks(yticks0[i_tt])
        sub0.text(0.975, 0.95, lbl, ha='right', va='top', transform=sub0.transAxes, fontsize=25)

        sub1 = gs1[i_tt] 
        sub1.plot(k_rsd, np.average(dp0dt_rsd_std, axis=0), c='k', label='standard')
        sub1.plot(k_rsd, np.average(dp0dt_rsd_pfd, axis=0), c='C1', label='paired-fixed')
        sub1.set_xlim(9e-3, 0.7) 
        sub1.set_xscale('log') 
        sub1.set_xticklabels([]) 
        sub1.set_ylim(ylims0[i_tt]) 
        sub1.set_yticks(yticks0[i_tt]) 
        sub1.text(0.975, 0.95, lbl, ha='right', va='top', transform=sub1.transAxes, fontsize=25)

        sub2 = gs2[i_tt] 
        sub2.plot(k_rsd, np.average(dp2dt_rsd_std, axis=0), c='k', label='standard')
        sub2.plot(k_rsd, np.average(dp2dt_rsd_pfd, axis=0), c='C1', label='paired-fixed')
        sub2.set_xlim(9e-3, 0.7) 
        sub2.set_xscale('log') 
        sub2.set_xticklabels([]) 
        sub2.set_ylim(ylims1[i_tt]) 
        sub2.set_yticks(yticks1[i_tt]) 

        if i_tt == 0: 
            sub0.set_ylabel(r'${\rm d}\log P/{\rm d} \theta$', fontsize=20) 
            sub1.set_ylabel(r'${\rm d}\log P_0/{\rm d} \theta$', fontsize=20) 
            sub2.set_ylabel(r'${\rm d}\log P_2/{\rm d} \theta$', fontsize=20) 
        if i_tt == len(thetas)-1: 
            sub0.legend(loc='lower left', handletextpad=0.3, fontsize=15)
            sub1.legend(loc='lower left', handletextpad=0.3, fontsize=15)
        
        sub3 = gs3[i_tt] 
        sub3.plot(k_real, bias_dpdt_real, c='C0', label='real $P_0$')
        sub3.fill_between([9e-3, 0.7], [-2, -2], [2, 2], color='k', alpha=0.2, linewidth=0.0) 
        sub3.plot([9e-3, 0.7], [0.0, 0.0], c='k', ls='--', zorder=10)
        sub3.set_xlim(9e-3, 0.7) 
        sub3.set_xscale('log') 
        sub3.set_ylim(-4.5, 4.5) 
        sub3.set_yticks([-4., -2., 0., 2., 4.]) 

        sub4 = gs4[i_tt] 
        _plt1, = sub4.plot(k_rsd, bias_dp0dt_rsd, c='C0')
        _plt2, = sub4.plot(k_rsd, bias_dp2dt_rsd, c='C2')
        sub4.fill_between([9e-3, 0.7], [-2, -2], [2, 2], color='k', alpha=0.2, linewidth=0.0) 
        sub4.plot([9e-3, 0.7], [0.0, 0.0], c='k', ls='--', zorder=10)
        sub4.set_xlim(9e-3, 0.7) 
        sub4.set_xscale('log') 
        sub4.set_ylim(-4.5, 4.5) 
        sub4.set_yticks([-4., -2., 0., 2., 4.]) 
        if i_tt == 0: 
            sub3.set_ylabel(r'bias ($\beta$)', fontsize=20) 
            sub4.set_ylabel(r'bias ($\beta$)', fontsize=20) 
            sub4.legend([_plt1], ['$\ell = 0$'], loc='lower left', bbox_to_anchor=(0.0, -0.05),
                    handletextpad=0.2, fontsize=15)
        if i_tt == 1: 
            sub4.legend([_plt2], ['$\ell = 2$'], loc='lower left', bbox_to_anchor=(0.0, -0.05),
                    handletextpad=0.2, fontsize=15)

    bkgd0 = fig0.add_subplot(111, frameon=False)
    bkgd0.set_xlabel('$k$', labelpad=10, fontsize=25) 
    bkgd0.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    
    bkgd1 = fig1.add_subplot(111, frameon=False)
    bkgd1.set_xlabel('$k$', labelpad=10, fontsize=25) 
    bkgd1.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    ffig = os.path.join(dir_fig, 'pf_dlogPdtheta.real.png')
    fig0.savefig(ffig, bbox_inches='tight') 

    ffig = os.path.join(dir_fig, 'pf_dlogPdtheta.rsd.png')
    fig1.savefig(ffig, bbox_inches='tight') 
    return None


def pf_dlogBdtheta(kmax=0.5): 
    ''' Comparison of dlogB/dtheta from paired-fixed vs standard N-body
    '''
    thetas  = ['Om', 'Ob2', 'h', 'ns', 's8']#, 'Mnu']
    lbls    = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$']#, r'$M_\nu$']
    ylims0  = [(-15., 5.), (-2., 24.), (-4., 0.5), (-5., -0.5), (-6., 0)]
    ylims1  = [(-7., 7.), (-25., 25.), (-4., 4), (-3., 3), (-5., 5.)]
    
    fig1 = plt.figure(figsize=(24,10))
    fig2 = plt.figure(figsize=(24,10))
    for i_tt, theta, lbl in zip(range(len(thetas)), thetas, lbls): 
        # real-space d logB / d theta (standard)
        i_k, j_k, l_k, dbdt_real_std = X_std('dlogbk', theta, rsd='real', silent=False) 
        # real-space d logP / d theta (paired-fixed)
        _, _, _, dbdt_real_pfd0 = X_pfd_1('dlogbk', theta, rsd='real', silent=False) 
        _, _, _, dbdt_real_pfd1 = X_pfd_2('dlogbk', theta, rsd='real', silent=False) 
        dbdt_real_pfd = 0.5 * (dbdt_real_pfd0 + dbdt_real_pfd1) 

        # redshift-space d logP0 / d theta (standard)
        _, _, _, dbdt_rsd_std = X_std('dlogbk', theta, rsd=0, silent=False) 
        # redshift-space d logP0 / d theta (paired-fixed)
        _, _, _, dbdt_rsd_pfd0 = X_pfd_1('dlogbk', theta, rsd=0, silent=False) 
        _, _, _, dbdt_rsd_pfd1 = X_pfd_2('dlogbk', theta, rsd=0, silent=False) 
        dbdt_rsd_pfd = 0.5 * (dbdt_rsd_pfd0 + dbdt_rsd_pfd1) 
        
        bias_dbdt_real = pf_bias(dbdt_real_std, dbdt_real_pfd)
        bias_dbdt_rsd = pf_bias(dbdt_rsd_std, dbdt_rsd_pfd)

        # get k limit
        if i_tt == 0: klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 

        sub1 = fig1.add_subplot(len(thetas),2,2*i_tt+1)
        sub2 = fig2.add_subplot(len(thetas),2,2*i_tt+1)

        _, _, _, _dbdt_real_std = Forecast.quijote_dBkdtheta(theta, log=True, rsd='real', flag='ncv', dmnu='fin')
        _, _, _, _dbdt_real_pfd = Forecast.quijote_dBkdtheta(theta, log=True, rsd='real', flag='ncv', dmnu='fin')
        sub1.plot(range(np.sum(klim)), _dbdt_real_std[klim], label='standard $N$-body')
        sub1.plot(range(np.sum(klim)), _dbdt_real_pfd[klim], lw=0.95, label='paired-fixed')
        sub1.set_xlim(0, np.sum(klim)) 
        sub1.set_ylim(ylims0[i_tt]) 

        sub2.plot(range(np.sum(klim)), np.average(dbdt_rsd_std, axis=0)[klim], label='standard $N$-body')
        sub2.plot(range(np.sum(klim)), np.average(dbdt_rsd_pfd, axis=0)[klim], lw=0.95, label='paired-fixed')
        sub2.set_xlim(0, np.sum(klim)) 
        sub2.set_ylim(ylims0[i_tt]) 

        if theta != thetas[-1]: 
            sub1.set_xticklabels([]) 
            sub2.set_xticklabels([]) 
        else: 
            sub1.set_xlabel('triangle configurations', fontsize=25)
            sub2.set_xlabel('triangle configurations', fontsize=25)
        if i_tt == 2: 
            sub1.set_ylabel(r'real-space $~~{\rm d}\log B/{\rm d} \theta$', fontsize=25) 
            sub2.set_ylabel(r'redshift-space $~~{\rm d}\log B_0/{\rm d} \theta$', fontsize=25) 
        if i_tt == 0: 
            sub1.legend(loc='upper left', handletextpad=0.2, ncol=2, fontsize=15)
            sub2.legend(loc='upper left', handletextpad=0.2, ncol=2, fontsize=15)
        #sub.text(0.975, 0.925, lbl, ha='right', va='top', transform=sub.transAxes, fontsize=25)

        sub1 = fig1.add_subplot(len(thetas),2,2*i_tt+2) 
        sub2 = fig2.add_subplot(len(thetas),2,2*i_tt+2) 

        sub1.plot(range(np.sum(klim)), bias_dbdt_real[klim])
        sub1.plot(range(np.sum(klim)), np.zeros(np.sum(klim)), c='k', ls='--')
        sub1.fill_between(range(np.sum(klim)), np.zeros(np.sum(klim))-2, np.zeros(np.sum(klim))+2, 
                color='k', linewidth=0, alpha=0.2)
        sub1.set_xlim(0, np.sum(klim)) 
        sub1.set_ylim(-4.5, 4.5) 

        sub2.plot(range(np.sum(klim)), bias_dbdt_rsd[klim])
        sub2.plot(range(np.sum(klim)), np.zeros(np.sum(klim)), c='k', ls='--')
        sub2.fill_between(range(np.sum(klim)), np.zeros(np.sum(klim))-2, np.zeros(np.sum(klim))+2, 
                color='k', linewidth=0, alpha=0.2)
        sub2.set_xlim(0, np.sum(klim)) 
        sub2.set_ylim(-4.5, 4.5) 

        if theta != thetas[-1]: 
            sub1.set_xticklabels([]) 
            sub2.set_xticklabels([]) 
        if i_tt == 2: 
            sub1.set_ylabel(r'bias $(\beta)$', fontsize=25) 
            sub2.set_ylabel(r'bias $(\beta)$', fontsize=25) 
        if i_tt == 4: 
            sub1.set_xlabel('triangle configurations', fontsize=25)
            sub2.set_xlabel('triangle configurations', fontsize=25)
        sub1.text(0.98, 0.925, lbl, ha='right', va='top', transform=sub1.transAxes, fontsize=20)
        sub2.text(0.98, 0.925, lbl, ha='right', va='top', transform=sub2.transAxes, fontsize=20)

    fig1.subplots_adjust(hspace=0.075, wspace=0.15) 
    ffig = os.path.join(dir_fig, 'pf_dlogBdtheta.real.kmax%.1f.png' % (kmax))
    fig1.savefig(ffig, bbox_inches='tight') 

    fig2.subplots_adjust(hspace=0.075, wspace=0.15) 
    ffig = os.path.join(dir_fig, 'pf_dlogBdtheta.rsd.kmax%.1f.png' % (kmax))
    fig2.savefig(ffig, bbox_inches='tight') 
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
    lbls        = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$']

    dpdts_std, dpdts_pfd = [], [] 
    for theta in thetas:
        if rsd == 'real': 
            # real-space d logP / d theta (standard)
            #_, dpdt_std = Forecast.quijote_dPkdtheta(theta, rsd='real', flag='reg', dmnu='fin_2lpt') 
            # real-space d logP / d theta (paired-fixed)
            #_, dpdt_pfd = Forecast.quijote_dPkdtheta(theta, rsd='real', flag='ncv', dmnu='fin_2lpt') 
            _, dpdt_std = X_std('dpk', theta, rsd=rsd, dmnu='fin_2lpt', silent=False) 
            _, dpdt_pfd0 = X_pfd_1('dpk', theta, rsd=rsd, dmnu='fin_2lpt', silent=False) 
            _, dpdt_pfd1 = X_pfd_2('dpk', theta, rsd=rsd, dmnu='fin_2lpt', silent=False) 
        else: 
            # redshift-space d logP / d theta (standard)
            #_, dpdt_std = Forecast.quijote_dP02kdtheta(theta, rsd=rsd, flag='reg', dmnu='fin_2lpt') 
            # redshift-space d logP / d theta (paired-fixed)
            #_, dpdt_pfd = Forecast.quijote_dP02kdtheta(theta, rsd=rsd, flag='ncv', dmnu='fin_2lpt') 
            _, dp0dt_std, dp2dt_std = X_std('dpk', theta, rsd=rsd, dmnu='fin_2lpt', silent=False) 
            dpdt_std = np.concatenate([dp0dt_std, dp2dt_std], axis=1) 
            _, dp0dt_pfd0, dp2dt_pfd0 = X_pfd_1('dpk', theta, rsd=rsd, dmnu='fin_2lpt', silent=False) 
            _, dp0dt_pfd1, dp2dt_pfd1 = X_pfd_2('dpk', theta, rsd=rsd, dmnu='fin_2lpt', silent=False) 
            dpdt_pfd0 = np.concatenate([dp0dt_pfd0, dp2dt_pfd0], axis=1) 
            dpdt_pfd1 = np.concatenate([dp0dt_pfd1, dp2dt_pfd1], axis=1) 
        #dpdts_std.append(dpdt_std[klim]) 
        #dpdts_pfd.append(dpdt_pfd[klim]) 
        dpdt_pfd = 0.5 * (dpdt_pfd0 + dpdt_pfd1) 
        dpdts_std.append(np.average(dpdt_std, axis=0)[klim]) 
        dpdts_pfd.append(np.average(dpdt_pfd, axis=0)[klim]) 

    Fij_std = Forecast.Fij(dpdts_std, C_inv) 
    Fij_pfd = Forecast.Fij(dpdts_pfd, C_inv) 

    fig = plt.figure(figsize=(17,5))
    sub = fig.add_subplot(131) 
    cm = sub.pcolormesh(Fij_std, cmap='RdBu', 
            norm=SymLogNorm(linthresh=1e4, vmin=-3e6, vmax=3e6),) 
    plt.colorbar(cm) 
    sub.set_xticks(np.arange(len(thetas))+0.5)
    sub.set_xticklabels(lbls) 
    sub.set_yticks(np.arange(len(thetas))+0.5)
    sub.set_yticklabels(lbls) 
    sub.set_title(r'standard $N$-body $F^{\rm std}_{ij}$', fontsize=20) 
    
    sub = fig.add_subplot(132) 
    cm = sub.pcolormesh(Fij_pfd, cmap='RdBu',
            norm=SymLogNorm(linthresh=1e4, vmin=-3e6, vmax=3e6),) 
    plt.colorbar(cm) 
    sub.set_xticks(np.arange(len(thetas))+0.5)
    sub.set_xticklabels(lbls) 
    sub.set_yticks(np.arange(len(thetas))+0.5)
    sub.set_yticklabels(lbls) 
    sub.set_title(r'paired-fixed $F^{\rm pf}_{ij}$', fontsize=20) 
    
    sub = fig.add_subplot(133) 
    cm = sub.pcolormesh(Fij_pfd/Fij_std - 1, vmin=-0.1, vmax=0.1, cmap='RdBu')
    plt.colorbar(cm) 
    sub.set_xticks(np.arange(len(thetas))+0.5)
    sub.set_xticklabels(lbls) 
    sub.set_yticks(np.arange(len(thetas))+0.5)
    sub.set_yticklabels(lbls) 
    sub.set_title(r'$(F^{\rm pf}_{ij} - F^{\rm std}_{ij})/ F^{\rm std}_{ij}$', fontsize=20) 
    
    fig.subplots_adjust(wspace=0.175) 
    if rsd == 'real': 
        ffig = os.path.join(dir_doc, 'pf_P_Fij.real.kmax%.1f.png' % kmax)
    else: 
        ffig = os.path.join(dir_doc, 'pf_P_Fij%s.kmax%.1f.png' % (_rsd_str(rsd), kmax)) 
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') 
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
    
    thetas      = ['Om', 'Ob2', 'h', 'ns', 's8', 'Mnu']
    lbls        = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$']
    dbdts_std, dbdts_pfd = [], [] 
    for theta in thetas:
        #if rsd == 'real': 
        #    # redshift-space d logP0 / d theta (standard)
        #    _, _, _, dbdt_std = Forecast.quijote_dBkdtheta(theta, rsd='real', flag='reg', dmnu='fin_2lpt')
        #    # redshift-space d logP0 / d theta (paired-fixed)
        #    _, _, _, dbdt_pfd = Forecast.quijote_dBkdtheta(theta, rsd='real', flag='ncv', dmnu='fin_2lpt')
        #else: 
        #    # real-space d logB / d theta (standard)
        #    _, _, _, dbdt_std = Forecast.quijote_dBkdtheta(theta, rsd=rsd, flag='reg', dmnu='fin_2lpt')
        #    # real-space d logP / d theta (paired-fixed)
        #    _, _, _, dbdt_pfd = Forecast.quijote_dBkdtheta(theta, rsd=rsd, flag='ncv', dmnu='fin_2lpt')
        #dbdts_std.append(dbdt_std[klim]) 
        #dbdts_pfd.append(dbdt_pfd[klim]) 

        # d logP / d theta (standard)
        _, _, _, dbdt_std = X_std('dbk', theta, rsd=rsd, dmnu='fin_2lpt', silent=False) 
        # d logP / d theta (paired-fixed)
        _, _, _, dbdt_pfd0 = X_pfd_1('dbk', theta, rsd=rsd, dmnu='fin_2lpt', silent=False) 
        _, _, _, dbdt_pfd1 = X_pfd_2('dbk', theta, rsd=rsd, dmnu='fin_2lpt', silent=False) 
        dbdt_pfd = 0.5 * (dbdt_pfd0 + dbdt_pfd1) 
        dbdts_std.append(np.average(dbdt_std, axis=0)[klim]) 
        dbdts_pfd.append(np.average(dbdt_pfd, axis=0)[klim]) 

    Fij_std = Forecast.Fij(dbdts_std, C_inv) 
    Fij_pfd = Forecast.Fij(dbdts_pfd, C_inv) 

    fig = plt.figure(figsize=(17,5))
    sub = fig.add_subplot(131) 
    cm = sub.pcolormesh(Fij_std, cmap='RdBu', 
            norm=SymLogNorm(linthresh=1e4, vmin=-3e6, vmax=3e6),) 
    plt.colorbar(cm) 
    sub.set_xticks(np.arange(len(thetas))+0.5)
    sub.set_xticklabels(lbls) 
    sub.set_yticks(np.arange(len(thetas))+0.5)
    sub.set_yticklabels(lbls) 
    sub.set_title(r'standard $N$-body $F^{\rm std}_{ij}$', fontsize=20) 
    
    sub = fig.add_subplot(132) 
    cm = sub.pcolormesh(Fij_pfd, cmap='RdBu',
            norm=SymLogNorm(linthresh=1e4, vmin=-3e6, vmax=3e6),) 
    plt.colorbar(cm) 
    sub.set_xticks(np.arange(len(thetas))+0.5)
    sub.set_xticklabels(lbls) 
    sub.set_yticks(np.arange(len(thetas))+0.5)
    sub.set_yticklabels(lbls) 
    sub.set_title(r'paired-fixed $F^{\rm pf}_{ij}$', fontsize=20) 
    
    sub = fig.add_subplot(133) 
    cm = sub.pcolormesh(Fij_pfd/Fij_std - 1, vmin=-0.1, vmax=0.1, cmap='RdBu')
    plt.colorbar(cm) 
    sub.set_xticks(np.arange(len(thetas))+0.5)
    sub.set_xticklabels(lbls) 
    sub.set_yticks(np.arange(len(thetas))+0.5)
    sub.set_yticklabels(lbls) 
    sub.set_title(r'$(F^{\rm pf}_{ij} - F^{\rm std}_{ij})/ F^{\rm std}_{ij}$', fontsize=20) 
    
    fig.subplots_adjust(wspace=0.175) 
    if rsd == 'real': 
        ffig = os.path.join(dir_doc, 'pf_B_Fij.real.kmax%.1f.png' % kmax)
    else: 
        ffig = os.path.join(dir_doc, 'pf_B_Fij%s.kmax%.1f.png' % (_rsd_str(rsd), kmax)) 
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') 
    return None 


def pf_P_posterior(rsd='all', kmax=0.5): 
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
    theta_lims  = [(0.29, 0.345), (0.028, 0.07), (0.4922, 0.85), (0.7748, 1.15), (0.808, 0.86), (-0.25, 0.25)]
    lbls        = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$']

    dpdts_std, dpdts_pfd = [], [] 
    for theta in thetas:
        if rsd == 'real': 
            # real-space d logP / d theta (standard)
            #_, dpdt_std = Forecast.quijote_dPkdtheta(theta, rsd='real', flag='reg', dmnu='fin_2lpt') 
            # real-space d logP / d theta (paired-fixed)
            #_, dpdt_pfd = Forecast.quijote_dPkdtheta(theta, rsd='real', flag='ncv', dmnu='fin_2lpt') 
            _, dpdt_std = X_std('dpk', theta, rsd=rsd, dmnu='fin_2lpt', silent=False) 
            _, dpdt_pfd0 = X_pfd_1('dpk', theta, rsd=rsd, dmnu='fin_2lpt', silent=False) 
            _, dpdt_pfd1 = X_pfd_2('dpk', theta, rsd=rsd, dmnu='fin_2lpt', silent=False) 
        else: 
            # redshift-space d logP / d theta (standard)
            #_, dpdt_std = Forecast.quijote_dP02kdtheta(theta, rsd=rsd, flag='reg', dmnu='fin_2lpt') 
            # redshift-space d logP / d theta (paired-fixed)
            #_, dpdt_pfd = Forecast.quijote_dP02kdtheta(theta, rsd=rsd, flag='ncv', dmnu='fin_2lpt') 
            _, dp0dt_std, dp2dt_std = X_std('dpk', theta, rsd=rsd, dmnu='fin_2lpt', silent=False) 
            dpdt_std = np.concatenate([dp0dt_std, dp2dt_std], axis=1) 
            _, dp0dt_pfd0, dp2dt_pfd0 = X_pfd_1('dpk', theta, rsd=rsd, dmnu='fin_2lpt', silent=False) 
            _, dp0dt_pfd1, dp2dt_pfd1 = X_pfd_2('dpk', theta, rsd=rsd, dmnu='fin_2lpt', silent=False) 
            dpdt_pfd0 = np.concatenate([dp0dt_pfd0, dp2dt_pfd0], axis=1) 
            dpdt_pfd1 = np.concatenate([dp0dt_pfd1, dp2dt_pfd1], axis=1) 
        #dpdts_std.append(dpdt_std[klim]) 
        #dpdts_pfd.append(dpdt_pfd[klim]) 
        dpdt_pfd = 0.5 * (dpdt_pfd0 + dpdt_pfd1) 
        dpdts_std.append(np.average(dpdt_std, axis=0)[klim]) 
        dpdts_pfd.append(np.average(dpdt_pfd, axis=0)[klim]) 
        Nstd = dpdt_std.shape[0]
        Npfd = dpdt_pfd.shape[0]

    Fij_std = Forecast.Fij(dpdts_std, C_inv) 
    Fij_pfd = Forecast.Fij(dpdts_pfd, C_inv) 

    Finv_std = np.linalg.inv(Fij_std) # invert fisher matrix 
    Finv_pfd = np.linalg.inv(Fij_pfd) # invert fisher matrix 

    sig_std = np.sqrt(np.diag(Finv_std))
    sig_pfd = np.sqrt(np.diag(Finv_pfd))

    titles = [r'$\sigma^{\rm pf}_{%s} = %.2f\times \sigma^{\rm std}_{%s}$' % (lbl.replace('$', ''), ratio, lbl.replace('$', ''))
            for ratio, lbl in zip((sig_pfd / sig_std), lbls)]
    fig = Forecast.plotFisher([Finv_pfd, Finv_std], theta_fid, ranges=theta_lims, colors=['C1', 'C0'], 
            labels=lbls, titles=titles, title_kwargs={'fontsize': 20})

    bkgd = fig.add_subplot(111, frameon=False)
    if rsd == 'real': 
        bkgd.text(0.67, 0.8, 'real-space $P(k)$', ha='center', va='bottom', transform=bkgd.transAxes, fontsize=25)
    else: 
        bkgd.text(0.67, 0.8, 'redshift-space $P_\ell(k)$', ha='center', va='bottom', transform=bkgd.transAxes, fontsize=25)
    bkgd.fill_between([],[],[], color='C0', label=r'%i standard $N$-body' % Nstd) 
    bkgd.fill_between([],[],[], color='C1', label=r'%i paired-fixed pairs' % Npfd) 
    bkgd.legend(loc='upper right', bbox_to_anchor=(0.8, 0.8), handletextpad=0.5, fontsize=20)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    fig.subplots_adjust(wspace=0.05, hspace=0.05) 
    if rsd == 'real': 
        ffig = os.path.join(dir_doc, 'pf_P_posterior.real.kmax%.1f.png' % kmax)
    else: 
        ffig = os.path.join(dir_doc, 'pf_P_posterior%s.kmax%.1f.png' % (_rsd_str(rsd), kmax))
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') 
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
    theta_lims  = [(0.3, 0.335), (0.04, 0.058), (0.6, 0.74), (0.9, 1.015), (0.808, 0.86), (-0.25, 0.25)]
    lbls        = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$']

    dbdts_std, dbdts_pfd = [], [] 
    for theta in thetas:
        #if rsd == 'real': 
        #    # real-space d logB / d theta (standard)
        #    _, _, _, dbdt_std = Forecast.quijote_dBkdtheta(theta, rsd='real', flag='reg', dmnu='fin_2lpt')
        #    # real-space d logP / d theta (paired-fixed)
        #    _, _, _, dbdt_pfd = Forecast.quijote_dBkdtheta(theta, rsd='real', flag='ncv', dmnu='fin_2lpt')
        #else: 
        #    # redshift-space d logP0 / d theta (standard)
        #    _, _, _, dbdt_std = Forecast.quijote_dBkdtheta(theta, rsd=rsd, flag='reg', dmnu='fin_2lpt')
        #    # redshift-space d logP0 / d theta (paired-fixed)
        #    _, _, _, dbdt_pfd = Forecast.quijote_dBkdtheta(theta, rsd=rsd, flag='ncv', dmnu='fin_2lpt')
        #dbdts_std.append(dbdt_std[klim]) 
        #dbdts_pfd.append(dbdt_pfd[klim]) 

        # d logP / d theta (standard)
        _, _, _, dbdt_std = X_std('dbk', theta, rsd=rsd, dmnu='fin_2lpt', silent=False) 
        # d logP / d theta (paired-fixed)
        _, _, _, dbdt_pfd0 = X_pfd_1('dbk', theta, rsd=rsd, dmnu='fin_2lpt', silent=False) 
        _, _, _, dbdt_pfd1 = X_pfd_2('dbk', theta, rsd=rsd, dmnu='fin_2lpt', silent=False) 
        dbdt_pfd = 0.5 * (dbdt_pfd0 + dbdt_pfd1) 

        dbdts_std.append(np.average(dbdt_std, axis=0)[klim]) 
        dbdts_pfd.append(np.average(dbdt_pfd, axis=0)[klim]) 
        
        N_std = dbdt_std.shape[0]
        N_pfd = dbdt_pfd.shape[0]

    Fij_std = Forecast.Fij(dbdts_std, C_inv) 
    Fij_pfd = Forecast.Fij(dbdts_pfd, C_inv) 
    
    Finv_std = np.linalg.inv(Fij_std) # invert fisher matrix 
    Finv_pfd = np.linalg.inv(Fij_pfd) # invert fisher matrix 
    
    sig_std = np.sqrt(np.diag(Finv_std))
    sig_pfd = np.sqrt(np.diag(Finv_pfd))
    
    titles = [r'$\sigma^{\rm pf}_{%s} = %.2f\times \sigma^{\rm std}_{%s}$' % (lbl.replace('$', ''), ratio, lbl.replace('$', ''))
            for ratio, lbl in zip((sig_pfd / sig_std), lbls)]
    fig = Forecast.plotFisher([Finv_pfd, Finv_std], theta_fid, ranges=theta_lims, colors=['C1', 'C0'], 
            labels=lbls, titles=titles, title_kwargs={'fontsize': 20})

    bkgd = fig.add_subplot(111, frameon=False)
    if rsd == 'real': 
        bkgd.text(0.67, 0.8, 'real-space $B(k_1, k_2, k_3)$', ha='center', va='bottom', transform=bkgd.transAxes, fontsize=25)
    else: 
        bkgd.text(0.67, 0.8, 'redshift-space $B_0(k_1, k_2, k_3)$', ha='center', va='bottom', transform=bkgd.transAxes, fontsize=25)
    bkgd.fill_between([],[],[], color='C0', label=r'%i standard $N$-body' % N_std) 
    bkgd.fill_between([],[],[], color='C1', label=r'%i paired-fixed pairs' % N_pfd) 
    bkgd.legend(loc='upper right', bbox_to_anchor=(0.8, 0.8), handletextpad=0.5, fontsize=20)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    fig.subplots_adjust(wspace=0.05, hspace=0.05) 
    if rsd == 'real': 
        ffig = os.path.join(dir_doc, 'pf_B_posterior.real.kmax%.1f.png' % kmax)
    else: 
        ffig = os.path.join(dir_doc, 'pf_B_posterior%s.kmax%.1f.png' % (_rsd_str(rsd), kmax)) 
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') 
    return None 


def pf_P_sigma_convergence(rsd='all', kmax=0.5): 
    ''' compare the fisher matrix derived from paired-fixed derivatives vs 
    standard n-body derivatives
    '''
    if rsd not in ['real', 'all']: raise ValueError
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
    theta_lims  = [(0.29, 0.345), (0.028, 0.07), (0.4922, 0.85), (0.7748, 1.15), (0.808, 0.86), (-0.25, 0.25)]
    lbls        = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$']

    dpdts_std, dpdts_pfd = [], [] 
    for theta in thetas:
        if rsd == 'real': 
            # real-space d logP / d theta (standard)
            #_, dpdt_std = Forecast.quijote_dPkdtheta(theta, rsd='real', flag='reg', dmnu='fin_2lpt') 
            # real-space d logP / d theta (paired-fixed)
            #_, dpdt_pfd = Forecast.quijote_dPkdtheta(theta, rsd='real', flag='ncv', dmnu='fin_2lpt') 
            _, dpdt_std = X_std('dpk', theta, rsd=rsd, dmnu='fin_2lpt', silent=False) 
            _, dpdt_pfd0 = X_pfd_1('dpk', theta, rsd=rsd, dmnu='fin_2lpt', silent=False) 
            _, dpdt_pfd1 = X_pfd_2('dpk', theta, rsd=rsd, dmnu='fin_2lpt', silent=False) 
        else: 
            # redshift-space d logP / d theta (standard)
            #_, dpdt_std = Forecast.quijote_dP02kdtheta(theta, rsd=rsd, flag='reg', dmnu='fin_2lpt') 
            # redshift-space d logP / d theta (paired-fixed)
            #_, dpdt_pfd = Forecast.quijote_dP02kdtheta(theta, rsd=rsd, flag='ncv', dmnu='fin_2lpt') 
            _, dp0dt_std, dp2dt_std = X_std('dpk', theta, rsd=rsd, dmnu='fin_2lpt', silent=False) 
            dpdt_std = np.concatenate([dp0dt_std, dp2dt_std], axis=1) 
            _, dp0dt_pfd0, dp2dt_pfd0 = X_pfd_1('dpk', theta, rsd=rsd, dmnu='fin_2lpt', silent=False) 
            _, dp0dt_pfd1, dp2dt_pfd1 = X_pfd_2('dpk', theta, rsd=rsd, dmnu='fin_2lpt', silent=False) 
            dpdt_pfd0 = np.concatenate([dp0dt_pfd0, dp2dt_pfd0], axis=1) 
            dpdt_pfd1 = np.concatenate([dp0dt_pfd1, dp2dt_pfd1], axis=1) 
        #dpdts_std.append(dpdt_std[klim]) 
        #dpdts_pfd.append(dpdt_pfd[klim]) 
        dpdt_pfd = 0.5 * (dpdt_pfd0 + dpdt_pfd1) 
        dpdts_std.append(dpdt_std[:,klim]) 
        dpdts_pfd.append(dpdt_pfd[:,klim]) 
        Nstd = dpdt_std.shape[0]
        Npfd = dpdt_pfd.shape[0]
    
    if Nstd == 1500: 
        Nstds = [200, 400, 600, 800, 1000, 1200, 1500]
    elif Nstd == 500:
        Nstds = [100, 200, 300, 400, 500]
    
    sig_std, sig_pfd = [], [] 
    for _Nstd in Nstds: 
        _dpdts_std = [np.average(dpdt[:_Nstd], axis=0) for dpdt in dpdts_std]
        _dpdts_pfd = [np.average(dpdt[:_Nstd/2], axis=0) for dpdt in dpdts_pfd]

        Fij_std = Forecast.Fij(_dpdts_std, C_inv) 
        Fij_pfd = Forecast.Fij(_dpdts_pfd, C_inv) 

        Finv_std = np.linalg.inv(Fij_std) # invert fisher matrix 
        Finv_pfd = np.linalg.inv(Fij_pfd) # invert fisher matrix 

        _sig_std = np.sqrt(np.diag(Finv_std))
        _sig_pfd = np.sqrt(np.diag(Finv_pfd))

        sig_std.append(_sig_std) 
        sig_pfd.append(_sig_pfd) 

    sig_std = np.array(sig_std)#/sig_std[-1]
    sig_pfd = np.array(sig_pfd)#/sig_pfd[-1]
    
    fig = plt.figure(figsize=(6,6))
    sub = fig.add_subplot(111)
    i = 5
    _plt0, = sub.plot(Nstds, sig_std[:,i], c='C0', ls='-', label=lbls[i]) 
    _plt1, = sub.plot(Nstds, sig_pfd[:,i], c='C0', ls='-.') 
    leg1 = sub.legend([_plt0, _plt1], ['standard', 'paired-fixed'], fontsize=20, loc='upper left') 
    #sub.legend(loc='lower right', fontsize=15) 
    plt.gca().add_artist(leg1)
    
    sub.set_xlabel(r'$N_{\rm deriv}$', fontsize=25) 
    sub.set_xlim(Nstds[0], Nstd) 
    sub.set_ylabel(r'$\sigma_{M_\nu}(N_{\rm deriv})/\sigma_{M_\nu}$', fontsize=25) 
    #sub.set_ylim(0.5, 1.1) 
    if rsd == 'real': 
        ffig = os.path.join(dir_fig, 'pf_P_sigma_convergence.real.kmax%.1f.png' % kmax)
    else: 
        ffig = os.path.join(dir_fig, 'pf_P_sigma_convergence.rsd.kmax%.1f.png' % kmax)
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def pf_B_sigma_convergence(rsd='all', kmax=0.5): 
    ''' compare the fisher matrix derived from paired-fixed derivatives vs 
    standard n-body derivatives
    '''
    if rsd not in ['real', 'all']: raise ValueError
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
    theta_lims  = [(0.3, 0.335), (0.04, 0.058), (0.6, 0.74), (0.9, 1.015), (0.808, 0.86), (-0.25, 0.25)]
    lbls        = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$']

    dbdts_std, dbdts_pfd = [], [] 
    for theta in thetas:
        #if rsd == 'real': 
        #    # real-space d logB / d theta (standard)
        #    _, _, _, dbdt_std = Forecast.quijote_dBkdtheta(theta, rsd='real', flag='reg', dmnu='fin_2lpt')
        #    # real-space d logP / d theta (paired-fixed)
        #    _, _, _, dbdt_pfd = Forecast.quijote_dBkdtheta(theta, rsd='real', flag='ncv', dmnu='fin_2lpt')
        #else: 
        #    # redshift-space d logP0 / d theta (standard)
        #    _, _, _, dbdt_std = Forecast.quijote_dBkdtheta(theta, rsd=rsd, flag='reg', dmnu='fin_2lpt')
        #    # redshift-space d logP0 / d theta (paired-fixed)
        #    _, _, _, dbdt_pfd = Forecast.quijote_dBkdtheta(theta, rsd=rsd, flag='ncv', dmnu='fin_2lpt')
        #dbdts_std.append(dbdt_std[klim]) 
        #dbdts_pfd.append(dbdt_pfd[klim]) 

        # d logP / d theta (standard)
        _, _, _, dbdt_std = X_std('dbk', theta, rsd=rsd, dmnu='fin_2lpt', silent=False) 
        # d logP / d theta (paired-fixed)
        _, _, _, dbdt_pfd0 = X_pfd_1('dbk', theta, rsd=rsd, dmnu='fin_2lpt', silent=False) 
        _, _, _, dbdt_pfd1 = X_pfd_2('dbk', theta, rsd=rsd, dmnu='fin_2lpt', silent=False) 
        dbdt_pfd = 0.5 * (dbdt_pfd0 + dbdt_pfd1) 

        dbdts_std.append(dbdt_std[:,klim]) 
        dbdts_pfd.append(dbdt_pfd[:,klim]) 
        
        Nstd = dbdt_std.shape[0]
        Npfd = dbdt_pfd.shape[0]

    if Nstd == 1500: 
        Nstds = [200, 400, 600, 800, 1000, 1200, 1500]
    elif Nstd == 500:
        Nstds = [100, 200, 300, 400, 500]
    
    sig_std, sig_pfd = [], [] 
    for _Nstd in Nstds: 
        _dbdts_std = [np.average(dbdt[:_Nstd], axis=0) for dbdt in dbdts_std]
        _dbdts_pfd = [np.average(dbdt[:_Nstd/2], axis=0) for dbdt in dbdts_pfd]

        Fij_std = Forecast.Fij(_dbdts_std, C_inv) 
        Fij_pfd = Forecast.Fij(_dbdts_pfd, C_inv) 

        Finv_std = np.linalg.inv(Fij_std) # invert fisher matrix 
        Finv_pfd = np.linalg.inv(Fij_pfd) # invert fisher matrix 

        _sig_std = np.sqrt(np.diag(Finv_std))
        _sig_pfd = np.sqrt(np.diag(Finv_pfd))

        sig_std.append(_sig_std) 
        sig_pfd.append(_sig_pfd) 

    sig_std = np.array(sig_std)#/sig_std[-1]
    sig_pfd = np.array(sig_pfd)#/sig_pfd[-1]
    
    fig = plt.figure(figsize=(6,6))
    sub = fig.add_subplot(111)
    i = 5
    _plt0, = sub.plot(Nstds, sig_std[:,i], c='C0', ls='-', label=lbls[i]) 
    _plt1, = sub.plot(Nstds, sig_pfd[:,i], c='C0', ls='-.') 
    leg1 = sub.legend([_plt0, _plt1], ['standard', 'paired-fixed'], fontsize=20, loc='upper left') 
    #sub.legend(loc='lower right', fontsize=15) 
    plt.gca().add_artist(leg1)
    
    sub.set_xlabel(r'$N_{\rm deriv}$', fontsize=25) 
    sub.set_xlim(Nstds[0], Nstd) 
    sub.set_ylabel(r'$\sigma_{M_\nu}(N_{\rm deriv})$ eV', fontsize=25) 
    #sub.set_ylim(0.5, 1.1) 
    if rsd == 'real': 
        ffig = os.path.join(dir_fig, 'pf_B_sigma_convergence.real.kmax%.1f.png' % kmax)
    else: 
        ffig = os.path.join(dir_fig, 'pf_B_sigma_convergence.rsd.kmax%.1f.png' % kmax)
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def _pf_B_posterior_noh(rsd=0, kmax=0.5): 
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
    
    thetas      = ['Om', 'Ob2', 'ns', 's8']#, 'Mnu']
    theta_fid   = [0.3175, 0.049, 0.9624, 0.834]#, 0.0] # fiducial theta 
    theta_lims  = [(0.3, 0.335), (0.04, 0.058), (0.9, 1.015), (0.808, 0.86)]#, (-0.25, 0.25)]
    lbls        = [r'$\Omega_m$', r'$\Omega_b$', r'$n_s$', r'$\sigma_8$']#, r'$M_\nu$']

    dbdts_std, dbdts_pfd = [], [] 
    for theta in thetas:
        if rsd == 'real': 
            # real-space d logB / d theta (standard)
            _, _, _, dbdt_std = X_std('dbk', theta, rsd='real', silent=False) 
            # real-space d logP / d theta (paired-fixed)
            _, _, _, dbdt_pfd0 = X_pfd_1('dbk', theta, rsd='real', silent=False) 
            _, _, _, dbdt_pfd1 = X_pfd_2('dbk', theta, rsd='real', silent=False) 
            dbdt_pfd = 0.5 * (dbdt_pfd0 + dbdt_pfd1) 
        else: 
            # redshift-space d logP0 / d theta (standard)
            _, _, _, dbdt_std = X_std('dbk', theta, rsd=rsd, silent=False) 
            # redshift-space d logP0 / d theta (paired-fixed)
            _, _, _, dbdt_pfd0 = X_pfd_1('dbk', theta, rsd=rsd, silent=False) 
            _, _, _, dbdt_pfd1 = X_pfd_2('dbk', theta, rsd=rsd, silent=False) 
            dbdt_pfd = 0.5 * (dbdt_pfd0 + dbdt_pfd1) 
        dbdts_std.append(np.average(dbdt_std, axis=0)[klim]) 
        dbdts_pfd.append(np.average(dbdt_pfd, axis=0)[klim]) 

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
    if rsd == 'real': 
        ffig = os.path.join(dir_fig, 'pf_B_posterior_noh.real.kmax%.1f.png' % kmax)
    else: 
        ffig = os.path.join(dir_fig, 'pf_B_posterior_noh.rsd.kmax%.1f.png' % kmax)
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') 
    return None 

############################################################
# etc 
############################################################
def X_std(obs, theta, rsd=2, dmnu='fin', silent=True): 
    ''' read in standard N-body observables 
    '''
    if obs == 'pk': 
        quij = Obvs.quijotePk(theta, rsd=rsd, flag='reg', silent=silent) 
        if rsd != 'real': # redshift-space 
            return quij['k'], quij['p0k'], quij['p2k']
        else: # real-space
            return quij['k'], quij['p0k']
    elif obs == 'bk': 
        quij = Obvs.quijoteBk(theta, rsd=rsd, flag='reg', silent=silent) 
        return quij['k1'], quij['k2'], quij['k3'], quij['b123']  
    elif obs == 'dlogpk': 
        if rsd != 'real': 
            k, dpdt_std = Forecast.quijote_dP02kdtheta(theta, log=True, rsd=rsd, flag='reg', dmnu=dmnu, average=False) 
            return k[:len(k)/2], dpdt_std[:,:len(k)/2], dpdt_std[:,len(k)/2:]
        else: 
            k, dpdt_std = Forecast.quijote_dPkdtheta(theta, log=True, rsd=rsd, flag='reg', dmnu=dmnu, average=False) 
            return k, dpdt_std
    elif obs == 'dlogbk': 
        return Forecast.quijote_dBkdtheta(theta, log=True, rsd=rsd, flag='reg', dmnu=dmnu, average=False)
    elif obs == 'dpk': 
        if rsd != 'real': 
            k, dpdt_std = Forecast.quijote_dP02kdtheta(theta, rsd=rsd, flag='reg', dmnu=dmnu, average=False) 
            return k[:len(k)/2], dpdt_std[:,:len(k)/2], dpdt_std[:,len(k)/2:]
        else: 
            k, dpdt_std = Forecast.quijote_dPkdtheta(theta, rsd=rsd, flag='reg', dmnu=dmnu, average=False) 
            return k, dpdt_std
    elif obs == 'dbk': 
        i_k, j_k, l_k, dbdt = Forecast.quijote_dBkdtheta(theta, rsd=rsd, flag='reg', dmnu=dmnu, average=False)
        return i_k, j_k, l_k, dbdt
    else: 
        raise ValueError


def X_pfd_1(obs, theta, rsd=2, dmnu='fin', silent=True): 
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
        quij = Obvs.quijoteBk(theta, rsd=rsd, flag='ncv', silent=silent) 
        return quij['k1'], quij['k2'], quij['k3'], quij['b123'][::2,:]
    elif obs == 'dlogpk': 
        if rsd != 'real': 
            k, dpdt_std = Forecast.quijote_dP02kdtheta(theta, log=True, rsd=rsd, flag='ncv', dmnu=dmnu, average=False) 
            return k[:len(k)/2], dpdt_std[::2,:len(k)/2], dpdt_std[::2,len(k)/2:]
        else: 
            k, dpdt_std = Forecast.quijote_dPkdtheta(theta, log=True, rsd=rsd, flag='ncv', dmnu=dmnu, average=False) 
            return k, dpdt_std[::2,:]
    elif obs == 'dlogbk': 
        i_k, j_k, l_k, dbdt = Forecast.quijote_dBkdtheta(theta, log=True, rsd=rsd, flag='ncv', dmnu=dmnu, average=False)
        return i_k, j_k, l_k, dbdt[::2,:]
    elif obs == 'dpk': 
        if rsd != 'real': 
            k, dpdt_std = Forecast.quijote_dP02kdtheta(theta, rsd=rsd, flag='ncv', dmnu=dmnu, average=False) 
            return k[:len(k)/2], dpdt_std[::2,:len(k)/2], dpdt_std[::2,len(k)/2:]
        else: 
            k, dpdt_std = Forecast.quijote_dPkdtheta(theta, rsd=rsd, flag='ncv', dmnu=dmnu, average=False) 
            return k, dpdt_std[::2,:]
    elif obs == 'dbk': 
        i_k, j_k, l_k, dbdt = Forecast.quijote_dBkdtheta(theta, rsd=rsd, flag='ncv', dmnu=dmnu, average=False)
        return i_k, j_k, l_k, dbdt[::2,:]
    else: 
        raise ValueError


def X_pfd_2(obs, theta, rsd=2, dmnu='fin', silent=True): 
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
        quij = Obvs.quijoteBk(theta, rsd=rsd, flag='ncv', silent=silent) 
        return quij['k1'], quij['k2'], quij['k3'], quij['b123'][1::2,:]
    elif obs == 'dlogpk': 
        if rsd != 'real': 
            k, dpdt_std = Forecast.quijote_dP02kdtheta(theta, log=True, rsd=rsd, flag='ncv', dmnu=dmnu, average=False) 
            return k[:len(k)/2], dpdt_std[1::2,:len(k)/2], dpdt_std[1::2,len(k)/2:]
        else: 
            k, dpdt_std = Forecast.quijote_dPkdtheta(theta, log=True, rsd=rsd, flag='ncv', dmnu=dmnu, average=False) 
            return k, dpdt_std[1::2,:]
    elif obs == 'dlogbk': 
        i_k, j_k, l_k, dbdt = Forecast.quijote_dBkdtheta(theta, log=True, rsd=rsd, flag='ncv', dmnu=dmnu, average=False)
        return i_k, j_k, l_k, dbdt[1::2,:]
    elif obs == 'dpk': 
        if rsd != 'real': 
            k, dpdt_std = Forecast.quijote_dP02kdtheta(theta, rsd=rsd, flag='ncv', dmnu=dmnu, average=False) 
            return k[:len(k)/2], dpdt_std[1::2,:len(k)/2], dpdt_std[1::2,len(k)/2:]
        else: 
            k, dpdt_std = Forecast.quijote_dPkdtheta(theta, rsd=rsd, flag='ncv', dmnu=dmnu, average=False) 
            return k, dpdt_std[1::2,:]
    elif obs == 'dbk': 
        i_k, j_k, l_k, dbdt = Forecast.quijote_dBkdtheta(theta, rsd=rsd, flag='ncv', dmnu=dmnu, average=False)
        return i_k, j_k, l_k, dbdt[1::2,:]
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


def sigma_ratio(X_std, X_pfd): 
    ''' calculate sigma_s / sigma_pfd 
    '''
    sig_std = np.std(X_std, axis=0) 
    sig_pfd = np.std(X_pfd, axis=0) 
    return sig_std / sig_pfd 


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
    if rsd == 'all': return '.rsd'
    elif rsd in [0, 1, 2]: return '.rsd%i' % rsd
    elif rsd == 'real': return '.real'
    else: raise NotImplementedError


def _flag_str(flag): 
    # assign string based on flag kwarg
    return ['.%s' % flag, ''][flag is None]



if __name__=="__main__": 
    #pf_Pk()
    #pf_Bk(kmax=0.5)
    #Pk_bias()
    #Bk_bias()
    #pf_dPdtheta()
    #pf_dBdtheta(kmax=0.5)
    #dPdtheta_bias(kmax=0.5)
    #dBdtheta_bias(kmax=0.5)
    #dBdtheta_chi2(kmax=0.2)
    #pf_dlogPdtheta()
    #pf_dlogBdtheta(kmax=0.5)
    #pf_P_Fij(rsd='real', kmax=0.5)
    #pf_P_Fij(rsd=0, kmax=0.5)
    #pf_P_Fij(rsd=1, kmax=0.5)
    #pf_P_Fij(rsd=2, kmax=0.5)
    #pf_P_Fij(rsd='all', kmax=0.5)
    #pf_B_Fij(rsd='real', kmax=0.5)
    #pf_B_Fij(rsd='all', kmax=0.5)

    #pf_P_posterior(rsd='real', kmax=0.5)
    #pf_P_posterior(rsd='all', kmax=0.5)

    #pf_B_posterior(rsd='real', kmax=0.5)
    #pf_B_posterior(rsd='all', kmax=0.5)
    
    #pf_P_sigma_convergence(rsd='real', kmax=0.5)
    #pf_P_sigma_convergence(rsd='all', kmax=0.5)
    
    pf_B_sigma_convergence(rsd='real', kmax=0.5)
    pf_B_sigma_convergence(rsd='all', kmax=0.5)

