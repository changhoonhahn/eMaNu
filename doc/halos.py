'''

figures looking at hades halo catalog with massive neutrinos


'''
import os 
import h5py
import numpy as np 
# --- eMaNu --- 
from emanu import util as UT
from emanu import obvs as Obvs
from emanu import plots as ePlots
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


##################################################################
# powerspectrum comparison 
##################################################################
def compare_Plk(nreals=range(1,71), krange=[0.03, 0.25]): 
    mnus = [0.0, 0.06, 0.1, 0.15]
    sig8s = [0.822, 0.818, 0.807, 0.798]
    kmin, kmax = krange 

    # read in all the powerspectrum
    p0ks, p2ks, p4ks = [], [], [] 
    for mnu in mnus: 
        k, p0k, p2k, p4k = readPlk(mnu, nreals, 4, zspace=True)
        p0ks.append(p0k)
        p2ks.append(p2k)
        p4ks.append(p4k)

    p0k_s8s, p2k_s8s, p4k_s8s = [], [], [] 
    for sig8 in sig8s: 
        k, p0k, p2k, p4k = readPlk_sigma8(sig8, nreals, 4, zspace=True)
        p0k_s8s.append(p0k) 
        p2k_s8s.append(p2k) 
        p4k_s8s.append(p4k) 

    sig_p0k, sig_p2k, sig_p4k = sigma_Plk(nreals, 4, zspace=True)

    klim = ((k <= kmax) & (k >= kmin))

    fig = plt.figure(figsize=(18,8))
    gs = mpl.gridspec.GridSpec(2,5, figure=fig) 
    sub = plt.subplot(gs[0,:-2]) 
    axins = inset_axes(sub, loc='upper right', width="35%", height="50%") 
    ii = 0 
    for mnu, p0k, p2k, p4k in zip(mnus, p0ks, p2ks, p4ks):
        sub.plot(k[klim], k[klim] * p0k[klim], c='C'+str(ii), label=str(mnu)+'eV') 
        axins.plot(k[klim], k[klim] * p0k[klim], c='C'+str(ii))
        ii += 1 
    
    for sig8, p0k, p2k, p4k in zip(sig8s, p0k_s8s, p2k_s8s, p4k_s8s):
        if ii < 10: 
            sub.plot(k[klim], k[klim] * p0k[klim], ls=':', c='C'+str(ii)) 
            axins.plot(k[klim], k[klim] * p0k[klim], ls=':', c='C'+str(ii))
        else: 
            sub.plot(k[klim], k[klim] * p0k[klim], ls=':', c='C9') 
            axins.plot(k[klim], k[klim] * p0k[klim], ls=':', c='C9')
        ii += 2 
    axins.set_xlim(0.15, 0.3)
    axins.set_ylim(1200, 1700) 
    axins.set_xticklabels('') 
    axins.set_yticklabels('') 
    #axins.yaxis.set_minor_formatter(NullFormatter())
    mark_inset(sub, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    sub.legend(loc='lower left', ncol=2, handletextpad=0.25, columnspacing=0.5, fontsize=18) 
    #sub.set_xlabel("$k$ [$h$/Mpc]", fontsize=25) 
    #sub.set_xscale('log') 
    sub.set_xlim([1e-2, 0.5])
    sub.set_ylim([500, 2700]) 
    sub.set_ylabel('$k\, P_0(k)$', fontsize=25) 

    sub = plt.subplot(gs[1,:-2]) 
    axins = inset_axes(sub, loc='upper right', width="35%", height="50%") 
    ii = 0 
    for mnu, p0k, p2k, p4k in zip(mnus, p0ks, p2ks, p4ks):
        sub.plot(k[klim], k[klim] * p2k[klim], c='C'+str(ii)) 
        axins.plot(k[klim], k[klim] * p2k[klim], c='C'+str(ii))
        ii += 1 
    
    for sig8, p0k, p2k, p4k in zip(sig8s, p0k_s8s, p2k_s8s, p4k_s8s):
        if ii < 10: 
            sub.plot(k[klim], k[klim] * p2k[klim], ls=':', c='C'+str(ii), label='$\sigma_8=$'+str(sig8)) 
            axins.plot(k[klim], k[klim] * p2k[klim], ls=':', c='C'+str(ii))
        else: 
            sub.plot(k[klim], k[klim] * p2k[klim], ls=':', c='C9', label='$\sigma_8=$'+str(sig8)) 
            axins.plot(k[klim], k[klim] * p2k[klim], ls=':', c='C9')
        ii += 2 
    axins.set_xlim(0.15, 0.3)
    axins.set_ylim(640, 840) 
    axins.set_xticklabels('') 
    axins.set_yticklabels('') 
    mark_inset(sub, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    sub.legend(loc='lower left', ncol=2, handletextpad=0.25, columnspacing=0.5, fontsize=18) 
    sub.set_xlabel("$k$ [$h$/Mpc]", fontsize=25) 
    sub.set_xlim([1e-2, 0.5])
    sub.set_ylim([300, 1300]) 
    sub.set_ylabel('$k\, P_2(k)$', fontsize=25) 

    sub = plt.subplot(gs[0,-2:]) 
    ii = 0 
    for mnu, p0k in zip(mnus, p0ks): 
        if mnu == 0.:  
            p0k_fid = p0k
        else: 
            sub.plot(k[klim], p0k[klim]/p0k_fid[klim], c='C'+str(ii), label=str(mnu)+'eV') 
        ii += 1 
    
    for sig8, p0k in zip(sig8s, p0k_s8s):
        if ii < 10: 
            sub.plot(k[klim], p0k[klim]/p0k_fid[klim], ls='-', lw=0.5, c='k') 
            sub.plot(k[klim], p0k[klim]/p0k_fid[klim], ls=':', c='C'+str(ii)) 
        else: 
            sub.plot(k[klim], p0k[klim]/p0k_fid[klim], ls='-', lw=0.5, c='k') 
            sub.plot(k[klim], p0k[klim]/p0k_fid[klim], ls=':', c='C9') 
        ii += 2 
    sub.fill_between(k, np.ones(len(k)), np.ones(len(k))+sig_p0k/p0k_fid, 
            color='k', alpha=0.2, linewidth=0) 
    sub.plot([1e-4, 10.], [1., 1.], c='k', ls='--') 
    #sub.legend(loc='upper left', ncol=2, handletextpad=0.25, columnspacing=0.5, fontsize=18) 
    sub.set_xscale("log") 
    sub.set_xlim([1e-2, 0.5])
    sub.set_ylim([0.98, 1.1]) 
    sub.set_yticks([1., 1.05, 1.1]) 
    sub.set_ylabel('$P_0(k)/P_0^\mathrm{(fid)}(k)$', fontsize=25) 

    sub = plt.subplot(gs[1,-2:]) 
    ii = 0 
    for mnu, p2k in zip(mnus, p2ks):
        if mnu == 0.: 
            p2k_fid = p2k
        else: 
            sub.plot(k[klim], p2k[klim]/p2k_fid[klim], c='C'+str(ii)) 
        ii += 1 
    
    for sig8, p2k in zip(sig8s, p2k_s8s):
        if ii < 10: 
            sub.plot(k[klim], p2k[klim]/p2k_fid[klim], ls='-', lw=0.5, c='k') 
            sub.plot(k[klim], p2k[klim]/p2k_fid[klim], ls=':', c='C'+str(ii), label='$\sigma_8=$'+str(sig8)) 
        else: 
            sub.plot(k[klim], p2k[klim]/p2k_fid[klim], ls='-', lw=0.5, c='k') 
            sub.plot(k[klim], p2k[klim]/p2k_fid[klim], ls=':', c='C9', label='$\sigma_8=$'+str(sig8)) 
        ii += 2 
    sub.fill_between(k, np.ones(len(k)), np.ones(len(k))+sig_p2k/p2k_fid, 
            color='k', alpha=0.2, linewidth=0) 
    sub.plot([1e-4, 10.], [1., 1.], c='k', ls='--') 
    #sub.legend(loc='upper left', ncol=2, handletextpad=0.25, columnspacing=0.5, fontsize=18) 
    sub.set_xlabel("$k$ [$h$/Mpc]", fontsize=25) 
    sub.set_xscale("log") 
    sub.set_xlim([1e-2, 0.5])
    sub.set_ylim([0.98, 1.1]) 
    sub.set_yticks([1., 1.05, 1.1]) 
    sub.set_ylabel('$P_2(k)/P_2^\mathrm{(fid)}(k)$', fontsize=25) 
    fig.subplots_adjust(wspace=0.7, hspace=0.125)
    fig.savefig(''.join([UT.doc_dir(), 'figs/haloPlk_rsd.pdf']), bbox_inches='tight') 
    return None


def ratio_Plk(nreals=range(1,101), krange=[0.03, 0.25]): 
    mnus = [0.0, 0.06, 0.1, 0.15]
    sig8s = [0.822, 0.818, 0.807, 0.798]
    kmin, kmax = krange 

    # read in all the powerspectrum
    p0ks, p2ks, p4ks = [], [], [] 
    for mnu in mnus: 
        k, p0k, p2k, p4k = readPlk(mnu, nreals, 4, zspace=True)
        p0ks.append(p0k)
        p2ks.append(p2k)
        p4ks.append(p4k)

    p0k_s8s, p2k_s8s, p4k_s8s = [], [], [] 
    for sig8 in sig8s: 
        k, p0k, p2k, p4k = readPlk_sigma8(sig8, nreals, 4, zspace=True)
        p0k_s8s.append(p0k) 
        p2k_s8s.append(p2k) 
        p4k_s8s.append(p4k) 

    #sig_p0k, sig_p2k, sig_p4k = sigma_Plk(nreals, 4, zspace=True)

    klim = ((k <= kmax) & (k >= kmin))

    fig = plt.figure(figsize=(15,5))
    gs = mpl.gridspec.GridSpec(1,2, figure=fig) 
    sub = plt.subplot(gs[0,0]) 
    ii = 0 
    for mnu, p0k in zip(mnus, p0ks): 
        if mnu == 0.:  
            p0k_fid = p0k
        else: 
            sub.plot(k[klim], p0k[klim]/p0k_fid[klim], c='C'+str(ii), label=str(mnu)+'eV') 
        ii += 1 
    lstyles = ['-', (0, (5, 7)), (0, (3, 5, 1, 5)), (0, (1, 5)) ] 
    for sig8, p0k, ls in zip(sig8s, p0k_s8s, lstyles):
        sub.plot(k[klim], p0k[klim]/p0k_fid[klim], ls=ls, lw=0.9, c='k') 
    #sub.fill_between(k, np.ones(len(k)), np.ones(len(k))+sig_p0k/p0k_fid, color='k', alpha=0.2, linewidth=0) 
    sub.plot([1e-4, 10.], [1., 1.], c='k', ls='--') 
    sub.legend(loc='upper left', ncol=1, handletextpad=0.25, columnspacing=0.5, fontsize=18) 
    sub.set_xscale("log") 
    sub.set_xlim([2e-2, 0.5])
    sub.set_xticks([2e-2, 1e-1, 0.5]) 
    sub.set_xticklabels([r'0.02', '0.1', r'0.5'])
    sub.set_ylim([0.98, 1.1]) 
    sub.set_yticks([1., 1.05, 1.1]) 
    sub.set_ylabel('$P_0(k)/P_0^\mathrm{fid}(k)$', fontsize=25) 

    sub = plt.subplot(gs[0,1]) 
    ii = 0 
    for mnu, p2k in zip(mnus, p2ks):
        if mnu == 0.: 
            p2k_fid = p2k
        else: 
            sub.plot(k[klim], p2k[klim]/p2k_fid[klim], c='C'+str(ii)) 
        ii += 1 
    
    for sig8, p2k, ls in zip(sig8s, p2k_s8s, lstyles):
        sub.plot(k[klim], p2k[klim]/p2k_fid[klim], ls=ls, lw=0.9, c='k', label='$\sigma_8=$'+str(sig8)) 
    #sub.fill_between(k, np.ones(len(k)), np.ones(len(k))+sig_p2k/p2k_fid, color='k', alpha=0.2, linewidth=0) 
    sub.plot([1e-4, 10.], [1., 1.], c='k', ls='--') 
    sub.legend(loc='upper left', ncol=1, handletextpad=0.25, columnspacing=0.5, fontsize=18) 
    #sub.set_xlabel("$k$ [$h$/Mpc]", fontsize=25) 
    sub.set_xscale("log") 
    sub.set_xlim([2e-2, 0.5])
    sub.set_xticks([2e-2, 1e-1, 0.5]) 
    sub.set_xticklabels([r'0.02', '0.1', r'0.5'])
    sub.set_ylim([0.98, 1.1]) 
    sub.set_yticks([1., 1.05, 1.1]) 
    sub.set_ylabel('$P_2(k)/P_2^\mathrm{fid}(k)$', fontsize=25) 
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel("$k$ [$h$/Mpc]", labelpad=10, fontsize=25) 
    fig.subplots_adjust(wspace=0.25)
    fig.savefig(''.join([UT.doc_dir(), 'figs/haloPlk_rsd_ratio.pdf']), bbox_inches='tight') 
    return None


def sigma_Plk(i, nzbin, zspace=False):
    '''diagonal element of the covariance matrix 
    '''
    pk_kwargs = {
            'zspace': zspace, 
            'mh_min': 3200., 
            'Ngrid': 360}

    k, p0k, p2k, p4k = [], [], [], []
    for ii, _i in enumerate(i):
        plk_i = Obvs.Plk_halo(0.0, _i, nzbin, **pk_kwargs)
        p0k.append(plk_i['p0k'])
        p2k.append(plk_i['p2k'])
        p4k.append(plk_i['p4k'])
    sig_p0k = np.std(p0k, axis=0)
    sig_p2k = np.std(p2k, axis=0)
    sig_p4k = np.std(p4k, axis=0)
    return sig_p0k, sig_p2k, sig_p4k


def readPlk(mneut, i, nzbin, zspace=False):
    ''' read in bispectrum of massive neutrino halo catalogs
    using the function Obvs.B123_halo
    '''
    pk_kwargs = {
            'zspace': zspace, 
            'mh_min': 3200., 
            'Ngrid': 360, 
            'Nbin': 120
            }

    if isinstance(i, int):
        plk_i = Obvs.Plk_halo(mneut, i, nzbin, **pk_kwargs)
        k = plk_i['k'] 
        p0k = plk_i['p0k'] 
        p2k = plk_i['p2k'] 
        p4k = plk_i['p4k'] 
    elif isinstance(i, (list, np.ndarray)):
        k, p0k, p2k, p4k = [], [], [], []
        for ii, _i in enumerate(i):
            plk_i = Obvs.Plk_halo(mneut, _i, nzbin, **pk_kwargs)
            k.append(plk_i['k'])
            p0k.append(plk_i['p0k'])
            p2k.append(plk_i['p2k'])
            p4k.append(plk_i['p4k'])
        k = np.average(k, axis=0)
        p0k = np.average(p0k, axis=0)
        p2k = np.average(p2k, axis=0)
        p4k = np.average(p4k, axis=0)
    return k, p0k, p2k, p4k


def readPlk_sigma8(sig8, i, nzbin, zspace=False):
    ''' read in bispectrum of massive neutrino halo catalogs
    using the function Obvs.B123_halo
    '''
    pk_kwargs = {
            'zspace': zspace, 
            'mh_min': 3200., 
            'Ngrid': 360, 
            'Nbin': 120
            }

    if isinstance(i, int):
        plk_i = Obvs.Plk_halo_sigma8(sig8, i, nzbin, **pk_kwargs)
        k = plk_i['k'] 
        p0k = plk_i['p0k'] 
        p2k = plk_i['p2k'] 
        p4k = plk_i['p4k'] 
    elif isinstance(i, (list, np.ndarray)):
        k, p0k, p2k, p4k = [], [], [], []
        for ii, _i in enumerate(i):
            plk_i = Obvs.Plk_halo_sigma8(sig8, _i, nzbin, **pk_kwargs)
            k.append(plk_i['k'])
            p0k.append(plk_i['p0k'])
            p2k.append(plk_i['p2k'])
            p4k.append(plk_i['p4k'])
        k = np.average(k, axis=0)
        p0k = np.average(p0k, axis=0)
        p2k = np.average(p2k, axis=0)
        p4k = np.average(p4k, axis=0)
    return k, p0k, p2k, p4k

##################################################################
# bispectrum comparison 
##################################################################
def compare_B123(typ, nreals=range(1,71), krange=[0.03, 0.25], nbin=50, zspace=False): 
    ''' Make various bispectrum plots as a function of m_nu 
    '''
    str_rsd = ''
    if zspace: str_rsd = '_rsd'
    mnus = [0.06, 0.1, 0.15]
    sig8s = [0.822, 0.818, 0.807, 0.798]
    kmin, kmax = krange 
    krange_str = str(kmin).replace('.', '')+'_'+str(kmax).replace('.', '') 

    _, _, _, B123_fid, cnts_fid, _ = readB123(0.0, nreals, 4, BorQ='B', zspace=zspace)

    B123s, cnts = [], [] 
    for mnu in mnus: 
        _i, _j, _l, _B123, _cnts, kf = readB123(mnu, nreals, 4, BorQ='B', zspace=zspace)
        B123s.append(_B123) 
        cnts.append(_cnts)

    B123_s8s, cnt_s8s = [], [] 
    for sig8 in sig8s: 
        _i, _j, _l, _B123, _cnts, kf = readB123_sigma8(sig8, nreals, 4, BorQ='B', zspace=zspace)
        B123_s8s.append(_B123) 
        cnt_s8s.append(_cnts)
        
    i_k, j_k, l_k = _i, _j, _l
    k3k1 = l_k.astype(float)/i_k.astype(float)
    k2k1 = j_k.astype(float)/i_k.astype(float)
    klim = ((i_k*kf <= kmax) & (i_k*kf >= kmin) &
            (j_k*kf <= kmax) & (j_k*kf >= kmin) & 
            (l_k*kf <= kmax) & (l_k*kf >= kmin)) 

    if typ == 'b_shape': 
        x_bins = np.linspace(0., 1., int(nbin)+1)
        y_bins = np.linspace(0.5, 1., int(0.5*nbin)+1)
        if zspace: norm = LogNorm(vmin=5e6, vmax=1e9)
        else: norm = SymLogNorm(vmin=-5e6, vmax=1e9, linthresh=1e6, linscale=1.)

        fig = plt.figure(figsize=(25,6))
        for i, mnu, B123_i, cnts_i in zip(range(len(mnus)+1), [0.0]+mnus, [B123_fid]+B123s, [cnts_fid]+cnts):
            BQgrid = ePlots._BorQgrid(k3k1[klim], k2k1[klim], B123_i[klim], cnts_i[klim], x_bins, y_bins)
            sub = fig.add_subplot(2,4,i+1)
            bplot = sub.pcolormesh(x_bins, y_bins, BQgrid.T, norm=norm, cmap='RdBu')
            sub.text(0.05, 0.05, str(mnu)+'eV', ha='left', va='bottom', 
                    transform=sub.transAxes, fontsize=20)
            if i > 0: 
                sub.set_xticklabels([]) 
                sub.set_yticklabels([]) 

        for i in range(3): 
            B123_i, cnts_i = B123_s8s[i], cnt_s8s[i]
            BQgrid = ePlots._BorQgrid(k3k1[klim], k2k1[klim], B123_i[klim], cnts_i[klim], x_bins, y_bins)
            sub = fig.add_subplot(2,4,i+6)
            bplot = sub.pcolormesh(x_bins, y_bins, BQgrid.T, norm=norm, cmap='RdBu')
            sub.text(0.05, 0.05, '0.0eV', ha='left', va='bottom', 
                    transform=sub.transAxes, fontsize=20)
            sub.text(0.975, 0.025, '$\sigma_8$='+str(round(sig8s[i],3)), ha='right', va='bottom', 
                    transform=sub.transAxes, fontsize=20)
            if i > 1: 
                sub.set_yticklabels([]) 

        bkgd = fig.add_subplot(111, frameon=False)
        bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        bkgd.set_xlabel('$k_3/k_1$', labelpad=10, fontsize=25)
        bkgd.set_ylabel('$k_2/k_1$', labelpad=5, fontsize=25)
        fig.subplots_adjust(wspace=0.15, hspace=0.2, right=0.935)
        cbar_ax = fig.add_axes([0.95, 0.15, 0.0125, 0.7])
        cbar = fig.colorbar(bplot, cax=cbar_ax)
        cbar.set_label('$B(k_1, k_2, k_3)$', rotation=90, fontsize=20)
        if not rsd: cbar.set_ticks([-1e6, 0., 1e6, 1e7, 1e8, 1e9]) 
        #cbar_ax = fig.add_axes([0.95, 0.125, 0.0125, 0.35])
        #cbar = fig.colorbar(dbplot, cax=cbar_ax)
        #cbar.set_label('$B(k_1, k_2, k_3) - B^\mathrm{(fid)}$', rotation=90, fontsize=20)
        fig.savefig(''.join([UT.doc_dir(), 'figs/haloB123_shape', '_', krange_str, str_rsd, '.pdf']), bbox_inches='tight') 

    elif typ == 'relative_shape':
        x_bins = np.linspace(0., 1., int(nbin)+1)
        y_bins = np.linspace(0.5, 1., int(0.5*nbin)+1)
        
        BQgrid_fid = ePlots._BorQgrid(k3k1[klim], k2k1[klim], B123_fid[klim], cnts[0][klim], x_bins, y_bins)

        fig = plt.figure(figsize=(25,6))
        for i in range(len(mnus)):
            B123_i, cnts_i = B123s[i], cnts[i]
            dB123 = B123_i - B123_fid
            dBQgrid = ePlots._BorQgrid(k3k1[klim], k2k1[klim], dB123[klim], cnts_i[klim], x_bins, y_bins)

            sub = fig.add_subplot(2,4,i+1)
            if rsd: 
                bplot = sub.pcolormesh(x_bins, y_bins, dBQgrid.T/BQgrid_fid.T, vmin=0., vmax=0.15, cmap='RdBu')
            else: 
                bplot = sub.pcolormesh(x_bins, y_bins, dBQgrid.T/BQgrid_fid.T, vmin=-0.1, vmax=0.15, cmap='RdBu')
            sub.text(0.05, 0.05, str(mnus[i])+'eV', ha='left', va='bottom', 
                    transform=sub.transAxes, fontsize=20)
            sub.set_xticklabels([]) 
            if i > 0: 
                sub.set_yticklabels([]) 

        for i in range(len(sig8s)): 
            B123_i, cnts_i = B123_s8s[i], cnt_s8s[i]
            dB123 = B123_i - B123_fid
            dBQgrid = ePlots._BorQgrid(k3k1[klim], k2k1[klim], dB123[klim], cnts_i[klim], x_bins, y_bins)
            sub = fig.add_subplot(2,4,i+5)
            if rsd: 
                bplot = sub.pcolormesh(x_bins, y_bins, dBQgrid.T/BQgrid_fid.T, vmin=0., vmax=0.15, cmap='RdBu')
            else: 
                bplot = sub.pcolormesh(x_bins, y_bins, dBQgrid.T/BQgrid_fid.T, vmin=-0.1, vmax=0.15, cmap='RdBu')
            sub.text(0.05, 0.05, '0.0eV', ha='left', va='bottom', 
                    transform=sub.transAxes, fontsize=20)
            sub.text(0.975, 0.025, '$\sigma_8$='+str(round(sig8s[i],3)), ha='right', va='bottom', 
                    transform=sub.transAxes, fontsize=20)
            if i > 0: 
                sub.set_yticklabels([]) 
                sub.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0]) 

        bkgd = fig.add_subplot(111, frameon=False)
        bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        bkgd.set_xlabel('$k_3/k_1$', labelpad=10, fontsize=25)
        bkgd.set_ylabel('$k_2/k_1$', labelpad=5, fontsize=25)
        fig.subplots_adjust(wspace=0.05, hspace=0.1, right=0.935)
        cbar_ax = fig.add_axes([0.95, 0.15, 0.0125, 0.7])
        cbar = fig.colorbar(bplot, cax=cbar_ax)
        cbar.set_label('$(B(k_1, k_2, k_3) - B^\mathrm{(fid)})/B^\mathrm{(fid)}$', rotation=90, fontsize=20)
        fig.savefig(''.join([UT.doc_dir(), 'figs/halodB123_relative_shape', '_', krange_str, str_rsd, '.pdf']), bbox_inches='tight') 

    elif typ == 'b_amp': 
        i_k = i_k[klim]
        j_k = j_k[klim]
        l_k = l_k[klim]
            
        ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # order of triangles 

        fig = plt.figure(figsize=(25,10))
        sub = fig.add_subplot(211)
        axins = inset_axes(sub, loc='upper right', width="40%", height="45%") 
        sub2 = fig.add_subplot(212)
        axins2 = inset_axes(sub2, loc='upper right', width="40%", height="45%") 
        ii = 0 
        for mnu, B123 in zip([0.0]+mnus, [B123_fid]+B123s):
            _b123 = B123[klim][ijl]
            sub.plot(range(np.sum(klim)), _b123, c='C'+str(ii), label=str(mnu)+'eV') 
            axins.plot(range(np.sum(klim)), _b123, c='C'+str(ii), label=str(mnu)+'eV') 
            if mnu == 0.0: 
                sub2.plot(range(np.sum(klim)), _b123, c='C0') 
                axins2.plot(range(np.sum(klim)), _b123, c='C0') 
            ii += 1 

        sub.legend(loc='lower left', ncol=4, columnspacing=0.5, markerscale=4, handletextpad=0.25, fontsize=20) 
        sub.set_yscale('log') 
        axins.set_yscale('log') 
        if rsd:
            sub.set_xlim([0, np.sum(klim)])
            sub.set_ylim([1e6, 1e10]) 
            axins.set_xlim(480, 500)
            axins.set_ylim(5e7, 2e8) 
        else: 
            sub.set_xlim([0, 1200])
            sub.set_ylim([1e5, 5e9]) 
            axins.set_xlim(480, 500)
            axins.set_ylim(2e7, 1.5e8) 
        
        axins.set_xticklabels('') 
        axins.yaxis.set_minor_formatter(NullFormatter())
        mark_inset(sub, axins, loc1=2, loc2=4, fc="none", ec="0.5")

        
        for sig8, B123 in zip(sig8s, B123_s8s):
            _b123 = B123[klim][ijl]
            if ii < 10: 
                sub2.plot(range(np.sum(klim)), _b123, c='C'+str(ii), label='$\sigma_8=$'+str(sig8)) 
                axins2.plot(range(np.sum(klim)), _b123, c='C'+str(ii)) 
            else: 
                sub2.plot(range(np.sum(klim)), _b123, c='C9', label='$\sigma_8=$'+str(sig8)) 
                axins2.plot(range(np.sum(klim)), _b123, c='C9') 
            ii += 2 
        sub2.legend(loc='lower left', ncol=4, columnspacing=0.5, markerscale=4, handletextpad=0.25, fontsize=20) 
        sub2.set_yscale('log') 
        axins2.set_yscale('log') 
        if rsd:
            sub2.set_xlim([0, np.sum(klim)])
            sub2.set_ylim([1e6, 1e10]) 
            axins2.set_xlim(480, 500)
            axins2.set_ylim(5e7, 2e8) 
        else: 
            sub2.set_xlim([0, 1200])
            sub2.set_ylim([1e5, 5e9]) 
            axins2.set_xlim(480, 500)
            axins2.set_ylim(2e7, 1.5e8) 
        
        axins2.set_xticklabels('') 
        axins2.yaxis.set_minor_formatter(NullFormatter())
        mark_inset(sub2, axins2, loc1=2, loc2=4, fc="none", ec="0.5")
        
        bkgd = fig.add_subplot(111, frameon=False)
        bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        bkgd.set_xlabel(r'$k_1 \le k_2 \le k_3$ triangle indices', labelpad=15, fontsize=25) 
        bkgd.set_ylabel('$B(k_1, k_2, k_3)$', labelpad=10, fontsize=25) 
        fig.subplots_adjust(hspace=0.15)
        fig.savefig(''.join([UT.doc_dir(), 'figs/haloB123_amp', '_', krange_str, str_rsd, '.pdf']), bbox_inches='tight') 
    
    elif typ == 'db_amp':
        fig = plt.figure(figsize=(25,8))
        sub = fig.add_subplot(211)
        sub2 = fig.add_subplot(212)
        i_k = i_k[klim]
        j_k = j_k[klim]
        l_k = l_k[klim]

        ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # triangle order

        ii = 0 
        for mnu, B123 in zip([0.0]+mnus, [B123_fid]+B123s):
            b123 = B123[klim] - B123_fid[klim]
            _b123 = b123[ijl] 

            if mnu != 0.0: 
                sub.plot(range(np.sum(klim)), _b123, c='C'+str(ii), label=str(mnu)+'eV') 
            else: 
                sub2.plot(range(np.sum(klim)), _b123, c='C0') 
            ii += 1 
        
        sub.legend(loc='upper right', markerscale=4, handletextpad=0.25, fontsize=20) 
        sub.set_xlim([0, np.sum(klim)])
        sub.set_yscale('log') 
        sub.set_ylim([1e6, 5e8]) 

        for sig8, B123 in zip(sig8s, B123_s8s):
            b123 = B123[klim] - B123_fid[klim]
            _b123 = b123[ijl] 
            if ii < 10: 
                sub2.plot(range(np.sum(klim)), _b123, c='C'+str(ii), label='$\sigma_8=$'+str(sig8)) 
            else: 
                sub2.plot(range(np.sum(klim)), _b123, c='C9', label='$\sigma_8=$'+str(sig8)) 
            ii += 2 
        sub2.legend(loc='upper right', markerscale=4, handletextpad=0.25, fontsize=20) 
        sub2.set_xlim([0, np.sum(klim)])
        sub2.set_yscale('log') 
        sub2.set_ylim([1e6, 5e8]) 
        
        bkgd = fig.add_subplot(111, frameon=False)
        bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        bkgd.set_xlabel(r'$k_1 \le k_2 \le k_3$ triangle indices', labelpad=10, fontsize=25) 
        bkgd.set_ylabel('$B(k_1, k_2, k_3)$', fontsize=25) 
        fig.subplots_adjust(hspace=0.15)
        fig.savefig(''.join([UT.doc_dir(), 'figs/halodB123_amp', '_', krange_str, str_rsd, '.pdf']), bbox_inches='tight') 
        
        ii = 0 
        fig = plt.figure(figsize=(18,6))
        sub = fig.add_subplot(111)
        for mnu, B123 in zip([0.0]+mnus, [B123_fid]+B123s):
            b123 = B123[klim] - B123_fid[klim]
            _b123 = b123[ijl]
            sub.plot(range(np.sum(klim)), _b123, c='C'+str(ii), lw=2, label=str(mnu)+'eV') 
            ii += 1 
        
        for sig8, B123 in zip(sig8s, B123_s8s):
            b123 = B123[klim] - B123_fid[klim]
            _b123 = b123[ijl] 
            if ii < 10: 
                sub.plot(range(np.sum(klim)), _b123, c='C'+str(ii), lw=0.75, label='$\sigma_8=$'+str(sig8)) 
            else: 
                sub.plot(range(np.sum(klim)), _b123, c='C9', lw=0.75, label='$\sigma_8=$'+str(sig8)) 
            ii += 2 
        
        sub.legend(loc='upper right', ncol=2, markerscale=4, handletextpad=0.25, fontsize=20) 
        sub.set_xlim([0, np.sum(klim)])
        sub.set_yscale('log') 
        sub.set_ylim([1e6, 5e8]) 
        fig.savefig(''.join([UT.doc_dir(), 'figs/halodB123_amp_comp', '_', krange_str, str_rsd, '.pdf']), bbox_inches='tight') 
    
    elif typ in ['b_amp_equilateral', 'b_amp_squeezed']: 
        # equilateral triangles 
        i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim]
        if typ == 'b_amp_equilateral': 
            tri = (i_k == j_k) & (j_k == l_k) # only equilateral 
        elif typ == 'b_amp_squeezed': 
            tri = (i_k == j_k) & (l_k == 3) # i_k >= j_k >= l_k (= 3*kf~0.01) 
            
        fig = plt.figure(figsize=(10,8))
        gs = mpl.gridspec.GridSpec(5,1, figure=fig) 
        sub = plt.subplot(gs[:3,0]) 
        axins = inset_axes(sub, loc='lower left', width="50%", height="50%") 
        sub2 = plt.subplot(gs[-2:,0]) 
        ii = 0 
        for mnu, B123 in zip([0.0]+mnus, [B123_fid]+B123s):
            _b123 = B123[klim][tri]
            sub.plot(kf*i_k[tri], _b123, c='C'+str(ii), label=str(mnu)+'eV') 
            axins.plot(kf*i_k[tri], _b123, c='C'+str(ii), label=str(mnu)+'eV') 
            if mnu != 0.0: 
                sub2.plot(kf*i_k[tri], _b123/B123_fid[klim][tri], c='C'+str(ii)) 
            ii += 1 

        for sig8, B123 in zip(sig8s, B123_s8s):
            _b123 = B123[klim][tri]
            if ii < 10: 
                sub.plot(kf*i_k[tri], _b123, c='C'+str(ii), ls='--', label='$\sigma_8=$'+str(sig8)) 
                axins.plot(kf*i_k[tri], _b123, c='C'+str(ii), ls='--') 
                sub2.plot(kf*i_k[tri], _b123/B123_fid[klim][tri], c='C'+str(ii), ls='--') 
            else: 
                sub.plot(kf*i_k[tri], _b123, c='C9', ls='--', label='$\sigma_8=$'+str(sig8)) 
                axins.plot(kf*i_k[tri], _b123, c='C9', ls='--') 
                sub2.plot(kf*i_k[tri], _b123/B123_fid[klim][tri], c='C9', ls='--') 
            ii += 2 
        
        sub2.plot([kmin, kmax], [1., 1.], c='k', ls='--', lw=2) 
        
        # subplot 1 inset 
        axins.set_xscale('log') 
        axins.set_xlim(0.2, 0.3)
        axins.set_yscale('log') 
        if typ == 'b_amp_equilateral': 
            axins.set_ylim(1.3e8, 2.3e8) 
        elif typ == 'b_amp_squeezed': 
            axins.set_ylim(1e9, 1.4e9) 
        axins.xaxis.set_minor_formatter(NullFormatter())
        axins.yaxis.set_minor_formatter(NullFormatter())
        axins.xaxis.set_major_formatter(NullFormatter())
        axins.yaxis.set_major_formatter(NullFormatter())
        mark_inset(sub, axins, loc1=3, loc2=4, fc="none", ec="0.5")

        sub.legend(loc='upper right', ncol=2, columnspacing=0.2, 
                markerscale=4, handletextpad=0.25, fontsize=18) 
        sub.set_xscale('log') 
        sub.set_xlim([kmin, kmax])
        sub.set_xticklabels([]) 
        sub.set_yscale('log') 
        
        sub2.set_xscale('log')
        sub2.set_xlim([kmin, kmax]) 
        sub2.set_ylim([0.95, 1.2]) 
        if typ == 'b_amp_equilateral': 
            sub.set_ylim([5e7, 1.25e10]) 
            sub.set_ylabel('$B(k, k, k)$', fontsize=25) 
            sub2.set_ylabel('$B(k, k, k)/B^\mathrm{(fid)}$', fontsize=25) 
            sub.set_title('{\em equilateral} configuration', pad=10, fontsize=25) 
        elif typ == 'b_amp_squeezed': 
            sub.set_ylim([7e8, 1.25e10]) 
            sub.set_ylabel('$B(k_1 = '+str(round(3*kf,3))+', k_2, k_2)$', fontsize=20) 
            sub2.set_ylabel('$B(k_1 = '+str(round(3*kf,3))+', k_2, k_2)/B^\mathrm{(fid)}$', fontsize=25) 
            sub.set_title('{\em squeezed} configuration', pad=10, fontsize=25) 
        
        bkgd = fig.add_subplot(111, frameon=False)
        bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        if typ == 'b_amp_equilateral': 
            bkgd.set_xlabel(r'$k$ [$h$/Mpc]', labelpad=10, fontsize=25) 
        elif typ == 'b_amp_squeezed': 
            bkgd.set_xlabel(r'$k_2$ [$h$/Mpc]', labelpad=10, fontsize=25) 

        fig.subplots_adjust(hspace=0.1)
        if typ == 'b_amp_equilateral': 
            fig.savefig(''.join([UT.doc_dir(), 'figs/haloB123_amp_equ', '_', krange_str, str_rsd, '.pdf']), bbox_inches='tight') 
        elif typ == 'b_amp_squeezed': 
            fig.savefig(''.join([UT.doc_dir(), 'figs/haloB123_amp_squ', '_', krange_str, str_rsd, '.pdf']), bbox_inches='tight') 

    elif typ == 'relative':
        fig = plt.figure(figsize=(18,18))
        i_k = i_k[klim]
        j_k = j_k[klim]
        l_k = l_k[klim]
        
        ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # triangle order

        ii = 1
        for i, mnu, sig8, B123, B123s8 in zip(range(3), mnus, sig8s, B123s, B123_s8s): 
            sub = fig.add_subplot(3,1,i+1) 

            b123 = (B123[klim] - B123_fid[klim])/B123_fid[klim]
            _b123 = b123[ijl]
            sub.plot(range(np.sum(klim)), _b123, lw=2, c='C'+str(ii), label=str(mnu)+'eV') 
            ii += 1 
        
            b123 = (B123s8[klim] - B123_fid[klim])/B123_fid[klim]
            _b123 = b123[ijl] 
            sub.plot(range(np.sum(klim)), _b123, lw=1, c='k', label='$\sigma_8=$'+str(sig8)) 
            sub.plot([0, np.sum(klim)], [0., 0.], c='k', ls='--', lw=2)
            if rsd: 
                sub.legend(loc='upper left', ncol=2, markerscale=4, handletextpad=0.5, fontsize=25) 
                sub.set_xlim([0, np.sum(klim)])
                sub.set_ylim([-0.01, 0.2]) 
                sub.set_yticks([0., 0.05, 0.1, 0.15, 0.2]) 
            else: 
                sub.legend(loc='upper left', frameon=True, ncol=2, markerscale=4, handletextpad=0.5, fontsize=25) 
                sub.set_xlim([0, 1200])
                sub.set_ylim([-0.01, 0.25]) 
                sub.set_yticks([0., 0.05, 0.1, 0.15, 0.2, 0.25]) 

            if i < 2: sub.set_xticklabels([]) 
        
        bkgd = fig.add_subplot(111, frameon=False)
        bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        bkgd.set_xlabel(r'$k_1 \le k_2 \le k_3$ triangle indices', labelpad=10, fontsize=30) 
        bkgd.set_ylabel('$B(k_1, k_2, k_3) / B^\mathrm{fid} - 1 $', labelpad=15, fontsize=30) 
        fig.subplots_adjust(hspace=0.1)
        fig.savefig(''.join([UT.doc_dir(), 'figs/halodB123_relative', '_', krange_str, str_rsd, '.pdf']), bbox_inches='tight') 

    elif typ == 'ratio':
        fig = plt.figure(figsize=(18,6))
        sub = fig.add_subplot(111)
        i_k = i_k[klim]
        j_k = j_k[klim]
        l_k = l_k[klim]
        
        ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # triangle order

        ii = 1 
        for mnu, B123 in zip(mnus, B123s):
            b123 = (B123[klim] / B123_fid[klim])
            _b123 = b123[ijl]
            sub.plot(range(np.sum(klim)), _b123, lw=2, c='C'+str(ii), label=str(mnu)+'eV') 
            ii += 1 
        
        for sig8, B123 in zip(sig8s, B123_s8s):
            b123 = (B123[klim] / B123_fid[klim])
            _b123 = b123[ijl] 
            if ii < 10: 
                sub.plot(range(np.sum(klim)), _b123, lw='1', c='C'+str(ii), label='$\sigma_8=$'+str(sig8)) 
            else: 
                sub.plot(range(np.sum(klim)), _b123, lw='1', c='C9', label='$\sigma_8=$'+str(sig8)) 
            ii += 2 
        sub.plot([0, np.sum(klim)], [1., 1.], c='k', ls='--', lw=2)
        sub.legend(loc='upper right', ncol=3, markerscale=4, handletextpad=0.25, fontsize=20) 
        sub.set_xlim([0, np.sum(klim)])
        sub.set_ylim([0.98, 1.2]) 
        #sub.set_yticks([0., 0.05, 0.1, 0.15, 0.2]) 
        
        bkgd = fig.add_subplot(111, frameon=False)
        bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        bkgd.set_xlabel(r'$k_1 \le k_2 \le k_3$ triangle indices', labelpad=10, fontsize=25) 
        bkgd.set_ylabel('$B(k_1, k_2, k_3) / B^\mathrm{(fid)}$', labelpad=15, fontsize=25) 
        fig.subplots_adjust(hspace=0.15)
        fig.savefig(''.join([UT.doc_dir(), 'figs/halodB123_ratio', str_rsd, '.pdf']), bbox_inches='tight') 
    return None 


def compare_B123_triangle(ik_pairs, nreals=range(1,71), krange=[0.03, 0.5], zspace=False): 
    ''' Make various bispectrum plots as a function of m_nu 
    '''
    str_rsd = ''
    if zspace: str_rsd = '_rsd'
    mnus = [0.06, 0.1, 0.15]
    sig8s = [0.822, 0.818, 0.807, 0.798]
    kmin, kmax = krange 
    krange_str = str(kmin).replace('.', '')+'_'+str(kmax).replace('.', '') 

    _, _, _, B123_fid, cnts_fid, _ = readB123(0.0, nreals, 4, BorQ='B', zspace=zspace)

    B123s, cnts = [], [] 
    for mnu in mnus: 
        _i, _j, _l, _B123, _cnts, kf = readB123(mnu, nreals, 4, BorQ='B', zspace=zspace)
        B123s.append(_B123) 
        cnts.append(_cnts)

    B123_s8s, cnt_s8s = [], [] 
    for sig8 in sig8s: 
        _i, _j, _l, _B123, _cnts, kf = readB123_sigma8(sig8, nreals, 4, BorQ='B', zspace=zspace)
        B123_s8s.append(_B123) 
        cnt_s8s.append(_cnts)
    i_k, j_k, l_k = _i, _j, _l
   
    fig = plt.figure(figsize=(6*len(ik_pairs),6))
    for i_p, ik_pair in enumerate(ik_pairs):  
        ik1, ik2 = ik_pair 

        klim = ((i_k == ik1) & (j_k == ik2)) | ((j_k == ik1) & (l_k == ik2)) 
        # angle between k1 and k2 
        ik3 = [] 
        for i, j, l in zip(i_k[klim], j_k[klim], l_k[klim]): 
            ijl = np.array([i, j, l]) 
            _ijl = np.delete(ijl, np.argmin(np.abs(ijl - ik1)))
            _ik3 = np.delete(_ijl, np.argmin(np.abs(_ijl - ik2)))[0]
            ik3.append(_ik3)
        ik3 = np.array(ik3) 
        theta12 = np.arccos((ik3**2 - ik1**2 - ik2**2)/2./ik1/ik2)

        theta_sort = np.argsort(theta12) 

        sub = fig.add_subplot(1, len(ik_pairs), i_p+1)
        ii = 1
        for mnu, B123 in zip(mnus, B123s): 
            lbl = None 
            if i_p == 0: lbl = str(mnu)+'eV'
            _b123 = (B123[klim][theta_sort] - B123_fid[klim][theta_sort])/B123_fid[klim][theta_sort]
            sub.plot(theta12[theta_sort]/np.pi, _b123, c='C'+str(ii), label=lbl)
            sub.set_xlim([0., 1.]) 
            ii += 1 
        ii = 1 
        if i_p == 0: sub.legend(loc='upper left', handletextpad=0.2, fontsize=20) 
        for i, sig8, B123 in zip(range(len(sig8s)), sig8s, B123_s8s): 
            _b123 = (B123[klim][theta_sort] - B123_fid[klim][theta_sort])/B123_fid[klim][theta_sort]
            if i < 3: 
                #sub.plot(theta12[theta_sort]/np.pi, _b123, c='k', lw=0.2)
                sub.plot(theta12[theta_sort]/np.pi, _b123, c='C'+str(ii), lw=1, 
                        ls=(0,(5,6)), label='$\sigma_8='+str(sig8)+'$')
            else: 
                sub.plot(theta12[theta_sort]/np.pi, _b123, c='k', 
                        ls=(0,(1,6)), lw=0.5, label='$\sigma_8='+str(sig8)+'$')
            sub.set_xlim([0., 1.]) 
            ii += 1 
        sub.set_ylim([0., 0.15]) 
        sub.set_yticks([0., 0.05, 0.1, 0.15]) 
        sub.set_title('$k_1 = %.2f,\,\,k_2 = %.2f$' % (ik1*kf, ik2*kf), fontsize=25)
        if i_p > 0: sub.set_yticklabels([]) 
        if i_p == 1: sub.legend(loc='upper left', handletextpad=0.2, fontsize=20) 
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'$\theta_{12}/\pi$', labelpad=10, fontsize=25) 
    bkgd.set_ylabel((r'$B(k_1, k_2, \theta_{12}) / B^\mathrm{fid} - 1$'), labelpad=10, fontsize=25) 
    fig.subplots_adjust(wspace=0.1) 
    fig.savefig(''.join([UT.doc_dir(), 'figs/', 
        'haloB123_triangles_', krange_str, str_rsd, '.pdf']), bbox_inches='tight') 
    return None 


def readB123(mneut, ireals, nzbin, BorQ='B', zspace=False):
    ''' read in bispectrum of massive neutrino halo catalogs with `mneut` eV
    `ireals` realizations, `nzbin` redshift bin 

    :param mneut: 
        neutrino mass

    :param ireals: 
        2 element array/list/tuple that specifies the realizations 

    :param nzbin: 
        redshift bin number 
    '''
    if zspace: str_rsd = '.zspace'
    else: str_rsd = '.rspace'
    fb123 = ''.join([UT.doc_dir(), 'dat/', 
        'halo_bispectrum.', str(mneut), 'eV.', 
        str(ireals[0]), '_', str(ireals[1]), '.z', str(nzbin), str_rsd, '.hdf5']) 

    if os.path.isfile(fb123): 
        f = h5py.File(fb123, 'r') 
        k_f = f.attrs['kf'] 

        i_k = f['i_k1'].value 
        j_k = f['i_k2'].value 
        l_k = f['i_k3'].value 
        
        p0k1 = f['p0k1'].value
        p0k2 = f['p0k2'].value
        p0k3 = f['p0k3'].value
        
        B123 = f['B123'].value
        B_SN = f['B_SN'].value 
        Q123 = f['Q123'].value
        counts = f['counts'].value
    else: 
        ffix = lambda mneut, nreal, nzbin: ''.join([UT.dat_dir(), 'bispectrum/', 
            'groups.', str(mneut), 'eV.', str(nreal), '.nzbin', str(nzbin), '.mhmin3200.0', str_rsd, 
            '.Ngrid360.Nmax40.Ncut3.step3.pyfftw.dat']) 
        
        p0k1, p0k2, p0k3 = [], [], [] 
        B123, Q123, B_SN, counts = [], [], [], [] 
        for ii, _i in enumerate(range(ireals[0], ireals[1]+1)):
            i_k, j_k, l_k, _p0k1, _p0k2, _p0k3, b123, q123, b_sn, cnts = np.loadtxt(ffix(mneut, _i, nzbin), 
                    skiprows=1, unpack=True, usecols=range(10)) 
            p0k1.append(_p0k1)
            p0k2.append(_p0k2)
            p0k3.append(_p0k3)
            B123.append(b123)
            B_SN.append(b_sn)
            Q123.append(q123)
            counts.append(cnts)
        
        k_f = 2.*np.pi/1000. # k_fundmanetal

        f = h5py.File(fb123, 'w') 
        f.attrs['kf'] = k_f
        f.create_dataset('i_k1', data=i_k)
        f.create_dataset('i_k2', data=j_k)
        f.create_dataset('i_k3', data=l_k)

        f.create_dataset('p0k1', data=np.array(p0k1))
        f.create_dataset('p0k2', data=np.array(p0k2))
        f.create_dataset('p0k3', data=np.array(p0k3))

        f.create_dataset('B123', data=np.array(B123))
        f.create_dataset('B_SN', data=np.array(B_SN))
        f.create_dataset('Q123', data=np.array(Q123))
        f.create_dataset('counts', data=np.array(counts)) 
        f.close() 

    B123 = np.average(B123, axis=0)
    Q123 = np.average(Q123, axis=0)
    counts = np.average(counts, axis=0)

    if BorQ == 'B':
        return i_k, j_k, l_k, B123, counts, k_f
    elif BorQ == 'Q':
        return i_k, j_k, l_k, Q123, counts, k_f 


def readB123_sigma8(sig8, ireals, nzbin, BorQ='B', zspace=False):
    ''' read in bispectrum of sigma8 varied m_nu = 0 halo catalogs
    using the function Obvs.B123_halo_sigma8
    '''
    if zspace: str_rsd = '.zspace'
    else: str_rsd = '.rspace'
    fb123 = ''.join([UT.doc_dir(), 'dat/', 
        'halo_bispectrum.0.0eV.sig8_', str(sig8), '.', 
        str(ireals[0]), '_', str(ireals[1]), '.z', str(nzbin), str_rsd, '.hdf5']) 

    if os.path.isfile(fb123): 
        f = h5py.File(fb123, 'r') 
        k_f = f.attrs['kf'] 

        i_k = f['i_k1'].value 
        j_k = f['i_k2'].value 
        l_k = f['i_k3'].value 
        
        B123 = f['B123'].value
        B_sn = f['B_SN'].value 
        Q123 = f['Q123'].value
        counts = f['counts'].value
    else: 
        ffix = lambda sig8, nreal, nzbin: ''.join([UT.dat_dir(), 'bispectrum/', 
            'groups.0.0eV.sig8_', str(sig8), '.', str(nreal), '.nzbin', str(nzbin), '.mhmin3200.0', str_rsd,
            '.Ngrid360.Nmax40.Ncut3.step3.pyfftw.dat']) 
        
        p0k1, p0k2, p0k3 = [], [], [] 
        B123, Q123, B_SN, counts = [], [], [], [] 
        for ii, _i in enumerate(range(ireals[0], ireals[1]+1)):
            i_k, j_k, l_k, _p0k1, _p0k2, _p0k3, b123, q123, b_sn, cnts = np.loadtxt(ffix(sig8, _i, nzbin), 
                    skiprows=1, unpack=True, usecols=range(10)) 
            p0k1.append(_p0k1)
            p0k2.append(_p0k2)
            p0k3.append(_p0k3)
            B123.append(b123)
            B_SN.append(b_sn)
            Q123.append(q123)
            counts.append(cnts)
        
        k_f = 2.*np.pi/1000. # k_fundmanetal

        f = h5py.File(fb123, 'w') 
        f.attrs['kf'] = k_f 
        f.create_dataset('i_k1', data=i_k)
        f.create_dataset('i_k2', data=j_k)
        f.create_dataset('i_k3', data=l_k)

        f.create_dataset('p0k1', data=np.array(p0k1))
        f.create_dataset('p0k2', data=np.array(p0k2))
        f.create_dataset('p0k3', data=np.array(p0k3))

        f.create_dataset('B123', data=np.array(B123))
        f.create_dataset('B_SN', data=np.array(B_SN))
        f.create_dataset('Q123', data=np.array(Q123))
        f.create_dataset('counts', data=np.array(counts)) 
        f.close() 

    B123 = np.average(B123, axis=0)
    Q123 = np.average(Q123, axis=0)
    counts = np.average(counts, axis=0)

    if BorQ == 'B':
        return i_k, j_k, l_k, B123, counts, k_f
    elif BorQ == 'Q':
        return i_k, j_k, l_k, Q123, counts, k_f

##################################################################
# qujiote fisher 
##################################################################
def quijote_covariance(krange=[0.01, 0.5]): 
    ''' plot the covariance matrix of the quijote fiducial 
    bispectrum. 
    '''
    kmin, kmax = krange  

    # read in covariance matrix 
    f = h5py.File(os.path.join(UT.dat_dir(), 'bispectrum', 'quijote_Cov_bk.%.2f_%.2f.gmorder.hdf5' % (kmin, kmax)), 'r') 
    C_bk = f['C_bk'].value 

    # plot the covariance matrix 
    fig = plt.figure(figsize=(10,8))
    sub = fig.add_subplot(111)
    cm = sub.pcolormesh(C_bk, norm=LogNorm(vmin=1e11, vmax=1e18))
    cbar = fig.colorbar(cm, ax=sub) 
    sub.set_title(r'Quijote $B(k_1, k_2, k_3)$ Covariance', fontsize=25)
    fig.savefig(os.path.join(UT.doc_dir(), 'figs', 'quijote_Cov.png'), bbox_inches='tight') 
    return None 


def quijote_forecast(krange=[0.01, 0.5], deriv='p'):
    ''' fisher forecast for qujote 
    
    :param krange: (default: [0.01, 0.5]) 
        tuple specifying the kranges of k1, k2, k3 in the bispectrum
    
    :param mpc: (default: 'p') 
        how the derivatives along Mnu are calculated. The options are 
        ['p', 'pp', 'ppp', 'fd'] 
    '''
    # read in  fisher matrix (Fij)
    bk_dir = os.path.join(UT.dat_dir(), 'bispectrum')
    f_ij = os.path.join(bk_dir, 'quijote_Fisher.%s.%.2f_%.2f.gmorder.hdf5' % (deriv, krange[0], krange[1]))
    f = h5py.File(f_ij, 'r') 
    Fij = f['Fij'].value

    Finv = np.linalg.inv(Fij) # invert fisher matrix 

    thetas = ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
    theta_lbls = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$']
    theta_lims = [(0.15, 0.5), (0., 0.1), (0., 1.5), (0.4, 1.5), (0.67, 1.), (-3, 3)]
    ntheta = len(thetas)
    
    theta_fid = {'Mnu': 0., 'Ob': 0.049, 'Om': 0.3175, 'h': 0.6711,  'ns': 0.9624,  's8': 0.834} # fiducial theta 
    for i in xrange(ntheta): 
        if thetas[i] == 'Mnu': print deriv, thetas[i], np.sqrt(Finv[i,i])
    fig = plt.figure(figsize=(17, 15))
    for i in xrange(ntheta-1): 
        for j in xrange(i+1, ntheta): 
            # sub inverse fisher matrix 
            Finv_sub = np.array([[Finv[i,i], Finv[i,j]], [Finv[j,i], Finv[j,j]]]) 

            theta_fid_i = theta_fid[thetas[i]]
            theta_fid_j = theta_fid[thetas[j]]
            
            # get ellipse parameters 
            a = np.sqrt(0.5*(Finv_sub[0,0] + Finv_sub[1,1]) + np.sqrt(0.25*(Finv_sub[0,0]-Finv_sub[1,1])**2 + Finv_sub[0,1]**2))
            b = np.sqrt(0.5*(Finv_sub[0,0] + Finv_sub[1,1]) - np.sqrt(0.25*(Finv_sub[0,0]-Finv_sub[1,1])**2 + Finv_sub[0,1]**2))
            theta = 0.5 * np.arctan2(2.0 * Finv_sub[0,1], (Finv_sub[0,0] - Finv_sub[1,1]))
    
            # plot the ellipse
            sub = fig.add_subplot(ntheta-1, ntheta-1, (ntheta-1) * (j-1) + i + 1) 
            for ii, alpha in enumerate([2.48, 1.52]):
                e = Ellipse(xy=(theta_fid_i, theta_fid_j), 
                        width=alpha * a, height=alpha * b, angle=theta * 360./(2.*np.pi))
                sub.add_artist(e)
                if ii == 0: alpha = 0.5
                if ii == 1: alpha = 1.
                e.set_alpha(alpha)
                e.set_facecolor('C0')

            x_range = np.sqrt(Finv[i,i]) * 1.5
            y_range = np.sqrt(Finv[j,j]) * 1.5
            
            #sub.set_xlim(theta_lims[i])
            #sub.set_ylim(theta_lims[j])
            sub.set_xlim([theta_fid_i - x_range, theta_fid_i + x_range])
            sub.set_ylim([theta_fid_j - y_range, theta_fid_j + y_range])
            if i == 0:   
                sub.set_ylabel(theta_lbls[j], fontsize=30) 
            else: 
                sub.set_yticks([])
                sub.set_yticklabels([])
            
            if j == ntheta-1: 
                sub.set_xlabel(theta_lbls[i], labelpad=10, fontsize=30) 
            else: 
                sub.set_xticks([])
                sub.set_xticklabels([]) 
    fig.subplots_adjust(wspace=0.05, hspace=0.05) 
    fig.savefig(os.path.join(UT.doc_dir(), 'figs', 'quijote_Fisher_%s_%.2f_%.2f.png' % (deriv, krange[0], krange[1])), bbox_inches='tight') 
    return None


if __name__=="__main__": 
    #compare_Plk(nreals=range(1,101), krange=[0.01, 0.5])
    #ratio_Plk(nreals=range(1,101), krange=[0.01, 0.5])
    for rsd in [False, True]:  
        if not rsd: nreals = (1, 100) 
        else: nreals = (1, 100) 
        for kmax in [0.5]: 
            continue 
            #compare_B123('b_shape', 
            #        nreals=nreals, krange=[0.01, kmax], zspace=rsd, nbin=31)
            #compare_B123('relative_shape', 
            #        nreals=nreals, krange=[0.01, kmax], zspace=rsd, nbin=31)
            #compare_B123('b_amp', 
            #        nreals=nreals, krange=[0.01, kmax], zspace=rsd)
            #compare_B123('b_amp_equilateral', 
            #        nreals=nreals, krange=[0.01, kmax], zspace=rsd)
            #compare_B123('b_amp_squeezed', 
            #        nreals=nreals, krange=[0.01, kmax], zspace=rsd)
            #compare_B123('relative', 
            #        nreals=nreals, krange=[0.01, kmax], zspace=rsd)
            #compare_B123_triangle([[30, 18], [18, 18], [12,9]], 
            #        nreals=nreals, krange=[0.01, kmax], zspace=rsd)
    #quijote_covariance(krange=[0.01, 0.5]) 
    for kmax in [0.5]: #[0.2, 0.3, 0.4, 0.5]: 
        for deriv in ['p', 'fd']: 
            quijote_forecast(krange=[0.01, kmax], deriv=deriv)
