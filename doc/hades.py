'''

analysis using the HADES halo catalog. These results mainly look at the 
Mnu-sigma8 degeneracy

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


##################################################################
# powerspectrum comparison 
##################################################################
def compare_Plk(kmax=0.25): 
    '''
    '''
    mnus = [0.0, 0.06, 0.1, 0.15]
    sig8s = [0.822, 0.818, 0.807, 0.798]

    # read in all the powerspectrum
    p0ks, p2ks, p4ks = [], [], [] 
    for mnu in mnus: 
        _hades = Obvs.hadesPlk(mnu) 
        p0ks.append(np.average(_hades['p0k'], axis=0))
        p2ks.append(np.average(_hades['p2k'], axis=0))
        p4ks.append(np.average(_hades['p4k'], axis=0))

    p0k_s8s, p2k_s8s, p4k_s8s = [], [], [] 
    for sig8 in sig8s: 
        _hades = Obvs.hadesPlk_s8(sig8) 
        p0k_s8s.append(np.average(_hades['p0k'], axis=0))
        p2k_s8s.append(np.average(_hades['p2k'], axis=0))
        p4k_s8s.append(np.average(_hades['p4k'], axis=0))
    k = _hades['k'] 

    klim = (k <= kmax) # k < k_max 

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
    sub.plot([1e-4, 10.], [1., 1.], c='k', ls='--') 
    #sub.legend(loc='upper left', ncol=2, handletextpad=0.25, columnspacing=0.5, fontsize=18) 
    sub.set_xlabel("$k$ [$h$/Mpc]", fontsize=25) 
    sub.set_xscale("log") 
    sub.set_xlim([1e-2, 0.5])
    sub.set_ylim([0.98, 1.1]) 
    sub.set_yticks([1., 1.05, 1.1]) 
    sub.set_ylabel('$P_2(k)/P_2^\mathrm{(fid)}(k)$', fontsize=25) 
    fig.subplots_adjust(wspace=0.7, hspace=0.125)
    fig.savefig(os.path.join(UT.doc_dir(), 'figs', 'haloPlk_rsd.pdf'), bbox_inches='tight') 
    return None


def ratio_Plk(kmax=0.25): 
    '''
    '''
    mnus = [0.0, 0.06, 0.1, 0.15]
    sig8s = [0.822, 0.818, 0.807, 0.798]

    # read in all the powerspectrum
    p0ks, p2ks, p4ks = [], [], [] 
    for mnu in mnus: 
        _hades = Obvs.hadesPlk(mnu) 
        p0ks.append(np.average(_hades['p0k'], axis=0))
        p2ks.append(np.average(_hades['p2k'], axis=0))
        p4ks.append(np.average(_hades['p4k'], axis=0))

    p0k_s8s, p2k_s8s, p4k_s8s = [], [], [] 
    for sig8 in sig8s: 
        _hades = Obvs.hadesPlk_s8(sig8) 
        p0k_s8s.append(np.average(_hades['p0k'], axis=0))
        p2k_s8s.append(np.average(_hades['p2k'], axis=0))
        p4k_s8s.append(np.average(_hades['p4k'], axis=0))
    k = _hades['k'] 

    klim = (k <= kmax) # k < k_max 

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
    fig.savefig(os.path.join(UT.doc_dir(), 'figs', 'haloPlk_rsd_ratio.pdf'), bbox_inches='tight') 
    return None

##################################################################
# bispectrum comparison 
##################################################################
def compare_Bk(kmax=0.5, rsd=True):  
    ''' Compare the amplitude of the bispectrum for the HADES simulations. 

    :param krange: (default: [0.01, 0.5]) 
        k-range of k1, k2, k3 

    :param rsd: (default: True) 
        RSD or not 
    '''
    mnus = [0.06, 0.1, 0.15]
    sig8s = [0.822, 0.818, 0.807, 0.798]

    hades_fid = Obvs.hadesBk(0.0, nzbin=4, rsd=rsd) # fiducial bispectrum
    Bk_fid = np.average(hades_fid['b123'], axis=0)
    Bk_sn = np.average(hades_fid['b_sn'], axis=0) 

    Bk_Mnu, Bk_s8s = [], [] 
    for mnu in mnus: # bispectrum for different Mnu
        hades_i = Obvs.hadesBk(mnu, nzbin=4, rsd=rsd)
        Bk_Mnu.append(np.average(hades_i['b123'], axis=0)) 
    for sig8 in sig8s: # bispectrum for different sigma8 
        hades_i = Obvs.hadesBk_s8(sig8, nzbin=4, rsd=rsd)
        Bk_s8s.append(np.average(hades_i['b123'], axis=0)) 

    i_k, j_k, l_k = hades_i['k1'], hades_i['k2'], hades_i['k3']
    kf = 2.*np.pi/1000. 
    klim = ((i_k * kf <= kmax) & (j_k * kf <= kmax) & (l_k * kf <= kmax)) 

    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 

    ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # order of triangles 

    fig = plt.figure(figsize=(25,10))
    sub = fig.add_subplot(211)
    axins = inset_axes(sub, loc='upper right', width="40%", height="45%") 
    sub2 = fig.add_subplot(212)
    axins2 = inset_axes(sub2, loc='upper right', width="40%", height="45%") 
    
    tri = np.arange(np.sum(klim))
    for ii, mnu, bk in zip(range(4), [0.0]+mnus, [Bk_fid]+Bk_Mnu):
        _bk = bk[klim][ijl]
        if mnu == 0.0: 
            sub2.plot(tri, _bk, c='C0') 
            axins2.plot(tri, _bk, c='C0') 
        sub.plot(tri, _bk, c='C'+str(ii), label=str(mnu)+'eV') 
        axins.plot(tri, _bk, c='C'+str(ii), label=str(mnu)+'eV') 

    for ii, sig8, bk in zip([4, 6, 8, 9], sig8s, Bk_s8s):
        _bk = bk[klim][ijl]
        sub2.plot(tri, _bk, c='C'+str(ii), label='$\sigma_8=%.3f$' % sig8) 
        axins2.plot(tri, _bk, c='C'+str(ii)) 
    #sub.plot(tri, Bk_sn[klim][ijl], c='k', ls=':') 

    #print(i_k[ijl][480:500]) 
    #print(j_k[ijl][480:500]) 
    #print(l_k[ijl][480:500]) 

    sub2.text(0.02, 0.15, '0.0 eV', ha='left', va='bottom', transform=sub2.transAxes, fontsize=20)
    sub.legend(loc='lower left', ncol=4, columnspacing=0.5, markerscale=4, handletextpad=0.25, fontsize=20) 
    sub2.legend(loc='lower left', ncol=4, columnspacing=0.5, markerscale=4, handletextpad=0.25, fontsize=20) 

    sub.set_yscale('log') 
    axins.set_yscale('log') 
    
    sub2.set_yscale('log') 
    axins2.set_yscale('log') 
    if rsd:
        sub.set_xlim([0, 1898])
        sub.set_ylim([1e6, 1e10]) 
        axins.set_xlim(480, 500)
        axins.set_ylim(5e7, 2e8) 
        sub2.set_xlim([0, 1898])
        sub2.set_ylim([1e6, 1e10]) 
        axins2.set_xlim(480, 500)
        axins2.set_ylim(5e7, 2e8) 
    else: 
        #sub.set_xlim([0, 1200])
        #sub.set_ylim([1e5, 5e9]) 
        sub.set_xlim([0, 1898])
        sub.set_ylim([1e6, 1e10]) 
        axins.set_xlim(480, 500)
        axins.set_ylim(2e7, 1.5e8) 
        sub2.set_xlim([0, 1200])
        sub2.set_ylim([1e5, 5e9]) 
        axins2.set_xlim(480, 500)
        axins2.set_ylim(2e7, 1.5e8) 

    axins.set_xticklabels('') 
    axins.yaxis.set_minor_formatter(NullFormatter())
    mark_inset(sub, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    axins2.set_xticklabels('') 
    axins2.yaxis.set_minor_formatter(NullFormatter())
    mark_inset(sub2, axins2, loc1=2, loc2=4, fc="none", ec="0.5")
    
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'triangle configurations', labelpad=15, fontsize=25) 
    bkgd.set_ylabel('$B(k_1, k_2, k_3)$', labelpad=10, fontsize=25) 
    fig.subplots_adjust(hspace=0.15)

    ffig = os.path.join(UT.doc_dir(), 'figs', 
            'haloBk_amp_kmax%s%s.pdf' % (str(kmax).replace('.', ''), ['', '_rsd'][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    
    # compare ratio of B(k) amplitude
    fig = plt.figure(figsize=(18,18))

    bk_fid = Bk_fid[klim][ijl]
    for i, mnu, sig8, bk, bks8 in zip(range(3), mnus, sig8s, Bk_Mnu, Bk_s8s): 
        sub = fig.add_subplot(3,1,i+1) 

        db = bk[klim][ijl]/bk_fid - 1.
        sub.plot(tri, db, lw=2, c='C'+str(i+1), label=str(mnu)+'eV') 
    
        db = bks8[klim][ijl]/bk_fid - 1.
        sub.plot(tri, db, lw=1, c='k', label='0.0 eV\n$\sigma_8=%.3f$' % sig8) 
        sub.plot([0, np.sum(klim)], [0., 0.], c='k', ls='--', lw=2)
        if rsd: 
            sub.legend(loc='upper left', ncol=2, markerscale=4, handletextpad=0.5, fontsize=25) 
            sub.set_xlim([0, np.sum(klim)])
            sub.set_ylim([-0.01, 0.25]) 
            sub.set_yticks([0., 0.05, 0.1, 0.15, 0.2, 0.25]) 
        else: 
            sub.legend(loc='upper left', frameon=True, ncol=2, markerscale=4, handletextpad=0.5, fontsize=25) 
            sub.set_xlim([0, 1200])
            sub.set_ylim([-0.01, 0.25]) 
            sub.set_yticks([0., 0.05, 0.1, 0.15, 0.2, 0.25]) 

        if i < 2: sub.set_xticklabels([]) 
    
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'triangle configurations', labelpad=15, fontsize=25) 
    bkgd.set_ylabel('$(B(k_1, k_2, k_3) - B^\mathrm{(fid)})/B^\mathrm{(fid)}$', labelpad=15, fontsize=30) 
    fig.subplots_adjust(hspace=0.1)
    ffig = os.path.join(UT.doc_dir(), 'figs', 
            'haloBk_residual_kmax%s%s.pdf' % (str(kmax).replace('.', ''), ['', '_rsd'][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def compare_Qk(kmax=0.5, rsd=True):  
    ''' Compare the reduced bispectrum for the HADES simulations. 

    :param kmax: (default: 0.5) 
        kmax 

    :param rsd: (default: True) 
        RSD or not 
    '''
    mnus = [0.06, 0.1, 0.15]
    sig8s = [0.822, 0.818, 0.807, 0.798]

    hades_fid = Obvs.hadesBk(0.0, nzbin=4, rsd=rsd) # fiducial bispectrum
    Qk_fid = np.average(hades_fid['q123'], axis=0)

    Qk_Mnu, Qk_s8s = [], [] 
    for mnu in mnus: # bispectrum for different Mnu
        hades_i = Obvs.hadesBk(mnu, nzbin=4, rsd=rsd)
        Qk_Mnu.append(np.average(hades_i['q123'], axis=0)) 
    for sig8 in sig8s: # bispectrum for different sigma8 
        hades_i = Obvs.hadesBk_s8(sig8, nzbin=4, rsd=rsd)
        Qk_s8s.append(np.average(hades_i['q123'], axis=0)) 

    i_k, j_k, l_k = hades_i['k1'], hades_i['k2'], hades_i['k3']
    kf = 2.*np.pi/1000. 
    klim = ((i_k * kf <= kmax) & (j_k * kf <= kmax) & (l_k * kf <= kmax)) 

    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 
    ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # order of triangles 

    fig = plt.figure(figsize=(25,10))
    sub = fig.add_subplot(211)
    axins = inset_axes(sub, loc='upper right', width="40%", height="45%") 
    sub2 = fig.add_subplot(212)
    axins2 = inset_axes(sub2, loc='upper right', width="40%", height="45%") 
    
    tri = np.arange(np.sum(klim))
    for ii, mnu, qk in zip(range(4), [0.0]+mnus, [Qk_fid]+Qk_Mnu):
        _qk = qk[klim][ijl]
        if mnu == 0.0: 
            sub2.plot(tri, _qk, c='C0') 
            axins2.plot(tri, _qk, c='C0') 
        sub.plot(tri, _qk, c='C'+str(ii), label=str(mnu)+'eV') 
        axins.plot(tri, _qk, c='C'+str(ii), label=str(mnu)+'eV') 

    for ii, sig8, qk in zip([4, 6, 8, 9], sig8s, Qk_s8s):
        _qk = qk[klim][ijl]
        sub2.plot(tri, _qk, c='C'+str(ii), label='$\sigma_8=%.3f$' % sig8) 
        axins2.plot(tri, _qk, c='C'+str(ii)) 

    sub2.text(0.02, 0.15, '0.0 eV', ha='left', va='bottom', transform=sub2.transAxes, fontsize=20)
    sub.legend(loc='lower left', ncol=4, columnspacing=0.5, markerscale=4, handletextpad=0.25, fontsize=20) 
    sub2.legend(loc='lower left', ncol=4, columnspacing=0.5, markerscale=4, handletextpad=0.25, fontsize=20) 

    sub.set_xlim([0, 1898])
    sub.set_ylim([0, 1]) 
    axins.set_xlim(480, 500)
    axins.set_ylim(0.3, 0.5) 
    sub2.set_xlim([0, 1898])
    sub2.set_ylim([0., 1.]) 
    axins2.set_xlim(480, 500)
    axins2.set_ylim(0.3, 0.5) 

    axins.set_xticklabels('') 
    axins.yaxis.set_minor_formatter(NullFormatter())
    mark_inset(sub, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    axins2.set_xticklabels('') 
    axins2.yaxis.set_minor_formatter(NullFormatter())
    mark_inset(sub2, axins2, loc1=2, loc2=4, fc="none", ec="0.5")
    
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'triangle configurations', labelpad=15, fontsize=25) 
    bkgd.set_ylabel('$Q(k_1, k_2, k_3)$', labelpad=10, fontsize=25) 
    fig.subplots_adjust(hspace=0.15)

    ffig = os.path.join(UT.doc_dir(), 'figs', 
            'haloQk_amp_kmax%s%s.pdf' % (str(kmax).replace('.', ''), ['', '_rsd'][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 

    # compare ratio of Q(k) amplitude
    fig = plt.figure(figsize=(18,18))

    qk_fid = Qk_fid[klim][ijl]
    for i, mnu, sig8, qk, qks8 in zip(range(3), mnus, sig8s, Qk_Mnu, Qk_s8s): 
        sub = fig.add_subplot(3,1,i+1) 

        dq = qk[klim][ijl]/qk_fid - 1.
        sub.plot(tri, dq, lw=2, c='C'+str(i+1), label=str(mnu)+'eV') 
    
        dq = qks8[klim][ijl]/qk_fid - 1.
        sub.plot(tri, dq, lw=1, c='k', label='0.0 eV\n$\sigma_8=%.3f$' % sig8) 
        sub.plot([0, np.sum(klim)], [0., 0.], c='k', ls='--', lw=2)
        sub.legend(loc='upper left', ncol=2, markerscale=4, handletextpad=0.5, fontsize=25) 
        sub.set_xlim([0, np.sum(klim)])
        sub.set_ylim([-0.01, 0.15]) 
        sub.set_yticks([0., 0.05, 0.1, 0.15]) 
        if i < 2: sub.set_xticklabels([]) 
    
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'triangle configurations', labelpad=15, fontsize=25) 
    bkgd.set_ylabel('$(Q(k_1, k_2, k_3) - Q^\mathrm{(fid)})/Q^\mathrm{(fid)}$', labelpad=15, fontsize=30) 
    fig.subplots_adjust(hspace=0.1)
    ffig = os.path.join(UT.doc_dir(), 'figs', 
            'haloQk_residual_kmax%s%s.pdf' % (str(kmax).replace('.', ''), ['', '_rsd'][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def compare_Bk_shape(kmax=0.5, rsd=True, nbin=31): 
    ''' Compare the amplitude of the bispectrum for the HADES simulations. 

    :param krange: (default: [0.01, 0.5]) 
        k-range of k1, k2, k3 

    :param rsd: (default: True) 
        RSD or not 

    :param nbin: (default 31)
        number of bins to divide k3/k1. 
    '''
    mnus = [0.06, 0.1, 0.15]
    sig8s = [0.822, 0.818, 0.807, 0.798]

    hades_fid = Obvs.hadesBk(0.0, nzbin=4, rsd=rsd) # fiducial bispectrum
    Bk_fid = np.average(hades_fid['b123'], axis=0)
    counts = hades_fid['counts']

    Bk_Mnu, Bk_s8s = [], [] 
    for mnu in mnus: # bispectrum for different Mnu
        hades_i = Obvs.hadesBk(mnu, nzbin=4, rsd=rsd)
        Bk_Mnu.append(np.average(hades_i['b123'], axis=0)) 
    for sig8 in sig8s: # bispectrum for different sigma8 
        hades_i = Obvs.hadesBk_s8(sig8, nzbin=4, rsd=rsd)
        Bk_s8s.append(np.average(hades_i['b123'], axis=0)) 

    i_k, j_k, l_k = hades_i['k1'], hades_i['k2'], hades_i['k3']
    k3k1 = l_k.astype(float)/i_k.astype(float)
    k2k1 = j_k.astype(float)/i_k.astype(float)
    kf = 2.*np.pi/1000. 
    klim = ((i_k * kf <= kmax) & (j_k * kf <= kmax) & (l_k * kf <= kmax)) 
        
    # plot B(k) shapes in triangle plot 
    x_bins = np.linspace(0., 1., int(nbin)+1)
    y_bins = np.linspace(0.5, 1., int(0.5*nbin)+1)
    if rsd: norm = LogNorm(vmin=5e6, vmax=1e9) # normalization 
    else: norm = SymLogNorm(vmin=-5e6, vmax=1e9, linthresh=1e6, linscale=1.)

    fig = plt.figure(figsize=(25,6))
    for i, mnu, bk in zip(range(4), [0.0]+mnus, [Bk_fid]+Bk_Mnu):
        sub = fig.add_subplot(2,4,i+1)
        BQgrid = ePlots._BorQgrid(k3k1[klim], k2k1[klim], bk[klim], counts[klim], x_bins, y_bins)
        bplot = sub.pcolormesh(x_bins, y_bins, BQgrid.T, norm=norm, cmap='RdBu')
        sub.text(0.05, 0.05, str(mnu)+'eV', ha='left', va='bottom', transform=sub.transAxes, fontsize=20)
        if i > 0: 
            sub.set_xticklabels([]) 
            sub.set_yticklabels([]) 

    for i, s8, bk in zip(range(3), sig8s, Bk_s8s): 
        sub = fig.add_subplot(2,4,i+6)
        BQgrid = ePlots._BorQgrid(k3k1[klim], k2k1[klim], bk[klim], counts[klim], x_bins, y_bins)
        bplot = sub.pcolormesh(x_bins, y_bins, BQgrid.T, norm=norm, cmap='RdBu')
        sub.text(0.05, 0.05, '0.0eV', ha='left', va='bottom', transform=sub.transAxes, fontsize=20)
        sub.text(0.975, 0.025, '$\sigma_8 = %.3f$' % sig8s[i], ha='right', va='bottom', 
                transform=sub.transAxes, fontsize=20)
        if i > 1: sub.set_yticklabels([]) 

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
    ffig = os.path.join(UT.doc_dir(), 'figs', 'haloBk_shape_kmax%s%s.pdf' % 
            (str(kmax).replace('.', ''), ['', '_rsd'][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 

    # plot residual of the B(k) shape dependence 
    fig = plt.figure(figsize=(25,6))
    for i, mnu, bk in zip(range(len(mnus)), mnus, Bk_Mnu): 
        sub = fig.add_subplot(2,4,i+1)
        dbk = bk/Bk_fid - 1.
        dBQgrid = ePlots._BorQgrid(k3k1[klim], k2k1[klim], dbk[klim], counts[klim], x_bins, y_bins)
        if rsd: bplot = sub.pcolormesh(x_bins, y_bins, dBQgrid.T, vmin=0., vmax=0.15, cmap='RdBu')
        else: bplot = sub.pcolormesh(x_bins, y_bins, dBQgrid.T, vmin=-0.1, vmax=0.15, cmap='RdBu')
        sub.text(0.05, 0.05, str(mnus[i])+'eV', ha='left', va='bottom', transform=sub.transAxes, fontsize=20)
        sub.set_xticklabels([]) 
        if i > 0: sub.set_yticklabels([]) 

    for i, s8, bk in zip(range(len(sig8s)), sig8s, Bk_s8s): 
        sub = fig.add_subplot(2,4,i+5)
        dbk = bk/Bk_fid - 1.
        dBQgrid = ePlots._BorQgrid(k3k1[klim], k2k1[klim], dbk[klim], counts[klim], x_bins, y_bins)
        if rsd: bplot = sub.pcolormesh(x_bins, y_bins, dBQgrid.T, vmin=0., vmax=0.15, cmap='RdBu')
        else: bplot = sub.pcolormesh(x_bins, y_bins, dBQgrid.T, vmin=-0.1, vmax=0.15, cmap='RdBu')
        sub.text(0.05, 0.05, '0.0eV', ha='left', va='bottom', transform=sub.transAxes, fontsize=20)
        sub.text(0.975, 0.025, '$\sigma_8 = %.3f$' % sig8s[i], ha='right', va='bottom', 
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
    ffig = os.path.join(UT.doc_dir(), 'figs', 'haloBk_dshape_kmax%s%s.pdf' % 
            (str(kmax).replace('.', ''), ['', '_rsd'][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def compare_Bk_triangle(typ, rsd=True):  
    ''' Compare the amplitude of specific triangle configuration of the bispectrum 
    for the HADES simulations. 

    :param tri: 
        string that specifies the triangle. Options are 'equ' 
        and 'squ'

    :param rsd: (default: True) 
        RSD or not 
    '''
    mnus = [0.06, 0.1, 0.15]
    sig8s = [0.822, 0.818, 0.807, 0.798]

    hades_fid = Obvs.hadesBk(0.0, nzbin=4, rsd=rsd) # fiducial bispectrum
    Bk_fid = np.average(hades_fid['b123'], axis=0)
    Bk_sn = np.average(hades_fid['b_sn'], axis=0) 
    i_k, j_k, l_k = hades_fid['k1'], hades_fid['k2'], hades_fid['k3']

    Bk_Mnu, Bk_s8s = [], [] 
    for mnu in mnus: # bispectrum for different Mnu
        hades_i = Obvs.hadesBk(mnu, nzbin=4, rsd=rsd)
        Bk_Mnu.append(np.average(hades_i['b123'], axis=0)) 
    for sig8 in sig8s: # bispectrum for different sigma8 
        hades_i = Obvs.hadesBk_s8(sig8, nzbin=4, rsd=rsd)
        Bk_s8s.append(np.average(hades_i['b123'], axis=0)) 

    kf = 2.*np.pi/1000.
    if typ == 'equ': # only equilateral 
        tri = (i_k == j_k) & (j_k == l_k) 
    elif typ == 'squ': # i_k >= j_k >= l_k (= 3*kf~0.01) 
        tri = (i_k == j_k) & (l_k == 3)
    
    # plot 
    fig = plt.figure(figsize=(10,8))
    gs = mpl.gridspec.GridSpec(5,1, figure=fig) 
    sub = plt.subplot(gs[:3,0]) 
    sub2 = plt.subplot(gs[-2:,0]) 
    for ii, mnu, bk in zip(range(4), [0.0]+mnus, [Bk_fid]+Bk_Mnu):
        if mnu == 0.0: 
            sub.fill_between(kf * i_k[tri], bk[tri] - sigBk, bk[tri] + sigBk, 
                    color='C'+str(ii), alpha=0.25, linewidth=0)
            sub2.fill_between(kf * i_k[tri], np.ones(np.sum(tri)), 1.+np.abs(sigBk/Bk_fid[tri]), 
                    color='C'+str(ii), alpha=0.25, linewidth=0)
        sub.plot(kf * i_k[tri], bk[tri], c='C'+str(ii), label=str(mnu)+'eV') 
        if mnu > 0.0: sub2.plot(kf * i_k[tri], bk[tri]/Bk_fid[tri], c='C'+str(ii)) 

    for ii, sig8, bk in zip([4, 6, 8, 9], sig8s, Bk_s8s):
        sub.plot(kf * i_k[tri], bk[tri], c='C'+str(ii), ls='--', label='$\sigma_8=$'+str(sig8)) 
        sub2.plot(kf * i_k[tri], bk[tri]/Bk_fid[tri], c='C'+str(ii), ls='--') 
    sub2.plot([1e-4, 1e3], [1., 1.], c='k', ls='--', lw=2) 
    sub.plot(kf * i_k[tri], Bk_sn[tri], c='k', ls=':')
    
    sub.legend(loc='upper right', ncol=2, columnspacing=0.2, 
            markerscale=4, handletextpad=0.25, fontsize=18) 
    sub.set_xscale('log') 
    sub.set_xlim([1e-2, 1e0])
    sub.set_xticklabels([]) 
    sub.set_yscale('log') 
    sub.set_ylim([1e5, 1e10]) 

    sub2.set_xscale('log')
    sub2.set_xlim([1e-2, 1e0])
    sub2.set_ylim([0.95, 1.25]) 
    if typ == 'equ': 
        sub.set_ylabel('$B(k, k, k)$', fontsize=25) 
        sub2.set_ylabel('$B(k, k, k)/B^\mathrm{(fid)}$', fontsize=25) 
        sub.set_title('{\em equilateral} configuration', pad=10, fontsize=25) 
    elif typ == 'squ': 
        sub.set_ylabel('$B(k_1 = %.3f, k_2, k_2)$' % (3.*kf), fontsize=20) 
        sub2.set_ylabel('$B(k_1 = %.3f, k_2, k_2)/B^\mathrm{(fid)}$' % (3.*kf), fontsize=25) 
        sub.set_title('{\em squeezed} configuration', pad=10, fontsize=25) 
    
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    if typ == 'equ': bkgd.set_xlabel(r'$k$ [$h$/Mpc]', labelpad=10, fontsize=25) 
    elif typ == 'squ': bkgd.set_xlabel(r'$k_2$ [$h$/Mpc]', labelpad=10, fontsize=25) 

    fig.subplots_adjust(hspace=0.1)
    ffig = os.path.join(UT.doc_dir(), 'figs', 'haloBk_amp_%s%s.pdf' % (typ, ['', '_rsd'][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    ffig = os.path.join(UT.doc_dir(), 'figs', 'haloBk_amp_%s%s.png' % (typ, ['', '_rsd'][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def compare_Bk_triangles(ik_pairs, rsd=True): 
    ''' Make various bispectrum plots as a function of m_nu 
    '''
    mnus = [0.06, 0.1, 0.15]
    sig8s = [0.822, 0.818, 0.807, 0.798]

    hades_fid = Obvs.hadesBk(0.0, nzbin=4, rsd=rsd) # fiducial bispectrum
    Bk_fid = np.average(hades_fid['b123'], axis=0)

    Bk_Mnu, Bk_s8s = [], [] 
    for mnu in mnus: # bispectrum for different Mnu
        hades_i = Obvs.hadesBk(mnu, nzbin=4, rsd=rsd)
        Bk_Mnu.append(np.average(hades_i['b123'], axis=0)) 
    for sig8 in sig8s: # bispectrum for different sigma8 
        hades_i = Obvs.hadesBk_s8(sig8, nzbin=4, rsd=rsd)
        Bk_s8s.append(np.average(hades_i['b123'], axis=0)) 
    
    kf = 2.*np.pi/1000.
    i_k, j_k, l_k = hades_i['k1'], hades_i['k2'], hades_i['k3']

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
        for ii, mnu, bk in zip(range(1,4), mnus, Bk_Mnu): 
            lbl = None 
            if i_p == 0: lbl = str(mnu)+'eV'
            _bk = (bk[klim][theta_sort]/Bk_fid[klim][theta_sort]) - 1. 
            sub.plot(theta12[theta_sort]/np.pi, _bk, c='C'+str(ii), label=lbl)
            sub.set_xlim([0., 1.]) 

        if i_p == 0: sub.legend(loc='upper left', handletextpad=0.2, fontsize=20) 

        for i, ii, sig8, bk in zip(range(len(sig8s)), [4, 6, 8, 9], sig8s, Bk_s8s): 
            _bk = (bk[klim][theta_sort]/Bk_fid[klim][theta_sort]) - 1.
            if i < 3: 
                sub.plot(theta12[theta_sort]/np.pi, _bk, c='C'+str(ii), lw=1, ls=(0,(5,6)), label='$\sigma_8 = %.3f$' % sig8)
            else: 
                sub.plot(theta12[theta_sort]/np.pi, _bk, c='k', ls=(0,(1,6)), lw=0.5, label='$\sigma_8 = %.3f$' % sig8)
            sub.set_xlim([0., 1.]) 
        sub.set_ylim([0., 0.15]) 
        sub.set_yticks([0., 0.05, 0.1, 0.15]) 
        sub.set_title('$k_1 = %.2f,\,\,k_2 = %.2f$' % (ik1*kf, ik2*kf), fontsize=25)
        if i_p > 0: sub.set_yticklabels([]) 
        if i_p == 1: sub.legend(loc='upper left', handletextpad=0.2, fontsize=20) 
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'$\theta_{12}/\pi$', labelpad=10, fontsize=30) 
    bkgd.set_ylabel((r'$B(k_1, k_2, \theta_{12}) / B^\mathrm{fid} - 1$'), labelpad=10, fontsize=25) 
    fig.subplots_adjust(wspace=0.1) 
    ffig = os.path.join(UT.doc_dir(), 'figs', 'haloBk_triangles%s.pdf' % (['', '_rsd'][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def B123_kmax_test(kmin=0.01, kmax1=0.4, kmax2=0.5, rsd=True): 
    ''' bispectrum for triangle configurations that are outside 
    the k-range kmin < k < kmax1 but inside kmin < k < kmax2. This 
    is to better understand the extra signal that comes from going 
    to higher kmax
    '''
    kf = 2.*np.pi/1000.
    hades = Obvs.hadesBk(0.0, nzbin=4, rsd=rsd)  
    i_k, j_k, l_k, bk = hades['k1'], hades['k2'], hades['k3'], hades['b123'] 

    quijo = quijoteBk('fiducial')
    bk_q = quijo['b123'] 

    # get triangles outside of klim1 but inside klim2 
    klim1 = ((i_k*kf <= kmax1) & (i_k*kf >= kmin) &
            (j_k*kf <= kmax1) & (j_k*kf >= kmin) & 
            (l_k*kf <= kmax1) & (l_k*kf >= kmin)) 
    
    klim2 = ((i_k*kf <= kmax2) & (i_k*kf >= kmin) &
            (j_k*kf <= kmax2) & (j_k*kf >= kmin) & 
            (l_k*kf <= kmax2) & (l_k*kf >= kmin)) 
    klim = ~klim1 & klim2
    
    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim]
    ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # order the triangles 
    
    bk = bk[:,klim][:,ijl]
    bk_q = bk_q[:,klim][:,ijl] 
    
    mu_bk = np.average(bk, axis=0) 
    sig_bk = np.std(bk_q, axis=0)
    
    fig = plt.figure(figsize=(25,5))
    sub = fig.add_subplot(111)
    sub.fill_between(np.arange(np.sum(klim)), mu_bk - sig_bk, mu_bk + sig_bk, 
            color='C0', alpha=0.25, linewidth=0)
    sub.plot(np.arange(np.sum(klim)), mu_bk, c='C0')
    sub.set_title(r'$%.1f < k_{\rm max} < %.1f$' % (kmax1, kmax2), fontsize=25) 
    sub.set_yscale('log') 
    sub.set_xlim([0, np.sum(klim)])
    sub.set_ylim([1e6, 1e9]) 
        
    fig.savefig(os.path.join(UT.doc_dir(), 'figs', 
        'B123_kmax_test_%.1f_%.1f_%.1f_%sspace.pdf' % (kmin, kmax1, kmax2, ['r', 'z'][rsd])), 
        bbox_inches='tight') 
    return None

##################################################################
# shotnoise uncorrected comparison 
##################################################################
def compare_Bk_SNuncorr(krange=[0.01, 0.5], rsd=True):  
    ''' Compare the amplitude of the shot-noise uncorrected bispectrum of 
    the HADES simulations. 

    :param krange: (default: [0.01, 0.5]) 
        k-range of k1, k2, k3 

    :param rsd: (default: True) 
        RSD or not 
    '''
    mnus = [0.06, 0.1, 0.15]
    sig8s = [0.822, 0.818, 0.807, 0.798]

    hades_fid = Obvs.hadesBk(0.0, nzbin=4, rsd=rsd) # fiducial bispectrum
    Bk_fid = np.average(hades_fid['b123'] + hades_fid['b_sn'], axis=0)

    Bk_Mnu, Bk_s8s = [], [] 
    for mnu in mnus: # bispectrum for different Mnu
        hades_i = Obvs.hadesBk(mnu, nzbin=4, rsd=rsd)
        Bk_Mnu.append(np.average(hades_i['b123'] + hades_i['b_sn'], axis=0)) 
    for sig8 in sig8s: # bispectrum for different sigma8 
        hades_i = Obvs.hadesBk_s8(sig8, nzbin=4, rsd=rsd)
        Bk_s8s.append(np.average(hades_i['b123'] + hades_i['b_sn'], axis=0)) 

    # covariance matrix
    C_full = quijoteCov()

    i_k, j_k, l_k = hades_i['k1'], hades_i['k2'], hades_i['k3']
    kf = 2.*np.pi/1000. 
    kmin, kmax = krange # impose k range 
    klim = ((i_k * kf <= kmax) & (i_k * kf >= kmin) &
            (j_k * kf <= kmax) & (j_k * kf >= kmin) & 
            (l_k * kf <= kmax) & (l_k * kf >= kmin)) 

    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 

    ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # order of triangles 

    sigBk = np.sqrt(np.diag(C_full))[klim][ijl] 

    fig = plt.figure(figsize=(25,10))
    sub = fig.add_subplot(211)
    axins = inset_axes(sub, loc='upper right', width="40%", height="45%") 
    sub2 = fig.add_subplot(212)
    axins2 = inset_axes(sub2, loc='upper right', width="40%", height="45%") 
    
    tri = np.arange(np.sum(klim))
    for ii, mnu, bk in zip(range(4), [0.0]+mnus, [Bk_fid]+Bk_Mnu):
        _bk = bk[klim][ijl]
        if mnu == 0.0: 
            sub.fill_between(tri, _bk - sigBk, _bk + sigBk, color='C'+str(ii), alpha=0.25, linewidth=0.) 
            axins.fill_between(tri, _bk - sigBk, _bk + sigBk, color='C'+str(ii), alpha=0.25, linewidth=0.) 
            sub2.fill_between(tri, _bk - sigBk, _bk + sigBk, color='C'+str(ii), alpha=0.25, linewidth=0.) 
            axins2.fill_between(tri, _bk - sigBk, _bk + sigBk, color='C'+str(ii), alpha=0.25, linewidth=0.) 
            sub2.plot(tri, _bk, c='C0') 
            axins2.plot(tri, _bk, c='C0') 
        sub.plot(tri, _bk, c='C'+str(ii), label=str(mnu)+'eV') 
        axins.plot(tri, _bk, c='C'+str(ii), label=str(mnu)+'eV') 

    for ii, sig8, bk in zip([4, 6, 8, 9], sig8s, Bk_s8s):
        _bk = bk[klim][ijl]
        sub2.plot(tri, _bk, c='C'+str(ii), label='$\sigma_8=%.3f$' % sig8) 
        axins2.plot(tri, _bk, c='C'+str(ii)) 

    sub2.text(0.02, 0.15, '0.0 eV', ha='left', va='bottom', transform=sub2.transAxes, fontsize=20)
    sub.legend(loc='lower left', ncol=4, columnspacing=0.5, markerscale=4, handletextpad=0.25, fontsize=20) 
    sub2.legend(loc='lower left', ncol=4, columnspacing=0.5, markerscale=4, handletextpad=0.25, fontsize=20) 

    sub.set_yscale('log') 
    axins.set_yscale('log') 
    
    sub2.set_yscale('log') 
    axins2.set_yscale('log') 
    if rsd:
        sub.set_xlim([0, 1898])
        sub.set_ylim([5e7, 1e10]) 
        axins.set_xlim(480, 500)
        axins.set_ylim(2e8, 5e8) 
        sub2.set_xlim([0, 1898])
        sub2.set_ylim([5e7, 1e10]) 
        axins2.set_xlim(480, 500)
        axins2.set_ylim(2e8, 5e8) 
    else: 
        #sub.set_xlim([0, 1200])
        #sub.set_ylim([1e5, 5e9]) 
        sub.set_xlim([0, 1898])
        sub.set_ylim([1e6, 1e10]) 
        axins.set_xlim(480, 500)
        axins.set_ylim(2e7, 1.5e8) 
        sub2.set_xlim([0, 1200])
        sub2.set_ylim([1e5, 5e9]) 
        axins2.set_xlim(480, 500)
        axins2.set_ylim(2e7, 1.5e8) 

    axins.set_xticklabels('') 
    axins.yaxis.set_minor_formatter(NullFormatter())
    mark_inset(sub, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    axins2.set_xticklabels('') 
    axins2.yaxis.set_minor_formatter(NullFormatter())
    mark_inset(sub2, axins2, loc1=2, loc2=4, fc="none", ec="0.5")
    
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'triangle configurations', labelpad=15, fontsize=25) 
    bkgd.set_ylabel('$B(k_1, k_2, k_3)$', labelpad=10, fontsize=25) 
    fig.subplots_adjust(hspace=0.15)

    krange_str = str(kmin).replace('.', '')+'_'+str(kmax).replace('.', '') 
    ffig = os.path.join(UT.fig_dir(), 
            'haloBkSNuncorr_amp_%s_%s%s.png' % 
            (str(kmin).replace('.', ''), str(kmax).replace('.', ''), ['', '_rsd'][rsd]))
    print ffig
    fig.savefig(ffig, bbox_inches='tight') 
    
    # compare ratio of B(k) amplitude
    fig = plt.figure(figsize=(18,18))

    bk_fid = Bk_fid[klim][ijl]
    for i, mnu, sig8, bk, bks8 in zip(range(3), mnus, sig8s, Bk_Mnu, Bk_s8s): 
        sub = fig.add_subplot(3,1,i+1) 

        db = bk[klim][ijl]/bk_fid - 1.
        sub.plot(tri, db, lw=2, c='C'+str(i+1), label=str(mnu)+'eV') 
        #if i == 0: sub.fill_between(tri, np.zeros(np.sum(klim)), sigBk/bk_fid, color='k', alpha=0.25, linewidth=0.) 
    
        db = bks8[klim][ijl]/bk_fid - 1.
        sub.plot(tri, db, lw=1, c='k', label='0.0 eV\n$\sigma_8=%.3f$' % sig8) 
        sub.plot([0, np.sum(klim)], [0., 0.], c='k', ls='--', lw=2)
        if rsd: 
            sub.legend(loc='upper left', ncol=2, markerscale=4, handletextpad=0.5, fontsize=25) 
            sub.set_xlim([0, np.sum(klim)])
            sub.set_ylim([-0.01, 0.25]) 
            sub.set_yticks([0., 0.05, 0.1, 0.15, 0.2, 0.25]) 
        else: 
            sub.legend(loc='upper left', frameon=True, ncol=2, markerscale=4, handletextpad=0.5, fontsize=25) 
            sub.set_xlim([0, 1200])
            sub.set_ylim([-0.01, 0.25]) 
            sub.set_yticks([0., 0.05, 0.1, 0.15, 0.2, 0.25]) 

        if i < 2: sub.set_xticklabels([]) 
    
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'triangle configurations', labelpad=15, fontsize=25) 
    bkgd.set_ylabel('$(B(k_1, k_2, k_3) - B^\mathrm{(fid)})/B^\mathrm{(fid)}$', labelpad=15, fontsize=30) 
    fig.subplots_adjust(hspace=0.1)
    ffig = os.path.join(UT.fig_dir(), 
            'haloBkSNuncorr_residual_%s_%s%s.png' % 
            (str(kmin).replace('.', ''), str(kmax).replace('.', ''), ['', '_rsd'][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def compare_Qk_SNuncorr(krange=[0.01, 0.5], rsd=True):  
    ''' Compare the shot noise uncorrected reduced bispectrum for the HADES simulations. 

    :param krange: (default: [0.01, 0.5]) 
        k-range of k1, k2, k3 

    :param rsd: (default: True) 
        RSD or not 
    '''
    mnus = [0.06, 0.1, 0.15]
    sig8s = [0.822, 0.818, 0.807, 0.798]

    hades_fid = Obvs.hadesBk(0.0, nzbin=4, rsd=rsd) # fiducial bispectrum
    bk = np.average((hades_fid['b123'] + hades_fid['b_sn']), axis=0)
    pk1 = np.average(hades_fid['p0k1'] + 1e9/hades_fid['Nhalos'][:,None], axis=0)
    pk2 = np.average(hades_fid['p0k2'] + 1e9/hades_fid['Nhalos'][:,None], axis=0)
    pk3 = np.average(hades_fid['p0k3'] + 1e9/hades_fid['Nhalos'][:,None], axis=0)
    Qk_fid = bk / (pk1 * pk2 + pk1 * pk3 + pk2 * pk3)

    Qk_Mnu, Qk_s8s = [], [] 
    for mnu in mnus: # bispectrum for different Mnu
        hades_i = Obvs.hadesBk(mnu, nzbin=4, rsd=rsd)
        bk = np.average((hades_i['b123'] + hades_i['b_sn']), axis=0)
        pk1 = np.average(hades_i['p0k1'] + 1e9/hades_i['Nhalos'][:,None], axis=0)
        pk2 = np.average(hades_i['p0k2'] + 1e9/hades_i['Nhalos'][:,None], axis=0)
        pk3 = np.average(hades_i['p0k3'] + 1e9/hades_i['Nhalos'][:,None], axis=0)
        Qk_Mnu.append(bk / (pk1 * pk2 + pk1 * pk3 + pk2 * pk3))
    for sig8 in sig8s: # bispectrum for different sigma8 
        hades_i = Obvs.hadesBk_s8(sig8, nzbin=4, rsd=rsd)
        bk = np.average((hades_i['b123'] + hades_i['b_sn']), axis=0)
        pk1 = np.average(hades_i['p0k1'] + 1e9/hades_i['Nhalos'][:,None], axis=0)
        pk2 = np.average(hades_i['p0k2'] + 1e9/hades_i['Nhalos'][:,None], axis=0)
        pk3 = np.average(hades_i['p0k3'] + 1e9/hades_i['Nhalos'][:,None], axis=0)
        Qk_s8s.append(bk / (pk1 * pk2 + pk1 * pk3 + pk2 * pk3))

    i_k, j_k, l_k = hades_i['k1'], hades_i['k2'], hades_i['k3']
    kf = 2.*np.pi/1000. 
    kmin, kmax = krange # impose k range 
    klim = ((i_k * kf <= kmax) & (i_k * kf >= kmin) &
            (j_k * kf <= kmax) & (j_k * kf >= kmin) & 
            (l_k * kf <= kmax) & (l_k * kf >= kmin)) 

    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 

    ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # order of triangles 

    fig = plt.figure(figsize=(25,10))
    sub = fig.add_subplot(211)
    axins = inset_axes(sub, loc='upper right', width="40%", height="45%") 
    sub2 = fig.add_subplot(212)
    axins2 = inset_axes(sub2, loc='upper right', width="40%", height="45%") 
    
    tri = np.arange(np.sum(klim))
    for ii, mnu, qk in zip(range(4), [0.0]+mnus, [Qk_fid]+Qk_Mnu):
        _qk = qk[klim][ijl]
        if mnu == 0.0: 
            sub2.plot(tri, _qk, c='C0') 
            axins2.plot(tri, _qk, c='C0') 
        sub.plot(tri, _qk, c='C'+str(ii), label=str(mnu)+'eV') 
        axins.plot(tri, _qk, c='C'+str(ii), label=str(mnu)+'eV') 

    for ii, sig8, qk in zip([4, 6, 8, 9], sig8s, Qk_s8s):
        _qk = qk[klim][ijl]
        sub2.plot(tri, _qk, c='C'+str(ii), label='$\sigma_8=%.3f$' % sig8) 
        axins2.plot(tri, _qk, c='C'+str(ii)) 

    sub2.text(0.02, 0.15, '0.0 eV', ha='left', va='bottom', transform=sub2.transAxes, fontsize=20)
    sub.legend(loc='lower left', ncol=4, columnspacing=0.5, markerscale=4, handletextpad=0.25, fontsize=20) 
    sub2.legend(loc='lower left', ncol=4, columnspacing=0.5, markerscale=4, handletextpad=0.25, fontsize=20) 

    sub.set_xlim([0, 1898])
    sub.set_ylim([0, 1]) 
    axins.set_xlim(480, 500)
    axins.set_ylim(0.3, 0.5) 
    sub2.set_xlim([0, 1898])
    sub2.set_ylim([0., 1.]) 
    axins2.set_xlim(480, 500)
    axins2.set_ylim(0.3, 0.5) 

    axins.set_xticklabels('') 
    axins.yaxis.set_minor_formatter(NullFormatter())
    mark_inset(sub, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    axins2.set_xticklabels('') 
    axins2.yaxis.set_minor_formatter(NullFormatter())
    mark_inset(sub2, axins2, loc1=2, loc2=4, fc="none", ec="0.5")
    
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'triangle configurations', labelpad=15, fontsize=25) 
    bkgd.set_ylabel('$Q(k_1, k_2, k_3)$', labelpad=10, fontsize=25) 
    fig.subplots_adjust(hspace=0.15)

    krange_str = str(kmin).replace('.', '')+'_'+str(kmax).replace('.', '') 
    ffig = os.path.join(UT.fig_dir(), 'haloQkSNuncorr_amp_%s_%s%s.pdf' % 
            (str(kmin).replace('.', ''), str(kmax).replace('.', ''), ['', '_rsd'][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 

    # compare ratio of Q(k) amplitude
    fig = plt.figure(figsize=(18,18))

    qk_fid = Qk_fid[klim][ijl]
    for i, mnu, sig8, qk, qks8 in zip(range(3), mnus, sig8s, Qk_Mnu, Qk_s8s): 
        sub = fig.add_subplot(3,1,i+1) 

        dq = qk[klim][ijl]/qk_fid - 1.
        sub.plot(tri, dq, lw=2, c='C'+str(i+1), label=str(mnu)+'eV') 
        #if i == 0: sub.fill_between(tri, np.zeros(np.sum(klim)), sigBk/bk_fid, color='k', alpha=0.25, linewidth=0.) 
    
        dq = qks8[klim][ijl]/qk_fid - 1.
        sub.plot(tri, dq, lw=1, c='k', label='0.0 eV\n$\sigma_8=%.3f$' % sig8) 
        sub.plot([0, np.sum(klim)], [0., 0.], c='k', ls='--', lw=2)
        sub.legend(loc='upper left', ncol=2, markerscale=4, handletextpad=0.5, fontsize=25) 
        sub.set_xlim([0, np.sum(klim)])
        sub.set_ylim([-0.01, 0.15]) 
        sub.set_yticks([0., 0.05, 0.1, 0.15]) 
        if i < 2: sub.set_xticklabels([]) 
    
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'triangle configurations', labelpad=15, fontsize=25) 
    bkgd.set_ylabel('$(Q(k_1, k_2, k_3) - Q^\mathrm{(fid)})/Q^\mathrm{(fid)}$', labelpad=15, fontsize=30) 
    fig.subplots_adjust(hspace=0.1)
    ffig = os.path.join(UT.fig_dir(), 'haloQkSNuncorr_residual_%s_%s%s.pdf' % 
            (str(kmin).replace('.', ''), str(kmax).replace('.', ''), ['', '_rsd'][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


if __name__=="__main__": 
    for kmax in [0.5]: 
        #compare_Plk(kmax=0.5)
        #ratio_Plk(kmax=0.5) 
        compare_Bk(kmax=kmax, rsd=True)
        compare_Bk_shape(kmax=kmax, rsd=True, nbin=31)
        compare_Qk(kmax=kmax, rsd=True)

    #compare_Bk_SNuncorr(krange=[0.01, 0.5], rsd=True)
    #compare_Qk_SNuncorr(krange=[0.01, 0.5], rsd=True)
