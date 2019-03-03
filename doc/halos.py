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


quijote_thetas = {
        'Mnu': [0.1, 0.2, 0.4], # +, ++, +++ 
        'Ob': [0.048, 0.050], # others are - + 
        'Om': [0.3075, 0.3275],
        'h': [0.6511, 0.6911],
        'ns': [0.9424, 0.9824],
        's8': [0.819, 0.849]} 
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
def hadesBk(mneut, nzbin=4, rsd=True):
    ''' read in bispectrum of hades massive neutrino halo catalogs 
    with Mnu = mneut eV and at nzbin redshift bin 

    :param mneut: 
        neutrino mass [0.0, 0.06, 0.1, 0.15]

    :param nzbin: (default: 4) 
        redshift bin number 
    '''
    dir_bk = os.path.join(UT.dat_dir(), 'bispectrum') 
    fbks = os.path.join(dir_bk, 
            'hades.%seV.nzbin%i.mhmin3200.0.%sspace.hdf5' % (str(mneut), nzbin, ['r', 'z'][rsd]))
    bks = h5py.File(fbks, 'r') 
    
    _bks = {}
    for k in bks.keys(): 
        _bks[k] = bks[k].value 
    return _bks


def hadesBk_s8(sig8, nzbin=4, rsd=True):
    ''' read in bispectrum of sigma8 matched Mnu = 0.0 eV halo catalogs
    
    :param sig8: 
        sigma_8 values [0.822, 0.818, 0.807, 0.798]

    :param nzbin: (default: 4) 
        redshift bin number 
    '''
    dir_bk = os.path.join(UT.dat_dir(), 'bispectrum') 
    fbks = os.path.join(dir_bk, 
            'hades.0.0eV.sig8_%s.nzbin%i.mhmin3200.0.%sspace.hdf5' % (str(sig8), nzbin, ['r', 'z'][rsd]))
    bks = h5py.File(fbks, 'r') 
    
    _bks = {}
    for k in bks.keys(): 
        _bks[k] = bks[k].value 
    return _bks


def compare_Bk(kmax=0.5, rsd=True):  
    ''' Compare the amplitude of the bispectrum for the HADES simulations. 

    :param krange: (default: [0.01, 0.5]) 
        k-range of k1, k2, k3 

    :param rsd: (default: True) 
        RSD or not 
    '''
    mnus = [0.06, 0.1, 0.15]
    sig8s = [0.822, 0.818, 0.807, 0.798]

    hades_fid = hadesBk(0.0, nzbin=4, rsd=rsd) # fiducial bispectrum
    Bk_fid = np.average(hades_fid['b123'], axis=0)
    Bk_sn = np.average(hades_fid['b_sn'], axis=0) 

    Bk_Mnu, Bk_s8s = [], [] 
    for mnu in mnus: # bispectrum for different Mnu
        hades_i = hadesBk(mnu, nzbin=4, rsd=rsd)
        Bk_Mnu.append(np.average(hades_i['b123'], axis=0)) 
    for sig8 in sig8s: # bispectrum for different sigma8 
        hades_i = hadesBk_s8(sig8, nzbin=4, rsd=rsd)
        Bk_s8s.append(np.average(hades_i['b123'], axis=0)) 

    # covariance matrix
    C_full = quijoteCov()

    i_k, j_k, l_k = hades_i['k1'], hades_i['k2'], hades_i['k3']
    kf = 2.*np.pi/1000. 
    klim = ((i_k * kf <= kmax) & (j_k * kf <= kmax) & (l_k * kf <= kmax)) 

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

    krange_str = str(kmin).replace('.', '')+'_'+str(kmax).replace('.', '') 
    ffig = os.path.join(UT.doc_dir(), 'figs', 
            'haloBk_amp_%s_%s%s.pdf' % 
            (str(kmin).replace('.', ''), str(kmax).replace('.', ''), ['', '_rsd'][rsd]))
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
    ffig = os.path.join(UT.doc_dir(), 'figs', 
            'haloBk_residual_%s_%s%s.pdf' % 
            (str(kmin).replace('.', ''), str(kmax).replace('.', ''), ['', '_rsd'][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def compare_Qk(krange=[0.01, 0.5], rsd=True):  
    ''' Compare the reduced bispectrum for the HADES simulations. 

    :param krange: (default: [0.01, 0.5]) 
        k-range of k1, k2, k3 

    :param rsd: (default: True) 
        RSD or not 
    '''
    mnus = [0.06, 0.1, 0.15]
    sig8s = [0.822, 0.818, 0.807, 0.798]

    hades_fid = hadesBk(0.0, nzbin=4, rsd=rsd) # fiducial bispectrum
    Qk_fid = np.average(hades_fid['q123'], axis=0)

    Qk_Mnu, Qk_s8s = [], [] 
    for mnu in mnus: # bispectrum for different Mnu
        hades_i = hadesBk(mnu, nzbin=4, rsd=rsd)
        Qk_Mnu.append(np.average(hades_i['q123'], axis=0)) 
    for sig8 in sig8s: # bispectrum for different sigma8 
        hades_i = hadesBk_s8(sig8, nzbin=4, rsd=rsd)
        Qk_s8s.append(np.average(hades_i['q123'], axis=0)) 

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
    ffig = os.path.join(UT.doc_dir(), 'figs', 
            'haloQk_amp_%s_%s%s.pdf' % 
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
    ffig = os.path.join(UT.doc_dir(), 'figs', 
            'haloQk_residual_%s_%s%s.pdf' % 
            (str(kmin).replace('.', ''), str(kmax).replace('.', ''), ['', '_rsd'][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def compare_Bk_shape(krange=[0.01, 0.5], rsd=True, nbin=31): 
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

    hades_fid = hadesBk(0.0, nzbin=4, rsd=rsd) # fiducial bispectrum
    Bk_fid = np.average(hades_fid['b123'], axis=0)
    counts = hades_fid['counts']

    Bk_Mnu, Bk_s8s = [], [] 
    for mnu in mnus: # bispectrum for different Mnu
        hades_i = hadesBk(mnu, nzbin=4, rsd=rsd)
        Bk_Mnu.append(np.average(hades_i['b123'], axis=0)) 
    for sig8 in sig8s: # bispectrum for different sigma8 
        hades_i = hadesBk_s8(sig8, nzbin=4, rsd=rsd)
        Bk_s8s.append(np.average(hades_i['b123'], axis=0)) 

    i_k, j_k, l_k = hades_i['k1'], hades_i['k2'], hades_i['k3']
    k3k1 = l_k.astype(float)/i_k.astype(float)
    k2k1 = j_k.astype(float)/i_k.astype(float)
    kf = 2.*np.pi/1000. 
    kmin, kmax = krange # impose k range 
    klim = ((i_k * kf <= kmax) & (i_k * kf >= kmin) &
            (j_k * kf <= kmax) & (j_k * kf >= kmin) & 
            (l_k * kf <= kmax) & (l_k * kf >= kmin)) 
        
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
    ffig = os.path.join(UT.doc_dir(), 'figs', 'haloBk_shape_%s_%s%s.pdf' % 
            (str(kmin).replace('.', ''), str(kmax).replace('.', ''), ['', '_rsd'][rsd]))
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
    ffig = os.path.join(UT.doc_dir(), 'figs', 'haloBk_dshape_%s_%s%s.pdf' % 
            (str(kmin).replace('.', ''), str(kmax).replace('.', ''), ['', '_rsd'][rsd]))
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

    hades_fid = hadesBk(0.0, nzbin=4, rsd=rsd) # fiducial bispectrum
    Bk_fid = np.average(hades_fid['b123'], axis=0)
    Bk_sn = np.average(hades_fid['b_sn'], axis=0) 

    Bk_Mnu, Bk_s8s = [], [] 
    for mnu in mnus: # bispectrum for different Mnu
        hades_i = hadesBk(mnu, nzbin=4, rsd=rsd)
        Bk_Mnu.append(np.average(hades_i['b123'], axis=0)) 
    for sig8 in sig8s: # bispectrum for different sigma8 
        hades_i = hadesBk_s8(sig8, nzbin=4, rsd=rsd)
        Bk_s8s.append(np.average(hades_i['b123'], axis=0)) 

    # covariance matrix
    i_k, j_k, l_k, C_full = quijoteCov()

    kf = 2.*np.pi/1000.
    if typ == 'equ': # only equilateral 
        tri = (i_k == j_k) & (j_k == l_k) 
    elif typ == 'squ': # i_k >= j_k >= l_k (= 3*kf~0.01) 
        tri = (i_k == j_k) & (l_k == 3)
    sigBk = np.sqrt(np.diag(C_full))[tri] 
    
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

    hades_fid = hadesBk(0.0, nzbin=4, rsd=rsd) # fiducial bispectrum
    Bk_fid = np.average(hades_fid['b123'], axis=0)

    Bk_Mnu, Bk_s8s = [], [] 
    for mnu in mnus: # bispectrum for different Mnu
        hades_i = hadesBk(mnu, nzbin=4, rsd=rsd)
        Bk_Mnu.append(np.average(hades_i['b123'], axis=0)) 
    for sig8 in sig8s: # bispectrum for different sigma8 
        hades_i = hadesBk_s8(sig8, nzbin=4, rsd=rsd)
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
    hades = hadesBk(0.0, nzbin=4, rsd=rsd)  
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

    hades_fid = hadesBk(0.0, nzbin=4, rsd=rsd) # fiducial bispectrum
    Bk_fid = np.average(hades_fid['b123'] + hades_fid['b_sn'], axis=0)

    Bk_Mnu, Bk_s8s = [], [] 
    for mnu in mnus: # bispectrum for different Mnu
        hades_i = hadesBk(mnu, nzbin=4, rsd=rsd)
        Bk_Mnu.append(np.average(hades_i['b123'] + hades_i['b_sn'], axis=0)) 
    for sig8 in sig8s: # bispectrum for different sigma8 
        hades_i = hadesBk_s8(sig8, nzbin=4, rsd=rsd)
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

    hades_fid = hadesBk(0.0, nzbin=4, rsd=rsd) # fiducial bispectrum
    bk = np.average((hades_fid['b123'] + hades_fid['b_sn']), axis=0)
    pk1 = np.average(hades_fid['p0k1'] + 1e9/hades_fid['Nhalos'][:,None], axis=0)
    pk2 = np.average(hades_fid['p0k2'] + 1e9/hades_fid['Nhalos'][:,None], axis=0)
    pk3 = np.average(hades_fid['p0k3'] + 1e9/hades_fid['Nhalos'][:,None], axis=0)
    Qk_fid = bk / (pk1 * pk2 + pk1 * pk3 + pk2 * pk3)

    Qk_Mnu, Qk_s8s = [], [] 
    for mnu in mnus: # bispectrum for different Mnu
        hades_i = hadesBk(mnu, nzbin=4, rsd=rsd)
        bk = np.average((hades_i['b123'] + hades_i['b_sn']), axis=0)
        pk1 = np.average(hades_i['p0k1'] + 1e9/hades_i['Nhalos'][:,None], axis=0)
        pk2 = np.average(hades_i['p0k2'] + 1e9/hades_i['Nhalos'][:,None], axis=0)
        pk3 = np.average(hades_i['p0k3'] + 1e9/hades_i['Nhalos'][:,None], axis=0)
        Qk_Mnu.append(bk / (pk1 * pk2 + pk1 * pk3 + pk2 * pk3))
    for sig8 in sig8s: # bispectrum for different sigma8 
        hades_i = hadesBk_s8(sig8, nzbin=4, rsd=rsd)
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

##################################################################
# qujiote fisher 
##################################################################
thetas = ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
theta_lbls = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$']
theta_lims = [(0.25, 0.385), (0.02, 0.08), (0.3, 1.1), (0.6, 1.3), (0.8, 0.88), (-0.45, 0.45)]
theta_fid = {'Mnu': 0., 'Ob': 0.049, 'Om': 0.3175, 'h': 0.6711,  'ns': 0.9624,  's8': 0.834} # fiducial theta 
ntheta = len(thetas)

def quijoteBk(theta, rsd=True): 
    ''' read in bispectra for specified theta of the quijote simulations. 
    
    :param theta: 
        string that specifies which quijote run. `theta==fiducial`  
        for the fiducial set of parameters. 

    :return _bks: 
        dictionary that contains all the bispectrum data from quijote simulation
    '''
    fbk = os.path.join(UT.dat_dir(), 'bispectrum', 'quijote_%s%s.hdf5' % (theta, ['.real', ''][rsd])) 
    bks = h5py.File(fbk, 'r') 

    _bks = {}
    for k in bks.keys(): 
        _bks[k] = bks[k].value 
    return _bks


def quijoteCov(rsd=True): 
    ''' return the full covariance matrix of the quijote bispectrum
    computed using 15000 simulations at the fiducial parameter values. 

    :return cov:
        big ass covariance matrix of all the triangle configurations in 
        the default ordering. 
    '''
    dir_bk = os.path.join(UT.dat_dir(), 'bispectrum')
    fcov = os.path.join(dir_bk, 'quijote_Cov_full%s.hdf5' % ['.real', ''][rsd])
    if os.path.isfile(fcov): 
        Fcov = h5py.File(fcov, 'r') # read in fiducial covariance matrix 
        cov = Fcov['C_bk'].value
        k1, k2, k3 = Fcov['k1'].value, Fcov['k2'].value, Fcov['k3'].value
    else: 
        fbks = h5py.File(os.path.join(dir_bk, 'quijote_fiducial%s.hdf5' % ['.real', ''][rsd]), 'r') 
        bks = fbks['b123'].value + fbks['b_sn'].value

        cov = np.cov(bks.T) # calculate the covariance
        k1, k2, k3 = fbks['k1'].value, fbks['k2'].value, fbks['k3'].value

        f = h5py.File(fcov, 'w') # write to hdf5 file 
        f.create_dataset('C_bk', data=cov) 
        f.create_dataset('k1', data=fbks['k1'].value) 
        f.create_dataset('k2', data=fbks['k2'].value) 
        f.create_dataset('k3', data=fbks['k3'].value) 
        f.close()
    return k1, k2, k3, cov


# covariance matrices 
def quijote_pkCov(kmax=0.5, rsd=True): 
    ''' plot the covariance matrix of the quijote fiducial 
    bispectrum. 
    '''
    # read in P(k) 
    quij = quijoteBk('fiducial', rsd=rsd) # theta_fiducial 
    pks = quij['p0k1'] + 1e9 / quij['Nhalos'][:,None]   # shotnoise uncorrected P(k) 
    i_k = quij['k1']
     
    # impose k limit on powerspectrum 
    kf = 2.*np.pi/1000. # fundmaentla mode
    _, _iuniq = np.unique(i_k, return_index=True)
    iuniq = np.zeros(len(i_k)).astype(bool) 
    iuniq[_iuniq] = True
    pklim = (iuniq & (i_k*kf <= kmax)) 
    pks = pks[:,pklim]
    
    C_pk = np.cov(pks.T) # covariance matrix 
    print('covariance matrix condition number = %.5e' % np.linalg.cond(C_pk)) 

    # plot the covariance matrix 
    fig = plt.figure(figsize=(10,8))
    sub = fig.add_subplot(111)
    cm = sub.pcolormesh(C_pk, norm=LogNorm(vmin=1e3, vmax=1e8))
    cbar = fig.colorbar(cm, ax=sub) 
    sub.set_title(r'Quijote $P(k)$ Covariance', fontsize=25)
    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_pkCov_kmax%s%s.png' % (str(kmax).replace('.', ''), ['_real', ''][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def quijote_bkCov(kmax=0.5, rsd=True): 
    ''' plot the covariance matrix of the quijote fiducial 
    bispectrum. 
    '''
    # read in B(k) of fiducial quijote simulation 
    quij = quijoteBk('fiducial', rsd=rsd) # theta_fiducial 
    bks = quij['b123'] + quij['b_sn']                   # shotnoise uncorrected B(k) 
    i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3']

    # impose k limit on bispectrum
    kf = 2.*np.pi/1000. # fundmaentla mode
    bklim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax))
    ijl = UT.ijl_order(i_k[bklim], j_k[bklim], l_k[bklim], typ='GM') # order of triangles 
    bks = bks[:,bklim][:,ijl] 

    C_bk = np.cov(bks.T) # covariance matrix 
    print('covariance matrix condition number = %.5e' % np.linalg.cond(C_bk)) 

    # plot the covariance matrix 
    fig = plt.figure(figsize=(10,8))
    sub = fig.add_subplot(111)
    cm = sub.pcolormesh(C_bk, norm=LogNorm(vmin=1e11, vmax=1e18))
    cbar = fig.colorbar(cm, ax=sub) 
    cbar.set_label(r'$B(k_1, k_2, k_3)$ covariance matrix, ${\bf C}_{B}$', fontsize=25, labelpad=10, rotation=90)
    #sub.set_title(r'Quijote $B(k_1, k_2, k_3)$ Covariance', fontsize=25)
    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_bkCov_kmax%s%s.png' % (str(kmax).replace('.', ''), ['_real', ''][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def quijote_pkbkCov(kmax=0.5, rsd=True): 
    ''' plot the covariance matrix of the quijote fiducial bispectrum. 
    '''
    # read in P(k) and B(k) 
    quij = quijoteBk('fiducial', rsd=rsd) # theta_fiducial 
    bks = quij['b123'] + quij['b_sn']                   # shotnoise uncorrected B(k) 
    pks = quij['p0k1'] + 1e9 / quij['Nhalos'][:,None]   # shotnoise uncorrected P(k) 
    i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3']
     
    # impose k limit on bispectrum
    kf = 2.*np.pi/1000. # fundmaentla mode
    bklim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 
    ijl = UT.ijl_order(i_k[bklim], j_k[bklim], l_k[bklim], typ='GM') # order of triangles 
    bks = bks[:,bklim][:,ijl] 
    # impose k limit on powerspectrum 
    _, _iuniq = np.unique(i_k, return_index=True)
    iuniq = np.zeros(len(i_k)).astype(bool) 
    iuniq[_iuniq] = True
    pklim = (iuniq & (i_k*kf <= kmax)) 
    pks = pks[:,pklim]
    
    pbks = np.concatenate([pks, bks], axis=1) # joint data vector

    C_pbk = np.cov(pbks.T) # covariance matrix 
    print('covariance matrix condition number = %.5e' % np.linalg.cond(C_pbk)) 

    # plot the covariance matrix 
    fig = plt.figure(figsize=(10,8))
    sub = fig.add_subplot(111)
    cm = sub.pcolormesh(C_pbk, norm=LogNorm(vmin=1e5, vmax=1e18))
    cbar = fig.colorbar(cm, ax=sub) 
    sub.set_title(r'Quijote $P(k)$ and $B(k_1, k_2, k_3)$ Covariance', fontsize=25)
    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_pbkCov_kmax%s%s.png' % 
            (str(kmax).replace('.', ''), ['_real', ''][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def _quijote_pkbkCov_triangle(typ, krange=[0.01, 0.5]): 
    ''' plot the covariance matrix of the quijote fiducial bispectrum. 
    '''
    # read in P(k) and B(k) 
    quij = quijoteBk('fiducial') # theta_fiducial 
    bks = quij['b123'] + quij['b_sn']                   # shotnoise uncorrected B(k) 
    pks = quij['p0k1'] + 1e9 / quij['Nhalos'][:,None]   # shotnoise uncorrected P(k) 
    i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3']
     
    # impose k limit on bispectrum
    kmin, kmax = krange 
    kf = 2.*np.pi/1000. # fundmaentla mode
    # impose k limit 
    if typ == 'equ': # only equilateral 
        tri = (i_k == j_k) & (j_k == l_k) 
    elif typ == 'squ': # i_k >= j_k >= l_k (= 3*kf~0.01) 
        tri = (i_k == j_k) & (l_k == 3)
    bklim = (tri & 
            (i_k*kf <= kmax) & (i_k*kf >= kmin) &
            (j_k*kf <= kmax) & (j_k*kf >= kmin) & 
            (l_k*kf <= kmax) & (l_k*kf >= kmin)) 
    ijl = UT.ijl_order(i_k[bklim], j_k[bklim], l_k[bklim], typ='GM') # order of triangles 
    bks = bks[:,bklim][:,ijl] 
    # impose k limit on powerspectrum 
    _, _iuniq = np.unique(i_k, return_index=True)
    iuniq = np.zeros(len(i_k)).astype(bool) 
    iuniq[_iuniq] = True
    pklim = (iuniq & (i_k*kf <= kmax) & (i_k*kf >= kmin)) 
    pks = pks[:,pklim]
    
    pbks = np.concatenate([pks, bks], axis=1) # joint data vector

    C_pbk = np.cov(pbks.T) # covariance matrix 
    print('covariance matrix condition number = %.5e' % np.linalg.cond(C_pbk)) 

    # plot the covariance matrix 
    fig = plt.figure(figsize=(10,8))
    sub = fig.add_subplot(111)
    cm = sub.pcolormesh(C_pbk, norm=LogNorm(vmin=1e5, vmax=1e18))
    cbar = fig.colorbar(cm, ax=sub) 
    sub.set_title(r'Quijote $P(k)$ and $B(k_1, k_2, k_3)$ Covariance', fontsize=25)
    ffig = os.path.join(UT.fig_dir(), 'quijote_pbk_Cov_%s_%s_%s.png' % 
            (typ, str(kmin).replace('.', ''), str(kmax).replace('.', '')))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


# B(k) fisher forecast --- 
def plotEllipse(Finv_sub, sub, theta_fid_ij=None, color='C0'): 
    ''' Given the inverse fisher sub-matrix, calculate ellipse parameters and
    add to subplot 
    '''
    theta_fid_i, theta_fid_j = theta_fid_ij
    # get ellipse parameters 
    a = np.sqrt(0.5*(Finv_sub[0,0] + Finv_sub[1,1]) + np.sqrt(0.25*(Finv_sub[0,0]-Finv_sub[1,1])**2 + Finv_sub[0,1]**2))
    b = np.sqrt(0.5*(Finv_sub[0,0] + Finv_sub[1,1]) - np.sqrt(0.25*(Finv_sub[0,0]-Finv_sub[1,1])**2 + Finv_sub[0,1]**2))
    theta = 0.5 * np.arctan2(2.0 * Finv_sub[0,1], (Finv_sub[0,0] - Finv_sub[1,1]))
    for ii, alpha in enumerate([2.48, 1.52]):
        e = Ellipse(xy=(theta_fid_i, theta_fid_j), 
                width=alpha * a, height=alpha * b, angle=theta * 360./(2.*np.pi))
        sub.add_artist(e)
        if ii == 0: alpha = 0.7
        if ii == 1: alpha = 1.
        e.set_alpha(alpha)
        e.set_facecolor(color) 
    return sub

# fisher derivatives
def quijote_dPk(theta, rsd=True, dmnu='fin'):
    ''' calculate d P(k)/d theta using the paired and fixed quijote simulations
    run on perturbed theta 

    :param theta: 
        string that specifies the parameter to take the 
        derivative by. 
    '''
    if theta == 'Mnu': 
        Pks = [] # read in the bispectrum a Mnu+, Mnu++, Mnu+++
        for tt in ['fiducial', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp']: 
            quij = quijoteBk(tt, rsd=rsd)
            Pks.append(np.average(quij['p0k1'], axis=0))
        Pk_fid, Pk_p, Pk_pp, Pk_ppp = Pks 

        h_p, h_pp, h_ppp = quijote_thetas['Mnu']
        # take the derivatives 
        if dmnu == 'p': 
            dPk = (Pk_p - Pk_fid)/h_p 
        elif dmnu == 'pp': 
            dPk = (Pk_pp - Pk_fid)/h_pp
        elif dmnu == 'ppp': 
            dPk = (Pk_ppp - Pk_fid)/h_ppp
        elif dmnu == 'fin': 
            dPk = (-21 * Pk_fid + 32 * Pk_p - 12 * Pk_pp + Pk_ppp)/(1.2) # finite difference coefficient
    elif theta == 'Amp': 
        # amplitude of P(k) is not a free parameter
        quij = quijoteBk('fiducial', rsd=rsd)
        dPk = np.average(quij['p0k1'], axis=0)
    else: 
        h = quijote_thetas[theta][1] - quijote_thetas[theta][0]
        
        quij = quijoteBk(theta+'_m', rsd=rsd)
        Pk_m = np.average(quij['p0k1'], axis=0) # Covariance matrix tt- 
        quij = quijoteBk(theta+'_p', rsd=rsd)
        Pk_p = np.average(quij['p0k1'], axis=0) # Covariance matrix tt+ 
        
        dPk = (Pk_p - Pk_m) / h # take the derivatives 
    return dPk


def quijote_dBk(theta, rsd=True, dmnu='fin'):
    ''' calculate d B(k)/d theta using the paired and fixed quijote simulations
    run on perturbed theta 

    :param theta: 
        string that specifies the parameter to take the 
        derivative by. 
    '''
    if theta == 'Mnu': 
        Bks = [] # read in the bispectrum at fiducial, Mnu+, Mnu++, Mnu+++
        for tt in ['fiducial', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp']: 
            quij = quijoteBk(tt, rsd=rsd)
            Bks.append(np.average(quij['b123'], axis=0))
        Bk_fid, Bk_p, Bk_pp, Bk_ppp = Bks 

        # take the derivatives 
        h_p, h_pp, h_ppp = quijote_thetas['Mnu']
        if dmnu == 'p': 
            dBk = (Bk_p - Bk_fid) / h_p 
        elif dmnu == 'pp': 
            dBk = (Bk_pp - Bk_fid) / h_pp
        elif dmnu == 'ppp': 
            dBk = (Bk_ppp - Bk_fid) / h_ppp
        else: 
            dBk = (-21. * Bk_fid + 32. * Bk_p - 12. * Bk_pp + Bk_ppp)/1.2 # finite difference coefficient
    elif theta == 'Amp': 
        # amplitude scaling is a free parameter
        quij = quijoteBk('fiducial', rsd=rsd)
        dBk = np.average(quij['b123'], axis=0)
    else: 
        h = quijote_thetas[theta][1] - quijote_thetas[theta][0]
        
        quij = quijoteBk(theta+'_m', rsd=rsd)
        Bk_m = np.average(quij['b123'], axis=0) # Covariance matrix tt- 
        quij = quijoteBk(theta+'_p', rsd=rsd)
        Bk_p = np.average(quij['b123'], axis=0) # Covariance matrix tt+ 
        
        dBk = (Bk_p - Bk_m) / h # take the derivatives 
    return dBk


def quijote_Bratio_thetas(kmax=0.5): 
    ''' compare the derivative of B along thetas 
    '''
    kf = 2.*np.pi/1000.
    quij = quijoteBk('fiducial', rsd=True)
    bk_fid = np.average(quij['b123'], axis=0) 
    i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3']

    klim = ((i_k * kf <= kmax) & (j_k * kf <= kmax) & (l_k * kf <= kmax)) 
    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 
    ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # order of triangles 

    fig = plt.figure(figsize=(20,8))
    sub = fig.add_subplot(111)
    for tt, lbl in zip(['Om_m', 'Ob_p', 'h_p', 'ns_m', 's8_p', 'Mnu_p'], theta_lbls): 
        quij = quijoteBk(tt, rsd=True) 
        sub.plot(range(np.sum(klim)), (np.average(quij['b123'], axis=0)/bk_fid)[klim][ijl], label=lbl)
    sub.plot([0., np.sum(klim)], [1., 1.], c='k', ls='--', zorder=0) 
    sub.legend(loc='upper right', ncol=2, frameon=True, fontsize=20) 
    sub.set_xlabel('triangle configurations', fontsize=25) 
    sub.set_xlim(0, np.sum(klim))
    sub.set_ylabel(r'$B(k_1, k_2, k_3)/B^{\rm fid}$', fontsize=25) 
    sub.set_ylim(0.9, 1.15) 
    ffig = os.path.join(UT.fig_dir(), 'quijote_Bratio_theta.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def quijote_B_relative_error(kmax=0.5): 
    kf = 2.*np.pi/1000.
    quij = quijoteBk('fiducial', rsd=True)
    bk_fid = np.average(quij['b123'], axis=0) 
    i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3']

    klim = ((i_k * kf <= kmax) & (j_k * kf <= kmax) & (l_k * kf <= kmax)) 
    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 
    ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # order of triangles 

    _, _, _, C_fid = quijoteCov(rsd=rsd)

    Cii = np.diag(C_fid)[klim][ijl]

    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    sub.plot(range(np.sum(klim)), np.sqrt(Cii)/bk_fid[klim][ijl]) 
    sub.set_xlabel('triangle configurations', fontsize=25) 
    sub.set_xlim(0, np.sum(klim))
    sub.set_ylabel(r'$\sqrt{C_{i,i}}/B^{\rm fid}_i$', fontsize=25) 
    sub.set_ylim(0.0, 2.5) 
    ffig = os.path.join(UT.fig_dir(), 'quijote_B_relative_error.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None



# fisher matrix 
def quijote_pkFisher(kmax=0.5, rsd=True, dmnu='fin', validate=False): 
    ''' calculate fisher matrix for parameters ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
    for P(k) analysis
    '''
    # calculate covariance matrix (with shotnoise; this is the correct one) 
    quij = quijoteBk('fiducial', rsd=rsd) # theta_fiducial 
    pks = quij['p0k1'] + 1e9/quij['Nhalos'][:,None] # uncorrect shotn oise 
    i_k = quij['k1']

    # impose k limit 
    _, _iuniq = np.unique(i_k, return_index=True)
    iuniq = np.zeros(len(i_k)).astype(bool) 
    iuniq[_iuniq] = True
    kf = 2.*np.pi/1000. # fundmaentla mode
    klim = (iuniq & (i_k*kf <= kmax)) 
    i_k = i_k[klim]

    C_fid = np.cov(pks[:,klim].T) 

    C_inv = np.linalg.inv(C_fid) # invert the covariance 
    
    dpk_dt = [] 
    for par in thetas: # calculate the derivative of Bk along all the thetas 
        dpk_dti = quijote_dPk(par, rsd=rsd, dmnu=dmnu)
        dpk_dt.append(dpk_dti[klim])
    Fij = Forecast.Fij(dpk_dt, C_inv) 
    
    if validate: 
        f_ij = os.path.join(UT.dat_dir(), 'bispectrum', 'quijote_pkFij.kmax%.2f%s.hdf5' % (kmax, ['.real', ''][rsd]))
        f = h5py.File(f_ij, 'w') 
        f.create_dataset('Fij', data=Fij) 
        f.create_dataset('C_fid', data=C_fid)
        f.create_dataset('C_inv', data=C_inv)
        f.close() 

        fig = plt.figure(figsize=(6,5))
        sub = fig.add_subplot(111)
        cm = sub.pcolormesh(Fij, norm=SymLogNorm(vmin=-2e5, vmax=5e5, linthresh=1e2, linscale=1.))
        sub.set_xticks(np.arange(Fij.shape[0]) + 0.5, minor=False)
        sub.set_xticklabels(theta_lbls, minor=False)
        sub.set_yticks(np.arange(Fij.shape[1]) + 0.5, minor=False)
        sub.set_yticklabels(theta_lbls, minor=False)
        sub.set_title(r'Fisher Matrix $F_{i,j}$', fontsize=25)
        fig.colorbar(cm)
        ffig = os.path.join(UT.doc_dir(), 'figs', os.path.basename(f_ij).replace('.hdf5', '.png'))
        fig.savefig(ffig, bbox_inches='tight') 
    return Fij 


def quijote_bkFisher(kmax=0.5, rsd=True, dmnu='fin', validate=False): 
    ''' calculate fisher matrix for parameters ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
    '''
    # read in full covariance matrix (with shotnoise; this is the correct one) 
    i_k, j_k, l_k, C_fid = quijoteCov(rsd=rsd)

    # impose k limit 
    kf = 2.*np.pi/1000. # fundmaentla mode
    klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 
    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 
    C_fid = C_fid[:,klim][klim,:]

    C_inv = np.linalg.inv(C_fid) # invert the covariance 
    
    dbk_dt = [] 
    for par in thetas: # derivatives of Bk along all the thetas 
        dbk_dti = quijote_dBk(par, rsd=rsd, dmnu=dmnu)
        dbk_dt.append(dbk_dti[klim])

    Fij = Forecast.Fij(dbk_dt, C_inv) 
    
    if validate: 
        f_ij = os.path.join(UT.dat_dir(), 'bispectrum', 'quijote_bkFij.kmax%.2f%s.hdf5' % (kmax, ['.real', ''][rsd]))
        f = h5py.File(f_ij, 'w') 
        f.create_dataset('Fij', data=Fij) 
        f.create_dataset('C_fid', data=C_fid)
        f.create_dataset('C_inv', data=C_inv)
        f.close() 

        fig = plt.figure(figsize=(6,5))
        sub = fig.add_subplot(111)
        cm = sub.pcolormesh(Fij, norm=SymLogNorm(vmin=-2e5, vmax=5e5, linthresh=1e2, linscale=1.))
        sub.set_xticks(np.arange(Fij.shape[0]) + 0.5, minor=False)
        sub.set_xticklabels(theta_lbls, minor=False)
        sub.set_yticks(np.arange(Fij.shape[1]) + 0.5, minor=False)
        sub.set_yticklabels(theta_lbls, minor=False)
        sub.set_title(r'Fisher Matrix $F_{i,j}$', fontsize=25)
        fig.colorbar(cm)
        ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_bkFij.kmax%.2f%s.png' % (kmax, ['.real', ''][rsd]))
        fig.savefig(ffig, bbox_inches='tight') 
    return Fij 


def quijote_bkFisher_triangle(typ, kmax=0.5, rsd=True, dmnu='fin'): 
    ''' calculate fisher matrix for parameters ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
    using only triangle configurations up to kmax

    :param typ: 
        string that specifies the triangle shape. typ in ['equ', 'squ']. 

    :param kmax: (default: 0.5) 
        kmax 
    '''
    # read in full covariance matrix (with shotnoise; this is the correct one) 
    i_k, j_k, l_k, C_fid = quijoteCov(rsd=rsd)

    # impose k limit 
    if typ == 'equ': # only equilateral 
        tri = (i_k == j_k) & (j_k == l_k) 
    elif typ == 'squ': # i_k >= j_k >= l_k (= 3*kf~0.01) 
        tri = (i_k == j_k) & (l_k == 3)
    kf = 2.*np.pi/1000. # fundmaentla mode
    klim = (tri & (i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 
    print('%i %s triangle configurations' % (np.sum(klim), typ))
    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 
    C_fid = C_fid[:,klim][klim,:]

    C_inv = np.linalg.inv(C_fid) # invert the covariance 
    
    dbk_dt = [] 
    for par in thetas: # calculate the derivative of Bk along all the thetas 
        dbk_dti = quijote_dBk(par, rsd=rsd, dmnu=dmnu)
        dbk_dt.append(dbk_dti[klim])
    
    return Forecast.Fij(dbk_dt, C_inv) 


# forecasts 
def quijote_Forecast(obs, kmax=0.5, rsd=True, dmnu='fin'):
    ''' fisher forecast for quijote observables
    
    :param krange: (default: 0.5) 
        tuple specifying the kranges of k1, k2, k3 in the bispectrum
    '''
    if obs == 'bk': 
        Fij = quijote_bkFisher(kmax=kmax, rsd=rsd, dmnu=dmnu, validate=False) # fisher matrix (Fij)
    elif obs == 'bk_equ': 
        Fij = quijote_bkFisher_triangle('equ', kmax=kmax, rsd=rsd, dmnu=dmnu)
    elif obs == 'bk_squ': 
        Fij = quijote_bkFisher_triangle('squ', kmax=kmax, rsd=rsd, dmnu=dmnu)
    elif obs == 'pk': 
        Fij = quijote_pkFisher(kmax=kmax, rsd=rsd, dmnu=dmnu, validate=False) # fisher matrix (Fij)
    else: 
        raise ValueError
    
    Finv = np.linalg.inv(Fij) # invert fisher matrix 
    print('sigma_s8 = %f' % np.sqrt(Finv[-2,-2]))
    print('sigma_Mnu = %f' % np.sqrt(Finv[-1,-1]))
    
    fig = plt.figure(figsize=(17, 15))
    for i in xrange(ntheta-1): 
        for j in xrange(i+1, ntheta): 
            # sub inverse fisher matrix 
            Finv_sub = np.array([[Finv[i,i], Finv[i,j]], [Finv[j,i], Finv[j,j]]]) 
            # plot the ellipse
            sub = fig.add_subplot(ntheta-1, ntheta-1, (ntheta-1) * (j-1) + i + 1) 
            plotEllipse(Finv_sub, sub, theta_fid_ij=[theta_fid[thetas[i]], theta_fid[thetas[j]]], color='C0')
            sub.set_xlim(theta_lims[i])
            sub.set_ylim(theta_lims[j])
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
    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_%sFisher_dmnu_%s_kmax%.2f%s.png' % (obs, dmnu, kmax, ['_real', ''][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def quijote_Forecast_kmax(obs, rsd=True, dmnu='fin'):
    ''' fisher forecast for quijote for different kmax values 
    '''
    kmaxs = [0.2, 0.3, 0.5]
    colrs = ['C2', 'C1', 'C0']
    alphas = [0.7, 0.8, 0.9] 
    
    # read in fisher matrix (Fij)
    Finvs = [] 
    print('%s %s-space' % (obs, ['real', 'redshift'][rsd]))
    for kmax in kmaxs: 
        if obs == 'bk': 
            Fij = quijote_bkFisher(kmax=kmax, rsd=rsd, dmnu=dmnu, validate=False)
        elif obs == 'pk': 
            Fij = quijote_pkFisher(kmax=kmax, rsd=rsd, dmnu=dmnu, validate=False)

        Finv = np.linalg.inv(Fij) # invert fisher matrix 
        Finvs.append(Finv)
        print('kmax=%f, sigma_Mnu=%f' % (kmax, np.sqrt(Finv[-1,-1])))

    fig = plt.figure(figsize=(17, 15))
    for i in xrange(ntheta-1): 
        for j in xrange(i+1, ntheta): 
            for i_k, Finv in enumerate(Finvs): 
                # sub inverse fisher matrix 
                Finv_sub = np.array([[Finv[i,i], Finv[i,j]], [Finv[j,i], Finv[j,j]]]) 

                theta_fid_i, theta_fid_j = theta_fid[thetas[i]], theta_fid[thetas[j]]
                
                # get ellipse parameters 
                a = np.sqrt(0.5*(Finv_sub[0,0] + Finv_sub[1,1]) + np.sqrt(0.25*(Finv_sub[0,0]-Finv_sub[1,1])**2 + Finv_sub[0,1]**2))
                b = np.sqrt(0.5*(Finv_sub[0,0] + Finv_sub[1,1]) - np.sqrt(0.25*(Finv_sub[0,0]-Finv_sub[1,1])**2 + Finv_sub[0,1]**2))
                theta = 0.5 * np.arctan2(2.0 * Finv_sub[0,1], (Finv_sub[0,0] - Finv_sub[1,1]))
        
                # plot the ellipse
                sub = fig.add_subplot(ntheta-1, ntheta-1, (ntheta-1) * (j-1) + i + 1) 

                e = Ellipse(xy=(theta_fid_i, theta_fid_j), width=2.48 * a, height=2.48 * b, angle=theta * 360./(2.*np.pi))
                sub.add_artist(e)
                e.set_alpha(alphas[i_k])
                e.set_facecolor(colrs[i_k])
                
                e = Ellipse(xy=(theta_fid_i, theta_fid_j), width=1.52 * a, height=1.52 * b, angle=theta * 360./(2.*np.pi),
                        fill=False, edgecolor='k', linestyle='--')
                sub.add_artist(e)

            sub.set_xlim(theta_lims[i])
            sub.set_ylim(theta_lims[j])
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

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.text(0.82, 0.77, r'$B^{\rm halo}(k_1, k_2, k_3)$', ha='right', va='bottom', 
                transform=bkgd.transAxes, fontsize=25)
    for colr, alpha, kmax in zip(colrs, alphas, kmaxs): 
        bkgd.fill_between([],[],[], color=colr, alpha=alpha, label=r'$k_1, k_2, k_3 < k_{\rm max} = %.1f$' % kmax) 
    bkgd.legend(loc='upper right', handletextpad=0.3, bbox_to_anchor=(0.925, 0.775), fontsize=25)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.05, hspace=0.05) 
    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_%sFisher_dmnu_%s_kmax%s.pdf' % (obs, dmnu, ['_real', ''][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_%sFisher_dmnu_%s_kmax%s.png' % (obs, dmnu, ['_real', ''][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def quijote_Forecast_dmnu(obs, rsd=True):
    ''' fisher forecast for quijote for different methods of calculating the 
    derivative along Mnu 
    '''
    dmnus = ['fin', 'p', 'pp', 'ppp'][::-1]
    colrs = ['C3', 'C2', 'C1', 'C0']
    alphas = [0.6, 0.7, 0.8, 0.9] 
    
    # read in fisher matrix (Fij)
    Finvs = [] 
    print('%s-space' % ['real', 'redshift'][rsd])
    for dmnu in dmnus:
        if obs == 'bk': 
            Fij = quijote_bkFisher(kmax=0.5, rsd=rsd, dmnu=dmnu, validate=False)
        elif obs == 'pk': 
            Fij = quijote_pkFisher(kmax=0.5, rsd=rsd, dmnu=dmnu, validate=False)

        Finv = np.linalg.inv(Fij) # invert fisher matrix 
        Finvs.append(Finv)
        print('dB/dMnu_%s, sigma_Mnu=%f' % (dmnu, np.sqrt(Finv[-1,-1])))

    fig = plt.figure(figsize=(17, 15))
    for i in xrange(ntheta-1): 
        for j in xrange(i+1, ntheta): 
            for i_k, Finv in enumerate(Finvs): 
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

                e = Ellipse(xy=(theta_fid_i, theta_fid_j), width=2.48 * a, height=2.48 * b, angle=theta * 360./(2.*np.pi))
                sub.add_artist(e)
                e.set_alpha(alphas[i_k])
                e.set_facecolor(colrs[i_k])
                
                e = Ellipse(xy=(theta_fid_i, theta_fid_j), width=1.52 * a, height=1.52 * b, angle=theta * 360./(2.*np.pi),
                        fill=False, edgecolor='k', linestyle='--')
                sub.add_artist(e)

            sub.set_xlim(theta_lims[i])
            sub.set_ylim(theta_lims[j])
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

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.text(0.82, 0.77, r'$B^{\rm halo}(k_1, k_2, k_3)$', ha='right', va='bottom', 
                transform=bkgd.transAxes, fontsize=25)
    for colr, alpha, dmnu in zip(colrs, alphas, dmnus): 
        bkgd.fill_between([],[],[], color=colr, alpha=alpha, label=r'${\rm d}B(k)/{\rm d}M_\nu$ %s' % dmnu) 
    bkgd.legend(loc='upper right', handletextpad=0.3, bbox_to_anchor=(0.925, 0.775), fontsize=25)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.05, hspace=0.05) 
    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_%sFisher_dmnu%s.pdf' % (obs, ['_real', ''][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_%sFisher_dmnu%s.png' % (obs, ['_real', ''][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    return None


# P(k), B(k) comparison 
def quijote_pbkForecast(kmax=0.5, rsd=True, dmnu='fin'):
    ''' fisher forecast for quijote from P(k) and B(k) overlayed on them 
    
    :param krange: (default: [0.01, 0.5]) 
        tuple specifying the kranges of k1, k2, k3 in the bispectrum
    '''
    # fisher matrices for powerspectrum and bispectrum
    pkFij = quijote_pkFisher(kmax=kmax, rsd=rsd, dmnu=dmnu, validate=False)
    bkFij = quijote_bkFisher(kmax=kmax, rsd=rsd, dmnu=dmnu, validate=False)

    pkFinv = np.linalg.inv(pkFij) # invert fisher matrix 
    bkFinv = np.linalg.inv(bkFij) # invert fisher matrix 

    print('P(k) marginalized constraint on Mnu = %f' % np.sqrt(pkFinv[-1,-1]))
    print('B(k1,k2,k3) marginalized constraint on Mnu = %f' % np.sqrt(bkFinv[-1,-1]))
    
    fig = plt.figure(figsize=(17, 15))
    for i in xrange(ntheta-1): 
        for j in xrange(i+1, ntheta): 
            theta_fid_i, theta_fid_j = theta_fid[thetas[i]], theta_fid[thetas[j]] # fiducial parameter 

            sub = fig.add_subplot(ntheta-1, ntheta-1, (ntheta-1) * (j-1) + i + 1) 
            for _i, Finv in enumerate([pkFinv, bkFinv]):
                # sub inverse fisher matrix 
                Finv_sub = np.array([[Finv[i,i], Finv[i,j]], [Finv[j,i], Finv[j,j]]]) 
                plotEllipse(Finv_sub, sub, theta_fid_ij=[theta_fid_i, theta_fid_j], color='C%i' % _i)
                
            sub.set_xlim(theta_lims[i])
            sub.set_ylim(theta_lims[j])
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

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.fill_between([],[],[], color='C0', label=r'$P^{\rm halo}_0(k)$') 
    bkgd.fill_between([],[],[], color='C1', label=r'$B^{\rm halo}(k_1, k_2, k_3)$') 
    bkgd.legend(loc='upper right', bbox_to_anchor=(0.875, 0.775), fontsize=25)
    bkgd.text(0.75, 0.61, r'$k_{\rm max} = %.1f$' % kmax, ha='right', va='bottom', 
            transform=bkgd.transAxes, fontsize=25)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.05, hspace=0.05) 
    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_pbkFisher_dmnu_%s_kmax%s%s.pdf' % 
            (dmnu, str(kmax).replace('.', ''), ['_real', ''][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_pbkFisher_dmnu_%s_kmax%s%s.png' % 
            (dmnu, str(kmax).replace('.', ''), ['_real', ''][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def quijote_pbkForecast_Mnu_s8(kmax=0.5):
    ''' fisher forecast for Mnu and sigma 8 using quijote from P(k) and B(k) 
    overlayed on them. Then overlay Hades parameter points ontop to see if
    things make somewhat sense.
    
    :param krange: (default: [0.01, 0.5]) 
        tuple specifying the kranges of k1, k2, k3 in the bispectrum
    '''
    # fisher matrices for powerspectrum and bispectrum
    pkFij = quijote_pkFisher(kmax=kmax, rsd=True, dmnu='fin', validate=False)
    bkFij = quijote_bkFisher(kmax=kmax, rsd=True, dmnu='fin', validate=False)

    pkFinv = np.linalg.inv(pkFij) # invert fisher matrix 
    bkFinv = np.linalg.inv(bkFij) # invert fisher matrix 

    print('P(k) marginalized constraint on Mnu = %f' % np.sqrt(pkFinv[-1,-1]))
    print('B(k1,k2,k3) marginalized constraint on Mnu = %f' % np.sqrt(bkFinv[-1,-1]))
    
    fig = plt.figure(figsize=(7,7))
    sub = fig.add_subplot(111) 
    i, j = 4, 5
    for _i, Finv in enumerate([pkFinv, bkFinv]):
        # sub inverse fisher matrix 
        Finv_sub = np.array([[Finv[i,i], Finv[i,j]], [Finv[j,i], Finv[j,j]]]) 
        theta_fid_i, theta_fid_j = theta_fid[thetas[j]], theta_fid[thetas[i]]
                
        plotEllipse(Finv_sub, sub, theta_fid_ij=[theta_fid_i, theta_fid_j], color='C%i' % _i)
        
    sub.set_xlabel(theta_lbls[i], labelpad=10, fontsize=30) 
    sub.set_xlim(theta_lims[i])
    sub.set_ylabel(theta_lbls[j], fontsize=30) 
    sub.set_ylim(theta_lims[j])
    sub.fill_between([],[],[], color='C0', label=r'$P^{\rm halo}_0(k)$') 
    sub.fill_between([],[],[], color='C1', label=r'$B^{\rm halo}(k_1, k_2, k_3)$') 
    # overlay hades
    hades_sig8 = [0.833, 0.822, 0.815, 0.806]
    hades_Mnus = [0., 0.06, 0.10, 0.15]
    sub.scatter(hades_sig8, hades_Mnus, c='k', marker='+', zorder=10, label='HADES sims')
    
    _hades_sig8 = [0.822, 0.818, 0.807, 0.798]
    sub.scatter(_hades_sig8, np.zeros(len(_hades_sig8)), c='k', marker='x', zorder=10)

    sub.legend(loc='upper right', handletextpad=0.2, markerscale=5, fontsize=20)
    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_pbkFisher_Mnu_sig8_kmax%s.png' % str(kmax).replace('.', ''))
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def hades_dchi2(krange=[0.01, 0.5]):
    ''' calculate delta chi-squared for the hades simulation using the quijote 
    simulation covariance matrix
    
    :param krange: (default: [0.01, 0.5]) 
        tuple specifying the kranges of k1, k2, k3 in the bispectrum
    '''
    # read in B(k) of fiducial quijote simulation for covariance matrix
    quij = quijoteBk('fiducial') # theta_fiducial 
    bks = quij['b123'] + quij['b_sn']                   # shotnoise uncorrected B(k) 
    i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3']

    # impose k limit on bispectrum
    kmin, kmax = krange 
    kf = 2.*np.pi/1000. # fundmaentla mode
    bklim = ((i_k*kf <= kmax) & (i_k*kf >= kmin) &
            (j_k*kf <= kmax) & (j_k*kf >= kmin) & 
            (l_k*kf <= kmax) & (l_k*kf >= kmin)) 
    ijl = UT.ijl_order(i_k[bklim], j_k[bklim], l_k[bklim], typ='GM') # order of triangles 
    bks = bks[:,bklim][:,ijl] 

    C_bk = np.cov(bks.T) # covariance matrix 
    C_inv = np.linalg.inv(C_bk) # invert the covariance 
    
    # fiducial hades B(k) 
    hades_fid = hadesBk(0.0, nzbin=4, rsd=True) # fiducial bispectrum
    Bk_fid = np.average(hades_fid['b123'], axis=0)

    mnus = [0.0, 0.06, 0.1, 0.15]
    sig8s = [0.822, 0.818, 0.807, 0.798]
    for mnu in mnus: # bispectrum for different Mnu
        hades_i = hadesBk(mnu, nzbin=4, rsd=True)
        _bk = np.average(hades_i['b123'], axis=0)
        dbk = (_bk - Bk_fid)[bklim][ijl]
        chi2 = np.sum(np.dot(dbk.T, np.dot(C_inv, dbk)))
        print('Mnu=%.2f, delta chi-squared %.2f' % (mnu, chi2))

    for sig8 in sig8s: # bispectrum for different sigma8 
        hades_i = hadesBk_s8(sig8, nzbin=4, rsd=True)
        _bk = np.average(hades_i['b123'], axis=0)
        dbk = (_bk - Bk_fid)[bklim][ijl]
        chi2 = np.sum(np.dot(dbk.T, np.dot(C_inv, dbk)))
        print('sig8=%.3f, delta chi-squared %.2f' % (sig8, chi2))
    return None


##################################################################
# forecasts with free scaling factor 
##################################################################
def quijote_pkFisher_freeScale(kmax=0.5, rsd=True, dmnu='fin'): 
    ''' calculate fisher matrix for P(k) analysis where we add a free 
    scaling parameter A into the analysis.
    P(k) = A * p(k) 
    '''
    # calculate covariance matrix (with shotnoise; this is the correct one) 
    quij = quijoteBk('fiducial', rsd=rsd) # theta_fiducial 
    pks = quij['p0k1'] + 1e9/quij['Nhalos'][:,None] # uncorrect shotn oise 
    i_k = quij['k1']

    # impose k limit 
    _, _iuniq = np.unique(i_k, return_index=True)
    iuniq = np.zeros(len(i_k)).astype(bool) 
    iuniq[_iuniq] = True
    kf = 2.*np.pi/1000. # fundmaentla mode
    klim = (iuniq & (i_k*kf <= kmax)) 
    i_k = i_k[klim]

    C_fid = np.cov(pks[:,klim].T) 

    C_inv = np.linalg.inv(C_fid) # invert the covariance 
    
    dpk_dt = [] 
    for par in thetas: # calculate the derivative of Bk along all the thetas 
        dpk_dti = quijote_dPk(par, rsd=rsd, dmnu=dmnu)
        dpk_dt.append(dpk_dti[klim])
    # d P(k) / d A = p(k) 
    dpk_dti = quijote_dPk('Amp', rsd=rsd, dmnu=dmnu)
    dpk_dt.append(dpk_dti[klim]) 

    Fij = Forecast.Fij(dpk_dt, C_inv) 
    return Fij 


def quijote_bkFisher_freeScale(kmax=0.5, rsd=True, dmnu='fin', validate=False): 
    ''' calculate fisher matrix for parameters ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
    '''
    # read in full covariance matrix (with shotnoise; this is the correct one) 
    i_k, j_k, l_k, C_fid = quijoteCov(rsd=rsd)

    # impose k limit 
    kf = 2.*np.pi/1000. # fundmaentla mode
    klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 
    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 
    C_fid = C_fid[:,klim][klim,:]

    C_inv = np.linalg.inv(C_fid) # invert the covariance 
    
    dbk_dt = [] 
    for par in thetas: # derivatives of Bk along all the thetas 
        dbk_dti = quijote_dBk(par, rsd=rsd, dmnu=dmnu)
        dbk_dt.append(dbk_dti[klim])
    # amplitude scaling factor Amp 
    dbk_dti = quijote_dBk('Amp', rsd=rsd, dmnu=dmnu)
    dbk_dt.append(dbk_dti[klim])

    Fij = Forecast.Fij(dbk_dt, C_inv) 
    return Fij 


def quijote_bkFisher_triangle_freeScale(typ, kmax=0.5, rsd=True, dmnu='fin'): 
    ''' calculate fisher matrix for parameters ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
    using only triangle configurations up to kmax

    :param typ: 
        string that specifies the triangle shape. typ in ['equ', 'squ']. 

    :param kmax: (default: 0.5) 
        kmax 
    '''
    # read in full covariance matrix (with shotnoise; this is the correct one) 
    i_k, j_k, l_k, C_fid = quijoteCov(rsd=rsd)

    # impose k limit 
    if typ == 'equ': # only equilateral 
        tri = (i_k == j_k) & (j_k == l_k) 
    elif typ == 'squ': # i_k >= j_k >= l_k (= 3*kf~0.01) 
        tri = (i_k == j_k) & (l_k == 3)
    kf = 2.*np.pi/1000. # fundmaentla mode
    klim = (tri & (i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 
    print('%i %s triangle configurations' % (np.sum(klim), typ))
    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 
    C_fid = C_fid[:,klim][klim,:]

    C_inv = np.linalg.inv(C_fid) # invert the covariance 
    
    dbk_dt = [] 
    for par in thetas: # calculate the derivative of Bk along all the thetas 
        dbk_dti = quijote_dBk(par, rsd=rsd, dmnu=dmnu)
        dbk_dt.append(dbk_dti[klim])
    # amplitude scaling factor Amp 
    dbk_dti = quijote_dBk('Amp', rsd=rsd, dmnu=dmnu)
    dbk_dt.append(dbk_dti[klim])
    
    Fij = Forecast.Fij(dbk_dt, C_inv) 
    return Fij 


def quijote_Forecast_freeScale(obs, kmax=0.5, rsd=True, dmnu='fin'):
    ''' fisher forecast for quijote observables where a scaling factor of the
    amplitude is added as a free parameter. This is to conservatively estimate
    what would happen if we wipe out any multiplicative amplitude information.
    
    :param kmax: (default: 0.5) 
        kmax of the analysis 
    '''
    if obs == 'bk': 
        _Fij = quijote_bkFisher(kmax=kmax, rsd=rsd, dmnu=dmnu) # fisher matrix (Fij)
        Fij = quijote_bkFisher_freeScale(kmax=kmax, rsd=rsd, dmnu=dmnu) # fisher matrix (Fij)
    elif obs == 'pk': 
        _Fij = quijote_pkFisher(kmax=kmax, rsd=rsd, dmnu=dmnu) # fisher matrix (Fij)
        Fij = quijote_pkFisher_freeScale(kmax=kmax, rsd=rsd, dmnu=dmnu) # fisher matrix (Fij)
    elif obs == 'bk_equ': 
        _Fij = quijote_bkFisher_triangle('equ', kmax=kmax, rsd=rsd, dmnu=dmnu) # fisher matrix (Fij)
        Fij = quijote_bkFisher_triangle_freeScale('equ', kmax=kmax, rsd=rsd, dmnu=dmnu) # fisher matrix (Fij)
    elif obs == 'bk_squ': 
        _Fij = quijote_bkFisher_triangle('squ', kmax=kmax, rsd=rsd, dmnu=dmnu) # fisher matrix (Fij)
        Fij = quijote_bkFisher_triangle_freeScale('squ', kmax=kmax, rsd=rsd, dmnu=dmnu) # fisher matrix (Fij)
    else: 
        raise ValueError
    
    _Finv = np.linalg.inv(_Fij) # invert fisher matrix 
    Finv = np.linalg.inv(Fij) # invert fisher matrix 
    print('sigma_Mnu = %f' % np.sqrt(_Finv[-1,-1]))
    print('sigma_Mnu = %f (Amp free)' % np.sqrt(Finv[-2,-2]))

    _thetas = thetas + ['Amp'] 
    _theta_fid = {'Mnu': 0., 'Ob': 0.049, 'Om': 0.3175, 'h': 0.6711,  'ns': 0.9624,  's8': 0.834, 'Amp':1.} # fiducial theta 
    if 'bk_' in obs: 
        _theta_lims = [(0.1, 0.5), (0.0, 0.2), (0.0, 1.3), (0.4, 1.6), (0.0, 2.), (-1., 1.), (0.5, 2.)]
    else: 
        _theta_lims = [(0.25, 0.385), (0.02, 0.08), (0.3, 1.1), (0.6, 1.3), (0.8, 0.88), (-0.4, 0.4), (0.8, 1.2)]
    _theta_lbls = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$', '$b_1$']
    
    fig = plt.figure(figsize=(17, 15))
    for i in xrange(ntheta): 
        for j in xrange(i+1, ntheta+1): 
            # sub inverse fisher matrix 
            Finv_sub = np.array([[Finv[i,i], Finv[i,j]], [Finv[j,i], Finv[j,j]]]) 
            # plot the ellipse
            sub = fig.add_subplot(ntheta, ntheta, ntheta * (j-1) + i + 1) 
            plotEllipse(Finv_sub, sub, theta_fid_ij=[_theta_fid[_thetas[i]], _theta_fid[_thetas[j]]], color='C0')
            if j < ntheta:
                _Finv_sub = np.array([[_Finv[i,i], _Finv[i,j]], [_Finv[j,i], _Finv[j,j]]]) 
                plotEllipse(_Finv_sub, sub, theta_fid_ij=[_theta_fid[_thetas[i]], _theta_fid[_thetas[j]]], color='C1')
            sub.set_xlim(_theta_lims[i])
            sub.set_ylim(_theta_lims[j])
            if i == 0:   
                sub.set_ylabel(_theta_lbls[j], fontsize=30) 
            else: 
                sub.set_yticks([])
                sub.set_yticklabels([])
            
            if j == ntheta: 
                sub.set_xlabel(_theta_lbls[i], labelpad=10, fontsize=30) 
            else: 
                sub.set_xticks([])
                sub.set_xticklabels([]) 
    fig.subplots_adjust(wspace=0.05, hspace=0.05) 
    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_%sFisher_freeScale_dmnu_%s_kmax%.2f%s.png' % 
            (obs, dmnu, kmax, ['_real', ''][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    return None


##################################################################
# shot noise uncorrected forecasts
##################################################################
def quijote_dPk_SNuncorr(theta, rsd=True, dmnu='fin'):
    ''' calculate shot noise uncorrected dP(k)/d theta using the paired 
    and fixed quijote simulations run on perturbed theta 

    :param theta: 
        string that specifies the parameter to take the 
        derivative by. 
    '''
    if theta == 'Mnu': 
        Pks = [] # read in the bispectrum a Mnu+, Mnu++, Mnu+++
        for tt in ['fiducial', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp']: 
            quij = quijoteBk(tt, rsd=rsd)
            Pks.append(np.average(quij['p0k1'] + 1e9/quij['Nhalos'][:,None], axis=0)) # SN uncorrected
        Pk_fid, Pk_p, Pk_pp, Pk_ppp = Pks 

        h_p, h_pp, h_ppp = quijote_thetas['Mnu']
        # take the derivatives 
        if dmnu == 'p': 
            dPk = (Pk_p - Pk_fid)/h_p 
        elif dmnu == 'pp': 
            dPk = (Pk_pp - Pk_fid)/h_pp
        elif dmnu == 'ppp': 
            dPk = (Pk_ppp - Pk_fid)/h_ppp
        elif dmnu == 'fin': 
            dPk = (-21 * Pk_fid + 32 * Pk_p - 12 * Pk_pp + Pk_ppp)/(1.2) # finite difference coefficient
    else: 
        h = quijote_thetas[theta][1] - quijote_thetas[theta][0] # step size
        
        quij = quijoteBk(theta+'_m', rsd=rsd)
        Pk_m = np.average(quij['p0k1'] + 1e9/quij['Nhalos'][:,None], axis=0) # Covariance matrix tt- 
        quij = quijoteBk(theta+'_p', rsd=rsd)
        Pk_p = np.average(quij['p0k1'] + 1e9/quij['Nhalos'][:,None], axis=0) # Covariance matrix tt+ 
        
        # take the derivatives 
        dPk = (Pk_p - Pk_m) / h 
    return dPk


def quijote_dBk_SNuncorr(theta, rsd=True, dmnu='fin'):
    ''' calculate shot noise uncorrected dB(k)/dtheta using the paired and fixed 
    quijote simulations run on perturbed theta 

    :param theta: 
        string that specifies the parameter to take the 
        derivative by. 
    '''
    if theta == 'Mnu': 
        Bks = [] # read in the bispectrum a Mnu+, Mnu++, Mnu+++
        for tt in ['fiducial', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp']: 
            quij = quijoteBk(tt, rsd=rsd)
            Bks.append(np.average(quij['b123'] + quij['b_sn'], axis=0))
        Bk_fid, Bk_p, Bk_pp, Bk_ppp = Bks 

        # take the derivatives 
        h_p, h_pp, h_ppp = quijote_thetas['Mnu']
        if dmnu == 'p': 
            dBk = (Bk_p - Bk_fid) / h_p 
        elif dmnu == 'pp': 
            dBk = (Bk_pp - Bk_fid) / h_pp
        elif dmnu == 'ppp': 
            dBk = (Bk_ppp - Bk_fid) / h_ppp
        else: 
            dBk = (-21. * Bk_fid + 32. * Bk_p - 12. * Bk_pp + Bk_ppp)/1.2 # finite difference coefficient
    else: 
        h = quijote_thetas[theta][1] - quijote_thetas[theta][0]

        quij = quijoteBk(theta+'_m', rsd=rsd)
        Bk_m = np.average(quij['b123'] + quij['b_sn'], axis=0) # Covariance matrix tt- 
        quij = quijoteBk(theta+'_p', rsd=rsd)
        Bk_p = np.average(quij['b123'] + quij['b_sn'], axis=0) # Covariance matrix tt+ 
        
        # take the derivatives 
        dBk = (Bk_p - Bk_m) / h 
    return dBk


def quijote_pkFisher_SNuncorr(kmax=0.5, rsd=True, dmnu='fin'): 
    ''' calculate fisher matrix for parameters ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
    for P(k) analysis without shot noise correction!
    '''
    # calculate covariance matrix (with shotnoise; this is the correct one) 
    quij = quijoteBk('fiducial', rsd=rsd) # theta_fiducial 
    pks = quij['p0k1'] + 1e9/quij['Nhalos'][:,None] # uncorrect shotn oise 
    i_k = quij['k1']

    # impose k limit 
    _, _iuniq = np.unique(i_k, return_index=True)
    iuniq = np.zeros(len(i_k)).astype(bool) 
    iuniq[_iuniq] = True
    kf = 2.*np.pi/1000. # fundmaentla mode
    klim = (iuniq & (i_k*kf <= kmax)) 
    i_k = i_k[klim]

    C_fid = np.cov(pks[:,klim].T) 

    C_inv = np.linalg.inv(C_fid) # invert the covariance 
    
    dpk_dt = [] 
    for par in thetas: 
        # calculate the derivative of Pk (uncorrected for SN) along all the thetas 
        dpk_dti = quijote_dPk_SNuncorr(par, rsd=rsd, dmnu=dmnu)
        dpk_dt.append(dpk_dti[klim])
    
    Fij = Forecast.Fij(dpk_dt, C_inv) 
    return Fij 


def quijote_bkFisher_SNuncorr(kmax=0.5, rsd=True, dmnu='fin'): 
    ''' calculate fisher matrix for parameters ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
    '''
    # read in full covariance matrix (with shotnoise; this is the correct one) 
    fcov = h5py.File(os.path.join(UT.dat_dir(), 'bispectrum', 'quijote_Cov_full.hdf5'), 'r') 
    C_fid = fcov['C_bk'].value
    i_k, j_k, l_k = fcov['k1'].value, fcov['k2'].value, fcov['k3'].value 

    # impose k limit 
    kf = 2.*np.pi/1000. # fundmaentla mode
    klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) &  (l_k*kf <= kmax)) 
    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 
    C_fid = C_fid[:,klim][klim,:]

    C_inv = np.linalg.inv(C_fid) # invert the covariance 
    
    dbk_dt = [] 
    for par in thetas: # calculate the derivative of Bk along all the thetas 
        dbk_dti = quijote_dBk_SNuncorr(par, rsd=rsd, dmnu=dmnu)
        dbk_dt.append(dbk_dti[klim])
    
    Fij = Forecast.Fij(dbk_dt, C_inv) 
    return Fij 


def quijote_Forecast_SNuncorr(obs, kmax=0.5, rsd=True, dmnu='fin'):
    ''' fisher forecast for quijote 
    
    :param krange: (default: [0.01, 0.5]) 
        tuple specifying the kranges of k1, k2, k3 in the bispectrum
    '''
    # fisher matrix (Fij)
    if obs == 'pk': 
        Fij = quijote_pkFisher_SNuncorr(kmax=kmax, rsd=rsd, dmnu=dmnu) 
    elif obs == 'bk': 
        Fij = quijote_bkFisher_SNuncorr(kmax=kmax, rsd=rsd, dmnu=dmnu) 

    Finv = np.linalg.inv(Fij) # invert fisher matrix 
    print('sigma_Mnu = %f' % np.sqrt(Finv[-1,-1])) 

    fig = plt.figure(figsize=(17, 15))
    for i in xrange(ntheta-1): 
        for j in xrange(i+1, ntheta): 
            # sub inverse fisher matrix 
            Finv_sub = np.array([[Finv[i,i], Finv[i,j]], [Finv[j,i], Finv[j,j]]]) 

            theta_fid_i, theta_fid_j = theta_fid[thetas[i]], theta_fid[thetas[j]]
            sub = fig.add_subplot(ntheta-1, ntheta-1, (ntheta-1) * (j-1) + i + 1) 

            plotEllipse(Finv_sub, sub, theta_fid_ij=[theta_fid_i, theta_fid_j], color='C0')
            
            sub.set_xlim(theta_lims[i])
            sub.set_ylim(theta_lims[j])
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
    ffig = os.path.join(UT.fig_dir(), 'quijote_%sSNuncorr_Fisher_kmax%.2f%s.png' % (obs, kmax, ['_real', ''][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    return None

##################################################################
# qujiote fisher tests 
##################################################################
def quijote_Fisher_nmock(nmock, kmax=0.5, rsd=True, dmnu='fin'): 
    ''' calculate fisher matrix for parameters ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
    using nmock of the 15,000 quijote simulations to calculate the covariance 
    matrix.

    :param typ: 
        string that specifies the triangle shape. typ in ['equ', 'squ']. 

    :param kmax: (default: 0.5) 
        kmax 
    '''
    # read in full bispectrum 
    kf = 2.*np.pi/1000. # fundmaentla mode

    quij = quijoteBk('fiducial', rsd=rsd) # theta_fiducial 
    bks = quij['b123'] + quij['b_sn']
    bks = bks[:nmock, :]
    C_fid = np.cov(bks.T) 

    i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3']
    klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 
    C_fid = C_fid[:,klim][klim,:]

    C_inv = np.linalg.inv(C_fid) # invert the covariance 
    
    dbk_dt = [] 
    for par in thetas: # calculate the derivative of Bk along all the thetas 
        dbk_dti = quijote_dBk(par, rsd=rsd, dmnu=dmnu)
        dbk_dt.append(dbk_dti[klim])
    
    Fij = Forecast.Fij(dbk_dt, C_inv) 
    return Fij 


def quijote_forecast_nmock(kmax=0.5, rsd=True, dmnu='fin'):
    ''' fisher forecast where we compute the covariance matrix using different 
    number of mocks. 
    
    :param kmax: (default: 0.5) 
        k1, k2, k3 <= kmax
    '''
    nmocks = [1000, 3000, 5000, 7000, 9000, 11000, 13000, 15000]
    # read in fisher matrix (Fij)
    Finvs = [] 
    for nmock in nmocks: 
        Fij = quijote_Fisher_nmock(nmock, kmax=kmax, rsd=rsd, dmnu=dmnu)
        Finvs.append(np.linalg.inv(Fij)) # invert fisher matrix 

    fig = plt.figure(figsize=(6,6))
    sub = fig.add_subplot(111) 
    sub.plot([1000, 15000], [1., 1.], c='k', ls='--', lw=1) 
    sub.plot([1000, 15000], [0.9, 0.9], c='k', ls=':', lw=1) 
    for i in xrange(ntheta): 
        sig_theta = np.zeros(len(nmocks))
        for ik in range(len(nmocks)): 
            sig_theta[ik] = np.sqrt(Finvs[ik][i,i])
        sub.plot(nmocks, sig_theta/sig_theta[-1], label=r'$%s$' % theta_lbls[i]) 
        sub.set_xlim([3000, 15000]) 
    sub.legend(loc='lower right', fontsize=20) 
    sub.set_ylabel(r'$\sigma_\theta(N_{\rm fid})/\sigma_\theta(N_{\rm fid}=15,000)$', fontsize=25)
    sub.set_ylim([0.5, 1.1]) 
    sub.set_xlabel(r"$N_{\rm fid}$ Quijote realizations", labelpad=10, fontsize=25) 
    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_Fisher_nmocks_dmnu_%s_kmax%s%s.pdf' % 
            (dmnu, str(kmax).replace('.', ''), ['_real', ''][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_Fisher_nmocks_dmnu_%s_kmax%s%s.png' % 
            (dmnu, str(kmax).replace('.', ''), ['_real', ''][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def quijote_dBk_nmock(nmock, theta, rsd=True, dmnu='fin'):
    ''' calculate d B(k)/d theta using the paired and fixed quijote simulations
    run on perturbed theta 

    :param theta: 
        string that specifies the parameter to take the 
        derivative by. 
    '''
    # get all bispectrum covariances along theta  
    nmocks = [100, 200, 300, 400, 500]
    dBks = [] 
    if theta == 'Mnu': 
        Bks = [] # read in the bispectrum a Mnu+, Mnu++, Mnu+++
        for tt in ['fiducial', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp']: 
            quij = quijoteBk(tt, rsd=rsd)
            Bks.append(np.average(quij['b123'][:nmock,:], axis=0))
        Bk_fid, Bk_p, Bk_pp, Bk_ppp = Bks 

        # take the derivatives 
        h_p, h_pp, h_ppp = quijote_thetas['Mnu']
        if dmnu == 'p': 
            dBk = (Bk_p - Bk_fid) / h_p 
        elif dmnu == 'pp': 
            dBk = (Bk_pp - Bk_fid) / h_pp
        elif dmnu == 'ppp': 
            dBk = (Bk_ppp - Bk_fid) / h_ppp
        elif dmnu == 'fin':
            dBk = (-21 * Bk_fid + 32 * Bk_p - 12 * Bk_pp + Bk_ppp)/1.2 # finite difference coefficient
    else: 
        h = quijote_thetas[theta][1] - quijote_thetas[theta][0]
        
        quij = quijoteBk(theta+'_m', rsd=rsd)
        Bk_m = np.average(quij['b123'][:nmock,:], axis=0) # Covariance matrix tt- 
        quij = quijoteBk(theta+'_p', rsd=rsd)
        Bk_p = np.average(quij['b123'][:nmock,:], axis=0) # Covariance matrix tt+ 
        
        # take the derivatives 
        dBk = (Bk_p - Bk_m) / h 
    return dBk 


def quijote_Fisher_dBk_nmock(nmock, kmax=0.5, rsd=True, dmnu='fin'): 
    ''' calculate fisher matrix for parameters ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
    using nmock of the 15,000 quijote simulations to calculate the covariance 
    matrix.

    :param typ: 
        string that specifies the triangle shape. typ in ['equ', 'squ']. 

    :param kmax: (default: 0.5) 
        kmax 
    '''
    kf = 2.*np.pi/1000. # fundmaentla mode
    i_k, j_k, l_k, C_fid = quijoteCov(rsd=rsd)
    klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 
    C_fid = C_fid[:,klim][klim,:]
    
    C_inv = np.linalg.inv(C_fid) # invert the covariance 
    
    dbk_dt = [] 
    for par in thetas: # calculate the derivative of Bk along all the thetas 
        dbk_dti = quijote_dBk_nmock(nmock, par, rsd=rsd, dmnu=dmnu)
        dbk_dt.append(dbk_dti[klim])
    return Forecast.Fij(dbk_dt, C_inv) 


def quijote_forecast_dBk_nmock(kmax=0.5, rsd=True, dmnu='fin'):
    ''' fisher forecast where we compute the derivatives using different number of mocks. 
    
    :param kmax: (default: 0.5) 
    '''
    nmocks = [100, 200, 300, 400, 500]
    # read in fisher matrix (Fij)
    Finvs = [] 
    for nmock in nmocks: 
        Fij = quijote_Fisher_dBk_nmock(nmock, kmax=kmax, rsd=rsd, dmnu=dmnu)
        Finvs.append(np.linalg.inv(Fij)) # invert fisher matrix 

    fig = plt.figure(figsize=(6,6))
    sub = fig.add_subplot(111)
    sub.plot([100., 500.], [1., 1.], c='k', ls='--', lw=1) 
    sub.plot([100., 500.], [0.9, 0.9], c='k', ls=':', lw=1) 
    for i in xrange(ntheta): 
        sig_theta = np.zeros(len(nmocks))
        for ik in range(len(nmocks)): 
            sig_theta[ik] = np.sqrt(Finvs[ik][i,i])
        sub.plot(nmocks, sig_theta/sig_theta[-1], label=(r'$%s$' % theta_lbls[i]))
        sub.set_xlim([100, 500]) 
    sub.legend(loc='lower right', fontsize=20) 
    sub.set_ylabel(r'$\sigma_\theta(N_{\rm mock})/\sigma_\theta(N_{\rm mock}=500)$', fontsize=25)
    sub.set_ylim([0.5, 1.1]) 
    sub.set_xlabel(r"$N_{\rm mock}$ Quijote realizations", labelpad=10, fontsize=25) 
    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_Fisher_dBk_nmocks_dmnu_%s_kmax%s%s.pdf' % 
            (dmnu, str(kmax).replace('.', ''), ['_real', ''][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_Fisher_dBk_nmocks_dmnu_%s_kmax%s%s.png' % 
            (dmnu, str(kmax).replace('.', ''), ['_real', ''][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    return None

############################################################
def quijote_nbars(): 
    thetas = ['fiducial', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp', 'Ob_m', 'Ob_p', 'Om_m', 'Om_p', 'h_m', 'h_p', 'ns_m', 'ns_p', 's8_m', 's8_p']
    nbars = [] 
    for sub in thetas:
        quij = quijoteBk(sub)
        nbars.append(np.average(quij['Nhalos'])/1.e9)
    print np.min(nbars)
    print np.min(nbars) * 1e9 
    print thetas[np.argmin(nbars)]
    return None

############################################################
# real vs rsd
############################################################
def compare_Pk_rsd(krange=[0.01, 0.5]):  
    ''' Compare the amplitude of the fiducial HADES and quijote bispectrum 
    in redshift space versus real space. 

    :param krange: (default: [0.01, 0.5]) 
        k-range of k1, k2, k3 
    '''
    kmin, kmax = krange # impose k range 
    kf = 2.*np.pi/1000. 

    fig = plt.figure(figsize=(5,5))
    sub = fig.add_subplot(111)

    hades_fid = hadesBk(0.0, nzbin=4, rsd=False) # fiducial bispectrum
    i_k = hades_fid['k1']
    Pk_fid = np.average(hades_fid['p0k1'], axis=0)
    hades_fid = hadesBk(0.0, nzbin=4, rsd=True) # fiducial bispectrum
    Pk_fid_rsd = np.average(hades_fid['p0k1'], axis=0)

    isort = np.argsort(i_k) 
    sub.plot(kf * i_k[isort], Pk_fid[isort], c='C0', label='real-space') 
    sub.plot(kf * i_k[isort], Pk_fid_rsd[isort], c='C1', label='redshift-space') 
    print('HADES', Pk_fid_rsd / Pk_fid) 

    quij_fid = quijoteBk('fiducial', rsd=False) # fiducial bispectrum
    i_k = quij_fid['k1']
    Pk_fid = np.average(quij_fid['p0k1'], axis=0)
    quij_fid = quijoteBk('fiducial', rsd=True) # fiducial bispectrum
    Pk_fid_rsd = np.average(quij_fid['p0k1'], axis=0)

    isort = np.argsort(i_k) 
    sub.plot(kf * i_k[isort], Pk_fid[isort], c='C0', ls=':', label='Quijote') 
    sub.plot(kf * i_k[isort], Pk_fid_rsd[isort], c='C1', ls=':') 
    print('Quijote', Pk_fid_rsd / Pk_fid)

    sub.legend(loc='lower left', fontsize=25) 
    sub.set_xlabel('$k$', labelpad=15, fontsize=25) 
    sub.set_xscale('log') 
    sub.set_xlim([1e-2, 1])
    sub.set_yscale('log') 
    sub.set_ylabel('$P(k_1)$', labelpad=10, fontsize=25) 
    krange_str = str(kmin).replace('.', '')+'_'+str(kmax).replace('.', '') 
    ffig = os.path.join(UT.doc_dir(), 'figs', 
            'haloPk_amp_%s_%s_rsd_comparison.pdf' % (str(kmin).replace('.', ''), str(kmax).replace('.', '')))
    fig.savefig(ffig, bbox_inches='tight') 
    ffig = os.path.join(UT.doc_dir(), 'figs', 
            'haloPk_amp_%s_%s_rsd_comparison.png' % (str(kmin).replace('.', ''), str(kmax).replace('.', '')))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def compare_Bk_rsd(krange=[0.01, 0.5]):  
    ''' Compare the amplitude of the fiducial HADES bispectrum in
    redshift space versus real space. 

    :param krange: (default: [0.01, 0.5]) 
        k-range of k1, k2, k3 
    '''
    fig = plt.figure(figsize=(25,5))
    sub = fig.add_subplot(111)

    hades_fid = hadesBk(0.0, nzbin=4, rsd=False) # fiducial bispectrum
    Bk_fid = np.average(hades_fid['b123'], axis=0)
    hades_fid = hadesBk(0.0, nzbin=4, rsd=True) # fiducial bispectrum
    Bk_fid_rsd = np.average(hades_fid['b123'], axis=0) 

    i_k, j_k, l_k = hades_fid['k1'], hades_fid['k2'], hades_fid['k3']
    kf = 2.*np.pi/1000. 
    kmin, kmax = krange # impose k range 
    klim = ((i_k * kf <= kmax) & (i_k * kf >= kmin) &
            (j_k * kf <= kmax) & (j_k * kf >= kmin) & 
            (l_k * kf <= kmax) & (l_k * kf >= kmin)) 

    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 

    ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # order of triangles 

    tri = np.arange(np.sum(klim))
    sub.plot(tri, Bk_fid[klim][ijl], c='C0', label='real-space') 
    sub.plot(tri, Bk_fid_rsd[klim][ijl], c='C1', label='redshift-space') 

    equ = ((i_k[ijl] == j_k[ijl]) & (l_k[ijl] == i_k[ijl]))
    sub.scatter(tri[equ], Bk_fid_rsd[klim][ijl][equ], c='k', zorder=10, label='equilateral') 
    for ii_k, x, y in zip(i_k[ijl][equ][::2], tri[equ][::2], Bk_fid_rsd[klim][ijl][equ][::2]):  
        sub.text(x, 1.1*y, '$k = %.2f$' % (ii_k * kf), ha='left', va='bottom', fontsize=15)

    quij_fid = quijoteBk('fiducial', rsd=False) # fiducial bispectrum
    _Bk_fid = np.average(quij_fid['b123'], axis=0)
    quij_fid = quijoteBk('fiducial', rsd=True) # fiducial bispectrum
    _Bk_fid_rsd = np.average(quij_fid['b123'], axis=0) 

    sub.plot(tri, _Bk_fid[klim][ijl], c='C0', ls=':', label='Quijote') 
    sub.plot(tri, _Bk_fid_rsd[klim][ijl], c='C1') 

    sub.legend(loc='upper right', fontsize=25) 
    sub.set_yscale('log') 
    sub.set_xlim([0, np.sum(klim)])
    sub.set_ylim([1e6, 1e10]) 
    sub.set_xlabel('triangle configurations', labelpad=15, fontsize=25) 
    sub.set_ylabel('$B(k_1, k_2, k_3)$', labelpad=10, fontsize=25) 
    krange_str = str(kmin).replace('.', '')+'_'+str(kmax).replace('.', '') 
    ffig = os.path.join(UT.doc_dir(), 'figs', 
            'haloBk_amp_%s_%s_rsd_comparison.pdf' % (str(kmin).replace('.', ''), str(kmax).replace('.', '')))
    fig.savefig(ffig, bbox_inches='tight') 
    ffig = os.path.join(UT.doc_dir(), 'figs', 
            'haloBk_amp_%s_%s_rsd_comparison.png' % (str(kmin).replace('.', ''), str(kmax).replace('.', '')))
    fig.savefig(ffig, bbox_inches='tight') 

    fig = plt.figure(figsize=(25,5))
    sub = fig.add_subplot(111)
    tri = np.arange(np.sum(klim))
    sub.plot(tri, Bk_fid_rsd[klim][ijl]/Bk_fid[klim][ijl], c='C0', label='HADES')
    sub.plot(tri, _Bk_fid_rsd[klim][ijl]/_Bk_fid[klim][ijl], c='C1', ls=':', label='Quijote')
    sub.set_xlim([0, np.sum(klim)])
    sub.set_ylim([1., 5.]) 
    sub.set_xlabel('triangle configurations', labelpad=15, fontsize=25) 
    sub.set_ylabel('$B^{(s)}/B$', labelpad=10, fontsize=25) 
    krange_str = str(kmin).replace('.', '')+'_'+str(kmax).replace('.', '') 
    ffig = os.path.join(UT.doc_dir(), 'figs', 
            'haloBk_amp_%s_%s_rsd_ratio.png' % (str(kmin).replace('.', ''), str(kmax).replace('.', '')))
    fig.savefig(ffig, bbox_inches='tight') 

    return None 


def compare_Qk_rsd(krange=[0.01, 0.5]):  
    ''' Compare the amplitude of the fiducial HADES bispectrum in
    redshift space versus real space. 

    :param krange: (default: [0.01, 0.5]) 
        k-range of k1, k2, k3 
    '''
    hades_fid = hadesBk(0.0, nzbin=4, rsd=False) # fiducial bispectrum
    Qk_fid = np.average(hades_fid['q123'], axis=0)
    hades_fid = hadesBk(0.0, nzbin=4, rsd=True) # fiducial bispectrum
    Qk_fid_rsd = np.average(hades_fid['q123'], axis=0) 

    i_k, j_k, l_k = hades_fid['k1'], hades_fid['k2'], hades_fid['k3']
    kf = 2.*np.pi/1000. 
    kmin, kmax = krange # impose k range 
    klim = ((i_k * kf <= kmax) & (i_k * kf >= kmin) &
            (j_k * kf <= kmax) & (j_k * kf >= kmin) & 
            (l_k * kf <= kmax) & (l_k * kf >= kmin)) 

    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 

    ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # order of triangles 

    fig = plt.figure(figsize=(25,5))
    sub = fig.add_subplot(111)
    tri = np.arange(np.sum(klim))
    sub.plot(tri, Qk_fid[klim][ijl], c='C0', label='real-space') 
    sub.plot(tri, Qk_fid_rsd[klim][ijl], c='C1', label='redshift-space') 
    # mark the equilateral triangles 
    equ = ((i_k[ijl] == j_k[ijl]) & (l_k[ijl] == i_k[ijl]))
    sub.scatter(tri[equ], Qk_fid_rsd[klim][ijl][equ], c='k', zorder=10, label='equilateral') 
    for ii_k, x, y in zip(i_k[ijl][equ][::2], tri[equ][::2], Qk_fid_rsd[klim][ijl][equ][::2]):  
        sub.text(x, 0.9*y, '$k = %.2f$' % (ii_k * kf), ha='left', va='top', fontsize=15)

    quijote_fid = quijoteBk('fiducial', rsd=False) # fiducial bispectrum
    _Qk_fid = np.average(quijote_fid['q123'], axis=0)
    quijote_fid = quijoteBk('fiducial', rsd=True) # fiducial bispectrum
    _Qk_fid_rsd = np.average(quijote_fid['q123'], axis=0) 
    sub.plot(tri, _Qk_fid[klim][ijl], c='C0', ls=':', label='Quijote') 
    sub.plot(tri, _Qk_fid_rsd[klim][ijl], c='C1', ls=':') 
    
    sub.legend(loc='upper right', fontsize=25) 
    sub.set_xlim([0, np.sum(klim)])
    sub.set_ylim(0, 1) 
    sub.set_xlabel('triangle configurations', labelpad=15, fontsize=25) 
    sub.set_ylabel('$Q(k_1, k_2, k_3)$', labelpad=10, fontsize=25) 
    krange_str = str(kmin).replace('.', '')+'_'+str(kmax).replace('.', '') 
    ffig = os.path.join(UT.doc_dir(), 'figs', 
            'haloQk_amp_%s_%s_rsd_comparison.pdf' % (str(kmin).replace('.', ''), str(kmax).replace('.', '')))
    fig.savefig(ffig, bbox_inches='tight') 
    ffig = os.path.join(UT.doc_dir(), 'figs', 
            'haloQk_amp_%s_%s_rsd_comparison.png' % (str(kmin).replace('.', ''), str(kmax).replace('.', '')))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def compare_Bk_rsd_SNuncorr(krange=[0.01, 0.5]):  
    ''' Compare the amplitude of the fiducial HADES bispectrum in
    redshift space versus real space. 

    :param krange: (default: [0.01, 0.5]) 
        k-range of k1, k2, k3 
    '''
    hades_fid = hadesBk(0.0, nzbin=4, rsd=False) # fiducial bispectrum
    Bk_fid = np.average(hades_fid['b123'] + hades_fid['b_sn'], axis=0)
    hades_fid = hadesBk(0.0, nzbin=4, rsd=True) # fiducial bispectrum
    Bk_fid_rsd = np.average(hades_fid['b123'] + hades_fid['b_sn'], axis=0) 

    i_k, j_k, l_k = hades_fid['k1'], hades_fid['k2'], hades_fid['k3']
    kf = 2.*np.pi/1000. 
    kmin, kmax = krange # impose k range 
    klim = ((i_k * kf <= kmax) & (i_k * kf >= kmin) &
            (j_k * kf <= kmax) & (j_k * kf >= kmin) & 
            (l_k * kf <= kmax) & (l_k * kf >= kmin)) 

    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 

    ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # order of triangles 

    fig = plt.figure(figsize=(25,5))
    sub = fig.add_subplot(111)
    tri = np.arange(np.sum(klim))
    sub.plot(tri, Bk_fid[klim][ijl], c='C0', label='real-space') 
    sub.plot(tri, Bk_fid_rsd[klim][ijl], c='C1', label='redshift-space') 

    equ = ((i_k[ijl] == j_k[ijl]) & (l_k[ijl] == i_k[ijl]))
    sub.scatter(tri[equ], Bk_fid_rsd[klim][ijl][equ], c='k', zorder=10, label='equilateral') 
    for ii_k, x, y in zip(i_k[ijl][equ][::2], tri[equ][::2], Bk_fid_rsd[klim][ijl][equ][::2]):  
        sub.text(x, 1.1*y, '$k = %.2f$' % (ii_k * kf), ha='left', va='bottom', fontsize=15)

    sub.legend(loc='upper right', fontsize=25) 
    sub.set_yscale('log') 
    sub.set_xlim([0, np.sum(klim)])
    sub.set_ylim([5e7, 1e10]) 
    sub.set_xlabel('triangle configurations', labelpad=15, fontsize=25) 
    sub.set_ylabel('$B(k_1, k_2, k_3)$ shot noise uncorrected', labelpad=10, fontsize=25) 
    krange_str = str(kmin).replace('.', '')+'_'+str(kmax).replace('.', '') 
    ffig = os.path.join(UT.fig_dir(), 
            'haloBkSNuncorr_amp_%s_%s_rsd_comparison.png' % (str(kmin).replace('.', ''), str(kmax).replace('.', '')))
    fig.savefig(ffig, bbox_inches='tight') 

    fig = plt.figure(figsize=(25,5))
    sub = fig.add_subplot(111)
    tri = np.arange(np.sum(klim))
    sub.plot(tri, Bk_fid_rsd[klim][ijl]/Bk_fid[klim][ijl], c='C0')
    sub.set_xlim([0, np.sum(klim)])
    sub.set_ylim([1., 2.]) 
    sub.set_xlabel('triangle configurations', labelpad=15, fontsize=25) 
    sub.set_ylabel('$B^{(s)}/B$', labelpad=10, fontsize=25) 
    krange_str = str(kmin).replace('.', '')+'_'+str(kmax).replace('.', '') 
    ffig = os.path.join(UT.fig_dir(), 
            'haloBkSNuncorr_amp_%s_%s_rsd_ratio.png' % (str(kmin).replace('.', ''), str(kmax).replace('.', '')))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def compare_Qk_rsd_SNuncorr(krange=[0.01, 0.5]):  
    ''' Compare the amplitude of the fiducial HADES bispectrum in
    redshift space versus real space. 

    :param krange: (default: [0.01, 0.5]) 
        k-range of k1, k2, k3 
    '''
    hades_fid = hadesBk(0.0, nzbin=4, rsd=False) # fiducial bispectrum
    bk = np.average((hades_fid['b123'] + hades_fid['b_sn']), axis=0)
    pk1 = np.average(hades_fid['p0k1'] + 1e9/hades_fid['Nhalos'][:,None], axis=0)
    pk2 = np.average(hades_fid['p0k2'] + 1e9/hades_fid['Nhalos'][:,None], axis=0)
    pk3 = np.average(hades_fid['p0k3'] + 1e9/hades_fid['Nhalos'][:,None], axis=0)
    Qk_fid = bk / (pk1 * pk2 + pk1 * pk3 + pk2 * pk3)
    hades_fid = hadesBk(0.0, nzbin=4, rsd=True) # fiducial bispectrum
    bk = np.average((hades_fid['b123'] + hades_fid['b_sn']), axis=0)
    pk1 = np.average(hades_fid['p0k1'] + 1e9/hades_fid['Nhalos'][:,None], axis=0)
    pk2 = np.average(hades_fid['p0k2'] + 1e9/hades_fid['Nhalos'][:,None], axis=0)
    pk3 = np.average(hades_fid['p0k3'] + 1e9/hades_fid['Nhalos'][:,None], axis=0)
    Qk_fid_rsd = bk / (pk1 * pk2 + pk1 * pk3 + pk2 * pk3)

    i_k, j_k, l_k = hades_fid['k1'], hades_fid['k2'], hades_fid['k3']
    kf = 2.*np.pi/1000. 
    kmin, kmax = krange # impose k range 
    klim = ((i_k * kf <= kmax) & (i_k * kf >= kmin) &
            (j_k * kf <= kmax) & (j_k * kf >= kmin) & 
            (l_k * kf <= kmax) & (l_k * kf >= kmin)) 

    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 

    ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # order of triangles 

    fig = plt.figure(figsize=(25,5))
    sub = fig.add_subplot(111)
    tri = np.arange(np.sum(klim))
    sub.plot(tri, Qk_fid[klim][ijl], c='C0', label='real-space') 
    sub.plot(tri, Qk_fid_rsd[klim][ijl], c='C1', label='redshift-space') 
    
    # mark the equilateral triangles 
    equ = ((i_k[ijl] == j_k[ijl]) & (l_k[ijl] == i_k[ijl]))
    sub.scatter(tri[equ], Qk_fid_rsd[klim][ijl][equ], c='k', s=2, zorder=10, label='equilateral') 
    for ii_k, x, y in zip(i_k[ijl][equ][::2], tri[equ][::2], Qk_fid_rsd[klim][ijl][equ][::2]):  
        sub.text(x, 0.9*y, '$k = %.2f$' % (ii_k * kf), ha='left', va='top', fontsize=15)

    sub.legend(loc='upper right', fontsize=25) 
    sub.set_xlim([0, np.sum(klim)])
    sub.set_ylim(0, 1) 
    sub.set_xlabel('triangle configurations', labelpad=15, fontsize=25) 
    sub.set_ylabel('$Q(k_1, k_2, k_3)$ shot noise uncorrected', labelpad=10, fontsize=25) 
    krange_str = str(kmin).replace('.', '')+'_'+str(kmax).replace('.', '') 
    ffig = os.path.join(UT.fig_dir(), 
            'haloQkSNuncorr_amp_%s_%s_rsd_comparison.png' % (str(kmin).replace('.', ''), str(kmax).replace('.', '')))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


if __name__=="__main__": 
    #compare_Plk(nreals=range(1,101), krange=[0.01, 0.5])
    #ratio_Plk(nreals=range(1,101), krange=[0.01, 0.5])
    for kmax in [0.5]: 
        continue 
        compare_Bk(krange=[0.01, kmax], rsd=True)
        compare_Bk_shape(krange=[0.01, kmax], rsd=True, nbin=31)
        compare_Qk(krange=[0.01, kmax], rsd=True)

    # covariance matrices
    for kmax in [0.5]: 
        continue 
        for rsd in [True, False]: 
            quijote_pkCov(kmax=kmax, rsd=rsd) 
            quijote_bkCov(kmax=kmax, rsd=rsd)
            quijote_pkbkCov(kmax=kmax, rsd=rsd)

    # fisher forecasts 
    for rsd in [False]: #, False]: 
        for kmax in [0.5]: #[0.2, 0.3, 0.4, 0.5]: 
            for dmnu in ['fin']: #['p', 'pp', 'ppp', 'fin']: 
                continue 
                quijote_Forecast('pk', kmax=kmax, rsd=rsd, dmnu=dmnu)
                quijote_Forecast('bk', kmax=kmax, rsd=rsd, dmnu=dmnu)
                #quijote_pbkForecast(kmax=kmax, rsd=rsd, dmnu=dmnu)
        #quijote_Forecast_kmax('pk', rsd=rsd) 
        #quijote_Forecast_kmax('bk', rsd=rsd)
        #quijote_pkForecast_dmnu(rsd=rsd)
        #quijote_bkForecast_dmnu(rsd=rsd)
    # rsd 
    #compare_Pk_rsd(krange=[0.01, 0.5])
    #compare_Bk_rsd(krange=[0.01, 0.5])
    #compare_Qk_rsd(krange=[0.01, 0.5])
    
    #hades_dchi2(krange=[0.01, 0.5])
    
    # amplitude scaling factor is a free parameter
    for kmax in [0.2, 0.5]: 
        continue 
        print('kmax = %.1f' % kmax) 
        quijote_Forecast_freeScale('pk', kmax=kmax, rsd=True, dmnu='fin')
        quijote_Forecast_freeScale('bk', kmax=kmax, rsd=True, dmnu='fin')
        #quijote_Forecast_freeScale('bk_equ', kmax=kmax, rsd=True, dmnu='fin')
        #quijote_Forecast_freeScale('bk_squ', kmax=kmax, rsd=True, dmnu='fin')
        #quijote_Forecast_freeScale('bk_equ', kmax=kmax, rsd=False, dmnu='fin')
        #quijote_Forecast_freeScale('bk_squ', kmax=kmax, rsd=False, dmnu='fin')
    quijote_Bratio_thetas(kmax=0.5)
    quijote_B_relative_error(kmax=0.5)

    # SN uncorrected forecasts 
    #compare_Bk_SNuncorr(krange=[0.01, 0.5], rsd=True)
    #compare_Qk_SNuncorr(krange=[0.01, 0.5], rsd=True)
    #compare_Bk_rsd_SNuncorr(krange=[0.01, 0.5])
    #compare_Qk_rsd_SNuncorr(krange=[0.01, 0.5])
    #for rsd in [True, False]:  
    #    quijote_Forecast_SNuncorr('pk', kmax=0.5, rsd=rsd, dmnu='fin')
    #    quijote_Forecast_SNuncorr('bk', kmax=0.5, rsd=rsd, dmnu='fin')
    #quijote_nbars()
        
    # kmax test
    #for k in [0.2, 0.3, 0.4]: 
    #    B123_kmax_test(kmin=0.01, kmax1=k, kmax2=k+0.1, rsd=True)

    # usingly only specific triangles
    #compare_Bk_triangles([[30, 18], [18, 18], [12,9]], rsd=True)

    # equilateral triangles 
    #for shape in ['equ', 'squ']: 
    #    compare_Bk_triangle(shape, rsd=True)
    #quijote_forecast_triangle_kmax('equ')
    #quijote_pkbkCov_triangle('equ', krange=[0.01, 0.5])
    
    # convergence tests 
    #quijote_forecast_nmock(kmax=0.5, rsd=True)
    #quijote_forecast_dBk_nmock(kmax=0.5, rsd=True)
