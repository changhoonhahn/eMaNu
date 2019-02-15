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


def quijoteBk(theta): 
    ''' read in bispectra for specified theta of the quijote simulations. 
    
    :param theta: 
        string that specifies which quijote run. `theta==fiducial`  
        for the fiducial set of parameters. 

    :return _bks: 
        dictionary that contains all the bispectrum data from quijote simulation
    '''
    dir_bk = os.path.join(UT.dat_dir(), 'bispectrum') 
    bks = h5py.File(os.path.join(dir_bk, 'quijote_%s.hdf5' % theta), 'r') 
    
    _bks = {}
    for k in bks.keys(): 
        _bks[k] = bks[k].value 
    return _bks


def quijoteCov(): 
    ''' read in the full covariance matrix of the quijote bispectrum
    computed using 15000 simulations at the fiducial parameter values. 

    :return cov:
        big ass covariance matrix of all the triangle configurations in 
        the default ordering. 
    '''
    # read in fiducial covariance matrix 
    fcov = h5py.File(os.path.join(UT.dat_dir(), 'bispectrum', 'quijote_Cov_full.hdf5'), 'r') 
    cov = fcov['C_bk'].value
    return cov


def compare_Bk(krange=[0.01, 0.5], rsd=True):  
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
        sub.set_xlim([0, 1200])
        sub.set_ylim([1e5, 5e9]) 
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
    bkgd.set_xlabel(r'$k_1 \le k_2 \le k_3$ triangle indices', labelpad=15, fontsize=25) 
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
    
        db = bks8[klim][ijl]/bk_fid - 1.
        sub.plot(tri, db, lw=1, c='k', label='$\sigma_8=%.3f$' % sig8) 
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
    bkgd.set_xlabel(r'$k_1 \le k_2 \le k_3$ triangle indices', labelpad=10, fontsize=30) 
    bkgd.set_ylabel('$B(k_1, k_2, k_3) / B^\mathrm{fid} - 1 $', labelpad=15, fontsize=30) 
    fig.subplots_adjust(hspace=0.1)
    ffig = os.path.join(UT.doc_dir(), 'figs', 
            'haloBk_residual_%s_%s%s.pdf' % 
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

    Bk_Mnu, Bk_s8s = [], [] 
    for mnu in mnus: # bispectrum for different Mnu
        hades_i = hadesBk(mnu, nzbin=4, rsd=rsd)
        Bk_Mnu.append(np.average(hades_i['b123'], axis=0)) 
    for sig8 in sig8s: # bispectrum for different sigma8 
        hades_i = hadesBk_s8(sig8, nzbin=4, rsd=rsd)
        Bk_s8s.append(np.average(hades_i['b123'], axis=0)) 

    # covariance matrix
    C_full = quijoteCov()

    kf = 2.*np.pi/1000.
    i_k, j_k, l_k = hades_i['k1'], hades_i['k2'], hades_i['k3']
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

##################################################################
# qujiote fisher 
##################################################################
def quijote_covariance(krange=[0.01, 0.5]): 
    ''' plot the covariance matrix of the quijote fiducial 
    bispectrum. 
    '''
    # read in full covariance matrix 
    fcov = h5py.File(os.path.join(UT.dat_dir(), 'bispectrum', 'quijote_Cov_full.hdf5'), 'r') 
    C_bk = fcov['C_bk'].value
    i_k, j_k, l_k = fcov['k1'].value, fcov['k2'].value, fcov['k3'].value 
     
    # impose k limit 
    kmin, kmax = krange 
    kf = 2.*np.pi/1000. # fundmaentla mode
    klim = ((i_k*kf <= kmax) & (i_k*kf >= kmin) &
            (j_k*kf <= kmax) & (j_k*kf >= kmin) & 
            (l_k*kf <= kmax) & (l_k*kf >= kmin)) 
    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 
    C_bk = C_bk[:,klim][klim,:]

    # order the triangles 
    ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # order of triangles 
    C_bk = C_bk[:,ijl][ijl,:] 

    # plot the covariance matrix 
    fig = plt.figure(figsize=(10,8))
    sub = fig.add_subplot(111)
    cm = sub.pcolormesh(C_bk, norm=LogNorm(vmin=1e11, vmax=1e18))
    cbar = fig.colorbar(cm, ax=sub) 
    sub.set_title(r'Quijote $B(k_1, k_2, k_3)$ Covariance', fontsize=25)
    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_Cov_%s_%s.png' % (str(kmin).replace('.', ''), str(kmax).replace('.', '')))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def quijote_dBk(theta):
    ''' calculate d B(k)/d theta using the paired and fixed quijote simulations
    run on perturbed theta 

    :param theta: 
        string that specifies the parameter to take the 
        derivative by. 
    '''
    quij = quijoteBk('fiducial') # theta_fiducial 
    Bk_fid = np.average(quij['b123'], axis=0) 
    i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3']

    # get all bispectrum covariances along theta  
    theta_dict = {
            'Mnu': [0.1, 0.2, 0.4], # +, ++, +++ 
            'Ob': [0.048, 0.050], # others are - + 
            'Om': [0.3075, 0.3275],
            'h': [0.6511, 0.6911],
            'ns': [0.9424, 0.9824],
            's8': [0.819, 0.849]} 

    if theta == 'Mnu': 
        h_p, h_pp, h_ppp = theta_dict['Mnu']
        
        Bks = [] # read in the bispectrum a Mnu+, Mnu++, Mnu+++
        for p in ['_p', '_pp', '_ppp']: 
            quij = quijoteBk(theta+'_p')
            Bks.append(np.average(quij['b123'], axis=0))
        Bk_p, Bk_pp, Bk_ppp = Bks 

        # take the derivatives 
        #dBk_p = (Bk_p - Bk_fid) / h_p 
        #dBk_pp = (Bk_pp - Bk_fid) / h_pp
        #dBk_ppp = (Bk_ppp - Bk_fid) / h_ppp
        dBk = (-21 * Bk_fid + 32 * Bk_p - 12 * Bk_pp + Bk_ppp)/(1.2) # finite difference coefficient
    else: 
        h = theta_dict[theta][1] - theta_dict[theta][0]
        
        quij = quijoteBk(theta+'_m')
        Bk_m = np.average(quij['b123'], axis=0) # Covariance matrix tt- 
        quij = quijoteBk(theta+'_p')
        Bk_p = np.average(quij['b123'], axis=0) # Covariance matrix tt+ 
        
        # take the derivatives 
        dBk = (Bk_p - Bk_m) / h 
    return dBk


def quijote_Fisher(krange=[0.01, 0.5]): 
    ''' calculate fisher matrix for parameters ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
    '''
    thetas = ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
    theta_lbls = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$']

    # read in full covariance matrix (with shotnoise; this is the correct one) 
    fcov = h5py.File(os.path.join(UT.dat_dir(), 'bispectrum', 'quijote_Cov_full.hdf5'), 'r') 
    C_fid = fcov['C_bk'].value
    i_k, j_k, l_k = fcov['k1'].value, fcov['k2'].value, fcov['k3'].value 

    # impose k limit 
    kmin, kmax = krange 
    kf = 2.*np.pi/1000. # fundmaentla mode
    klim = ((i_k*kf <= kmax) & (i_k*kf >= kmin) &
            (j_k*kf <= kmax) & (j_k*kf >= kmin) & 
            (l_k*kf <= kmax) & (l_k*kf >= kmin)) 
    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 
    C_fid = C_fid[:,klim][klim,:]

    C_inv = np.linalg.inv(C_fid) # invert the covariance 
    
    dbk_dt = [] 
    for par in thetas: # calculate the derivative of Bk along all the thetas 
        dbk_dti = quijote_dBk(par)
        dbk_dt.append(dbk_dti[klim])
    
    Fij = np.zeros((len(thetas), len(thetas)))
    for i, par_i in enumerate(thetas): 
        for j, par_j in enumerate(thetas): 
            dbk_dtt_i, dbk_dtt_j = dbk_dt[i], dbk_dt[j]

            # calculate Mij 
            Mij = np.dot(dbk_dtt_i[:,None], dbk_dtt_j[None,:]) + np.dot(dbk_dtt_j[:,None], dbk_dtt_i[None,:])

            fij = 0.5 * np.trace(np.dot(C_inv, Mij))
            Fij[i,j] = fij 
    
    f_ij = os.path.join(UT.dat_dir(), 'bispectrum', 
            'quijote_Fisher.%.2f_%.2f.gmorder.hdf5' % (kmin, kmax))
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

    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_Fisher.%.2f_%.2f.gmorder.pdf' % (kmin, kmax))
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def quijote_forecast(krange=[0.01, 0.5]):
    ''' fisher forecast for quijote 
    
    :param krange: (default: [0.01, 0.5]) 
        tuple specifying the kranges of k1, k2, k3 in the bispectrum
    '''
    # read in fisher matrix (Fij)
    bk_dir = os.path.join(UT.dat_dir(), 'bispectrum')
    f_ij = os.path.join(bk_dir, 'quijote_Fisher.%.2f_%.2f.gmorder.hdf5' % (krange[0], krange[1]))
    if not os.path.isfile(f_ij): quijote_Fisher(krange=krange)
    f = h5py.File(f_ij, 'r') 
    Fij = f['Fij'].value

    Finv = np.linalg.inv(Fij) # invert fisher matrix 

    thetas = ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
    theta_lbls = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$']
    theta_lims = [(0.275, 0.375), (0.03, 0.07), (0.5, 0.9), (0.75, 1.2), (0.8, 0.87), (-0.25, 0.25)]
    theta_fid = {'Mnu': 0., 'Ob': 0.049, 'Om': 0.3175, 'h': 0.6711,  'ns': 0.9624,  's8': 0.834} # fiducial theta 
    ntheta = len(thetas)
    
    for i in xrange(ntheta): 
        if thetas[i] == 'Mnu': print thetas[i], np.sqrt(Finv[i,i])

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
            
            sub.set_xlim(theta_lims[i])
            sub.set_ylim(theta_lims[j])
            #sub.set_xlim([theta_fid_i - x_range, theta_fid_i + x_range])
            #sub.set_ylim([theta_fid_j - y_range, theta_fid_j + y_range])
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
    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_Fisher_%.2f_%.2f.png' % (krange[0], krange[1]))
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def quijote_Fisher_triangle(typ, krange=[0.01, 0.5]): 
    ''' calculate fisher matrix for parameters ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
    using only triangle configurations up to kmax

    :param typ: 
        string that specifies the triangle shape. typ in ['equ', 'squ']. 

    :param kmax: (default: 0.5) 
        kmax 
    '''
    thetas = ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
    theta_lbls = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$']

    # read in full covariance matrix (with shotnoise; this is the correct one) 
    fcov = h5py.File(os.path.join(UT.dat_dir(), 'bispectrum', 'quijote_Cov_full.hdf5'), 'r') 
    C_fid = fcov['C_bk'].value
    i_k, j_k, l_k = fcov['k1'].value, fcov['k2'].value, fcov['k3'].value 

    # impose k limit 
    if typ == 'equ': # only equilateral 
        tri = (i_k == j_k) & (j_k == l_k) 
    elif typ == 'squ': # i_k >= j_k >= l_k (= 3*kf~0.01) 
        tri = (i_k == j_k) & (l_k == 3)
    kf = 2.*np.pi/1000. # fundmaentla mode
    kmin, kmax = krange 
    klim = (tri & 
            (i_k*kf <= kmax) & (i_k*kf >= kmin) &
            (j_k*kf <= kmax) & (j_k*kf >= kmin) & 
            (l_k*kf <= kmax) & (l_k*kf >= kmin)) 
    print('%i %s triangle configurations' % (np.sum(klim), typ))
    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 
    C_fid = C_fid[:,klim][klim,:]

    C_inv = np.linalg.inv(C_fid) # invert the covariance 
    
    dbk_dt = [] 
    for par in thetas: # calculate the derivative of Bk along all the thetas 
        dbk_dti = quijote_dBk(par)
        dbk_dt.append(dbk_dti[klim])
    
    Fij = np.zeros((len(thetas), len(thetas)))
    for i, par_i in enumerate(thetas): 
        for j, par_j in enumerate(thetas): 
            dbk_dtt_i, dbk_dtt_j = dbk_dt[i], dbk_dt[j]

            # calculate Mij 
            Mij = np.dot(dbk_dtt_i[:,None], dbk_dtt_j[None,:]) + np.dot(dbk_dtt_j[:,None], dbk_dtt_i[None,:])

            fij = 0.5 * np.trace(np.dot(C_inv, Mij))
            Fij[i,j] = fij 
    
    f_ij = os.path.join(UT.dat_dir(), 'bispectrum', 
            'quijote_Fisher.%s.%.2f_%.2f.gmorder.hdf5' % (typ, kmin, kmax))
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
    if typ == 'equ': 
        sub.set_title(r'Fisher Matrix $F_{i,j}$; {\em equilateral} triangles only', fontsize=25)
    elif typ == 'squ': 
        sub.set_title(r'Fisher Matrix $F_{i,j}$; {\em squeezed} triangles only', fontsize=25)
    fig.colorbar(cm)
    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_Fisher.%s.%.2f_%.2f.gmorder.pdf' % (typ, kmin, kmax))
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def quijote_forecast_triangle_kmax(typ):
    ''' fisher forecast for quijote using only specific triangle configuration 
    
    :param krange: (default: [0.01, 0.5]) 
        tuple specifying the kranges of k1, k2, k3 in the bispectrum
    '''
    bk_dir = os.path.join(UT.dat_dir(), 'bispectrum')
    kmaxs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    # read in fisher matrix (Fij)
    Finvs = [] 
    for kmax in kmaxs: 
        f_ij = os.path.join(bk_dir, 'quijote_Fisher.%s.%.2f_%.2f.gmorder.hdf5' % (typ, 0.01, kmax))
        if not os.path.isfile(f_ij): quijote_Fisher_triangle(typ, krange=[0.01, kmax])
        f = h5py.File(f_ij, 'r') 
        Fij = f['Fij'].value
        Finvs.append(np.linalg.inv(Fij)) # invert fisher matrix 

    thetas = ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
    theta_lbls = [r'\Omega_m', r'\Omega_b', r'h', r'n_s', r'\sigma_8', r'M_\nu']
    theta_lims = [(0.2, 0.3), (0.01, 0.15), (0.2, 1.5), (0.6, 1.6), (0.6, 0.8), (-0.4, 0.5)]
    theta_fid = {'Mnu': 0., 'Ob': 0.049, 'Om': 0.3175, 'h': 0.6711,  'ns': 0.9624,  's8': 0.834} # fiducial theta 
    ntheta = len(thetas)
    
    fig = plt.figure(figsize=(15, 10))
    for i in xrange(ntheta): 
        sub = fig.add_subplot(2,ntheta/2,i+1) 
        sig_theta = np.zeros(len(kmaxs))
        for ik, kmax in enumerate(kmaxs): 
            sig_theta[ik] = np.sqrt(Finvs[ik][i,i])

        sub.plot(kmaxs, sig_theta) 

        sub.set_xlim([kmaxs[0], kmaxs[-1]]) 
        sub.set_ylabel(r'$\sigma_{%s}$' % theta_lbls[i], fontsize=25)
        sub.set_ylim([0., theta_lims[i][1]]) 

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r"$k_{\rm max}$ [$h$/Mpc]", labelpad=10, fontsize=25) 
    fig.subplots_adjust(wspace=0.3, hspace=0.1) 
    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_Fisher_%s_kmax.png' % (typ))
    fig.savefig(ffig, bbox_inches='tight') 
    return None


if __name__=="__main__": 
    #compare_Plk(nreals=range(1,101), krange=[0.01, 0.5])
    #ratio_Plk(nreals=range(1,101), krange=[0.01, 0.5])

    for kmax in [0.2, 0.3, 0.4, 0.5]: 
        continue 
        #compare_Bk(krange=[0.01, kmax], rsd=rsd)
        #compare_Bk_shape(krange=[0.01, kmax], rsd=rsd, nbin=31)
        #quijote_covariance(krange=[0.01, kmax])
        #quijote_forecast(krange=[0.01, kmax])
    
    # kmax text
    #for k in [0.2, 0.3, 0.4]: 
    #    B123_kmax_test(kmin=0.01, kmax1=k, kmax2=k+0.1, rsd=True)

    # usingly only specific triangles
    #compare_Bk_triangles([[30, 18], [18, 18], [12,9]], rsd=True)

    # equilateral triangles 
    compare_Bk_triangle('equ', rsd=True)
    quijote_forecast_triangle_kmax('equ')

    # squeezed triangles 
    compare_Bk_triangle('squ', rsd=True)
    quijote_forecast_triangle_kmax('squ')
