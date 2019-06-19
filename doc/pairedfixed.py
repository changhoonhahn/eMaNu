'''

investigate paired-fixed simulations for the bispectrum

'''
import os 
import h5py
import numpy as np 
from copy import copy as copy
# --- eMaNu --- 
from emanu import util as UT
from emanu import obvs as Obvs
# --- plotting --- 
import matplotlib as mpl
import matplotlib.pyplot as plt
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

dir_fig = UT.fig_dir()
kf = 2.*np.pi/1000. 


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
        sub.plot(range(np.sum(klim)), (bk_fid_ncv/bk_fid)[klim], c='k', lw=0.1) 
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
    for kmax in [0.5, 0.4, 0.3, 0.2, 0.1]: 
        klim = ((i_k * kf <= kmax) & (j_k * kf <= kmax) & (l_k * kf <= kmax)) 
        sub.plot(range(np.sum(klim)), np.average(ratios, axis=0)[klim], label=r'$k_{\rm max} = %.1f$' % kmax) 
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



if __name__=="__main__": 
    #for theta in ['fiducial', 'Om_m', 'Om_p', 'Ob2_m', 'Ob2_p', 'h_m', 'h_p', 'ns_m', 'ns_p', 's8_m', 's8_p', 
    #        'Mnu_p', 'Mnu_pp', 'Mnu_ppp']: 
    #    pairedfixed_theta(theta, kmax=0.5) 
    pairedfixed_allthetas()
