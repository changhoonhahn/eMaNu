'''

figures looking at hades halo catalog with massive neutrinos


'''
import os 
import h5py
import numpy as np 
# --- eMaNu --- 
from emanu import util as UT
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
mpl.rcParams['legend.frameon'] = False


def checkCCA(): 
    ''' check that the Paco's CCA bispectrum runs are consistent with my 
    local bispectrum runs
    '''
    # read in my runs
    fb123 = os.path.join(UT.doc_dir(), 'dat/', 'halo_bispectrum.0.0eV.1_100.z4.zspace.hdf5') 
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

    # read in cca runs 
    fcca = lambda i: os.path.join(UT.dat_dir(), 'bispectrum', 'cca', 'Bk_%i.dat' % i) 
    nbk = 0 

    p0k1s, p0k2s, p0k3s = [], [], [] 
    B123s, Q123s, B_SNs, countss = [], [], [], [] 
    for i in range(200): 
        try: 
            _,_,_, _p0k1, _p0k2, _p0k3, b123, q123, b_sn, cnts = np.loadtxt(fcca(i), 
                    skiprows=1, unpack=True, usecols=range(10)) 
            p0k1s.append(_p0k1)
            p0k2s.append(_p0k2)
            p0k3s.append(_p0k3)
            B123s.append(b123)
            B_SNs.append(b_sn)
            Q123s.append(q123)
            countss.append(cnts)
            nbk += 1
        except IOError: 
            pass 
    
    # check powerspectrum
    fig = plt.figure(figsize=(6,6))
    sub = fig.add_subplot(111)
    isort = np.argsort(i_k) 
    for p0k in p0k1s: 
        sub.plot(i_k[isort] * k_f, p0k[isort], c='k', alpha=0.5, lw=0.2)
    sub.plot(i_k[isort] * k_f, np.average(p0k1, axis=0)[isort], c='C0', lw=2, ls='--', label='old runs')
    sub.plot(i_k[isort] * k_f, np.average(p0k1s, axis=0)[isort], c='C1', lw=2, ls=':', label='new runs')
    sub.legend(loc='lower left', fontsize=20) 
    sub.set_xscale("log") 
    sub.set_xlabel('$k$', fontsize=25) 
    sub.set_xlim([1e-2, 0.5])
    sub.set_yscale("log") 
    sub.set_ylim([1e3, 2e5]) 
    sub.set_ylabel('$P_0(k)$', fontsize=25) 
    fig.savefig(os.path.join(UT.fig_dir(), 'pk.ccacheck.png'), bbox_inches='tight') 

    klim = ((i_k*k_f <= 0.5) & (i_k*k_f >= 0.01) &
            (j_k*k_f <= 0.5) & (j_k*k_f >= 0.01) & 
            (l_k*k_f <= 0.5) & (l_k*k_f >= 0.01)) 

    ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # order of triangles 

    fig = plt.figure(figsize=(25,8))
    sub = fig.add_subplot(111)
    for _b123 in B123s: 
        sub.plot(range(np.sum(klim)), _b123[ijl][klim], c='k', alpha=0.5, lw=0.2)
    sub.plot(range(np.sum(klim)), np.average(B123, axis=0)[ijl][klim], c='C0', lw=2, ls='--', label='old runs') 
    sub.plot(range(np.sum(klim)), np.average(B123s, axis=0)[ijl][klim], c='C1', lw=1, ls=':', label='new runs')
    sub.legend(loc='upper right', fontsize=20) 
    sub.set_yscale('log') 
    sub.set_xlim([0, 1e2]) #np.sum(klim)])
    sub.set_ylim([1e7, 1e10]) 
    fig.savefig(os.path.join(UT.fig_dir(), 'bk.ccacheck.png'), bbox_inches='tight') 

    return None 


if __name__=="__main__": 
    checkCCA()
