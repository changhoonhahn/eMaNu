'''

script to check the B derivatives from Quijote 

'''
import os
import numpy as np 
from emanu import util as UT 
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

thetas = ['Om', 'Ob2', 'h', 'ns', 's8', 'Mmin'] 
theta_lbls = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_{min}$']#, r'$M_\nu$']

kf = 2.*np.pi/1000.

def dlogBdthetas(kmax=0.5):
    ''' Compare the d logB / d theta for the fixed-paired 1 RSD direction, N-body 1 RSD direction, 
    and full fixed-paired 3 RSD directions + N-body 3 RSD directions 
    '''
    fig = plt.figure(figsize=(20, 5*len(thetas)))
    for i_tt, tt, lbl in zip(range(len(thetas)), thetas, theta_lbls): 
        ffid = os.path.join(UT.doc_dir(), 'dat', 'dBdtheta.%s.dat' % tt) 
        
        i_k, j_k, l_k, dlogBdt_fid = np.loadtxt(ffid, skiprows=1, unpack=True, usecols=[0,1,2,4]) 

        fncv = os.path.join(UT.doc_dir(), 'dat', 'dBdtheta.%s.ncv.rsd0.dat' % tt)
        dlogBdt_ncv = np.loadtxt(fncv, skiprows=1, unpack=True, usecols=[4]) 

        freg = os.path.join(UT.doc_dir(), 'dat', 'dBdtheta.%s.reg.rsd0.dat' % tt)
        dlogBdt_reg = np.loadtxt(freg, skiprows=1, unpack=True, usecols=[4]) 
        
        klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 
        #ijl = UT.ijl_order(i_k[klim], j_k[klim], l_k[klim], typ='GM') # order of triangles 

        sub = fig.add_subplot(len(thetas),1,i_tt+1)
        plt_ncv, = sub.plot(range(np.sum(klim)), dlogBdt_ncv[klim], c='C0', ls='--') 
        plt_reg, = sub.plot(range(np.sum(klim)), dlogBdt_reg[klim], c='C1', ls=':') 
        plt_fid, = sub.plot(range(np.sum(klim)), dlogBdt_fid[klim], c='k', lw=1) 

        sub.set_xlim(0, np.sum(klim)) 
        if i_tt == 0:
            sub.legend([plt_fid, plt_ncv, plt_reg], ['full', 'fixed-paired', 'N-body'], loc='upper left', fontsize=20)
        sub.text(0.975, 0.95, lbl, ha='right', va='top', transform=sub.transAxes, fontsize=30)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel('triangle configuration', fontsize=25) 
    bkgd.set_ylabel(r'${\rm d}\log B/{\rm d} \theta$', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ffig = os.path.join(UT.fig_dir(), '_quijote_dlogBdthetas_check.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def dlogBdthetas_ratio(kmax=0.5):
    ''' Compare the derivatives of the bispectrum 
    '''
    fig = plt.figure(figsize=(20, 5*len(thetas)))
    for i_tt, tt, lbl in zip(range(len(thetas)), thetas, theta_lbls): 
        ffid = os.path.join(UT.doc_dir(), 'dat', 'dBdtheta.%s.dat' % tt) 
        
        i_k, j_k, l_k, dlogBdt_fid = np.loadtxt(ffid, skiprows=1, unpack=True, usecols=[0,1,2,4]) 

        fncv = os.path.join(UT.doc_dir(), 'dat', 'dBdtheta.%s.ncv.rsd0.dat' % tt)
        dlogBdt_ncv = np.loadtxt(fncv, skiprows=1, unpack=True, usecols=[4]) 

        freg = os.path.join(UT.doc_dir(), 'dat', 'dBdtheta.%s.reg.rsd0.dat' % tt)
        dlogBdt_reg = np.loadtxt(freg, skiprows=1, unpack=True, usecols=[4]) 
        
        klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 
        #ijl = UT.ijl_order(i_k[klim], j_k[klim], l_k[klim], typ='GM') # order of triangles 

        sub = fig.add_subplot(len(thetas),1,i_tt+1)
        plt_ncv, = sub.plot(range(np.sum(klim)), dlogBdt_ncv[klim]/dlogBdt_fid[klim], c='C0') 
        plt_reg, = sub.plot(range(np.sum(klim)), dlogBdt_reg[klim]/dlogBdt_fid[klim], c='C1') 
        sub.plot([0., np.sum(klim)], [1., 1.], c='k', ls='--') 

        sub.set_xlim(0, np.sum(klim)) 
        sub.set_ylim(0.6, 1.4) 
        if i_tt == 0: sub.legend([plt_ncv, plt_reg], ['fixed-paired', 'N-body'], loc='upper left', fontsize=20)
        sub.text(0.975, 0.95, lbl, ha='right', va='top', transform=sub.transAxes, fontsize=30)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel('triangle configuration', fontsize=25) 
    bkgd.set_ylabel(r'${\rm d}\log B/{\rm d} \theta$ ratio', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ffig = os.path.join(UT.fig_dir(), '_quijote_dlogBdthetas_ratio_check.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None



if __name__=="__main__": 
    dlogBdthetas(kmax=0.5)
    dlogBdthetas_ratio(kmax=0.5) 
