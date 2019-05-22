'''

quick script to compare dlogB / dtheta using the different 
derivative methods 

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

thetas = ['Om', 'Ob2', 'h', 'ns', 's8', 'Mnu', 'Mmin'] 
theta_lbls = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$', r'$M_{min}$']

kf = 2.*np.pi/1000.

def dlogBdMnu(kmax=0.5):
    ''' Compare the d logB / d theta for the fixed-paired 1 RSD direction, N-body 1 RSD direction, 
    and full fixed-paired 3 RSD directions + N-body 3 RSD directions 
    '''

    ffid = os.path.join(UT.doc_dir(), 'dat', 'dBdtheta.Mnu.dat') 
        
    i_k, j_k, l_k, dlogBdt_fid_fin, dlogBdt_fid_fin0, dlogBdt_fid_p = np.loadtxt(ffid, skiprows=1, unpack=True, 
            usecols=[0,1,2,4,6,8]) 
    dlogBdts_fid = [dlogBdt_fid_fin, dlogBdt_fid_fin0, dlogBdt_fid_p] # fin. diff w/ 0, 0.1, 0.2, 0.4, fin. diff w/ 0, 0.1 0.2, 0, 0.1 

    fncv = os.path.join(UT.doc_dir(), 'dat', 'dBdtheta.Mnu.ncv.rsd0.dat')
    dlogBdts_ncv = np.loadtxt(fncv, skiprows=1, unpack=True, usecols=[4,6,8]) 

    freg = os.path.join(UT.doc_dir(), 'dat', 'dBdtheta.Mnu.reg.rsd0.dat')
    dlogBdts_reg = np.loadtxt(freg, skiprows=1, unpack=True, usecols=[4,6,8]) 
        
    klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 
    
    plts_fid = [] 
    fig = plt.figure(figsize=(20, 10))
    sub = fig.add_subplot(111)
    for i, d0, d1, d2 in zip(range(3), dlogBdts_fid, dlogBdts_ncv, dlogBdts_reg): 
        plt_ncv, = sub.plot(range(np.sum(klim)), d2[klim], c='C%i' % i, ls=':')
        plt_reg, = sub.plot(range(np.sum(klim)), d1[klim], c='C%i' % i, ls='--') 
        plt_fid, = sub.plot(range(np.sum(klim)), d0[klim], c='C%i' % i, lw=1, ls='-') 
        sub.plot(range(np.sum(klim)), d0[klim], c='k', lw=0.5, ls='-') 

        #sub.legend([plt_fid, plt_ncv, plt_reg], ['full', 'fixed-paired', 'N-body'], loc='upper left', fontsize=20)
        plts_fid.append(plt_fid) 
    sub.set_xlim(0, np.sum(klim)) 
    sub.set_ylim(-1., 1.5) 
    sub.legend(plts_fid, ['0, 0.1, 0.2, 0.4eV', '0, 0.1, 0.2eV', '0, 0.1eV'], loc='upper right', fontsize=20)
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel('triangle configuration', fontsize=25) 
    bkgd.set_ylabel(r'${\rm d}\log B/{\rm d} M_\nu$', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ffig = os.path.join(UT.fig_dir(), '_quijote_dlogBdMnu_check.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None


if __name__=="__main__": 
    dlogBdMnu(kmax=0.5)
