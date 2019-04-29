''' quick script to check the derivatives of P and B w.r.t. Mmin 
'''
import os
import h5py
import numpy as np 
# --- emanu --- 
from emanu import util as UT
from emanu import obvs as Obvs
# -- plotting -- 
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

kf = 2.*np.pi/1000.
# read in P0(k)s 
quij = Obvs.quijoteP0k('fiducial') 
p0k_fid = np.average(quij['p0k'], axis=0) 
quij = Obvs.quijoteP0k('Mmin_m') 
p0k_Mmin_m = np.average(quij['p0k'], axis=0) 
quij = Obvs.quijoteP0k('Mmin_p') 
p0k_Mmin_p = np.average(quij['p0k'], axis=0) 

# compare P0(k)s 
fig = plt.figure(figsize=(6,8)) 
sub = fig.add_subplot(211) 
sub.plot(quij['k'], quij['k'] * p0k_Mmin_m, c='C0', lw=1, label=r'$M_{\rm min}^{-}$') 
sub.plot(quij['k'], quij['k'] * p0k_fid, c='k', lw=1, label=r'fiducial') 
sub.plot(quij['k'], quij['k'] * p0k_Mmin_p, c='C1', lw=1, label=r'$M_{\rm min}^{+}$') 
sub.legend(loc='lower left', fontsize=20) 
sub.set_xscale('log') 
sub.set_xlim(6e-3, 1) 
sub.set_ylabel('$k P_0(k)$', fontsize=25) 
sub.set_yscale('log') 
sub.set_ylim(1e2, 3e3) 
sub = fig.add_subplot(212) 
sub.plot(quij['k'], p0k_Mmin_m/p0k_fid, c='C0', lw=1) 
sub.plot(quij['k'], p0k_Mmin_p/p0k_fid, c='C1', lw=1) 
sub.plot([1e-3, 1], [1., 1.], c='k', ls='--') 
sub.set_xlabel('$k$', fontsize=25) 
sub.set_xscale('log') 
sub.set_xlim(6e-3, 1) 
sub.set_ylabel(r'$P_0/P^{\rm fid}_0$', fontsize=25) 
sub.set_ylim(0.9, 1.1) 
fig.savefig(os.path.join(UT.fig_dir(), '_p0k.Mmin.png'), bbox_inches='tight')

# dP / dMmin 
dpkdmin_mp = (p0k_Mmin_p - p0k_Mmin_m)/0.2
dpkdmin_p = (p0k_Mmin_p - p0k_fid)/0.1

# compare dP0/dMin 
fig = plt.figure() 
sub = fig.add_subplot(111) 
sub.plot(quij['k'], dpkdmin_mp, c='k', lw=1, label='-,+')
sub.plot(quij['k'], dpkdmin_p, c='C0', lw=1, label='fid.,+') 
sub.legend(loc='lower left', fontsize=20) 
sub.set_xlabel('$k$', fontsize=25) 
sub.set_xscale('log') 
sub.set_xlim(6e-3, 1) 
sub.set_ylabel(r'$dP_0/dM_{\rm min}$', fontsize=25) 
sub.set_yscale('log') 
sub.set_ylim(1e1, 3e4) 
fig.savefig(os.path.join(UT.fig_dir(), '_dp0kdMmin.png'), bbox_inches='tight')

# read in B(k)s 
quij = Obvs.quijoteBk('fiducial') 
bk_fid = np.average(quij['b123'], axis=0) 
quij = Obvs.quijoteBk('Mmin_m') 
bk_Mmin_m = np.average(quij['b123'], axis=0) 
quij = Obvs.quijoteBk('Mmin_p') 
bk_Mmin_p = np.average(quij['b123'], axis=0) 
quij = Obvs.quijoteBk('Mmin_pp') 
bk_Mmin_pp = np.average(quij['b123'], axis=0) 
    
i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3']
kmax = 0.5 
klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 
ijl = UT.ijl_order(i_k[klim], j_k[klim], l_k[klim], typ='GM') # order of triangles 

# compare B(k)s 
fig = plt.figure(figsize=(12,8)) 
sub = fig.add_subplot(211) 
sub.plot(range(np.sum(klim)), bk_Mmin_m[klim][ijl], c='C0', lw=1, label=r'$M_{\rm min}^{-}$') 
sub.plot(range(np.sum(klim)), bk_fid[klim][ijl], c='k', lw=1, label=r'fiducial') 
sub.plot(range(np.sum(klim)), bk_Mmin_p[klim][ijl], c='C1', lw=1, label=r'$M_{\rm min}^{+}$') 
sub.plot(range(np.sum(klim)), bk_Mmin_pp[klim][ijl], c='C2', lw=1, label=r'$M_{\rm min}^{++}$') 
sub.legend(loc='upper right', fontsize=20) 
sub.set_xlim(0, np.sum(klim)) 
sub.set_ylabel('$B(k)$', fontsize=25) 
sub.set_yscale('log') 

sub = fig.add_subplot(212) 
sub.plot(range(np.sum(klim)), (bk_Mmin_m/bk_fid)[klim][ijl], c='C0', lw=1) 
sub.plot(range(np.sum(klim)), (bk_Mmin_p/bk_fid)[klim][ijl], c='C1', lw=1) 
sub.plot(range(np.sum(klim)), (bk_Mmin_pp/bk_fid)[klim][ijl], c='C2', lw=1) 
sub.plot([0, np.sum(klim)], [1., 1.], c='k', ls='--') 
sub.set_xlabel('triangles', fontsize=25) 
sub.set_xlim(0, np.sum(klim)) 
sub.set_ylabel(r'$B/B^{\rm fid}_0$', fontsize=25) 
sub.set_ylim(0.9, 1.1) 
fig.savefig(os.path.join(UT.fig_dir(), '_bk.Mmin.png'), bbox_inches='tight')

# dB / dMmin 
dbkdmin_mp = (bk_Mmin_p - bk_Mmin_m)/0.2
dbkdmin_p = (bk_Mmin_p - bk_fid)/0.1
dbkdmin_pp = (bk_Mmin_pp - bk_fid)/0.2

# compare dB/dMin 
fig = plt.figure(figsize=(12,8)) 
sub = fig.add_subplot(211) 
sub.plot(range(np.sum(klim)), dbkdmin_mp[klim][ijl], c='k', ls='-', lw=1, label='-,+')
sub.plot(range(np.sum(klim)), dbkdmin_p[klim][ijl], c='C0', ls='--', lw=1, label='fid.,+') 
sub.plot(range(np.sum(klim)), dbkdmin_pp[klim][ijl], c='C1', ls='-.', lw=1, label='fid.,++') 
sub.legend(loc='upper right', fontsize=20) 
sub.set_xlim(0, np.sum(klim)) 
sub.set_ylabel(r'$dB/dM_{\rm min}$', fontsize=25) 
sub.set_yscale('log') 
sub.set_ylim(1e5, 5e9) 
sub = fig.add_subplot(212) 
sub.plot(range(np.sum(klim)), (dbkdmin_p/dbkdmin_mp)[klim][ijl], c='C0', lw=1, label='(fid.,+)/(-,+)') 
sub.plot(range(np.sum(klim)), (dbkdmin_pp/dbkdmin_mp)[klim][ijl], c='C1', lw=1, label='(fid.,++)/(-,+)') 
sub.plot([0., np.sum(klim)], [1., 1.], c='k', ls='--', lw=1) 
sub.legend(loc='lower left', frameon=True, fontsize=20) 
sub.set_xlabel('triangles', fontsize=25) 
sub.set_xlim(0, np.sum(klim)) 
sub.set_ylabel(r'deriv. ratio', fontsize=25) 
sub.set_ylim(0.4, 1.6) 
fig.savefig(os.path.join(UT.fig_dir(), '_dbkdMmin.png'), bbox_inches='tight')
