''' quick script to check that the fine resolution P(k) 
calculations that are in agreement with previous P(k) calculated
with the bispectrum. 
'''
import os
import h5py
import numpy as np 
# --- emanu --- 
from emanu import util as UT
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


dir_quij = os.path.join(UT.dat_dir(), 'bispectrum') 
# read in k_fund gridded P(k) 
f_kfund = h5py.File(os.path.join(dir_quij, 'quijote_Pk_fiducial.hdf5'), 'r') 
k, p0k_kfund = f_kfund['k'].value, f_kfund['p0k'].value 

# read in P(k) calculated with bispectrum code 
f_bisp = h5py.File(os.path.join(dir_quij, 'quijote_fiducial.hdf5'), 'r') 
i_k, p0k_bisp = f_bisp['k1'].value, f_bisp['p0k1'].value 

kfund = 2.*np.pi/1000.
_, _iuniq = np.unique(i_k, return_index=True)
iuniq = np.zeros(len(i_k)).astype(bool) 
iuniq[_iuniq] = True

fig = plt.figure()
sub = fig.add_subplot(111)
for i in range(10): 
    sub.plot(k, p0k_kfund[i,:] * 0.8**i, c='C0', lw=1) 
    sub.plot(kfund * i_k[iuniq], p0k_bisp[i,iuniq] * 0.8**i, c='k', ls='--', lw=1) 
sub.set_xlabel('$k$', fontsize=25) 
sub.set_xscale('log') 
sub.set_xlim(kfund, 1.) 
sub.set_ylabel('$\propto P_0(k)$', fontsize=25) 
sub.set_yscale('log') 
sub.set_ylim(1e2, 1.5e5) 
fig.savefig(os.path.join(UT.fig_dir(), 'pk_kfund_check.png'), bbox_inches='tight') 
