
import os 
import h5py
import numpy as np 
from emanu import util as UT
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


kf = np.pi/500.

old_dir = '/Volumes/chang_eHDD/projects/emanu/bispectrum/archive/'
new_dir = '/Users/ChangHoon/data/emanu/bispectrum/quijote/z0/'

fold_fid = h5py.File(os.path.join(old_dir, 'quijote_fiducial.hdf5'), 'r') 
b123_fid = np.average(fold_fid['b123'][...], axis=0) 

fnew_fid = h5py.File(os.path.join(new_dir, 'quijote_fiducial.reg.rsd2.hdf5'), 'r') 
bold_fid = b123_fid 
bnew_fid = np.average(fnew_fid['b123'][...], axis=0) 

k1, k2, k3 = fold_fid['k1'][...], fold_fid['k2'][...], fold_fid['k3'][...] 
klim = ((k1 * kf <= 0.5) & (k2 * kf <= 0.5) & (k3 * kf <= 0.5)) 

fig = plt.figure(figsize=(10,5)) 
sub = fig.add_subplot(111) 
sub.plot(range(np.sum(klim)), np.average(fnew_fid['b123'][...], axis=0)[klim]/b123_fid[klim], 
        c='k', ls='--', lw=0.5) 
sub.set_xlabel('triangle configurations', fontsize=20) 
sub.set_xlim(0, np.sum(klim))
fig.savefig(os.path.join(UT.fig_dir(), '_check_quijote.fiducial.png'), bbox_inches='tight')  
plt.close() 

#thetas = ['Om', 'Ob2', 'h', 'ns', 's8']
quijote_thetas = {
        'Mnu': [0.1, 0.2, 0.4], # +, ++, +++ 
        'Ob': [0.048, 0.050],   # others are - + 
        'Ob2': [0.047, 0.051],   
        'Om': [0.3075, 0.3275],
        'h': [0.6511, 0.6911],
        'ns': [0.9424, 0.9824],
        's8': [0.819, 0.849]}
thetas = ['Om', 'h', 'ns', 's8'] 
for theta in thetas: 
    fold_m = h5py.File(os.path.join(old_dir, 'quijote_%s_m.ncv.hdf5' % theta), 'r') 
    fold_p = h5py.File(os.path.join(old_dir, 'quijote_%s_p.ncv.hdf5' % theta), 'r') 

    fnew_m = h5py.File(os.path.join(new_dir, 'quijote_%s_m.ncv.rsd2.hdf5' % theta), 'r') 
    fnew_p = h5py.File(os.path.join(new_dir, 'quijote_%s_p.ncv.rsd2.hdf5' % theta), 'r') 

    k1, k2, k3 = fold_m['k1'][...], fold_m['k2'][...], fold_m['k3'][...] 

    klim = ((k1 * kf <= 0.5) & (k2 * kf <= 0.5) & (k3 * kf <= 0.5)) 

    bold_m = np.average(fold_m['b123'][...], axis=0)
    bold_p = np.average(fold_p['b123'][...], axis=0)
    bnew_m = np.average(fnew_m['b123'][...], axis=0)
    bnew_p = np.average(fnew_p['b123'][...], axis=0)

    fig = plt.figure(figsize=(10,5)) 
    sub = fig.add_subplot(111) 
    sub.plot(range(np.sum(klim)), bold_m[klim]/b123_fid[klim], c='C1') 
    sub.plot(range(np.sum(klim)), bold_p[klim]/b123_fid[klim], c='C0') 
    sub.plot(range(np.sum(klim)), bnew_m[klim]/b123_fid[klim], c='k', ls='--', lw=0.5) 
    sub.plot(range(np.sum(klim)), bnew_p[klim]/b123_fid[klim], c='k', ls='--', lw=0.5) 
    sub.set_xlabel('triangle configurations', fontsize=20) 
    sub.set_xlim(0, np.sum(klim))
    fig.savefig(os.path.join(UT.fig_dir(), '_check_quijote.%s.png' % theta), bbox_inches='tight')  
    plt.close() 

    h = (quijote_thetas[theta][1] - quijote_thetas[theta][0])
    fig = plt.figure(figsize=(10,5)) 
    sub = fig.add_subplot(111) 
    sub.plot(range(np.sum(klim)), (bold_p - bold_m)[klim]/h/bold_fid[klim], c='C1')
    sub.plot(range(np.sum(klim)), (bnew_p - bnew_m)[klim]/h/bnew_fid[klim], c='k', ls='--', lw=0.5) 
    sub.set_xlabel('triangle configurations', fontsize=20) 
    sub.set_xlim(0, np.sum(klim))
    fig.savefig(os.path.join(UT.fig_dir(), '_check_quijote.d%s.png' % theta), bbox_inches='tight')  
    plt.close() 

fold_p = h5py.File(os.path.join(old_dir, 'quijote_Mnu_p.ncv.hdf5'), 'r') 
fold_pp = h5py.File(os.path.join(old_dir, 'quijote_Mnu_pp.hdf5'), 'r') 
fold_ppp = h5py.File(os.path.join(old_dir, 'quijote_Mnu_ppp.hdf5'), 'r') 
fnew_p = h5py.File(os.path.join(new_dir, 'quijote_Mnu_p.ncv.rsd2.hdf5'), 'r') 
fnew_pp = h5py.File(os.path.join(new_dir, 'quijote_Mnu_pp.ncv.rsd2.hdf5'), 'r') 
fnew_ppp = h5py.File(os.path.join(new_dir, 'quijote_Mnu_ppp.ncv.rsd2.hdf5'), 'r') 

bold_p = np.average(fold_p['b123'][...], axis=0)
bold_pp = np.average(fold_pp['b123'][...], axis=0)
bold_ppp = np.average(fold_ppp['b123'][...], axis=0)
bnew_p = np.average(fnew_p['b123'][...], axis=0)
bnew_pp = np.average(fnew_pp['b123'][...], axis=0)
bnew_ppp = np.average(fnew_ppp['b123'][...], axis=0)

fig = plt.figure(figsize=(10,5)) 
sub = fig.add_subplot(111) 
sub.plot(range(np.sum(klim)), np.average(fold_p['b123'][...], axis=0)[klim]/b123_fid[klim], c='C0') 
sub.plot(range(np.sum(klim)), np.average(fnew_p['b123'][...], axis=0)[klim]/b123_fid[klim], c='k', 
        ls='--', lw=0.5) 
sub.plot(range(np.sum(klim)), np.average(fold_pp['b123'][...], axis=0)[klim]/b123_fid[klim], c='C1') 
sub.plot(range(np.sum(klim)), np.average(fnew_pp['b123'][...], axis=0)[klim]/b123_fid[klim], c='k', 
        ls='--', lw=0.5) 
sub.plot(range(np.sum(klim)), np.average(fold_ppp['b123'][...], axis=0)[klim]/b123_fid[klim], c='C2') 
sub.plot(range(np.sum(klim)), np.average(fnew_ppp['b123'][...], axis=0)[klim]/b123_fid[klim], c='k', 
        ls='--', lw=0.5) 
sub.set_xlabel('triangle configurations', fontsize=20) 
sub.set_xlim(0, np.sum(klim))
fig.savefig(os.path.join(UT.fig_dir(), '_check_quijote.Mnu.png'), bbox_inches='tight')  
plt.close() 
        
