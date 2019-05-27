
import os 
import h5py
import numpy as np 
from emanu import util as UT
from emanu import forecast as Forecast
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

# covariance matrix 
_bks = fold_fid['b123'][...] + fold_fid['b_sn'][...]
C_old = np.cov(_bks.T) 
_bks = fnew_fid['b123'][...] + fnew_fid['b_sn'][...]
C_new = np.cov(_bks.T) 
fcov = h5py.File('/Users/ChangHoon/data/emanu/bispectrum/quijote_bCov_full.rsd2.reg.hdf5', 'r') 
C_for = fcov['C_bk'][...]
nmock = fold_fid['b123'][...].shape[0] 

k1, k2, k3 = fold_fid['k1'][...], fold_fid['k2'][...], fold_fid['k3'][...] 
klim = ((k1 * kf <= 0.5) & (k2 * kf <= 0.5) & (k3 * kf <= 0.5)) 
ndata = np.sum(klim) 

# calculate precision matrix 
f_hartlap = 1. #float(nmock - ndata - 2)/float(nmock - 1) 
# now lets check the forecasts
Ci_old = f_hartlap * np.linalg.inv(C_old[:,klim][klim,:]) 
Ci_new = f_hartlap * np.linalg.inv(C_new[:,klim][klim,:]) 
Ci_for = f_hartlap * np.linalg.inv(C_for[:,klim][klim,:]) 

fig = plt.figure(figsize=(10,5)) 
sub = fig.add_subplot(111) 
sub.plot(range(np.sum(klim)), np.average(fnew_fid['b123'][...], axis=0)[klim]/b123_fid[klim], 
        c='k', ls='--', lw=0.5) 
sub.set_xlabel('triangle configurations', fontsize=20) 
sub.set_xlim(0, np.sum(klim))
fig.savefig(os.path.join(UT.fig_dir(), '_check_quijote.fiducial.png'), bbox_inches='tight')  
plt.close() 

quijote_thetas = {
        'Mnu': [0.1, 0.2, 0.4], # +, ++, +++ 
        'Ob': [0.048, 0.050],   # others are - + 
        'Ob2': [0.047, 0.051],   
        'Om': [0.3075, 0.3275],
        'h': [0.6511, 0.6911],
        'ns': [0.9424, 0.9824],
        's8': [0.819, 0.849]}
thetas = ['Om', 'Ob2', 'h', 'ns', 's8'] 
theta_lbls = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$']

dbks_old = [] 
dbks_new = [] 
dbks_for = [] 
for theta in thetas: 
    if theta != 'Ob2':
        fold_m = h5py.File(os.path.join(old_dir, 'quijote_%s_m.ncv.hdf5' % theta), 'r') 
        fold_p = h5py.File(os.path.join(old_dir, 'quijote_%s_p.ncv.hdf5' % theta), 'r') 
        h_old = (quijote_thetas[theta][1] - quijote_thetas[theta][0])
    else: 
        fold_m = h5py.File(os.path.join(old_dir, 'quijote_Ob_m.hdf5'), 'r') 
        fold_p = h5py.File(os.path.join(old_dir, 'quijote_Ob_p.hdf5'), 'r') 
        h_old = 0.002 

    fnew_m = h5py.File(os.path.join(new_dir, 'quijote_%s_m.ncv.rsd2.hdf5' % theta), 'r') 
    fnew_p = h5py.File(os.path.join(new_dir, 'quijote_%s_p.ncv.rsd2.hdf5' % theta), 'r') 
    h_new = (quijote_thetas[theta][1] - quijote_thetas[theta][0])

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

    _, _, _, dbk = Forecast.quijote_dBkdtheta(theta, log=False, flag='ncv', rsd=2)
    dbks_old.append(((bold_p - bold_m)/h_old)[klim]) 
    dbks_new.append(((bnew_p - bnew_m)/h_new)[klim]) 
    dbks_for.append(dbk[klim]) 
    dold = (bold_p - bold_m)/h_old/bold_fid
    dnew = (bnew_p - bnew_m)/h_new/bnew_fid
    dbk /= bnew_fid

    print (dnew/dold)[klim]
    print (dbk/dold)[klim]
    fig = plt.figure(figsize=(10,5)) 
    sub = fig.add_subplot(111) 
    sub.plot(range(np.sum(klim)), dold[klim], c='C1')
    sub.plot(range(np.sum(klim)), dbk[klim], c='C0', ls=':') 
    sub.plot(range(np.sum(klim)), dnew[klim], c='k', ls='--', lw=0.5) 
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

dold = (-21. * bold_fid + 32. * bold_p - 12. * bold_pp + bold_ppp)/1.2
dnew = (-21. * bnew_fid + 32. * bnew_p - 12. * bnew_pp + bnew_ppp)/1.2
_, _, _, dbk = Forecast.quijote_dBkdtheta('Mnu', log=False, flag='ncv', rsd=2)
dbks_old.append(dold[klim]) 
dbks_new.append(dnew[klim]) 
dbks_for.append(dbk[klim]) 

dold /=bold_fid
dnew /=bnew_fid
dbk /= bnew_fid
print (dnew/dold)[klim]
print (dbk/dold)[klim]

fig = plt.figure(figsize=(10,5)) 
sub = fig.add_subplot(111) 
sub.plot(range(np.sum(klim)), dold[klim], c='C1') 
sub.plot(range(np.sum(klim)), dbk[klim], c='C0', ls=':') 
sub.plot(range(np.sum(klim)), dnew[klim], c='k', ls='--', lw=0.5) 
sub.set_xlabel('triangle configurations', fontsize=20) 
sub.set_xlim(0, np.sum(klim))
fig.savefig(os.path.join(UT.fig_dir(), '_check_quijote.dMnu.png'), bbox_inches='tight')  
plt.close() 

Fij_old = Forecast.Fij(dbks_old, Ci_old) 
Fij_new = Forecast.Fij(dbks_new, Ci_new) 
Fij_for = Forecast.Fij(dbks_for, Ci_for) 

print Fij_new/Fij_old
print Fij_for/Fij_old

ij_pairs = [] 
ij_pairs_str = [] 

_thetas = ['Om', 'Ob2', 'h', 'ns', 's8', 'Mnu'] 
for i in xrange(len(_thetas)): 
    for j in xrange(i, len(_thetas)): 
        ij_pairs.append((i,j))
        ij_pairs_str.append(','.join([theta_lbls[i], theta_lbls[j]]))
ij_pairs = np.array(ij_pairs) 

Nderivs = [100, 250, 400, 500]
Fijs_old, Fijs_new, Fijs_for = [], [], [] 
for Nderiv in Nderivs: 
    dbks_old, dbks_new, dbks_for = [], [], []
    for theta in thetas: 
        if theta != 'Ob2':
            fold_m = h5py.File(os.path.join(old_dir, 'quijote_%s_m.ncv.hdf5' % theta), 'r') 
            fold_p = h5py.File(os.path.join(old_dir, 'quijote_%s_p.ncv.hdf5' % theta), 'r') 
            h_old = (quijote_thetas[theta][1] - quijote_thetas[theta][0])
        else: 
            fold_m = h5py.File(os.path.join(old_dir, 'quijote_Ob_m.hdf5'), 'r') 
            fold_p = h5py.File(os.path.join(old_dir, 'quijote_Ob_p.hdf5'), 'r') 
            h_old = 0.002 

        fnew_m = h5py.File(os.path.join(new_dir, 'quijote_%s_m.ncv.rsd2.hdf5' % theta), 'r') 
        fnew_p = h5py.File(os.path.join(new_dir, 'quijote_%s_p.ncv.rsd2.hdf5' % theta), 'r') 
        h_new = (quijote_thetas[theta][1] - quijote_thetas[theta][0])
    
        irand = np.random.choice(np.arange(500), Nderiv, replace=False) 
        isort = np.argsort(fold_m['b123'][...][:,0])
        bold_m = np.average(fold_m['b123'][...][isort][irand], axis=0)
        isort = np.argsort(fold_p['b123'][...][:,0])
        bold_p = np.average(fold_p['b123'][...][isort][irand], axis=0)
        isort = np.argsort(fnew_m['b123'][...][:,0])
        bnew_m = np.average(fnew_m['b123'][...][isort][irand], axis=0)
        isort = np.argsort(fnew_p['b123'][...][:,0])
        bnew_p = np.average(fnew_p['b123'][...][isort][irand], axis=0)

        _, _, _, dbk = Forecast.quijote_dBkdtheta(theta, log=False, flag='ncv', rsd=2, Nderiv=Nderiv)
        dbk_old = (bold_p - bold_m)/h_old
        dbk_new = (bnew_p - bnew_m)/h_new
        dbks_old.append(dbk_old[klim]) 
        dbks_new.append(dbk_new[klim]) 
        dbks_for.append(dbk[klim]) 
        print dbk_old
        print dbk_new        
        print dbk

    fold_p = h5py.File(os.path.join(old_dir, 'quijote_Mnu_p.ncv.hdf5'), 'r') 
    fold_pp = h5py.File(os.path.join(old_dir, 'quijote_Mnu_pp.hdf5'), 'r') 
    fold_ppp = h5py.File(os.path.join(old_dir, 'quijote_Mnu_ppp.hdf5'), 'r') 
    fnew_p = h5py.File(os.path.join(new_dir, 'quijote_Mnu_p.ncv.rsd2.hdf5'), 'r') 
    fnew_pp = h5py.File(os.path.join(new_dir, 'quijote_Mnu_pp.ncv.rsd2.hdf5'), 'r') 
    fnew_ppp = h5py.File(os.path.join(new_dir, 'quijote_Mnu_ppp.ncv.rsd2.hdf5'), 'r') 

    irand = np.random.choice(np.arange(500), Nderiv, replace=False) 
    isort = np.argsort(fold_p['b123'][...][:,0])
    bold_p      = np.average(fold_p['b123'][...][isort][irand], axis=0)
    isort = np.argsort(fold_pp['b123'][...][:,0])
    bold_pp     = np.average(fold_pp['b123'][...][isort][irand], axis=0)
    isort = np.argsort(fold_ppp['b123'][...][:,0])
    bold_ppp    = np.average(fold_ppp['b123'][...][isort][irand], axis=0)
    isort = np.argsort(fnew_p['b123'][...][:,0])
    bnew_p      = np.average(fnew_p['b123'][...][isort][irand], axis=0)
    isort = np.argsort(fnew_pp['b123'][...][:,0])
    bnew_pp     = np.average(fnew_pp['b123'][...][isort][irand], axis=0)
    isort = np.argsort(fnew_ppp['b123'][...][:,0])
    bnew_ppp    = np.average(fnew_ppp['b123'][...][isort][irand], axis=0)

    dold = (-21. * bold_fid + 32. * bold_p - 12. * bold_pp + bold_ppp)/1.2
    dnew = (-21. * bnew_fid + 32. * bnew_p - 12. * bnew_pp + bnew_ppp)/1.2
    _, _, _, dbk = Forecast.quijote_dBkdtheta('Mnu', log=False, flag='ncv', rsd=2, Nderiv=Nderiv)
    dbks_old.append(dold[klim]) 
    dbks_new.append(dnew[klim]) 
    dbks_for.append(dbk[klim]) 
    print dbk_old
    print dbk_new        
    print dbk

    Fijs_old.append(Forecast.Fij(dbks_old, Ci_old))
    Fijs_new.append(Forecast.Fij(dbks_new, Ci_new))
    Fijs_for.append(Forecast.Fij(dbks_for, Ci_for))

Fijs_old = np.array(Fijs_old)
Fijs_new = np.array(Fijs_new)
Fijs_for = np.array(Fijs_for)
print Fijs_old.shape
print Fijs_new.shape
print Fijs_for.shape

fig = plt.figure(figsize=(6,15))
sub = fig.add_subplot(111)
for _i, ij in enumerate(ij_pairs): 
    sub.fill_between([100, 500], [1.-_i*0.3-0.05, 1.-_i*0.3-0.05], [1.-_i*0.3+0.05, 1.-_i*0.3+0.05],
            color='k', linewidth=0, alpha=0.25) 
    sub.plot([100., 500.], [1.-_i*0.3, 1.-_i*0.3], c='k', ls='--', lw=1) 
    #_Fij_old = np.array([Fijs_old[ik][ij[0],ij[1]] for ik in range(len(Nderivs))]) 
    #_Fij_new = np.array([Fijs_new[ik][ij[0],ij[1]] for ik in range(len(Nderivs))]) 
    #_Fij_for = np.array([Fijs_for[ik][ij[0],ij[1]] for ik in range(len(Nderivs))]) 
    _Fij_old = Fijs_old[:,ij[0],ij[1]]
    _Fij_new = Fijs_new[:,ij[0],ij[1]]
    _Fij_for = Fijs_for[:,ij[0],ij[1]]
    sub.plot(Nderivs, _Fij_new/_Fij_new[-1] - _i*0.3, c='C1', lw=0.5)
    sub.plot(Nderivs, _Fij_for/_Fij_for[-1] - _i*0.3, c='C0', lw=0.5)
    sub.plot(Nderivs, _Fij_old/_Fij_old[-1] - _i*0.3, c='k', lw=0.5)
sub.set_xlim(100, 500)
sub.set_ylim([1.-0.3*len(ij_pairs), 1.3]) 
sub.set_yticks([1. - 0.3 * ii for ii in range(len(ij_pairs))])
sub.set_yticklabels(ij_pairs_str) 
fig.savefig(os.path.join(UT.fig_dir(), '_check_quijote.Fij_converge.png'), bbox_inches='tight')  
