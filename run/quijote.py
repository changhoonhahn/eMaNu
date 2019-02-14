'''

Some scripts for dealing with the quijote simulation bispectrum
measurements. 


'''
import os 
import glob
import h5py 
import numpy as np 
# -- emanu --
from emanu import util as UT
# -- plotting -- 
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


def quijote_covariance(krange=[0.01, 0.5]):
    ''' calculate the covariance matrix using the 15,000 fiducial 
    bispetra.
    '''
    kmin, kmax = krange  
    fcov = os.path.join(UT.dat_dir(), 'bispectrum', 'quijote_Cov_bk.%.2f_%.2f.gmorder.hdf5' % (kmin, kmax))

    if os.path.isfile(fcov): 
        f = h5py.File(fcov, 'r') 
        C_bk = f['C_bk'].value # fiducial covariance matrix 
    else: 
        # read in fiducial bispectra
        bk = quijote_bk('fiducial', krange=krange)
        C_bk = np.cov(bk.T) 

        # write C_bk to hdf5 file 
        f = h5py.File(fcov, 'w') 
        f.create_dataset('C_bk', data=C_bk) 
        f.close()
    return C_bk  


def quijote_cov_perturb(theta, krange=[0.01, 0.5]):
    ''' covariance matrix at some perturbed theta

    :param theta: 
        string that specifies which parameter
    '''
    kmin, kmax = krange  
    fcov = os.path.join(UT.dat_dir(), 'bispectrum', 'quijote_Cov_bk.%s.%.2f_%.2f.gmorder.hdf5' % (theta, kmin, kmax))
    
    if os.path.isfile(fcov): 
        f = h5py.File(fcov, 'r') 
        C_bk = f['C_bk'].value 
    else:
        # read in fiducial bispectra
        bk = quijote_bk(theta, krange=krange)
        C_bk = np.cov(bk.T) 

        # write C_bk to hdf5 file 
        f = h5py.File(fcov, 'w') 
        f.create_dataset('C_bk', data=C_bk) 
        f.close()
    return C_bk 


def quijote_cov_perturb_comparison(theta, krange=[0.01, 0.5]): 
    ''' Compare fiducial covariance matrix to perturbed theta
    covariance matrices 
    '''
    # fiducial covariance matrix 
    C_fid = quijote_covariance(krange=krange) 

    if theta == 'Mnu':
        perturbs = ['p', 'pp', 'ppp'] 
        pert_lbls = ['+', '++', '+++'] 
    else:
        perturbs = ['m', 'p'] 
        pert_lbls = ['-', '+'] 

    Cs = [] 
    for perturb in perturbs: # read in covariance matrices
        Cs.append(quijote_cov_perturb(theta+'_'+perturb, krange=krange)) 

    fig = plt.figure(figsize=(20,10))
    gs = mpl.gridspec.GridSpec(2, 4, figure=fig) 
    sub = plt.subplot(gs[0, 0]) 
    cm = sub.pcolormesh(C_fid, norm=LogNorm(vmin=1e11, vmax=1e18))
    sub.set_title(r'$C^{\rm (fid)}_{B(k_1, k_2, k_3)}$', fontsize=25)
    for ii, pert, C in zip(range(len(perturbs)), pert_lbls, Cs): 
        sub = plt.subplot(gs[0, ii+1]) 
        cm = sub.pcolormesh(C, norm=LogNorm(vmin=1e11, vmax=1e18))
        sub.set_title(r'$C^{(%s)}_{B(k_1, k_2, k_3)}$' % pert, fontsize=25)
        sub.set_yticklabels([]) 
    
    sub = plt.subplot(gs[1,:]) 
    for ii, pert, C in zip(range(len(perturbs)), pert_lbls, Cs): 
        sub.plot(range(C.shape[0]), np.diag(C), lw=4-ii, label=r'$C^{(%s)}_{i,i}$' % pert)
    sub.plot(range(C_fid.shape[0]), np.diag(C_fid), c='k', lw=1, ls='--', label=r'$C^{\rm (fid)}_{i,i}$')
    sub.set_xlabel(r'$k_1 \le k_2 \le k_3$ triangle indices', fontsize=25) 
    sub.set_xlim([0., 1000.]) 
    sub.set_yscale('log') 
    sub.set_ylim([1e13, 5e19]) 
    sub.legend(loc='upper right', fontsize=20) 
    fig.subplots_adjust(wspace=0.1, right=0.925)
    cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
    fig.colorbar(cm, cax=cbar_ax)
    fig.savefig(os.path.join(UT.dat_dir(), 'bispectrum', 'quijote_Cov_bk_d%s_comparison.png' % theta), 
        bbox_inches='tight') 

##################################################################
# qujiote fisher stuff
##################################################################
def quijote_dCov_dtheta(theta, krange=[0.01, 0.5], validate=False): 
    ''' return derivative of the covariance matrix along theta. 

    :param theta: 
        string that specifies which parameter 
    ''' 
    kmin, kmax = krange  
    # fiducial theta and covariance matrix 
    fid_dict = {'Mnu': 0., 'Ob': 0.049, 'Om': 0.3175, 'h': 0.6711,  'ns': 0.9624,  's8': 0.834}
    tt_fid = fid_dict[theta] 
    C_fid = quijote_covariance(krange=krange) 

    # get all bispectrum covariances along theta  
    theta_dict = {
            'Mnu': [0.1, 0.2, 0.4], # +, ++, +++ 
            'Ob': [0.048, 0.050], # others are - + 
            'Om': [0.3075, 0.3275],
            'h': [0.6511, 0.6911],
            'ns': [0.9424, 0.9824],
            's8': [0.819, 0.849]} 

    if theta == 'Mnu': 
        tt_p, tt_pp, tt_ppp = theta_dict['Mnu'] 
        h_p = tt_p - tt_fid # step + 
        h_pp = tt_pp - tt_fid # step ++ 
        h_ppp = tt_ppp - tt_fid # step +++ 
        
        C_p = quijote_cov_perturb(theta+'_p', krange=krange) # Covariance matrix tt+ 
        C_pp = quijote_cov_perturb(theta+'_pp', krange=krange) # Covariance matrix tt++
        C_ppp = quijote_cov_perturb(theta+'_ppp', krange=krange) # Covariance matrix tt+++

        # take the derivatives 
        dC_p = (C_p - C_fid) / h_p 
        dC_pp = (C_pp - C_fid) / h_pp
        dC_ppp = (C_ppp - C_fid) / h_ppp

        for mpc, dC in zip(['p', 'pp', 'ppp'], [dC_p, dC_pp, dC_ppp]): 
            fdcov = os.path.join(UT.dat_dir(), 'bispectrum', 'quijote_dCov_bk_d%s.%s.%.2f_%.2f.gmorder.hdf5' % (theta, mpc, kmin, kmax))
            f = h5py.File(fdcov, 'w') 
            f.create_dataset('dC_bk_dtheta', data=dC) 
            f.close() 

        if validate: 
            fig = plt.figure(figsize=(15,10))
            gs = mpl.gridspec.GridSpec(2,3,figure=fig) 
            for ii, mpc, dC, h in zip(range(3), ['+', '++', '+++'], [dC_p, dC_pp, dC_ppp], [h_p, h_pp, h_ppp]): 
                sub = plt.subplot(gs[0, ii]) 
                cm = sub.pcolormesh(dC * h, norm=LogNorm(vmin=1e11, vmax=1e18))
                sub.set_title(r'$\Delta C^{(%s)}_{B(k_1, k_2, k_3)}$' % mpc, fontsize=25)
                if ii != 0: sub.set_yticklabels([]) 
            
            sub = plt.subplot(gs[1,:]) 
            for ii, mpc, dC, h in zip(range(3), ['+', '++', '+++'], [dC_p, dC_pp, dC_ppp], [h_p, h_pp, h_ppp]): 
                #sub.plot(range(dC.shape[0]), np.diag(dC) * h, lw=4-ii, label=r'$\Delta C^{(%s)}_{i,i}$' % mpc)
                sub.scatter(range(dC.shape[0]), np.diag(dC) * h, s=1, label=r'$\Delta C^{(%s)}_{i,i}$' % mpc)
            sub.set_xlabel(r'$k_1 \le k_2 \le k_3$ triangle indices', fontsize=25) 
            sub.set_xlim([0., 1000.]) 
            sub.set_yscale('symlog') 
            sub.set_ylim([-1e19, 1e19]) 
            sub.legend(loc='upper right', markerscale=10, handletextpad=-0.1, fontsize=20) 
            fig.subplots_adjust(wspace=0.1, right=0.925)
            cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
            fig.colorbar(cm, cax=cbar_ax)
            fig.savefig(os.path.join(UT.dat_dir(), 'bispectrum', 'quijote_dCov_bk_d%s.png' % theta), 
                    bbox_inches='tight') 
    else: 
        tt_m, tt_p = theta_dict[theta]
        h_m = tt_fid - tt_m # step - 
        h_p = tt_p - tt_fid # step + 
        
        C_m = quijote_cov_perturb(theta+'_m', krange=krange) # Covariance matrix tt- 
        C_p = quijote_cov_perturb(theta+'_p', krange=krange) # Covariance matrix tt+ 
        
        # take the derivatives 
        dC_m = (C_fid - C_m) / h_m 
        dC_p = (C_p - C_fid) / h_p 
        dC_c = (C_p - C_m) / (h_m + h_p) 
        
        for mpc, dC in zip(['m', 'p', 'c'], [dC_m, dC_p, dC_c]): 
            fdcov = os.path.join(UT.dat_dir(), 'bispectrum', 'quijote_dCov_bk_d%s.%s.%.2f_%.2f.gmorder.hdf5' % (theta, mpc, kmin, kmax))
            f = h5py.File(fdcov, 'w') 
            f.create_dataset('dC_bk_dtheta', data=dC) 
            f.close() 

        if validate: 
            fig = plt.figure(figsize=(15,10))
            gs = mpl.gridspec.GridSpec(2,3,figure=fig) 
            for ii, mpc, dC, h in zip(range(3), ['-', '+', '-+'], [dC_m, dC_p, dC_c], [h_m, h_p, h_m+h_p]): 
                sub = plt.subplot(gs[0, ii]) 
                cm = sub.pcolormesh(dC * h, norm=LogNorm(vmin=1e11, vmax=1e18))
                sub.set_title(r'$\Delta C^{(%s)}_{B(k_1, k_2, k_3)}$' % mpc, fontsize=25)
                if ii != 0: sub.set_yticklabels([]) 
            
            sub = plt.subplot(gs[1,:]) 
            for ii, mpc, dC, ls, h in zip(range(3), ['-', '+', '-+'], [dC_m, dC_p, dC_c], ['-', '--', ':'], [h_m, h_p, h_m+h_p]): 
                #sub.plot(range(dC.shape[0]), np.diag(dC) * h, lw=4-ii, label=r'$\Delta C^{(%s)}_{i,i}$' % mpc)
                sub.scatter(range(dC.shape[0]), np.diag(dC) * h, s=1, label=r'$\Delta C^{(%s)}_{i,i}$' % mpc)
            sub.set_xlabel(r'$k_1 \le k_2 \le k_3$ triangle indices', fontsize=25) 
            sub.set_xlim([0., 1000.]) 
            sub.set_yscale('symlog') 
            sub.set_ylim([-1e19, 1e19]) 
            sub.legend(loc='upper right', markerscale=10, handletextpad=-0.1, fontsize=20) 
            fig.subplots_adjust(wspace=0.1, right=0.925)
            cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
            fig.colorbar(cm, cax=cbar_ax)
            fig.savefig(os.path.join(UT.dat_dir(), 'bispectrum', 'quijote_dCov_bk_d%s.png' % theta), 
                    bbox_inches='tight') 
    return None 


def quijote_Fisher(mpc='c', krange=[0.01, 0.5], validate=False): 
    ''' calculate fisher matrix for thetas = ['Mnu', 'Ob', 'Om', 'h', 'ns', 's8']
    '''
    kmin, kmax = krange
    thetas = ['Mnu', 'Ob', 'Om', 'h', 'ns', 's8']
    Fij = np.zeros((len(thetas), len(thetas)))
    for i, par_i in enumerate(thetas): 
        for j, par_j in enumerate(thetas): 
            fij = quijote_Fij(par_i, par_j, mpc=mpc, krange=krange)
            Fij[i,j] = fij 
    
    bk_dir = os.path.join(UT.dat_dir(), 'bispectrum')
    f_ij = os.path.join(bk_dir, 'quijote_Fisher.%s.%.2f_%.2f.gmorder.hdf5' % (mpc, kmin, kmax))
    f = h5py.File(f_ij, 'w') 
    f.create_dataset('Fij', data=Fij) 
    f.close() 
    if validate: 
        fig = plt.figure(figsize=(6,5))
        sub = fig.add_subplot(111)
        print('%.2e, %.2e' % (Fij.min(), Fij.max()))
        cm = sub.pcolormesh(Fij, norm=SymLogNorm(vmin=-5e6, vmax=5e8, linthresh=1e5, linscale=1.))

        sub.set_xticks(np.arange(Fij.shape[0]) + 0.5, minor=False)
        sub.set_xticklabels([r'$M_\nu$', r'$\Omega_b$', r'$\Omega_m$',r'$h$',r'$n_s$',r'$\sigma_8$'], minor=False)
        sub.set_yticks(np.arange(Fij.shape[1]) + 0.5, minor=False)
        sub.set_yticklabels([r'$M_\nu$', r'$\Omega_b$', r'$\Omega_m$',r'$h$',r'$n_s$',r'$\sigma_8$'], minor=False)
        sub.set_title(r'Fisher Matrix $F_{i,j}^{(%s)}$' % mpc, fontsize=25)
        fig.colorbar(cm)
        fig.savefig(f_ij.replace('.hdf5', '.png'), bbox_inches='tight') 
    return None


def quijote_Fij(theta_i, theta_j, mpc='c', krange=[0.01, 0.5]): 
    ''' calculate Fij = 1/2 * Tr[C^-1 C,i C^-1 C,j + C^-1 Mij]
    where 
    Mij = dBk/dtheta_i dBk/dtheta_j.T + dBk/dtheta_j dBk/dtheta_i.T

    :param theta_i: 
        string specificying theta_i 

    :param theta_j: 
        string specificying theta_j 

    :param mpc: (default: 'c') 
        string specifying the derivative. The options are 'm', 'c', 'p'. 
        For theta == 'Mnu', this corresponds to 'p', 'pp', 'ppp'. 
    
    :param krange: (default: [0.01, 0.5]) 
        k-range of the bispectrum measurements. Note that the
        Gil-Marin triangle order will also be implemented. 

    :param validate: (default: False) 
        If True, the code will generate a plot to validate the 
        derivatives
    '''
    kmin, kmax = krange
    bk_dir = os.path.join(UT.dat_dir(), 'bispectrum')
    fij = os.path.join(bk_dir, 'Fij', 'quijote_Fij.%s.%s.%s.%.2f_%.2f.gmorder.hdf5' % 
            (theta_i, theta_j, mpc, kmin, kmax))
    if os.path.isfile(fij): 
        f = h5py.File(fij, 'r') 
        Fij = f['Fij'].value 
    else: 
        mpc_i = mpc 
        mpc_j = mpc 
        if theta_i == 'Mnu': mpc_i = {'m': 'p', 'c': 'pp', 'p': 'ppp'}[mpc]
        if theta_j == 'Mnu': mpc_j = {'m': 'p', 'c': 'pp', 'p': 'ppp'}[mpc]

        C_fid = quijote_covariance(krange=krange) # fiducial covariance
        Cinv = np.linalg.inv(C_fid) # invert the covariance 
        
        fdCdi = os.path.join(bk_dir, 'quijote_dCov_bk_d%s.%s.%.2f_%.2f.gmorder.hdf5' % (theta_i, mpc_i, kmin, kmax))
        fdCdj = os.path.join(bk_dir, 'quijote_dCov_bk_d%s.%s.%.2f_%.2f.gmorder.hdf5' % (theta_j, mpc_j, kmin, kmax))

        fi = h5py.File(fdCdi, 'r') 
        fj = h5py.File(fdCdj, 'r') 

        dCdi = fi['dC_bk_dtheta'].value 
        dCdj = fj['dC_bk_dtheta'].value 

        fdbk_i = os.path.join(bk_dir, 'quijote_dbk_d%s.%s.%.2f_%.2f.gmorder.hdf5' % (theta_i, mpc_i, kmin, kmax))
        fdbk_j = os.path.join(bk_dir, 'quijote_dbk_d%s.%s.%.2f_%.2f.gmorder.hdf5' % (theta_j, mpc_j, kmin, kmax))

        # read in dBk/dtheta_i and dBk/dtheta_j 
        f_i = h5py.File(fdbk_i, 'r') 
        dbk_dtt_i = f_i['dbk_dtheta'].value
        f_j = h5py.File(fdbk_j, 'r') 
        dbk_dtt_j = f_j['dbk_dtheta'].value

        # calculate Mij 
        Mij = np.dot(dbk_dtt_i[:,None], dbk_dtt_j[None,:]) + np.dot(dbk_dtt_j[:,None], dbk_dtt_i[None,:])

        Fij = 0.5 * np.trace(np.dot(np.dot(Cinv, dCdi), np.dot(Cinv, dCdj)) + np.dot(Cinv, Mij))

        f = h5py.File(fij, 'w') 
        f.create_dataset('Fij', data=Fij) 
        f.create_dataset('Cinv', data=Cinv) 
        f.create_dataset('dCdi', data=dCdi) 
        f.create_dataset('dCdj', data=dCdj) 
        f.create_dataset('Mij', data=Mij) 
        f.close() 
    return Fij 


def quijote_dBk(theta, krange=[0.01, 0.5], validate=False):
    ''' calculate d B(k)/d theta. 

    :param theta: 
        string that specifies the parameter to take the 
        derivative by. 

    :param krange: (default: [0.01, 0.5]) 
        k-range of the bispectrum measurements. Note that the
        Gil-Marin triangle order will also be implemented. 

    :param validate: (default: False) 
        If True, the code will generate a plot to validate the 
        derivatives
    '''
    kmin, kmax = krange  
    # fiducial theta and covariance matrix 
    fid_dict = {'Mnu': 0., 'Ob': 0.049, 'Om': 0.3175, 'h': 0.6711,  'ns': 0.9624,  's8': 0.834}
    tt_fid = fid_dict[theta] 
    Bk_fid = np.average(quijote_bk('fiducial', krange=krange), axis=0) 

    # get all bispectrum covariances along theta  
    theta_dict = {
            'Mnu': [0.1, 0.2, 0.4], # +, ++, +++ 
            'Ob': [0.048, 0.050], # others are - + 
            'Om': [0.3075, 0.3275],
            'h': [0.6511, 0.6911],
            'ns': [0.9424, 0.9824],
            's8': [0.819, 0.849]} 

    if theta == 'Mnu': 
        tt_p, tt_pp, tt_ppp = theta_dict['Mnu'] 
        h_p = tt_p - tt_fid # step + 
        h_pp = tt_pp - tt_fid # step ++ 
        h_ppp = tt_ppp - tt_fid # step +++ 
        
        Bk_p    = np.average(quijote_bk(theta+'_p', krange=krange), axis=0) # Covariance matrix tt+ 
        Bk_pp   = np.average(quijote_bk(theta+'_pp', krange=krange), axis=0) # Covariance matrix tt++
        Bk_ppp  = np.average(quijote_bk(theta+'_ppp', krange=krange), axis=0) # Covariance matrix tt+++

        # take the derivatives 
        dBk_p = (Bk_p - Bk_fid) / h_p 
        dBk_pp = (Bk_pp - Bk_fid) / h_pp
        dBk_ppp = (Bk_ppp - Bk_fid) / h_ppp

        for mpc, dBk in zip(['p', 'pp', 'ppp'], [dBk_p, dBk_pp, dBk_ppp]): 
            fdbk = os.path.join(UT.dat_dir(), 'bispectrum', 
                    'quijote_dbk_d%s.%s.%.2f_%.2f.gmorder.hdf5' % (theta, mpc, kmin, kmax))
            f = h5py.File(fdbk, 'w') 
            f.create_dataset('dbk_dtheta', data=dBk) 
            f.close() 

        if validate: 
            fig = plt.figure(figsize=(15,5))
            sub = fig.add_subplot(111)
            for mpc, dBk in zip(['+', '++', '+++'], [dBk_p, dBk_pp, dBk_ppp]): 
                sub.plot(range(len(Bk_p)), dBk, 
                        label=r'$(B^{(%s)} - B^{\rm fid})/(M_\nu^{(%s)} - M_\nu^{\rm fid})$' % (mpc, mpc))
            sub.legend(loc='upper right', fontsize=20 ) 
            sub.set_xlabel(r'$k_1 \le k_2 \le k_3$ triangle indices', fontsize=25) 
            sub.set_xlim([0., 500.])#len(Bk_p)])  
            sub.set_yscale('log') 
            sub.set_ylim([5e6, 1e10]) 
            fig.savefig(os.path.join(UT.dat_dir(), 'bispectrum', 'quijote_dbk_d%s.png' % theta), 
                    bbox_inches='tight') 
    else: 
        tt_m, tt_p = theta_dict[theta]
        h_m = tt_fid - tt_m # step - 
        h_p = tt_p - tt_fid # step + 
        
        Bk_m = np.average(quijote_bk(theta+'_m', krange=krange), axis=0) # Covariance matrix tt- 
        Bk_p = np.average(quijote_bk(theta+'_p', krange=krange), axis=0) # Covariance matrix tt+ 
        
        # take the derivatives 
        dBk_m = (Bk_fid - Bk_m) / h_m 
        dBk_p = (Bk_p - Bk_fid) / h_p 
        dBk_c = (Bk_p - Bk_m) / (h_m + h_p) 
        
        for mpc, dBk in zip(['m', 'p', 'c'], [dBk_m, dBk_p, dBk_c]): 
            fdbk = os.path.join(UT.dat_dir(), 'bispectrum', 
                    'quijote_dbk_d%s.%s.%.2f_%.2f.gmorder.hdf5' % (theta, mpc, kmin, kmax))
            f = h5py.File(fdbk, 'w') 
            f.create_dataset('dbk_dtheta', data=dBk) 
            f.close() 

        if validate: 
            fig = plt.figure(figsize=(15,5))
            sub = fig.add_subplot(111)
            for ii, mpc, dBk in zip(range(3), ['-', '+', '-+'], [dBk_m, dBk_p, dBk_c]): 
                sub.plot(range(len(Bk_p)), dBk, 
                        label=r'${\rm d}B(k_1, k_2, k_3)^{(%s)}/{\rm d}\theta$' % mpc)
            sub.set_xlabel(r'$k_1 \le k_2 \le k_3$ triangle indices', fontsize=25) 
            sub.set_xlim([0., 500.]) #len(Bk_p)])  
            if theta in ['Ob']: 
                sub.set_yscale('log') 
                sub.set_ylim([1e7, 1e12]) 
                sub.legend(loc='upper right', fontsize=20 ) 
            elif theta in ['Om', 'h', 'ns', 's8']: 
                sub.set_yscale('symlog') 
                sub.set_ylim([-1e7, -1e12]) 
                sub.legend(loc='upper right', fontsize=20 ) 
            else: 
                sub.legend(loc='lower left', fontsize=20 ) 
            fig.savefig(os.path.join(UT.dat_dir(), 'bispectrum', 'quijote_dbk_d%s.png' % theta), 
                    bbox_inches='tight') 
    return None 


def quijote_bk(theta, krange=[0.01, 0.5]): 
    ''' read in bispectra for specified theta 
    '''
    kmin, kmax = krange 
    # read in fiducial biectra
    dir_bk = os.path.join(UT.dat_dir(), 'bispectrum') 
    bks = h5py.File(os.path.join(dir_bk, 'quijote_%s.hdf5' % theta), 'r') 
    i_k, j_k, l_k = bks['k1'].value, bks['k2'].value, bks['k3'].value 
    bk = bks['b123'].value 

    # k range limit of the bispectrum
    kf = 2.*np.pi/1000. # fundmaentla mode
    klim = ((i_k*kf <= kmax) & (i_k*kf >= kmin) &
            (j_k*kf <= kmax) & (j_k*kf >= kmin) & 
            (l_k*kf <= kmax) & (l_k*kf >= kmin)) 
    
    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 
    ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # order of triangles 
    _bk = bk[:,klim]
    return _bk[:,ijl.flatten()]


def quijote_comparison(par, krange=[0.01, 0.5]): 
    ''' Compare quijote simulations across a specified parameter `par`
    
    :param par: 
        string that specifies the parameter to compare along  

    :param krange: (default: [0.01, 0.5]) 
        k range to limit the bispectrum
    '''
    kmin, kmax = krange  
    kf = 2.*np.pi/1000. # fundmaentla mode

    # gather quijote bispectra for parameter par 
    dir_bk = os.path.join(UT.dat_dir(), 'bispectrum') 
    fbks = glob.glob(dir_bk+'/'+'quijote_%s*.hdf5' % par)
    # labels for the files 
    lbls = [fbk.rsplit('quijote_', 1)[-1].rsplit('.hdf5')[0].replace('_p', '+').replace('p', '+').replace('_m', '-') 
            for fbk in fbks]

    fig = plt.figure(figsize=(30,5))
    gs = mpl.gridspec.GridSpec(1,7, figure=fig) 
    sub0 = plt.subplot(gs[0,:2])  # pk panel
    sub1 = plt.subplot(gs[0,2:]) # bk panel
    axins = inset_axes(sub1, loc='upper right', width="40%", height="45%") 

    for i, fbk, lbl in zip(range(len(fbks)), fbks, lbls): # loop through the bispectrum
        bks = h5py.File(fbk, 'r') 
        i_k, j_k, l_k = bks['k1'].value, bks['k2'].value, bks['k3'].value 
        pk1, pk2, pk3 = bks['p0k1'].value, bks['p0k2'].value, bks['p0k3'].value
        bk = bks['b123'].value 
    
        # plot P(k) 
        klim = (i_k*kf <= kmax) & (i_k*kf >= kmin) 
        isort = np.argsort(i_k[klim]) 
        kk = kf * i_k[klim][isort]
        avgpk = np.average(pk1, axis=0)[klim][isort]
        sub0.plot(kk, avgpk, c='C'+str(i), label=lbl) 
        
        # plot B(k1,k2,k3) 
        klim = ((i_k*kf <= kmax) & (i_k*kf >= kmin) &
                (j_k*kf <= kmax) & (j_k*kf >= kmin) & 
                (l_k*kf <= kmax) & (l_k*kf >= kmin)) 
        i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 

        ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # order of triangles 
        avgbk = np.average(bk, axis=0)[klim][ijl]
        sub1.plot(range(np.sum(klim)), avgbk, c='C'+str(i)) 
        axins.plot(range(np.sum(klim)), avgbk, c='C'+str(i))

    # plot fiducial 
    bks = h5py.File(os.path.join(dir_bk, 'quijote_fiducial.hdf5'), 'r') 
    i_k, j_k, l_k = bks['k1'].value, bks['k2'].value, bks['k3'].value 
    pk1, pk2, pk3 = bks['p0k1'].value, bks['p0k2'].value, bks['p0k3'].value
    bk = bks['b123'].value 
    # plot P(k) 
    klim = (i_k*kf <= kmax) & (i_k*kf >= kmin) 
    isort = np.argsort(i_k[klim]) 
    kk = kf * i_k[klim][isort]
    avgpk = np.average(pk1, axis=0)[klim][isort]
    stdpk = np.std(pk1, axis=0)[klim][isort]
    sub0.fill_between(kk, avgpk - stdpk, avgpk + stdpk, color='k', alpha=0.25, linewidth=0, label='fiducial') 
    
    # plot B(k1,k2,k3) 
    klim = ((i_k*kf <= kmax) & (i_k*kf >= kmin) &
            (j_k*kf <= kmax) & (j_k*kf >= kmin) & 
            (l_k*kf <= kmax) & (l_k*kf >= kmin)) 
    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 
    ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # order of triangles 
    avgbk = np.average(bk, axis=0)[klim][ijl]
    stdbk = np.std(bk, axis=0)[klim][ijl]
    sub1.fill_between(np.arange(np.sum(klim)), (avgbk - stdbk).flatten(), (avgbk + stdbk).flatten(), 
            color='k', alpha=0.25, linewidth=0) 
    axins.fill_between(np.arange(np.sum(klim)), (avgbk - stdbk).flatten(), (avgbk + stdbk).flatten(), 
            color='k', alpha=0.25, linewidth=0) 

    # pk panel 
    sub0.legend(loc='lower left', ncol=1, handletextpad=0.25, columnspacing=0.5, fontsize=18) 
    sub0.set_xlabel("$k$ [$h$/Mpc]", fontsize=25) 
    sub0.set_xscale("log") 
    sub0.set_xlim([1e-2, 0.5])
    sub0.set_xticks([1e-2, 1e-1, 0.5]) 
    sub0.set_xticklabels([r'0.01', '0.1', r'0.5'])
    sub0.set_ylabel('$P_0(k)$', fontsize=25) 
    sub0.set_yscale("log") 
    sub0.set_ylim([1e3, 1e5]) 
    # bk panel
    sub1.set_yscale('log') 
    axins.set_yscale('log') 
    sub1.set_xlabel(r'$k_1 \le k_2 \le k_3$ triangle indices', labelpad=5, fontsize=25) 
    sub1.set_xlim([0, np.sum(klim)])
    sub1.set_ylabel('$B(k_1, k_2, k_3)$', labelpad=5, fontsize=25) 
    sub1.set_ylim([1e6, 1e10]) 
    axins.set_xlim(480, 500)
    axins.set_ylim(5e7, 2e8) 
    axins.set_xticklabels('') 
    axins.yaxis.set_minor_formatter(NullFormatter())
    mark_inset(sub1, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    fig.subplots_adjust(wspace=1.) 
    fig.savefig(os.path.join(UT.fig_dir(), 'quijote_%s.png' % par), bbox_inches='tight') 
    return None 


def _hdf5_quijote(subdir): 
    ''' write out quijote bispectrum transferred from cca server to hdf5 
    file for fast and easier access in the future. 

    :param subdir: 
        name of subdirectory which also describes the run.
    '''
    dir_quij = os.path.join(UT.dat_dir(), 'bispectrum', 'quijote', subdir) 
    fbks = os.listdir(dir_quij) 
    nbk = len(fbks) 
    print('%i bispectrum files in /%s' % (nbk, subdir))  
    
    # check the number of bispectrum
    if subdir == 'fiducial': assert nbk == 15000, "not the right number of files"
    else: assert nbk == 500, "not the right number of files"
    
    # load in all the files 
    for i, fbk in enumerate(fbks):
        i_k, j_k, l_k, _p0k1, _p0k2, _p0k3, b123, q123, b_sn, cnts = np.loadtxt(
                os.path.join(dir_quij, fbk), skiprows=1, unpack=True, usecols=range(10)) 
        if i == 0: 
            p0k1 = np.zeros((nbk, len(i_k)))
            p0k2 = np.zeros((nbk, len(i_k)))
            p0k3 = np.zeros((nbk, len(i_k)))
            bks = np.zeros((nbk, len(i_k)))
            qks = np.zeros((nbk, len(i_k)))
            bsn = np.zeros((nbk, len(i_k)))
        p0k1[i,:] = _p0k1
        p0k2[i,:] = _p0k2
        p0k3[i,:] = _p0k3
        bks[i,:] = b123
        qks[i,:] = q123
        bsn[i,:] = b_sn 

    # save to hdf5 file 
    f = h5py.File(os.path.join(UT.dat_dir(), 'bispectrum', 'quijote_%s.hdf5' % subdir), 'w') 
    f.create_dataset('k1', data=i_k)
    f.create_dataset('k2', data=j_k)
    f.create_dataset('k3', data=l_k)
    f.create_dataset('p0k1', data=p0k1) 
    f.create_dataset('p0k2', data=p0k2) 
    f.create_dataset('p0k3', data=p0k3) 
    f.create_dataset('b123', data=bks) 
    f.create_dataset('q123', data=qks) 
    f.create_dataset('b_sn', data=bsn) 
    f.create_dataset('counts', data=cnts) 
    f.close()
    return None 


if __name__=="__main__": 
    # write to hdf5 
    #for sub in ['Mnu_p', 'Mnu_pp', 'Mnu_ppp', 'Ob_m', 'Ob_p', 'Om_m', 'Om_p', 
    #        'fiducial', 'fiducial_NCV', 'h_m', 'h_p', 'ns_m', 'ns_p', 's8_m', 's8_p']: 
    #    _hdf5_quijote(sub)
    
    # check along parameter axis
    #for par in ['Mnu', 'Ob', 'Om', 'h', 'ns', 's8']: 
    #    quijote_comparison(par, krange=[0.01, 0.5])
    
    # covariance matrix 
    #quijote_covariance(krange=[0.01, 0.5])
    for par in ['Mnu', 'Ob', 'Om', 'h', 'ns', 's8']: 
        #quijote_cov_perturb_comparison(par, krange=[0.01, 0.5])
        quijote_dCov_dtheta(par, krange=[0.01, 0.5], validate=True)
        #quijote_dBk(par, krange=[0.01, 0.5], validate=True)
    #quijote_Fisher(mpc='m', krange=[0.01, 0.5], validate=True)   
    #quijote_Fisher(mpc='c', krange=[0.01, 0.5], validate=True)   
    #quijote_Fisher(mpc='p', krange=[0.01, 0.5], validate=True)   
