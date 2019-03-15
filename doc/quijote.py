'''

forecasts using Quijote simulations 

'''
import os 
import h5py
import numpy as np 
# --- eMaNu --- 
from emanu import util as UT
from emanu import obvs as Obvs
from emanu import plots as ePlots
from emanu import forecast as Forecast
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


quijote_thetas = {
        'Mnu': [0.1, 0.2, 0.4], # +, ++, +++ 
        'Ob': [0.048, 0.050], # others are - + 
        'Om': [0.3075, 0.3275],
        'h': [0.6511, 0.6911],
        'ns': [0.9424, 0.9824],
        's8': [0.819, 0.849] }
##################################################################
# qujiote fisher 
##################################################################
thetas = ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
theta_lbls = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$']
theta_lims = [(0.25, 0.385), (0.02, 0.08), (0.3, 1.1), (0.6, 1.3), (0.8, 0.88), (-0.45, 0.45)]
theta_fid = {'Mnu': 0., 'Ob': 0.049, 'Om': 0.3175, 'h': 0.6711,  'ns': 0.9624,  's8': 0.834} # fiducial theta 
ntheta = len(thetas)


def quijoteCov(rsd=True): 
    ''' return the full covariance matrix of the quijote bispectrum
    computed using 15000 simulations at the fiducial parameter values. 

    :return cov:
        big ass covariance matrix of all the triangle configurations in 
        the default ordering. 
    '''
    fcov = os.path.join(UT.dat_dir(), 'bispectrum', 'quijote_Cov_full%s.hdf5' % ['.real', ''][rsd])
    if os.path.isfile(fcov): 
        Fcov = h5py.File(fcov, 'r') # read in fiducial covariance matrix 
        cov = Fcov['C_bk'].value
        k1, k2, k3 = Fcov['k1'].value, Fcov['k2'].value, Fcov['k3'].value
    else: 
        quij = Obvs.quijoteBk('fiducial', rsd=rsd) 
        bks = quij['b123'] + quij['b_sn']

        cov = np.cov(bks.T) # calculate the covariance

        f = h5py.File(fcov, 'w') # write to hdf5 file 
        f.create_dataset('C_bk', data=cov) 
        f.create_dataset('k1', data=quij['k1']) 
        f.create_dataset('k2', data=quij['k2']) 
        f.create_dataset('k3', data=quij['k3']) 
        f.close()
    return k1, k2, k3, cov

# covariance matrices 
def quijote_pkCov(kmax=0.5, rsd=True): 
    ''' plot the covariance matrix of the quijote fiducial 
    bispectrum. 
    '''
    # read in P(k) 
    quij = Obvs.quijoteBk('fiducial', rsd=rsd) # theta_fiducial 
    pks = quij['p0k1'] + 1e9 / quij['Nhalos'][:,None]   # shotnoise uncorrected P(k) 
    i_k = quij['k1']
     
    # impose k limit on powerspectrum 
    kf = 2.*np.pi/1000. # fundmaentla mode
    _, _iuniq = np.unique(i_k, return_index=True)
    iuniq = np.zeros(len(i_k)).astype(bool) 
    iuniq[_iuniq] = True
    pklim = (iuniq & (i_k*kf <= kmax)) 
    pks = pks[:,pklim]
    
    C_pk = np.cov(pks.T) # covariance matrix 
    print('covariance matrix condition number = %.5e' % np.linalg.cond(C_pk)) 

    # plot the covariance matrix 
    fig = plt.figure(figsize=(10,8))
    sub = fig.add_subplot(111)
    cm = sub.pcolormesh(C_pk, norm=LogNorm(vmin=1e3, vmax=1e8))
    cbar = fig.colorbar(cm, ax=sub) 
    sub.set_title(r'Quijote $P(k)$ Covariance', fontsize=25)
    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_pkCov_kmax%s%s.png' % (str(kmax).replace('.', ''), ['_real', ''][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def quijote_bkCov(kmax=0.5, rsd=True): 
    ''' plot the covariance matrix of the quijote fiducial 
    bispectrum. 
    '''
    i_k, j_k, l_k, C_bk = quijoteCov(rsd=rsd) 

    # impose k limit on bispectrum
    kf = 2.*np.pi/1000. # fundmaentla mode
    bklim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax))
    C_bk = C_bk[bklim,:][:,bklim]

    ijl = UT.ijl_order(i_k[bklim], j_k[bklim], l_k[bklim], typ='GM') # order of triangles 
    C_bk = C_bk[ijl,:][:,ijl]
    print('covariance matrix condition number = %.5e' % np.linalg.cond(C_bk)) 

    # plot the covariance matrix 
    fig = plt.figure(figsize=(10,8))
    sub = fig.add_subplot(111)
    cm = sub.pcolormesh(C_bk, norm=LogNorm(vmin=1e11, vmax=1e18))
    cbar = fig.colorbar(cm, ax=sub) 
    cbar.set_label(r'$B(k_1, k_2, k_3)$ covariance matrix, ${\bf C}_{B}$', fontsize=25, labelpad=10, rotation=90)
    #sub.set_title(r'Quijote $B(k_1, k_2, k_3)$ Covariance', fontsize=25)
    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_bkCov_kmax%s%s.png' % (str(kmax).replace('.', ''), ['_real', ''][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def quijote_pkbkCov(kmax=0.5, rsd=True): 
    ''' plot the covariance matrix of the quijote fiducial bispectrum and powerspectrum. 
    '''
    # read in P(k) and B(k) 
    quij = Obvs.quijoteBk('fiducial', rsd=rsd) # theta_fiducial 
    bks = quij['b123'] + quij['b_sn']                   # shotnoise uncorrected B(k) 
    pks = quij['p0k1'] + 1e9 / quij['Nhalos'][:,None]   # shotnoise uncorrected P(k) 
    i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3']
     
    # impose k limit on bispectrum
    kf = 2.*np.pi/1000. # fundmaentla mode
    bklim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 
    ijl = UT.ijl_order(i_k[bklim], j_k[bklim], l_k[bklim], typ='GM') # order of triangles 
    bks = bks[:,bklim][:,ijl] 
    # impose k limit on powerspectrum 
    _, _iuniq = np.unique(i_k, return_index=True)
    iuniq = np.zeros(len(i_k)).astype(bool) 
    iuniq[_iuniq] = True
    pklim = (iuniq & (i_k*kf <= kmax)) 
    pks = pks[:,pklim]
    # powerspectrum amplitude has to be scaled 
    
    pbks = np.concatenate([pks, bks], axis=1) # joint data vector

    C_pbk = np.cov(pbks.T) # covariance matrix 
    print('covariance matrix condition number = %.5e' % np.linalg.cond(C_pbk)) 

    # plot the covariance matrix 
    fig = plt.figure(figsize=(10,8))
    sub = fig.add_subplot(111)
    cm = sub.pcolormesh(C_pbk, norm=LogNorm(vmin=1e5, vmax=1e18))
    cbar = fig.colorbar(cm, ax=sub) 
    sub.set_title(r'Quijote $P(k)$ and $B(k_1, k_2, k_3)$ Covariance', fontsize=25)
    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_pbkCov_kmax%s%s.png' % 
            (str(kmax).replace('.', ''), ['_real', ''][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


# fisher derivatives
def quijote_dPk(theta, rsd=True, dmnu='fin'):
    ''' calculate d P(k)/d theta using the paired and fixed quijote simulations
    run on perturbed theta 

    :param theta: 
        string that specifies the parameter to take the 
        derivative by. 
    '''
    if theta == 'Mnu': 
        Pks = [] # read in the bispectrum a Mnu+, Mnu++, Mnu+++
        for tt in ['fiducial', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp']: 
            quij = Obvs.quijoteBk(tt, rsd=rsd)
            Pks.append(np.average(quij['p0k1'], axis=0))
        Pk_fid, Pk_p, Pk_pp, Pk_ppp = Pks 

        h_p, h_pp, h_ppp = quijote_thetas['Mnu']
        # take the derivatives 
        if dmnu == 'p': 
            dPk = (Pk_p - Pk_fid)/h_p 
        elif dmnu == 'pp': 
            dPk = (Pk_pp - Pk_fid)/h_pp
        elif dmnu == 'ppp': 
            dPk = (Pk_ppp - Pk_fid)/h_ppp
        elif dmnu == 'fin': 
            dPk = (-21 * Pk_fid + 32 * Pk_p - 12 * Pk_pp + Pk_ppp)/(1.2) # finite difference coefficient
    elif theta == 'Mmin': 
        Pks = [] # read in the bispectrum for fiducial, Mmin+, Mmin++ 
        for tt in ['fiducial', 'Mmin_p', 'Mmin_pp']: 
            quij = Obvs.quijoteBk(tt, rsd=rsd)
            Pks.append(np.average(quij['p0k1'], axis=0))
        Pk_fid, Pk_p, Pk_pp = Pks 
    
        # step sizes 
        h_p = 0.1 # 3.3 x10^14 - 3.2 x10^14
        h_pp = 0.2 # 3.3 x10^14 - 3.2 x10^14 

        # take the derivatives 
        if dmnu == 'p': 
            dPk = (Pk_p - Pk_fid)/h_p 
        elif dmnu in ['pp', 'ppp']: 
            dPk = (Pk_pp - Pk_fid)/h_pp
        elif dmnu == 'fin': 
            dPk = (-15. * Pk_fid + 20. * Pk_p - 5. * Pk_pp) # finite difference coefficient
    elif theta == 'Amp': 
        # amplitude of P(k) is a free parameter
        quij = Obvs.quijoteBk('fiducial', rsd=rsd)
        dPk = np.average(quij['p0k1'], axis=0)
    else: 
        h = quijote_thetas[theta][1] - quijote_thetas[theta][0]
        
        quij = Obvs.quijoteBk(theta+'_m', rsd=rsd)
        Pk_m = np.average(quij['p0k1'], axis=0) # Covariance matrix tt- 
        quij = Obvs.quijoteBk(theta+'_p', rsd=rsd)
        Pk_p = np.average(quij['p0k1'], axis=0) # Covariance matrix tt+ 
        
        dPk = (Pk_p - Pk_m) / h # take the derivatives 
    return dPk


def quijote_dBk(theta, rsd=True, dmnu='fin'):
    ''' calculate d B(k)/d theta using the paired and fixed quijote simulations
    run on perturbed theta 

    :param theta: 
        string that specifies the parameter to take the 
        derivative by. 
    '''
    if theta == 'Mnu': 
        Bks = [] # read in the bispectrum at fiducial, Mnu+, Mnu++, Mnu+++
        for tt in ['fiducial', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp']: 
            quij = Obvs.quijoteBk(tt, rsd=rsd)
            Bks.append(np.average(quij['b123'], axis=0))
        Bk_fid, Bk_p, Bk_pp, Bk_ppp = Bks 

        # take the derivatives 
        h_p, h_pp, h_ppp = quijote_thetas['Mnu']
        if dmnu == 'p': 
            dBk = (Bk_p - Bk_fid) / h_p 
        elif dmnu == 'pp': 
            dBk = (Bk_pp - Bk_fid) / h_pp
        elif dmnu == 'ppp': 
            dBk = (Bk_ppp - Bk_fid) / h_ppp
        else: 
            dBk = (-21. * Bk_fid + 32. * Bk_p - 12. * Bk_pp + Bk_ppp)/1.2 # finite difference coefficient
    elif theta == 'Mmin': 
        Bks = [] # read in the bispectrum at fiducial, Mmin+, Mmin++
        for tt in ['fiducial', 'Mmin_p', 'Mmin_pp']: 
            quij = Obvs.quijoteBk(tt, rsd=rsd)
            Bks.append(np.average(quij['b123'], axis=0))
        Bk_fid, Bk_p, Bk_pp = Bks 

        # take the derivatives 
        h_p = 0.1 # 3.3 x10^14 - 3.2 x10^14
        h_pp = 0.2 # 3.3 x10^14 - 3.2 x10^14 
        if dmnu == 'p': 
            dBk = (Bk_p - Bk_fid) / h_p 
        elif dmnu in ['pp', 'ppp']: 
            dBk = (Bk_pp - Bk_fid) / h_pp
        else: 
            dBk = (-3. * Bk_fid + 4. * Bk_p - Bk_pp)/0.2 # finite difference coefficient
    elif theta == 'Amp': 
        # amplitude scaling is a free parameter
        quij = Obvs.quijoteBk('fiducial', rsd=rsd)
        dBk = np.average(quij['b123'], axis=0)
    else: 
        h = quijote_thetas[theta][1] - quijote_thetas[theta][0]
        
        quij = Obvs.quijoteBk(theta+'_m', rsd=rsd)
        Bk_m = np.average(quij['b123'], axis=0) # Covariance matrix tt- 
        quij = Obvs.quijoteBk(theta+'_p', rsd=rsd)
        Bk_p = np.average(quij['b123'], axis=0) # Covariance matrix tt+ 
        
        dBk = (Bk_p - Bk_m) / h # take the derivatives 
    return dBk


def quijote_dPdthetas(dmnu='fin'):
    ''' Compare the derivatives of the powerspectrum 
    '''
    # fiducial P0(k)  
    quij = Obvs.quijoteBk('fiducial', rsd=True)
    pk_fid = np.average(quij['p0k1'], axis=0) 
    i_k = quij['k1']
    kf = 2.*np.pi/1000.
    klim = (kf * i_k < 0.5)

    fig = plt.figure(figsize=(7,8))
    sub = fig.add_subplot(111)
    _thetas = thetas + ['Mmin', 'Amp'] 
    _theta_lbls = theta_lbls + [r'$M_{\rm min}$', '$b_1$']
    for tt, lbl in zip(_thetas, _theta_lbls): 
        dpdt = quijote_dPk(tt, rsd=True, dmnu=dmnu)
        if dpdt[klim].min() < 0: 
            sub.plot(i_k[klim] * kf, np.abs(dpdt[klim]), label='-'+lbl) 
        else: 
            sub.plot(i_k[klim] * kf, dpdt[klim], label=lbl) 

    sub.legend(loc='upper right', ncol=2, fontsize=15) 
    sub.set_xlabel('$k$', fontsize=25) 
    sub.set_xscale('log') 
    sub.set_xlim(1.8e-2, 0.5) 
    sub.set_ylabel(r'$|{\rm d}P/d\theta|$', fontsize=25) 
    sub.set_yscale('log') 
    sub.set_ylim(1e2, 1e6) 
    ffig = os.path.join(UT.fig_dir(), 'quijote_dPdthetas.%s.png' % dmnu)
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def quijote_dBdthetas(kmax=0.5, dmnu='fin'):
    ''' Compare the derivatives of the bispectrum 
    '''
    # fiducial P0(k)  
    quij = Obvs.quijoteBk('fiducial', rsd=True)
    pk_fid = np.average(quij['p0k1'], axis=0) 
    i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3']
    kf = 2.*np.pi/1000.

    klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 
    ijl = UT.ijl_order(i_k[klim], j_k[klim], l_k[klim], typ='GM') # order of triangles 

    fig = plt.figure(figsize=(20, 5))
    sub = fig.add_subplot(111)
    _thetas = thetas + ['Mmin', 'Amp'] 
    _theta_lbls = theta_lbls + [r'$M_{\rm min}$', '$b_1$']
    for tt, lbl in zip(_thetas, _theta_lbls): 
        dpdt = quijote_dBk(tt, rsd=True, dmnu=dmnu)
        if dpdt[klim].mean() < 0: 
            sub.plot(range(np.sum(klim)), np.abs(dpdt[klim]), label='-'+lbl) 
        else: 
            sub.plot(range(np.sum(klim)), dpdt[klim], label=lbl) 

    sub.legend(loc='upper right', ncol=2, fontsize=15, frameon=True) 
    sub.set_xlabel('triangle configuration', fontsize=25) 
    sub.set_xlim(0, np.sum(klim)) 
    sub.set_ylabel(r'$|{\rm d}B/d\theta|$', fontsize=25) 
    sub.set_yscale('log') 
    ffig = os.path.join(UT.fig_dir(), 'quijote_dBdthetas.%s.png' % dmnu)
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def quijote_P_theta(theta): 
    ''' Compare the quijote powerspectrum evaluated along theta axis  
    '''
    # fiducial P0(k)  
    quij = Obvs.quijoteBk('fiducial', rsd=True)
    pk_fid = np.average(quij['p0k1'], axis=0) 
    i_k = quij['k1']
    kf = 2.*np.pi/1000.
    klim = (kf * i_k < 0.5)

    pks = [] 
    if theta == 'Mnu': 
        for tt in ['Mnu_p', 'Mnu_pp', 'Mnu_ppp']: 
            _quij = Obvs.quijoteBk(tt, rsd=True)
            pks.append(np.average(_quij['p0k1'], axis=0)) 
    elif theta == 'Mmin': 
        for tt in ['Mmin_p', 'Mmin_pp']: 
            _quij = Obvs.quijoteBk(tt, rsd=True)
            pks.append(np.average(_quij['p0k1'], axis=0)) 
    else: 
        for tt in ['_m', '_p']:
            _quij = Obvs.quijoteBk(theta+tt, rsd=True)
            pks.append(np.average(_quij['p0k1'], axis=0)) 

    quijote_thetas['Mmin'] = [3.2, 3.3]
    _thetas = thetas + ['Mmin'] 
    _theta_lbls = theta_lbls + [r'$M_{\rm min}$']  

    fig = plt.figure(figsize=(7,8))
    sub = fig.add_subplot(111)
    sub.plot(i_k[klim] * kf, np.ones(np.sum(klim)), c='k', ls='--')
    for tt, pk in zip(quijote_thetas[theta], pks): 
        print('%s = %f' % (theta, tt))
        sub.plot(i_k[klim] * kf, pk[klim]/pk_fid[klim], label='%s=%.2f' % 
                (_theta_lbls[_thetas.index(theta)], tt))

    sub.legend(loc='upper right', ncol=2, fontsize=15) 
    sub.set_xlabel('$k$', fontsize=25) 
    sub.set_xscale('log') 
    sub.set_xlim(1.8e-2, 0.5) 
    sub.set_ylabel(r'$P(k)/P_{\rm fid}$ along $\theta$', fontsize=25) 
    sub.set_ylim(0.9, 1.1) 
    ffig = os.path.join(UT.fig_dir(), 'quijote_P_%s.png' % theta)
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def quijote_B_theta(theta, kmax=0.5): 
    ''' compare the B along thetas 
    '''
    kf = 2.*np.pi/1000.
    quij = Obvs.quijoteBk('fiducial', rsd=True)
    bk_fid = np.average(quij['b123'], axis=0) 
    i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3']

    klim = ((i_k * kf <= kmax) & (j_k * kf <= kmax) & (l_k * kf <= kmax)) 
    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 
    ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # order of triangles 

    fig = plt.figure(figsize=(20,8))
    sub = fig.add_subplot(111)
    for tt, lbl in zip(['Om_m', 'Ob_p', 'h_p', 'ns_m', 's8_p', 'Mnu_p'], theta_lbls): 
        quij = Obvs.quijoteBk(tt, rsd=True) 
        sub.plot(range(np.sum(klim)), (np.average(quij['b123'], axis=0)/bk_fid)[klim][ijl], label=lbl)
    sub.plot([0., np.sum(klim)], [1., 1.], c='k', ls='--', zorder=0) 
    sub.legend(loc='upper right', ncol=2, frameon=True, fontsize=20) 
    sub.set_xlabel('triangle configurations', fontsize=25) 
    sub.set_xlim(0, np.sum(klim))
    sub.set_ylabel(r'$B(k_1, k_2, k_3)/B^{\rm fid}$', fontsize=25) 
    sub.set_ylim(0.9, 1.15) 
    ffig = os.path.join(UT.fig_dir(), 'quijote_Bratio_theta.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def quijote_B_relative_error(kmax=0.5): 
    kf = 2.*np.pi/1000.
    quij = quijoteBk('fiducial', rsd=True)
    bk_fid = np.average(quij['b123'], axis=0) 
    i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3']

    klim = ((i_k * kf <= kmax) & (j_k * kf <= kmax) & (l_k * kf <= kmax)) 
    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 
    ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # order of triangles 

    _, _, _, C_fid = quijoteCov(rsd=rsd)

    Cii = np.diag(C_fid)[klim][ijl]

    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    sub.plot(range(np.sum(klim)), np.sqrt(Cii)/bk_fid[klim][ijl]) 
    sub.set_xlabel('triangle configurations', fontsize=25) 
    sub.set_xlim(0, np.sum(klim))
    sub.set_ylabel(r'$\sqrt{C_{i,i}}/B^{\rm fid}_i$', fontsize=25) 
    sub.set_ylim(0.0, 2.5) 
    ffig = os.path.join(UT.fig_dir(), 'quijote_B_relative_error.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None


# fisher matrix 
def quijote_pkFisher(kmax=0.5, rsd=True, dmnu='fin', validate=False): 
    ''' calculate fisher matrix for parameters ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu', 'Mmin']
    for P(k) analysis
    '''
    # calculate covariance matrix (with shotnoise; this is the correct one) 
    quij = quijoteBk('fiducial', rsd=rsd) # theta_fiducial 
    pks = quij['p0k1'] + 1e9/quij['Nhalos'][:,None] # uncorrect shotn oise 
    i_k = quij['k1']

    # impose k limit 
    _, _iuniq = np.unique(i_k, return_index=True)
    iuniq = np.zeros(len(i_k)).astype(bool) 
    iuniq[_iuniq] = True
    kf = 2.*np.pi/1000. # fundmaentla mode
    klim = (iuniq & (i_k*kf <= kmax)) 
    i_k = i_k[klim]

    C_fid = np.cov(pks[:,klim].T) 

    C_inv = np.linalg.inv(C_fid) # invert the covariance 
    
    dpk_dt = [] 
    for par in thetas: # calculate the derivative of Bk along all the thetas 
        dpk_dti = quijote_dPk(par, rsd=rsd, dmnu=dmnu)
        dpk_dt.append(dpk_dti[klim])
    Fij = Forecast.Fij(dpk_dt, C_inv) 
    
    if validate: 
        f_ij = os.path.join(UT.dat_dir(), 'bispectrum', 'quijote_pkFij.kmax%.2f%s.hdf5' % (kmax, ['.real', ''][rsd]))
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
        ffig = os.path.join(UT.doc_dir(), 'figs', os.path.basename(f_ij).replace('.hdf5', '.png'))
        fig.savefig(ffig, bbox_inches='tight') 
    return Fij 


def quijote_bkFisher(kmax=0.5, rsd=True, dmnu='fin', validate=False): 
    ''' calculate fisher matrix for parameters ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
    '''
    # read in full covariance matrix (with shotnoise; this is the correct one) 
    i_k, j_k, l_k, C_fid = quijoteCov(rsd=rsd)

    # impose k limit 
    kf = 2.*np.pi/1000. # fundmaentla mode
    klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 
    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 
    C_fid = C_fid[:,klim][klim,:]

    C_inv = np.linalg.inv(C_fid) # invert the covariance 
    
    dbk_dt = [] 
    for par in thetas: # derivatives of Bk along all the thetas 
        dbk_dti = quijote_dBk(par, rsd=rsd, dmnu=dmnu)
        dbk_dt.append(dbk_dti[klim])

    Fij = Forecast.Fij(dbk_dt, C_inv) 
    
    if validate: 
        f_ij = os.path.join(UT.dat_dir(), 'bispectrum', 'quijote_bkFij.kmax%.2f%s.hdf5' % (kmax, ['.real', ''][rsd]))
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
        ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_bkFij.kmax%.2f%s.png' % (kmax, ['.real', ''][rsd]))
        fig.savefig(ffig, bbox_inches='tight') 
    return Fij 


def quijote_bkFisher_triangle(typ, kmax=0.5, rsd=True, dmnu='fin'): 
    ''' calculate fisher matrix for parameters ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
    using only triangle configurations up to kmax

    :param typ: 
        string that specifies the triangle shape. typ in ['equ', 'squ']. 

    :param kmax: (default: 0.5) 
        kmax 
    '''
    # read in full covariance matrix (with shotnoise; this is the correct one) 
    i_k, j_k, l_k, C_fid = quijoteCov(rsd=rsd)

    # impose k limit 
    if typ == 'equ': # only equilateral 
        tri = (i_k == j_k) & (j_k == l_k) 
    elif typ == 'squ': # i_k >= j_k >= l_k (= 3*kf~0.01) 
        tri = (i_k == j_k) & (l_k == 3)
    kf = 2.*np.pi/1000. # fundmaentla mode
    klim = (tri & (i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 
    print('%i %s triangle configurations' % (np.sum(klim), typ))
    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 
    C_fid = C_fid[:,klim][klim,:]

    C_inv = np.linalg.inv(C_fid) # invert the covariance 
    
    dbk_dt = [] 
    for par in thetas: # calculate the derivative of Bk along all the thetas 
        dbk_dti = quijote_dBk(par, rsd=rsd, dmnu=dmnu)
        dbk_dt.append(dbk_dti[klim])
    
    return Forecast.Fij(dbk_dt, C_inv) 


# forecasts 
def quijote_Forecast(obs, kmax=0.5, rsd=True, dmnu='fin'):
    ''' fisher forecast for quijote observables
    
    :param krange: (default: 0.5) 
        tuple specifying the kranges of k1, k2, k3 in the bispectrum
    '''
    if obs == 'bk': 
        Fij = quijote_bkFisher(kmax=kmax, rsd=rsd, dmnu=dmnu, validate=False) # fisher matrix (Fij)
    elif obs == 'bk_equ': 
        Fij = quijote_bkFisher_triangle('equ', kmax=kmax, rsd=rsd, dmnu=dmnu)
    elif obs == 'bk_squ': 
        Fij = quijote_bkFisher_triangle('squ', kmax=kmax, rsd=rsd, dmnu=dmnu)
    elif obs == 'pk': 
        Fij = quijote_pkFisher(kmax=kmax, rsd=rsd, dmnu=dmnu, validate=False) # fisher matrix (Fij)
    else: 
        raise ValueError
    
    print Fij[-1,-1]
    Finv = np.linalg.inv(Fij) # invert fisher matrix 
    
    i_s8 = thetas.index('s8')
    print('sigma_s8 = %f' % np.sqrt(Finv[i_s8,i_s8]))
    i_Mnu = thetas.index('Mnu')
    print('sigma_Mnu = %f' % np.sqrt(Finv[i_Mnu,i_Mnu]))
    
    fig = plt.figure(figsize=(17, 15))
    for i in xrange(ntheta-1): 
        for j in xrange(i+1, ntheta): 
            # sub inverse fisher matrix 
            Finv_sub = np.array([[Finv[i,i], Finv[i,j]], [Finv[j,i], Finv[j,j]]]) 
            # plot the ellipse
            sub = fig.add_subplot(ntheta-1, ntheta-1, (ntheta-1) * (j-1) + i + 1) 
            plotEllipse(Finv_sub, sub, theta_fid_ij=[theta_fid[thetas[i]], theta_fid[thetas[j]]], color='C0')
            sub.set_xlim(theta_lims[i])
            sub.set_ylim(theta_lims[j])
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
    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_%sFisher_dmnu_%s_kmax%.2f%s.png' % (obs, dmnu, kmax, ['_real', ''][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def quijote_Forecast_kmax(obs, rsd=True, dmnu='fin'):
    ''' fisher forecast for quijote for different kmax values 
    '''
    kmaxs = [0.2, 0.3, 0.5]
    colrs = ['C2', 'C1', 'C0']
    alphas = [0.7, 0.8, 0.9] 
    
    # read in fisher matrix (Fij)
    Finvs = [] 
    print('%s %s-space' % (obs, ['real', 'redshift'][rsd]))
    for kmax in kmaxs: 
        if obs == 'bk': 
            Fij = quijote_bkFisher(kmax=kmax, rsd=rsd, dmnu=dmnu, validate=False)
        elif obs == 'pk': 
            Fij = quijote_pkFisher(kmax=kmax, rsd=rsd, dmnu=dmnu, validate=False)

        Finv = np.linalg.inv(Fij) # invert fisher matrix 
        Finvs.append(Finv)
        print('kmax=%f, sigma_Mnu=%f' % (kmax, np.sqrt(Finv[-1,-1])))

    fig = plt.figure(figsize=(17, 15))
    for i in xrange(ntheta-1): 
        for j in xrange(i+1, ntheta): 
            for i_k, Finv in enumerate(Finvs): 
                # sub inverse fisher matrix 
                Finv_sub = np.array([[Finv[i,i], Finv[i,j]], [Finv[j,i], Finv[j,j]]]) 

                theta_fid_i, theta_fid_j = theta_fid[thetas[i]], theta_fid[thetas[j]]
                
                # get ellipse parameters 
                a = np.sqrt(0.5*(Finv_sub[0,0] + Finv_sub[1,1]) + np.sqrt(0.25*(Finv_sub[0,0]-Finv_sub[1,1])**2 + Finv_sub[0,1]**2))
                b = np.sqrt(0.5*(Finv_sub[0,0] + Finv_sub[1,1]) - np.sqrt(0.25*(Finv_sub[0,0]-Finv_sub[1,1])**2 + Finv_sub[0,1]**2))
                theta = 0.5 * np.arctan2(2.0 * Finv_sub[0,1], (Finv_sub[0,0] - Finv_sub[1,1]))
        
                # plot the ellipse
                sub = fig.add_subplot(ntheta-1, ntheta-1, (ntheta-1) * (j-1) + i + 1) 

                e = Ellipse(xy=(theta_fid_i, theta_fid_j), width=2.48 * a, height=2.48 * b, angle=theta * 360./(2.*np.pi))
                sub.add_artist(e)
                e.set_alpha(alphas[i_k])
                e.set_facecolor(colrs[i_k])
                
                e = Ellipse(xy=(theta_fid_i, theta_fid_j), width=1.52 * a, height=1.52 * b, angle=theta * 360./(2.*np.pi),
                        fill=False, edgecolor='k', linestyle='--')
                sub.add_artist(e)

            sub.set_xlim(theta_lims[i])
            sub.set_ylim(theta_lims[j])
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

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.text(0.82, 0.77, r'$B^{\rm halo}(k_1, k_2, k_3)$', ha='right', va='bottom', 
                transform=bkgd.transAxes, fontsize=25)
    for colr, alpha, kmax in zip(colrs, alphas, kmaxs): 
        bkgd.fill_between([],[],[], color=colr, alpha=alpha, label=r'$k_1, k_2, k_3 < k_{\rm max} = %.1f$' % kmax) 
    bkgd.legend(loc='upper right', handletextpad=0.3, bbox_to_anchor=(0.925, 0.775), fontsize=25)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.05, hspace=0.05) 
    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_%sFisher_dmnu_%s_kmax%s.pdf' % (obs, dmnu, ['_real', ''][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_%sFisher_dmnu_%s_kmax%s.png' % (obs, dmnu, ['_real', ''][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def quijote_Forecast_dmnu(obs, rsd=True):
    ''' fisher forecast for quijote for different methods of calculating the 
    derivative along Mnu 
    '''
    dmnus = ['fin', 'p', 'pp', 'ppp'][::-1]
    colrs = ['C3', 'C2', 'C1', 'C0']
    alphas = [0.6, 0.7, 0.8, 0.9] 
    
    # read in fisher matrix (Fij)
    Finvs = [] 
    print('%s-space' % ['real', 'redshift'][rsd])
    for dmnu in dmnus:
        if obs == 'bk': 
            Fij = quijote_bkFisher(kmax=0.5, rsd=rsd, dmnu=dmnu, validate=False)
        elif obs == 'pk': 
            Fij = quijote_pkFisher(kmax=0.5, rsd=rsd, dmnu=dmnu, validate=False)

        Finv = np.linalg.inv(Fij) # invert fisher matrix 
        Finvs.append(Finv)
        print('dB/dMnu_%s, sigma_Mnu=%f' % (dmnu, np.sqrt(Finv[-1,-1])))

    fig = plt.figure(figsize=(17, 15))
    for i in xrange(ntheta-1): 
        for j in xrange(i+1, ntheta): 
            for i_k, Finv in enumerate(Finvs): 
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

                e = Ellipse(xy=(theta_fid_i, theta_fid_j), width=2.48 * a, height=2.48 * b, angle=theta * 360./(2.*np.pi))
                sub.add_artist(e)
                e.set_alpha(alphas[i_k])
                e.set_facecolor(colrs[i_k])
                
                e = Ellipse(xy=(theta_fid_i, theta_fid_j), width=1.52 * a, height=1.52 * b, angle=theta * 360./(2.*np.pi),
                        fill=False, edgecolor='k', linestyle='--')
                sub.add_artist(e)

            sub.set_xlim(theta_lims[i])
            sub.set_ylim(theta_lims[j])
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

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.text(0.82, 0.77, r'$B^{\rm halo}(k_1, k_2, k_3)$', ha='right', va='bottom', 
                transform=bkgd.transAxes, fontsize=25)
    for colr, alpha, dmnu in zip(colrs, alphas, dmnus): 
        bkgd.fill_between([],[],[], color=colr, alpha=alpha, label=r'${\rm d}B(k)/{\rm d}M_\nu$ %s' % dmnu) 
    bkgd.legend(loc='upper right', handletextpad=0.3, bbox_to_anchor=(0.925, 0.775), fontsize=25)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.05, hspace=0.05) 
    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_%sFisher_dmnu%s.pdf' % (obs, ['_real', ''][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_%sFisher_dmnu%s.png' % (obs, ['_real', ''][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    return None


# P(k), B(k) comparison 
def quijote_pbkForecast(kmax=0.5, rsd=True, dmnu='fin'):
    ''' fisher forecast for quijote from P(k) and B(k) overlayed on them 
    
    :param krange: (default: [0.01, 0.5]) 
        tuple specifying the kranges of k1, k2, k3 in the bispectrum
    '''
    # fisher matrices for powerspectrum and bispectrum
    pkFij = quijote_pkFisher(kmax=kmax, rsd=rsd, dmnu=dmnu, validate=False)
    bkFij = quijote_bkFisher(kmax=kmax, rsd=rsd, dmnu=dmnu, validate=False)

    pkFinv = np.linalg.inv(pkFij) # invert fisher matrix 
    bkFinv = np.linalg.inv(bkFij) # invert fisher matrix 

    print('P(k) marginalized constraint on Mnu = %f' % np.sqrt(pkFinv[-1,-1]))
    print('B(k1,k2,k3) marginalized constraint on Mnu = %f' % np.sqrt(bkFinv[-1,-1]))
    
    fig = plt.figure(figsize=(17, 15))
    for i in xrange(ntheta-1): 
        for j in xrange(i+1, ntheta): 
            theta_fid_i, theta_fid_j = theta_fid[thetas[i]], theta_fid[thetas[j]] # fiducial parameter 

            sub = fig.add_subplot(ntheta-1, ntheta-1, (ntheta-1) * (j-1) + i + 1) 
            for _i, Finv in enumerate([pkFinv, bkFinv]):
                # sub inverse fisher matrix 
                Finv_sub = np.array([[Finv[i,i], Finv[i,j]], [Finv[j,i], Finv[j,j]]]) 
                plotEllipse(Finv_sub, sub, theta_fid_ij=[theta_fid_i, theta_fid_j], color='C%i' % _i)
                
            sub.set_xlim(theta_lims[i])
            sub.set_ylim(theta_lims[j])
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

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.fill_between([],[],[], color='C0', label=r'$P^{\rm halo}_0(k)$') 
    bkgd.fill_between([],[],[], color='C1', label=r'$B^{\rm halo}(k_1, k_2, k_3)$') 
    bkgd.legend(loc='upper right', bbox_to_anchor=(0.875, 0.775), fontsize=25)
    bkgd.text(0.75, 0.61, r'$k_{\rm max} = %.1f$' % kmax, ha='right', va='bottom', 
            transform=bkgd.transAxes, fontsize=25)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.05, hspace=0.05) 
    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_pbkFisher_dmnu_%s_kmax%s%s.pdf' % 
            (dmnu, str(kmax).replace('.', ''), ['_real', ''][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_pbkFisher_dmnu_%s_kmax%s%s.png' % 
            (dmnu, str(kmax).replace('.', ''), ['_real', ''][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def quijote_pbkForecast_Mnu_s8(kmax=0.5):
    ''' fisher forecast for Mnu and sigma 8 using quijote from P(k) and B(k) 
    overlayed on them. Then overlay Hades parameter points ontop to see if
    things make somewhat sense.
    
    :param krange: (default: [0.01, 0.5]) 
        tuple specifying the kranges of k1, k2, k3 in the bispectrum
    '''
    # fisher matrices for powerspectrum and bispectrum
    pkFij = quijote_pkFisher(kmax=kmax, rsd=True, dmnu='fin', validate=False)
    bkFij = quijote_bkFisher(kmax=kmax, rsd=True, dmnu='fin', validate=False)

    pkFinv = np.linalg.inv(pkFij) # invert fisher matrix 
    bkFinv = np.linalg.inv(bkFij) # invert fisher matrix 

    print('P(k) marginalized constraint on Mnu = %f' % np.sqrt(pkFinv[-1,-1]))
    print('B(k1,k2,k3) marginalized constraint on Mnu = %f' % np.sqrt(bkFinv[-1,-1]))
    
    fig = plt.figure(figsize=(7,7))
    sub = fig.add_subplot(111) 
    i, j = 4, 5
    for _i, Finv in enumerate([pkFinv, bkFinv]):
        # sub inverse fisher matrix 
        Finv_sub = np.array([[Finv[i,i], Finv[i,j]], [Finv[j,i], Finv[j,j]]]) 
        theta_fid_i, theta_fid_j = theta_fid[thetas[j]], theta_fid[thetas[i]]
                
        plotEllipse(Finv_sub, sub, theta_fid_ij=[theta_fid_i, theta_fid_j], color='C%i' % _i)
        
    sub.set_xlabel(theta_lbls[i], labelpad=10, fontsize=30) 
    sub.set_xlim(theta_lims[i])
    sub.set_ylabel(theta_lbls[j], fontsize=30) 
    sub.set_ylim(theta_lims[j])
    sub.fill_between([],[],[], color='C0', label=r'$P^{\rm halo}_0(k)$') 
    sub.fill_between([],[],[], color='C1', label=r'$B^{\rm halo}(k_1, k_2, k_3)$') 
    # overlay hades
    hades_sig8 = [0.833, 0.822, 0.815, 0.806]
    hades_Mnus = [0., 0.06, 0.10, 0.15]
    sub.scatter(hades_sig8, hades_Mnus, c='k', marker='+', zorder=10, label='HADES sims')
    
    _hades_sig8 = [0.822, 0.818, 0.807, 0.798]
    sub.scatter(_hades_sig8, np.zeros(len(_hades_sig8)), c='k', marker='x', zorder=10)

    sub.legend(loc='upper right', handletextpad=0.2, markerscale=5, fontsize=20)
    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_pbkFisher_Mnu_sig8_kmax%s.png' % str(kmax).replace('.', ''))
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def hades_dchi2(krange=[0.01, 0.5]):
    ''' calculate delta chi-squared for the hades simulation using the quijote 
    simulation covariance matrix
    
    :param krange: (default: [0.01, 0.5]) 
        tuple specifying the kranges of k1, k2, k3 in the bispectrum
    '''
    # read in B(k) of fiducial quijote simulation for covariance matrix
    quij = quijoteBk('fiducial') # theta_fiducial 
    bks = quij['b123'] + quij['b_sn']                   # shotnoise uncorrected B(k) 
    i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3']

    # impose k limit on bispectrum
    kmin, kmax = krange 
    kf = 2.*np.pi/1000. # fundmaentla mode
    bklim = ((i_k*kf <= kmax) & (i_k*kf >= kmin) &
            (j_k*kf <= kmax) & (j_k*kf >= kmin) & 
            (l_k*kf <= kmax) & (l_k*kf >= kmin)) 
    ijl = UT.ijl_order(i_k[bklim], j_k[bklim], l_k[bklim], typ='GM') # order of triangles 
    bks = bks[:,bklim][:,ijl] 

    C_bk = np.cov(bks.T) # covariance matrix 
    C_inv = np.linalg.inv(C_bk) # invert the covariance 
    
    # fiducial hades B(k) 
    hades_fid = hadesBk(0.0, nzbin=4, rsd=True) # fiducial bispectrum
    Bk_fid = np.average(hades_fid['b123'], axis=0)

    mnus = [0.0, 0.06, 0.1, 0.15]
    sig8s = [0.822, 0.818, 0.807, 0.798]
    for mnu in mnus: # bispectrum for different Mnu
        hades_i = hadesBk(mnu, nzbin=4, rsd=True)
        _bk = np.average(hades_i['b123'], axis=0)
        dbk = (_bk - Bk_fid)[bklim][ijl]
        chi2 = np.sum(np.dot(dbk.T, np.dot(C_inv, dbk)))
        print('Mnu=%.2f, delta chi-squared %.2f' % (mnu, chi2))

    for sig8 in sig8s: # bispectrum for different sigma8 
        hades_i = hadesBk_s8(sig8, nzbin=4, rsd=True)
        _bk = np.average(hades_i['b123'], axis=0)
        dbk = (_bk - Bk_fid)[bklim][ijl]
        chi2 = np.sum(np.dot(dbk.T, np.dot(C_inv, dbk)))
        print('sig8=%.3f, delta chi-squared %.2f' % (sig8, chi2))
    return None


##################################################################
# forecasts with free scaling factor 
##################################################################
def quijote_pkFisher_freeScale(kmax=0.5, rsd=True, dmnu='fin'): 
    ''' calculate fisher matrix for P(k) analysis where we add a free 
    scaling parameter A into the analysis.
    P(k) = A * p(k) 
    '''
    # calculate covariance matrix (with shotnoise; this is the correct one) 
    quij = quijoteBk('fiducial', rsd=rsd) # theta_fiducial 
    pks = quij['p0k1'] + 1e9/quij['Nhalos'][:,None] # uncorrect shotn oise 
    i_k = quij['k1']

    # impose k limit 
    _, _iuniq = np.unique(i_k, return_index=True)
    iuniq = np.zeros(len(i_k)).astype(bool) 
    iuniq[_iuniq] = True
    kf = 2.*np.pi/1000. # fundmaentla mode
    klim = (iuniq & (i_k*kf <= kmax)) 
    i_k = i_k[klim]

    C_fid = np.cov(pks[:,klim].T) 

    C_inv = np.linalg.inv(C_fid) # invert the covariance 
    
    dpk_dt = [] 
    for par in thetas: # calculate the derivative of Bk along all the thetas 
        dpk_dti = quijote_dPk(par, rsd=rsd, dmnu=dmnu)
        dpk_dt.append(dpk_dti[klim])
    # d P(k) / d A = p(k) 
    dpk_dti = quijote_dPk('Amp', rsd=rsd, dmnu=dmnu)
    dpk_dt.append(dpk_dti[klim]) 

    Fij = Forecast.Fij(dpk_dt, C_inv) 
    return Fij 


def quijote_bkFisher_freeScale(kmax=0.5, rsd=True, dmnu='fin', validate=False): 
    ''' calculate fisher matrix for parameters ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
    '''
    # read in full covariance matrix (with shotnoise; this is the correct one) 
    i_k, j_k, l_k, C_fid = quijoteCov(rsd=rsd)

    # impose k limit 
    kf = 2.*np.pi/1000. # fundmaentla mode
    klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 
    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 
    C_fid = C_fid[:,klim][klim,:]

    C_inv = np.linalg.inv(C_fid) # invert the covariance 
    
    dbk_dt = [] 
    for par in thetas: # derivatives of Bk along all the thetas 
        dbk_dti = quijote_dBk(par, rsd=rsd, dmnu=dmnu)
        dbk_dt.append(dbk_dti[klim])
    # amplitude scaling factor Amp 
    dbk_dti = quijote_dBk('Amp', rsd=rsd, dmnu=dmnu)
    dbk_dt.append(dbk_dti[klim])

    Fij = Forecast.Fij(dbk_dt, C_inv) 
    return Fij 


def quijote_bkFisher_triangle_freeScale(typ, kmax=0.5, rsd=True, dmnu='fin'): 
    ''' calculate fisher matrix for parameters ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
    using only triangle configurations up to kmax

    :param typ: 
        string that specifies the triangle shape. typ in ['equ', 'squ']. 

    :param kmax: (default: 0.5) 
        kmax 
    '''
    # read in full covariance matrix (with shotnoise; this is the correct one) 
    i_k, j_k, l_k, C_fid = quijoteCov(rsd=rsd)

    # impose k limit 
    if typ == 'equ': # only equilateral 
        tri = (i_k == j_k) & (j_k == l_k) 
    elif typ == 'squ': # i_k >= j_k >= l_k (= 3*kf~0.01) 
        tri = (i_k == j_k) & (l_k == 3)
    kf = 2.*np.pi/1000. # fundmaentla mode
    klim = (tri & (i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 
    print('%i %s triangle configurations' % (np.sum(klim), typ))
    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 
    C_fid = C_fid[:,klim][klim,:]

    C_inv = np.linalg.inv(C_fid) # invert the covariance 
    
    dbk_dt = [] 
    for par in thetas: # calculate the derivative of Bk along all the thetas 
        dbk_dti = quijote_dBk(par, rsd=rsd, dmnu=dmnu)
        dbk_dt.append(dbk_dti[klim])
    # amplitude scaling factor Amp 
    dbk_dti = quijote_dBk('Amp', rsd=rsd, dmnu=dmnu)
    dbk_dt.append(dbk_dti[klim])
    
    Fij = Forecast.Fij(dbk_dt, C_inv) 
    return Fij 


def quijote_Forecast_freeScale(obs, kmax=0.5, rsd=True, dmnu='fin'):
    ''' fisher forecast for quijote observables where a scaling factor of the
    amplitude is added as a free parameter. This is to conservatively estimate
    what would happen if we wipe out any multiplicative amplitude information.
    
    :param kmax: (default: 0.5) 
        kmax of the analysis 
    '''
    if obs == 'bk': 
        _Fij = quijote_bkFisher(kmax=kmax, rsd=rsd, dmnu=dmnu) # fisher matrix (Fij)
        Fij = quijote_bkFisher_freeScale(kmax=kmax, rsd=rsd, dmnu=dmnu) # fisher matrix (Fij)
    elif obs == 'pk': 
        _Fij = quijote_pkFisher(kmax=kmax, rsd=rsd, dmnu=dmnu) # fisher matrix (Fij)
        Fij = quijote_pkFisher_freeScale(kmax=kmax, rsd=rsd, dmnu=dmnu) # fisher matrix (Fij)
    elif obs == 'bk_equ': 
        _Fij = quijote_bkFisher_triangle('equ', kmax=kmax, rsd=rsd, dmnu=dmnu) # fisher matrix (Fij)
        Fij = quijote_bkFisher_triangle_freeScale('equ', kmax=kmax, rsd=rsd, dmnu=dmnu) # fisher matrix (Fij)
    elif obs == 'bk_squ': 
        _Fij = quijote_bkFisher_triangle('squ', kmax=kmax, rsd=rsd, dmnu=dmnu) # fisher matrix (Fij)
        Fij = quijote_bkFisher_triangle_freeScale('squ', kmax=kmax, rsd=rsd, dmnu=dmnu) # fisher matrix (Fij)
    else: 
        raise ValueError
    
    _Finv = np.linalg.inv(_Fij) # invert fisher matrix 
    Finv = np.linalg.inv(Fij) # invert fisher matrix 
    print('sigma_Mnu = %f' % np.sqrt(_Finv[-1,-1]))
    print('sigma_Mnu = %f (Amp free)' % np.sqrt(Finv[-2,-2]))

    _thetas = thetas + ['Amp'] 
    _theta_fid = {'Mnu': 0., 'Ob': 0.049, 'Om': 0.3175, 'h': 0.6711,  'ns': 0.9624,  's8': 0.834, 'Amp':1.} # fiducial theta 
    if 'bk_' in obs: 
        _theta_lims = [(0.1, 0.5), (0.0, 0.2), (0.0, 1.3), (0.4, 1.6), (0.0, 2.), (-1., 1.), (0.5, 2.)]
    else: 
        _theta_lims = [(0.25, 0.385), (0.02, 0.08), (0.3, 1.1), (0.6, 1.3), (0.8, 0.88), (-0.4, 0.4), (0.8, 1.2)]
    _theta_lbls = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$', '$b_1$']
    
    fig = plt.figure(figsize=(17, 15))
    for i in xrange(ntheta): 
        for j in xrange(i+1, ntheta+1): 
            # sub inverse fisher matrix 
            Finv_sub = np.array([[Finv[i,i], Finv[i,j]], [Finv[j,i], Finv[j,j]]]) 
            # plot the ellipse
            sub = fig.add_subplot(ntheta, ntheta, ntheta * (j-1) + i + 1) 
            plotEllipse(Finv_sub, sub, theta_fid_ij=[_theta_fid[_thetas[i]], _theta_fid[_thetas[j]]], color='C0')
            if j < ntheta:
                _Finv_sub = np.array([[_Finv[i,i], _Finv[i,j]], [_Finv[j,i], _Finv[j,j]]]) 
                plotEllipse(_Finv_sub, sub, theta_fid_ij=[_theta_fid[_thetas[i]], _theta_fid[_thetas[j]]], color='C1')
            sub.set_xlim(_theta_lims[i])
            sub.set_ylim(_theta_lims[j])
            if i == 0:   
                sub.set_ylabel(_theta_lbls[j], fontsize=30) 
            else: 
                sub.set_yticks([])
                sub.set_yticklabels([])
            
            if j == ntheta: 
                sub.set_xlabel(_theta_lbls[i], labelpad=10, fontsize=30) 
            else: 
                sub.set_xticks([])
                sub.set_xticklabels([]) 
    fig.subplots_adjust(wspace=0.05, hspace=0.05) 
    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_%sFisher_freeScale_dmnu_%s_kmax%.2f%s.png' % 
            (obs, dmnu, kmax, ['_real', ''][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    return None

##################################################################
# forecasts with free Mmin 
##################################################################
def quijote_dbk_dMnu_dMmin(kmax=0.5, rsd=True, dmnu='fin'): 
    ''' Compare the dB(k)/dMnu versus dB(k)/dMnu 
    '''
    quij = quijoteBk('fiducial', rsd=rsd) 
    i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3'] 
    bk_fid = np.average(quij['b123'], axis=0)

    # impose k limit 
    kf = 2.*np.pi/1000. # fundmaentla mode
    klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 
    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 
    ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # order of triangles 

    dbk_dmnu = quijote_dBk('Mnu', rsd=rsd, dmnu=dmnu)
    dbk_dmmin = quijote_dBk('Mmin', rsd=rsd, dmnu=dmnu)

    fig = plt.figure(figsize=(20,10))
    sub = fig.add_subplot(211) 
    for tt, ls, lbl in zip(['Mmin_p', 'Mmin_pp'], ['-', ':'], ['Mmin+', 'Mmin++']): 
        quij = quijoteBk(tt, rsd=rsd)
        _bk = np.average(quij['b123'], axis=0) 
        sub.plot(range(np.sum(klim)), _bk[klim][ijl]/bk_fid[klim][ijl], c='C0', ls=ls, label=lbl)  
    
    for tt, ls, lbl in zip(['Mnu_p', 'Mnu_pp'], ['-', ':'], ['Mnu+', 'Mnu++']): 
            quij = quijoteBk(tt, rsd=rsd)
            _bk = np.average(quij['b123'], axis=0) 
            sub.plot(range(np.sum(klim)), _bk[klim][ijl]/bk_fid[klim][ijl], c='C1', ls=ls, label=lbl)  
    sub.plot([0., np.sum(klim)], [1., 1.], c='k', ls='--') 
    sub.legend(loc='upper right', fontsize=20)     
    sub.set_xlabel('triangle configuration', fontsize=25) 
    sub.set_xlim(0, np.sum(klim))
    sub.set_ylabel(r'$B(k)/B_{\rm fid}(k)$', fontsize=25) 
    sub.set_ylim(0.95, 1.15) 

    sub = fig.add_subplot(212) 
    sub.plot(range(np.sum(klim)), dbk_dmnu[klim][ijl], c='k', label=r'$M_\nu$')
    sub.plot(range(np.sum(klim)), dbk_dmmin[klim][ijl], c='C1', label=r'$M_{\rm min}$') 
    sub.legend(loc='upper right', fontsize=20) 
    sub.set_xlabel('triangle configuration', fontsize=25) 
    sub.set_xlim(0, np.sum(klim))
    sub.set_ylabel(r'$dB(k)/d\theta$', fontsize=25) 
    sub.set_yscale('log') 
    ffig = os.path.join(UT.fig_dir(), 'quijote_dbk_dMnu_dMmin.%.2f.%s.png' % (kmax, dmnu))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def quijote_pkFisher_freeMmin(kmax=0.5, rsd=True, dmnu='fin'): 
    ''' calculate fisher matrix for P(k) analysis where we allow Mmin to 
    be a free parameter
    '''
    # calculate covariance matrix (with shotnoise; this is the correct one) 
    quij = quijoteBk('fiducial', rsd=rsd) # theta_fiducial 
    pks = quij['p0k1'] + 1e9/quij['Nhalos'][:,None] # uncorrect shotn oise 
    i_k = quij['k1']

    # impose k limit 
    _, _iuniq = np.unique(i_k, return_index=True)
    iuniq = np.zeros(len(i_k)).astype(bool) 
    iuniq[_iuniq] = True
    kf = 2.*np.pi/1000. # fundmaentla mode
    klim = (iuniq & (i_k*kf <= kmax)) 
    i_k = i_k[klim]

    C_fid = np.cov(pks[:,klim].T) 

    C_inv = np.linalg.inv(C_fid) # invert the covariance 
    
    dpk_dt = [] 
    for par in thetas: # calculate the derivative of Bk along all the thetas 
        dpk_dti = quijote_dPk(par, rsd=rsd, dmnu=dmnu)
        dpk_dt.append(dpk_dti[klim])
    # d P(k) / d Mmin 
    dpk_dti = quijote_dPk('Mmin', rsd=rsd, dmnu=dmnu)
    dpk_dt.append(dpk_dti[klim]) 
    # Also Amplitude
    dpk_dti = quijote_dPk('Amp', rsd=rsd, dmnu=dmnu)
    dpk_dt.append(dpk_dti[klim]) 

    Fij = Forecast.Fij(dpk_dt, C_inv) 
    return Fij 


def quijote_bkFisher_freeMmin(kmax=0.5, rsd=True, dmnu='fin', validate=False): 
    ''' calculate fisher matrix for B(k) analysis where we allow Mmin to 
    be a free parameter
    '''
    # read in full covariance matrix (with shotnoise; this is the correct one) 
    i_k, j_k, l_k, C_fid = quijoteCov(rsd=rsd)

    # impose k limit 
    kf = 2.*np.pi/1000. # fundmaentla mode
    klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 
    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 
    C_fid = C_fid[:,klim][klim,:]

    C_inv = np.linalg.inv(C_fid) # invert the covariance 
    
    dbk_dt = [] 
    for par in thetas: # derivatives of Bk along all the thetas 
        dbk_dti = quijote_dBk(par, rsd=rsd, dmnu=dmnu)
        dbk_dt.append(dbk_dti[klim])
    # d B(k) / d Mmin 
    dbk_dti = quijote_dBk('Mmin', rsd=rsd, dmnu=dmnu)
    dbk_dt.append(dbk_dti[klim])
    # Also Amplitude
    dbk_dti = quijote_dBk('Amp', rsd=rsd, dmnu=dmnu)
    dbk_dt.append(dbk_dti[klim]) 

    Fij = Forecast.Fij(dbk_dt, C_inv) 
    return Fij 


def quijote_Forecast_freeMmin(obs, kmax=0.5, rsd=True, dmnu='fin'):
    ''' fisher forecast for quijote observables where a scaling factor of the
    amplitude is added as a free parameter. This is to conservatively estimate
    what would happen if we wipe out any multiplicative amplitude information.
    
    :param kmax: (default: 0.5) 
        kmax of the analysis 
    '''
    # fisher matrix (Fij)
    if obs == 'bk': 
        _Fij = quijote_bkFisher(kmax=kmax, rsd=rsd, dmnu=dmnu)          # original 
        Fij = quijote_bkFisher_freeMmin(kmax=kmax, rsd=rsd, dmnu=dmnu)   # w/ free Mmin 
    elif obs == 'pk': 
        _Fij = quijote_pkFisher(kmax=kmax, rsd=rsd, dmnu=dmnu)          # original 
        Fij = quijote_pkFisher_freeMmin(kmax=kmax, rsd=rsd, dmnu=dmnu)  # w/ free Mmin
    else: 
        raise ValueError
    
    _Finv = np.linalg.inv(_Fij) # invert fisher matrix 
    Finv = np.linalg.inv(Fij) # invert fisher matrix 

    #i_s8 = thetas.index('s8')
    #print('original sigma_s8 = %f' % np.sqrt(_Finv[i_s8,i_s8]))
    #print('w/ Mmin sigma_s8 = %f' % np.sqrt(Finv[i_s8,i_s8]))
    i_Mnu = thetas.index('Mnu')
    print('original sigma_Mnu = %f' % np.sqrt(_Finv[i_Mnu,i_Mnu]))
    print('w/ Mmin sigma_Mnu = %f' % np.sqrt(Finv[i_Mnu,i_Mnu]))

    _thetas = thetas + ['Mmin', 'Amp'] 
    _theta_fid = {'Mnu': 0., 'Ob': 0.049, 'Om': 0.3175, 'h': 0.6711,  'ns': 0.9624,  's8': 0.834, 'Mmin':3.2, 'Amp': 1.} # fiducial theta 
    if 'bk_' in obs: 
        _theta_lims = [(0.1, 0.5), (0.0, 0.2), (0.0, 1.3), (0.4, 1.6), (0.0, 2.), (-1., 1.), (2.8, 3.6), (0.8, 1.2)]
    else: 
        _theta_lims = [(0.25, 0.385), (0.02, 0.08), (0.3, 1.1), (0.6, 1.3), (0.8, 0.88), (-0.4, 0.4), (2.8, 3.6), (0.8, 1.2)]
    _theta_lbls = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$', r'$M_{\rm min}$', '$b_1$']
    
    fig = plt.figure(figsize=(17, 15))
    for i in xrange(ntheta+1): 
        for j in xrange(i+1, ntheta+2): 
            # sub inverse fisher matrix 
            Finv_sub = np.array([[Finv[i,i], Finv[i,j]], [Finv[j,i], Finv[j,j]]]) 
            # plot the ellipse
            sub = fig.add_subplot(ntheta+1, ntheta+1, (ntheta+1) * (j-1) + i + 1) 
            plotEllipse(Finv_sub, sub, theta_fid_ij=[_theta_fid[_thetas[i]], _theta_fid[_thetas[j]]], color='C0')
            if j < ntheta:
                _Finv_sub = np.array([[_Finv[i,i], _Finv[i,j]], [_Finv[j,i], _Finv[j,j]]]) 
                plotEllipse(_Finv_sub, sub, theta_fid_ij=[_theta_fid[_thetas[i]], _theta_fid[_thetas[j]]], color='C1')
            sub.set_xlim(_theta_lims[i])
            sub.set_ylim(_theta_lims[j])
            if i == 0:   
                sub.set_ylabel(_theta_lbls[j], fontsize=30) 
            else: 
                sub.set_yticks([])
                sub.set_yticklabels([])
            
            if j == ntheta+1: 
                sub.set_xlabel(_theta_lbls[i], labelpad=10, fontsize=30) 
            else: 
                sub.set_xticks([])
                sub.set_xticklabels([]) 
    fig.subplots_adjust(wspace=0.05, hspace=0.05) 
    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_%sFisher_freeMmin_dmnu_%s_kmax%.2f%s.png' % 
            (obs, dmnu, kmax, ['_real', ''][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    return None


##################################################################
# forecasts with fixed nbar 
##################################################################
def quijote_bkCov_fixednbar(kmax=0.5, rsd=True): 
    ''' plot the covariance matrix of the quijote fiducial 
    bispectrum. 
    '''
    # read in B(k) of fiducial quijote simulation 
    quij = quijoteBk('fiducial', rsd=rsd, flag='.fixed_nbar') # theta_fiducial 
    bks = quij['b123'] + quij['b_sn']                   # shotnoise uncorrected B(k) 
    i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3']

    # impose k limit on bispectrum
    kf = 2.*np.pi/1000. # fundmaentla mode
    bklim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax))
    ijl = UT.ijl_order(i_k[bklim], j_k[bklim], l_k[bklim], typ='GM') # order of triangles 
    bks = bks[:,bklim][:,ijl] 

    C_bk = np.cov(bks.T) # covariance matrix 
    print('covariance matrix condition number = %.5e' % np.linalg.cond(C_bk)) 

    # plot the covariance matrix 
    fig = plt.figure(figsize=(10,8))
    sub = fig.add_subplot(111)
    cm = sub.pcolormesh(C_bk, norm=LogNorm(vmin=1e11, vmax=1e18))
    cbar = fig.colorbar(cm, ax=sub) 
    cbar.set_label(r'$B(k_1, k_2, k_3)$ covariance matrix, ${\bf C}_{B}$', fontsize=25, labelpad=10, rotation=90)
    #sub.set_title(r'Quijote $B(k_1, k_2, k_3)$ Covariance', fontsize=25)
    ffig = os.path.join(UT.fig_dir(), 'quijote_bkCov.fixed_nbar.kmax%s%s.png' % (str(kmax).replace('.', ''), ['_real', ''][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def quijote_dPk_fixednbar(theta, dmnu='fin'):
    ''' calculate shot noise uncorrected dP(k)/d theta using the paired 
    and fixed quijote simulations run on perturbed theta 

    :param theta: 
        string that specifies the parameter to take the 
        derivative by. 
    '''
    if theta == 'Mnu': 
        Pks = [] # read in the bispectrum a Mnu+, Mnu++, Mnu+++
        for tt in ['fiducial', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp']: 
            quij = quijoteBk(tt, rsd=True, flag='.fixed_nbar')
            Pks.append(np.average(quij['p0k1'] + 1e9/quij['Nhalos'][:,None], axis=0)) # SN uncorrected
        Pk_fid, Pk_p, Pk_pp, Pk_ppp = Pks 

        h_p, h_pp, h_ppp = quijote_thetas['Mnu']
        # take the derivatives 
        if dmnu == 'p': 
            dPk = (Pk_p - Pk_fid)/h_p 
        elif dmnu == 'pp': 
            dPk = (Pk_pp - Pk_fid)/h_pp
        elif dmnu == 'ppp': 
            dPk = (Pk_ppp - Pk_fid)/h_ppp
        elif dmnu == 'fin': 
            dPk = (-21 * Pk_fid + 32 * Pk_p - 12 * Pk_pp + Pk_ppp)/(1.2) # finite difference coefficient
    else: 
        h = quijote_thetas[theta][1] - quijote_thetas[theta][0] # step size
        
        quij = quijoteBk(theta+'_m', rsd=True, flag='.fixed_nbar')
        Pk_m = np.average(quij['p0k1'] + 1e9/quij['Nhalos'][:,None], axis=0) # Covariance matrix tt- 
        quij = quijoteBk(theta+'_p', rsd=True, flag='.fixed_nbar')
        Pk_p = np.average(quij['p0k1'] + 1e9/quij['Nhalos'][:,None], axis=0) # Covariance matrix tt+ 
        
        # take the derivatives 
        dPk = (Pk_p - Pk_m) / h 
    return dPk


def quijote_dBk_fixednbar(theta, dmnu='fin'):
    ''' calculate fixed nbar dB(k)/dtheta using the paired and fixed 
    quijote simulations run on perturbed theta 

    :param theta: 
        string that specifies the parameter to take the 
        derivative by. 
    '''
    if theta == 'Mnu': 
        Bks = [] # read in the bispectrum a Mnu+, Mnu++, Mnu+++
        for tt in ['fiducial', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp']: 
            quij = quijoteBk(tt, rsd=True, flag='.fixed_nbar')
            Bks.append(np.average(quij['b123'] + quij['b_sn'], axis=0))
        Bk_fid, Bk_p, Bk_pp, Bk_ppp = Bks 

        # take the derivatives 
        h_p, h_pp, h_ppp = quijote_thetas['Mnu']
        if dmnu == 'p': 
            dBk = (Bk_p - Bk_fid) / h_p 
        elif dmnu == 'pp': 
            dBk = (Bk_pp - Bk_fid) / h_pp
        elif dmnu == 'ppp': 
            dBk = (Bk_ppp - Bk_fid) / h_ppp
        else: 
            dBk = (-21. * Bk_fid + 32. * Bk_p - 12. * Bk_pp + Bk_ppp)/1.2 # finite difference coefficient
    else: 
        h = quijote_thetas[theta][1] - quijote_thetas[theta][0]

        quij = quijoteBk(theta+'_m', rsd=True, flag='.fixed_nbar')
        Bk_m = np.average(quij['b123'] + quij['b_sn'], axis=0) # Covariance matrix tt- 
        quij = quijoteBk(theta+'_p', rsd=True, flag='.fixed_nbar')
        Bk_p = np.average(quij['b123'] + quij['b_sn'], axis=0) # Covariance matrix tt+ 
        
        # take the derivatives 
        dBk = (Bk_p - Bk_m) / h 
    return dBk


def quijote_fixednbar_derivative_comparison(obs, kmax=0.5): 
    ''' compare the derivatives d P/B(k) / d theta where nbar is fixed and not fixed
    '''
    fig = plt.figure(figsize=(20, 5*len(thetas)))
    quij = quijoteBk('fiducial', rsd=True)
    kf = 2.*np.pi/1000.
    i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3'] 

    for i, tt in enumerate(thetas): 
        sub = fig.add_subplot(len(thetas),1,i+1) 
        if obs == 'pk': 
            _dobsdt = quijote_dPk(tt, rsd=True, dmnu='fin') # original deriative 
            dobsdt = quijote_dPk_fixednbar(tt, dmnu='fin')  # fixed nbar 
        
            klim = (i_k * kf < kmax) 

            sub.plot(kf * i_k[klim], _dobsdt[klim], c='k', label='original') 
            sub.plot(kf * i_k[klim], dobsdt[klim], c='C1', label=r'fixed $\bar{n}$') 
            sub.set_xscale('log') 
            sub.set_xlim(1.8e-2, 0.5) 
            sub.set_ylabel(r"d$P$/d%s" % tt, fontsize=20) 
        elif obs == 'bk': 
            _dobsdt = quijote_dBk(tt, rsd=True, dmnu='fin') # original deriative 
            dobsdt = quijote_dBk_fixednbar(tt, dmnu='fin')  # fixed nbar 

            klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) &  (l_k*kf <= kmax)) 
            sub.plot(range(np.sum(klim)), _dobsdt[klim], c='k', label='original') 
            sub.plot(range(np.sum(klim)), dobsdt[klim], c='C1', label=r'fixed $\bar{n}$') 
            sub.set_xlim(0, np.sum(klim))
            sub.set_yscale("symlog") 
            sub.set_ylabel(r"d$B$/d%s" % tt, fontsize=20) 
        if i == 0: sub.legend(loc='upper right', fontsize=20) 

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    if obs == 'pk': 
        bkgd.set_xlabel("$k$ [$h$/Mpc]", labelpad=10, fontsize=25) 
    elif obs == 'bk': 
        bkgd.set_xlabel("triangle configuration", labelpad=10, fontsize=25) 

    ffig = os.path.join(UT.fig_dir(), 'quijote_fixednbar_d%s_comparison_kmax%.2f.png' % (obs, kmax))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def quijote_pkFisher_fixednbar(kmax=0.5, rsd=True, dmnu='fin'): 
    ''' calculate fisher matrix for parameters ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
    for P(k) analysis without shot noise correction!
    '''
    # calculate covariance matrix (with shotnoise; this is the correct one) 
    quij = quijoteBk('fiducial', rsd=True, flag='.fixed_nbar') # theta_fiducial 
    pks = quij['p0k1'] + 1e9/quij['Nhalos'][:,None] # uncorrect shotn oise 
    i_k = quij['k1']

    # impose k limit 
    _, _iuniq = np.unique(i_k, return_index=True)
    iuniq = np.zeros(len(i_k)).astype(bool) 
    iuniq[_iuniq] = True
    kf = 2.*np.pi/1000. # fundmaentla mode
    klim = (iuniq & (i_k*kf <= kmax)) 
    i_k = i_k[klim]

    C_fid = np.cov(pks[:,klim].T) 

    C_inv = np.linalg.inv(C_fid) # invert the covariance 
    
    dpk_dt = [] 
    for par in thetas: 
        # calculate the derivative of Pk (uncorrected for SN) along all the thetas 
        dpk_dti = quijote_dPk_SNuncorr(par, dmnu=dmnu)
        dpk_dt.append(dpk_dti[klim])
    
    Fij = Forecast.Fij(dpk_dt, C_inv) 
    return Fij 


def quijote_bkFisher_fixednbar(kmax=0.5, dmnu='fin'): 
    ''' calculate fisher matrix for parameters ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
    '''
    # calculate full covariance matrix (with shotnoise; this is the correct one) 
    # read in B(k) of fiducial quijote simulation 
    quij = quijoteBk('fiducial', rsd=True, flag='.fixed_nbar')   # theta_fiducial for fixed nbar 
    bks = quij['b123'] + quij['b_sn']                           # uncorrect shotnoise 
    i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3']
    C_fid = np.cov(bks.T)  # full covariance matrix

    # impose k limit 
    kf = 2.*np.pi/1000. # fundmaentla mode
    klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) &  (l_k*kf <= kmax)) 
    C_fid = C_fid[:,klim][klim,:]

    C_inv = np.linalg.inv(C_fid) # invert the covariance 
    
    dbk_dt = [] 
    for par in thetas: # calculate the derivative of Bk along all the thetas 
        dbk_dti = quijote_dBk_fixednbar(par, dmnu=dmnu)
        dbk_dt.append(dbk_dti[klim])
    
    Fij = Forecast.Fij(dbk_dt, C_inv) 
    return Fij 


def quijote_Forecast_fixednbar(obs, kmax=0.5, dmnu='fin'):
    ''' fisher forecast for quijote 
    
    :param krange: (default: [0.01, 0.5]) 
        tuple specifying the kranges of k1, k2, k3 in the bispectrum
    '''
    # fisher matrix (Fij)
    if obs == 'pk': 
        _Fij = quijote_pkFisher(kmax=kmax, dmnu=dmnu)
        Fij = quijote_pkFisher_fixednbar(kmax=kmax, dmnu=dmnu) 
    elif obs == 'bk': 
        _Fij = quijote_bkFisher(kmax=kmax, dmnu=dmnu)
        Fij = quijote_bkFisher_fixednbar(kmax=kmax, dmnu=dmnu) 
    
    _Finv = np.linalg.inv(_Fij)
    Finv = np.linalg.inv(Fij) # invert fisher matrix 
    i_s8 = thetas.index('s8')
    print('original sigma_s8 = %f' % np.sqrt(_Finv[i_s8,i_s8]))
    print('fixed nbar sigma_s8 = %f' % np.sqrt(Finv[i_s8,i_s8]))
    i_Mnu = thetas.index('Mnu')
    print('original sigma_Mnu = %f' % np.sqrt(_Finv[i_Mnu,i_Mnu]))
    print('fixed nbar sigma_Mnu = %f' % np.sqrt(Finv[i_Mnu,i_Mnu]))

    fig = plt.figure(figsize=(17, 15))
    for i in xrange(ntheta-1): 
        for j in xrange(i+1, ntheta): 
            # sub inverse fisher matrix 
            Finv_sub = np.array([[Finv[i,i], Finv[i,j]], [Finv[j,i], Finv[j,j]]]) 

            theta_fid_i, theta_fid_j = theta_fid[thetas[i]], theta_fid[thetas[j]]
            sub = fig.add_subplot(ntheta-1, ntheta-1, (ntheta-1) * (j-1) + i + 1) 

            plotEllipse(Finv_sub, sub, theta_fid_ij=[theta_fid_i, theta_fid_j], color='C0')
                
            _Finv_sub = np.array([[_Finv[i,i], _Finv[i,j]], [_Finv[j,i], _Finv[j,j]]]) 
            plotEllipse(_Finv_sub, sub, theta_fid_ij=[theta_fid_i, theta_fid_j], color='C1')
            
            sub.set_xlim(theta_lims[i])
            sub.set_ylim(theta_lims[j])
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
    ffig = os.path.join(UT.fig_dir(), 'quijote_%s_fixednbar_Fisher_kmax%.2f.png' % (obs, kmax))
    fig.savefig(ffig, bbox_inches='tight') 
    return None

##################################################################
# shot noise uncorrected forecasts
##################################################################
def quijote_dPk_SNuncorr(theta, rsd=True, dmnu='fin'):
    ''' calculate shot noise uncorrected dP(k)/d theta using the paired 
    and fixed quijote simulations run on perturbed theta 

    :param theta: 
        string that specifies the parameter to take the 
        derivative by. 
    '''
    if theta == 'Mnu': 
        Pks = [] # read in the bispectrum a Mnu+, Mnu++, Mnu+++
        for tt in ['fiducial', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp']: 
            quij = quijoteBk(tt, rsd=rsd)
            Pks.append(np.average(quij['p0k1'] + 1e9/quij['Nhalos'][:,None], axis=0)) # SN uncorrected
        Pk_fid, Pk_p, Pk_pp, Pk_ppp = Pks 

        h_p, h_pp, h_ppp = quijote_thetas['Mnu']
        # take the derivatives 
        if dmnu == 'p': 
            dPk = (Pk_p - Pk_fid)/h_p 
        elif dmnu == 'pp': 
            dPk = (Pk_pp - Pk_fid)/h_pp
        elif dmnu == 'ppp': 
            dPk = (Pk_ppp - Pk_fid)/h_ppp
        elif dmnu == 'fin': 
            dPk = (-21 * Pk_fid + 32 * Pk_p - 12 * Pk_pp + Pk_ppp)/(1.2) # finite difference coefficient
    else: 
        h = quijote_thetas[theta][1] - quijote_thetas[theta][0] # step size
        
        quij = quijoteBk(theta+'_m', rsd=rsd)
        Pk_m = np.average(quij['p0k1'] + 1e9/quij['Nhalos'][:,None], axis=0) # Covariance matrix tt- 
        quij = quijoteBk(theta+'_p', rsd=rsd)
        Pk_p = np.average(quij['p0k1'] + 1e9/quij['Nhalos'][:,None], axis=0) # Covariance matrix tt+ 
        
        # take the derivatives 
        dPk = (Pk_p - Pk_m) / h 
    return dPk


def quijote_dBk_SNuncorr(theta, rsd=True, dmnu='fin'):
    ''' calculate shot noise uncorrected dB(k)/dtheta using the paired and fixed 
    quijote simulations run on perturbed theta 

    :param theta: 
        string that specifies the parameter to take the 
        derivative by. 
    '''
    if theta == 'Mnu': 
        Bks = [] # read in the bispectrum a Mnu+, Mnu++, Mnu+++
        for tt in ['fiducial', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp']: 
            quij = quijoteBk(tt, rsd=rsd)
            Bks.append(np.average(quij['b123'] + quij['b_sn'], axis=0))
        Bk_fid, Bk_p, Bk_pp, Bk_ppp = Bks 

        # take the derivatives 
        h_p, h_pp, h_ppp = quijote_thetas['Mnu']
        if dmnu == 'p': 
            dBk = (Bk_p - Bk_fid) / h_p 
        elif dmnu == 'pp': 
            dBk = (Bk_pp - Bk_fid) / h_pp
        elif dmnu == 'ppp': 
            dBk = (Bk_ppp - Bk_fid) / h_ppp
        else: 
            dBk = (-21. * Bk_fid + 32. * Bk_p - 12. * Bk_pp + Bk_ppp)/1.2 # finite difference coefficient
    else: 
        h = quijote_thetas[theta][1] - quijote_thetas[theta][0]

        quij = quijoteBk(theta+'_m', rsd=rsd)
        Bk_m = np.average(quij['b123'] + quij['b_sn'], axis=0) # Covariance matrix tt- 
        quij = quijoteBk(theta+'_p', rsd=rsd)
        Bk_p = np.average(quij['b123'] + quij['b_sn'], axis=0) # Covariance matrix tt+ 
        
        # take the derivatives 
        dBk = (Bk_p - Bk_m) / h 
    return dBk


def quijote_pkFisher_SNuncorr(kmax=0.5, rsd=True, dmnu='fin'): 
    ''' calculate fisher matrix for parameters ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
    for P(k) analysis without shot noise correction!
    '''
    # calculate covariance matrix (with shotnoise; this is the correct one) 
    quij = quijoteBk('fiducial', rsd=rsd) # theta_fiducial 
    pks = quij['p0k1'] + 1e9/quij['Nhalos'][:,None] # uncorrect shotn oise 
    i_k = quij['k1']

    # impose k limit 
    _, _iuniq = np.unique(i_k, return_index=True)
    iuniq = np.zeros(len(i_k)).astype(bool) 
    iuniq[_iuniq] = True
    kf = 2.*np.pi/1000. # fundmaentla mode
    klim = (iuniq & (i_k*kf <= kmax)) 
    i_k = i_k[klim]

    C_fid = np.cov(pks[:,klim].T) 

    C_inv = np.linalg.inv(C_fid) # invert the covariance 
    
    dpk_dt = [] 
    for par in thetas: 
        # calculate the derivative of Pk (uncorrected for SN) along all the thetas 
        dpk_dti = quijote_dPk_SNuncorr(par, rsd=rsd, dmnu=dmnu)
        dpk_dt.append(dpk_dti[klim])
    
    Fij = Forecast.Fij(dpk_dt, C_inv) 
    return Fij 


def quijote_bkFisher_SNuncorr(kmax=0.5, rsd=True, dmnu='fin'): 
    ''' calculate fisher matrix for parameters ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
    '''
    # read in full covariance matrix (with shotnoise; this is the correct one) 
    fcov = h5py.File(os.path.join(UT.dat_dir(), 'bispectrum', 'quijote_Cov_full.hdf5'), 'r') 
    C_fid = fcov['C_bk'].value
    i_k, j_k, l_k = fcov['k1'].value, fcov['k2'].value, fcov['k3'].value 

    # impose k limit 
    kf = 2.*np.pi/1000. # fundmaentla mode
    klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) &  (l_k*kf <= kmax)) 
    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 
    C_fid = C_fid[:,klim][klim,:]

    C_inv = np.linalg.inv(C_fid) # invert the covariance 
    
    dbk_dt = [] 
    for par in thetas: # calculate the derivative of Bk along all the thetas 
        dbk_dti = quijote_dBk_SNuncorr(par, rsd=rsd, dmnu=dmnu)
        dbk_dt.append(dbk_dti[klim])
    
    Fij = Forecast.Fij(dbk_dt, C_inv) 
    return Fij 


def quijote_Forecast_SNuncorr(obs, kmax=0.5, rsd=True, dmnu='fin'):
    ''' fisher forecast for quijote 
    
    :param krange: (default: [0.01, 0.5]) 
        tuple specifying the kranges of k1, k2, k3 in the bispectrum
    '''
    # fisher matrix (Fij)
    if obs == 'pk': 
        Fij = quijote_pkFisher_SNuncorr(kmax=kmax, rsd=rsd, dmnu=dmnu) 
    elif obs == 'bk': 
        Fij = quijote_bkFisher_SNuncorr(kmax=kmax, rsd=rsd, dmnu=dmnu) 

    Finv = np.linalg.inv(Fij) # invert fisher matrix 
    print('sigma_Mnu = %f' % np.sqrt(Finv[-1,-1])) 

    fig = plt.figure(figsize=(17, 15))
    for i in xrange(ntheta-1): 
        for j in xrange(i+1, ntheta): 
            # sub inverse fisher matrix 
            Finv_sub = np.array([[Finv[i,i], Finv[i,j]], [Finv[j,i], Finv[j,j]]]) 

            theta_fid_i, theta_fid_j = theta_fid[thetas[i]], theta_fid[thetas[j]]
            sub = fig.add_subplot(ntheta-1, ntheta-1, (ntheta-1) * (j-1) + i + 1) 

            plotEllipse(Finv_sub, sub, theta_fid_ij=[theta_fid_i, theta_fid_j], color='C0')
            
            sub.set_xlim(theta_lims[i])
            sub.set_ylim(theta_lims[j])
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
    ffig = os.path.join(UT.fig_dir(), 'quijote_%sSNuncorr_Fisher_kmax%.2f%s.png' % (obs, kmax, ['_real', ''][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    return None

##################################################################
# qujiote fisher tests 
##################################################################
def quijote_Fisher_nmock(nmock, kmax=0.5, rsd=True, dmnu='fin'): 
    ''' calculate fisher matrix for parameters ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
    using nmock of the 15,000 quijote simulations to calculate the covariance 
    matrix.

    :param typ: 
        string that specifies the triangle shape. typ in ['equ', 'squ']. 

    :param kmax: (default: 0.5) 
        kmax 
    '''
    # read in full bispectrum 
    kf = 2.*np.pi/1000. # fundmaentla mode

    quij = quijoteBk('fiducial', rsd=rsd) # theta_fiducial 
    bks = quij['b123'] + quij['b_sn']
    bks = bks[:nmock, :]
    C_fid = np.cov(bks.T) 

    i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3']
    klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 
    C_fid = C_fid[:,klim][klim,:]

    C_inv = np.linalg.inv(C_fid) # invert the covariance 
    
    dbk_dt = [] 
    for par in thetas: # calculate the derivative of Bk along all the thetas 
        dbk_dti = quijote_dBk(par, rsd=rsd, dmnu=dmnu)
        dbk_dt.append(dbk_dti[klim])
    
    Fij = Forecast.Fij(dbk_dt, C_inv) 
    return Fij 


def quijote_forecast_nmock(kmax=0.5, rsd=True, dmnu='fin'):
    ''' fisher forecast where we compute the covariance matrix using different 
    number of mocks. 
    
    :param kmax: (default: 0.5) 
        k1, k2, k3 <= kmax
    '''
    nmocks = [1000, 3000, 5000, 7000, 9000, 11000, 13000, 15000]
    # read in fisher matrix (Fij)
    Finvs = [] 
    for nmock in nmocks: 
        Fij = quijote_Fisher_nmock(nmock, kmax=kmax, rsd=rsd, dmnu=dmnu)
        Finvs.append(np.linalg.inv(Fij)) # invert fisher matrix 

    fig = plt.figure(figsize=(6,6))
    sub = fig.add_subplot(111) 
    sub.plot([1000, 15000], [1., 1.], c='k', ls='--', lw=1) 
    sub.plot([1000, 15000], [0.9, 0.9], c='k', ls=':', lw=1) 
    for i in xrange(ntheta): 
        sig_theta = np.zeros(len(nmocks))
        for ik in range(len(nmocks)): 
            sig_theta[ik] = np.sqrt(Finvs[ik][i,i])
        sub.plot(nmocks, sig_theta/sig_theta[-1], label=r'$%s$' % theta_lbls[i]) 
        sub.set_xlim([3000, 15000]) 
    sub.legend(loc='lower right', fontsize=20) 
    sub.set_ylabel(r'$\sigma_\theta(N_{\rm fid})/\sigma_\theta(N_{\rm fid}=15,000)$', fontsize=25)
    sub.set_ylim([0.5, 1.1]) 
    sub.set_xlabel(r"$N_{\rm fid}$ Quijote realizations", labelpad=10, fontsize=25) 
    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_Fisher_nmocks_dmnu_%s_kmax%s%s.pdf' % 
            (dmnu, str(kmax).replace('.', ''), ['_real', ''][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_Fisher_nmocks_dmnu_%s_kmax%s%s.png' % 
            (dmnu, str(kmax).replace('.', ''), ['_real', ''][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def quijote_dBk_nmock(nmock, theta, rsd=True, dmnu='fin'):
    ''' calculate d B(k)/d theta using the paired and fixed quijote simulations
    run on perturbed theta 

    :param theta: 
        string that specifies the parameter to take the 
        derivative by. 
    '''
    # get all bispectrum covariances along theta  
    nmocks = [100, 200, 300, 400, 500]
    dBks = [] 
    if theta == 'Mnu': 
        Bks = [] # read in the bispectrum a Mnu+, Mnu++, Mnu+++
        for tt in ['fiducial', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp']: 
            quij = quijoteBk(tt, rsd=rsd)
            Bks.append(np.average(quij['b123'][:nmock,:], axis=0))
        Bk_fid, Bk_p, Bk_pp, Bk_ppp = Bks 

        # take the derivatives 
        h_p, h_pp, h_ppp = quijote_thetas['Mnu']
        if dmnu == 'p': 
            dBk = (Bk_p - Bk_fid) / h_p 
        elif dmnu == 'pp': 
            dBk = (Bk_pp - Bk_fid) / h_pp
        elif dmnu == 'ppp': 
            dBk = (Bk_ppp - Bk_fid) / h_ppp
        elif dmnu == 'fin':
            dBk = (-21 * Bk_fid + 32 * Bk_p - 12 * Bk_pp + Bk_ppp)/1.2 # finite difference coefficient
    else: 
        h = quijote_thetas[theta][1] - quijote_thetas[theta][0]
        
        quij = quijoteBk(theta+'_m', rsd=rsd)
        Bk_m = np.average(quij['b123'][:nmock,:], axis=0) # Covariance matrix tt- 
        quij = quijoteBk(theta+'_p', rsd=rsd)
        Bk_p = np.average(quij['b123'][:nmock,:], axis=0) # Covariance matrix tt+ 
        
        # take the derivatives 
        dBk = (Bk_p - Bk_m) / h 
    return dBk 


def quijote_Fisher_dBk_nmock(nmock, kmax=0.5, rsd=True, dmnu='fin'): 
    ''' calculate fisher matrix for parameters ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
    using nmock of the 15,000 quijote simulations to calculate the covariance 
    matrix.

    :param typ: 
        string that specifies the triangle shape. typ in ['equ', 'squ']. 

    :param kmax: (default: 0.5) 
        kmax 
    '''
    kf = 2.*np.pi/1000. # fundmaentla mode
    i_k, j_k, l_k, C_fid = quijoteCov(rsd=rsd)
    klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 
    C_fid = C_fid[:,klim][klim,:]
    
    C_inv = np.linalg.inv(C_fid) # invert the covariance 
    
    dbk_dt = [] 
    for par in thetas: # calculate the derivative of Bk along all the thetas 
        dbk_dti = quijote_dBk_nmock(nmock, par, rsd=rsd, dmnu=dmnu)
        dbk_dt.append(dbk_dti[klim])
    return Forecast.Fij(dbk_dt, C_inv) 


def quijote_forecast_dBk_nmock(kmax=0.5, rsd=True, dmnu='fin'):
    ''' fisher forecast where we compute the derivatives using different number of mocks. 
    
    :param kmax: (default: 0.5) 
    '''
    nmocks = [100, 200, 300, 400, 500]
    # read in fisher matrix (Fij)
    Finvs = [] 
    for nmock in nmocks: 
        Fij = quijote_Fisher_dBk_nmock(nmock, kmax=kmax, rsd=rsd, dmnu=dmnu)
        Finvs.append(np.linalg.inv(Fij)) # invert fisher matrix 

    fig = plt.figure(figsize=(6,6))
    sub = fig.add_subplot(111)
    sub.plot([100., 500.], [1., 1.], c='k', ls='--', lw=1) 
    sub.plot([100., 500.], [0.9, 0.9], c='k', ls=':', lw=1) 
    for i in xrange(ntheta): 
        sig_theta = np.zeros(len(nmocks))
        for ik in range(len(nmocks)): 
            sig_theta[ik] = np.sqrt(Finvs[ik][i,i])
        sub.plot(nmocks, sig_theta/sig_theta[-1], label=(r'$%s$' % theta_lbls[i]))
        sub.set_xlim([100, 500]) 
    sub.legend(loc='lower right', fontsize=20) 
    sub.set_ylabel(r'$\sigma_\theta(N_{\rm mock})/\sigma_\theta(N_{\rm mock}=500)$', fontsize=25)
    sub.set_ylim([0.5, 1.1]) 
    sub.set_xlabel(r"$N_{\rm mock}$ Quijote realizations", labelpad=10, fontsize=25) 
    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_Fisher_dBk_nmocks_dmnu_%s_kmax%s%s.pdf' % 
            (dmnu, str(kmax).replace('.', ''), ['_real', ''][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_Fisher_dBk_nmocks_dmnu_%s_kmax%s%s.png' % 
            (dmnu, str(kmax).replace('.', ''), ['_real', ''][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    return None

############################################################
def quijote_nbars(): 
    thetas = ['fiducial', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp', 'Ob_m', 'Ob_p', 'Om_m', 'Om_p', 'h_m', 'h_p', 'ns_m', 'ns_p', 's8_m', 's8_p']
    nbars = [] 
    for sub in thetas:
        quij = quijoteBk(sub)
        nbars.append(np.average(quij['Nhalos'])/1.e9)
        print sub, np.average(quij['Nhalos'])/1.e9
    print np.min(nbars)
    print np.min(nbars) * 1e9 
    print thetas[np.argmin(nbars)]
    return None


def quijote_pairfixed_test(kmax=0.5): 
    ''' compare the bispectrum of the fiducial and fiducial pair-fixed 
    '''
    quij_fid = quijoteBk('fiducial', rsd=True) 
    i_k, j_k, l_k = quij_fid['k1'], quij_fid['k2'], quij_fid['k3']
    bk_fid = np.average(quij_fid['b123'], axis=0) 
    quij_fid_ncv = quijoteBk('fiducial_NCV', rsd=True) 
    bk_fid_ncv = np.average(quij_fid_ncv['b123'], axis=0) 
    
    kf = 2.*np.pi/1000. 
    klim = ((i_k * kf <= kmax) & (j_k * kf <= kmax) & (l_k * kf <= kmax)) 

    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 
    ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # order of triangles 

    fig = plt.figure(figsize=(20,10)) 
    sub = fig.add_subplot(211)
    sub.plot(range(np.sum(klim)), bk_fid[klim][ijl], c='k', label='fiducial') 
    sub.plot(range(np.sum(klim)), bk_fid_ncv[klim][ijl], c='C1', ls="--", label='fiducial NCV') 
    sub.legend(loc='upper right', fontsize=20) 
    sub.set_xlabel('triangle configuration', fontsize=25) 
    sub.set_xlim(0, np.sum(klim))
    sub.set_ylabel('$B(k)$', fontsize=25)
    sub.set_yscale('log') 
    sub.set_ylim(1e5, 1e10)

    sub = fig.add_subplot(212)
    sub.plot(range(np.sum(klim)), bk_fid_ncv[klim][ijl]/bk_fid[klim][ijl], c='k' ) 
    sub.legend(loc='upper right', fontsize=20) 
    sub.set_xlabel('triangle configuration', fontsize=25) 
    sub.set_xlim(0, np.sum(klim))
    sub.set_ylabel(r'$B^{\rm fid;NCV}/B^{\rm fid}$', fontsize=25)
    sub.set_ylim(0.9, 1.1)

    ffig = os.path.join(UT.fig_dir(), 'quijote_pairfixed_test.kmax%.2f.png' % kmax) 
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


############################################################
# real vs rsd
############################################################
def compare_Pk_rsd(krange=[0.01, 0.5]):  
    ''' Compare the amplitude of the fiducial HADES and quijote bispectrum 
    in redshift space versus real space. 

    :param krange: (default: [0.01, 0.5]) 
        k-range of k1, k2, k3 
    '''
    kmin, kmax = krange # impose k range 
    kf = 2.*np.pi/1000. 

    fig = plt.figure(figsize=(5,5))
    sub = fig.add_subplot(111)

    hades_fid = hadesBk(0.0, nzbin=4, rsd=False) # fiducial bispectrum
    i_k = hades_fid['k1']
    Pk_fid = np.average(hades_fid['p0k1'], axis=0)
    hades_fid = hadesBk(0.0, nzbin=4, rsd=True) # fiducial bispectrum
    Pk_fid_rsd = np.average(hades_fid['p0k1'], axis=0)

    isort = np.argsort(i_k) 
    sub.plot(kf * i_k[isort], Pk_fid[isort], c='C0', label='real-space') 
    sub.plot(kf * i_k[isort], Pk_fid_rsd[isort], c='C1', label='redshift-space') 
    print('HADES', Pk_fid_rsd / Pk_fid) 

    quij_fid = quijoteBk('fiducial', rsd=False) # fiducial bispectrum
    i_k = quij_fid['k1']
    Pk_fid = np.average(quij_fid['p0k1'], axis=0)
    quij_fid = quijoteBk('fiducial', rsd=True) # fiducial bispectrum
    Pk_fid_rsd = np.average(quij_fid['p0k1'], axis=0)

    isort = np.argsort(i_k) 
    sub.plot(kf * i_k[isort], Pk_fid[isort], c='C0', ls=':', label='Quijote') 
    sub.plot(kf * i_k[isort], Pk_fid_rsd[isort], c='C1', ls=':') 
    print('Quijote', Pk_fid_rsd / Pk_fid)

    sub.legend(loc='lower left', fontsize=25) 
    sub.set_xlabel('$k$', labelpad=15, fontsize=25) 
    sub.set_xscale('log') 
    sub.set_xlim([1e-2, 1])
    sub.set_yscale('log') 
    sub.set_ylabel('$P(k_1)$', labelpad=10, fontsize=25) 
    krange_str = str(kmin).replace('.', '')+'_'+str(kmax).replace('.', '') 
    ffig = os.path.join(UT.doc_dir(), 'figs', 
            'haloPk_amp_%s_%s_rsd_comparison.pdf' % (str(kmin).replace('.', ''), str(kmax).replace('.', '')))
    fig.savefig(ffig, bbox_inches='tight') 
    ffig = os.path.join(UT.doc_dir(), 'figs', 
            'haloPk_amp_%s_%s_rsd_comparison.png' % (str(kmin).replace('.', ''), str(kmax).replace('.', '')))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def compare_Bk_rsd(kmax=0.5):  
    ''' Compare the amplitude of the fiducial HADES bispectrum in
    redshift space versus real space. 

    :param kmax: (default: 0.5) 
        k_max of k1, k2, k3 
    '''
    fig = plt.figure(figsize=(25,5))
    sub = fig.add_subplot(111)

    hades_fid = hadesBk(0.0, nzbin=4, rsd=False) # fiducial bispectrum
    Bk_fid = np.average(hades_fid['b123'], axis=0)
    hades_fid = hadesBk(0.0, nzbin=4, rsd=True) # fiducial bispectrum
    Bk_fid_rsd = np.average(hades_fid['b123'], axis=0) 

    i_k, j_k, l_k = hades_fid['k1'], hades_fid['k2'], hades_fid['k3']
    kf = 2.*np.pi/1000. 
    klim = ((i_k * kf <= kmax) & (j_k * kf <= kmax) & (l_k * kf <= kmax)) 

    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 
    ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # order of triangles 

    tri = np.arange(np.sum(klim))
    sub.plot(tri, Bk_fid[klim][ijl], c='C0', label='real-space') 
    sub.plot(tri, Bk_fid_rsd[klim][ijl], c='C1', label='redshift-space') 

    equ = ((i_k[ijl] == j_k[ijl]) & (l_k[ijl] == i_k[ijl]))
    sub.scatter(tri[equ], Bk_fid_rsd[klim][ijl][equ], marker='^', 
            facecolors='none', edgecolors='k', zorder=10, label='equilateral\ntriangles') 
    for iii, ii_k, x, y in zip(range(10000), i_k[ijl][equ][1::2], tri[equ][1::2], Bk_fid_rsd[klim][ijl][equ][1::2]):  
        if iii < 1: 
            sub.text(x, 1.2*y, '$k = %.2f$' % (ii_k * kf), ha='center', va='bottom', fontsize=15)
        elif iii < 6: 
            sub.text(x, 2.5*y, '$k = %.2f$' % (ii_k * kf), ha='center', va='bottom', fontsize=15)
        elif iii < 10: 
            sub.text(x, 1.5*y, '$k = %.2f$' % (ii_k * kf), ha='center', va='bottom', fontsize=15)

    quij_fid = quijoteBk('fiducial', rsd=False) # fiducial bispectrum
    _Bk_fid = np.average(quij_fid['b123'], axis=0)
    quij_fid = quijoteBk('fiducial', rsd=True) # fiducial bispectrum
    _Bk_fid_rsd = np.average(quij_fid['b123'], axis=0) 

    #sub.plot(tri, _Bk_fid[klim][ijl], c='C0', ls=':', label='Quijote') 
    #sub.plot(tri, _Bk_fid_rsd[klim][ijl], c='C1') 

    sub.legend(loc='upper right', handletextpad=0.5, markerscale=2, fontsize=22) 
    sub.set_yscale('log') 
    sub.set_xlim([0, np.sum(klim)])
    sub.set_ylim([1e5, 1e10]) 
    sub.set_xlabel('triangle configurations', labelpad=15, fontsize=25) 
    sub.set_ylabel('$B(k_1, k_2, k_3)$', labelpad=10, fontsize=25) 
    ffig = os.path.join(UT.doc_dir(), 'figs', 
            'haloBk_amp_kmax%s_rsd_comparison.pdf' % str(kmax).replace('.', ''))
    fig.savefig(ffig, bbox_inches='tight') 
    ffig = os.path.join(UT.doc_dir(), 'figs', 
            'haloBk_amp_kmax%s_rsd_comparison.png' % str(kmax).replace('.', ''))
    fig.savefig(ffig, bbox_inches='tight') 

    fig = plt.figure(figsize=(25,5))
    sub = fig.add_subplot(111)
    tri = np.arange(np.sum(klim))
    sub.plot(tri, Bk_fid_rsd[klim][ijl]/Bk_fid[klim][ijl], c='C0', label='HADES')
    sub.plot(tri, _Bk_fid_rsd[klim][ijl]/_Bk_fid[klim][ijl], c='C1', ls=':', label='Quijote')
    sub.set_xlim([0, np.sum(klim)])
    sub.set_ylim([1., 5.]) 
    sub.set_xlabel('triangle configurations', labelpad=15, fontsize=25) 
    sub.set_ylabel('$B^{(s)}/B$', labelpad=10, fontsize=25) 
    ffig = os.path.join(UT.doc_dir(), 'figs', 
            'haloBk_amp_kmax%s_rsd_ratio.png' % str(kmax).replace('.', ''))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def compare_Qk_rsd(krange=[0.01, 0.5]):  
    ''' Compare the amplitude of the fiducial HADES bispectrum in
    redshift space versus real space. 

    :param krange: (default: [0.01, 0.5]) 
        k-range of k1, k2, k3 
    '''
    hades_fid = hadesBk(0.0, nzbin=4, rsd=False) # fiducial bispectrum
    Qk_fid = np.average(hades_fid['q123'], axis=0)
    hades_fid = hadesBk(0.0, nzbin=4, rsd=True) # fiducial bispectrum
    Qk_fid_rsd = np.average(hades_fid['q123'], axis=0) 

    i_k, j_k, l_k = hades_fid['k1'], hades_fid['k2'], hades_fid['k3']
    kf = 2.*np.pi/1000. 
    kmin, kmax = krange # impose k range 
    klim = ((i_k * kf <= kmax) & (i_k * kf >= kmin) &
            (j_k * kf <= kmax) & (j_k * kf >= kmin) & 
            (l_k * kf <= kmax) & (l_k * kf >= kmin)) 

    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 

    ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # order of triangles 

    fig = plt.figure(figsize=(25,5))
    sub = fig.add_subplot(111)
    tri = np.arange(np.sum(klim))
    sub.plot(tri, Qk_fid[klim][ijl], c='C0', label='real-space') 
    sub.plot(tri, Qk_fid_rsd[klim][ijl], c='C1', label='redshift-space') 
    # mark the equilateral triangles 
    equ = ((i_k[ijl] == j_k[ijl]) & (l_k[ijl] == i_k[ijl]))
    sub.scatter(tri[equ], Qk_fid_rsd[klim][ijl][equ], c='k', zorder=10, label='equilateral') 
    for ii_k, x, y in zip(i_k[ijl][equ][::2], tri[equ][::2], Qk_fid_rsd[klim][ijl][equ][::2]):  
        sub.text(x, 0.9*y, '$k = %.2f$' % (ii_k * kf), ha='left', va='top', fontsize=15)

    quijote_fid = quijoteBk('fiducial', rsd=False) # fiducial bispectrum
    _Qk_fid = np.average(quijote_fid['q123'], axis=0)
    quijote_fid = quijoteBk('fiducial', rsd=True) # fiducial bispectrum
    _Qk_fid_rsd = np.average(quijote_fid['q123'], axis=0) 
    sub.plot(tri, _Qk_fid[klim][ijl], c='C0', ls=':', label='Quijote') 
    sub.plot(tri, _Qk_fid_rsd[klim][ijl], c='C1', ls=':') 
    
    sub.legend(loc='upper right', fontsize=25) 
    sub.set_xlim([0, np.sum(klim)])
    sub.set_ylim(0, 1) 
    sub.set_xlabel('triangle configurations', labelpad=15, fontsize=25) 
    sub.set_ylabel('$Q(k_1, k_2, k_3)$', labelpad=10, fontsize=25) 
    krange_str = str(kmin).replace('.', '')+'_'+str(kmax).replace('.', '') 
    ffig = os.path.join(UT.doc_dir(), 'figs', 
            'haloQk_amp_%s_%s_rsd_comparison.pdf' % (str(kmin).replace('.', ''), str(kmax).replace('.', '')))
    fig.savefig(ffig, bbox_inches='tight') 
    ffig = os.path.join(UT.doc_dir(), 'figs', 
            'haloQk_amp_%s_%s_rsd_comparison.png' % (str(kmin).replace('.', ''), str(kmax).replace('.', '')))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def compare_Bk_rsd_SNuncorr(krange=[0.01, 0.5]):  
    ''' Compare the amplitude of the fiducial HADES bispectrum in
    redshift space versus real space. 

    :param krange: (default: [0.01, 0.5]) 
        k-range of k1, k2, k3 
    '''
    hades_fid = hadesBk(0.0, nzbin=4, rsd=False) # fiducial bispectrum
    Bk_fid = np.average(hades_fid['b123'] + hades_fid['b_sn'], axis=0)
    hades_fid = hadesBk(0.0, nzbin=4, rsd=True) # fiducial bispectrum
    Bk_fid_rsd = np.average(hades_fid['b123'] + hades_fid['b_sn'], axis=0) 

    i_k, j_k, l_k = hades_fid['k1'], hades_fid['k2'], hades_fid['k3']
    kf = 2.*np.pi/1000. 
    kmin, kmax = krange # impose k range 
    klim = ((i_k * kf <= kmax) & (i_k * kf >= kmin) &
            (j_k * kf <= kmax) & (j_k * kf >= kmin) & 
            (l_k * kf <= kmax) & (l_k * kf >= kmin)) 

    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 

    ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # order of triangles 

    fig = plt.figure(figsize=(25,5))
    sub = fig.add_subplot(111)
    tri = np.arange(np.sum(klim))
    sub.plot(tri, Bk_fid[klim][ijl], c='C0', label='real-space') 
    sub.plot(tri, Bk_fid_rsd[klim][ijl], c='C1', label='redshift-space') 

    equ = ((i_k[ijl] == j_k[ijl]) & (l_k[ijl] == i_k[ijl]))
    sub.scatter(tri[equ], Bk_fid_rsd[klim][ijl][equ], c='k', zorder=10, label='equilateral') 
    for ii_k, x, y in zip(i_k[ijl][equ][::2], tri[equ][::2], Bk_fid_rsd[klim][ijl][equ][::2]):  
        sub.text(x, 1.1*y, '$k = %.2f$' % (ii_k * kf), ha='left', va='bottom', fontsize=15)

    sub.legend(loc='upper right', fontsize=25) 
    sub.set_yscale('log') 
    sub.set_xlim([0, np.sum(klim)])
    sub.set_ylim([5e7, 1e10]) 
    sub.set_xlabel('triangle configurations', labelpad=15, fontsize=25) 
    sub.set_ylabel('$B(k_1, k_2, k_3)$ shot noise uncorrected', labelpad=10, fontsize=25) 
    krange_str = str(kmin).replace('.', '')+'_'+str(kmax).replace('.', '') 
    ffig = os.path.join(UT.fig_dir(), 
            'haloBkSNuncorr_amp_%s_%s_rsd_comparison.png' % (str(kmin).replace('.', ''), str(kmax).replace('.', '')))
    fig.savefig(ffig, bbox_inches='tight') 

    fig = plt.figure(figsize=(25,5))
    sub = fig.add_subplot(111)
    tri = np.arange(np.sum(klim))
    sub.plot(tri, Bk_fid_rsd[klim][ijl]/Bk_fid[klim][ijl], c='C0')
    sub.set_xlim([0, np.sum(klim)])
    sub.set_ylim([1., 2.]) 
    sub.set_xlabel('triangle configurations', labelpad=15, fontsize=25) 
    sub.set_ylabel('$B^{(s)}/B$', labelpad=10, fontsize=25) 
    krange_str = str(kmin).replace('.', '')+'_'+str(kmax).replace('.', '') 
    ffig = os.path.join(UT.fig_dir(), 
            'haloBkSNuncorr_amp_%s_%s_rsd_ratio.png' % (str(kmin).replace('.', ''), str(kmax).replace('.', '')))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def compare_Qk_rsd_SNuncorr(krange=[0.01, 0.5]):  
    ''' Compare the amplitude of the fiducial HADES bispectrum in
    redshift space versus real space. 

    :param krange: (default: [0.01, 0.5]) 
        k-range of k1, k2, k3 
    '''
    hades_fid = hadesBk(0.0, nzbin=4, rsd=False) # fiducial bispectrum
    bk = np.average((hades_fid['b123'] + hades_fid['b_sn']), axis=0)
    pk1 = np.average(hades_fid['p0k1'] + 1e9/hades_fid['Nhalos'][:,None], axis=0)
    pk2 = np.average(hades_fid['p0k2'] + 1e9/hades_fid['Nhalos'][:,None], axis=0)
    pk3 = np.average(hades_fid['p0k3'] + 1e9/hades_fid['Nhalos'][:,None], axis=0)
    Qk_fid = bk / (pk1 * pk2 + pk1 * pk3 + pk2 * pk3)
    hades_fid = hadesBk(0.0, nzbin=4, rsd=True) # fiducial bispectrum
    bk = np.average((hades_fid['b123'] + hades_fid['b_sn']), axis=0)
    pk1 = np.average(hades_fid['p0k1'] + 1e9/hades_fid['Nhalos'][:,None], axis=0)
    pk2 = np.average(hades_fid['p0k2'] + 1e9/hades_fid['Nhalos'][:,None], axis=0)
    pk3 = np.average(hades_fid['p0k3'] + 1e9/hades_fid['Nhalos'][:,None], axis=0)
    Qk_fid_rsd = bk / (pk1 * pk2 + pk1 * pk3 + pk2 * pk3)

    i_k, j_k, l_k = hades_fid['k1'], hades_fid['k2'], hades_fid['k3']
    kf = 2.*np.pi/1000. 
    kmin, kmax = krange # impose k range 
    klim = ((i_k * kf <= kmax) & (i_k * kf >= kmin) &
            (j_k * kf <= kmax) & (j_k * kf >= kmin) & 
            (l_k * kf <= kmax) & (l_k * kf >= kmin)) 

    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 

    ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # order of triangles 

    fig = plt.figure(figsize=(25,5))
    sub = fig.add_subplot(111)
    tri = np.arange(np.sum(klim))
    sub.plot(tri, Qk_fid[klim][ijl], c='C0', label='real-space') 
    sub.plot(tri, Qk_fid_rsd[klim][ijl], c='C1', label='redshift-space') 
    
    # mark the equilateral triangles 
    equ = ((i_k[ijl] == j_k[ijl]) & (l_k[ijl] == i_k[ijl]))
    sub.scatter(tri[equ], Qk_fid_rsd[klim][ijl][equ], c='k', s=2, zorder=10, label='equilateral') 
    for ii_k, x, y in zip(i_k[ijl][equ][::2], tri[equ][::2], Qk_fid_rsd[klim][ijl][equ][::2]):  
        sub.text(x, 0.9*y, '$k = %.2f$' % (ii_k * kf), ha='left', va='top', fontsize=15)

    sub.legend(loc='upper right', fontsize=25) 
    sub.set_xlim([0, np.sum(klim)])
    sub.set_ylim(0, 1) 
    sub.set_xlabel('triangle configurations', labelpad=15, fontsize=25) 
    sub.set_ylabel('$Q(k_1, k_2, k_3)$ shot noise uncorrected', labelpad=10, fontsize=25) 
    krange_str = str(kmin).replace('.', '')+'_'+str(kmax).replace('.', '') 
    ffig = os.path.join(UT.fig_dir(), 
            'haloQkSNuncorr_amp_%s_%s_rsd_comparison.png' % (str(kmin).replace('.', ''), str(kmax).replace('.', '')))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def quijotePk_scalefactor(rsd=True, validate=False): 
    ''' we have to scale the amplitude of P(k) so that we can invert the joint
    P(k) + B(k) covariance matrix 
    '''
    # lets start with something stupid 
    quij = quijoteBk('fiducial', rsd=rsd)  
    i_k = quij['k1'] 
    mu_pk = np.average(quij['p0k1'], axis=0)    # get average quijote pk 
    mu_bk = np.average(quij['b123'], axis=0)    # get average quijote bk 

    # only keep 
    _, iuniq = np.unique(i_k, return_index=True) 
    med_pkamp = np.median(mu_pk[iuniq]) # median pk amplitude
    med_bkamp = np.median(mu_bk) # median bk amplitude 
    print('median P(k) amplitude = %.3e' % med_pkamp) 
    print('median B(k) amplitude = %.3e' % med_bkamp) 
    scalefactor = med_bkamp / med_pkamp 
    print('scale factor = %.3e' % scalefactor) 
    if validate: 
        fig = plt.figure(figsize=(20,5))
        sub = fig.add_subplot(111)
        sub.scatter(range(len(iuniq)), mu_pk[iuniq], c='C1', s=1, label='unscale $P_0(k)$') 
        sub.scatter(range(len(iuniq)), scalefactor * mu_pk[iuniq], c='C0', s=1, label='scaled $P_0(k)$') 
        sub.scatter(range(len(iuniq), len(iuniq)+len(mu_bk)), mu_bk, c='k', s=1, label='$B_0(k)$')
        sub.legend(loc='upper right', markerscale=10, handletextpad=0.25, fontsize=20) 
        sub.set_xlim(0, len(iuniq)+len(mu_bk)) 
        sub.set_yscale('log') 
        ffig = os.path.join(UT.fig_dir(), 'quijotePk_scalefactor%s.png' % ['_real', ''][rsd])
        fig.savefig(ffig, bbox_inches='tight') 
    return scalefactor  

def quijote_scaledpkbkCov(kmax=0.5, rsd=True): 
    ''' plot the covariance matrix of the quijote fiducial bispectrum and scaled
    fiducial powerspectrum. 
    '''
    # read in P(k) and B(k) 
    quij = quijoteBk('fiducial', rsd=rsd) # theta_fiducial 
    bks = quij['b123'] + quij['b_sn']                   # shotnoise uncorrected B(k) 
    pks = quij['p0k1'] + 1e9 / quij['Nhalos'][:,None]   # shotnoise uncorrected P(k) 
    i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3']
     
    # impose k limit on bispectrum
    kf = 2.*np.pi/1000. # fundmaentla mode
    bklim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax)) 
    ijl = UT.ijl_order(i_k[bklim], j_k[bklim], l_k[bklim], typ='GM') # order of triangles 
    bks = bks[:,bklim][:,ijl] 
    # impose k limit on powerspectrum 
    _, _iuniq = np.unique(i_k, return_index=True)
    iuniq = np.zeros(len(i_k)).astype(bool) 
    iuniq[_iuniq] = True
    pklim = (iuniq & (i_k*kf <= kmax)) 
    pks = pks[:,pklim]

    # powerspectrum amplitude has to be scaled 
    f_pks = quijotePk_scalefactor(rsd=rsd, validate=False)
    pks *= f_pks 
    
    pbks = np.concatenate([pks, bks], axis=1) # joint data vector

    C_pbk = np.cov(pbks.T) # covariance matrix 
    print('covariance matrix condition number = %.5e' % np.linalg.cond(C_pbk)) 

    # plot the covariance matrix 
    fig = plt.figure(figsize=(10,8))
    sub = fig.add_subplot(111)
    cm = sub.pcolormesh(C_pbk, norm=LogNorm(vmin=1e5, vmax=1e18))
    cbar = fig.colorbar(cm, ax=sub) 
    sub.set_title(r'Quijote $P(k)$ and $B(k_1, k_2, k_3)$ Covariance', fontsize=25)
    ffig = os.path.join(UT.doc_dir(), 'figs', 'quijote_scaledpbkCov_kmax%s%s.png' % 
            (str(kmax).replace('.', ''), ['_real', ''][rsd]))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def _quijote_pkbkCov_triangle(typ, krange=[0.01, 0.5]): 
    ''' plot the covariance matrix of the quijote fiducial bispectrum. 
    '''
    # read in P(k) and B(k) 
    quij = quijoteBk('fiducial') # theta_fiducial 
    bks = quij['b123'] + quij['b_sn']                   # shotnoise uncorrected B(k) 
    pks = quij['p0k1'] + 1e9 / quij['Nhalos'][:,None]   # shotnoise uncorrected P(k) 
    i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3']
     
    # impose k limit on bispectrum
    kmin, kmax = krange 
    kf = 2.*np.pi/1000. # fundmaentla mode
    # impose k limit 
    if typ == 'equ': # only equilateral 
        tri = (i_k == j_k) & (j_k == l_k) 
    elif typ == 'squ': # i_k >= j_k >= l_k (= 3*kf~0.01) 
        tri = (i_k == j_k) & (l_k == 3)
    bklim = (tri & 
            (i_k*kf <= kmax) & (i_k*kf >= kmin) &
            (j_k*kf <= kmax) & (j_k*kf >= kmin) & 
            (l_k*kf <= kmax) & (l_k*kf >= kmin)) 
    ijl = UT.ijl_order(i_k[bklim], j_k[bklim], l_k[bklim], typ='GM') # order of triangles 
    bks = bks[:,bklim][:,ijl] 
    # impose k limit on powerspectrum 
    _, _iuniq = np.unique(i_k, return_index=True)
    iuniq = np.zeros(len(i_k)).astype(bool) 
    iuniq[_iuniq] = True
    pklim = (iuniq & (i_k*kf <= kmax) & (i_k*kf >= kmin)) 
    pks = pks[:,pklim]
    
    pbks = np.concatenate([pks, bks], axis=1) # joint data vector

    C_pbk = np.cov(pbks.T) # covariance matrix 
    print('covariance matrix condition number = %.5e' % np.linalg.cond(C_pbk)) 

    # plot the covariance matrix 
    fig = plt.figure(figsize=(10,8))
    sub = fig.add_subplot(111)
    cm = sub.pcolormesh(C_pbk, norm=LogNorm(vmin=1e5, vmax=1e18))
    cbar = fig.colorbar(cm, ax=sub) 
    sub.set_title(r'Quijote $P(k)$ and $B(k_1, k_2, k_3)$ Covariance', fontsize=25)
    ffig = os.path.join(UT.fig_dir(), 'quijote_pbk_Cov_%s_%s_%s.png' % 
            (typ, str(kmin).replace('.', ''), str(kmax).replace('.', '')))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 



if __name__=="__main__": 
    # covariance matrices
    for kmax in [0.5]: 
        for rsd in [True]:#, False]: 
            continue 
            quijote_pkCov(kmax=kmax, rsd=rsd) 
            quijote_bkCov(kmax=kmax, rsd=rsd) # condition number 1.73518e+08
            quijote_pkbkCov(kmax=kmax, rsd=rsd) 
            quijote_scaledpkbkCov(kmax=kmax, rsd=rsd) # condition number 3.26234e+11
    
    # deriatives 
    #quijote_dPdthetas(dmnu='fin')
    #quijote_dPdthetas(dmnu='p')
    #quijote_dBdthetas(dmnu='fin')
    #quijote_dBdthetas(dmnu='p')
    quijote_Bratio_thetas(kmax=0.5)
    #quijote_B_relative_error(kmax=0.5)
    #for tt in ['s8', 'Mnu', 'Mmin']: 
    #    quijote_P_theta(tt)

    # fisher forecasts 
    for rsd in [True]: #, False]: 
        for kmax in [0.5]: #[0.2, 0.3, 0.4, 0.5]: 
            for dmnu in ['fin']: #['p', 'pp', 'ppp', 'fin']: 
                continue 
                quijote_Forecast('pk', kmax=kmax, rsd=rsd, dmnu=dmnu)
                quijote_Forecast('bk', kmax=kmax, rsd=rsd, dmnu=dmnu)
                quijote_pbkForecast(kmax=kmax, rsd=rsd, dmnu=dmnu)
        #quijote_Forecast_kmax('pk', rsd=rsd) 
        #quijote_Forecast_kmax('bk', rsd=rsd)
        #quijote_pkForecast_dmnu(rsd=rsd)
        #quijote_bkForecast_dmnu(rsd=rsd)
    # rsd 
    #compare_Pk_rsd(krange=[0.01, 0.5])
    #compare_Bk_rsd(kmax=0.5)
    #compare_Qk_rsd(krange=[0.01, 0.5])
    
    #hades_dchi2(krange=[0.01, 0.5])
    
    # amplitude scaling factor is a free parameter
    for kmax in [0.2, 0.5]: 
        continue 
        print('kmax = %.1f' % kmax) 
        quijote_Forecast_freeScale('pk', kmax=kmax, rsd=True, dmnu='fin')
        quijote_Forecast_freeScale('bk', kmax=kmax, rsd=True, dmnu='fin')
        #quijote_Forecast_freeScale('bk_equ', kmax=kmax, rsd=True, dmnu='fin')
        #quijote_Forecast_freeScale('bk_squ', kmax=kmax, rsd=True, dmnu='fin')
        #quijote_Forecast_freeScale('bk_equ', kmax=kmax, rsd=False, dmnu='fin')
        #quijote_Forecast_freeScale('bk_squ', kmax=kmax, rsd=False, dmnu='fin')

    # Mmin is a free parameter
    for kmax in [0.2, 0.5]: 
        continue 
        quijote_Forecast_freeMmin('pk', kmax=kmax, rsd=True, dmnu='fin')
        print('kmax = %.1f' % kmax) 
        quijote_dbk_dMnu_dMmin(kmax=kmax, rsd=True, dmnu='fin')
        quijote_Forecast_freeMmin('bk', kmax=kmax, rsd=True, dmnu='fin')
    
    # fixed nbar forecasts 
    #quijote_bkCov_fixednbar(kmax=0.5, rsd=True)
    for kmax in [0.5]: 
        continue 
        print('kmax = %.1f' % kmax) 
        quijote_fixednbar_derivative_comparison('pk', kmax=kmax)
        quijote_fixednbar_derivative_comparison('bk', kmax=kmax)
        quijote_Forecast_fixednbar('pk', kmax=kmax, dmnu='fin')
        quijote_Forecast_fixednbar('bk', kmax=kmax, dmnu='fin')


    # SN uncorrected forecasts 
    #compare_Bk_rsd_SNuncorr(krange=[0.01, 0.5])
    #compare_Qk_rsd_SNuncorr(krange=[0.01, 0.5])
    #for rsd in [True, False]:  
    #    quijote_Forecast_SNuncorr('pk', kmax=0.5, rsd=rsd, dmnu='fin')
    #    quijote_Forecast_SNuncorr('bk', kmax=0.5, rsd=rsd, dmnu='fin')
    #quijote_nbars()

    # quijote pair fixed test  
    #quijote_pairfixed_test(kmax=0.5)
        
    # kmax test
    #for k in [0.2, 0.3, 0.4]: 
    #    B123_kmax_test(kmin=0.01, kmax1=k, kmax2=k+0.1, rsd=True)

    # usingly only specific triangles
    #compare_Bk_triangles([[30, 18], [18, 18], [12,9]], rsd=True)

    # equilateral triangles 
    #for shape in ['equ', 'squ']: 
    #    compare_Bk_triangle(shape, rsd=True)
    #quijote_forecast_triangle_kmax('equ')
    #quijote_pkbkCov_triangle('equ', krange=[0.01, 0.5])
    
    # convergence tests 
    #quijote_forecast_nmock(kmax=0.5, rsd=True)
    #quijote_forecast_dBk_nmock(kmax=0.5, rsd=True)

    #quijotePk_scalefactor(rsd=True, validate=True)