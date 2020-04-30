''' 

script for calculating the derivatives for wp and xim measurements of quijote
hod catalogs that Jeremy made 

'''
import os 
import h5py 
import numpy as np 
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

if 'machine' in os.environ and os.environ['machine'] == 'mbp': 
    dat_dir = '/Users/ChangHoon/data/emanu/jlt_wp_xi/'
else: 
    raise ValueError


thetas = ['Om', 'Ob2', 'h', 'ns', 's8', 'Mnu', 'logMmin', 'sigma_logM', 'alpha', 'logM1'] 

theta_lbls = [r'$\Omega_m$', r'$\Omega_b$', '$h$', '$n_s$', r'$\sigma_8$', r'$M_\nu$', 
        r'$\log M_{\rm min}$', r'$\sigma_{\log M}$', r'$\alpha$', r'$\log M_1$'] 

theta_fid = {'Mnu': 0., 'Ob': 0.049, 'Ob2': 0.049, 'Om': 0.3175, 'h': 0.6711,  'ns': 0.9624,  's8': 0.834, 
        'logMmin': 13.65, 'sigma_logM': 0.2, 'logM0': 14.0, 'alpha': 1.1, 'logM1': 14.0} # fiducial theta 


def forecast(obs):
    ''' fisher forecast  
    '''
    Fij = FisherMatrix(obs) # fisher matrix (Fij)

    cond = np.linalg.cond(Fij)
    if cond > 1e16: print('Fij is ill-conditioned %.5e' % cond)

    Finv    = np.linalg.inv(Fij) # invert fisher matrix 
    print('sigmas %s' % ', '.join(['%.2e' % sii for sii in np.sqrt(np.diag(Finv))]))

    # x, y ranges
    dlim = [0.1, 0.06, 0.5, 0.5, 0.15, 0.75, 0.2, 0.4, 0.2, 0.2]

    fig = plt.figure(figsize=(17, 15))
    for i in range(len(thetas)+1): 
        for j in range(i+1, len(thetas)): 
            # sub inverse fisher matrix 
            Finv_sub = np.array([[Finv[i,i], Finv[i,j]], [Finv[j,i], Finv[j,j]]]) 
            # plot the ellipse
            sub = fig.add_subplot(len(thetas)-1, len(thetas)-1, (len(thetas)-1) * (j-1) + i + 1) 

            Forecast.plotEllipse(Finv_sub, sub, 
                    theta_fid_ij=[theta_fid[thetas[i]], theta_fid[thetas[j]]], color='C0')

            sub.set_xlim(theta_fid[thetas[i]] - dlim[i], theta_fid[thetas[i]] + dlim[i])
            sub.set_ylim(theta_fid[thetas[j]] - dlim[j], theta_fid[thetas[j]] + dlim[j])
            if i == 0:   
                sub.set_ylabel(theta_lbls[j], fontsize=20) 
            else: 
                sub.set_yticks([])
                sub.set_yticklabels([])
            
            if j == len(thetas)-1: 
                sub.set_xlabel(theta_lbls[i], labelpad=10, fontsize=20) 
            else: 
                sub.set_xticks([])
                sub.set_xticklabels([]) 

    fig.subplots_adjust(wspace=0.05, hspace=0.05) 
    ffig = os.path.join(dat_dir, '%s.forecast.png' % obs)
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def plot_FisherMatrix(obs): 
    ''' plot the Fisher matrix
    '''
    Fij = FisherMatrix(obs)

    theta_lbls = [r'$\Omega_m$', r'$\Omega_b$', '$h$', '$n_s$', r'$\sigma_8$',
            r'$\log M_{\rm min}$', r'$\sigma_{\log M}$', r'$\alpha$', 
            r'$\log M_1$'] 
    
    # plot the  fisher matrix 
    fig = plt.figure(figsize=(20,7))
    sub = fig.add_subplot(121)
    cm = sub.pcolormesh(Fij)#, vmin=-0.025, vmax=1.) 
    cbar = fig.colorbar(cm, ax=sub) 

    sub.set_xticks(np.arange(len(theta_lbls))+0.5) 
    sub.set_xticklabels(theta_lbls) 
    sub.set_yticks(np.arange(len(theta_lbls))+0.5) 
    sub.set_yticklabels(theta_lbls) 
    
    if obs == 'wp': 
        obs_lbl = '$w_p$'
    elif obs == 'xi': 
        obs_lbl = r'$\xi_0, \xi_2$'
    elif obs == 'wpxi':
        obs_lbl = r'$w_p, \xi_0, \xi_2$'
    cbar.set_label(r'%s fisher matrix' % obs_lbl, fontsize=25, labelpad=10, rotation=90)

    ffig = os.path.join(dat_dir, 'fij_%s.png' % obs)
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


# Fisher Matrix
def FisherMatrix(obs): 
    ''' calculate fisher matrix 
    '''
    # calculate covariance matrix (with shotnoise; this is the correct one) 
    # *** rsd and flag kwargs are ignored for the covariace matirx ***
    if obs == 'wp': 
        C_fid, nmock = cov_wp()
        dobsdt = dwpdtheta
    elif obs == 'xi': 
        C_fid, nmock = cov_xi()
        dobsdt = dxidtheta
    elif obs == 'wpxi': 
        C_fid, nmock = cov_wpxi()
        dobsdt = dwpxidtheta
    else: 
        raise NotImplementedError
    print('covariance matrix from %i mocks' % nmock)
    print('     condition number %e' % np.linalg.cond(C_fid))
    
    ndata = C_fid.shape[0] 
    f_hartlap = float(nmock - ndata - 2)/float(nmock - 1) 
    C_inv = f_hartlap * np.linalg.inv(C_fid) # invert the covariance 
    
    # calculate the derivatives along all the thetas 

    dobs_dt = [] 
    for par in thetas: 
        _, dobsdt_i = dobsdt(par)
        dobs_dt.append(dobsdt_i) 
            
    Fij = Forecast.Fij(dobs_dt, C_inv) 
    return Fij 


def plot_dwpxi(): 
    ''' compare wp, xi0, and xi2 derivatives w.r.t. the different thetas
    '''
    # y range of P0, P2 plots 
    #logplims = [(-10., 5.), (-2, 15), (-3., 1.), (-4., 0.), (-2., 2.), (-0.5, 0.5), (-2., 4), (-2., 1.), None, (-1., 2.), (-5., 1.)] 
    #logblims = [(-13., -6.), (-2., 24.), (-5., -0.5), (-5., -0.5), (-2., 2.), (0.2, 0.7), (3., 6.), None, None, (2, 6), None] 

    fig = plt.figure(figsize=(15,3*len(thetas)))
    for i, theta in enumerate(thetas): 
        r, dwp = dwpdtheta(theta)
        
        # plot dwp/dtheta
        sub = fig.add_subplot(len(thetas), 3, 3*i+1)
        sub.plot(r, dwp, c='C0')
        sub.set_xscale('log') 
        sub.set_xlim(0.1, 30.) 
        if i != len(thetas)-1: sub.set_xticklabels([]) 
        sub.set_yscale('symlog', linthreshy=1e3) 
        #sub.set_ylim(logplims[i]) 
        if i == 4: sub.set_ylabel(r'${\rm d} w_{\rm p}/{\rm d}\theta$', fontsize=25) 
        
        r, dxis = dxidtheta(theta)
        dxi0 = dxis[:len(dxis)//2]
        dxi2 = dxis[len(dxis)//2:]

        # plot dB/dtheta
        sub = fig.add_subplot(len(thetas), 3, 3*i+2)
        sub.plot(r, dxi0)
        sub.set_xscale('log') 
        sub.set_xlim(0.1, 30.) 
        if i != len(thetas)-1: sub.set_xticklabels([]) 
        else: sub.set_xlabel('$r$', fontsize=25) 
        sub.set_yscale('symlog', linthreshy=1e3) 
        #sub.set_ylim(logblims[i]) 
        if i == 4: sub.set_ylabel(r'${\rm d} \xi_0/{\rm d}\theta$', fontsize=25) 

        sub = fig.add_subplot(len(thetas), 3, 3*i+3)
        sub.plot(r, dxi2)
        sub.set_xscale('log') 
        sub.set_xlim(0.1, 30.) 
        sub.set_yscale('symlog', linthreshy=1e3) 
        #sub.set_ylim(logblims[i]) 
        if i == 4: sub.set_ylabel(r'${\rm d} \xi_2/{\rm d}\theta$', fontsize=25) 

        sub.text(0.9, 0.9, theta_lbls[i], ha='right', va='top', 
                transform=sub.transAxes, fontsize=25,
                bbox=dict(facecolor='white', alpha=0.75, edgecolor='None')) 
        if i != len(thetas)-1: sub.set_xticklabels([]) 

    ffig = os.path.join(dat_dir, 'dwpxidtheta.png')
    fig.subplots_adjust(wspace=0.5, hspace=0.02) 
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def plot_cov(): 
    ''' compare covariance of wp, xi0, and xi2
    '''
    C_wp, _ = cov_wp()
    C_xi, _ = cov_xi()
    C_wpxi, _ = cov_wpxi() 
    print('condition number for C_wp: %e' % np.linalg.cond(C_wp))
    print('condition number for C_xi: %e' % np.linalg.cond(C_xi))
    print('condition number for C_wpxi: %e' % np.linalg.cond(C_wpxi))
    
    Ccorr_wp = ((C_wp / np.sqrt(np.diag(C_wp))).T / np.sqrt(np.diag(C_wp))).T
    Ccorr_xi = ((C_xi / np.sqrt(np.diag(C_xi))).T / np.sqrt(np.diag(C_xi))).T
    Ccorr_wpxi = ((C_wpxi / np.sqrt(np.diag(C_wpxi))).T / np.sqrt(np.diag(C_wpxi))).T

    # plot the covariance matrix 
    fig = plt.figure(figsize=(30,7))
    sub = fig.add_subplot(131)
    cm = sub.pcolormesh(Ccorr_wp, vmin=-0.025, vmax=1.) 
    cbar = fig.colorbar(cm, ax=sub) 
    cbar.set_label(r'$w_p$ correlation matrix', fontsize=25, labelpad=10, rotation=90)

    sub = fig.add_subplot(132)
    cm = sub.pcolormesh(Ccorr_xi, vmin=-0.025, vmax=1.) 
    cbar = fig.colorbar(cm, ax=sub) 
    cbar.set_label(r'$\xi_0, \xi_2$ correlation matrix', fontsize=25, labelpad=10, rotation=90)
    
    sub = fig.add_subplot(133)
    cm = sub.pcolormesh(Ccorr_wpxi, vmin=-0.025, vmax=1.) 
    cbar = fig.colorbar(cm, ax=sub) 
    cbar.set_label(r'$w_p, \xi_0, \xi_2$ correlation matrix', fontsize=25, labelpad=10, rotation=90)

    ffig = os.path.join(dat_dir, 'cov_wpxi.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def cov_wpxi(): 
    ''' covariance of wp and xi combined
    '''
    quij    = quijhod_wp('fiducial')
    wps     = quij['wp']
    
    quij    = quijhod_xi('fiducial')
    xi0     = quij['xi0']
    xi2     = quij['xi2']
    
    data_vector = np.concatenate([wps, xi0, xi2], axis=1)

    nmock = data_vector.shape[0]
    cov = np.cov(data_vector.T) # calculate the covariance
    return cov, nmock


def cov_wp(): 
    ''' calculate covariance of wp 
    '''
    quij    = quijhod_wp('fiducial')
    wps     = quij['wp']
    nmock   = wps.shape[0] 
    cov     = np.cov(wps.T) # calculate the covariance
    return cov, nmock


def cov_xi(): 
    ''' calculate covariance of wp 
    '''
    quij = quijhod_xi('fiducial')
    xi0 = quij['xi0']
    xi2 = quij['xi2']
    
    xis = np.concatenate([xi0, xi2], axis=1)

    nmock = xis.shape[0]
    cov = np.cov(xis.T) # calculate the covariance
    return cov, nmock


def dwpxidtheta(theta): 
    ''' derivative of [wp, xi0, xi2] w.r.t. theta 
    ''' 
    r0, dwp = dwpdtheta(theta)
    r1, dxi = dxidtheta(theta) 

    return r0, np.concatenate([dwp, dxi]) 


def dwpdtheta(theta): 
    ''' derivative of wp w.r.t theta  
    '''
    quijote_thetas = {
            'Mnu': [0.1, 0.2, 0.4], # +, ++, +++ 
            'Ob': [0.048, 0.050],   # others are - + 
            'Ob2': [0.047, 0.051],   # others are - + 
            'Om': [0.3075, 0.3275],
            'h': [0.6511, 0.6911],
            'ns': [0.9424, 0.9824],
            's8': [0.819, 0.849], 
            'logMmin': [13.6, 13.7], 
            'sigma_logM': [0.18, 0.22], 
            'logM0': [13.8, 14.2],
            'alpha': [0.9, 1.3], 
            'logM1': [13.8, 14.2]}

    if theta == 'Mnu': 
        # this is wrong. It should fiducial_ZA 
        steps = ['fiducial', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp'] 
        coeffs = [-21., 32., -12., 1.] # finite difference coefficient
        h = 1.2
    else: 
        steps = ['%s_%s' % (theta, step) for step in ['m', 'p']]
        coeffs = [-1, 1.]
        h = quijote_thetas[theta][1] - quijote_thetas[theta][0] 

    for i, step, coeff in zip(range(len(steps)), steps, coeffs): 
        quij = quijhod_wp(step)

        if i == 0: 
            dwp = np.zeros(quij['wp'].shape[1])
            r   = np.zeros(quij['r'].shape[1])

        dwp += coeff * np.average(quij['wp'], axis=0) 
        r   += np.average(quij['r'], axis=0) 
    
    dwp  /= h 
    r   /= float(len(steps))
    return r, dwp 


def dxidtheta(theta): 
    ''' derivative of wp w.r.t theta  
    '''
    quijote_thetas = {
            'Mnu': [0.1, 0.2, 0.4], # +, ++, +++ 
            'Ob': [0.048, 0.050],   # others are - + 
            'Ob2': [0.047, 0.051],   # others are - + 
            'Om': [0.3075, 0.3275],
            'h': [0.6511, 0.6911],
            'ns': [0.9424, 0.9824],
            's8': [0.819, 0.849], 
            'logMmin': [13.6, 13.7], 
            'sigma_logM': [0.18, 0.22], 
            'logM0': [13.8, 14.2],
            'alpha': [0.9, 1.3], 
            'logM1': [13.8, 14.2]}

    if theta == 'Mnu': 
        # this is wrong. It should fiducial_ZA 
        steps = ['fiducial', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp'] 
        coeffs = [-21., 32., -12., 1.] # finite difference coefficient
        h = 1.2
    else: 
        steps = ['%s_%s' % (theta, step) for step in ['m', 'p']]
        coeffs = [-1, 1.]
        h = quijote_thetas[theta][1] - quijote_thetas[theta][0] 

    for i, step, coeff in zip(range(len(steps)), steps, coeffs): 
        quij = quijhod_xi(step)

        if i == 0: 
            r       = np.zeros(quij['r'].shape[1])
            dxi0    = np.zeros(quij['xi0'].shape[1])
            dxi2    = np.zeros(quij['xi2'].shape[1])

        r       += np.average(quij['r'], axis=0) 
        dxi0    += coeff * np.average(quij['xi0'], axis=0) 
        dxi2    += coeff * np.average(quij['xi2'], axis=0) 
    
    r       /= float(len(steps))
    dxi0    /= h 
    dxi2    /= h 

    dxis = np.concatenate([dxi0, dxi2])
    return r, dxis


def quijhod_wp(theta): 
    ''' read wp at given `theta` from the directory `dat_dir` 
    '''
    if theta in ['alpha_m', 'alpha_p', 'logM1_m', 'logM1_p', 
            'logMmin_m', 'logMmin_p', 'sigma_logM_m', 'sigma_logM_p']: 
        # hod parameters
        fwp = os.path.join(dat_dir, 'wp_fiducial_%s.hdf5' % theta) 
    else: 
        fwp = os.path.join(dat_dir, 'wp_%s.hdf5' % theta) 
    print(fwp)     
    wps = h5py.File(fwp, 'r') 

    out = {}
    for k in wps.keys(): 
        out[k] = wps[k][...]
    return out


def quijhod_xi(theta): 
    ''' read xi multipoles at given `theta` from the directory `dat_dir` 
    '''
    if theta in ['alpha_m', 'alpha_p', 'logM1_m', 'logM1_p', 
            'logMmin_m', 'logMmin_p', 'sigma_logM_m', 'sigma_logM_p']: 
        # hod parameters
        fxi = os.path.join(dat_dir, 'xi_fiducial_%s.hdf5' % theta) 
    else: 
        fxi = os.path.join(dat_dir, 'xi_%s.hdf5' % theta) 
    print(fxi)    
    xim = h5py.File(fxi, 'r') 

    out = {}
    for k in xim.keys(): 
        out[k] = xim[k][...]
    return out


if __name__=='__main__': 
    plot_cov()
    #plot_dwpxi()
    #plot_FisherMatrix('wp')
    #plot_FisherMatrix('xi')
    plot_FisherMatrix('wpxi')
    #forecast('wp')
    #forecast('xi')
    forecast('wpxi')
