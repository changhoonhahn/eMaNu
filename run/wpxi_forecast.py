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


def wpxi(): 
    ''' plot wp and xi0 + xi2 for fiducial value
    '''
    fig = plt.figure(figsize=(14,4))
    sub0 = fig.add_subplot(131) 
    sub0.set_xlabel(r'$r$', fontsize=25) 
    sub0.set_xscale('log') 
    sub0.set_xlim(1e-1, 3e1) 
    sub0.set_ylabel(r'$w_p(r)$', fontsize=25) 
    sub0.set_yscale('log') 

    sub1 = fig.add_subplot(132) 
    sub1.set_xlabel(r'$r$', fontsize=25) 
    sub1.set_xscale('log') 
    sub1.set_xlim(1e-1, 3e1) 
    sub1.set_ylabel(r'$r^2\xi_0(r)$', fontsize=25) 
    sub1.set_ylim(-200, 400) 

    sub2 = fig.add_subplot(133) 
    sub2.set_xlabel(r'$r$', fontsize=25) 
    sub2.set_xscale('log') 
    sub2.set_xlim(1e-1, 3e1) 
    sub2.set_ylabel(r'$r^2\xi_2(r)$', fontsize=25) 
    sub2.set_ylim(-200, 400) 

    for i, theta in enumerate(thetas+['fiducial']): 
        if theta == 'fiducial': 
            wp = quijhod_wp(theta)
            xi = quijhod_xi(theta) 
        
            sub0.errorbar(
                    np.average(wp['r'], axis=0), 
                    np.average(wp['wp'], axis=0), 
                    yerr=np.average(wp['err_wp'], axis=0), 
                    fmt='.k') 
            
            sub1.errorbar(
                    np.average(xi['r'], axis=0), 
                    np.average(xi['r']**2 * xi['xi0'], axis=0), 
                    yerr=np.average(xi['r']**2 * xi['err_xi0'], axis=0),
                    fmt='.k') 

            sub2.errorbar(
                    np.average(xi['r'], axis=0), 
                    np.average(xi['r']**2 * xi['xi2'], axis=0), 
                    yerr=np.average(xi['r']**2 * xi['err_xi2'], axis=0),
                    fmt='.k') 
        else: 
            if theta != 'Mnu': 
                steps = ['m', 'p']
            else: 
                steps = ['p', 'pp', 'ppp'] 
            lstyles = ['-', '--', ':']
            for ii, step in enumerate(steps): 
                wp = quijhod_wp('%s_%s' % (theta, step))
                xi = quijhod_xi('%s_%s' % (theta, step)) 

                sub0.plot(np.average(wp['r'], axis=0), 
                        np.average(wp['wp'], axis=0), 
                        c='C%i' % i, ls=lstyles[ii], lw=0.5)
                sub1.plot(np.average(xi['r'], axis=0), 
                        np.average(xi['r']**2 * xi['xi0'], axis=0), 
                        c='C%i' % i, ls=lstyles[ii], lw=0.5)
                sub2.plot(np.average(xi['r'], axis=0), 
                        np.average(xi['r']**2 * xi['xi2'], axis=0), 
                        c='C%i' % i, ls=lstyles[ii], lw=0.5)
                if ii == 0: 
                    sub2.plot([0], [0], label=theta_lbls[i])
    sub2.legend(loc='lower left', handletextpad=0.1, ncol=3)
    fig.subplots_adjust(wspace=0.4) 
    ffig = os.path.join(dat_dir, 'wpxi.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def _wpxi_pm(theta): 
    ''' plot wp and xi0 + xi2
    '''
    fig = plt.figure(figsize=(14,4))
    sub0 = fig.add_subplot(131) 
    sub0.set_xlabel(r'$r$', fontsize=25) 
    sub0.set_xscale('log') 
    sub0.set_xlim(1e-1, 3e1) 
    sub0.set_ylabel(r'$w_p(r)$', fontsize=25) 
    sub0.set_yscale('log') 

    sub1 = fig.add_subplot(132) 
    sub1.set_xlabel(r'$r$', fontsize=25) 
    sub1.set_xscale('log') 
    sub1.set_xlim(1e-1, 3e1) 
    sub1.set_ylabel(r'$r^2\xi_0(r)$', fontsize=25) 
    sub1.set_ylim(-200, 400) 
    #sub1.set_ylabel(r'$\xi_0(r)$', fontsize=25) 
    #sub1.set_yscale('log')

    sub2 = fig.add_subplot(133) 
    sub2.set_xlabel(r'$r$', fontsize=25) 
    sub2.set_xscale('log') 
    sub2.set_xlim(1e-1, 3e1) 
    sub2.set_ylabel(r'$r^2\xi_2(r)$', fontsize=25) 
    sub2.set_ylim(-200, 400) 
    #sub2.set_ylabel(r'$\xi_2(r)$', fontsize=25) 
    #sub2.set_yscale('log')

    for i, _theta in enumerate([theta, 'fiducial']): 
        if _theta  == 'fiducial': 
            steps = ['']
            clr = 'k'
        elif _theta != 'Mnu': 
            steps = ['_m', '_p']
            clr = 'C0'
        else: 
            steps = ['_p', '_pp', '_ppp'] 
            clr = 'C0'
        lstyles = ['-', '--', ':']
    
        wps, xis = [], [] 
        for ii, step in enumerate(steps): 
            wp = quijhod_wp('%s%s' % (_theta, step))
            xi = quijhod_xi('%s%s' % (_theta, step)) 

            sub0.plot(np.average(wp['r'], axis=0), 
                    np.average(wp['wp'], axis=0), 
                    c=clr, ls=lstyles[ii], lw=0.5)
            sub1.plot(np.average(xi['r'], axis=0), 
                    np.average(xi['r']**2 * xi['xi0'], axis=0), 
                    #np.average(xi['xi0'], axis=0), 
                    c=clr, ls=lstyles[ii], lw=0.5)
            sub2.plot(np.average(xi['r'], axis=0), 
                    np.average(xi['r']**2 * xi['xi2'], axis=0), 
                    #np.average(xi['xi2'], axis=0), 
                    c=clr, ls=lstyles[ii], lw=0.5)
            if ii == 0 and _theta != 'fiducial': 
                sub2.plot([0], [0], c=clr, label=theta_lbls[thetas.index(_theta)])

            wps.append(wp)
            xis.append(xi) 

        if _theta not in ['fiducial', 'Mnu']: 
            sub0.plot(np.average(wps[0]['r'], axis=0), 
                    np.average(wps[1]['wp'] - wps[0]['wp'], axis=0), 
                    c='C1', ls=lstyles[ii], lw=0.5)
            sub1.plot(np.average(xis[0]['r'], axis=0), 
                    np.average(xis[1]['r']**2 * xis[1]['xi0'] - xis[0]['r']**2 * xis[0]['xi0'], axis=0), 
                    #np.average(xi['xi0'], axis=0), 
                    c='C1', ls=lstyles[ii], lw=0.5)
            sub2.plot(np.average(xi['r'], axis=0), 
                    np.average(xis[1]['r']**2 * xis[1]['xi2'] - xis[0]['r']**2 * xis[0]['xi2'], axis=0), 
                    #np.average(xi['xi2'], axis=0), 
                    c='C1', ls=lstyles[ii], lw=0.5)
            sub2.plot([0], [0], c='C1', label=r'$\theta^+ - \theta^-$')

    sub2.legend(loc='lower left', handletextpad=0.1, ncol=3)
    fig.subplots_adjust(wspace=0.4) 
    ffig = os.path.join(dat_dir, 'wpxi.%s.png' % theta)
    fig.savefig(ffig, bbox_inches='tight') 
    return None


def forecast(obs, params='default'):
    ''' fisher forecast  
    '''
    if isinstance(obs, list): 
        obs_list = obs 
    elif isinstance(obs, str): 
        obs_list = [obs] 

    # x, y ranges
    if params == 'default': 
        _thetas = thetas
    elif isinstance(params, list): 
        _thetas = params
    else: 
        raise NotImplementedError

    dlim = [[0.1, 0.06, 0.5, 0.5, 0.15, 0.75, 0.2, 0.4, 0.2,
        0.2][thetas.index(_tt)] for _tt in _thetas]

    fig = plt.figure(figsize=(17, 15))
    for i_obs, obs in enumerate(obs_list): 
        Fij = FisherMatrix(obs, params=params) # fisher matrix (Fij)

        cond = np.linalg.cond(Fij)
        if cond > 1e16: print('Fij is ill-conditioned %.5e' % cond)

        Finv    = np.linalg.inv(Fij) # invert fisher matrix 
        print('sigmas %s' % ', '.join(['%.2e' % sii for sii in np.sqrt(np.diag(Finv))]))

        for i in range(len(_thetas)+1): 
            for j in range(i+1, len(_thetas)): 
                # sub inverse fisher matrix 
                Finv_sub = np.array([[Finv[i,i], Finv[i,j]], [Finv[j,i], Finv[j,j]]]) 
                # plot the ellipse
                sub = fig.add_subplot(len(_thetas)-1, len(_thetas)-1, (len(_thetas)-1) * (j-1) + i + 1) 

                Forecast.plotEllipse(Finv_sub, sub, 
                        theta_fid_ij=[theta_fid[_thetas[i]],
                            theta_fid[_thetas[j]]], color='C%i'% i_obs)

                sub.set_xlim(theta_fid[_thetas[i]] - dlim[i], theta_fid[_thetas[i]] + dlim[i])
                sub.set_ylim(theta_fid[_thetas[j]] - dlim[j], theta_fid[_thetas[j]] + dlim[j])
                if i == 0:   
                    sub.set_ylabel(theta_lbls[thetas.index(_thetas[j])], fontsize=20) 
                else: 
                    sub.set_yticks([])
                    sub.set_yticklabels([])
                
                if j == len(_thetas)-1: 
                    sub.set_xlabel(theta_lbls[thetas.index(_thetas[i])], labelpad=10, fontsize=20) 
                else: 
                    sub.set_xticks([])
                    sub.set_xticklabels([]) 

        # write out Fisher matrix  
        hdr = '%s\n%s' % (', '.join(_thetas), '1sigma %s' % ', '.join(['%.2e' %
            sii for sii in np.sqrt(np.diag(Finv))]))
        
        ffish = os.path.join(dat_dir, 'fij_%s%s.dat' % (obs, _params_str(params)))
        np.savetxt(ffish, Fij, delimiter='\t', header=hdr)
    
    fig.subplots_adjust(wspace=0.05, hspace=0.05) 

    bkgd = fig.add_subplot(111, frameon=False)
    obs_lbl_dict = {'wp': r'$w_p$', 'xi0': r'$\xi_0$', 'xi': r'$\xi_\ell$',
            'wpxi': r'$w_p + \xi_\ell$'} 
    for i_obs, obs in enumerate(obs_list): 
        bkgd.fill_between([],[],[], color='C%i' % i_obs, label=obs_lbl_dict[obs]) 
    bkgd.legend(loc='upper right', bbox_to_anchor=(0.875, 0.775), fontsize=25)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    
    obs_str = '.'.join(obs_list) 
    ffig = os.path.join(dat_dir, '%s%s.forecast.png' % (obs_str,
        _params_str(params)))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def _forecast_converge(obs, params='default'):
    ''' fisher forecast  
    '''
    assert isinstance(obs, str)
    Nderivs = [200, 300, 350, 400, 425, 450, 475, 500]

    # x, y ranges
    if params == 'default': 
        _thetas = thetas
    elif isinstance(params, list): 
        _thetas = params
    else: 
        raise NotImplementedError

    dlim = [[0.1, 0.06, 0.5, 0.5, 0.15, 0.75, 0.2, 0.4, 0.2,
        0.2][thetas.index(_tt)] for _tt in _thetas]
    
    onesigs = [] 
    for i_deriv, Nderiv in enumerate(Nderivs): 
        Fij = FisherMatrix(obs, params=params, Nderiv=Nderiv) # fisher matrix (Fij)

        cond = np.linalg.cond(Fij)
        if cond > 1e16: print('Fij is ill-conditioned %.5e' % cond)

        Finv = np.linalg.inv(Fij) # invert fisher matrix 
        onesig = np.sqrt(np.diag(Finv))
        print('sigmas %s' % ', '.join(['%.2e' % sii for sii in onesig]))
        onesigs.append(onesig) 
    onesigs = np.array(onesigs) 

    fig = plt.figure(figsize=(6,6))
    sub = fig.add_subplot(111)
    for i, theta in enumerate(_thetas): 
        sub.plot(Nderivs, onesigs[:,i]/onesigs[-1,i], c='C%i' % i,
                label=theta_lbls[thetas.index(theta)]) 
    sub.plot([0, 1000], [1., 1.], c='k', ls='--') 
    sub.legend(loc='lower left', fontsize=20)   
    sub.set_xlabel(r'$N_{\rm deriv}$', fontsize=20) 
    sub.set_xlim(np.min(Nderivs), 500) 
    sub.set_ylabel(r'$N_{\rm deriv}$', fontsize=20) 
    sub.set_ylim(0.8, 1.1) 
    ffig = os.path.join(dat_dir, '_%s%s.forecast_converge.png' % (obs,
        _params_str(params)))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def plot_FisherMatrix(obs, params='default'): 
    ''' plot the Fisher matrix
    '''
    Fij = FisherMatrix(obs, params=params)
    
    if params == 'default': 
        theta_lbls = [r'$\Omega_m$', r'$\Omega_b$', '$h$', '$n_s$', r'$\sigma_8$',
                r'$\log M_{\rm min}$', r'$\sigma_{\log M}$', r'$\alpha$', r'$\log M_1$'] 
    else: 
        raise NotImplementedError
    
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
def FisherMatrix(obs, params='default', Nderiv=None): 
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
    elif obs == 'xi0': 
        _C_fid, nmock = cov_xi()
        dobsdt = dxidtheta
    
        nbin = _C_fid.shape[0] // 2
        C_fid = _C_fid[:nbin,:nbin]
    elif obs == 'xi2': 
        _C_fid, nmock = cov_xi()
        dobsdt = dxidtheta
    
        nbin = _C_fid.shape[0] // 2
        C_fid = _C_fid[nbin:,nbin:]
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
        _, dobsdt_i = dobsdt(par, Nderiv=Nderiv)
        
        if obs == 'xi0': 
            dobsdt_i = dobsdt_i[:nbin]
        elif obs == 'xi2': 
            dobsdt_i = dobsdt_i[nbin:]

        dobs_dt.append(dobsdt_i) 
            
    Fij = Forecast.Fij(dobs_dt, C_inv) 

    if params == 'default': 
        return Fij 
    elif isinstance(params, list): 
        i_thetas = np.array([thetas.index(param) for param in params]) 
        return Fij[i_thetas,:][:,i_thetas]


def plot_dwpxi(): 
    ''' compare wp, xi0, and xi2 derivatives w.r.t. the different thetas
    '''
    fig = plt.figure(figsize=(14,4))
    sub0 = fig.add_subplot(131)
    sub0.set_xscale('log') 
    sub0.set_xlim(0.1, 30.) 
    sub0.set_ylabel(r'$|{\rm d} w_{\rm p}/{\rm d}\theta|$', fontsize=25) 
    sub0.set_yscale('log') 
    sub0.set_ylim(0.5, 1e5) 
    
    sub1 = fig.add_subplot(132)
    sub1.set_xscale('log') 
    sub1.set_xlim(0.1, 30.) 
    sub1.set_xlabel('$r$', fontsize=25)
    sub1.set_yscale('log') 
    sub1.set_ylabel(r'$|{\rm d} r^2\xi_0/{\rm d}\theta|$', fontsize=25) 
        
    sub2 = fig.add_subplot(133)
    sub2.set_xscale('log') 
    sub2.set_xlim(0.1, 30.) 
    sub2.set_yscale('log') 
    sub2.set_ylabel(r'$|{\rm d} r^2\xi_2/{\rm d}\theta|$', fontsize=25) 

    for i, theta in enumerate(thetas): 
        r, dwp = dwpdtheta(theta)

        line, = sub0.plot(r, np.abs(dwp), c='C%i' % i, ls='--')
        dwp[dwp < 0] = np.nan
        sub0.plot(r, dwp, color=line.get_color(), ls='-', label=theta_lbls[i])

        r, dxis = dxidtheta(theta)
        dxi0 = dxis[:len(dxis)//2]
        dxi2 = dxis[len(dxis)//2:]

        # plot dxi_ell/dtheta
        sub1.plot(r, np.abs(r**2*dxi0), c='C%i' %i, ls='--')
        dxi0[dxi0 < 0] = np.nan 
        sub1.plot(r, r**2*dxi0, c=line.get_color(), ls='-')

        sub2.plot(r, np.abs(r**2*dxi2), c='C%i' %i, ls='--')
        dxi2[dxi2 < 0] = np.nan
        sub2.plot(r, r**2*dxi2, c=line.get_color(), ls='-')

    sub0.legend(loc='lower left', handletextpad=0.1, ncol=2)

    ffig = os.path.join(dat_dir, 'dwpxidtheta.png')
    fig.subplots_adjust(wspace=0.4) 
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def _plot_dwpxi_theta(theta): 
    ''' compare wp, xi0, and xi2 derivatives w.r.t. input theta
    '''
    i_theta = thetas.index(theta) 
    theta_lbl = theta_lbls[i_theta].replace('$', '') 

    fig = plt.figure(figsize=(14,4))
    sub0 = fig.add_subplot(131)
    sub0.set_xscale('log') 
    sub0.set_xlim(0.1, 30.) 
    sub0.set_ylabel(r'$|{\rm d} w_{\rm p}/{\rm d}%s|$' % theta_lbl, fontsize=25) 
    sub0.set_yscale('log') 
    sub0.set_ylim(0.5, 1e5) 
    
    sub1 = fig.add_subplot(132)
    sub1.set_xscale('log') 
    sub1.set_xlim(0.1, 30.) 
    sub1.set_xlabel('$r$', fontsize=25)
    sub1.set_yscale('log') 
    sub1.set_ylabel(r'$|{\rm d} r^2\xi_0/{\rm d}%s|$' % theta_lbl, fontsize=25) 
        
    sub2 = fig.add_subplot(133)
    sub2.set_xscale('log') 
    sub2.set_xlim(0.1, 30.) 
    sub2.set_yscale('log') 
    sub2.set_ylabel(r'$|{\rm d} r^2\xi_2/{\rm d}%s|$' % theta_lbl, fontsize=25) 

    r, dwp = dwpdtheta(theta)

    line, = sub0.plot(r, np.abs(dwp), c='k', ls='--')
    dwp[dwp < 0] = np.nan
    sub0.plot(r, dwp, color=line.get_color(), ls='-')

    r, dxis = dxidtheta(theta)
    dxi0 = dxis[:len(dxis)//2]
    dxi2 = dxis[len(dxis)//2:]

    # plot dxi_ell/dtheta
    sub1.plot(r, np.abs(r**2*dxi0), c='k', ls='--')
    dxi0[dxi0 < 0] = np.nan 
    sub1.plot(r, r**2*dxi0, c=line.get_color(), ls='-')

    sub2.plot(r, np.abs(r**2*dxi2), c='k', ls='--')
    dxi2[dxi2 < 0] = np.nan
    sub2.plot(r, r**2*dxi2, c=line.get_color(), ls='-')

    ffig = os.path.join(dat_dir, 'dwpxid%s.png' % theta)
    fig.subplots_adjust(wspace=0.4) 
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


def dwpxidtheta(theta, Nderiv=None): 
    ''' derivative of [wp, xi0, xi2] w.r.t. theta 
    ''' 
    r0, dwp = dwpdtheta(theta, Nderiv=Nderiv)
    r1, dxi = dxidtheta(theta, Nderiv=Nderiv) 

    return r0, np.concatenate([dwp, dxi]) 


def dwpdtheta(theta, Nderiv=None): 
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
        
        r   += np.average(quij['r'], axis=0) 
        if Nderiv is None: 
            dwp += coeff * np.average(quij['wp'], axis=0) 
        else: 
            dwp += coeff * np.average(quij['wp'][:Nderiv], axis=0) 
    
    dwp  /= h 
    r   /= float(len(steps))
    return r, dwp 


def dxidtheta(theta, Nderiv=None): 
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
        if Nderiv is None: 
            dxi0    += coeff * np.average(quij['xi0'], axis=0) 
            dxi2    += coeff * np.average(quij['xi2'], axis=0) 
        else: 
            dxi0    += coeff * np.average(quij['xi0'][:Nderiv], axis=0) 
            dxi2    += coeff * np.average(quij['xi2'][:Nderiv], axis=0) 
    
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


def _params_str(params): 
    if params == 'default': return ''
    elif isinstance(params, list): 
        return '.%s' % ''.join(params)


if __name__=='__main__': 
    #wpxi()
    #for theta in thetas: 
    #    _wpxi_pm(theta)
    #    _plot_dwpxi_theta(theta)
    #plot_cov()
    #plot_dwpxi()
    #plot_FisherMatrix('wp')
    #plot_FisherMatrix('xi')
    #plot_FisherMatrix('wpxi')
    #forecast('wp')
    #forecast('xi')
    #forecast('wpxi')
    #forecast(['wp', 'xi0', 'xi', 'wpxi'], params='default')
    #forecast(['wp', 'xi0', 'xi', 'wpxi'], params=['Om', 'h', 'ns', 's8', 'Mnu', 'logMmin', 'sigma_logM', 'alpha', 'logM1'])
    for obs in ['wp', 'xi0', 'xi', 'wpxi']:
        _forecast_converge(obs, params='default')
