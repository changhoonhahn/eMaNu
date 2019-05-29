'''
'''
import numpy as np 
from . import obvs as Obvs
# -- mpl -- 
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def quijote_dPkdtheta(theta, log=False, rsd='all', flag=None, dmnu='fin', z=0, Nderiv=None, silent=True):
    ''' calculate d P0(k)/d theta using the paired and fixed quijote simulations
    run on perturbed theta 

    :param theta: 
        string that specifies the parameter to take the 
        derivative by. 
    '''
    quijote_thetas = {
            'Mnu': [0.1, 0.2, 0.4], # +, ++, +++ 
            'Ob': [0.048, 0.050],   # others are - + 
            'Ob2': [0.047, 0.051],   # others are - + 
            'Om': [0.3075, 0.3275],
            'h': [0.6511, 0.6911],
            'ns': [0.9424, 0.9824],
            's8': [0.819, 0.849]}

    if z != 0: raise NotImplementedError
    c_dpk = 0.
    if theta == 'Mnu': 
        if not silent: print("--- calculating dP/d%s using %s ---" % (theta, dmnu)) 
        tts = ['fiducial', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp']
        if dmnu == 'p': 
            coeffs = [-1., 1., 0., 0.]
            h = 0.1
        elif dmnu == 'pp': 
            coeffs = [-1., 0., 1., 0.]
            h = 0.2
        elif dmnu == 'ppp': 
            coeffs = [-1., 0., 0., 1.]
            h = 0.4
        elif dmnu == 'fin0': 
            coeffs = [-3., 4., -1., 0.] # finite difference coefficient
            h = 0.2
        elif dmnu == 'fin': 
            coeffs = [-21., 32., -12., 1.] # finite difference coefficient
            h = 1.2
    elif theta == 'Mmin': # halo mass limit 
        if not silent: print("--- calculating dP/dMmin ---") 
        tts = ['Mmin_m', 'Mmin_p'] 
        coeffs = [-1., 1.] 
        h = 0.2 # 3.3 - 3.1 x 10^13 Msun 
    elif theta == 'Amp': 
        if not silent: print("--- calculating dP/db' ---") 
        # amplitude of P(k) is a free parameter
        tts = ['fiducial'] 
        coeffs = [0.] 
        h = 1. 
        quij = Obvs.quijotePk('fiducial', z=z, flag=flag, rsd=rsd, silent=silent)
        if not log: c_dpk = np.average(quij['p0k'], axis=0) 
        else: c_dpk = np.ones(quij['p0k'].shape[1]) 
    elif theta == 'Asn' : 
        if not silent: print("--- calculating dP/dAsn ---") 
        # constant shot noise term is a free parameter
        tts = ['fiducial'] 
        coeffs = [0.] 
        h = 1. 
        quij = Obvs.quijotePk('fiducial', z=z, flag=flag, rsd=rsd, silent=silent)
        if not log: c_dpk = np.ones(quij['p0k'].shape[1]) 
        else: c_dpk = 1./np.average(quij['p0k'], axis=0) 
    else: 
        if not silent: print("--- calculating dP/d%s ---" % theta) 
        tts = [theta+'_m', theta+'_p'] 
        coeffs = [-1., 1.] 
        h = quijote_thetas[theta][1] - quijote_thetas[theta][0]

    for i_tt, tt, coeff in zip(range(len(tts)), tts, coeffs): 
        quij = Obvs.quijotePk(tt, z=z, flag=flag, rsd=rsd, silent=silent) # read Pk 
        if i_tt == 0: dpk = np.zeros(quij['p0k'].shape[1]) 
    
        if Nderiv is not None and tt != 'fiducial': 
            __pk = quij['p0k'][:Nderiv]
            if not silent: print('only using %i of %i' % (__pk.shape[0], quij['p0k'].shape[0])) 
            _pk = np.average(quij['p0k'][:Nderiv], axis=0)  
        else: 
            _pk = np.average(quij['p0k'], axis=0)  

        if log: _pk = np.log(_pk) # log 

        dpk += coeff * _pk 
    return quij['k'], dpk / h + c_dpk 


def quijote_dBkdtheta(theta, log=False, rsd='all', flag=None, z=0, dmnu='fin', Nderiv=None, silent=True):
    ''' calculate d B(k)/d theta using quijote simulations run on perturbed theta 

    :param theta: 
        string that specifies the parameter to take the 
        derivative by. 

    :param log: (default: False) 
        boolean that specifies whether to return dB/dtheta or dlogB/dtheta 
   
    :param rsd: (default: 'all') 
        rsd kwarg that specifies rsd set up for B(k). 
        If rsd == 'all', include 3 RSD directions. 
        If rsd in [0,1,2] include one of the directions
    
    :param flag: (default: None) 
        kwarg specifying the flag for B(k). 
        If `flag is None`, include paired-fixed and regular N-body simulation.(not advised)
        If `flag == 'ncv'` only include paired-fixed. 
        If `flag == 'reg'` only include regular N-body

    :param dmnu: (default: 'fin') 
        stirng that specifies the derivative method for dB/dMnu. Default 
        is finite difference method using 0, 0.1, 0.2, 0.4 eVs

    :return k1, k2, k3, dbk
        triangle sides and derivatives
    '''
    quijote_thetas = {
            'Mnu': [0.1, 0.2, 0.4], # +, ++, +++ 
            'Ob': [0.048, 0.050],   # others are - + 
            'Ob2': [0.047, 0.051],   
            'Om': [0.3075, 0.3275],
            'h': [0.6511, 0.6911],
            'ns': [0.9424, 0.9824],
            's8': [0.819, 0.849]}

    c_dbk = 0. 
    if theta == 'Mnu': 
        if not silent: print("--- calculating dB/d%s using %s ---" % (theta, dmnu)) 
        # derivative w.r.t. Mnu using 0, 0.1, 0.2, 0.4 eV
        tts = ['fiducial', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp']
        if dmnu == 'p': 
            coeffs = [-1., 1., 0., 0.]  # derivative at 0.05eV
            h = 0.1
        elif dmnu == 'pp': 
            coeffs = [-1., 0., 1., 0.]  # derivative at 0.1eV
            h = 0.2
        elif dmnu == 'ppp': 
            coeffs = [-1., 0., 0., 1.]  # derivative at 0.2eV
            h = 0.4
        elif dmnu == 'fin0': 
            coeffs = [-3., 4., -1., 0.] # finite difference coefficient
            h = 0.2
        elif dmnu == 'fin': 
            coeffs = [-21., 32., -12., 1.] # finite difference coefficient
            h = 1.2
    elif theta == 'Mmin': 
        if not silent: print("--- calculating dB/dMmin ---") 
        tts = ['Mmin_m', 'Mmin_p'] 
        coeffs = [-1., 1.] 
        h = 0.2 # 3.3x10^13 - 3.1x10^13 Msun 
    elif theta == 'Amp': 
        if not silent: print("--- calculating dB/db' ---") 
        # amplitude scaling is a free parameter
        tts = ['fiducial'] 
        coeffs = [0.] 
        h = 1. 
        quij = Obvs.quijoteBk('fiducial', z=z, flag=flag, rsd=rsd, silent=silent)
        if not log: c_dbk = np.average(quij['b123'], axis=0) 
        else: c_dbk = np.ones(quij['b123'].shape[1])
    elif theta == 'b2': 
        if not silent: print("--- calculating dB/db2' ---") 
        # analytic deivative of b2 where we have 
        # B = b' B_nbody + b2 (P1P2 + P2P3 + P3P1) 
        tts = ['fiducial'] 
        coeffs = [0.] 
        h = 1. 
        quij = Obvs.quijoteBk('fiducial', z=z, flag=flag, rsd=rsd, silent=silent)
        P1 = np.average(quij['p0k1'], axis=0) 
        P2 = np.average(quij['p0k2'], axis=0) 
        P3 = np.average(quij['p0k3'], axis=0) 
        if not log: c_dbk = P1*P2 + P2*P3 + P3*P1 
        else: c_dbk = (P1*P2 + P2*P3 + P3*P1)/np.average(quij['b123'], axis=0) 
    elif theta == 'g2': 
        if not silent: print("--- calculating dB/dg2' ---") 
        # analytic derivative of gamma2 where we have 
        # B = b' B + b2 (P1P2 + P2P3 + P3P1) + g2 (K12 P1 P2 + K23 P2 P3 + K31 P3 P1) 
        tts = ['fiducial'] 
        coeffs = [0.] 
        h = 1. 

        quij = Obvs.quijoteBk('fiducial', z=z, flag=flag, rsd=rsd, silent=silent)
        i_k, j_k, l_k = quij['k1'], quij['k2'], quij['k3']
        K12 = (i_k**2 + j_k**2 - l_k**2)/(2. * i_k * j_k) # cos theta_12
        K23 = (j_k**2 + l_k**2 - i_k**2)/(2. * j_k * l_k) # cos theta_23
        K31 = (l_k**2 + i_k**2 - j_k**2)/(2. * l_k * i_k) # cos theta_31

        P1 = np.average(quij['p0k1'], axis=0) 
        P2 = np.average(quij['p0k2'], axis=0) 
        P3 = np.average(quij['p0k3'], axis=0) 

        if not log: c_dbk = K12 * P1 * P2 + K23 * P2 * P3 + K31 * P3 * P1 
        else: c_dbk = (K12 * P1 * P2 + K23 * P2 * P3 + K31 * P3 * P1)/np.average(quij['b123'], axis=0) 
    elif theta == 'Asn' : 
        # free parameter that's supposed to account for the constant shot noise term -- 1/n^2
        # B = B_nbody + Bsn * (P1 + P2 + P3) + Asn
        tts = ['fiducial'] 
        coeffs = [0.] 
        h = 1. 

        quij = Obvs.quijoteBk('fiducial', z=z, flag=flag, rsd=rsd, silent=silent)
        if not log: c_dbk = np.ones(quij['b123'].shape[1]) * 1.e8 
        else: c_dbk = 1.e8/np.average(quij['b123'], axis=0) 
    elif theta == 'Bsn': 
        # free parameter that's suppose to account for the powerspectrum dependent shot noise term -- 1/n 
        # B = B_nbody + Bsn * (P1 + P2 + P3) + Asn
        tts = ['fiducial'] 
        coeffs = [0.] 
        h = 1. 

        quij = Obvs.quijoteBk('fiducial', z=z, flag=flag, rsd=rsd, silent=silent)
        if not log: c_dbk = np.average(quij['p0k1'] + quij['p0k2'] + quij['p0k3'], axis=0)
        else: c_dbk = np.average(quij['p0k1'] + quij['p0k2'] + quij['p0k3'], axis=0) / np.average(quij['b123'], axis=0)
    else: 
        if not silent: print("--- calculating dB/d%s ---" % theta) 
        tts = [theta+'_m', theta+'_p'] 
        coeffs = [-1., 1.]
        h = quijote_thetas[theta][1] - quijote_thetas[theta][0]

    for i_tt, tt, coeff in zip(range(len(tts)), tts, coeffs): 
        quij = Obvs.quijoteBk(tt, z=z, flag=flag, rsd=rsd, silent=silent)

        if i_tt == 0: dbk = np.zeros(quij['b123'].shape[1]) 
        if Nderiv is not None and tt != 'fiducial': 
            __bk = quij['b123'][:Nderiv]
            if not silent: print('only using %i of %i' % (__bk.shape[0], quij['b123'].shape[0])) 
            _bk = np.average(__bk, axis=0)  
        else: 
            _bk = np.average(quij['b123'], axis=0)  

        if log: _bk = np.log(_bk) 
        dbk += coeff * _bk 

    return quij['k1'], quij['k2'], quij['k3'], dbk / h + c_dbk 


def Fij(dmudts, Cinv): 
    ''' given derivative of observable along thetas and
    inverse covariance matrix, return fisher matrix Fij

    :param dmudts: 
        List of derivatives d mu/d theta_i 

    :param Cinv: 
        precision matrix 

    :return Fij: 
        Fisher matrix 
    '''
    ntheta = len(dmudts) 
    Fij = np.zeros((ntheta, ntheta))
    for i in range(ntheta): 
        for j in range(ntheta): 
            dmu_dt_i, dmu_dt_j = dmudts[i], dmudts[j]

            # calculate Mij 
            Mij = np.dot(dmu_dt_i[:,None], dmu_dt_j[None,:]) + np.dot(dmu_dt_j[:,None], dmu_dt_i[None,:])
            Fij[i,j] = 0.5 * np.trace(np.dot(Cinv, Mij))
    return Fij 
    

def plotEllipse(Finv_sub, sub, theta_fid_ij=None, color='C0'): 
    ''' Given the inverse fisher sub-matrix, calculate ellipse parameters and
    add to subplot 
    '''
    theta_fid_i, theta_fid_j = theta_fid_ij
    # get ellipse parameters 
    a = np.sqrt(0.5*(Finv_sub[0,0] + Finv_sub[1,1]) + np.sqrt(0.25*(Finv_sub[0,0]-Finv_sub[1,1])**2 + Finv_sub[0,1]**2))
    b = np.sqrt(0.5*(Finv_sub[0,0] + Finv_sub[1,1]) - np.sqrt(0.25*(Finv_sub[0,0]-Finv_sub[1,1])**2 + Finv_sub[0,1]**2))
    theta = 0.5 * np.arctan2(2.0 * Finv_sub[0,1], (Finv_sub[0,0] - Finv_sub[1,1]))
    for ii, alpha in enumerate([2.48, 1.52]):
        e = Ellipse(xy=(theta_fid_i, theta_fid_j), 
                width=alpha * a, height=alpha * b, angle=theta * 360./(2.*np.pi))
        sub.add_artist(e)
        if ii == 0: alpha = 0.7
        if ii == 1: alpha = 1.
        e.set_alpha(alpha)
        e.set_facecolor(color) 
    return sub
