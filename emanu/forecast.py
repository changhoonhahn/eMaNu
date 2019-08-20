'''
'''
import numpy as np 
from . import obvs as Obvs
# -- mpl -- 
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def quijote_dPkdtheta(theta, log=False, rsd='all', flag=None, dmnu='fin', z=0, Nderiv=None, average=True, silent=True):
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
        tts = ['fiducial_za', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp']
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
        elif dmnu == 'fin_2lpt': 
            tts = ['fiducial', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp']
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
        if Nderiv is not None: 
            if (flag == 'reg') and (tt == 'fiducial'): 
                if average: 
                    _pk = np.average(quij['p0k'], axis=0)  
                else: 
                    _pk = quij['p0k']
            else: 
                __pk = quij['p0k'][:Nderiv]
                if not silent: print('only using %i of %i' % (__pk.shape[0], quij['p0k'].shape[0])) 
                if average: 
                    _pk = np.average(quij['p0k'][:Nderiv], axis=0)  
                else: 
                    _pk = quij['p0k'][:Nderiv]
        else: 
            if average: 
                _pk = np.average(quij['p0k'], axis=0)  
            else: 
                _pk = quij['p0k']
                if theta == 'Mnu' and dmnu == 'fin_2lpt' and not average: 
                    if rsd == 'all': 
                        _pk = quij['p0k'][:1500,:]
                    else: 
                        _pk = quij['p0k'][:500,:]
        
        if i_tt == 0: dpk = np.zeros(_pk.shape) 

        if log: _pk = np.log(_pk) # log 

        dpk += coeff * _pk 
    return quij['k'], dpk / h + c_dpk 


def quijote_dP02kdtheta(theta, log=False, rsd='all', flag=None, dmnu='fin', z=0, Nderiv=None, average=True, silent=True):
    ''' calculate the derivative d [P0(k), P2(k)] /d theta using the paired and fixed quijote simulations
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
    assert rsd != 'real', "there's no quadrupole in real-space use quijote_dPkdtheta instead" 

    if z != 0: raise NotImplementedError
    c_dpk = 0.
    if theta == 'Mnu': 
        if not silent: print("--- calculating dP/d%s using %s ---" % (theta, dmnu)) 
        tts = ['fiducial_za', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp']
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
        elif dmnu == 'fin_2lpt': 
            tts = ['fiducial', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp']
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
        _p02ks = np.concatenate([quij['p0k'], quij['p2k']], axis=1)
        if not log: c_dpk = np.average(_p02ks, axis=0) 
        else: c_dpk = np.ones(_p02ks.shape[1]) 
    elif theta == 'Asn' : 
        if not silent: print("--- calculating dP/dAsn ---") 
        # constant shot noise term is a free parameter
        tts = ['fiducial'] 
        coeffs = [0.] 
        h = 1. 
        quij = Obvs.quijotePk('fiducial', z=z, flag=flag, rsd=rsd, silent=silent)
        _p02ks = np.concatenate([quij['p0k'], quij['p2k']], axis=1)
        if not log: c_dpk = np.ones(_p02ks.shape[1]) 
        else: c_dpk = 1./np.average(_p02ks, axis=0) 
    else: 
        if not silent: print("--- calculating dP/d%s ---" % theta) 
        tts = [theta+'_m', theta+'_p'] 
        coeffs = [-1., 1.] 
        h = quijote_thetas[theta][1] - quijote_thetas[theta][0]

    for i_tt, tt, coeff in zip(range(len(tts)), tts, coeffs): 
        quij = Obvs.quijotePk(tt, z=z, flag=flag, rsd=rsd, silent=silent) # read Pk 
    
        p02ks = np.concatenate([quij['p0k'], quij['p2k']], axis=1) # [P0, P2] 
        if Nderiv is not None: 
            if (flag == 'reg') and (tt == 'fiducial'): 
                if average: 
                    _pk = np.average(p02ks, axis=0)  
                else: 
                    _pk = p02ks
            else: 
                __pk = p02ks[:Nderiv]
                if not silent: print('only using %i of %i' % (__pk.shape[0], quij['p0k'].shape[0])) 
                if average: 
                    _pk = np.average(__pk, axis=0)  
                else: 
                    _pk = __pk
        else: 
            if average: 
                _pk = np.average(p02ks, axis=0)  
            else: 
                _pk = p02ks 
                if theta == 'Mnu' and dmnu == 'fin_2lpt' and not average: 
                    if rsd == 'all': 
                        _pk = p02ks[:1500,:]
                    else: 
                        _pk = p02ks[:500,:]
        
        if i_tt == 0: dpk = np.zeros(_pk.shape) 

        if log: _pk = np.log(_pk) # log 

        dpk += coeff * _pk 
    return np.concatenate([quij['k'], quij['k']]) , dpk / h + c_dpk 


def quijote_dBkdtheta(theta, log=False, rsd='all', flag=None, z=0, dmnu='fin', Nderiv=None, average=True, silent=True):
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
        tts = ['fiducial_za', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp']
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
        elif dmnu == 'fin_2lpt': 
            tts = ['fiducial', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp'] # use ZA inital conditions
            coeffs = [-21., 32., -12., 1.] # finite difference coefficient
            h = 1.2
        elif dmnu == '0.2eV_2LPTZA': # derivative @ 0.2 eV (not using 0.0eV which has 2LPT IC) 
            coeffs = [0., -20., 15., 5.] 
            h = 3. 
        elif dmnu == '0.2eV_ZA': # derivative @ 0.2 eV (not using 0.0eV which has 2LPT IC) 
            coeffs = [15., -80., 60., 5.] 
            h = 6. 
        else: 
            raise NotImplementedError
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

        if Nderiv is not None: 
            if (flag == 'reg') and (tt == 'fiducial'):
                if average: 
                    _bk = np.average(quij['b123'], axis=0)  
                else: 
                    _bk = quij['b123']
            else: 
                __bk = quij['b123'][:Nderiv]
                if not silent: print('only using %i of %i' % (__bk.shape[0], quij['b123'].shape[0])) 
                if average: 
                    _bk = np.average(__bk, axis=0)  
                else: 
                    _bk = __bk
        else: 
            if average: 
                _bk = np.average(quij['b123'], axis=0)  
            else: 
                _bk = quij['b123']
                if theta == 'Mnu' and dmnu == 'fin_2lpt' and not average: 
                    if rsd == 'all': 
                        _bk = quij['b123'][:1500,:]
                    else: 
                        _bk = quij['b123'][:500,:]
        
        if i_tt == 0: dbk = np.zeros(_bk.shape) 

        if log: _bk = np.log(_bk) 
        dbk += coeff * _bk 

    return quij['k1'], quij['k2'], quij['k3'], dbk / h + c_dbk 


def _PF_quijote_dPkdtheta(theta, log=False, rsd='all', flag='reg', z=0, dmnu='fin', dh='pm', average=True, silent=True):
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
    if theta not in quijote_thetas.keys(): raise ValueError

    if theta == 'Mnu': 
        if not silent: print("--- calculating dP/d%s using %s ---" % (theta, dmnu)) 
        tts = ['fiducial_za', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp']
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
        elif dmnu == 'fin_2lpt': 
            tts = ['fiducial', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp']
            coeffs = [-21., 32., -12., 1.] # finite difference coefficient
            h = 1.2
    elif theta == 'h': 
        if not silent: print("--- calculating dB/d%s ---" % theta) 
        if dh == 'pm':  # standard derivative using h+ and h- 
            tts = [theta+'_m', theta+'_p'] 
            h = quijote_thetas[theta][1] - quijote_thetas[theta][0]
        elif dh == 'm_only': 
            tts = [theta+'_m', 'fiducial'] 
            h = 0.6711 - quijote_thetas[theta][0]
        elif dh == 'p_only': 
            tts = ['fiducial', theta+'_p'] 
            h = quijote_thetas[theta][1] - 0.6711
        coeffs = [-1., 1.]
    else: 
        if not silent: print("--- calculating dP/d%s ---" % theta) 
        tts = [theta+'_m', theta+'_p'] 
        coeffs = [-1., 1.] 
        h = quijote_thetas[theta][1] - quijote_thetas[theta][0]

    for i_tt, tt, coeff in zip(range(len(tts)), tts, coeffs): 
        quij = Obvs.quijotePk(tt, z=z, flag=flag, rsd=rsd, silent=silent) # read Pk 
    
        _pk = quij['p0k'] 
        if theta == 'Mnu' and dmnu == 'fin_2lpt': 
            if rsd == 'all': 
                _pk = _pk[:1500,:]
            else: 
                _pk = _pk[:500,:]
        elif theta == 'h' and dh in ['m_only', 'p_only']: 
            if rsd == 'all': 
                _pk = _pk[:1500,:]
            else: 
                _pk = _pk[:500,:]
        
        if average: _pk = np.average(quij['p0k'], axis=0)  
        
        if log: _pk = np.log(_pk) # log 

        if i_tt == 0: dpk = np.zeros(_pk.shape) 
        dpk += coeff * _pk 

    return quij['k'], dpk / h


def _PF_quijote_dP02kdtheta(theta, log=False, rsd='all', flag='reg', dmnu='fin', dh='pm', z=0, average=True, silent=True):
    ''' calculate the derivative d [P0(k), P2(k)] /d theta using the paired and fixed quijote simulations
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
    assert rsd != 'real', "there's no quadrupole in real-space use quijote_dPkdtheta instead" 

    if z != 0: raise NotImplementedError

    if theta == 'Mnu': 
        if not silent: print("--- calculating dP/d%s using %s ---" % (theta, dmnu)) 
        tts = ['fiducial_za', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp']
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
        elif dmnu == 'fin_2lpt': 
            tts = ['fiducial', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp']
            coeffs = [-21., 32., -12., 1.] # finite difference coefficient
            h = 1.2
    elif theta == 'h': 
        if not silent: print("--- calculating dB/d%s ---" % theta) 
        if dh == 'pm':  # standard derivative using h+ and h- 
            tts = [theta+'_m', theta+'_p'] 
            h = quijote_thetas[theta][1] - quijote_thetas[theta][0]
        elif dh == 'm_only': 
            tts = [theta+'_m', 'fiducial'] 
            h = 0.6711 - quijote_thetas[theta][0]
        elif dh == 'p_only': 
            tts = ['fiducial', theta+'_p'] 
            h = quijote_thetas[theta][1] - 0.6711
        coeffs = [-1., 1.]
    else: 
        if not silent: print("--- calculating dP/d%s ---" % theta) 
        tts = [theta+'_m', theta+'_p'] 
        coeffs = [-1., 1.] 
        h = quijote_thetas[theta][1] - quijote_thetas[theta][0]

    for i_tt, tt, coeff in zip(range(len(tts)), tts, coeffs): 
        quij = Obvs.quijotePk(tt, z=z, flag=flag, rsd=rsd, silent=silent) # read Pk 
    
        _pk = np.concatenate([quij['p0k'], quij['p2k']], axis=1) # [P0, P2] 

        if theta == 'Mnu' and dmnu == 'fin_2lpt': 
            if rsd == 'all': 
                _pk = _pk[:1500,:]
            else: 
                _pk = _pk[:500,:]
        elif theta == 'h' and dh in ['m_only', 'p_only']:
            if rsd == 'all': 
                _pk = _pk[:1500,:]
            else: 
                _pk = _pk[:500,:]

        if average: _pk = np.average(_pk, axis=0)  
        if log: _pk = np.log(_pk) # log 
        
        if i_tt == 0: dpk = np.zeros(_pk.shape) 

        dpk += coeff * _pk 
    return np.concatenate([quij['k'], quij['k']]) , dpk / h


def _PF_quijote_dBkdtheta(theta, log=False, rsd='all', flag='reg', dmnu='fin', dh='pm', z=0, average=True, corr_sn=True, silent=True):
    ''' d B(k)/d theta calculations for the paired-fixed investigation. Lots of bells and whistles 
    that I want separate from the main analysis. 

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
    if theta not in quijote_thetas.keys(): raise ValueError
    if flag not in ['reg', 'ncv']: raise ValueError 

    if theta == 'Mnu': 
        if not silent: print("--- calculating dB/d%s using %s ---" % (theta, dmnu)) 
        # derivative w.r.t. Mnu using 0, 0.1, 0.2, 0.4 eV
        tts = ['fiducial_za', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp']
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
        elif dmnu == 'fin_2lpt': 
            tts = ['fiducial', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp'] # use ZA inital conditions
            coeffs = [-21., 32., -12., 1.] # finite difference coefficient
            h = 1.2
        elif dmnu == '0.2eV_2LPTZA': # derivative @ 0.2 eV (not using 0.0eV which has 2LPT IC) 
            coeffs = [0., -20., 15., 5.] 
            h = 3. 
        elif dmnu == '0.2eV_ZA': # derivative @ 0.2 eV (not using 0.0eV which has 2LPT IC) 
            coeffs = [15., -80., 60., 5.] 
            h = 6. 
        else: 
            raise NotImplementedError
    elif theta == 'h': 
        if not silent: print("--- calculating dB/d%s ---" % theta) 
        if dh == 'pm':  # standard derivative using h+ and h- 
            tts = [theta+'_m', theta+'_p'] 
            h = quijote_thetas[theta][1] - quijote_thetas[theta][0]
        elif dh == 'm_only': 
            tts = [theta+'_m', 'fiducial'] 
            h = 0.6711 - quijote_thetas[theta][0]
        elif dh == 'p_only': 
            tts = ['fiducial', theta+'_p'] 
            h = quijote_thetas[theta][1] - 0.6711
        coeffs = [-1., 1.]
    else: 
        if not silent: print("--- calculating dB/d%s ---" % theta) 
        tts = [theta+'_m', theta+'_p'] 
        coeffs = [-1., 1.]
        h = quijote_thetas[theta][1] - quijote_thetas[theta][0]

    for i_tt, tt, coeff in zip(range(len(tts)), tts, coeffs): 
        quij = Obvs.quijoteBk(tt, z=z, flag=flag, rsd=rsd, silent=silent)
        b_sn = quij['b_sn']
        
        _bk = quij['b123'] 
        if not corr_sn: _bk += b_sn  # uncorrect for shot-noise 

        if theta == 'Mnu' and dmnu == 'fin_2lpt': # fiducial is 2LPT and has different number of sims 
            if rsd == 'all': 
                _bk = _bk[:1500,:]
            else: 
                _bk = _bk[:500,:]
        elif theta == 'h' and dh in ['m_only', 'p_only']: 
            if rsd == 'all': 
                _bk = _bk[:1500,:]
            else: 
                _bk = _bk[:500,:]
        
        if average: _bk = np.average(_bk, axis=0)  
        if log: _bk = np.log(_bk) 
    
        if i_tt == 0: 
            dbk = np.zeros(_bk.shape) 
        dbk += coeff * _bk 

    return quij['k1'], quij['k2'], quij['k3'], dbk / h 


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
        if ii == 0: e.set_alpha(0.7)
        if ii == 1: e.set_alpha(1.0)
        e.set_facecolor(color) 
    return sub


def plotFisher(Finvs, theta_fid, colors=None, linestyles=None, ranges=None, titles=None, title_kwargs=None, labels=None): 
    ''' Given a list of inverse Fisher matrices, plot the Fisher contours
    '''
    ntheta = Finvs[0].shape[0] # number of parameters 
    assert ntheta == len(theta_fid)
    
    # check inputs 
    if colors is None: colors = ['C%i' % i for i in range(len(Finvs))]
    else: assert len(Finvs) == len(colors) 

    if ranges is not None: assert ntheta == len(ranges) 
    if labels is not None: assert ntheta == len(labels) 
    
    # calculate the marginalized constraints 
    onesigmas = [np.sqrt(np.diag(_Finv)) for _Finv in Finvs]

    fig = plt.figure(figsize=(3*ntheta, 3*ntheta))
    for i in range(ntheta): 
        for j in range(i,ntheta): 
            sub = fig.add_subplot(ntheta,ntheta,ntheta*j+i+1)

            if i != j: # 2D contour 
                theta_fid_i, theta_fid_j = theta_fid[i], theta_fid[j] # fiducial parameter 
                for _i, _Finv in enumerate(Finvs):
                    Finv_sub = np.array([[_Finv[i,i], _Finv[i,j]], [_Finv[j,i], _Finv[j,j]]]) # sub inverse fisher matrix 
                    plotEllipse(Finv_sub, sub, theta_fid_ij=[theta_fid_i, theta_fid_j], color=colors[_i])
            
                if ranges is not None: 
                    sub.set_xlim(ranges[i])
                    sub.set_ylim(ranges[j])
                # y-axes
                if i == 0:   
                    if labels is not None: 
                        sub.set_ylabel(labels[j], labelpad=5, fontsize=28) 
                        sub.get_yaxis().set_label_coords(-0.35,0.5)
                else: 
                    sub.set_yticks([])
                    sub.set_yticklabels([])
                # x-axes
                if j == ntheta-1: 
                    if labels is not None: 
                        sub.set_xlabel(labels[i], fontsize=26) 
                else: 
                    sub.set_xticks([])
                    sub.set_xticklabels([]) 

            elif i == j: # marginalized gaussian constraints on the diagonal  
                if ranges is None: 
                    x = np.linspace(0.5 * theta_fid[i], 1.5 * theta_fid[i], 100) 
                else: 
                    x = np.linspace(ranges[i][0], ranges[i][1], 100) 
                for _i, onesigma in enumerate(onesigmas): 
                    if linestyles is None: 
                        sub.plot(x, _gaussian(x, theta_fid[i], onesigma[i]), c=colors[_i])
                    else: 
                        sub.plot(x, _gaussian(x, theta_fid[i], onesigma[i]), c=colors[_i], ls=linestyles[_i])

                if ranges is not None: sub.set_xlim(ranges[i])
                if j != ntheta-1: 
                    sub.set_xticklabels([]) 
                sub.set_ylim(0., None) 
                sub.set_yticks([]) 
                sub.set_yticklabels([]) 
                if titles is not None: 
                    sub.set_title(titles[i], **title_kwargs) 

    fig.subplots_adjust(wspace=0.05, hspace=0.05) 
    return fig 


def _gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
