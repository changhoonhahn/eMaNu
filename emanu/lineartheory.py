'''
submodule for linear theory calculations done using CLASS 
'''
import os 
import numpy as np 
from scipy.interpolate import interp1d
# --- eMaNu --- 
from . import util as UT


def Fij_Pm(k, kmax=0.5, npoints=5, thetas=None, flag=None): 
    ''' calculate fisher matrix for linear theory Pm
    '''
    if thetas is None: thetas = ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu'] # default theta 
    klim = (k < kmax) 
    k = k[klim] 

    Pm_fid = _Pm_Mnu(0.0, k) 
    factor = 1.e9 / (4. * np.pi**2) 
    
    Fij = np.zeros((len(thetas), len(thetas)))
    for i, tt_i in enumerate(thetas): 
        for j, tt_j in enumerate(thetas): 
            # get d ln Pm(k) / d thetas
            #dlnPdtti = dPmdtheta(tt_i, k, log=True, npoints=npoints) 
            #dlnPdttj = dPmdtheta(tt_j, k, log=True, npoints=npoints) 
            #Fij[i,j] = np.trapz(k**3 * dlnPdtti * dlnPdttj, np.log(k))
            dPdtti = dPmdtheta(tt_i, k, log=False, npoints=npoints, flag=flag) 
            dPdttj = dPmdtheta(tt_j, k, log=False, npoints=npoints, flag=flag) 
            Fij[i,j] = np.trapz(k**2 * dPdtti * dPdttj / Pm_fid**2, k)
    return Fij * factor 


def dPmdtheta(theta, k, log=False, npoints=5, flag=None): 
    ''' derivative of Pm w.r.t. to theta 
    '''
    if theta == 'Mnu': 
        return dPmdMnu(k, npoints=npoints, log=log, flag=flag) 
    else:
        h_dict = {'Ob': 0.002, 'Om': 0.02, 'h': 0.04, 'ns': 0.04, 's8': 0.03, 'As': 0.4} 
        h = h_dict[theta]
        Pm_m = _Pm_theta('%s_m' % theta, k, flag=flag) 
        Pm_p = _Pm_theta('%s_p' % theta, k, flag=flag) 
        if not log: 
            return (Pm_p - Pm_m)/h 
        else: # dlnP/dtheta
            return (np.log(Pm_p) - np.log(Pm_m))/h


def _Pm_theta(theta, karr, flag=None): 
    ''' linear theory matter power spectrum with fiducial parameters except theta for k 
    '''
    _thetas = ['Ob_m', 'Ob_p', 'Om_m', 'Om_p', 'h_m', 'h_p', 'ns_m', 'ns_p', 's8_m', 's8_p', 'As_m', 'As_p'] 
    if theta not in _thetas: 
        raise ValueError 
    if flag in ['fixAs', 'fixAs_cb']: 
        f = os.path.join(UT.dat_dir(), 'lt', 'output', '%s_fixAs_pk.dat' % theta)
    else: 
        f = os.path.join(UT.dat_dir(), 'lt', 'output', '%s_pk.dat' % theta)
    k, pmk = np.loadtxt(f, unpack=True, skiprows=4, usecols=[0,1]) 
    fpmk = interp1d(k, pmk, kind='cubic') 
    return fpmk(karr) 


def dPmdMnu(k, npoints=5, log=False, flag=None): 
    '''calculate derivatives of the linear theory matter power spectrum  
    at 0.0 eV using npoints different points 
    '''
    if npoints == 1: 
        mnus = [0.0, 0.025]
        coeffs = [-1., 1.]
        fdenom = 1. * 0.025
    elif npoints == 2:  
        mnus = [0.0, 0.025, 0.05]
        coeffs = [-3., 4., -1.]
        fdenom = 2. * 0.025
    elif npoints == 3:  
        mnus = [0.0, 0.025, 0.05, 0.075]
        coeffs = [-11., 18., -9., 2.]
        fdenom = 6. * 0.025
    elif npoints == 4:  
        mnus = [0.0, 0.025, 0.05, 0.075, 0.1]
        coeffs = [-25., 48., -36., 16., -3.]
        fdenom = 12.* 0.025
    elif npoints == 5:  
        mnus = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125]
        coeffs = [-137., 300., -300., 200., -75., 12.]
        fdenom = 60.* 0.025
    elif npoints == 'quijote': 
        mnus = [0.0, 0.1, 0.2, 0.4]
        coeffs = [-21., 32., -12., 1.]
        fdenom = 1.2 
    elif npoints == '0.05eV': 
        mnus = [0.0, 0.1]
        coeffs = [-1., 1.]
        fdenom = 0.1 
    elif npoints == '0.1eV':
        mnus = [0.075, 0.125]
        coeffs = [-1., 1.]
        fdenom = 0.05 
    else: 
        raise ValueError
    
    dPm = np.zeros(len(k))
    for i, mnu, coeff in zip(range(len(mnus)), mnus, coeffs): 
        if not log: 
            pm = _Pm_Mnu(mnu, k, flag=flag) 
        else: 
            pm = np.log(_Pm_Mnu(mnu, k, flag=flag))
        dPm += coeff * pm 
    dPm /= fdenom 
    return dPm 


def _Pm_Mnu(mnu, karr, flag=None): 
    ''' linear theory matter P(k) with fiducial parameters and input Mnu
    calculated using the high precision CLASS setting 
    '''
    if flag is None: 
        if mnu == 0.0: 
            fema = os.path.join(UT.dat_dir(), 'lt', 'output', '0eV_pk.dat')
        elif mnu == 0.025: 
            fema = os.path.join(UT.dat_dir(), 'lt', 'output', '0p025eV_pk.dat')
        elif mnu == 0.05: 
            fema = os.path.join(UT.dat_dir(), 'lt', 'output', '0p05eV_pk.dat')
        elif mnu == 0.075: 
            fema = os.path.join(UT.dat_dir(), 'lt', 'output', '0p075eV_pk.dat')
        elif mnu == 0.1: 
            fema = os.path.join(UT.dat_dir(), 'lt', 'output', '0p1eV_pk.dat')
        elif mnu == 0.125: 
            fema = os.path.join(UT.dat_dir(), 'lt', 'output', '0p125eV_pk.dat')
        elif mnu == 0.2: 
            fema = os.path.join(UT.dat_dir(), 'lt', 'output', '0p2eV_pk.dat')
        elif mnu == 0.4: 
            fema = os.path.join(UT.dat_dir(), 'lt', 'output', '0p4eV_pk.dat')
        k, pmk = np.loadtxt(fema, unpack=True, skiprows=4, usecols=[0,1]) 
        fpmk = interp1d(k, pmk, kind='cubic') 
        return fpmk(karr)
    elif flag == 'cb': 
        if mnu == 0.0: 
            fema = os.path.join(UT.dat_dir(), 'lt', 'output', '0eV_pk.dat')
        elif mnu == 0.025: 
            fema = os.path.join(UT.dat_dir(), 'lt', 'output', '0p025eV_pk_cb.dat')
        elif mnu == 0.05: 
            fema = os.path.join(UT.dat_dir(), 'lt', 'output', '0p05eV_pk_cb.dat')
        elif mnu == 0.075: 
            fema = os.path.join(UT.dat_dir(), 'lt', 'output', '0p075eV_pk_cb.dat')
        elif mnu == 0.1: 
            fema = os.path.join(UT.dat_dir(), 'lt', 'output', '0p1eV_pk_cb.dat')
        elif mnu == 0.125: 
            fema = os.path.join(UT.dat_dir(), 'lt', 'output', '0p125eV_pk_cb.dat')
        elif mnu == 0.2: 
            fema = os.path.join(UT.dat_dir(), 'lt', 'output', '0p2eV_pk_cb.dat')
        elif mnu == 0.4: 
            fema = os.path.join(UT.dat_dir(), 'lt', 'output', '0p4eV_pk_cb.dat')
        k, pmk = np.loadtxt(fema, unpack=True, skiprows=4, usecols=[0,1]) 
        fpmk = interp1d(k, pmk, kind='cubic') 
        return fpmk(karr)
    elif flag == 'fixAs': 
        if mnu == 0.0: 
            fema = os.path.join(UT.dat_dir(), 'lt', 'output', 'fid_fixAs_pk.dat')
        elif mnu == 0.1: 
            fema = os.path.join(UT.dat_dir(), 'lt', 'output', '0p1eV_fixAs_pk.dat')
        elif mnu == 0.2: 
            fema = os.path.join(UT.dat_dir(), 'lt', 'output', '0p2eV_fixAs_pk.dat')
        elif mnu == 0.4: 
            fema = os.path.join(UT.dat_dir(), 'lt', 'output', '0p4eV_fixAs_pk.dat')
        else: 
            raise NotImplementedError
        k, pmk = np.loadtxt(fema, unpack=True, skiprows=4, usecols=[0,1]) 
        fpmk = interp1d(k, pmk, kind='cubic') 
        return fpmk(karr)
    elif flag == 'fixAs_cb': 
        if mnu == 0.0: 
            fema = os.path.join(UT.dat_dir(), 'lt', 'output', 'fid_fixAs_pk.dat')
        elif mnu == 0.1: 
            fema = os.path.join(UT.dat_dir(), 'lt', 'output', '0p1eV_fixAs_pk_cb.dat')
        elif mnu == 0.2: 
            fema = os.path.join(UT.dat_dir(), 'lt', 'output', '0p2eV_fixAs_pk_cb.dat')
        elif mnu == 0.4: 
            fema = os.path.join(UT.dat_dir(), 'lt', 'output', '0p4eV_fixAs_pk_cb.dat')
        else: 
            raise NotImplementedError
        k, pmk = np.loadtxt(fema, unpack=True, skiprows=4, usecols=[0,1]) 
        fpmk = interp1d(k, pmk, kind='cubic') 
        return fpmk(karr)
    elif flag == 'ema': 
        if mnu == 0.0: 
            fema = os.path.join(UT.dat_dir(), 'ema', 'HADES_0p00eV_z1_pk.dat')
            fAs = 0.9818992345104124**2
        elif mnu == 0.025: 
            fema = os.path.join(UT.dat_dir(), 'ema', 'HADES_0p025eV_z1_pk.dat')
            fAs = 0.9879232989481492**2
        elif mnu == 0.05: 
            fema = os.path.join(UT.dat_dir(), 'ema', 'HADES_0p05eV_z1_pk.dat')
            fAs = 0.9961075140612549**2
        elif mnu == 0.075: 
            fema = os.path.join(UT.dat_dir(), 'ema', 'HADES_0p075eV_z1_pk.dat')
            fAs = 1.00415885618811**2
        elif mnu == 0.1: 
            fema = os.path.join(UT.dat_dir(), 'ema', 'HADES_0p10eV_z1_pk.dat')
            fAs = 1.0123494887789048**2
        elif mnu == 0.125: 
            fema = os.path.join(UT.dat_dir(), 'ema', 'HADES_0p125eV_z1_pk.dat')
            fAs = 1.0205994937246368**2
        k, pmk = np.loadtxt(fema, unpack=True, skiprows=4, usecols=[0,1]) 
        fpmk = interp1d(k, pmk, kind='cubic') 
        return fAs * fpmk(karr) 
    elif flag == 'ema_cb': 
        if mnu == 0.0: 
            fema = os.path.join(UT.dat_dir(), 'lt', 'output', 'HADES_0p00eV_z1_pk.dat')
            fAs = 0.9818992345104124**2
        elif mnu == 0.025: 
            fema = os.path.join(UT.dat_dir(), 'lt', 'output', 'HADES_0p025eV_z1_pk_cb.dat')
            fAs = 0.9879232989481492**2
        elif mnu == 0.05: 
            fema = os.path.join(UT.dat_dir(), 'lt', 'output', 'HADES_0p05eV_z1_pk_cb.dat')
            fAs = 0.9961075140612549**2
        elif mnu == 0.075: 
            fema = os.path.join(UT.dat_dir(), 'lt', 'output', 'HADES_0p075eV_z1_pk_cb.dat')
            fAs = 1.00415885618811**2
        elif mnu == 0.1: 
            fema = os.path.join(UT.dat_dir(), 'lt', 'output', 'HADES_0p10eV_z1_pk_cb.dat')
            fAs = 1.0123494887789048**2
        elif mnu == 0.125: 
            fema = os.path.join(UT.dat_dir(), 'lt', 'output', 'HADES_0p125eV_z1_pk_cb.dat')
            fAs = 1.0205994937246368**2
        k, pmk = np.loadtxt(fema, unpack=True, skiprows=4, usecols=[0,1]) 
        fpmk = interp1d(k, pmk, kind='cubic') 
        return fAs * fpmk(karr) 
    elif flag == 'paco': 
        if mnu == 0.0: 
            fpaco = os.path.join(UT.dat_dir(), 'paco', '0.00eV.txt')
        elif mnu == 0.025: 
            fpaco = os.path.join(UT.dat_dir(), 'paco', '0.025eV.txt')
        elif mnu == 0.05: 
            fpaco = os.path.join(UT.dat_dir(), 'paco', '0.050eV.txt')
        elif mnu == 0.075: 
            fpaco = os.path.join(UT.dat_dir(), 'paco', '0.075eV.txt')
        elif mnu == 0.1: 
            fpaco = os.path.join(UT.dat_dir(), 'paco', '0.100eV.txt')
        elif mnu == 0.125: 
            fpaco = os.path.join(UT.dat_dir(), 'paco', '0.125eV.txt')
        k, pmk = np.loadtxt(fpaco, unpack=True, usecols=[0,1]) 
        fpmk = interp1d(k, pmk, kind='cubic') 
        return fpmk(karr) 
    else: 
        raise NotImplementedError


def dfgdtheta(theta, npoints=5, flag=None): 
    ''' derivative of growth factor w.r.t. to theta 
    '''
    if theta == 'Mnu': 
        return dfgdMnu(npoints=npoints, flag=flag) 
    else:
        h_dict = {'Ob': 0.002, 'Om': 0.02, 'h': 0.04, 'ns': 0.04, 's8': 0.03} 
        h = h_dict[theta]
        fg_m = fgrowth_theta('%s_m' % theta) 
        fg_p = fgrowth_theta('%s_p' % theta) 
        return (fg_p - fg_m)/h 


def dfgdMnu(npoints=5, flag=None):
    if npoints == 1: 
        mnus = [0.0, 0.025]
        coeffs = [-1., 1.]
        fdenom = 1. * 0.025
    elif npoints == 2:  
        mnus = [0.0, 0.025, 0.05]
        coeffs = [-3., 4., -1.]
        fdenom = 2. * 0.025
    elif npoints == 3:  
        mnus = [0.0, 0.025, 0.05, 0.075]
        coeffs = [-11., 18., -9., 2.]
        fdenom = 6. * 0.025
    elif npoints == 4:  
        mnus = [0.0, 0.025, 0.05, 0.075, 0.1]
        coeffs = [-25., 48., -36., 16., -3.]
        fdenom = 12.* 0.025
    elif npoints == 5:  
        mnus = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125]
        coeffs = [-137., 300., -300., 200., -75., 12.]
        fdenom = 60.* 0.025
    elif npoints == 'quijote': 
        mnus = [0.0, 0.1, 0.2, 0.4]
        coeffs = [-21., 32., -12., 1.]
        fdenom = 1.2 
    elif npoints == '0.1eV':
        mnus = [0.075, 0.125]
        coeffs = [-1., 1.]
        fdenom = 0.05 
    else: 
        raise ValueError
    
    dfg = 0.
    for i, mnu, coeff in zip(range(len(mnus)), mnus, coeffs): 
        fg = fgrowth_Mnu(mnu, flag=flag) 
        dfg += coeff * fg 
    dfg /= fdenom 
    return dfg


def fgrowth_Mnu(mnu, flag=None): 
    ''' given theta read in growth factor at z=0 from CLASS background file
    '''
    if flag is None: 
        if mnu == 0.0: 
            fema = os.path.join(UT.dat_dir(), 'lt', 'output', '0eV_background.dat')
        elif mnu == 0.025: 
            fema = os.path.join(UT.dat_dir(), 'lt', 'output', '0p025eV_background.dat')
        elif mnu == 0.05: 
            fema = os.path.join(UT.dat_dir(), 'lt', 'output', '0p05eV_background.dat')
        elif mnu == 0.075: 
            fema = os.path.join(UT.dat_dir(), 'lt', 'output', '0p075eV_background.dat')
        elif mnu == 0.1: 
            fema = os.path.join(UT.dat_dir(), 'lt', 'output', '0p1eV_background.dat')
        elif mnu == 0.125: 
            fema = os.path.join(UT.dat_dir(), 'lt', 'output', '0p125eV_background.dat')
        elif mnu == 0.2: 
            fema = os.path.join(UT.dat_dir(), 'lt', 'output', '0p2eV_background.dat')
        elif mnu == 0.4: 
            fema = os.path.join(UT.dat_dir(), 'lt', 'output', '0p4eV_background.dat')
    elif flag == 'ema': 
        if mnu == 0.0: 
            fema = os.path.join(UT.dat_dir(), 'ema', 'HADES_0p00eV_z1_background.dat')
        elif mnu == 0.025: 
            fema = os.path.join(UT.dat_dir(), 'ema', 'HADES_0p025eV_z1_background.dat')
        elif mnu == 0.05: 
            fema = os.path.join(UT.dat_dir(), 'ema', 'HADES_0p05eV_z1_background.dat')
        elif mnu == 0.075: 
            fema = os.path.join(UT.dat_dir(), 'ema', 'HADES_0p075eV_z1_background.dat')
        elif mnu == 0.1: 
            fema = os.path.join(UT.dat_dir(), 'ema', 'HADES_0p10eV_z1_background.dat')
        elif mnu == 0.125: 
            fema = os.path.join(UT.dat_dir(), 'ema', 'HADES_0p125eV_z1_background.dat')
    else: 
        raise NotImplementedError
    fgs = np.loadtxt(fema, unpack=True, skiprows=4, usecols=[15]) 
    return fgs[-1]


def fgrowth_theta(theta): 
    ''' given theta read in growth factor at z=0 from CLASS background file
    '''
    if theta not in ['Ob_m', 'Ob_p', 'Om_m', 'Om_p', 'h_m', 'h_p', 'ns_m', 'ns_p', 's8_m', 's8_p',
            'fid_As', 'ns_m_As', 'ns_p_As' ]: 
        raise ValueError 
    f = os.path.join(UT.dat_dir(), 'lt', 'output', '%s_background.dat' % theta)
    fgs = np.loadtxt(f, unpack=True, skiprows=4, usecols=[15]) 
    return fgs[-1]


def dPrsdbdtheta(theta, b, karr, Prsdb, npoints=5, flag=None):  
    ''' get derivative d P / d theta w/ bias b and RSD 
    '''
    Pcb = _Pm_Mnu(0.0, karr, flag='cb') # fiducial Pcb = Pm  
    dlogPcb = dPmdtheta(theta, karr, log=True, npoints=npoints, flag='cb')

    fg = fgrowth_Mnu(0.0, flag=flag) # fiducial f_growth 
    dfg = dfgdtheta(theta, npoints=npoints, flag=flag)

    dPrsdb = Prsdb * dlogPcb + (2./3. * b + 0.4 * fg) * dfg * Pcb 
    return dPrsdb
