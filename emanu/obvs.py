'''

IO for observables 

'''
import os 
import h5py 
import numpy as np 
from . import util as UT 


def hadesPlk(mneut, nzbin=4, rsd=True):  
    ''' read in real/rsd halo powerspectrum of hades simulations with Mnu = mneut eV at z = zbin[nzbin] 

    :param mneut: 
        neutrino mass [0.0, 0.06, 0.1, 0.15] 

    :param nzbin: (default: 4) 
        redshift bin index. At the moment only 4 is implemented. 
    
    :param rsd: (default: True) 
        boolean on whether real- or redshift-space. 
    
    :return _pks: 
        returns a dictionary with the bispectrum values 
    '''
    assert mneut in [0.0, 0.06, 0.1, 0.15]
    fpks = os.path.join(UT.dat_dir(), 'plk', 
            'hades.%seV.nzbin%i.mhmin3200.0.%sspace.hdf5' % (str(mneut), nzbin, ['r', 'z'][rsd]))
    pks = h5py.File(fpks, 'r') 
    
    _pks = {}
    for k in pks.keys(): 
        _pks[k] = pks[k].value 
    return _pks


def hadesPlk_s8(sig8, nzbin=4, rsd=True):  
    ''' read in sigma8 matched 0eV real/rsd halo powerspectrum of hades simulations 
    at z = zbin[nzbin] 

    :param sig8: 
        sigma_8 values [0.822, 0.818, 0.807, 0.798]

    :param nzbin: (default: 4) 
        redshift bin index. At the moment only 4 is implemented. 
    
    :param rsd: (default: True) 
        boolean on whether real- or redshift-space. 
    
    :return _bks: 
        returns a dictionary with the bispectrum values 
    '''
    assert sig8 in [0.822, 0.818, 0.807, 0.798]

    fpks = os.path.join(UT.dat_dir(), 'plk', 
            'hades.0.0eV.sig8_%.3f.nzbin%i.mhmin3200.0.%sspace.hdf5' % 
            (sig8, nzbin, ['r', 'z'][rsd]))
    pks = h5py.File(fpks, 'r') 
    
    _pks = {}
    for k in pks.keys(): 
        _pks[k] = pks[k].value 
    return _pks


def hadesBk(mneut, nzbin=4, rsd=True): 
    ''' read in real/rsd halo bispectrum of hades simulations with Mnu = mneut eV at z = zbin[nzbin] 

    :param mneut: 
        neutrino mass [0.0, 0.06, 0.1, 0.15] 

    :param nzbin: (default: 4) 
        redshift bin index. At the moment only 4 is implemented. 
    
    :param rsd: (default: True) 
        boolean on whether real- or redshift-space. 
    
    :return _bks: 
        returns a dictionary with the bispectrum values 
    '''
    assert mneut in [0.0, 0.06, 0.1, 0.15]
    fbks = os.path.join(UT.dat_dir(), 'bispectrum', 
            'hades.%seV.nzbin%i.mhmin3200.0.%sspace.hdf5' % (str(mneut), nzbin, ['r', 'z'][rsd]))
    bks = h5py.File(fbks, 'r') 
    
    _bks = {}
    for k in bks.keys(): 
        _bks[k] = bks[k].value 
    return _bks


def hadesBk_s8(sig8, nzbin=4, rsd=True): 
    ''' read in sigma8 matched 0eV real/rsd halo bispectrum of hades simulations 
    at z = zbin[nzbin] 

    :param sig8: 
        sigma_8 values [0.822, 0.818, 0.807, 0.798]

    :param nzbin: (default: 4) 
        redshift bin index. At the moment only 4 is implemented. 
    
    :param rsd: (default: True) 
        boolean on whether real- or redshift-space. 
    
    :return _bks: 
        returns a dictionary with the bispectrum values 
    '''
    assert sig8 in [0.822, 0.818, 0.807, 0.798]
    fbks = os.path.join(UT.dat_dir(), 'bispectrum', 
            'hades.0.0eV.sig8_%s.nzbin%i.mhmin3200.0.%sspace.hdf5' % (str(sig8), nzbin, ['r', 'z'][rsd]))
    bks = h5py.File(fbks, 'r') 

    _bks = {}
    for k in bks.keys(): 
        _bks[k] = bks[k].value 
    return _bks


def quijoteBk(theta, z=0, rsd=True, flag=None, silent=True): 
    ''' read in real/redshift-space bispectrum for specified model (theta) of the quijote simulations. 
    
    :param theta: 
        string that specifies which quijote run. `theta==fiducial`  
        for the fiducial set of parameters. 

    :param rsd: (default: True) 
        if True, read in z-space bispectrum (including all 3 RSD directions). 
        if False, read in real-space bispectrum
        if int, rsd specifies the direction of the RSD 

    :param z: (default:0) 
        redshift z=0. currently only supports z=0 
    
    :param flag: (default: None) 
        specify more specific bispectrum runs. currently only flags within 
        [None, 'ncv', 'reg']  are supported. ncv only includes paired-fixed 
        simulations. reg only includes regular n-body simulations.

    :return _bks: 
        dictionary that contains all the bispectrum data from quijote simulation
    '''
    if z == 0: zdir = 'z0'
    else: raise NotImplementedError
    
    assert flag in [None, 'ncv', 'reg'], "flag=%s unspecified" % flag
    if flag is None: assert rsd in [True, 'real'] 
    
    quij_dir = os.path.join(UT.dat_dir(), 'bispectrum', 'quijote', zdir) 
    if (type(rsd) == bool) and rsd: # rsd=True (include all 3 rsd directions) 
        if flag is None: # combine ncv and reg. n-body sims 
            fbks = [] 
            for fl in ['ncv', 'reg']: 
                for irsd in [0, 1, 2]: 
                    fbks.append('quijote_%s.%s.rsd%i.hdf5' % (theta, fl, irsd))
        else: 
            fbks = ['quijote_%s.%s.rsd%i.hdf5' % (theta, flag, irsd) for irsd in [0, 1, 2]] 
    elif rsd == 'real': # real-space 
        if flag is None: # combine ncv and reg. n-body sims 
            fbks = [] 
            for fl in ['ncv', 'reg']: 
                fbks.append('quijote_%s.%s.real.hdf5' % (theta, fl))
        else: fbks = ['quijote_%s.%s.real.hdf5' % (theta, flag)]
    elif rsd in [0, 1, 2]:  
        fbks = ['quijote_%s.%s.rsd%i.hdf5' % (theta, flag, rsd)]

    if not silent: print(fbks) 
    for i_f, fbk in enumerate(fbks): 
        bks = h5py.File(os.path.join(quij_dir, fbk), 'r') 
        if i_f == 0:  
            _bks = {}
            for k in bks.keys(): 
                _bks[k] = bks[k].value 
        else: 
            for k in ['p0k1', 'p0k2', 'p0k3', 'b123', 'b_sn', 'q123']: 
                _bks[k] = np.concatenate([_bks[k], bks[k].value]) 
    return _bks


def quijoteP0k(theta): 
    ''' read in redshift-space power spectrum for specified model (theta) of the quijote simulations. 
    
    :param theta: 
        string that specifies which quijote run. `theta==fiducial`  
        for the fiducial set of parameters. 

    :return _pks: 
        dictionary that contains all the bispectrum data from quijote simulation
    '''
    fpk = os.path.join(UT.dat_dir(), 'powerspectrum', 'quijote', 'z0', 'quijote_Pk_%s.hdf5' % theta)
    pks = h5py.File(fpk, 'r') 

    _pks = {}
    for k in pks.keys(): 
        _pks[k] = pks[k].value 
    return _pks
