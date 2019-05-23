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


def quijoteBk(theta, z=0, rsd=True, flag=None): 
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
    assert flag in [None, '.fixed_nbar', 'ncv', 'reg'], "flag=%s unspecified" % flag

    if z == 0: zdir = 'z0'
    else: raise NotImplementedError
    
    quij_dir = os.path.join(UT.dat_dir(), 'bispectrum', 'quijote', zdir) 
    if rsd != 'real':  # reshift space 
        if flag is None: fbk = os.path.join(quij_dir, 'quijote_%s.hdf5' % theta)
        else: 
            if rsd in [0, 1, 2]: 
                fbk = os.path.join(quij_dir, 'quijote_%s.%s.rsd%i.hdf5' % (theta, flag, rsd))
            else: raise ValueError
    else: 
        if flag is None: fbk = os.path.join(quij_dir, 'quijote_%s.real.hdf5' % theta)
        else: fbk = os.path.join(quij_dir, 'quijote_%s.%s.real.hdf5' % (theta, flag))
    bks = h5py.File(fbk, 'r') 

    _bks = {}
    for k in bks.keys(): 
        _bks[k] = bks[k].value 
    return _bks


def quijoteP0k(theta): 
    ''' read in redshift-space power spectrum for specified model (theta) of the quijote simulations. 
    
    :param theta: 
        string that specifies which quijote run. `theta==fiducial`  
        for the fiducial set of parameters. 

    :return _pks: 
        dictionary that contains all the bispectrum data from quijote simulation
    '''
    fpk = os.path.join(UT.dat_dir(), 'bispectrum', 'quijote_Pk_%s.hdf5' % theta)
    pks = h5py.File(fpk, 'r') 

    _pks = {}
    for k in pks.keys(): 
        _pks[k] = pks[k].value 
    return _pks
