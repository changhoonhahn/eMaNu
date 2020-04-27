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
        _pks[k] = pks[k][...]
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
        _pks[k] = pks[k][...]
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
        _bks[k] = bks[k][...]
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
        _bks[k] = bks[k][...]
    return _bks


def quijhod_Bk(theta, z=0, rsd='all', flag=None, silent=True): 
    ''' read in real/redshift-space bispectrum for specified model (theta) of the 
    quijote HOD (quijhod) catalogs. 
    
    :param theta: 
        string that specifies which quijote run. `theta==fiducial`  
        for the fiducial set of parameters. 

    :param rsd: (default: 'all') 
        if rsd == 'all', read in z-space bispectrum (including all 3 RSD directions). 
        if rsd in [0, 1, 2], read in z-space bispecturm in one RSD direction 
        if False, read in real-space bispectrum

    :param z: (default:0) 
        redshift z=0. currently only supports z=0 
    
    :param flag: (default: None) 
        specify more specific bispectrum runs. currently only flags within 
        [None, 'ncv', 'reg']  are supported. ncv only includes paired-fixed 
        simulations. reg only includes regular n-body simulations.

    :return _bks: 
        dictionary that contains all the bispectrum data from quijhod 
    '''
    if z == 0: zdir = 'z0'
    else: raise NotImplementedError
    
    assert flag in [None, 'ncv', 'reg'], "flag=%s unspecified" % flag
    assert rsd in [0, 1, 2, 'all', 'real'] 
    
    # get files to read-in 
    if 'machine' in os.environ and os.environ['machine'] == 'mbp': 
        quij_dir = os.path.join(UT.dat_dir(), 'Galaxies')
        if rsd == 'all': # include all 3 rsd directions
            fbks = ['quijhod_B_%s.%s.rsd%i.hdf5' % (theta, flag, irsd) for irsd in [0, 1, 2]] 
        elif rsd in [0, 1, 2]: # include a single rsd direction 
            fbks = ['quijhod_B_%s.%s.rsd%i.hdf5' % (theta, flag, rsd)]
        elif rsd == 'real': # real-space 
            fbks = ['quijhod_B_%s.%s.real.hdf5' % (theta, flag)]
    else: 
        quij_dir = os.path.join(UT.dat_dir(), 'bispectrum', 'quijote_hod', zdir) 
        if rsd == 'all': # include all 3 rsd directions
            fbks = ['quijhod_%s.%s.rsd%i.hdf5' % (theta, flag, irsd) for irsd in [0, 1, 2]] 
        elif rsd in [0, 1, 2]: # include a single rsd direction 
            fbks = ['quijhod_%s.%s.rsd%i.hdf5' % (theta, flag, rsd)]
        elif rsd == 'real': # real-space 
            fbks = ['quijhod_%s.%s.real.hdf5' % (theta, flag)]
    
    # combine the files  
    if not silent: print(fbks) 
    _bks = {}
    for i_f, fbk in enumerate(fbks): 
        bks = h5py.File(os.path.join(quij_dir, fbk), 'r') 
        if i_f == 0:  
            for k in bks.keys(): 
                _bks[k] = bks[k][...]
        else: 
            for k in ['p0k1', 'p0k2', 'p0k3', 'b123', 'b_sn', 'q123', 'Ngalaxies']: 
                _bks[k] = np.concatenate([_bks[k], bks[k][...]]) 
    return _bks


def quijhod_Pk(theta, z=0, rsd='all', flag=None, silent=True): 
    ''' read in real/redshift-space powerspectrum monopole (for RSD quadru- and hexadeca-poles) 
    for specified model (theta) of the quijote hod (quijhod) simulations. 
    
    :param theta: 
        string that specifies which quijote run. `theta==fiducial`  
        for the fiducial set of parameters. 

    :param z: (default:0) 
        redshift z=0. currently only supports z=0 

    :param rsd: (default: 'all') 
        if rsd == 'all', read in z-space bispectrum (including all 3 RSD directions). 
        if rsd in [0, 1, 2], read in z-space bispecturm in one RSD direction 
        if False, read in real-space bispectrum
    
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
    assert rsd in [0, 1, 2, 'all', 'real'] 
    
    # get files to read-in 
    if 'machine' in os.environ and os.environ['machine'] == 'mbp': 
        quij_dir = os.path.join(UT.dat_dir(), 'Galaxies') 
        if rsd == 'all': # include all 3 rsd directions
            fpks = ['quijhod_P_%s.%s.rsd%i.hdf5' % (theta, flag, irsd) for irsd in [0, 1, 2]] 
        elif rsd in [0, 1, 2]: # include a single rsd direction 
            fpks = ['quijhod_P_%s.%s.rsd%i.hdf5' % (theta, flag, rsd)]
        elif rsd == 'real': # real-space 
            fpks = ['quijhod_P_%s.%s.real.hdf5' % (theta, flag)]
    else: 
        quij_dir = os.path.join(UT.dat_dir(), 'powerspectrum', 'quijote_hod', zdir) 
        if rsd == 'all': # include all 3 rsd directions
            fpks = ['quijhod_%s.%s.rsd%i.hdf5' % (theta, flag, irsd) for irsd in [0, 1, 2]] 
        elif rsd in [0, 1, 2]: # include a single rsd direction 
            fpks = ['quijhod_%s.%s.rsd%i.hdf5' % (theta, flag, rsd)]
        elif rsd == 'real': # real-space 
            fpks = ['quijhod_%s.%s.real.hdf5' % (theta, flag)]
        
    # combine the files  
    if not silent: print(fpks) 
    _pks = {}
    for i_f, fpk in enumerate(fpks): 
        pks = h5py.File(os.path.join(quij_dir, fpk), 'r') 
        if i_f == 0:  
            for k in pks.keys(): 
                _pks[k] = pks[k][...]
        else: 
            for k in ['p0k', 'p2k', 'p4k', 'p_sn']: 
                _pks[k] = np.concatenate([_pks[k], pks[k][...]]) 
    return _pks


def quijoteBk(theta, z=0, rsd='all', flag=None, silent=True): 
    ''' read in real/redshift-space bispectrum for specified model (theta) of the quijote simulations. 
    
    :param theta: 
        string that specifies which quijote run. `theta==fiducial`  
        for the fiducial set of parameters. 

    :param rsd: (default: 'all') 
        if rsd == 'all', read in z-space bispectrum (including all 3 RSD directions). 
        if rsd in [0, 1, 2], read in z-space bispecturm in one RSD direction 
        if False, read in real-space bispectrum

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
    assert rsd in [0, 1, 2, 'all', 'real'] 
    
    # get files to read-in 
    quij_dir = os.path.join(UT.dat_dir(), 'bispectrum', 'quijote', zdir) 
    if rsd == 'all': # include all 3 rsd directions
        if flag is None: # combine ncv and reg. n-body sims (no advised) 
            fbks = [] 
            for fl in ['ncv', 'reg']: 
                for irsd in [0, 1, 2]: 
                    fbks.append('quijote_%s.%s.rsd%i.hdf5' % (theta, fl, irsd))
        else: 
            fbks = ['quijote_%s.%s.rsd%i.hdf5' % (theta, flag, irsd) for irsd in [0, 1, 2]] 
    elif rsd in [0, 1, 2]: # include a single rsd direction 
        fbks = ['quijote_%s.%s.rsd%i.hdf5' % (theta, flag, rsd)]
    elif rsd == 'real': # real-space 
        if flag is None: # combine ncv and reg. n-body sims 
            fbks = [] 
            for fl in ['ncv', 'reg']: 
                fbks.append('quijote_%s.%s.real.hdf5' % (theta, fl))
        else: fbks = ['quijote_%s.%s.real.hdf5' % (theta, flag)]
    
    # combine the files  
    if not silent: print(fbks) 
    _bks = {}
    for i_f, fbk in enumerate(fbks): 
        bks = h5py.File(os.path.join(quij_dir, fbk), 'r') 
        if i_f == 0:  
            for k in bks.keys(): 
                _bks[k] = bks[k][...] 
        else: 
            for k in ['p0k1', 'p0k2', 'p0k3', 'b123', 'b_sn', 'q123', 'Nhalos']: 
                _bks[k] = np.concatenate([_bks[k], bks[k][...]]) 
    return _bks


def quijotePk(theta, z=0, rsd='all', flag=None, silent=True): 
    ''' read in real/redshift-space powerspectrum monopole (for RSD quadru- and hexadeca-poles) 
    for specified model (theta) of the quijote simulations. 
    
    :param theta: 
        string that specifies which quijote run. `theta==fiducial`  
        for the fiducial set of parameters. 

    :param z: (default:0) 
        redshift z=0. currently only supports z=0 

    :param rsd: (default: 'all') 
        if rsd == 'all', read in z-space bispectrum (including all 3 RSD directions). 
        if rsd in [0, 1, 2], read in z-space bispecturm in one RSD direction 
        if False, read in real-space bispectrum
    
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
    assert rsd in [0, 1, 2, 'all', 'real'] 
    
    # get files to read-in 
    quij_dir = os.path.join(UT.dat_dir(), 'powerspectrum', 'quijote', zdir) 
    if rsd == 'all': # include all 3 rsd directions
        if flag is None: # combine ncv and reg. n-body sims (no advised) 
            fpks = [] 
            for fl in ['ncv', 'reg']: 
                for irsd in [0, 1, 2]: 
                    fpks.append('quijote_%s.%s.rsd%i.hdf5' % (theta, fl, irsd))
        else: 
            fpks = ['quijote_%s.%s.rsd%i.hdf5' % (theta, flag, irsd) for irsd in [0, 1, 2]] 
    elif rsd in [0, 1, 2]: # include a single rsd direction 
        fpks = ['quijote_%s.%s.rsd%i.hdf5' % (theta, flag, rsd)]
    elif rsd == 'real': # real-space 
        if flag is None: # combine ncv and reg. n-body sims 
            fpks = [] 
            for fl in ['ncv', 'reg']: 
                fpks.append('quijote_%s.%s.real.hdf5' % (theta, fl))
        else: fpks = ['quijote_%s.%s.real.hdf5' % (theta, flag)]
    
    # combine the files  
    if not silent: print(fpks) 
    _pks = {}
    for i_f, fpk in enumerate(fpks): 
        pks = h5py.File(os.path.join(quij_dir, fpk), 'r') 
        if i_f == 0:  
            for k in pks.keys(): 
                _pks[k] = pks[k][...]
        else: 
            for k in ['p0k', 'p2k', 'p4k', 'p_sn']: 
                _pks[k] = np.concatenate([_pks[k], pks[k][...]]) 
    return _pks
