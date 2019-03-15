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


def quijoteBk(theta, rsd=True, flag=None): 
    ''' read in real/redshift-space bispectrum for specified model (theta) of the quijote simulations. 
    
    :param theta: 
        string that specifies which quijote run. `theta==fiducial`  
        for the fiducial set of parameters. 

    :param rsd: 
        boolean if False reads in real bispectrum  
    
    :param flag: 
        for more specific runs such as flag = '.fixed_nbar' 

    :return _bks: 
        dictionary that contains all the bispectrum data from quijote simulation
    '''
    fbk = os.path.join(UT.dat_dir(), 'bispectrum', 
            'quijote_%s%s%s.hdf5' % (theta, ['.real', ''][rsd], [flag, ''][flag is None])) 
    bks = h5py.File(fbk, 'r') 

    _bks = {}
    for k in bks.keys(): 
        _bks[k] = bks[k].value 
    return _bks



"""
    def Plk_halo(mneut, nreal, nzbin, zspace=False, mh_min=3200., Ngrid=360, Nbin=120): 
        ''' Return the powerspectrum multipoles for halo catalog with
        total neutrino mass `mneut`, realization `nreal`, and redshift 
        bin `nzbin` in either real/redshift space. 
        '''
        if zspace: str_space = 'z' # redshift space 
        else: str_space = 'r' # real 
        fpk = ''.join([UT.dat_dir(), 'plk/', 
            'plk.groups.', str(mneut), 'eV.', str(nreal), '.nzbin', str(nzbin), 
            '.', str_space, 'space', 
            '.mhmin', str(mh_min), 
            '.Nmesh', str(Ngrid), 
            '.Nbin', str(Nbin), 
            '.dat']) 
        if not os.path.isfile(fpk): 
            raise ValueError("--- %s does not exist ---" % fpk) 

        # read in plk 
        k,p0k,p2k,p4k = np.loadtxt(fpk, skiprows=3, unpack=True, usecols=[0,1,2,3]) 
        plk = {'k': k, 'p0k': p0k, 'p2k': p2k, 'p4k':p4k} 

        # readin shot-noise from header 
        with open(fpk) as lines: 
            for i_l, line in enumerate(lines):
                if i_l == 1: 
                    str_sn = line
                    break 
        plk['shotnoise'] = float(str_sn.strip().split('P_shotnoise')[-1])
        return plk 


    def Plk_halo_sigma8(sig8, nreal, nzbin, zspace=False, mh_min=3200., Ngrid=360, Nbin=120): 
        ''' Return the powerspectrum multipoles for halo catalog with
        total neutrino mass `mneut`, realization `nreal`, and redshift 
        bin `nzbin` in either real/redshift space. 
        '''
        if zspace: str_space = 'z' # redshift space 
        else: str_space = 'r' # real 
        fpk = ''.join([UT.dat_dir(), 'plk/', 
            'plk.groups.0.0eV.sig8', str(sig8), '.', str(nreal), '.nzbin', str(nzbin), 
            '.', str_space, 'space', 
            '.mhmin', str(mh_min), 
            '.Nmesh', str(Ngrid), 
            '.Nbin', str(Nbin), 
            '.dat']) 
        if not os.path.isfile(fpk): 
            raise ValueError("--- %s does not exist ---" % fpk) 

        # read in plk 
        k,p0k,p2k,p4k = np.loadtxt(fpk, skiprows=3, unpack=True, usecols=[0,1,2,3]) 
        plk = {'k': k, 'p0k': p0k, 'p2k': p2k, 'p4k':p4k} 

        # readin shot-noise from header 
        with open(fpk) as lines: 
            for i_l, line in enumerate(lines):
                if i_l == 1: 
                    str_sn = line
                    break 
        plk['shotnoise'] = float(str_sn.strip().split('P_shotnoise')[-1])
        return plk 


    def B123_halo(mneut, nreal, nzbin, Lbox=1000., zspace=False, mh_min=3200., Ngrid=360, Nmax=40, Ncut=3, step=3): 
        ''' return halo bispectrum 
        '''
        if zspace: str_space = 'z' # redshift space 
        else: str_space = 'r' # real 
        fbk = ''.join([UT.dat_dir(), 'bispectrum/', 
            'groups.', str(mneut), 'eV.', str(nreal), '.nzbin', str(nzbin), '.mhmin', str(mh_min),
            '.', str_space, 'space',
            '.Ngrid', str(Ngrid), 
            '.Nmax', str(Nmax),
            '.Ncut', str(Ncut),
            '.step', str(step), 
            '.pyfftw', 
            '.dat']) 
        if not os.path.isfile(fbk): 
            raise ValueError("--- %s does not exist ---" % fbk) 

        k_f = 2*np.pi/Lbox
        
        # read in file
        i_k, j_k, l_k, b123, q123, cnts = np.loadtxt(fbk, unpack=True, skiprows=1, usecols=[0,1,2,3,4,5]) 
        return i_k, j_k, l_k, b123, q123, cnts, k_f


    def B123_halo_sigma8(sig8, nreal, nzbin, Lbox=1000., zspace=False, mh_min=3200., Ngrid=360, Nmax=40, Ncut=3, step=3, silent=True): 
        ''' return halo bispectrum 
        '''
        if sig8 not in [0.822, 0.818, 0.807, 0.798]: 
            raise ValueError("sigma_8 = %f is not available" % sig8) 
        if zspace: str_space = 'z' # redhsift
        else: str_space = 'r' # real 
        # sigma 8 look up table

        fbk = ''.join([UT.dat_dir(), 'bispectrum/', 
            'groups.0.0eV.sig8_', str(sig8), '.', str(nreal), '.nzbin', str(nzbin), '.mhmin', str(mh_min),
            '.', str_space, 'space',
            '.Ngrid', str(Ngrid), 
            '.Nmax', str(Nmax),
            '.Ncut', str(Ncut),
            '.step', str(step), 
            '.pyfftw', 
            '.dat']) 
        if not os.path.isfile(fbk): 
            raise ValueError("--- %s does not exist ---" % fbk) 
        if not silent: 
            print('--- reading in %s ---' % fbk) 

        k_f = 2*np.pi/Lbox
        
        # read in file
        i_k, j_k, l_k, b123, q123, cnts = np.loadtxt(fbk, unpack=True, skiprows=1, usecols=[0,1,2,3,4,5]) 
        return i_k, j_k, l_k, b123, q123, cnts, k_f


    def cthreePCF_halo(mneut, nreal, nzbin, zspace=False, i_dr=0, nside=20, nbin=20, rmax=200., sim='hades', rrr_norm=False, factor=None): 
        ''' return compressed 3pcf following Eq. 72 of Slepian & Eisenstein (2015) 
        '''
        # read in 3pcf 
        tpcf = threePCF_halo(mneut, nreal, nzbin, zspace=zspace, i_dr=i_dr, nside=nside, nbin=nbin, sim=sim, rrr_norm=rrr_norm)
        ells = tpcf.keys() # list of ells
        ells.remove('meta') 
        
        r1_edge = np.linspace(0., rmax, nbin+1) 
        r1_mid = 0.5 * (r1_edge[1:] + r1_edge[:-1]) 
        r2_edge = np.linspace(0., rmax, nbin+1) 
        r2_mid = 0.5 * (r2_edge[1:] + r2_edge[:-1]) 

        dr = r1_edge[1] - r1_edge[0]
        dVr2 = (r2_edge[1:]**3 - r2_edge[:-1]**3)
        
        if factor is None: 
            factor = int(np.ceil(18./dr))
        
        ctpcf = {} 
        ctpcf['meta'] = tpcf['meta'].copy() 
        for ell in ells: 
            ctpcf[ell] = np.zeros(nbin)
            for i_r1 in range(nbin): 
                Sr1 = (r2_mid > factor*dr) & (r2_mid < (r1_mid[i_r1] - factor*dr)) 
                if np.sum(Sr1) > 0:  
                    zeta = tpcf[ell][i_r1, Sr1]
                    ctpcf[ell][i_r1] = (zeta * dVr2[Sr1]).sum() / dVr2[Sr1].sum() 
        return ctpcf 


    def threePCF_halo(mneut, nreal, nzbin, zspace=False, i_dr=0, nside=20, nbin=20, sim='hades', rrr_norm=False): 
        '''Return isotropic three point correlation function (3PCF) from Slepian & Eisenstein. 
        More specifically this function outputs the averaged NNN (N = D-R) 3pcf measurements.
        *** document this ***
        *** document this ***
        *** document this ***
        *** document this ***
        '''
        if isinstance(i_dr, str): # strng 
            if i_dr != 'all': raise ValueError
            i_drs = range(50)
        elif isinstance(i_dr, int): 
            i_drs = [i_dr] 
        elif isinstance(i_dr, list): 
            i_drs = i_dr
        else: 
            raise ValueError

        for ii, i in enumerate(i_drs): 
            tpcf_i = _threePCF_halo_i_dr(mneut, nreal, nzbin, zspace=zspace, 
                    i_dr=i, nside=nside, nbin=nbin, sim=sim)
            
            if ii == 0: 
                tpcfs = tpcf_i
                ells = tpcfs.keys()
                ells.remove('meta') 
            else: 
                for ell in ells:
                    tpcfs[ell] += tpcf_i[ell] 
                # I *should* check the meta data but bleh.
                    
        for ell in ells: 
            tpcfs[ell] /= float(len(i_drs))
        
        if rrr_norm: 
            # normalize by RRR_0
            rrr = _threePCF_halo_rrr(mneut, nzbin, zspace=zspace, nside=nside, nbin=nbin, sim=sim)
            for ell in ells: 
                tpcfs[ell] /= rrr[0]
        return tpcfs


    def _threePCF_halo_i_dr(mneut, nreal, nzbin, zspace=False, i_dr=0, nside=20, nbin=20, sim='hades'): 
        ''' Return isotropic three point correlation function (3PCF) from Slepian & Eisenstein. 
        More specifically this function reads in the i^th NNN (N = D-R) 3pcf calculation.
        To reduce shot-noise ~50 NNN 3pcf should be averaged. 
        '''
        # file name 
        f = _obvs_fname('3pcf', mneut, nreal, nzbin, zspace, i_dr=i_dr, nside=nside, nbin=nbin)

        # read in meta data 
        tpcf = {} 
        tpcf['meta'] = {} 
        with open(f) as fp: 
            # save meta data
            line = fp.readline() # box size
            tpcf['meta']['box'] = float(line.split('=')[1])
            line = fp.readline() # grid 
            tpcf['meta']['Ngrid'] = int(line.split('=')[1])
            line = fp.readline() # max radius 
            tpcf['meta']['max_radius'] = float(line.split('=')[1])
            line = fp.readline() # max radius in units of grid size 
            line = fp.readline() # number of bins 
            nbin = int(line.split('=')[1])
            tpcf['meta']['Nbin'] = nbin
            line = fp.readline() # orders 
            order = int(line.split('=')[1])
            tpcf['meta']['order'] = order 

            mpoles = [np.zeros((nbin, nbin)) for oo in range(order+1)]

            ii = 1 

            while line: 
                line = fp.readline() 
                if len(line.split()) < 3: 
                    continue
                if (line.split()[0] == 'Multipole') and (int(line.split()[2]) > 0): 
                    _ = fp.readline() # skip row 
                    for _i in range(ii):
                        line = fp.readline() 
                        ll = line.split()
                        i = int(ll[0])
                        j = int(ll[1])

                        for ell in range(order+1):
                            if ell == 0: 
                                amp = float(ll[2])
                                mpoles[ell][i,j] = amp 
                                mpoles[ell][j,i] = amp 
                            else:
                                mpoles[ell][i,j] = amp * float(ll[ell+2])
                                mpoles[ell][j,i] = amp * float(ll[ell+2])
                    ii += 1
            
            for ell, mpole in zip(range(order+1), mpoles):
                tpcf[ell] = mpole
        return tpcf 


    def _threePCF_halo_rrr(mneut, nzbin, zspace=False, nside=20, nbin=20, sim='hades'): 
        ''' Read in the RRR normalization from randoms 
        '''
        # file name 
        f = _obvs_fname('rrr', mneut, 1, nzbin, zspace, nside=nside, nbin=nbin)

        # read in meta data 
        tpcf = {} 
        tpcf['meta'] = {} 
        with open(f) as fp: 
            # save meta data
            line = fp.readline() # box size
            tpcf['meta']['box'] = float(line.split('=')[1])
            line = fp.readline() # grid 
            tpcf['meta']['Ngrid'] = int(line.split('=')[1])
            line = fp.readline() # max radius 
            tpcf['meta']['max_radius'] = float(line.split('=')[1])
            line = fp.readline() # max radius in units of grid size 
            line = fp.readline() # number of bins 
            nbin = int(line.split('=')[1])
            tpcf['meta']['Nbin'] = nbin
            line = fp.readline() # orders 
            order = int(line.split('=')[1])
            tpcf['meta']['order'] = order 

            mpoles = [np.zeros((nbin, nbin)) for oo in range(order+1)]

            ii = 1 

            while line: 
                line = fp.readline() 
                if len(line.split()) < 3: 
                    continue
                if (line.split()[0] == 'Multipole') and (int(line.split()[2]) > 0): 
                    _ = fp.readline() # skip row 
                    for _i in range(ii):
                        line = fp.readline() 
                        ll = line.split()
                        i = int(ll[0])
                        j = int(ll[1])

                        for ell in range(order+1):
                            if ell == 0: 
                                amp = float(ll[2])
                                mpoles[ell][i,j] = amp 
                                mpoles[ell][j,i] = amp 
                            else:
                                mpoles[ell][i,j] = amp * float(ll[ell+2])
                                mpoles[ell][j,i] = amp * float(ll[ell+2])
                    ii += 1
            
            for ell, mpole in zip(range(order+1), mpoles):
                tpcf[ell] = mpole
        return tpcf 


    def _obvs_fname(obvs, mneut, nreal, nzbin, zspace, **kwargs): 
        if obvs not in ['plk', '3pcf', 'rrr']: 
            raise ValueError('Currently we only support Pl(k) and 3PCF') 

        if zspace: str_space = 'z' # redhsift
        else: str_space = 'r' # real 
        
        if obvs == 'plk': 
            f = ''.join([UT.dat_dir(), 'plk/', 
                'plk.groups.', str(mneut), 'eV.', str(nreal), '.nzbin', str(nzbin), #'.nmesh', str(kwargs['Nmesh']), 
                '.', str_space, 'space.dat']) 
        elif obvs == '3pcf': 
            if 'nside' not in kwargs.keys():  
                raise ValueError
            if 'nbin' not in kwargs.keys():  
                raise ValueError
            f = ''.join([UT.dat_dir(), '3pcf/', 
                '3pcf.groups.', str(mneut), 'eV.', str(nreal), '.nzbin', str(nzbin), 
                '.nside', str(kwargs['nside']), 
                '.nbin', str(kwargs['nbin']), 
                '.', str_space, 'space.nnn', str(kwargs['i_dr']), '.dat']) 
        elif obvs == 'rrr': 
            # 3pcf.groups.0.06eV.nzbin4.r0.nside20.nbin20.rspace.rrr.dat
            if 'nside' not in kwargs.keys():  
                raise ValueError
            if 'nbin' not in kwargs.keys():  
                raise ValueError
            f = ''.join([UT.dat_dir(), '3pcf/', 
                '3pcf.groups.', str(mneut), 'eV.nzbin', str(nzbin), '.r0', 
                '.nside', str(kwargs['nside']), 
                '.nbin', str(kwargs['nbin']), 
                '.', str_space, 'space.rrr.dat']) 

        #if not os.path.isfile(f): 
        #    raise ValueError('%s does not exist' % f) 
        return f 
"""
