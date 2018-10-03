'''

code for dealing with observables

'''
import os 
import numpy as np 

import util as UT 


def Plk_halo(mneut, nreal, nzbin, zspace=False, sim='hades'): 
    ''' Return the powerspectrum multipoles for halo catalog with
    total neutrino mass `mneut`, realization `nreal`, and redshift 
    bin `nzbin` in either real/redshift space. 
    '''
    f = _obvs_fname('plk', mneut, nreal, nzbin, zspace)
    # read in plk 
    k,p0k,p2k,p4k = np.loadtxt(f, skiprows=3, unpack=True, usecols=[0,1,2,3]) 
    plk = {'k': k, 'p0k': p0k, 'p2k': p2k, 'p4k':p4k} 

    # readin shot-noise from header 
    with open(f) as lines: 
        for i_l, line in enumerate(lines):
            if i_l == 1: 
                str_sn = line
                break 
    plk['shotnoise'] = float(str_sn.strip().split('P_shotnoise')[-1])
    return plk 


def threePCF_halo(mneut, nreal, nzbin, zspace=False, i_dr=0, nside=20, nbin=20, sim='hades'): 
    if isinstance(i_dr, str): # strng 
        if i_dr != 'all': raise ValueError
        i_drs = range(50)
    else: 
        i_drs = [i_dr] 

    for ii, i in enumerate(i_drs): 
        tpcf_i = _threePCF_halo_i_dr(mneut, nreal, nzbin, zspace=zspace, 
                i_dr=i, nside=nside, nbin=nbin, sim=sim)
        
        if ii == 0: 
            tpcfs = tpcf_i
        else: 
            for ell in tpcfs.keys():
                if ell != 'meta': 
                    tpcfs[ell] += tpcf_i[ell] 


def _threePCF_halo_i_dr(mneut, nreal, nzbin, zspace=False, i_dr=0, nside=20, nbin=20, sim='hades'): 
    ''' Return isotropic three point correlation function (3PCF) from Slepian & Eisenstein. 
    This function reads in the i^th NNN (N = D-R) 3pcf calculation.  
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


def _obvs_fname(obvs, mneut, nreal, nzbin, zspace, **kwargs): 
    if obvs not in ['plk', '3pcf']: 
        raise ValueError('Currently we only support Pl(k) and 3PCF') 

    if mneut == 0.1: str_mneut = '0.10eV'
    else: str_mneut = str(mneut)+'eV'

    if zspace: str_space = 'z' # redhsift
    else: str_space = 'r' # real 
    
    if obvs == 'plk': 
        f = ''.join([UT.dat_dir(), 'plk.', 
            'plk.groups.', str(mneut), 'eV.', str(nreal), '.nzbin', str(nzbin), 
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
            '.', str_space, 'space.d_r', str(kwargs['i_dr']), 'dat']) 

    if not os.path.isfile(f): 
        raise ValueError('%s does not exist' % f) 
    return f 
