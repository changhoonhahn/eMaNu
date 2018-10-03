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


def cthreePCF_halo(mneut, nreal, nzbin, zspace=False, i_dr=0, nside=20, nbin=20, rmax=200., sim='hades'): 
    ''' return compressed 3pcf following Eq. 72 of Slepian & Eisenstein (2015) 
    '''
    # read in 3pcf 
    tpcf = threePCF_halo(mneut, nreal, nzbin, zspace=zspace, i_dr=i_dr, nside=nside, nbin=nbin, sim=sim)
    ells = tpcf.keys() # list of ells
    ells.remove('meta') 
    
    r1_edge = np.linspace(0., rmax, nbin+1) 
    r1_mid = 0.5 * (r1_edge[1:] + r1_edge[:-1]) 
    r2_edge = np.linspace(0., rmax, nbin+1) 
    r2_mid = 0.5 * (r2_edge[1:] + r2_edge[:-1]) 

    dr = r1_edge[1] - r1_edge[0]
    dVr2 = (r2_edge[1:]**3 - r2_edge[:-1]**3)

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


def threePCF_halo(mneut, nreal, nzbin, zspace=False, i_dr=0, nside=20, nbin=20, sim='hades'): 
    '''Return isotropic three point correlation function (3PCF) from Slepian & Eisenstein. 
    More specifically this function outputs the averaged NNN (N = D-R) 3pcf measurements.
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
            '.', str_space, 'space.nnn', str(kwargs['i_dr']), '.dat']) 

    if not os.path.isfile(f): 
        raise ValueError('%s does not exist' % f) 
    return f 
