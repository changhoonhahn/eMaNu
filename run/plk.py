#!/bin/python
import os 
import sys 
import numpy as np 
# -- nbodykit -- 
import nbodykit.lab as NBlab
# -- eMaNu -- 
from emanu import util as UT
from emanu.hades import data as Dat
from emanu import forwardmodel as FM


def hadesHalo_Plk(mneut, nreal, nzbin, zspace=False, Lbox=1000., mh_min=3200., Nmesh=360, overwrite=False): 
    ''' Calculate the powerspectrum multipoles for Paco's Neutrio 
    halo catalogs

    notes
    -----
    * Takes ~20 seconds.
    '''
    if zspace: raise NotImplementedError
    else: str_space = 'r' 
    # P_l(k) output file 
    fplk = ''.join([UT.dat_dir(), 'plk/', 
        'plk.', 
        'groups.', str(mneut), 'eV.', str(nreal), '.nzbin', str(nzbin), '.', str_space, 'space',
        '.mhmin', str(mh_min), 
        'Nmesh', str(Nmesh), 
        '.dat']) 

    if os.path.isfile(fplk) and not overwrite: 
        print('--- already written to ---\n %s' % (fplk))
        return None 

    # read in Neutrino halo with mneut eV, realization # nreal, at z specified by nzbin 
    halos = Dat.NeutHalos(mneut, nreal, nzbin, mh_min=mh_min, silent=True) 
    # calculate P_l(k) 
    plk = FM.Observables(halos, observable='plk', rsd=zspace, dk=0.005, kmin=0.005, Nmesh=Nmesh)

    # header 
    hdr = ''.join(['P_l(k) measurements for m neutrino = ', str(mneut), ' eV, realization ', str(nreal), ', zbin ', str(nzbin), 
        '\n P_shotnoise ', str(plk['shotnoise']), 
        '\n cols: k, P_0, P_2, P_4']) 
    # write to file 
    np.savetxt(fplk, np.array([plk['k'], plk['p0k'], plk['p2k'], plk['p4k']]).T, header=hdr) 
    return None 


def hadesHalo_Plk_sigma8(sig8, nreal, nzbin, zspace=False, Lbox=1000., mh_min=3200., Nmesh=360, overwrite=False): 
    ''' Calculate the powerspectrum multipoles for Paco's Neutrino halo catalogs

    notes
    -----
    * Takes ~20 seconds.
    '''
    if zspace: raise NotImplementedError
    else: str_space = 'r' 
    if sig8 not in [0.822, 0.818, 0.807, 0.798]: 
        raise ValueError("sigma_8 value not available") 
    # plk output file 
    fplk = ''.join([UT.dat_dir(), 'plk/', 
        'plk.', 
        'groups.0.0eV.sig8', str(sig8), '.', str(nreal), '.nzbin', str(nzbin), '.', str_space, 'space',
        '.mhmin', str(mh_min),
        'Nmesh', str(Nmesh), 
        '.dat']) 

    if os.path.isfile(fplk) and not overwrite: 
        print('--- already written to ---\n %s' % (fplk))
        return None 
    
    # read in Neutrino halo with 0.0eV but with sigma_8 matched to some massive eV catalog, 
    # realization # nreal, at z specified by nzbin 
    halos = Dat.Sig8Halos(Sig8, nreal, nzbin, mh_min=mh_min, silent=True) 
    # calculate P_l(k) 
    plk = FM.Observables(halos, observable='plk', rsd=zspace, dk=0.005, kmin=0.005, Nmesh=Nmesh)
    # header 
    hdr = ''.join(['P_l(k) measurements for m neutrino = ', str(mneut), ' eV, realization ', str(nreal), ', zbin ', str(nzbin), 
        '\n P_shotnoise ', str(plk['shotnoise']), 
        '\n cols: k, P_0, P_2, P_4']) 

    # write to file 
    np.savetxt(fplk, np.array([plk['k'], plk['p0k'], plk['p2k'], plk['p4k']]).T, header=hdr) 
    return None 


if __name__=='__main__': 
    run = sys.argv[1]
    if run == 'halo': 
        # python plk.py halo mneut nreal nzbin zstr
        mneut = float(sys.argv[2])
        nreal = int(sys.argv[3]) 
        nzbin = int(sys.argv[4])
        zstr = sys.argv[5]
        nmesh = int(sys.argv[6])
        if zstr == 'z': zbool = True
        elif zstr == 'real': zbool = False
        hadesHalo_Plk(mneut, nreal, nzbin, zspace=zbool, Lbox=1000., Nmesh=nmesh)
    elif run == 'halo_sig8': 
        # python plk.py halo mneut nreal nzbin zstr
        sig8 = float(sys.argv[2])
        nreal = int(sys.argv[3]) 
        nzbin = int(sys.argv[4])
        zstr = sys.argv[5]
        nmesh = int(sys.argv[6])
        if zstr == 'z': zbool = True
        elif zstr == 'real': zbool = False
        hadesHalo_Plk_sigma8(sig8, nreal, nzbin, zspace=zbool, Lbox=1000., Nmesh=nmesh)
    else: 
        raise ValueError
