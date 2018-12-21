#!/bin/python
import os 
import sys 
import time 
import numpy as np 
import multiprocessing as MP 
# -- eMaNu -- 
from emanu import util as UT
from emanu.hades import data as Dat
from emanu import forwardmodel as FM


def hadesHalo_xyz(mneut, nreal, nzbin, zspace=False, Lbox=1000., mh_min=3200., overwrite=False): 
    ''' output x,y,z positions of the HADES halo catalog given
    mneut, realization #, and redshift bin #
    '''
    # output file 
    if zspace: str_space = 'z'
    else: str_space = 'r'
    fout = ''.join([UT.dat_dir(), 'halos/',
        'groups.', str(mneut), 'eV.', str(nreal), '.nzbin', str(nzbin), 
        '.', str_space, 'space.mhmin', str(mh_min), '.dat']) 

    if os.path.isfile(fout) and not overwrite: 
        print('--- already written to ---\n %s' % (fout))
        return None 
    
    # import Neutrino halo with mneut eV, realization # nreal, at z specified by nzbin 
    halos = Dat.NeutHalos(mneut, nreal, nzbin, mh_min=mh_min, silent=True) 
    # halo positions
    if zspace: 
        xyz = np.array(FM.RSD(halos, LOS=[0,0,1], Lbox=Lbox))
    else: 
        xyz = np.array(halos['Position'])
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]
    assert np.max(x) <= Lbox
    assert np.max(y) <= Lbox
    assert np.max(z) <= Lbox
    # weights (all ones!) 
    w = np.ones(len(x)) 
    
    # header 
    hdr = ''.join([
        'm neutrino = ', str(mneut), ' eV, ', 
        'realization ', str(nreal), ', ', 
        'zbin ', str(nzbin), ', ',
        str_space, '-space']) 

    outarr = np.array([x, y, z, w]).T
    # write to file 
    np.savetxt(fout, outarr, header=hdr) 
    print('--- halo written to ---\n %s' % (fout))
    return None 


def _hadesHalo_xyz(args):
    # wrapper for hadesHalo_xyz that works with multiprocessing 
    mneut = args[0]
    nreal = args[1] 
    nzbin = args[2] 
    zbool = args[3]
    hadesHalo_xyz(mneut, nreal, nzbin, zspace=zbool, overwrite=True)
    return None


def hadesHalo_sigma8_xyz(sig8, nreal, nzbin, zspace=False, Lbox=1000., mh_min=3200., overwrite=False): 
    ''' output x,y,z positions of the 0eV HADES halo catalog that has
    the same sigma8 as the HADES halo catalog with m_nu = mneut.
    '''
    if sig8 not in [0.822, 0.818, 0.807, 0.798]: 
        raise ValueError("sigma8=%f not available" % sig8) 
    # output file 
    if zspace: str_space = 'z'
    else: str_space = 'r'

    fout = ''.join([UT.dat_dir(), 'halos/',
        'groups.0.0eV.sig8', str(sig8), 
        '.', str(nreal), 
        '.nzbin', str(nzbin), 
        '.', str_space, 'space',
        '.mhmin', str(mh_min), 
        '.dat']) 

    if os.path.isfile(fout) and not overwrite: 
        print('--- already written to ---\n %s' % (fout))
        return None 
    
    # import Neutrino halo with 0.0 eV, sig8 = tbl_sig[mneut], realization # nreal, at z specified by nzbin 
    halos = Dat.Sig8Halos(sig8, nreal, nzbin, mh_min=mh_min, silent=True) 
    # halo positions
    if zspace: 
        xyz = np.array(FM.RSD(halos, LOS=[0,0,1], Lbox=Lbox))
    else: 
        xyz = np.array(halos['Position'])
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]
    assert np.max(x) <= Lbox
    assert np.max(y) <= Lbox
    assert np.max(z) <= Lbox
    # weights (all ones!) 
    w = np.ones(len(x)) 
    
    # header 
    hdr = ''.join([
        'm neutrino = 0.0 eV, ', 
        'sigma_8 = ', str(sig8), 
        'realization ', str(nreal), ', ', 
        'zbin ', str(nzbin), ', ',
        str_space, '-space']) 

    outarr = np.array([x, y, z, w]).T
    # write to file 
    np.savetxt(fout, outarr, header=hdr) 
    print('--- halo written to ---\n %s' % (fout))
    return None 


def _hadesHalo_sigma8_xyz(args):
    # wrapper for hadesHalo_sigma8_xyz that works with multiprocessing 
    mneut = args[0]
    nreal = args[1] 
    nzbin = args[2] 
    zbool = args[3]
    hadesHalo_sigma8_xyz(mneut, nreal, nzbin, zspace=zbool, overwrite=True)
    return None


if __name__=='__main__': 
    run = sys.argv[1]
    nthread = int(sys.argv[2])
    if nthread == 1: 
        # python threepcf.py data 1 mneut nreal nzbin zstr
        mneut_or_sig8 = float(sys.argv[3])
        nreal = int(sys.argv[4]) 
        nzbin = int(sys.argv[5])
        zstr = sys.argv[6]
        if zstr == 'z': zbool = True
        elif zstr == 'real': zbool = False
        if run == 'mneut': 
            hadesHalo_xyz(mneut_or_sig8, nreal, nzbin, zspace=zbool, Lbox=1000., overwrite=False)
        elif run == 'sig8': 
            hadesHalo_sigma8_xyz(mneut_or_sig8, nreal, nzbin, zspace=zbool, Lbox=1000., overwrite=False)
    elif nthread > 1: 
        # python threepcf.py data 1 mneut nreal_i nreal_f nzbin zstr
        mneut_or_sig8 = float(sys.argv[3])
        nreal_i = int(sys.argv[4]) 
        nreal_f = int(sys.argv[5]) 
        nzbin = int(sys.argv[6])
        zstr = sys.argv[7]
        if zstr == 'z': zbool = True
        elif zstr == 'real': zbool = False
        # multiprocessing 
        args_list = [(mneut_or_sig8, ireal, nzbin, zbool) for ireal in np.arange(nreal_i, nreal_f+1)]
        pewl = MP.Pool(processes=nthread)
        if run == 'mneut': 
            pewl.map(_hadesHalo_xyz, args_list)
        elif run == 'sig8': 
            pewl.map(_hadesHalo_sigma8_xyz, args_list)
        pewl.close()
        pewl.terminate()
        pewl.join() 
