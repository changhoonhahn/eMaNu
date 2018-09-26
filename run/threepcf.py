#!/bin/python
import os 
import sys 
import numpy as np 
import multiprocessing as MP 
# -- eMaNu -- 
from emanu import util as UT
from emanu import forwardmodel as FM
from emanu.hades import data as Dat


def hadesHalo_pre3PCF(mneut, nreal, nzbin, zspace=False, Lbox=1000., overwrite=False): 
    ''' pre-process halo catalogs for 3PCF run. In addition to 
    reading in halo catalog and outputing the input format for Daniel 
    Eisenstein's code, it also appends 1.76 * N_data randoms points 
    for computing NNN (N = D-R). 
    '''
    # output file 
    if zspace: str_space = '.z'
    else: str_space = '.r'
    fout = ''.join([UT.hades_dir(mneut, nreal), 
        'groups.nzbin', str(nzbin), str_space, 'space.dat']) 

    if os.path.isfile(fout) and not overwrite: 
        print('--- already written to ---\n %s' % (fout))
        return None 
    
    # import Neutrino halo with mneut eV, realization # nreal, at z specified by nzbin 
    halos = Dat.NeutHalos(mneut, nreal, nzbin) 
    # halo positions
    if zspace: 
        xyz = FM.RSD(halos, LOS=[0,0,1])
    else: 
        xyz = np.array(halos['Position'])
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]
    assert x.max() <= Lbox
    assert y.max() <= Lbox
    assert z.max() <= Lbox
    # weights (all ones!) 
    w = np.ones(len(x)) 

    ndat = len(x) 
    nran = int(1.76 * float(ndat))
    f_dr = float(ndat)/float(nran) 
    x = np.concatenate([x, Lbox * np.random.uniform(size=nran)]) 
    y = np.concatenate([y, Lbox * np.random.uniform(size=nran)]) 
    z = np.concatenate([z, Lbox * np.random.uniform(size=nran)]) 
    w = np.concatenate([w, -1. * np.repeat(f_dr, nran)]) 
    
    # header 
    hdr = ''.join([
        'm neutrino = ', str(mneut), ' eV, ', 
        'realization ', str(nreal), ', ', 
        'zbin ', str(nzbin)]) 

    outarr = np.array([x, y, z, w]).T
    # write to file 
    np.savetxt(fout, outarr, header=hdr) 
    print('--- halo written to ---\n %s' % (fout))
    return None 


def _hadesHalo_pre3PCF(args):
    # wrapper for NeutHalo that works with multiprocessing 
    mneut = args[0]
    nreal = args[1] 
    nzbin = args[2] 
    zspace = args[3]
    hadesHalo_pre3PCF(mneut, nreal, nzbin, zspace=zbool)
    return None


if __name__=='__main__': 
    nthread = int(sys.argv[1])
    if nthread == 1: 
        # python threepcf.py 1 mneut nreal nzbin zstr
        mneut = float(sys.argv[2])
        nreal = int(sys.argv[3]) 
        nzbin = int(sys.argv[4])
        zstr = sys.argv[5]
        if zstr == 'z': zbool = True
        elif zstr == 'real': zbool = False
        hadesHalo_pre3PCF(mneut, nreal, nzbin, zspace=zbool, Lbox=1000., overwrite=True)
    elif nthread > 1: 
        # python threepcf.py 1 mneut nreal nzbin zstr
        mneut = float(sys.argv[2])
        nreal_i = int(sys.argv[3]) 
        nreal_f = int(sys.argv[4]) 
        nzbin = int(sys.argv[5])
        zstr = sys.argv[6]
        if zstr == 'z': zbool = True
        elif zstr == 'real': zbool = False
        # multiprocessing 
        args_list = [(mneut, ireal, nzbin, zbool) for ireal in np.arange(nreal_i, nreal_f+1)]
        pewl = MP.Pool(processes=nthread)
        pewl.map(_hadesHalo_pre3PCF, args_list)
        pewl.close()
        pewl.terminate()
        pewl.join() 
