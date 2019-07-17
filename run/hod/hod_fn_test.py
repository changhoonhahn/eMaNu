'''
script for testing HOD pipeline in new env

conda activate hodtest
# required packages
- eMaNu
- pySpectrum
- nbodykit

'''
import os 
import numpy as np 
# --- eMaNu --- 
from emanu import forwardmodel as FM
from emanu.sims import data as simData


def HOD(tt_hod, folder, snapnum, do_RSD, seed, Ob, ns, s8):
    # halo catalogs 
    halos = simData.hqHalos(folder, snapnum, Ob=Ob, ns=ns, s8=s8)
    # populate halos 
    hod = FM.hodGalaxies(halos, {'logMmin': tt_hod[0], 'sigma_logM': tt_hod[1], 'logM0': tt_hod[2], 'alpha': tt_hod[3], 'logM1': tt_hod[4]}, seed=seed) 
    # apply RSD 
    if do_RSD: xyz = FM.RSD(hod) 
    else: xyz = np.array(hod['Position']) 
    return xyz 


if __name__=="__main__": 
    tt_hod = np.array([14.30, 0.45, 14.10, 0.87, 14.80])
    folder = '/Users/ChangHoon/data/emanu/halos/hades/0.0eV/1'
    snapnum = 4
    do_RSD = True
    seed = 0 
    xyz = HOD(tt_hod, folder, snapnum, do_RSD, seed, 0.049, 0.9624, 0.834)
