'''
script for testing HOD pipeline in new env

conda activate hodtest
# required packages
- eMaNu
- nbodykit

# bispectrum requires 
- pySpectrum 
- pyfftw

'''
import os 
import numpy as np 
# --- eMaNu --- 
from emanu import forwardmodel as FM
from emanu.sims import data as simData
# -- pySpectrum -- 
from pyspectrum import pyspectrum as pySpec


def HOD(tt_hod, folder, snapnum, do_RSD, seed): #, Ob, ns, s8):
    # halo catalogs 
    halos = simData.hqHalos(folder, snapnum)#, Ob=Ob, ns=ns, s8=s8)
    # populate halos 
    hod = FM.hodGalaxies(halos, {'logMmin': tt_hod[0], 'sigma_logM': tt_hod[1], 'logM0': tt_hod[2], 'alpha': tt_hod[3], 'logM1': tt_hod[4]}, seed=seed) 
    # apply RSD 
    if do_RSD: xyz = FM.RSD(hod) 
    else: xyz = np.array(hod['Position']) 
    return xyz 


if __name__=="__main__": 
    tt_hod = np.array([13.65, 0.2, 14., 1.1, 14.]) # fiducial HOD 
    #folder = '/Users/ChangHoon/data/emanu/halos/hades/0.0eV/1'
    folder  = '/global/cscratch1/sd/chahah/emanu/0.0eV/1' # on cori 
    snapnum = 4
    do_RSD  = True
    seed    = 0 
    xyz = HOD(tt_hod, folder, snapnum, do_RSD, seed)
    print(xyz[:10]) 
    raise ValueError
    # calculate bispectrum
    BoxSize = 1000. 
    Ngrid   = 360
    Nmax = 40
    Ncut = 3
    step = 3
    b123out = pySpec.Bk_periodic(xyz.T, Lbox=BoxSize, Ngrid=Ngrid, step=step, 
            Ncut=Ncut, Nmax=Nmax, fft='pyfftw', nthreads=8, silent=False) 
