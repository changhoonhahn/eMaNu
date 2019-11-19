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
import h5py 
import numpy as np 
# --- eMaNu --- 
from emanu import forwardmodel as FM
from emanu.sims import data as simData
# -- pySpectrum -- 
from pyspectrum import pyspectrum as pySpec


def HOD(tt_hod, halo_folder, snap_folder, snapnum, do_RSD, seed): #, Ob, ns, s8):
    # halo catalogs 
    halos = simData.hqHalos(halo_folder, snap_folder, snapnum)#, Ob=Ob, ns=ns, s8=s8)
    # populate halos 
    hod = FM.hodGalaxies(halos, {'logMmin': tt_hod[0], 'sigma_logM': tt_hod[1], 'logM0': tt_hod[2], 'alpha': tt_hod[3], 'logM1': tt_hod[4]}, seed=seed) 
    # apply RSD 
    if do_RSD: xyz = FM.RSD(hod) 
    else: xyz = np.array(hod['Position']) 
    return xyz 


def create_HOD(tt_hod, halo_folder, snap_folder, snapnum, do_RSD, seed, fhod): #, Ob, ns, s8):
    # halo catalogs 
    halos = simData.hqHalos(halo_folder, snap_folder, snapnum)#, Ob=Ob, ns=ns, s8=s8)
    # populate halos 
    hod = FM.hodGalaxies(halos, {'logMmin': tt_hod[0], 'sigma_logM': tt_hod[1], 'logM0': tt_hod[2], 'alpha': tt_hod[3], 'logM1': tt_hod[4]}, seed=seed) 

    # get positions and velocities of the galaxies
    pos = np.array(hod['Position'])
    vel = np.array(hod['Velocity'])

    # extra columns (for Christina and Alice) 
    # halo position 
    x_h = np.array(hod['halo_x']) 
    y_h = np.array(hod['halo_y']) 
    z_h = np.array(hod['halo_z']) 
    pos_halo = np.array([x_h, y_h, z_h]).T

    # halo velocity 
    vx_h = np.array(hod['halo_vx']) 
    vy_h = np.array(hod['halo_vy']) 
    vz_h = np.array(hod['halo_vz']) 
    vel_halo = np.array([vx_h, vy_h, vz_h]).T
    
    # halo virial mass 
    mvir_halo = np.array(hod['halo_mvir']) 
    # halo virial radius
    rvir_halo = np.array(hod['halo_rvir']) 
    # halo id 
    id_halo = np.array(hod['halo_id']) 
    # gal_type (0:central, 1:satellite) 
    gal_type = np.array(hod['gal_type'])

    # save catalogue to file
    f = h5py.File(fhod, 'w')
    f.create_dataset('pos', data=pos)
    f.create_dataset('vel', data=vel)
    f.create_dataset('halo_pos', data=pos_halo) 
    f.create_dataset('halo_vel', data=vel_halo) 
    f.create_dataset('halo_mvir', data=mvir_halo) 
    f.create_dataset('halo_rvir', data=rvir_halo) 
    f.create_dataset('halo_id', data=id_halo) 
    f.create_dataset('gal_type', data=gal_type) 
    f.close()
    return None 


if __name__=="__main__": 
    tt_hod = np.array([13.65, 0.2, 14., 1.1, 14.]) # fiducial HOD 
    #folder = '/Users/ChangHoon/data/emanu/halos/hades/0.0eV/1'
    halo_folder  = '/global/cscratch1/sd/chahah/emanu/0.0eV/1' # on cori 
    snap_folder  = '/global/cscratch1/sd/chahah/emanu/snapshot' # on cori 
    snapnum = 4
    do_RSD  = True
    seed    = 0 
    #xyz = HOD(tt_hod, halo_folder, snap_folder, snapnum, do_RSD, seed)
    fhod = '/global/cscratch1/sd/chahah/emanu/test_hod_catalog.hdf5' 
    create_HOD(tt_hod, halo_folder, snap_folder, snapnum, do_RSD, seed, fhod)
    raise ValueError
    # calculate bispectrum
    BoxSize = 1000. 
    Ngrid   = 360
    Nmax = 40
    Ncut = 3
    step = 3
    b123out = pySpec.Bk_periodic(xyz.T, Lbox=BoxSize, Ngrid=Ngrid, step=step, 
            Ncut=Ncut, Nmax=Nmax, fft='pyfftw', nthreads=8, silent=False) 
