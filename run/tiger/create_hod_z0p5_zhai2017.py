# This script computes the galaxy bispectrum in real- and redshift-space. It takes as
# input the first and last number of the wanted realizations, the cosmology and the 
# snapnum. In redshift-space it computes the bispectrum along the 3 different axes. 
import argparse
from mpi4py import MPI
import numpy as np
import sys,os,h5py
# -- emanu -- 
from emanu import util as UT 
from emanu import forwardmodel as FM
from emanu.sims import data as simData
from pyspectrum import pyspectrum as pySpec
# -- pylians3 -- 
import MAS_library as MASL
import Pk_library as PKL

###### MPI DEFINITIONS ######                                    
comm   = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()

# read the first and last realization to identify voids
parser = argparse.ArgumentParser(description="This script constructs the HOD galaxy catalog with Zhai+(2017) at z=0.5")
parser.add_argument("first",      help="first realization number", type=int)
parser.add_argument("last",       help="last  realization number", type=int)
parser.add_argument("cosmo",      help="folder with the realizations")
parser.add_argument("seed",       help="seed",                     type=int)
args = parser.parse_args()
first, last, cosmo, snapnum = args.first, args.last, args.cosmo
seed = args.seed # HODrandom seed


def create_HOD(halo_folder, snapnum, Om, Ol, z, h, Hz, hod_dict, seed, fGC):
    ''' Compute and saves the galaxy catalog to hdf5 
    '''
    # read in the halo catalog
    halos = simData.hqHalos(halo_folder, None, snapnum, Om=Om, Ol=Ol, z=z, h=h, Hz=Hz) 

    # populate halos with galaxies
    hod = FM.hodGalaxies(halos, hod_dict, seed=seed)
    
    # get positions and velocities of the galaxies
    pos = np.array(hod['Position'])
    vel = np.array(hod['Velocity'])
    vel_offset = np.array(hod['VelocityOffset']) 

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
    #print('--- creating %s ---' % os.path.basename(fGC))
    f = h5py.File(fGC, 'w')
    f.create_dataset('pos', data=pos)
    f.create_dataset('vel', data=vel)
    f.create_dataset('vel_offset', data=vel_offset) 
    f.create_dataset('halo_pos', data=pos_halo) 
    f.create_dataset('halo_vel', data=vel_halo) 
    f.create_dataset('halo_mvir', data=mvir_halo) 
    f.create_dataset('halo_rvir', data=rvir_halo) 
    f.create_dataset('halo_id', data=id_halo) 
    f.create_dataset('gal_type', data=gal_type) 
    f.close()
    return None 


##################################### INPUT #########################################
# general parameters
root    = '/projects/QUIJOTE'

# Zhai+2017 HOD parameter values
# zhai2017 = np.array([13.67, 0.81, 11.62, 0.43, 14.93])
hod_dict = {
    'logMmin': 13.67, 
    'sigma_logM': 0.81, 
    'logM0': 11.62, 
    'alpha': 0.43, 
    'logM1': 14.93}

# output folder name
root_out = '/projects/QUIJOTE/Galaxies' 
#####################################################################################

# find the redshift
snapnum = 3 
z = {4:0, 3:0.5, 2:1, 1:2, 0:3}[snapnum]

cosmo2 = cosmo 

# create output folder if it does not exist
if myrank==0:
    if not os.path.exists(os.path.join(root_out, cosmo2)):  os.system('mkdir %s' % os.path.join(root_out,cosmo2))

# get the realizations each cpu works on
numbers = np.where(np.arange(args.first, args.last)%nprocs==myrank)[0]
numbers = np.arange(args.first, args.last)[numbers]

# look up header info 
hdr = np.genfromtxt(os.path.join(UT.dat_dir(), 'quijote_header_lookup.dat'), 
        delimiter='\t', dtype=None, names=('theta', 'snapnum', 'Om', 'Ol', 'z', 'h', 'Hz'))
_cosmos = list(hdr['theta'].astype(str)) 
assert cosmo in _cosmos
i_hdr = _cosmos.index(cosmo)  

######## standard simulations #########
for i in numbers:
    # find halo and snap folders
    halo_folder = '%s/Halos/%s/%d' % (root,cosmo,i)

    # find output folder and create it if it doesnt exist
    folder_out = '%s/%s/%d/' % (root_out, cosmo2, i)
    if not os.path.exists(folder_out):  
        os.system('mkdir %s' % folder_out)

    fGC = '%s/GC_zhai2017_%d_z=%s.hdf5' % (folder_out, seed, z)
    if not os.path.exists(fGC):  
        print('--- creating %s ---' % fGC) 
        create_HOD(halo_folder, snapnum, 
                hdr['Om'][i_hdr], hdr['Ol'][i_hdr], hdr['z'][i_hdr], hdr['h'][i_hdr], hdr['Hz'][i_hdr], 
                hod_dict, seed, fGC)
    else:
        print('--- already exists %s ---' % fGC) 
