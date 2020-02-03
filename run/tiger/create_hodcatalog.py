# This script computes the galaxy bispectrum in real- and redshift-space. It takes as
# input the first and last number of the wanted realizations, the cosmology and the 
# snapnum. In redshift-space it computes the bispectrum along the 3 different axes. 
import argparse
from mpi4py import MPI
import numpy as np
import sys,os,h5py
from emanu import forwardmodel as FM
from emanu.sims import data as simData
from pyspectrum import pyspectrum as pySpec

###### MPI DEFINITIONS ######                                    
comm   = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()

# read the first and last realization to identify voids
parser = argparse.ArgumentParser(description="This script constructs the HOD galaxy catalog")
parser.add_argument("first",      help="first realization number", type=int)
parser.add_argument("last",       help="last  realization number", type=int)
parser.add_argument("cosmo",      help="folder with the realizations")
parser.add_argument("snapnum",    help="snapshot number",          type=int)
parser.add_argument("logMmin",    help="logMmin",                  type=float)
parser.add_argument("sigma_logM", help="sigma_logM",               type=float)
parser.add_argument("logM0",      help="logM0",                    type=float)
parser.add_argument("alpha",      help="alpha",                    type=float)
parser.add_argument("logM1",      help="logM1",                    type=float)
args = parser.parse_args()
first, last, cosmo, snapnum = args.first, args.last, args.cosmo, args.snapnum
logMmin, sigma_logM = args.logMmin, args.sigma_logM
logM0, alpha, logM1 = args.logM0, args.alpha, args.logM1

def create_HOD(halo_folder, snap_folder, snapnum, hod_dict, seed, fGC):
    ''' Compute and saves the galaxy catalog to hdf5 
    '''
    # read in the halo catalog
    halos = simData.hqHalos(halo_folder, snap_folder, snapnum) 

    # populate halos with galaxies
    hod = FM.hodGalaxies(halos, hod_dict, seed=seed)
    
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
    f = h5py.File(fGC, 'w')
    f.create_dataset('pos', data=pos)
    f.create_dataset('vel', data=vel)
    f.create_dataset('halo_pos', data=pos_halo) 
    f.create_dataset('halo_vel', data=vel_halo) 
    f.create_dataset('halo_mvir', data=mvir_halo) 
    f.create_dataset('halo_rvir', data=rvir_halo) 
    f.create_dataset('halo_id', data=id_halo) 
    f.create_dataset('gal_type', data=gal_type) 
    f.close() 
    # real, RSD x, y, z 
	for 
    if do_RSD: 
        if   axis==0:  xyz = FM.RSD(hod, LOS=[1,0,0]) 
        elif axis==1:  xyz = FM.RSD(hod, LOS=[0,1,0]) 
        elif axis==2:  xyz = FM.RSD(hod, LOS=[0,0,1])
        else:  raise Exception('wrong axis')
    else:      xyz = np.array(hod['Position']) 

    # calculate bispectrum 
    b123out = pySpec.Bk_periodic(xyz.T, Lbox=BoxSize, Ngrid=Ngrid, step=step,
                Ncut=Ncut, Nmax=Nmax, fft='pyfftw', nthreads=2, silent=False)

    i_k  = b123out['i_k1']
    j_k  = b123out['i_k2']
    l_k  = b123out['i_k3']
    p0k1 = b123out['p0k1']
    p0k2 = b123out['p0k2']
    p0k3 = b123out['p0k3']
    b123 = b123out['b123']
    b_sn = b123out['b123_sn']
    q123 = b123out['q123']
    cnts = b123out['counts']

    # save results to file
    hdr = ('galaxy Bk for cosmology=%s, redshift bin %i; k_f = 2pi/%.1f, Ngal=%i'%\
           (cosmo, snapnum, BoxSize, xyz.shape[0]))
    np.savetxt(fbk, np.array([i_k,j_k,l_k,p0k1,p0k2, p0k3, b123, q123, b_sn, cnts]).T, 
               fmt='%i %i %i %.5e %.5e %.5e %.5e %.5e %.5e %.5e', 
               delimiter='\t', header=hdr)





    return None 


def create_HOD_all(halo_folder, snap_folder, snapnum, hod_dict, seed, fGC):
    ''' Compute and saves the galaxy catalog to hdf5 
    '''
    # read in the halo catalog
    halos = simData.hqHalos(halo_folder, snap_folder, snapnum) 

    # populate halos with galaxies
    hod = FM.hodGalaxies(halos, hod_dict, seed=seed)
    
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
    f = h5py.File(fGC, 'w')
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

##################################### INPUT #########################################
# general parameters
root    = '/projects/QUIJOTE'
snapnum = 4 #4(z=0), 3(z=0.5), 2(z=1), 1(z=2), 0(z=3)

# HOD parameter values
seed     = 0    #random seed
hod_dict = {
    'logMmin': logMmin, 
    'sigma_logM': sigma_logM, 
    'logM0': logM0, 
    'alpha': alpha, 
    'logM1': logM1}

# output folder name
root_out = '/projects/QUIJOTE/Galaxies/' 
#####################################################################################

# find the redshift
z = {4:0, 3:0.5, 2:1, 1:2, 0:3}[snapnum]

suffix = None
if logMmin!=13.65:   suffix='logMmin=%.2f' % logMmin
if sigma_logM!=0.2:  suffix='sigma_logM=%.2f' % sigma_logM
if logM0!=14.0:      suffix='logM0=%.1f' % logM0
if alpha!=1.1:       suffix='alpha=%.1f' % alpha
if logM1!=14.0:      suffix='logM1=%.1f' % logM1

if suffix!=None:  cosmo2 = '%s_%s'%(cosmo, suffix)
else:             cosmo2 = cosmo

# create output folder if it does not exist
if myrank==0:
    if not os.path.exists(root_out+cosmo2):  os.system('mkdir %s/%s/'%(root_out,cosmo2))

# get the realizations each cpu works on
numbers = np.where(np.arange(args.first, args.last)%nprocs==myrank)[0]
numbers = np.arange(args.first, args.last)[numbers]

######## standard simulations #########
for i in numbers:
    print(i)

    # find halo and snap folders
    halo_folder = '%s/Halos/%s/%d' % (root,cosmo,i)
    snap_folder = '%s/Snapshots/%s/%d' % (root,cosmo,i)
    if not os.path.exists(snap_folder):
        continue

    # find output folder and create it if it doesnt exist
    folder_out = '%s/%s/%d/' % (root_out, cosmo2, i)
    if not os.path.exists(folder_out):  
        os.system('mkdir %s'%folder_out)

    # create galaxy catalogue
    fGC = '%s/GC_%d_z=%s.hdf5' % (folder_out, seed, z)
    if not os.path.exists(fGC):  
        create_HOD(halo_folder, snap_folder, snapnum, hod_dict, seed, fGC)

###### paired fixed realizations ######
for i in numbers:

    for pair in [0,1]:

        # find halo and snap folders
        halo_folder = '%s/Halos/%s/NCV_%d_%d'%(root,cosmo,pair,i)
        snap_folder = '%s/Snapshots/%s/NCV_%d_%d'%(root,cosmo,pair,i)
        if not(os.path.exists(snap_folder)):  continue

        # find output folder and create it if it doesnt exist
        folder_out = '%s/%s/NCV_%d_%d/'%(root_out, cosmo2, pair, i)
        if not(os.path.exists(folder_out)):  os.system('mkdir %s'%folder_out)

        # create galaxy catalogue
        fGC = '%s/GC_%d_z=%s.hdf5'%(folder_out,seed,z)
        if not(os.path.exists(fGC)):  
            create_HOD(halo_folder, snap_folder, snapnum, hod_dict, seed, fGC)
