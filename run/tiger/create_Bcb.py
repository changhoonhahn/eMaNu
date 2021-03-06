'''

This script computes the real-space bispectrum for CDM + baryons. It takes as
input the first and last number of the wanted realizations, the cosmology and the 
snapnum

'''
import argparse
from mpi4py import MPI
import numpy as np
import sys,os,h5py
# -- emanu -- 
from emanu import util as UT 
from emanu.sims import readgadget as RG 
from pyspectrum import pyspectrum as pySpec
# -- pylians3 -- 
import MAS_library as MASL
import Pk_library as PKL

###### MPI DEFINITIONS ######                                    
comm   = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()

# read the first and last realization to identify voids
parser = argparse.ArgumentParser(description="This script constructs bispectrum for CDM+B snapshot")
parser.add_argument("first",      help="first realization number", type=int)
parser.add_argument("last",       help="last  realization number", type=int)
parser.add_argument("cosmo",      help="folder with the realizations")
parser.add_argument("snapnum",    help="snapshot number",          type=int)
args = parser.parse_args()
first, last, cosmo, snapnum = args.first, args.last, args.cosmo, args.snapnum

def create_B(ireal):
    ''' read in snapshots and compute B
    '''
    # read gadget snapshot 
    ptype    = [1] #[1](CDM), [2](neutrinos) or [1,2](CDM+neutrinos)

    # read positions, velocities and IDs of the particles
    fsnap = os.path.join('/projects/QUIJOTE/Snapshots', 
            cosmo, str(ireal), 'snapdir_%s' % str(snapnum).zfill(3), 
            'snap_%s' % str(snapnum).zfill(3))

    fbk = os.path.join('/projects/QUIJOTE/Bk_cdmb', cosmo, 'Bk_%i_snap%i.txt' % (ireal, snapnum))
    if os.path.exists(fbk): 
        print('    %s ... already exists' % fbk) 
        return None  
    else: 
        print('    %s ... calculating' % fbk) 
        pass 

    xyz = RG.read_block(fsnap, "POS ", ptype)/1e3 #positions in Mpc/h
    xyz = xyz.astype(np.float32).T
    
    # calculate bispectrum 
    b123out = pySpec.Bk_periodic(xyz, 
            Lbox=1000.0, #Mpc/h
            Ngrid=360, 
            step=3, 
            Ncut=3, 
            Nmax=40, 
            fft='pyfftw', 
            nthreads=2, 
            silent=False)

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
    hdr = ('CDM+B Bk for cosmology=%s, snapshot %i; k_f = 2pi/%.1f, Nparticles=%i'%\
           (cosmo, snapnum, 1000., xyz.shape[0]))
    print('--- creating %s ---' % os.path.basename(fbk)) 
    np.savetxt(fbk, np.array([i_k,j_k,l_k,p0k1,p0k2, p0k3, b123, q123, b_sn, cnts]).T, 
               fmt='%i %i %i %.5e %.5e %.5e %.5e %.5e %.5e %.5e', 
               delimiter='\t', header=hdr)
    return None 

#####################################################################################
# create output folder if it does not exist
if myrank==0:
    if not os.path.exists(os.path.join('/projects/QUIJOTE/Bk_cdmb', cosmo)):  
        os.system('mkdir %s' % os.path.join('/projects/QUIJOTE/Bk_cdmb',cosmo))

# get the realizations each cpu works on
numbers = np.where(np.arange(args.first, args.last)%nprocs==myrank)[0]
numbers = np.arange(args.first, args.last)[numbers]

######## standard simulations #########
for i in numbers:
    create_B(i)
