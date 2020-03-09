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


def create_P(fGC):
    ''' Compute and saves the galaxy catalog to hdf5 and quickly computes P
    '''
    # read catalog from file 
    print('--- reading %s ---' % os.path.basename(fGC))
    f = h5py.File(fGC, 'r')
    hod = {} 
    hod['Position'] = f['pos'][...]
    hod['VelocityOffset'] = f['vel_offset'][...] 
    f.close()

    # loop through  real, RSD x, y, z 
    #for rsd in ['real', 0, 1, 2]: 
    for rsd in [0, 1, 2]: 
        if rsd == 'real': 
            xyz = np.array(hod['Position'])
            rsd_str = 'real'
            axis = 0 
        elif rsd == 0:  
            xyz = FM.RSD(hod, LOS=[1,0,0]) 
            xyz = xyz.astype(np.float32)
            rsd_str = 'RS0'
            axis = 0
        elif rsd == 1: 
            xyz = FM.RSD(hod, LOS=[0,1,0]) 
            xyz = xyz.astype(np.float32)
            rsd_str = 'RS1'
            axis = 1
        elif rsd == 2: 
            xyz = FM.RSD(hod, LOS=[0,0,1])
            xyz = xyz.astype(np.float32)
            rsd_str = 'RS2'
            axis = 2
        fpk = os.path.join(os.path.dirname(fGC), 'Pk_%s_%s' % (rsd_str, os.path.basename(fGC).replace('.hdf5', '.txt')))
        if os.path.exists(fpk): 
            print('    %s ... already exists' % fpk) 
            continue 
        else: 
            print('    %s ... calculating' % fpk) 
            pass 

        # calculate powerspectrum 
        delta = np.zeros((1024, 1024, 1024), dtype=np.float32)
        MASL.MA(xyz, delta, 1000., 'CIC')
        delta /= np.mean(delta, dtype=np.float64)
        delta -= 1.0 
        Pk = PKL.Pk(delta, 
                1000., 
                axis, 
                'CIC', 
                threads=1)

        hdr = ('Ngalaxies=%i BoxSize=%.3f' % (xyz.shape[0], 1000.))    
        print('--- creating %s ---' % os.path.basename(fpk)) 
        if rsd == 'real': 
            np.savetxt(fpk, np.transpose([Pk.k3D, Pk.Pk[:,0]]), delimiter='\t', header=hdr)
        else:
            np.savetxt(fpk, np.transpose([Pk.k3D, Pk.Pk[:,0], Pk.Pk[:,1], Pk.Pk[:,2]]), delimiter='\t', header=hdr)
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
root_out = '/projects/QUIJOTE/Galaxies' 
#####################################################################################

# find the redshift
z = {4:0, 3:0.5, 2:1, 1:2, 0:3}[snapnum]

suffix = None
if logMmin!=13.65:   suffix='logMmin=%.2f' % logMmin
if sigma_logM!=0.2:  suffix='sigma_logM=%.2f' % sigma_logM
if logM0!=14.0:      suffix='logM0=%.1f' % logM0
if alpha!=1.1:       suffix='alpha=%.1f' % alpha
if logM1!=14.0:      suffix='logM1=%.1f' % logM1

if suffix is not None:  cosmo2 = '%s_%s'%(cosmo, suffix)
else:             cosmo2 = cosmo

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
        os.system('mkdir %s'%folder_out)

    # create galaxy catalogue
    fGC = '%sGC_%d_z=%s.hdf5' % (folder_out, seed, z)
    print(fGC)
    create_P(fGC)

'''
    ###### paired fixed realizations ######
    for i in numbers:
        for pair in [0,1]:
            # find halo and snap folders
            halo_folder = '%s/Halos/%s/NCV_%d_%d'%(root,cosmo,pair,i)

            # find output folder and create it if it doesnt exist
            folder_out = '%s/%s/NCV_%d_%d/'%(root_out, cosmo2, pair, i)
            if not(os.path.exists(folder_out)):  os.system('mkdir %s'%folder_out)

            # create galaxy catalogue
            fGC = '%s/GC_%d_z=%s.hdf5'%(folder_out,seed,z)
            if not(os.path.exists(fGC)):  
                print('--- creating %s ---' % os.path.basename(fGC)) 
                create_HOD(halo_folder, snapnum, 
                        hdr['Om'][i_hdr], hdr['Ol'][i_hdr], hdr['z'][i_hdr], hdr['h'][i_hdr], hdr['Hz'][i_hdr], 
                        hod_dict, seed, fGC)
            else:
                print('--- already exists %s ---' % os.path.basename(fGC)) 
            #create_ALL(halo_folder, snapnum, hdr['Om'][i_hdr], hdr['Ol'][i_hdr], hdr['z'][i_hdr], hdr['h'][i_hdr], hdr['Hz'][i_hdr], hod_dict, seed, fGC)
'''
