'''
evaluate w_p for a grid of hod parameters
'''
import os 
import scipy as sp 
import numpy as np 
# --- eMaNu --- 
from emanu import util as UT
from emanu import forwardmodel as FM
from emanu.hades import data as hadesData
# -- pyspectrum -- 
from pyspectrum import pyspectrum as pySpec

# halo catalogs 
halos = hadesData.hadesMnuHalos(0., 1, 4, mh_min=None, dir='/Users/ChangHoon/data/emanu/halos/hades/0.0eV/1')
# hi res halo catalog
halos_hires = hadesData.hadesMnuHalos(0., '1_hires', 4, mh_min=None, dir='/Users/ChangHoon/data/emanu/halos/hades/0.0eV/1_hires')

thetas = ['logMmin', 'siglogM', 'logM0', 'alpha', 'logM1']
theta_lbls = [r'$\log M_{\rm min}$', r'$\sigma_{\log M}$',r'$\log M_0$',r'$\alpha$',r'$\log M_1$']
hod_fid = np.array([14.22, 0.55, 14.00, 0.87, 14.69]) # fiducial HOD parameters (selected from hod_fiducial.ipynb)
hod_Mr22 = np.array([14.22, 0.77, 14.00, 0.87, 14.69]) # Zheng+(2007) Mr<-22

def pk_model(halos, tt, rsd=True, seed=None): 
    ''' wrapper for populating halos and calculating wp 
    '''
    # population halos 
    hod = FM.hodGalaxies(halos, {'logMmin': tt[0], 'sigma_logM': tt[1], 'logM0': tt[2], 'alpha': tt[3], 'logM1': tt[4]}, seed=seed)
    # apply RSD 
    if rsd: xyz = FM.RSD(hod) 
    else: xyz = np.array(hod['Position']) 
    # calculate wp 
    _pk = pySpec.Pk_periodic(xyz.T, Lbox=1000, Ngrid=360, fft='pyfftw', silent=True)
    return _pk['k'], _pk['p0k']


def fpk(tt, res='HR', rsd=True, seed=None): 
    ''' return file name for wp given HOD parameters
    '''
    pk_dir = os.path.join('/Users/ChangHoon/data/emanu/hod', 'pk') 
    return os.path.join(pk_dir, 
            'pk.z07hod%.2f_%.2f_%.2f_%.2f_%.2f.%s.%i.%s.dat' % 
            (tt[0], tt[1], tt[2], tt[3], tt[4], ['rspace', 'zspace'][rsd], seed, res))


def write_pk(tt, res='HR', rsd=True, seed=None): 
    ''' wrapper for the above two functions 
    '''
    if res == 'HR': # high resolution 
        _halos = halos_hires
    elif res == 'LR': # low res
        _halos = halos 
    _k, _pk = pk_model(_halos, tt, rsd=rsd, seed=seed) 

    _fpk = fpk(tt, res=res, rsd=rsd, seed=seed) 
    np.savetxt(_fpk, np.array([_k, _pk]).T) 
    return None 

'''
# hods from the literature (single realization) 
z07_21_5 = np.array([13.38, 0.51, 13.94, 1.04, 13.91]) # Zheng+(2007) Mr<-21.5
g15_21_5 = np.array([13.53, 0.72, 13.13, 1.14, 14.52]) # Guo+(2015) Mr<-21.5
v19_21_5 = np.array([13.39, 0.56, 12.87, 1.26, 14.51]) # Vakili+(2019) Mr<-21.5
z07_22_0 = np.array([14.22, 0.77, 14.00, 0.87, 14.69]) # Zheng+(2007) Mr<-22
for hod in [z07_21_5, g15_21_5, v19_21_5, z07_22_0]: 
    write_pk(hod, res='LR', rsd=True, seed=0) 
    write_pk(hod, res='HR', rsd=True, seed=0) 
'''
# fiducial HOD
for i in range(20): 
    write_pk(hod_fid, res='LR', rsd=True, seed=i) 
    write_pk(hod_fid, res='HR', rsd=True, seed=i) 

# fiducial HOD with different sigma_logM
for sig in [0.7, 0.6, 0.5, 0.4]:
    for i in range(20): 
        _hod = hod_fid.copy() 
        _hod[1] = sig 
        write_pk(_hod, res='LR', rsd=True, seed=i) 
        write_pk(_hod, res='HR', rsd=True, seed=i) 
