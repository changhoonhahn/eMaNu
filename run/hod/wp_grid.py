'''
evaluate w_p for a grid around the fiducial 
'''
import os 
import scipy as sp 
import numpy as np 
# --- eMaNu --- 
from emanu import util as UT
from emanu import forwardmodel as FM
from emanu.hades import data as hadesData
# --- corrfunc -- 
from Corrfunc.theory import wp as wpCF

# halo catalogs 
halos = hadesData.hadesMnuHalos(0., 1, 4, mh_min=None, dir='/Users/ChangHoon/data/emanu/halos/hades/0.0eV/1')
# hi res halo catalog
halos_hires = hadesData.hadesMnuHalos(0., '1_hires', 4, mh_min=None, dir='/Users/ChangHoon/data/emanu/halos/hades/0.0eV/1_hires')

thetas = ['logMmin', 'siglogM', 'logM0', 'alpha', 'logM1']
theta_lbls = [r'$\log M_{\rm min}$', r'$\sigma_{\log M}$',r'$\log M_0$',r'$\alpha$',r'$\log M_1$']
hod_fid = np.array([14.22, 0.55, 14.00, 0.87, 14.69]) # fiducial HOD parameters (selected from hod_fiducial.ipynb)
hod_Mr22 = np.array([14.22, 0.77, 14.00, 0.87, 14.69]) # Zheng+(2007) Mr<-22

rbins = np.array([0.1, 0.15848932, 0.25118864, 0.39810717, 0.63095734, 1., 1.58489319, 2.51188643, 3.98107171, 6.30957344, 10., 15.84893192, 25.11886432]) 
def wp_model(halos, tt, rsd=True, seed=None): 
    ''' wrapper for populating halos and calculating wp 
    '''
    # population halos 
    hod = FM.hodGalaxies(halos, {'logMmin': tt[0], 'sigma_logM': tt[1], 'logM0': tt[2], 'alpha': tt[3], 'logM1': tt[4]}, seed=seed) 
    # apply RSD 
    if rsd: xyz = FM.RSD(hod) 
    else: xyz = np.array(hod['Position']) 
    # calculate wp 
    _wp = wpCF(1000., 40., 1, rbins, xyz[:,0], xyz[:,1], xyz[:,2], verbose=False, output_rpavg=False) 
    return _wp['wp']



steps = [[-0.15, -0.125,  -0.1, -0.075, -0.05, 0.05, 0.075, 0.1, 0.125, 0.15], 


for i in range(hod_fid): 
