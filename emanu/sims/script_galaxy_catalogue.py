'''

Code for generating galaxy catalogs from fastpm 

fastpm --halofinder--> halo catalog --HOD--> galaxy catalog 

author: Elena Massara 

'''
import fastpm
from fastpm.core import Solver, leapfrog
from fastpm.nbkit import FastPMCatalogSource
from nbodykit.lab import *
from nbodykit import setup_logging

import sys
import dask
from pylab import *

from pmesh.pm import ParticleMesh
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as interpolate

########################################################################
def func_gal_catalogue(bs, nc, seed, nstep, seed_hod, Omega_m, p_alpha, p_logMin, p_logM1, p_logM0, p_sigma_logM):

    folder = "L%04d_N%04d_S%04d_%02dstep"%(bs, nc, seed, nstep)
    
    # setup initial conditions
    Omegacdm = Omega_m-0.049,
    cosmo   = cosmology.Planck15.clone(Omega_cdm=Omegacdm, 
                                       h = 0.6711, Omega_b = 0.049)
    power   = cosmology.LinearPower(cosmo, 0)
    klin    = np.logspace(-4,2,1000) 
    plin    = power(klin)
    pkfunc  = interpolate(klin, plin)
    
    
    # run the simulation
    pm      = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])
    Q       = pm.generate_uniform_particle_grid()
    stages  = numpy.linspace(0.1, 1.0, nstep, endpoint=True)
    solver  = Solver(pm, cosmo, B=2)
    wn      = solver.whitenoise(seed)
    dlin    = solver.linear(wn, pkfunc)
    state   = solver.lpt(dlin, Q, stages[0])
    state   = solver.nbody(state, leapfrog(stages))
    
    # create the catalogue
    cat = ArrayCatalog({'Position' : state.X, 
    	  		'Velocity' : state.V, 
			'Displacement' : state.S,
			'Density' : state.RHO}, 
			BoxSize=pm.BoxSize, 
			Nmesh=pm.Nmesh,
			M0 = Omega_m * 27.75e10 * bs**3 / (nc/2.0)**3)
    cat['KDDensity'] = KDDensity(cat).density
    cat.save('%s/Matter'%(folder), ('Position', 'Velocity', 'Density', 'KDDensity'))

    # run FOF 
    fof = FOF(cat, linking_length=0.2, nmin=12)
    fofcat = fof.to_halos(particle_mass=cat.attrs['M0'],
                          cosmo = cosmo,
                          redshift = 0.0)      
    fofcat.save('%s/FOF'%(folder), ('Position', 'Velocity', 'Mass', 'Radius'))
   
    
    # run HOD
    params = {'alpha':p_alpha, 'logMmin':p_logMin, 'logM1':p_logM1, 
              'logM0':p_logM0, 'sigma_logM':p_sigma_logM}
    halos = HaloCatalog(fofcat, cosmo=cosmo, redshift=0.0, mdef='vir')
    halocat = halos.to_halotools(halos.attrs['BoxSize'])
    hod = HODCatalog(halocat, seed=seed_hod, **params)
    
    hod.save('%s/HOD'%(folder), ('Position', 'Velocity'))
    
    return folder,cat,fofcat,hod


########## INPUT ##########

bs, nc, seed, nstep, seed_hod, Omegam, p_alpha, p_logMin, p_logM1, p_logM0, p_sigma_logM = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8]), float(sys.argv[9]), float(sys.argv[10]), float(sys.argv[11])


###################################################################
pm      = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])

if pm.comm.rank == 0:
   print('starting the simulation')

folder,cat,fofcat,hod = func_gal_catalogue(bs, nc, seed, nstep, seed_hod, Omegam, p_alpha, p_logMin, p_logM1, p_logM0, p_sigma_logM)

if pm.comm.rank == 0:
   print('Done!')

