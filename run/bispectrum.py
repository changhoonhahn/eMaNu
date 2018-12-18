'''

calculate bispectrum 


'''
import os 
import sys 
import numpy as np 
# -- pyspectrum -- 
from pyspectrum import pyspectrum as pySpec
# -- emanu -- 
from emanu import util as UT 


def haloBispectrum(mneut, nreal, nzbin, Lbox=1000., zspace=False, mh_min=3200.,
        Ngrid=360, Nmax=40, Ncut=3, step=3, silent=True, overwrite=False): 
    if zspace: raise NotImplementedError
    else: str_space = 'r' 
    fhalo = ''.join([UT.dat_dir(), 'halos/', 
        'groups.', str(mneut), 'eV.', str(nreal), '.nzbin', str(nzbin), '.', str_space, 'space',
        '.mhmin', str(mh_min), '.dat']) 

    fbk = ''.join([UT.dat_dir(), 'bispectrum/', 
         fhalo.split('/')[-1].rsplit('.dat', 1)[0],
        '.Ngrid', str(Ngrid), 
        '.Nmax', str(Nmax),
        '.Ncut', str(Ncut),
        '.step', str(step), 
        '.pyfftw', 
        '.dat']) 
    if not os.path.isfile(fbk) or overwrite: 
        # read in data file  
        if not silent: print('--- calculating %s ---' % fbk) 
        if not silent: print('--- reading %s ---' % fhalo) 
        x, y, z = np.loadtxt(fhalo, skiprows=1, unpack=True, usecols=[0,1,2]) 
        xyz = np.zeros((3, len(x)))
        
        xyz[0,:] = x
        xyz[1,:] = y 
        xyz[2,:] = z
        
        delta = pySpec.FFTperiodic(xyz, fft='fortran', Lbox=Lbox, Ngrid=Ngrid, silent=silent) 
        delta_fft = pySpec.reflect_delta(delta, Ngrid=Ngrid) 
        
        # calculate bispectrum 
        i_k, j_k, l_k, b123, q123, cnts = pySpec.Bk123_periodic(
                delta_fft, Nmax=Nmax, Ncut=Ncut, step=step, fft_method='pyfftw', nthreads=1, silent=silent) 

        hdr = ('halo bispectrum for mneut=%f, realization %i, redshift bin %i; k_f = 2pi/%f' % 
                (mneut, nreal, nzbin, Lbox))
        np.savetxt(fbk, np.array([i_k, j_k, l_k, b123, q123, cnts]).T, fmt='%i %i %i %.5e %.5e %.5e', 
                delimiter='\t', header=hdr)
    else: 
        if not silent: print('--- %s already exists ---' % fbk) 
    return None 


def haloBispectrum_sigma8(sig8, nreal, nzbin, Lbox=1000., zspace=False, mh_min=3200.,
        Ngrid=360, Nmax=40, Ncut=3, step=3, silent=True, overwrite=False): 
    if zspace: raise NotImplementedError
    else: str_space = 'r' 
    fhalo = ''.join([UT.dat_dir(), 'halos/', 
        'groups.0.0eV.sig8', str(sig8), '.', str(nreal), '.nzbin', str(nzbin), '.', str_space, 'space',
        '.mhmin', str(mh_min), '.dat']) 

    fbk = ''.join([UT.dat_dir(), 'bispectrum/', 
         fhalo.split('/')[-1].rsplit('.dat', 1)[0],
        '.Ngrid', str(Ngrid), 
        '.Nmax', str(Nmax),
        '.Ncut', str(Ncut),
        '.step', str(step), 
        '.pyfftw', 
        '.dat']) 
    if not os.path.isfile(fbk) or overwrite: 
        # read in data file  
        if not silent: print('--- calculating %s ---' % fbk) 
        if not silent: print('--- reading %s ---' % fhalo) 
        x, y, z = np.loadtxt(fhalo, skiprows=1, unpack=True, usecols=[0,1,2]) 
        xyz = np.zeros((3, len(x)))
        
        xyz[0,:] = x
        xyz[1,:] = y 
        xyz[2,:] = z
        
        delta = pySpec.FFTperiodic(xyz, fft='fortran', Lbox=Lbox, Ngrid=Ngrid, silent=silent) 
        delta_fft = pySpec.reflect_delta(delta, Ngrid=Ngrid) 
        
        # calculate bispectrum 
        i_k, j_k, l_k, b123, q123, cnts = pySpec.Bk123_periodic(
                delta_fft, Nmax=Nmax, Ncut=Ncut, step=step, fft_method='pyfftw', nthreads=1, silent=silent) 

        hdr = ('halo bispectrum for sigma_8=%f, realization %i, redshift bin %i; k_f = 2pi/%f' % 
                (sig8, nreal, nzbin, Lbox))
        np.savetxt(fbk, np.array([i_k, j_k, l_k, b123, q123, cnts]).T, fmt='%i %i %i %.5e %.5e %.5e', 
                delimiter='\t', header=hdr)
    else: 
        if not silent: print('--- %s already exists ---' % fbk) 
    return None 


if __name__=="__main__":
    run = sys.argv[1]
    if run == 'mneut': 
        mneut = float(sys.argv[2]) 
        nreal = int(sys.argv[3]) 
        nzbin = int(sys.argv[4]) 
        haloBispectrum(mneut, nreal, nzbin, Lbox=1000., zspace=False, 
                Ngrid=360, Nmax=40, Ncut=3, step=3, silent=True, overwrite=False)
    elif run == 'sigma8': 
        sig8 = float(sys.argv[2]) 
        nreal = int(sys.argv[3]) 
        nzbin = int(sys.argv[4]) 
        haloBispectrum_sigma8(sig8, nreal, nzbin, Lbox=1000., zspace=False, 
                Ngrid=360, Nmax=40, Ncut=3, step=3, silent=True, overwrite=False)
