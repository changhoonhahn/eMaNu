'''

calculate bispectrum 


'''
import sys 
import numpy as np 
# -- pyspectrum -- 
from pyspectrum import pyspectrum as pySpec
# -- emanu -- 
from emanu import util as UT 


def haloBispectrum(mneut, nreal, nzbin, Lbox=1000., zspace=False, 
        Ngrid=360, Nmax=40, Ncut=3, step=3, silent=True): 
    if zspace: raise NotImplementedError
    else: str_space = 'r' 
    fhalo = ''.join([UT.dat_dir(), 'halos/', 
        'groups.', str(mneut), 'eV.', str(nreal), '.nzbin', str(nzbin), '.', str_space, 'space.dat']) 

    # read in data file  
    if not silent: print('--- reading %s ---' % fhalo) 
    x, y, z = np.loadtxt(fhalo, skiprows=1, unpack=True, usecols=[0,1,2]) 
    xyz = np.zeros((3, len(x)))
    
    xyz[0,:] = x
    xyz[1,:] = y 
    xyz[2,:] = z
    
    delta = pySpec.FFTperiodic(xyz, fft='fortran', Lbox=Lbox, Ngrid=Ngrid, silent=silent) 
    delta_fft = pySpec.reflect_delta(delta, Ngrid=Ngrid) 
    
    # calculate bispectrum 
    i_k, j_k, l_k, b123, q123 = pySpec.Bk123_periodic(
            delta_fft, Nmax=Nmax, Ncut=Ncut, step=step, fft_method='pyfftw', nthreads=1, silent=silent) 

    hdr = ('halo bispectrum for mneut=%f, realization %i, redshift bin %i; k_f = 2pi/%f' % 
            (mneut, nreal, nzbin, Lbox))
    fbk = ''.join([UT.dat_dir(), 'bispectrum/', 
         fhalo.split('/')[-1].rsplit('.dat', 1)[0],
        '.Ngrid', str(Ngrid), 
        '.Nmax', str(Nmax),
        '.Ncut', str(Ncut),
        '.step', str(step), 
        '.pyfftw', 
        '.dat']) 
    np.savetxt(fbk, np.array([i_k, j_k, l_k, b123, q123]).T, fmt='%i %i %i %.5e %.5e', 
            delimiter='\t', header=hdr)
    return None 


if __name__=="__main__":
    mneut = float(sys.argv[1]) 
    nreal = int(sys.argv[2]) 
    nzbin = int(sys.argv[3]) 
    haloBispectrum(mneut, nreal, nzbin, Lbox=1000., zspace=False, 
            Ngrid=360, Nmax=40, Ncut=3, step=3, silent=True)
