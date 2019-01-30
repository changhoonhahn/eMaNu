'''

calculate bispectrum 


'''
import os 
import sys 
import h5py
import numpy as np 
# -- pyspectrum -- 
from pyspectrum import pyspectrum as pySpec
# -- emanu -- 
from emanu import util as UT 

def haloBispectrum(mneut, nreal, nzbin, zspace=False, mh_min=3200.,
        Ngrid=360, Nmax=40, Ncut=3, step=3, silent=True, overwrite=False): 
    if zspace: str_space = 'z' 
    else: str_space = 'r' 
    fhalo = ''.join([UT.dat_dir(), 'halos/', 
        'groups.', 
        str(mneut), 'eV.',      # mnu eV
        str(nreal),             # realization #
        '.nzbin', str(nzbin),   # zbin 
        '.mhmin', str(mh_min), '.hdf5']) 

    fbk = ''.join([UT.dat_dir(), 'bispectrum/', 
        fhalo.split('/')[-1].rsplit('.hdf5', 1)[0],
        '.', str_space, 'space',
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
        f = h5py.File(fhalo, 'r') 
        Lbox = f.attrs['Lbox']
        xyz = f['Position'].value.T
        
        if zspace: # impose redshift space distortions on the z-axis 
            v_offset = f['VelocityOffset'].value.T
            xyz += (f['VelocityOffset'].value * [0, 0, 1]).T + Lbox
            xyz = (xyz % Lbox) 

        if not silent: print('FFTing the halo field') 
        delta = pySpec.FFTperiodic(xyz, fft='fortran', Lbox=Lbox, Ngrid=Ngrid, silent=silent) 
        delta_fft = pySpec.reflect_delta(delta, Ngrid=Ngrid) 
        
        # calculate bispectrum 
        if not silent: print('calculating the halo bispectrum') 
        b123out = pySpec.Bk123_periodic(
                delta_fft, Nmax=Nmax, Ncut=Ncut, step=step, fft_method='pyfftw', nthreads=1, silent=silent) 
        i_k = b123out['i_k1']
        j_k = b123out['i_k2']
        l_k = b123out['i_k3']
        pki = b123out['p0k1']
        pkj = b123out['p0k2']
        pkl = b123out['p0k3']
        b123 = b123out['b123'] 
        q123 = b123out['q123'] 
        cnts = b123out['counts'] 

        hdr = ('halo bispectrum for mneut=%f, realization %i, redshift bin %i; k_f = 2pi/%f, Nhalo=%i' % 
                (mneut, nreal, nzbin, Lbox, xyz.shape[1]))
        np.savetxt(fbk, np.array([i_k, j_k, l_k, b123, q123, cnts, pki, pkj, pkl]).T, 
                fmt='%i %i %i %.5e %.5e %.5e %.5e %.5e %.5e', delimiter='\t', header=hdr)
    else: 
        if not silent: print('--- %s already exists ---' % fbk) 
    return None 


def haloBispectrum_sigma8(sig8, nreal, nzbin, zspace=False, mh_min=3200.,
        Ngrid=360, Nmax=40, Ncut=3, step=3, silent=True, overwrite=False): 
    if zspace: str_space = 'z'  
    else: str_space = 'r' 
    fhalo = ''.join([UT.dat_dir(), 'halos/', 
        'groups.', 
        '0.0eV.sig8_', str(sig8),   # 0.0eV, sigma8
        '.', str(nreal),                 # realization #
        '.nzbin', str(nzbin),       # zbin 
        '.mhmin', str(mh_min), '.hdf5']) 

    fbk = ''.join([UT.dat_dir(), 'bispectrum/', 
         fhalo.split('/')[-1].rsplit('.hdf5', 1)[0],
        '.', str_space, 'space',
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
        f = h5py.File(fhalo, 'r') 
        Lbox = f.attrs['Lbox']
        xyz = f['Position'].value.T
        
        if zspace: # impose redshift space distortions on the z-axis 
            v_offset = f['VelocityOffset'].value.T
            xyz += (f['VelocityOffset'].value * [0, 0, 1]).T + Lbox
            xyz = (xyz % Lbox) 
        
        delta = pySpec.FFTperiodic(xyz, fft='fortran', Lbox=Lbox, Ngrid=Ngrid, silent=silent) 
        delta_fft = pySpec.reflect_delta(delta, Ngrid=Ngrid) 
        
        # calculate bispectrum 
        b123out = pySpec.Bk123_periodic(
                delta_fft, Nmax=Nmax, Ncut=Ncut, step=step, fft_method='pyfftw', nthreads=1, silent=silent) 
        i_k = b123out['i_k1']
        j_k = b123out['i_k2']
        l_k = b123out['i_k3']
        pki = b123out['p0k1']
        pkj = b123out['p0k2']
        pkl = b123out['p0k3']
        b123 = b123out['b123'] 
        q123 = b123out['q123'] 
        cnts = b123out['counts'] 

        hdr = ('halo bispectrum for sigma_8=%f, realization %i, redshift bin %i; k_f = 2pi/%f, Nhalo=%i' % 
                (sig8, nreal, nzbin, Lbox, xyz.shape[1]))
        np.savetxt(fbk, np.array([i_k, j_k, l_k, b123, q123, cnts, pki, pkj, pkl]).T, 
                fmt='%i %i %i %.5e %.5e %.5e %.5e %.5e %.5e', delimiter='\t', header=hdr)
    else: 
        if not silent: print('--- %s already exists ---' % fbk) 
    return None 


if __name__=="__main__":
    run = sys.argv[1]
    mnu_or_sig8 = float(sys.argv[2]) 
    nreal = int(sys.argv[3]) 
    nzbin = int(sys.argv[4]) 
    rsd = sys.argv[5]
    if rsd == 'r': zspace = False
    elif rsd == 'z': zspace = True

    if run == 'mneut': 
        haloBispectrum(mnu_or_sig8, nreal, nzbin, zspace=zspace, 
                Ngrid=360, Nmax=40, Ncut=3, step=3, silent=False, overwrite=False)
    elif run == 'sigma8': 
        haloBispectrum_sigma8(mnu_or_sig8, nreal, nzbin, zspace=zspace, 
                Ngrid=360, Nmax=40, Ncut=3, step=3, silent=False, overwrite=False)
    elif run == 'mneut_pk': 
        haloPk_periodic(mnu_or_sig8, nreal, nzbin, zspace=zspace, 
                Ngrid=360, Nmax=40, Ncut=3, step=3, silent=False, overwrite=False)
    elif run == 'sigma8_pk': 
        haloPk_periodic_sigma8(mnu_or_sig8, nreal, nzbin, zspace=zspace, 
                Ngrid=360, Nmax=40, Ncut=3, step=3, silent=False, overwrite=False)
