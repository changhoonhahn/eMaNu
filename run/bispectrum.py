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

        # calculate bispectrum 
        b123out = pySpec.Bk_periodic(xyz, Lbox=Lbox, Ngrid=Ngrid, 
                step=step, Ncut=Ncut, Nmax=Nmax, fft='pyfftw', nthreads=1, silent=silent) 
        i_k = b123out['i_k1']
        j_k = b123out['i_k2']
        l_k = b123out['i_k3']
        p0k1 = b123out['p0k1']
        p0k2 = b123out['p0k2']
        p0k3 = b123out['p0k3']
        b123 = b123out['b123'] 
        b_sn = b123out['b123_sn'] 
        q123 = b123out['q123'] 
        cnts = b123out['counts'] 
       
        hdr = ('halo bispectrum for mneut=%f, realization %i, redshift bin %i; k_f = 2pi/%.1f, Nhalo=%i' % 
                (mneut, nreal, nzbin, Lbox, xyz.shape[1]))
        np.savetxt(fbk, np.array([i_k, j_k, l_k, p0k1, p0k2, p0k3, b123, q123, b_sn, cnts]).T, 
                        fmt='%i %i %i %.5e %.5e %.5e %.5e %.5e %.5e %.5e', delimiter='\t', header=hdr)
        #np.savetxt(fbk, np.array([i_k, j_k, l_k, b123, q123, cnts, pki, pkj, pkl]).T, 
        #        fmt='%i %i %i %.5e %.5e %.5e %.5e %.5e %.5e', delimiter='\t', header=hdr)
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
        
        # calculate bispectrum 
        b123out = pySpec.Bk_periodic(xyz, Lbox=Lbox, Ngrid=Ngrid, 
                step=step, Ncut=Ncut, Nmax=Nmax, fft='pyfftw', nthreads=1, silent=silent) 
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
        np.savetxt(fbk, np.array([i_k, j_k, l_k, p0k1, p0k2, p0k3, b123, q123, b_sn, cnts]).T, 
                fmt='%i %i %i %.5e %.5e %.5e %.5e %.5e %.5e %.5e', delimiter='\t', header=hdr)
    else: 
        if not silent: print('--- %s already exists ---' % fbk) 
    return None 


def fix_B123_shotnoise(): 
    ''' fix HADES halo bispecturm outputs to correctly account for shot noise 
    without having to rerun the bispectrum Jan 31, 2019 
    '''
    fhalo = lambda mneut, nreal, str_rsd: ''.join([UT.dat_dir(), 'halos/', 
        'groups.', str(mneut), 'eV.', str(nreal), '.nzbin4.mhmin3200.0.hdf5']) 

    fbk = lambda mneut, nreal, str_rsd: ''.join([UT.dat_dir(), 'bispectrum/', 
        'groups.', str(mneut), 'eV.', str(nreal), '.nzbin4.mhmin3200.0.', str_rsd, 'space',
        '.Ngrid360.Nmax40.Ncut3.step3.pyfftw.dat']) 
    
    ffix = lambda mneut, nreal, str_rsd: ''.join([UT.dat_dir(), 'bispectrum/fix/', 
        'groups.', str(mneut), 'eV.', str(nreal), '.nzbin4.mhmin3200.0.', str_rsd, 'space',
        '.Ngrid360.Nmax40.Ncut3.step3.pyfftw.dat']) 

    for mneut in [0.0, 0.06, 0.1, 0.15]: 
        for i in range(1,101): 
            for zstr in ['r', 'z']: 
                halo = h5py.File(fhalo(mneut, i, zstr), 'r') 
                Lbox = halo.attrs['Lbox']
                kf = 2 * np.pi / Lbox

                Nhalo = halo['Position'].value.shape[0]
                nhalo = float(Nhalo)/Lbox**3
                print('nbar = %f' % nhalo) 
                print('1/nbar = %f' % (1./nhalo))

                i_k, j_k, l_k, b123, q123, cnts = np.loadtxt(fbk(mneut, i, zstr), 
                        unpack=True, skiprows=1, usecols=[0,1,2,3,4,5]) 
        
                xyz = halo['Position'].value.T
                if zstr == 'z': # impose redshift space distortions on the z-axis 
                    v_offset = halo['VelocityOffset'].value.T
                    xyz += (halo['VelocityOffset'].value * [0, 0, 1]).T + Lbox
                    xyz = (xyz % Lbox) 
    
                delta = pySpec.FFTperiodic(xyz, fft='fortran', Lbox=Lbox, Ngrid=360, silent=True) 
                delta_fft = pySpec.reflect_delta(delta, Ngrid=360) 
                p0k1, p0k2, p0k3 = pySpec._pk_Bk123_periodic(delta_fft, 
                        Nmax=40, Ncut=3, step=3, fft='pyfftw', nthreads=1, silent=True)
                assert len(p0k1) == len(i_k) 
                assert len(p0k2) == len(j_k) 
                assert len(p0k3) == len(l_k) 

                p0k1 = p0k1 * (2 * np.pi)**3 / kf**3 - 1./nhalo
                p0k2 = p0k2 * (2 * np.pi)**3 / kf**3 - 1./nhalo
                p0k3 = p0k3 * (2 * np.pi)**3 / kf**3 - 1./nhalo
                b_sn = (p0k1 + p0k2 + p0k3)/nhalo + 1./nhalo**2

                b123 = b123 * (2.*np.pi)**6 / kf**6 - b_sn 
                q123 = b123 / (p0k1*p0k2 + p0k1*p0k3 + p0k2*p0k3) 
        
                hdr = ('halo bispectrum for mneut=%.2f, realization %i, redshift bin %i; k_f = 2pi/%.1f, Nhalo=%i' % 
                        (mneut, i, 4, Lbox, Nhalo))
                np.savetxt(ffix(mneut, i, zstr), 
                        np.array([i_k, j_k, l_k, p0k1, p0k2, p0k3, b123, q123, b_sn, cnts]).T, 
                        fmt='%i %i %i %.5e %.5e %.5e %.5e %.5e %.5e %.5e', delimiter='\t', header=hdr)

    fhalo = lambda sig8, nreal, str_rsd: ''.join([UT.dat_dir(), 'halos/', 
        'groups.0.0eV.sig8_', str(sig8), '.', str(nreal), '.nzbin4.mhmin3200.0.hdf5']) 

    fbk = lambda sig8, nreal, str_rsd: ''.join([UT.dat_dir(), 'bispectrum/', 
        'groups.0.0eV.sig8_', str(sig8), '.', str(nreal), '.nzbin4.mhmin3200.0.', str_rsd, 'space',
        '.Ngrid360.Nmax40.Ncut3.step3.pyfftw.dat']) 
    
    ffix = lambda sig8, nreal, str_rsd: ''.join([UT.dat_dir(), 'bispectrum/fix/', 
        'groups.0.0eV.sig8_', str(sig8), '.', str(nreal), '.nzbin4.mhmin3200.0.', str_rsd, 'space',
        '.Ngrid360.Nmax40.Ncut3.step3.pyfftw.dat']) 

    for sig8 in [0.798, 0.807, 0.818, 0.822]: 
        for i in range(1,101): 
            for zstr in ['r', 'z']: 
                halo = h5py.File(fhalo(sig8, i, zstr), 'r') 
                Lbox = halo.attrs['Lbox']
                kf = 2 * np.pi / Lbox

                Nhalo = halo['Position'].value.shape[0]
                nhalo = float(Nhalo)/Lbox**3
                print('nbar = %f' % nhalo) 
                print('1/nbar = %f' % (1./nhalo))

                i_k, j_k, l_k, b123, q123, cnts = np.loadtxt(fbk(sig8, i, zstr), 
                        unpack=True, skiprows=1, usecols=[0,1,2,3,4,5]) 
        
                xyz = halo['Position'].value.T
                if zstr == 'z': # impose redshift space distortions on the z-axis 
                    v_offset = halo['VelocityOffset'].value.T
                    xyz += (halo['VelocityOffset'].value * [0, 0, 1]).T + Lbox
                    xyz = (xyz % Lbox) 
    
                delta = pySpec.FFTperiodic(xyz, fft='fortran', Lbox=Lbox, Ngrid=360, silent=True) 
                delta_fft = pySpec.reflect_delta(delta, Ngrid=360) 
                p0k1, p0k2, p0k3 = pySpec._pk_Bk123_periodic(delta_fft, 
                        Nmax=40, Ncut=3, step=3, fft='pyfftw', nthreads=1, silent=True)
                assert len(p0k1) == len(i_k) 
                assert len(p0k2) == len(j_k) 
                assert len(p0k3) == len(l_k) 
                p0k1 = p0k1 * (2 * np.pi)**3 / kf**3 - 1./nhalo
                p0k2 = p0k2 * (2 * np.pi)**3 / kf**3 - 1./nhalo
                p0k3 = p0k3 * (2 * np.pi)**3 / kf**3 - 1./nhalo

                b_sn = (p0k1 + p0k2 + p0k3)/nhalo + 1./nhalo**2

                b123 = b123 * (2.*np.pi)**6 / kf**6 - b_sn 
                q123 = b123 / (p0k1*p0k2 + p0k1*p0k3 + p0k2*p0k3) 
        
                hdr = ('halo bispectrum for sigma_8=%f, realization %i, redshift bin %i; k_f = 2pi/%.1f, Nhalo=%i' % 
                        (sig8, i, 4, Lbox, Nhalo))
        
                np.savetxt(ffix(sig8, i, zstr), 
                        np.array([i_k, j_k, l_k, p0k1, p0k2, p0k3, b123, q123, b_sn, cnts]).T, 
                        fmt='%i %i %i %.5e %.5e %.5e %.5e %.5e %.5e %.5e', delimiter='\t', header=hdr)
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
    elif run == 'fix': 
        fix_B123_shotnoise()
