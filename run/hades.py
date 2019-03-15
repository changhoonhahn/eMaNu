'''

Scripts for creating data products of hades simulation bispectrum
measurements that's easy to deal with in the analysis. 

'''
import os 
import glob
import h5py 
import numpy as np 
# -- emanu --
from emanu import util as UT


def hadesPk_Mnu_hdf5(mneut, nzbin=4, rsd=True): 
    ''' write out HADES halo powerspectrum for given mnu and redshift bin to 
    hdf5 file for fast and easier access. 

    :param mneut: 
        Mnu value options are 0.0, 0.06, 0.1, 0.15 eV

    :param nzbin: (default: 4) 
        redshift bin number. So far only 4 is computed 
    '''
    # RSD or not  
    if rsd: str_rsd = '.zspace'
    else: str_rsd = '.rspace'
    # hades B(k) directory 
    hades_dir = os.path.join(UT.dat_dir(), 'plk', 'hades') 
    fpks = glob.glob(os.path.join(hades_dir, 
        'plk.groups.%seV.[0-9]*.nzbin%i%s.mhmin3200.0.Nmesh360.Nbin120.dat' %
        (str(mneut), nzbin, str_rsd)))
    npk = len(fpks) 

    print('\n%i powerspectrum files with Mnu=%.2feV zbin=%i in %s\n' % (npk, mneut, nzbin, str_rsd.strip('.')))  
    assert npk == 100, "you're missing powerspectrum files" 

    # load in all the files 
    for i, fpk in enumerate(fpks):
        # read in plk 
        k, _p0k, _p2k, _p4k = np.loadtxt(os.path.join(hades_dir, fpk), 
                skiprows=3, unpack=True, usecols=[0,1,2,3]) 
        # readin shot-noise from header 
        with open(fpk) as lines: 
            for i_l, line in enumerate(lines):
                if i_l == 1: 
                    str_sn = line
                    break 
        _sn = float(str_sn.strip().split('P_shotnoise')[-1])
        if i == 0: 
            sns = np.zeros((npk))
            p0k = np.zeros((npk, len(k)))
            p2k = np.zeros((npk, len(k)))
            p4k = np.zeros((npk, len(k)))
        sns[i] = _sn
        p0k[i,:] = _p0k
        p2k[i,:] = _p2k
        p4k[i,:] = _p4k

    # save to hdf5 file 
    fhdf5 = os.path.join(UT.dat_dir(), 'plk', 'hades.%seV.nzbin%i.mhmin3200.0%s.hdf5' % 
            (str(mneut), nzbin, str_rsd))
    print('will be written to %s\n' % fhdf5)
    f = h5py.File(fhdf5, 'w') 
    f.create_dataset('k', data=k)
    f.create_dataset('p0k', data=p0k) 
    f.create_dataset('p2k', data=p2k) 
    f.create_dataset('p4k', data=p4k) 
    f.create_dataset('shotnoise', data=sns) 
    f.close()
    return None 


def hadesPk_sig8_hdf5(sig8, nzbin=4, rsd=True): 
    ''' write out real-/redshift-space HADES halo powerspectrum with 0.0eV for given sigma8 
    and redshift bin to hdf5 file for fast and easier access. 

    :param sig8: 
        sigma8 value

    :param nzbin: (default: 4) 
        redshift bin number. So far only 4 is computed 
    '''

    # RSD or not  
    if rsd: str_rsd = '.zspace'
    else: str_rsd = '.rspace'
    # hades B(k) directory 
    hades_dir = os.path.join(UT.dat_dir(), 'plk', 'hades') 
    fpks = glob.glob(os.path.join(hades_dir, 
        'plk.groups.0.0eV.sig8%s.[0-9]*.nzbin%i%s.mhmin3200.0.Nmesh360.Nbin120.dat' %
        (str(sig8), nzbin, str_rsd)))
    npk = len(fpks) 

    print('\n%i powerspectrum files with Mnu=0eV sigma8=%.3f zbin=%i in %s\n' % 
            (npk, sig8, nzbin, str_rsd.strip('.')))  
    assert npk == 100, "you're missing powerspectrum files" 

    # load in all the files 
    for i, fpk in enumerate(fpks):
        # read in plk 
        k, _p0k, _p2k, _p4k = np.loadtxt(os.path.join(hades_dir, fpk), 
                skiprows=3, unpack=True, usecols=[0,1,2,3]) 
        # readin shot-noise from header 
        with open(fpk) as lines: 
            for i_l, line in enumerate(lines):
                if i_l == 1: 
                    str_sn = line
                    break 
        _sn = float(str_sn.strip().split('P_shotnoise')[-1])
        if i == 0: 
            sns = np.zeros((npk))
            p0k = np.zeros((npk, len(k)))
            p2k = np.zeros((npk, len(k)))
            p4k = np.zeros((npk, len(k)))
        sns[i] = _sn
        p0k[i,:] = _p0k
        p2k[i,:] = _p2k
        p4k[i,:] = _p4k

    # save to hdf5 file 
    fhdf5 = os.path.join(UT.dat_dir(), 'plk', 'hades.0.0eV.sig8_%.3f.nzbin%i.mhmin3200.0%s.hdf5' % 
            (sig8, nzbin, str_rsd))
    print('will be written to %s\n' % fhdf5)
    f = h5py.File(fhdf5, 'w') 
    f.create_dataset('k', data=k)
    f.create_dataset('p0k', data=p0k) 
    f.create_dataset('p2k', data=p2k) 
    f.create_dataset('p4k', data=p4k) 
    f.create_dataset('shotnoise', data=sns) 
    f.close()
    return None 


def hadesBk_Mnu_hdf5(mneut, nzbin=4, rsd=True): 
    ''' write out HADES halo bispectrum for given mnu and redshift bin to 
    hdf5 file for fast and easier access. 

    :param mneut: 
        Mnu value options are 0.0, 0.06, 0.1, 0.15 eV

    :param nzbin: (default: 4) 
        redshift bin number. So far only 4 is computed 
    '''
    # RSD or not  
    if rsd: str_rsd = '.zspace'
    else: str_rsd = '.rspace'
    # hades B(k) directory 
    hades_dir = os.path.join(UT.dat_dir(), 'bispectrum', 'hades') 
    fbks = glob.glob(os.path.join(hades_dir, 
        'groups.%seV.[0-9]*.nzbin%i.mhmin3200.0%s.Ngrid360.Nmax40.Ncut3.step3.pyfftw.dat' %
        (str(mneut), nzbin, str_rsd)))
    nbk = len(fbks) 

    print('\n%i bispectrum files with Mnu=%.2feV zbin=%i in %s\n' % (nbk, mneut, nzbin, str_rsd.strip('.')))  
    
    assert nbk == 100, "not the correct number of files!"
    
    # load in all the files 
    for i, fbk in enumerate(fbks):
        i_k, j_k, l_k, _p0k1, _p0k2, _p0k3, b123, q123, b_sn, cnts = np.loadtxt(
                os.path.join(hades_dir, fbk), skiprows=1, unpack=True, usecols=range(10)) 
        hdr = open(os.path.join(hades_dir, fbk)).readline().rstrip() 
        Nhalo = int(hdr.split('Nhalo=')[-1])
        if i == 0: 
            Nhalos = np.zeros((nbk)) 
            p0k1 = np.zeros((nbk, len(i_k)))
            p0k2 = np.zeros((nbk, len(i_k)))
            p0k3 = np.zeros((nbk, len(i_k)))
            bks = np.zeros((nbk, len(i_k)))
            qks = np.zeros((nbk, len(i_k)))
            bsn = np.zeros((nbk, len(i_k)))
        Nhalos[i] = Nhalo
        p0k1[i,:] = _p0k1
        p0k2[i,:] = _p0k2
        p0k3[i,:] = _p0k3
        bks[i,:] = b123
        qks[i,:] = q123
        bsn[i,:] = b_sn 

    # save to hdf5 file 
    fhdf5 = os.path.join(UT.dat_dir(), 'bispectrum', 'hades.%seV.nzbin%i.mhmin3200.0%s.hdf5' % (str(mneut), nzbin, str_rsd))
    print('will be written to %s\n' % fhdf5)
    f = h5py.File(fhdf5, 'w') 
    f.create_dataset('k1', data=i_k)
    f.create_dataset('k2', data=j_k)
    f.create_dataset('k3', data=l_k)
    f.create_dataset('p0k1', data=p0k1) 
    f.create_dataset('p0k2', data=p0k2) 
    f.create_dataset('p0k3', data=p0k3) 
    f.create_dataset('b123', data=bks) 
    f.create_dataset('q123', data=qks) 
    f.create_dataset('b_sn', data=bsn) 
    f.create_dataset('counts', data=cnts) 
    f.create_dataset('Nhalos', data=Nhalos) 
    f.close()
    return None 


def hadesBk_sig8_hdf5(sig8, nzbin=4, rsd=True): 
    ''' write out HADES halo bispectrum for given mnu and redshift bin to 
    hdf5 file for fast and easier access. 

    :param mneut: 
        Mnu value options are 0.0, 0.06, 0.1, 0.15 eV

    :param nzbin: (default: 4) 
        redshift bin number. So far only 4 is computed 
    '''
    # RSD or not  
    if rsd: str_rsd = '.zspace'
    else: str_rsd = '.rspace'
    # hades B(k) directory 
    hades_dir = os.path.join(UT.dat_dir(), 'bispectrum', 'hades') 
    fbks = glob.glob(os.path.join(hades_dir, 
        'groups.0.0eV.sig8_%s.*.nzbin%i.mhmin3200.0%s.Ngrid360.Nmax40.Ncut3.step3.pyfftw.dat' %
        (str(sig8), nzbin, str_rsd)))
    nbk = len(fbks) 

    print('\n%i bispectrum files with Mnu=0.0eV sigma8=%.2f zbin=%i in %s\n' % (nbk, sig8, nzbin, str_rsd.strip('.')))  
    
    assert nbk == 100, "not the correct number of files!"
    
    # load in all the files 
    for i, fbk in enumerate(fbks):
        i_k, j_k, l_k, _p0k1, _p0k2, _p0k3, b123, q123, b_sn, cnts = np.loadtxt(
                os.path.join(hades_dir, fbk), skiprows=1, unpack=True, usecols=range(10)) 
        hdr = open(os.path.join(hades_dir, fbk)).readline().rstrip() 
        Nhalo = int(hdr.split('Nhalo=')[-1])
        if i == 0: 
            Nhalos = np.zeros((nbk)) 
            p0k1 = np.zeros((nbk, len(i_k)))
            p0k2 = np.zeros((nbk, len(i_k)))
            p0k3 = np.zeros((nbk, len(i_k)))
            bks = np.zeros((nbk, len(i_k)))
            qks = np.zeros((nbk, len(i_k)))
            bsn = np.zeros((nbk, len(i_k)))
        Nhalos[i] = Nhalo
        p0k1[i,:] = _p0k1
        p0k2[i,:] = _p0k2
        p0k3[i,:] = _p0k3
        bks[i,:] = b123
        qks[i,:] = q123
        bsn[i,:] = b_sn 

    # save to hdf5 file 
    fhdf5 = os.path.join(UT.dat_dir(), 'bispectrum', 
            'hades.0.0eV.sig8_%s.nzbin%i.mhmin3200.0%s.hdf5' % (str(sig8), nzbin, str_rsd))
    print('will be written to %s\n' % fhdf5)
    f = h5py.File(fhdf5, 'w') 
    f.create_dataset('k1', data=i_k)
    f.create_dataset('k2', data=j_k)
    f.create_dataset('k3', data=l_k)
    f.create_dataset('p0k1', data=p0k1) 
    f.create_dataset('p0k2', data=p0k2) 
    f.create_dataset('p0k3', data=p0k3) 
    f.create_dataset('b123', data=bks) 
    f.create_dataset('q123', data=qks) 
    f.create_dataset('b_sn', data=bsn) 
    f.create_dataset('counts', data=cnts) 
    f.create_dataset('Nhalos', data=Nhalos) 
    f.close()
    return None 


if __name__=="__main__":
    for mnu in [0.0, 0.06, 0.1, 0.15]: 
        hadesPk_Mnu_hdf5(mnu, nzbin=4, rsd=False)
        hadesPk_Mnu_hdf5(mnu, nzbin=4, rsd=True)
        #hadesBk_Mnu_hdf5(mnu, nzbin=4, rsd=False)
        #hadesBk_Mnu_hdf5(mnu, nzbin=4, rsd=True)
    for sig8 in [0.822, 0.818, 0.807, 0.798]:
        hadesPk_sig8_hdf5(sig8, nzbin=4, rsd=False)
        hadesPk_sig8_hdf5(sig8, nzbin=4, rsd=True)
        #hadesBk_sig8_hdf5(sig8, nzbin=4, rsd=False)
        #hadesBk_sig8_hdf5(sig8, nzbin=4, rsd=True)
