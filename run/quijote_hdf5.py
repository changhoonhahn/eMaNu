'''

script for processing B(k) files of Paco's sims into easy to 
access hdf5 files

'''
import os 
import glob
import h5py 
import numpy as np 
# -- emanu --
from emanu import util as UT


def quijoteP_hdf5(subdir, machine='mbp', rsd=0, flag=None): 
    ''' write out quijote powerspectrum transferred from cca server to hdf5 
    file for fast and easier access in the future. 
    **currently only implemented for z = 0**

    :param subdir: 
        name of subdirectory which also describes the run.

    :param machine: (default: 'mbp') 
        string that specifies which machine. 

    :param rsd: (default: 0) 
        int (0, 1, 2) determines the direction of the RSD. string 'real' is for real-space

    :param flag: (default: None) 
        string specifying the runs. For the perturbed thetas this refers to 
        either regular Nbody runs or fixed-paired (reg vs ncv). If None it reads 
        in all the files in the directory 
    '''
    # directory where the quijote bispectrum are at 
    if machine in ['mbp', 'cori']: dir_quij = os.path.join(UT.dat_dir(), 'bispectrum', 'quijote', subdir) 
    else: raise NotImplementedError 
    
    if rsd not in [0, 1, 2, 'real']: raise ValueError 
    if subdir == 'fiducial' and flag == 'reg': nmocks = 15000
    elif subdir == 'fiducial' and flag == 'ncv': nmocks = 500
    else: nmocks = 500 

    fpks = [] 
    if rsd != 'real' and flag == 'reg': 
        for i in range(nmocks): 
            fpks.append('Pk_RS%i_%i_z=0.txt' % (rsd, i)) 
    elif rsd != 'real' and flag == 'ncv':
        for i in range(nmocks/2): 
            fpks.append('Pk_RS%i_NCV_0_%i_z=0.txt' % (rsd, i)) 
            fpks.append('Pk_RS%i_NCV_1_%i_z=0.txt' % (rsd, i)) 
    elif rsd == 'real' and flag == 'reg': 
        for i in range(nmocks): 
            fpks.append('Pk_%i_z=0.txt' % i) 
    elif rsd == 'real' and flag == 'ncv': 
        for i in range(nmocks/2): 
            fpks.append('Pk_NCV_0_%i_z=0.txt' % i)
            fpks.append('Pk_NCV_1_%i_z=0.txt' % i)

    print([os.path.basename(fpk) for fpk in fpks[::10][:10]])
    npk = len(fpks) 
    print('%i bispectrum files in /%s' % (npk, subdir))  
    
    # check the number of bispectrum
    #if subdir == 'fiducial': assert nbk == 15000, "not the right number of files"
    #else: if nbk != 500 and nbk != 1000: print('not the right number of files') 
    
    # load in all the files 
    for i, fpk in enumerate(fpks):
        k, _p0k, _p2k, _p4k = np.loadtxt(os.path.join(dir_quij, fpk), skiprows=1, unpack=True, usecols=range(4)) 
        hdr = open(os.path.join(dir_quij, fpk)).readline().rstrip() 
        # Nhalos=159315 BoxSize=1000.000
        Nhalo = int(hdr.split(' ')[0].split('Nhalo=')[-1])
        nbar = float(Nhalo)/1000.**3 
 
        if i == 0: 
            Nhalos = np.zeros((nbk)) 
            p0k = np.zeros((nbk, len(k)))
            p2k = np.zeros((nbk, len(k)))
            p4k = np.zeros((nbk, len(k)))
            psn = np.zeros((nbk))
        Nhalos[i] = Nhalo
        psn[i] = 1./nbar
        p0k[i,:] = _p0k - 1./nbar 
        p2k[i,:] = _p2k
        p4k[i,:] = _p4k

    # save to hdf5 file 
    quij_dir = os.path.join(UT.dat_dir(), 'powerspectrum', 'quijote', 'z0') 
    if rsd != 'real':  # reshift space 
        fhdf5 = os.path.join(quij_dir, 'quijote_%s.%s.rsd%i.hdf5' % (subdir, flag, rsd))
    else: 
        fhdf5 = os.path.join(quij_dir, 'quijote_%s.%s.real.hdf5' % (subdir, flag))
    f = h5py.File(fhdf5, 'w') 
    f.create_dataset('k', data=i_k)
    f.create_dataset('p0k', data=p0k1) 
    f.create_dataset('p2k', data=p0k2) 
    f.create_dataset('p4k', data=p0k3) 
    f.create_dataset('p_sn', data=bsn) 
    f.create_dataset('Nhalos', data=Nhalos) 
    f.create_dataset('files', data=fpks) 
    f.close()
    return None 


def quijoteB_hdf5(subdir, machine='mbp', rsd=0, flag=None): 
    ''' write out quijote bispectrum transferred from cca server to hdf5 
    file for fast and easier access in the future. 
    **currently only implemented for z = 0**

    :param subdir: 
        name of subdirectory which also describes the run.

    :param machine: (default: 'mbp') 
        string that specifies which machine. 

    :param rsd: (default: 0) 
        int (0, 1, 2) determines the direction of the RSD. string 'real' is for real-space

    :param flag: (default: None) 
        string specifying the runs. For the perturbed thetas this refers to 
        either regular Nbody runs or fixed-paired (reg vs ncv). If None it reads 
        in all the files in the directory 
    '''
    # directory where the quijote bispectrum are at 
    if machine in ['mbp', 'cori']: dir_quij = os.path.join(UT.dat_dir(), 'bispectrum', 'quijote', subdir) 
    else: raise NotImplementedError 
    
    if rsd not in [0, 1, 2, 'real']: raise ValueError 
    if subdir == 'fiducial' and flag == 'reg': nmocks = 15000
    elif subdir == 'fiducial' and flag == 'ncv': nmocks = 500
    else: nmocks = 500 

    fbks = [] 
    if rsd != 'real' and flag == 'reg': 
        for i in range(nmocks): 
            fbks.append('Bk_RS%i_%i_z=0.txt' % (rsd, i)) 
    elif rsd != 'real' and flag == 'ncv':
        for i in range(nmocks/2): 
            fbks.append('Bk_RS%i_NCV_0_%i_z=0.txt' % (rsd, i)) 
            fbks.append('Bk_RS%i_NCV_1_%i_z=0.txt' % (rsd, i)) 
    elif rsd == 'real' and flag == 'reg': 
        for i in range(nmocks): 
            fbks.append('Bk_%i_z=0.txt' % i) 
    elif rsd == 'real' and flag == 'ncv': 
        for i in range(nmocks/2): 
            fbks.append('Bk_NCV_0_%i_z=0.txt' % i)
            fbks.append('Bk_NCV_1_%i_z=0.txt' % i)

    print([os.path.basename(fbk) for fbk in fbks[::10][:10]])
    nbk = len(fbks) 
    print('%i bispectrum files in /%s' % (nbk, subdir))  
    
    # check the number of bispectrum
    #if subdir == 'fiducial': assert nbk == 15000, "not the right number of files"
    #else: if nbk != 500 and nbk != 1000: print('not the right number of files') 
    
    # load in all the files 
    for i, fbk in enumerate(fbks):
        i_k, j_k, l_k, _p0k1, _p0k2, _p0k3, b123, q123, b_sn, cnts = np.loadtxt(
                os.path.join(dir_quij, fbk), skiprows=1, unpack=True, usecols=range(10)) 
        hdr = open(os.path.join(dir_quij, fbk)).readline().rstrip() 
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
    quij_dir = os.path.join(UT.dat_dir(), 'bispectrum', 'quijote', 'z0') 
    if rsd != 'real':  # reshift space 
        fhdf5 = os.path.join(quij_dir, 'quijote_%s.%s.rsd%i.hdf5' % (subdir, flag, rsd))
    else: 
        fhdf5 = os.path.join(quij_dir, 'quijote_%s.%s.real.hdf5' % (subdir, flag))
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
    f.create_dataset('files', data=fbks) 
    f.close()
    return None 


if __name__=="__main__": 
    thetas = ['Mnu_p', 'Mnu_pp', 'Mnu_ppp', 'Om_m', 'Om_p', 'Ob2_m', 'Ob2_p', 'h_m', 'h_p', 'ns_m', 'ns_p', 's8_m', 's8_p', 
            'Mmin_m', 'Mmin_p', 'fiducial']
    for sub in thetas:
        print('---%s---' % sub) 
        for rsd in [0, 1, 2, 'real']: 
            quijoteP_hdf5(sub, rsd=rsd, flag='ncv')
            quijoteP_hdf5(sub, rsd=rsd, flag='reg')
            continue 
            quijoteB_hdf5(sub, rsd=rsd, flag='ncv')
            quijoteB_hdf5(sub, rsd=rsd, flag='reg')
