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


def quijote_hdf5(subdir, machine='mbp', rsd=0, flag=None): 
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

    if subdir == 'fiducial' and flag is not None: raise ValueError('fiducial does not have flag') 

    if rsd != 'real': # redshift-space B
        if flag is None: 
            # include all redshift-space files (fixed paired and reg. N-body in all RSD directions) 
            fbks = glob.glob(os.path.join(dir_quij, '*RS*'))
        elif flag == 'ncv': 
            # fixed pair (rsd direction) 
            fbks = glob.glob(os.path.join(dir_quij, '*RS%i_NCV*' % rsd)) 
        elif flag == 'reg':
            # regular N-body (rsd direction) 
            fbks = glob.glob(os.path.join(dir_quij, '*RS%i*' % rsd))
            fbks_ncv = glob.glob(os.path.join(dir_quij, '*RS%i_NCV*' % rsd)) 
            fbks = list(set(fbks) - set(fbks_ncv))
    else: # real-space B 
        if flag is None: 
            # include all redshift-space files (fixed paired and reg. N-body in all RSD directions) 
            fbks = glob.glob(os.path.join(dir_quij, '*'))
            fbks_rsd = glob.glob(os.path.join(dir_quij, '*RS*'))
            fbks = list(set(fbks) - set(fbks_rsd))
        elif flag == 'ncv': 
            # fixed pair (rsd direction) 
            fbks = glob.glob(os.path.join(dir_quij, '*NCV*')) 
            fbks_rsd = glob.glob(os.path.join(dir_quij, '*RS*_NCV*')) 
            fbks = list(set(fbks) - set(fbks_rsd))
        elif flag == 'reg':
            # regular N-body (rsd direction) 
            fbks = glob.glob(os.path.join(dir_quij, '*'))
            fbks_rsd = glob.glob(os.path.join(dir_quij, '*RS*')) 
            fbks_ncv = glob.glob(os.path.join(dir_quij, '*NCV*')) 
            fbks = list(set(fbks) - set(fbks_rsd) - set(fbks_ncv))

    print([os.path.basename(fbk) for fbk in fbks[::10]])
    nbk = len(fbks) 
    print('%i bispectrum files in /%s' % (nbk, subdir))  
    
    # check the number of bispectrum
    if subdir == 'fiducial': assert nbk == 15000, "not the right number of files"
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
        if flag is None: fhdf5 = os.path.join(quij_dir, 'quijote_%s.hdf5' % subdir)
        else: fhdf5 = os.path.join(quij_dir, 'quijote_%s.%s.rsd%i.hdf5' % (subdir, flag, rsd))
    else: 
        if flag is None: fhdf5 = os.path.join(quij_dir, 'quijote_%s.real.hdf5' % subdir)
        else: fhdf5 = os.path.join(quij_dir, 'quijote_%s.%s.real.hdf5' % (subdir, flag))
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
    thetas = ['Mnu_p', 'Mnu_pp', 'Mnu_ppp', 'Om_m', 'Om_p', 'Ob2_m', 'Ob2_p', 'h_m', 'h_p', 'ns_m', 'ns_p', 's8_m', 's8_p', 
            'Mmin_m', 'Mmin_p']
    thetas = ['h_p', 's8_p', 'Mmin_m', 'Mmin_p'] 
    for sub in thetas:
        print('---%s---' % sub) 
        quijote_hdf5(sub) # all redshift-space files
        quijote_hdf5(sub, rsd='real') # all real-space files
        for rsd in [0, 1, 2, 'real']: 
            quijote_hdf5(sub, flag='ncv', rsd=rsd)
            quijote_hdf5(sub, flag='reg', rsd=rsd)
    #quijote_hdf5('fiducial')
