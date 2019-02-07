'''

Some scripts for dealing with the quijote simulation bispectrum
measurements. 


'''
import os 
import glob
import h5py 
import numpy as np 
# -- emanu --
from emanu import util as UT
# -- plotting -- 
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter, NullFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.xmargin'] = 1
mpl.rcParams['xtick.labelsize'] = 'x-large'
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.labelsize'] = 'x-large'
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['legend.frameon'] = False


def quijote_comparison(par): 
    ''' Compare quijote simulations across a specified parameter `par`
    
    :param par: 
        string that specifies the parameter to compare along  
    '''
    # gather quijote bispectra for parameter par 
    dir_bk = os.path.join(UT.dat_dir(), 'bispectrum') 
    fbks = glob.glob(dir_bk+'/'+'quijote_%s*.hdf5' % par)

    fbks =  
    fig = plt.figure()


def _hdf5_quijote(subdir): 
    ''' write out quijote bispectrum transferred from cca server to hdf5 
    file for fast and easier access in the future. 

    :param subdir: 
        name of subdirectory which also describes the run.
    '''
    dir_quij = os.path.join(UT.dat_dir(), 'bispectrum', 'quijote', subdir) 
    fbks = os.listdir(dir_quij) 
    nbk = len(fbks) 
    print('%i bispectrum files in /%s' % (nbk, subdir))  
    
    # check the number of bispectrum
    if subdir == 'fiducial': assert nbk == 15000, "not the right number of files"
    else: assert nbk == 500, "not the right number of files"
    
    # load in all the files 
    for i, fbk in enumerate(fbks):
        i_k, j_k, l_k, _p0k1, _p0k2, _p0k3, b123, q123, b_sn, cnts = np.loadtxt(
                os.path.join(dir_quij, fbk), skiprows=1, unpack=True, usecols=range(10)) 
        if i == 0: 
            p0k1 = np.zeros((nbk, len(i_k)))
            p0k2 = np.zeros((nbk, len(i_k)))
            p0k3 = np.zeros((nbk, len(i_k)))
            bks = np.zeros((nbk, len(i_k)))
            qks = np.zeros((nbk, len(i_k)))
            bsn = np.zeros((nbk, len(i_k)))
        p0k1[i,:] = _p0k1
        p0k2[i,:] = _p0k2
        p0k3[i,:] = _p0k3
        bks[i,:] = b123
        qks[i,:] = q123
        bsn[i,:] = b_sn 

    # save to hdf5 file 
    f = h5py.File(os.path.join(UT.dat_dir(), 'bispectrum', 'quijote_%s.hdf5' % subdir), 'w') 
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
    f.close()
    return None 


if __name__=="__main__": 
    quijote_comparison('Mnu')
    #for sub in ['Mnu_p', 'Mnu_pp', 'Mnu_ppp', 'Ob_m', 'Ob_p', 'Om_m', 'Om_p', 
    #        'fiducial', 'fiducial_NCV', 'h_m', 'h_p', 'ns_m', 'ns_p', 's8_m', 's8_p']: 
    #    _hdf5_quijote(sub)
