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


def quijote_covariance(krange=[0.01, 0.5]):
    ''' calculate the covariance matrix using the 15,000 fiducial 
    bispetra.
    '''
    # read in fiducial bispectra
    dir_bk = os.path.join(UT.dat_dir(), 'bispectrum') 
    bks = h5py.File(os.path.join(dir_bk, 'quijote_fiducial.hdf5'), 'r') 
    i_k, j_k, l_k = bks['k1'].value, bks['k2'].value, bks['k3'].value 
    pk1, pk2, pk3 = bks['p0k1'].value, bks['p0k2'].value, bks['p0k3'].value
    bk = bks['b123'].value 
    
    # k range limit of the bispectrum
    kmin, kmax = krange  
    kf = 2.*np.pi/1000. # fundmaentla mode
    klim = ((i_k*kf <= kmax) & (i_k*kf >= kmin) &
            (j_k*kf <= kmax) & (j_k*kf >= kmin) & 
            (l_k*kf <= kmax) & (l_k*kf >= kmin)) 
    
    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 
    ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # order of triangles 

    # evaluate covariance matrix
    _bk = bk[:,klim]
    C_bk = np.cov(_bk[:,ijl.flatten()].T) 
    assert np.sum(klim) == C_bk.shape[0]

    # write C_bk to hdf5 file 
    f = h5py.File(os.path.join(UT.dat_dir(), 'bispectrum', 'quijote_Cov_bk.%.2f_%.2f.gmorder.hdf5' % (kmin, kmax)), 'w') 
    f.create_dataset('k1', data=i_k[ijl])
    f.create_dataset('k2', data=j_k[ijl])
    f.create_dataset('k3', data=l_k[ijl])
    f.create_dataset('C_bk', data=C_bk) 
    f.close()
    return None 


def quijote_comparison(par, krange=[0.01, 0.5]): 
    ''' Compare quijote simulations across a specified parameter `par`
    
    :param par: 
        string that specifies the parameter to compare along  

    :param krange: (default: [0.01, 0.5]) 
        k range to limit the bispectrum
    '''
    kmin, kmax = krange  
    kf = 2.*np.pi/1000. # fundmaentla mode

    # gather quijote bispectra for parameter par 
    dir_bk = os.path.join(UT.dat_dir(), 'bispectrum') 
    fbks = glob.glob(dir_bk+'/'+'quijote_%s*.hdf5' % par)
    # labels for the files 
    lbls = [fbk.rsplit('quijote_', 1)[-1].rsplit('.hdf5')[0].replace('_p', '+').replace('p', '+').replace('_m', '-') 
            for fbk in fbks]

    fig = plt.figure(figsize=(30,5))
    gs = mpl.gridspec.GridSpec(1,7, figure=fig) 
    sub0 = plt.subplot(gs[0,:2])  # pk panel
    sub1 = plt.subplot(gs[0,2:]) # bk panel
    axins = inset_axes(sub1, loc='upper right', width="40%", height="45%") 

    for i, fbk, lbl in zip(range(len(fbks)), fbks, lbls): # loop through the bispectrum
        bks = h5py.File(fbk, 'r') 
        i_k, j_k, l_k = bks['k1'].value, bks['k2'].value, bks['k3'].value 
        pk1, pk2, pk3 = bks['p0k1'].value, bks['p0k2'].value, bks['p0k3'].value
        bk = bks['b123'].value 
    
        # plot P(k) 
        klim = (i_k*kf <= kmax) & (i_k*kf >= kmin) 
        isort = np.argsort(i_k[klim]) 
        kk = kf * i_k[klim][isort]
        avgpk = np.average(pk1, axis=0)[klim][isort]
        sub0.plot(kk, avgpk, c='C'+str(i), label=lbl) 
        
        # plot B(k1,k2,k3) 
        klim = ((i_k*kf <= kmax) & (i_k*kf >= kmin) &
                (j_k*kf <= kmax) & (j_k*kf >= kmin) & 
                (l_k*kf <= kmax) & (l_k*kf >= kmin)) 
        i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 

        ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # order of triangles 
        avgbk = np.average(bk, axis=0)[klim][ijl]
        sub1.plot(range(np.sum(klim)), avgbk, c='C'+str(i)) 
        axins.plot(range(np.sum(klim)), avgbk, c='C'+str(i))

    # plot fiducial 
    bks = h5py.File(os.path.join(dir_bk, 'quijote_fiducial.hdf5'), 'r') 
    i_k, j_k, l_k = bks['k1'].value, bks['k2'].value, bks['k3'].value 
    pk1, pk2, pk3 = bks['p0k1'].value, bks['p0k2'].value, bks['p0k3'].value
    bk = bks['b123'].value 
    # plot P(k) 
    klim = (i_k*kf <= kmax) & (i_k*kf >= kmin) 
    isort = np.argsort(i_k[klim]) 
    kk = kf * i_k[klim][isort]
    avgpk = np.average(pk1, axis=0)[klim][isort]
    stdpk = np.std(pk1, axis=0)[klim][isort]
    sub0.fill_between(kk, avgpk - stdpk, avgpk + stdpk, color='k', alpha=0.25, linewidth=0, label='fiducial') 
    
    # plot B(k1,k2,k3) 
    klim = ((i_k*kf <= kmax) & (i_k*kf >= kmin) &
            (j_k*kf <= kmax) & (j_k*kf >= kmin) & 
            (l_k*kf <= kmax) & (l_k*kf >= kmin)) 
    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 
    ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # order of triangles 
    avgbk = np.average(bk, axis=0)[klim][ijl]
    stdbk = np.std(bk, axis=0)[klim][ijl]
    sub1.fill_between(np.arange(np.sum(klim)), (avgbk - stdbk).flatten(), (avgbk + stdbk).flatten(), 
            color='k', alpha=0.25, linewidth=0) 
    axins.fill_between(np.arange(np.sum(klim)), (avgbk - stdbk).flatten(), (avgbk + stdbk).flatten(), 
            color='k', alpha=0.25, linewidth=0) 

    # pk panel 
    sub0.legend(loc='lower left', ncol=1, handletextpad=0.25, columnspacing=0.5, fontsize=18) 
    sub0.set_xlabel("$k$ [$h$/Mpc]", fontsize=25) 
    sub0.set_xscale("log") 
    sub0.set_xlim([1e-2, 0.5])
    sub0.set_xticks([1e-2, 1e-1, 0.5]) 
    sub0.set_xticklabels([r'0.01', '0.1', r'0.5'])
    sub0.set_ylabel('$P_0(k)$', fontsize=25) 
    sub0.set_yscale("log") 
    sub0.set_ylim([1e3, 1e5]) 
    # bk panel
    sub1.set_yscale('log') 
    axins.set_yscale('log') 
    sub1.set_xlabel(r'$k_1 \le k_2 \le k_3$ triangle indices', labelpad=5, fontsize=25) 
    sub1.set_xlim([0, np.sum(klim)])
    sub1.set_ylabel('$B(k_1, k_2, k_3)$', labelpad=5, fontsize=25) 
    sub1.set_ylim([1e6, 1e10]) 
    axins.set_xlim(480, 500)
    axins.set_ylim(5e7, 2e8) 
    axins.set_xticklabels('') 
    axins.yaxis.set_minor_formatter(NullFormatter())
    mark_inset(sub1, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    fig.subplots_adjust(wspace=1.) 
    fig.savefig(os.path.join(UT.fig_dir(), 'quijote_%s.png' % par), bbox_inches='tight') 
    return None 


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
    # write to hdf5 
    #for sub in ['Mnu_p', 'Mnu_pp', 'Mnu_ppp', 'Ob_m', 'Ob_p', 'Om_m', 'Om_p', 
    #        'fiducial', 'fiducial_NCV', 'h_m', 'h_p', 'ns_m', 'ns_p', 's8_m', 's8_p']: 
    #    _hdf5_quijote(sub)
    
    # check along parameter axis
    #for par in ['Mnu', 'Ob', 'Om', 'h', 'ns', 's8']: 
    #    quijote_comparison(par, krange=[0.01, 0.5])
    
    # covariance matrix 
    quijote_covariance(krange=[0.01, 0.5])
