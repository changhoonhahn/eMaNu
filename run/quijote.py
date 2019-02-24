'''

Scripts for dealing with the quijote simulation bispectrum
measurements. THis includes measuring the covariance matrix
and fisher matrix. 


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


def quijote_hdf5(subdir): 
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
    f.create_dataset('Nhalos', data=Nhalos) 
    f.close()
    return None 


def quijote_Cov_full(shotnoise=True): 
    ''' calculate the *full* covariance of the fiducial quijote bispectra

    :param shotnoise: (default: True)
        whether or not to include shotnoise in the covarinace. The correct
        thing to do is to include it. If True includes shotnoise 
    '''
    dir_bk = os.path.join(UT.dat_dir(), 'bispectrum') 
    fbks = h5py.File(os.path.join(dir_bk, 'quijote_fiducial.hdf5'), 'r') 
    if shotnoise: 
        bks = fbks['b123'].value + fbks['b_sn'].value
    else: 
        bks = fbks['b123'].value

    C_bk = np.cov(bks.T) 

    # write C_bk to hdf5 file 
    fcov = os.path.join(UT.dat_dir(), 'bispectrum', 'quijote_Cov_full%s.hdf5' % ['.sn_corr', ''][shotnoise])
    f = h5py.File(fcov, 'w') 
    f.create_dataset('C_bk', data=C_bk) 
    f.create_dataset('k1', data=fbks['k1'].value) 
    f.create_dataset('k2', data=fbks['k2'].value) 
    f.create_dataset('k3', data=fbks['k3'].value) 
    f.close()
    return None 


def quijote_covariance(krange=[0.01, 0.5], shotnoise=True):
    ''' calculate the covariance matrix using the 15,000 fiducial bispetra.

    :param krange: (default: [0.01, 0.5]) 
        k range of k1, k2, k3:
        
    :param shotnoise: (default: True)
        By default, shotnoise should be included in the covariance matrix. 
    '''
    kmin, kmax = krange  
    if shotnoise: 
        fcov = os.path.join(UT.dat_dir(), 'bispectrum', 'quijote_Cov_bk.sn.%.2f_%.2f.gmorder.hdf5' % (kmin, kmax))
    else: 
        fcov = os.path.join(UT.dat_dir(), 'bispectrum', 'quijote_Cov_bk.%.2f_%.2f.gmorder.hdf5' % (kmin, kmax))

    if os.path.isfile(fcov): 
        f = h5py.File(fcov, 'r') 
        C_bk = f['C_bk'].value # fiducial covariance matrix 
    else: 
        # read in fiducial bispectra
        bk = quijote_bk('fiducial', krange=krange, shotnoise=shotnoise)
        C_bk = np.cov(bk.T) 

        # write C_bk to hdf5 file 
        f = h5py.File(fcov, 'w') 
        f.create_dataset('C_bk', data=C_bk) 
        f.close()
    return C_bk  


def quijote_Cov_SNcomparison(krange=[0.01, 0.5]): 
    ''' Compare covariance with and without shotnoise. With shot noise 
    is the correct covariance matrix.
    '''
    C_sn = quijote_covariance(krange=krange, shotnoise=True)
    C_nosn = quijote_covariance(krange=krange, shotnoise=False)

    # plot the covariance matrices 
    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(121)
    cm = sub.pcolormesh(C_sn, norm=LogNorm(vmin=1e11, vmax=1e18))
    sub.set_title(r'with shotnoise (correct)', fontsize=20)
    
    sub = fig.add_subplot(122)
    cm = sub.pcolormesh(C_nosn, norm=LogNorm(vmin=1e11, vmax=1e18))
    sub.set_title(r'without shotnoise (incorrect)', fontsize=20)
    sub.set_yticklabels([])
    fig.subplots_adjust(wspace=0.1, right=0.925)
    cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
    fig.colorbar(cm, cax=cbar_ax)
    fig.savefig(os.path.join(UT.fig_dir(), 'quijote_Cov_SN.png'), bbox_inches='tight') 
    return None 


def qujjote_covariance_convergence(krange=[0.01, 0.5]): 
    ''' test the convergence of covariance matrix 
    '''
    kmin, kmax = krange  
    bk = quijote_bk('fiducial', krange=krange) # fiducial bispectra
    ijs = [] 
    for i in range(bk.shape[1])[::50]: 
        for j in range(i, bk.shape[1])[::50]: 
            ijs.append((i, j))
    print len(ijs)

    nmocks = np.linspace(1000, bk.shape[0], 15).astype(int)
    Ciis, Cijs = [], [] 
    for n_bk in nmocks: 
        C_bk = np.cov(bk[:n_bk,:].T) 
        Ciis.append(np.diag(C_bk)) 
        print C_bk[np.array(ijs)].shape
        Cijs.append(C_bk[np.array(ijs)])
    Ciis = np.array(Ciis)  
    Cijs = np.array(Cijs)
    print Cijs.shape

    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(121)
    for i in range(Ciis.shape[1])[10::100]: 
        sub.plot(nmocks, Ciis[:,i])
    for j in range(Cijs.shape[1]): 
        sub.plot(nmocks, Cijs[:,i]) 

    sub.set_xlabel(r"$N_{\rm mocks}$", fontsize=25) 
    sub.set_xlim([0, bk.shape[0]]) 
    sub.set_ylabel(r"$C_{i,i}$", fontsize=25) 
    sub.set_yscale('log') 
    fig.savefig(os.path.join(UT.fig_dir(), 'quijote_Covii_converge.png'), bbox_inches='tight') 
    return None


def quijote_bk(theta, krange=[0.01, 0.5], shotnoise=False): 
    ''' read in bispectra for specified theta 
    
    :param theta: 
        string that specifies which quijote run. `theta==fiducial`  
        for the fiducial set of parameters. 

    :param krange: (default: [0.01, 0.5]) 
        k range of k1, k2, and k3. 

    :param shotnoise: (default: False)
        If False, function returns shot noise *corrected* B(k) -- B(k)
        If True, function returns shot noise *uncorrected* B(k) -- \hat{B(k)}
        where \hat{B(k)} = B(k) + B_SN 
    '''
    kmin, kmax = krange 
    # read in fiducial biectra
    dir_bk = os.path.join(UT.dat_dir(), 'bispectrum') 
    bks = h5py.File(os.path.join(dir_bk, 'quijote_%s.hdf5' % theta), 'r') 
    i_k, j_k, l_k = bks['k1'].value, bks['k2'].value, bks['k3'].value 
    bk_sn = bks['b_sn'].value  # shot noise 

    if shotnoise: 
        bk = bks['b123'].value + bk_sn # shot noise uncorrected
    else: 
        bk = bks['b123'].value # shot noise corrected

    # k range limit of the bispectrum
    kf = 2.*np.pi/1000. # fundmaentla mode
    klim = ((i_k*kf <= kmax) & (i_k*kf >= kmin) &
            (j_k*kf <= kmax) & (j_k*kf >= kmin) & 
            (l_k*kf <= kmax) & (l_k*kf >= kmin)) 
    
    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim] 
    ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # order of triangles 
    _bk = bk[:,klim]
    return _bk[:,ijl.flatten()]

##################################################################
# qujiote fisher stuff
##################################################################
def quijote_Fisher(krange=[0.01, 0.5], deriv='p', shotnoise=True, validate=False): 
    ''' calculate fisher matrix for parameters ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
    '''
    kmin, kmax = krange
    thetas = ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
    theta_lbls = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_\nu$']

    C_fid = quijote_covariance(krange=krange, shotnoise=shotnoise) # fiducial covariance (with shotnoise is the correct one) 
    C_inv = np.linalg.inv(C_fid) # invert the covariance 

    dbk_dt = [] 
    for par in thetas:  # read in in dBk/dtheta_i 
        str_d = '' 
        if par == 'Mnu': str_d = '.'+deriv
        f = h5py.File(os.path.join(UT.dat_dir(), 'bispectrum', 
            'quijote_dbk_d%s%s.%.2f_%.2f.gmorder.hdf5' % (par, str_d, kmin, kmax)), 'r') 
        dbk_dt.append(f['dbk_dtheta'].value)
    
    Fij = np.zeros((len(thetas), len(thetas)))
    for i, par_i in enumerate(thetas): 
        for j, par_j in enumerate(thetas): 
            dbk_dtt_i = dbk_dt[i]
            dbk_dtt_j = dbk_dt[j]

            # calculate Mij 
            Mij = np.dot(dbk_dtt_i[:,None], dbk_dtt_j[None,:]) + np.dot(dbk_dtt_j[:,None], dbk_dtt_i[None,:])

            fij = 0.5 * np.trace(np.dot(C_inv, Mij))
            Fij[i,j] = fij 
    
    sn_str = ''
    if shotnoise: sn_str = '.sn'
    f_ij = os.path.join(UT.dat_dir(), 'bispectrum', 
            'quijote_Fisher%s.%s.%.2f_%.2f.gmorder.hdf5' % (sn_str, deriv, kmin, kmax))
    f = h5py.File(f_ij, 'w') 
    f.create_dataset('Fij', data=Fij) 
    f.create_dataset('C_fid', data=C_fid)
    f.create_dataset('C_inv', data=C_inv)
    f.close() 

    if validate: 
        fig = plt.figure(figsize=(6,5))
        sub = fig.add_subplot(111)

        cm = sub.pcolormesh(Fij, norm=SymLogNorm(vmin=-2e5, vmax=5e5, linthresh=1e2, linscale=1.))

        sub.set_xticks(np.arange(Fij.shape[0]) + 0.5, minor=False)
        sub.set_xticklabels(theta_lbls, minor=False)
        sub.set_yticks(np.arange(Fij.shape[1]) + 0.5, minor=False)
        sub.set_yticklabels(theta_lbls, minor=False)
        sub.set_title(r'Fisher Matrix $F_{i,j}^{(%s)}$' % deriv, fontsize=25)
        fig.colorbar(cm)
        fig.savefig(f_ij.replace('.hdf5', '.png'), bbox_inches='tight') 
    return None


def quijote_dBk(theta, krange=[0.01, 0.5], validate=False):
    ''' calculate d B(k)/d theta using the paired and fixed quijote simulations
    run on perturbed theta 

    :param theta: 
        string that specifies the parameter to take the 
        derivative by. 

    :param krange: (default: [0.01, 0.5]) 
        k-range of the bispectrum measurements. Note that the
        Gil-Marin triangle order will also be implemented. 

    :param validate: (default: False) 
        If True, the code will generate a plot to validate the 
        derivatives
    '''
    kmin, kmax = krange  
    # fiducial theta and covariance matrix 
    fid_dict = {'Mnu': 0., 'Ob': 0.049, 'Om': 0.3175, 'h': 0.6711,  'ns': 0.9624,  's8': 0.834}
    tt_fid = fid_dict[theta] 
    Bk_fid = np.average(quijote_bk('fiducial', krange=krange), axis=0) 

    # get all bispectrum covariances along theta  
    theta_dict = {
            'Mnu': [0.1, 0.2, 0.4], # +, ++, +++ 
            'Ob': [0.048, 0.050], # others are - + 
            'Om': [0.3075, 0.3275],
            'h': [0.6511, 0.6911],
            'ns': [0.9424, 0.9824],
            's8': [0.819, 0.849]} 

    if theta == 'Mnu': 
        tt_p, tt_pp, tt_ppp = theta_dict['Mnu'] 
        h_p = tt_p - tt_fid # step + 
        h_pp = tt_pp - tt_fid # step ++ 
        h_ppp = tt_ppp - tt_fid # step +++ 
        
        Bk_p    = np.average(quijote_bk(theta+'_p', krange=krange), axis=0) # Covariance matrix tt+ 
        Bk_pp   = np.average(quijote_bk(theta+'_pp', krange=krange), axis=0) # Covariance matrix tt++
        Bk_ppp  = np.average(quijote_bk(theta+'_ppp', krange=krange), axis=0) # Covariance matrix tt+++

        # take the derivatives 
        dBk_p = (Bk_p - Bk_fid) / h_p 
        dBk_pp = (Bk_pp - Bk_fid) / h_pp
        dBk_ppp = (Bk_ppp - Bk_fid) / h_ppp
        dBk_diff1 = (-21 * Bk_fid + 32 * Bk_p - 12 * Bk_pp + Bk_ppp)/(12*0.1) # finite difference coefficient
        dBk_diff2 = (-105 * Bk_fid + 160 * Bk_p - 60 * Bk_pp + 5 * Bk_ppp)/6. # finite difference coefficient

        for mpc, dBk in zip(['p', 'pp', 'ppp', 'fd1', 'fd2'], [dBk_p, dBk_pp, dBk_ppp, dBk_diff1, dBk_diff2]): 
            fdbk = os.path.join(UT.dat_dir(), 'bispectrum', 
                    'quijote_dbk_d%s.%s.%.2f_%.2f.gmorder.hdf5' % (theta, mpc, kmin, kmax))
            f = h5py.File(fdbk, 'w') 
            f.create_dataset('dbk_dtheta', data=dBk) 
            f.close() 

        if validate: 
            fig = plt.figure(figsize=(15,5))
            sub = fig.add_subplot(111)
            for mpc, dBk in zip(['+', '++', '+++'], [dBk_p, dBk_pp, dBk_ppp]): 
                sub.plot(range(len(Bk_p)), dBk, 
                        label=r'$(B^{(%s)} - B^{\rm fid})/(M_\nu^{(%s)} - M_\nu^{\rm fid})$' % (mpc, mpc))
            sub.plot(range(len(Bk_p)), dBk_diff1, label=r'${\rm d}B/{\rm d} M_\nu$ (finite diff. 1)')
            sub.plot(range(len(Bk_p)), dBk_diff2, label=r'${\rm d}B/{\rm d} M_\nu$ (finite diff. 2)')
            sub.legend(loc='upper right', fontsize=20 ) 
            sub.set_xlabel(r'$k_1 \le k_2 \le k_3$ triangle indices', fontsize=25) 
            sub.set_xlim([0., len(Bk_p)])  
            sub.set_yscale('log') 
            sub.set_ylim([1e6, 1e10]) 
            fig.savefig(os.path.join(UT.dat_dir(), 'bispectrum', 'quijote_dbk_d%s.png' % theta), 
                    bbox_inches='tight') 
    else: 
        tt_m, tt_p = theta_dict[theta]
        h_m = tt_fid - tt_m # step - 
        h_p = tt_p - tt_fid # step + 
        
        Bk_m = np.average(quijote_bk(theta+'_m', krange=krange), axis=0) # Covariance matrix tt- 
        Bk_p = np.average(quijote_bk(theta+'_p', krange=krange), axis=0) # Covariance matrix tt+ 
        
        # take the derivatives 
        dBk_m = (Bk_fid - Bk_m) / h_m 
        dBk_p = (Bk_p - Bk_fid) / h_p 
        dBk_c = (Bk_p - Bk_m) / (h_m + h_p) 
        
        for mpc, dBk in zip(['.m', '.p', ''], [dBk_m, dBk_p, dBk_c]): 
            fdbk = os.path.join(UT.dat_dir(), 'bispectrum', 
                    'quijote_dbk_d%s%s.%.2f_%.2f.gmorder.hdf5' % (theta, mpc, kmin, kmax))
            f = h5py.File(fdbk, 'w') 
            f.create_dataset('dbk_dtheta', data=dBk) 
            f.close() 

        if validate: 
            fig = plt.figure(figsize=(15,5))
            sub = fig.add_subplot(111)
            for ii, mpc, dBk in zip(range(3), ['-', '+'], [dBk_m, dBk_p]): 
                sub.plot(range(len(Bk_p)), dBk, label=r'${\rm d}B^{(%s)}/{\rm d}\theta$' % mpc)
            sub.plot(range(len(Bk_p)), dBk_c, c='k', ls='--', label=r'${\rm d}B/{\rm d}\theta$')
            sub.set_xlabel(r'$k_1 \le k_2 \le k_3$ triangle indices', fontsize=25) 
            sub.set_xlim([0., 500.]) #len(Bk_p)])  
            if theta in ['Ob']: 
                sub.set_yscale('log') 
                sub.set_ylim([1e7, 1e12]) 
                sub.legend(loc='upper right', fontsize=20 ) 
            elif theta in ['Om', 'h', 'ns', 's8']: 
                sub.set_yscale('symlog') 
                sub.set_ylim([-1e7, -1e12]) 
                sub.legend(loc='upper right', fontsize=20 ) 
            else: 
                sub.legend(loc='lower left', fontsize=20 ) 
            fig.savefig(os.path.join(UT.dat_dir(), 'bispectrum', 'quijote_dbk_d%s.png' % theta), 
                    bbox_inches='tight') 
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


if __name__=="__main__": 
    # write to hdf5 
    for sub in ['Mnu_pp', 'Mnu_ppp', 'Ob_m', 'Ob_p', 'Om_m', 'Om_p', 'h_m', 'h_p', 'ns_m', 'ns_p', 's8_m', 's8_p']: #['Mnu_p']:  
        quijote_hdf5(sub)
    #quijote_hdf5('fiducial') 
    #quijote_Cov_full(shotnoise=True)   
    #quijote_Cov_full(shotnoise=False)   

    # check along parameter axis
    #for par in ['Mnu', 'Ob', 'Om', 'h', 'ns', 's8']: 
    #    quijote_comparison(par, krange=[0.01, 0.5])
    
    # covariance matrix 
    #quijote_covariance(krange=[0.01, 0.5], shotnoise=True) # covariance matrix with shotnoise (i.e. correct covariance) 
    #quijote_Cov_SNcomparison(krange=[0.01, 0.5])
    #qujjote_covariance_convergence(krange=[0.01, 0.5])

    #for par in ['Mnu']:#, 'Ob', 'Om', 'h', 'ns', 's8']: 
    #    quijote_dBk(par, krange=[0.01, 0.5], validate=True)
    for kmax in [0.2, 0.3, 0.4, 0.5]: 
        for deriv in ['p', 'fd']: 
            pass
            #quijote_Fisher(krange=[0.01, kmax], deriv=deriv, shotnoise=True, validate=True)   
            #quijote_Fisher(krange=[0.01, kmax], deriv=deriv, shotnoise=False, validate=True)   
