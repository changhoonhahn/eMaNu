import numpy as np 
from emanu import util as UT
from emanu import obvs as Obvs

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
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


def hubble2018(): 
    ''' Figure for Hubble proposal 
    '''
    fig = plt.figure(figsize=(15,6))
    bkgd = fig.add_subplot(111, frameon=False)
    
    # P(k) for neutrino mass 
    gs1 = mpl.gridspec.GridSpec(1,1, figure=fig) 
    gs1.update(left=0.02, right=0.25)#, wspace=0.05)
    sub = plt.subplot(gs1[0])
    _, plk_fid = Pellk(0, 0.0)
    for mneut in [0.06, 0.1, 0.15, 0.6]:
        k, plks = Pellk(0, mneut)
        if mneut == 0.06: 
            sub.plot(k, (np.average(plks, axis=0)-np.average(plk_fid, axis=0))/np.average(plk_fid, axis=0), 
                    lw=2, label=r'$\sum m_\nu ='+str(mneut)+'$eV')
        else: 
            sub.plot(k, (np.average(plks, axis=0)-np.average(plk_fid, axis=0))/np.average(plk_fid, axis=0), 
                    lw=2, label=str(mneut)+'eV')
    sub.plot([1e-2, 1.], [0., 0.], c='k', ls='--', lw=2)
    sub.set_xlabel('k [$h$/Mpc]', fontsize=25)
    sub.set_xlim([0.01, 1.])
    sub.set_xscale('log')
    sub.set_ylabel(r'$\Delta P/P(k)$', fontsize=25)
    sub.set_ylim([-0.025, 0.5])
    sub.legend(loc='upper right', handletextpad=0.2, fontsize=20) 
    # 3PCF 
    gs2 = mpl.gridspec.GridSpec(2,4, figure=fig) 
    gs2.update(left=0.32, right=0.95, hspace=0.1, wspace=0.1)
    mu_tpcf0 = mu_3PCF(0.0)
    for i_m, mn in enumerate([0.06, 0.1, 0.15, 0.6]): 
        mu_tpcf = mu_3PCF(mn)

        for i_l, ell in enumerate([0,2]):
            sub = plt.subplot(gs2[i_l,i_m]) 
            vlim = [-5e5, 5e5]
            cont = sub.pcolormesh(np.linspace(0., 200., 21), 
                                  np.linspace(0., 200., 21), 
                                  mu_tpcf[ell] - mu_tpcf0[ell], 
                                  cmap='RdBu_r', vmin=vlim[0], vmax=vlim[1])
            if ell == 0:
                if i_m == 0:
                    sub.set_title(str(mn)+'eV', fontsize=20)
                else: 
                    sub.set_title(str(mn)+'eV', fontsize=20)
                sub.set_xticklabels([]) 
            if i_m > 0: 
                sub.set_yticklabels([]) 
                if i_l > 0: sub.set_xticklabels(['', '100', '200']) 
            else: 
                sub.text(0.05, 0.05, r'$\ell='+str(ell)+'$', transform=sub.transAxes, fontsize=20)
    fig.text(0.3425, 0.8925, r'$\sum m_\nu$=', fontsize=17, ha='center') 
    fig.text(0.6, 0.0, r'$r_1$ [Mpc/$h$]', fontsize=25, ha='center')
    fig.text(0.2675, 0.5, r'$r_2$ [Mpc/$h$]', fontsize=25, va='center', rotation='vertical')
    fig.text(1., 0.5, r"$\Delta \zeta_\ell(r_1, r_2)$", fontsize=25, va='center', rotation=270)#, rotation='vertical')
    
    cbar_ax = fig.add_axes([0.955, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(cont, format=ticker.FuncFormatter(fmt), extend='both', cax=cbar_ax) # format='%.e', 
    #cbar.set_label(r"$\Delta \zeta_\ell(r_1, r_2)$", size=20)
    cbar.ax.tick_params(labelsize=10)
    fig.savefig('hubble_3pcf.png', bbox_inches='tight') 
    plt.close() 
    return None 


def Pellk(ell, mneut, nzbin=4, zspace=False): 
    pks = [] 
    for i in range(1,101): 
        plk_i = Obvs.Plk_halo(mneut, i, nzbin, zspace=zspace)
        pks.append(plk_i['p'+str(ell)+'k']) 
    return plk_i['k'], np.array(pks)


def mu_3PCF(mneut, nreal=100, ell=10, i_dr=[0,1], nside=20, nbin=20): 
    nzbin = 4
    zspace = False

    tpcfs = [] 
    for ireal in range(1,nreal+1): 
        tpcf_i = Obvs.threePCF_halo(mneut, ireal, nzbin, zspace=zspace, i_dr=i_dr, nside=nside, nbin=nbin)
        for ell in range(ell+1): 
            if ell == 0: 
                tpcf_mat = np.zeros((11, tpcf_i[0].shape[0], tpcf_i[0].shape[1]))
                tpcf_mat[0] = tpcf_i[0]
            else: 
                tpcf_mat[ell] = tpcf_i[ell]

        tpcfs.append(tpcf_mat) 
    tpcfs = np.array(tpcfs)
    mu_tpcfs = np.average(tpcfs, axis=0)
    return mu_tpcfs 


def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    a = a.split('.')[0]
    b = int(b)
    if b == 0: 
        return r'${}$'.format(a)
    else: 
        return r'${}\times10^{{{}}}$'.format(a, b)


# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


if __name__=='__main__': 
    hubble2018()
