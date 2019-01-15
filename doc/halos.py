'''

figures looking at hades halo catalog with massive neutrinos


'''
import numpy as np 
# --- eMaNu --- 
from emanu import util as UT
from emanu import obvs as Obvs
from emanu import plots as ePlots
# --- plotting --- 
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


def compare_dB123(typ, nreals=range(1,71), krange=[0.03, 0.25], nbin=50, zspace=False): 
    ''' Make various bispectrum plots as a function of m_nu 
    '''
    str_rsd = ''
    if zspace: str_rsd = '_rsd'
    mnus = [0.06, 0.1, 0.15]
    sig8s = [0.822, 0.818, 0.807, 0.798]
    kmin, kmax = krange 

    _, _, _, B123_fid, _, _ = readB123(0.0, nreals, 4, BorQ='B', zspace=zspace)

    B123s, cnts = [], [] 
    for mnu in mnus: 
        _i, _j, _l, _B123, _cnts, kf = readB123(mnu, nreals, 4, BorQ='B', zspace=zspace)
        B123s.append(_B123) 
        cnts.append(_cnts)

    B123_s8s, cnt_s8s = [], [] 
    for sig8 in sig8s: 
        _i, _j, _l, _B123, _cnts, kf = readB123_sigma8(sig8, nreals, 4, BorQ='B', zspace=zspace)
        B123_s8s.append(_B123) 
        cnt_s8s.append(_cnts)
        
    i_k, j_k, l_k = _i, _j, _l
    k3k1 = l_k.astype(float)/i_k.astype(float)
    k2k1 = j_k.astype(float)/i_k.astype(float)
    klim = ((i_k*kf <= kmax) & (i_k*kf >= kmin) &
            (j_k*kf <= kmax) & (j_k*kf >= kmin) & 
            (l_k*kf <= kmax) & (l_k*kf >= kmin)) 

    if typ == 'shape':
        x_bins = np.linspace(0., 1., int(nbin)+1)
        y_bins = np.linspace(0.5, 1., int(0.5*nbin)+1)

        fig = plt.figure(figsize=(25,6))
        for i in range(len(mnus)):
            B123_i, cnts_i = B123s[i], cnts[i]
            dB123 = B123_i - B123_fid
            BQgrid = ePlots._BorQgrid(k3k1[klim], k2k1[klim], dB123[klim], cnts_i[klim], x_bins, y_bins)

            sub = fig.add_subplot(2,4,i+1)
            bplot = sub.pcolormesh(x_bins, y_bins, BQgrid.T,
                    norm=LogNorm(vmin=1e6, vmax=5e7), cmap='RdBu')
            sub.text(0.05, 0.05, str(mnus[i])+'eV', ha='left', va='bottom', 
                    transform=sub.transAxes, fontsize=20)
            if i > 0: 
                sub.set_xticklabels([]) 
                sub.set_yticklabels([]) 

        for i in range(len(sig8s)): 
            B123_i, cnts_i = B123_s8s[i], cnt_s8s[i]
            dB123 = B123_i - B123_fid
            BQgrid = ePlots._BorQgrid(k3k1[klim], k2k1[klim], dB123[klim], cnts_i[klim], x_bins, y_bins)
            sub = fig.add_subplot(2,4,i+5)
            bplot = sub.pcolormesh(x_bins, y_bins, BQgrid.T,
                    norm=LogNorm(vmin=1e6, vmax=5e7), cmap='RdBu')
            sub.text(0.05, 0.05, '0.0eV', ha='left', va='bottom', 
                    transform=sub.transAxes, fontsize=20)
            sub.text(0.975, 0.025, '$\sigma_8$='+str(round(sig8s[i],3)), ha='right', va='bottom', 
                    transform=sub.transAxes, fontsize=20)
            if i > 1: 
                sub.set_yticklabels([]) 

        bkgd = fig.add_subplot(111, frameon=False)
        bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        bkgd.set_xlabel('$k_3/k_1$', labelpad=10, fontsize=25)
        bkgd.set_ylabel('$k_2/k_1$', labelpad=5, fontsize=25)
        fig.subplots_adjust(wspace=0.15, hspace=0.2, right=0.935)
        cbar_ax = fig.add_axes([0.95, 0.15, 0.0125, 0.7])
        cbar = fig.colorbar(bplot, cax=cbar_ax)
        cbar.set_label('$\Delta B = B(k_1, k_2, k_3) - B^\mathrm{(fid)}$', rotation=90, fontsize=20)
        fig.savefig(''.join([UT.doc_dir(), 'figs/halodB123_shape', str_rsd, '.pdf']), bbox_inches='tight') 

    elif typ == 'amp':
        fig = plt.figure(figsize=(25,8))
        sub = fig.add_subplot(211)
        sub2 = fig.add_subplot(212)
        i_k = i_k[klim]
        j_k = j_k[klim]
        l_k = l_k[klim]
        ii = 0 
        for mnu, B123 in zip([0.0]+mnus, [B123_fid]+B123s):
            b123 = B123[klim] - B123_fid[klim]

            _b123 = [] 
            l_usort = np.sort(np.unique(l_k))
            for l in l_usort: 
                j_usort = np.sort(np.unique(j_k[l_k == l]))
                for j in j_usort: 
                    i_usort = np.sort(np.unique(i_k[(l_k == l) & (j_k == j)]))
                    for i in i_usort: 
                        _b123.append(b123[(i_k == i) & (j_k == j) & (l_k == l)])
            if mnu != 0.0: 
                sub.plot(range(np.sum(klim)), _b123, c='C'+str(ii), label=str(mnu)+'eV') 
            else: 
                sub2.plot(range(np.sum(klim)), _b123, c='C0') 
            ii += 1 
        
        sub.legend(loc='upper right', markerscale=4, handletextpad=0.25, fontsize=20) 
        sub.set_xlim([0, np.sum(klim)])
        sub.set_yscale('log') 
        sub.set_ylim([1e6, 5e8]) 

        for sig8, B123 in zip(sig8s, B123_s8s):
            b123 = B123[klim] - B123_fid[klim]
            _b123 = [] 
            l_usort = np.sort(np.unique(l_k))
            for l in l_usort: 
                j_usort = np.sort(np.unique(j_k[l_k == l]))
                for j in j_usort: 
                    i_usort = np.sort(np.unique(i_k[(l_k == l) & (j_k == j)]))
                    for i in i_usort: 
                        _b123.append(b123[(i_k == i) & (j_k == j) & (l_k == l)])
            if ii < 10: 
                sub2.plot(range(np.sum(klim)), _b123, c='C'+str(ii), label='$\sigma_8=$'+str(sig8)) 
            else: 
                sub2.plot(range(np.sum(klim)), _b123, c='C9', label='$\sigma_8=$'+str(sig8)) 
            ii += 2 
        sub2.legend(loc='upper right', markerscale=4, handletextpad=0.25, fontsize=20) 
        sub2.set_xlim([0, np.sum(klim)])
        sub2.set_yscale('log') 
        sub2.set_ylim([1e6, 5e8]) 
        
        bkgd = fig.add_subplot(111, frameon=False)
        bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        bkgd.set_xlabel(r'$k_1 \le k_2 \le k_3$ triangle indices', labelpad=10, fontsize=25) 
        bkgd.set_ylabel('$B(k_1, k_2, k_3)$', fontsize=25) 
        fig.subplots_adjust(hspace=0.15)
        fig.savefig(''.join([UT.doc_dir(), 'figs/halodB123_amp', str_rsd, '.pdf']), bbox_inches='tight') 
        
        ii = 0 
        fig = plt.figure(figsize=(18,6))
        sub = fig.add_subplot(111)
        for mnu, B123 in zip([0.0]+mnus, [B123_fid]+B123s):
            b123 = B123[klim] - B123_fid[klim]

            _b123 = [] 
            l_usort = np.sort(np.unique(l_k))
            for l in l_usort: 
                j_usort = np.sort(np.unique(j_k[l_k == l]))
                for j in j_usort: 
                    i_usort = np.sort(np.unique(i_k[(l_k == l) & (j_k == j)]))
                    for i in i_usort: 
                        _b123.append(b123[(i_k == i) & (j_k == j) & (l_k == l)])
            sub.plot(range(np.sum(klim)), _b123, c='C'+str(ii), lw=2, label=str(mnu)+'eV') 
            ii += 1 
        
        for sig8, B123 in zip(sig8s, B123_s8s):
            b123 = B123[klim] - B123_fid[klim]
            _b123 = [] 
            l_usort = np.sort(np.unique(l_k))
            for l in l_usort: 
                j_usort = np.sort(np.unique(j_k[l_k == l]))
                for j in j_usort: 
                    i_usort = np.sort(np.unique(i_k[(l_k == l) & (j_k == j)]))
                    for i in i_usort: 
                        _b123.append(b123[(i_k == i) & (j_k == j) & (l_k == l)])
            if ii < 10: 
                sub.plot(range(np.sum(klim)), _b123, c='C'+str(ii), lw=0.75, label='$\sigma_8=$'+str(sig8)) 
            else: 
                sub.plot(range(np.sum(klim)), _b123, c='C9', lw=0.75, label='$\sigma_8=$'+str(sig8)) 
            ii += 2 
        
        sub.legend(loc='upper right', ncol=2, markerscale=4, handletextpad=0.25, fontsize=20) 
        sub.set_xlim([0, np.sum(klim)])
        sub.set_yscale('log') 
        sub.set_ylim([1e6, 5e8]) 
        fig.savefig(''.join([UT.doc_dir(), 'figs/halodB123_amp_comp', str_rsd, '.pdf']), bbox_inches='tight') 

    elif typ == 'relative':
        fig = plt.figure(figsize=(18,6))
        sub = fig.add_subplot(111)
        i_k = i_k[klim]
        j_k = j_k[klim]
        l_k = l_k[klim]
        ii = 1 
        for mnu, B123 in zip(mnus, B123s):
            b123 = (B123[klim] - B123_fid[klim])/B123_fid[klim]

            _b123 = [] 
            l_usort = np.sort(np.unique(l_k))
            for l in l_usort: 
                j_usort = np.sort(np.unique(j_k[l_k == l]))
                for j in j_usort: 
                    i_usort = np.sort(np.unique(i_k[(l_k == l) & (j_k == j)]))
                    for i in i_usort: 
                        _b123.append(b123[(i_k == i) & (j_k == j) & (l_k == l)])
            sub.plot(range(np.sum(klim)), _b123, lw=2, c='C'+str(ii), label=str(mnu)+'eV') 
            ii += 1 
        
        for sig8, B123 in zip(sig8s, B123_s8s):
            b123 = (B123[klim] - B123_fid[klim])/B123_fid[klim]
            _b123 = [] 
            l_usort = np.sort(np.unique(l_k))
            for l in l_usort: 
                j_usort = np.sort(np.unique(j_k[l_k == l]))
                for j in j_usort: 
                    i_usort = np.sort(np.unique(i_k[(l_k == l) & (j_k == j)]))
                    for i in i_usort: 
                        _b123.append(b123[(i_k == i) & (j_k == j) & (l_k == l)])
            if ii < 10: 
                sub.plot(range(np.sum(klim)), _b123, lw='1', c='C'+str(ii), label='$\sigma_8=$'+str(sig8)) 
            else: 
                sub.plot(range(np.sum(klim)), _b123, lw='1', c='C9', label='$\sigma_8=$'+str(sig8)) 
            ii += 2 
        sub.plot([0, np.sum(klim)], [0., 0.], c='k', ls='--', lw=2)
        sub.legend(loc='upper right', ncol=3, markerscale=4, handletextpad=0.25, fontsize=20) 
        sub.set_xlim([0, np.sum(klim)])
        sub.set_ylim([-0.01, 0.2]) 
        sub.set_yticks([0., 0.05, 0.1, 0.15, 0.2]) 
        
        bkgd = fig.add_subplot(111, frameon=False)
        bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        bkgd.set_xlabel(r'$k_1 \le k_2 \le k_3$ triangle indices', labelpad=10, fontsize=25) 
        bkgd.set_ylabel('$(B(k_1, k_2, k_3) - B^\mathrm{(fid)})/B^\mathrm{(fid)}$', labelpad=15, fontsize=25) 
        fig.subplots_adjust(hspace=0.15)
        fig.savefig(''.join([UT.doc_dir(), 'figs/halodB123_relative', str_rsd, '.pdf']), bbox_inches='tight') 
    return None 


def compare_B123(typ, nreals=range(1,71), krange=[0.03, 0.25], nbin=50, zspace=False): 
    ''' Make various bispectrum plots as a function of m_nu 
    '''
    str_rsd = ''
    if zspace: str_rsd = '_rsd'
    mnus = [0.0, 0.06, 0.1, 0.15]
    sig8s = [0.822, 0.818, 0.807, 0.798]
    kmin, kmax = krange 

    B123s, cnts = [], [] 
    for mnu in mnus: 
        _i, _j, _l, _B123, _cnts, kf = readB123(mnu, nreals, 4, BorQ='B', zspace=zspace)
        B123s.append(_B123) 
        cnts.append(_cnts)

    B123_s8s, cnt_s8s = [], [] 
    for sig8 in sig8s: 
        _i, _j, _l, _B123, _cnts, kf = readB123_sigma8(sig8, nreals, 4, BorQ='B', zspace=zspace)
        B123_s8s.append(_B123) 
        cnt_s8s.append(_cnts)
        
    i_k, j_k, l_k = _i, _j, _l
    k3k1 = l_k.astype(float)/i_k.astype(float)
    k2k1 = j_k.astype(float)/i_k.astype(float)
    klim = ((i_k*kf <= kmax) & (i_k*kf >= kmin) &
            (j_k*kf <= kmax) & (j_k*kf >= kmin) & 
            (l_k*kf <= kmax) & (l_k*kf >= kmin)) 

    if typ == 'shape':
        x_bins = np.linspace(0., 1., int(nbin)+1)
        y_bins = np.linspace(0.5, 1., int(0.5*nbin)+1)

        fig = plt.figure(figsize=(25,6))
        for i in range(len(mnus)):
            B123_i, cnts_i = B123s[i], cnts[i]
            BQgrid = ePlots._BorQgrid(k3k1[klim], k2k1[klim], B123_i[klim], cnts_i[klim], x_bins, y_bins)

            sub = fig.add_subplot(2,4,i+1)
            bplot = sub.pcolormesh(x_bins, y_bins, BQgrid.T,
                    norm=LogNorm(vmin=5e7, vmax=1e9), cmap='RdBu')
            sub.text(0.05, 0.05, str(mnus[i])+'eV', ha='left', va='bottom', 
                    transform=sub.transAxes, fontsize=20)
            if i > 0: 
                sub.set_xticklabels([]) 
                sub.set_yticklabels([]) 

        for i in range(len(sig8s)): 
            B123_i, cnts_i = B123_s8s[i], cnt_s8s[i]
            BQgrid = ePlots._BorQgrid(k3k1[klim], k2k1[klim], B123_i[klim], cnts_i[klim], x_bins, y_bins)
            sub = fig.add_subplot(2,4,i+6)
            bplot = sub.pcolormesh(x_bins, y_bins, BQgrid.T,
                    norm=LogNorm(vmin=5e7, vmax=1e9), cmap='RdBu')
            sub.text(0.05, 0.05, '0.0eV', ha='left', va='bottom', 
                    transform=sub.transAxes, fontsize=20)
            sub.text(0.975, 0.025, '$\sigma_8$='+str(round(sig8s[i],3)), ha='right', va='bottom', 
                    transform=sub.transAxes, fontsize=20)
            if i > 1: 
                sub.set_yticklabels([]) 

        bkgd = fig.add_subplot(111, frameon=False)
        bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        bkgd.set_xlabel('$k_3/k_1$', labelpad=10, fontsize=25)
        bkgd.set_ylabel('$k_2/k_1$', labelpad=5, fontsize=25)
        fig.subplots_adjust(wspace=0.15, hspace=0.2, right=0.935)
        cbar_ax = fig.add_axes([0.95, 0.15, 0.0125, 0.7])
        cbar = fig.colorbar(bplot, cax=cbar_ax)
        cbar.set_label('$B(k_1, k_2, k_3)$', rotation=90, fontsize=20)
        #cbar_ax = fig.add_axes([0.95, 0.125, 0.0125, 0.35])
        #cbar = fig.colorbar(dbplot, cax=cbar_ax)
        #cbar.set_label('$B(k_1, k_2, k_3) - B^\mathrm{(fid)}$', rotation=90, fontsize=20)
        fig.savefig(''.join([UT.doc_dir(), 'figs/haloB123_shape', str_rsd, '.pdf']), bbox_inches='tight') 

    elif typ == 'amp':
        fig = plt.figure(figsize=(25,8))
        sub = fig.add_subplot(211)
        axins = inset_axes(sub, loc='upper center', width="40%", height="55%") 
        sub2 = fig.add_subplot(212)
        axins2 = inset_axes(sub2, loc='upper center', width="40%", height="55%") 
        i_k = i_k[klim]
        j_k = j_k[klim]
        l_k = l_k[klim]
        ii = 0 
        for mnu, B123 in zip(mnus, B123s):
            b123 = B123[klim]

            _b123 = [] 
            l_usort = np.sort(np.unique(l_k))
            for l in l_usort: 
                j_usort = np.sort(np.unique(j_k[l_k == l]))
                for j in j_usort: 
                    i_usort = np.sort(np.unique(i_k[(l_k == l) & (j_k == j)]))
                    for i in i_usort: 
                        _b123.append(b123[(i_k == i) & (j_k == j) & (l_k == l)])
            #sub.scatter(range(np.sum(klim)), _b123, s=5, c='C'+str(ii), label=str(mnu)+'eV') 
            sub.plot(range(np.sum(klim)), _b123, c='C'+str(ii), label=str(mnu)+'eV') 
            # inset axes....
            #axins.scatter(range(np.sum(klim)), _b123, s=5, c='C'+str(ii), label=str(mnu)+'eV') 
            axins.plot(range(np.sum(klim)), _b123, c='C'+str(ii), label=str(mnu)+'eV') 
            
            if mnu == 0.0: 
                #sub2.scatter(range(np.sum(klim)), _b123, s=5, c='C0') 
                sub2.plot(range(np.sum(klim)), _b123, c='C0') 
                #axins2.scatter(range(np.sum(klim)), _b123, s=5, c='C0') 
                axins2.plot(range(np.sum(klim)), _b123, c='C0') 
            ii += 1 

        axins.set_xlim(900, 1050)
        axins.set_yscale('log') 
        axins.set_ylim(7e7, 3e8) 
        axins.set_xticklabels('') 
        axins.yaxis.set_minor_formatter(NullFormatter())
        mark_inset(sub, axins, loc1=3, loc2=4, fc="none", ec="0.5")

        sub.legend(loc='upper right', markerscale=4, handletextpad=0.25, fontsize=20) 
        sub.set_xlim([0, np.sum(klim)])
        sub.set_yscale('log') 
        sub.set_ylim([5e7, 8e9]) 
        
        for sig8, B123 in zip(sig8s, B123_s8s):
            b123 = B123[klim]
            _b123 = [] 
            l_usort = np.sort(np.unique(l_k))
            for l in l_usort: 
                j_usort = np.sort(np.unique(j_k[l_k == l]))
                for j in j_usort: 
                    i_usort = np.sort(np.unique(i_k[(l_k == l) & (j_k == j)]))
                    for i in i_usort: 
                        _b123.append(b123[(i_k == i) & (j_k == j) & (l_k == l)])
            #sub2.scatter(range(np.sum(klim)), _b123, s=5, c='C'+str(ii), label='$\sigma_8=$'+str(sig8)) 
            #sub2.plot(range(np.sum(klim)), _b123, c='C'+str(ii), label='$\sigma_8=$'+str(sig8)) 
            # inset axes....
            if ii < 10: 
                sub2.plot(range(np.sum(klim)), _b123, c='C'+str(ii), label='$\sigma_8=$'+str(sig8)) 
                axins2.plot(range(np.sum(klim)), _b123, c='C'+str(ii)) 
            else: 
                sub2.plot(range(np.sum(klim)), _b123, c='C9', label='$\sigma_8=$'+str(sig8)) 
                axins2.plot(range(np.sum(klim)), _b123, c='C9') 
            ii += 2 
        sub2.legend(loc='upper right', markerscale=4, handletextpad=0.25, fontsize=20) 
        sub2.set_xlim([0, np.sum(klim)])
        sub2.set_yscale('log') 
        sub2.set_ylim([5e7, 8e9]) 
        
        axins2.set_xlim(900, 1050)
        axins2.set_yscale('log') 
        axins2.set_ylim(7e7, 3e8) 
        axins2.set_xticklabels('') 
        axins2.yaxis.set_minor_formatter(NullFormatter())
        mark_inset(sub2, axins2, loc1=3, loc2=4, fc="none", ec="0.5")
        
        bkgd = fig.add_subplot(111, frameon=False)
        bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        bkgd.set_xlabel(r'$k_1 \le k_2 \le k_3$ triangle indices', labelpad=10, fontsize=25) 
        bkgd.set_ylabel('$B(k_1, k_2, k_3)$', fontsize=25) 
        fig.subplots_adjust(hspace=0.15)
        fig.savefig(''.join([UT.doc_dir(), 'figs/haloB123_amp', str_rsd, '.pdf']), bbox_inches='tight') 
    return None 


def readB123(mneut, i, nzbin, BorQ='B', zspace=False):
    ''' read in bispectrum of massive neutrino halo catalogs
    using the function Obvs.B123_halo
    '''
    bk_kwargs = {
            'Lbox': 1000., 
            'zspace': zspace, 
            'mh_min': 3200., 
            'Ngrid': 360, 
            'Nmax': 40, 
            'Ncut': 3, 
            'step': 3}

    if isinstance(i, int):
        i_k, j_k, l_k, B123, Q123, cnts, k_f = Obvs.B123_halo(mneut, i, nzbin, **bk_kwargs)
    elif isinstance(i, (list, np.ndarray)):
        i_k, j_k, l_k, B123, Q123, cnts = [], [], [], [], [], []
        for ii, _i in enumerate(i):
            i_k_i, j_k_i, l_k_i, B123_i, Q123_i, cnts_i, k_f = Obvs.B123_halo(mneut, _i, nzbin, **bk_kwargs)
            i_k.append(i_k_i)
            j_k.append(j_k_i)
            l_k.append(l_k_i)
            B123.append(B123_i)
            Q123.append(Q123_i)
            cnts.append(cnts_i)
        i_k = np.average(i_k, axis=0)
        j_k = np.average(j_k, axis=0)
        l_k = np.average(l_k, axis=0)
        B123 = np.average(B123, axis=0)
        Q123 = np.average(Q123, axis=0)
        cnts = np.average(cnts, axis=0)

    if BorQ == 'B':
        return i_k, j_k, l_k, (2*np.pi)**6 * B123 / k_f**6, cnts, k_f
    elif BorQ == 'Q':
        return i_k, j_k, l_k, Q123, cnts, k_f 


def readB123_sigma8(sig8, i, nzbin, BorQ='B', zspace=False):
    ''' read in bispectrum of sigma8 varied m_nu = 0 halo catalogs
    using the function Obvs.B123_halo_sigma8
    '''
    bk_kwargs = {
            'Lbox': 1000., 
            'zspace': zspace, 
            'mh_min': 3200., 
            'Ngrid': 360, 
            'Nmax': 40, 
            'Ncut': 3, 
            'step': 3}

    if isinstance(i, int):
        i_k, j_k, l_k, B123, Q123, cnts, k_f = Obvs.B123_halo_sigma8(sig8, i, nzbin, **bk_kwargs)
    elif isinstance(i, (list, np.ndarray)):
        i_k, j_k, l_k, B123, Q123, cnts = [], [], [], [], [], []
        for ii, _i in enumerate(i):
            i_k_i, j_k_i, l_k_i, B123_i, Q123_i, cnts_i, k_f = Obvs.B123_halo_sigma8(sig8, _i, nzbin, **bk_kwargs)
            i_k.append(i_k_i)
            j_k.append(j_k_i)
            l_k.append(l_k_i)
            B123.append(B123_i)
            Q123.append(Q123_i)
            cnts.append(cnts_i)
        i_k = np.average(i_k, axis=0)
        j_k = np.average(j_k, axis=0)
        l_k = np.average(l_k, axis=0)
        B123 = np.average(B123, axis=0)
        Q123 = np.average(Q123, axis=0)
        cnts = np.average(cnts, axis=0)

    if BorQ == 'B':
        return i_k, j_k, l_k, (2*np.pi)**6 * B123 / k_f**6, cnts, k_f
    elif BorQ == 'Q':
        return i_k, j_k, l_k, Q123, cnts, k_f


if __name__=="__main__": 
    #compare_B123('shape', nreals=range(1,71), krange=[0.03, 0.5], nbin=31)
    #compare_dB123('shape', nreals=range(1,71), krange=[0.03, 0.5], nbin=31)
    #compare_B123('amp', nreals=range(1,71), krange=[0.03, 0.5], zspace=False)
    #compare_dB123('amp', nreals=range(1,71), krange=[0.03, 0.5], zspace=False)
    #compare_dB123('relative', nreals=range(1,71), krange=[0.03, 0.5], nbin=25)
    
    compare_B123('amp', nreals=range(1,2), krange=[0.03, 0.5], zspace=True)
    compare_dB123('amp', nreals=range(1,2), krange=[0.03, 0.5], zspace=True)
    compare_dB123('relative', nreals=range(1,2), krange=[0.03, 0.5], zspace=True)
