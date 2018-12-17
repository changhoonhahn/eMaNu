'''

some convenience functions for plotting observables  


'''
import numpy as np 
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
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


def plotBk_shape(k1, k2, k3, BorQ, nbin=20, fig=None, ax=None, cmap='viridis'): 
    ''' Make triangle plot that illustrates the shape of the bispectrum
    '''
    if fig is None and ax is None: 
        fig = plt.figure()
        sub = fig.add_subplot(111)
    if ax is not None: 
        sub = ax 
    
    # k1 >= k2 >= k3 
    _k1, _k2, _k3 = [], [], [] 
    for k1i, k2i, k3i in zip(k1, k2, k3): 
        _k1.append(np.sort([k1i, k2i, k3i])[2]) 
        _k2.append(np.sort([k1i, k2i, k3i])[1]) 
        _k3.append(np.sort([k1i, k2i, k3i])[0]) 

    k3k1 = np.array(_k3)/np.array(_k1)
    k2k1 = np.array(_k2)/np.array(_k1)

    x_bins = np.linspace(0., 1., int(nbin)+1)
    y_bins = np.linspace(0.5, 1., int(0.5*nbin)+1)
    BorQ_grid = np.zeros((len(x_bins)-1, len(y_bins)-1))
    for i_x in range(len(x_bins)-1): 
        for i_y in range(len(y_bins)-1): 
            lim = ((k2k1 >= y_bins[i_y]) & 
                    (k2k1 < y_bins[i_y+1]) & 
                    (k3k1 >= x_bins[i_x]) & 
                    (k3k1 < x_bins[i_x+1]))
            if np.sum(lim) > 0: 
                BorQ_grid[i_x, i_y] = np.sum(BorQ[lim])/float(np.sum(lim))
            else: 
                BorQ_grid[i_x, i_y] = -np.inf
    BorQ_grid /= BorQ_grid.max() 
    bplot = plt.pcolormesh(x_bins, y_bins, BorQ_grid.T, vmin=0., vmax=1., cmap=cmap)
    cbar = plt.colorbar(bplot, orientation='vertical') 

    sub.set_xlabel(r'$k_3/k_1$', fontsize=25)
    sub.set_xlim([0.0, 1.0]) 
    sub.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0]) 
    sub.set_ylabel(r'$k_2/k_1$', fontsize=25)
    sub.set_ylim([0.5, 1.0]) 
    sub.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.]) 
    if ax is not None: 
        return sub 
    else: 
        return fig
