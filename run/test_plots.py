import numpy as np 
import matplotlib.pyplot as plt 

from emanu import util as UT
from emanu.plots import plotBk


if __name__=="__main__": 
    bkfile = ''.join([UT.dat_dir(), 'bispectrum/',
        'groups.0.0eV.1.nzbin4.rspace.Ngrid360.Nmax40.Ncut3.step3.pyfftw.dat']) 
    i, j, l, b123, q123 = np.loadtxt(bkfile, unpack=True, skiprows=1, usecols=[0,1,2,3,4]) 
    fig = plt.figure()
    sub = fig.add_subplot(111)
    plotBk('shape', i, j, l, q123, nbin=50, ax=sub) 
    fig.savefig('shape.png', bbox_inches='tight') 

    fig = plt.figure(figsize=(15,5))
    fig = plotBk('amplitude', i, j, l, q123, nbin=20, fig=fig) 
    fig.savefig('amp.png', bbox_inches='tight')
