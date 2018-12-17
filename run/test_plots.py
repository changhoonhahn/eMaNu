import numpy as np 
import matplotlib.pyplot as plt 

from emanu.plots import plotBk_shape


if __name__=="__main__": 
    bkfile = '/Users/ChangHoon/data/pyspectrum/pySpec.B123.BoxN1.mock.Ngrid360.Ngrid360.Nmax40.Ncut3.step3.pyfftw.dat'
    i, j, l, b123, q123 = np.loadtxt(bkfile, unpack=True, skiprows=1, usecols=[0,1,2,3,4]) 
    fig = plt.figure()
    sub = fig.add_subplot(111)
    plotBk_shape(i, j, l, q123, nbin=50, ax=sub) 
    fig.savefig('test.png', bbox_inches='tight') 
