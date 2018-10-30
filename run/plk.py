#!/bin/python
import os 
import sys 
import numpy as np 
# -- eMaNu -- 
from emanu import util as UT
from emanu import obvs as Obvs
from emanu import forwardmodel as FM
from emanu.hades import data as Dat


def hadesHalo_Plk(mneut, nreal, nzbin, zspace=False, Lbox=1000., Nmesh=360): 
    ''' Calculate the powerspectrum multipoles for Paco's Neutrio 
    halo catalogs

    notes
    -----
    * Takes ~20 seconds.
    '''
    # output file 
    fout = Obvs._obvs_fname('plk', mneut, nreal, nzbin, zspace, Nmesh=Nmesh) 

    # import Neutrino halo with mneut eV, realization # nreal, at z specified by nzbin 
    halos = Dat.NeutHalos(mneut, nreal, nzbin) 
    if zspace: 
        halos['RSDPosition'] = FM.RSD(halos, LOS=[0,0,1], Lbox=Lbox)
    
    # calculate P_l(k) 
    plk = FM.Observables(halos, observable='plk', rsd=zspace, dk=0.005, kmin=0.005, Nmesh=Nmesh)

    # header 
    hdr = ''.join(['P_l(k) measurements for m neutrino = ', str(mneut), ' eV, realization ', str(nreal), ', zbin ', str(nzbin), 
        '\n P_shotnoise ', str(plk['shotnoise']), 
        '\n cols: k, P_0, P_2, P_4']) 

    # write to file 
    np.savetxt(fout, np.array([plk['k'], plk['p0k'], plk['p2k'], plk['p4k']]).T, header=hdr) 
    return None 


if __name__=='__main__': 
    run = sys.argv[1]
    if run == 'halo': 
        # python plk.py halo mneut nreal nzbin zstr
        mneut = float(sys.argv[2])
        nreal = int(sys.argv[3]) 
        nzbin = int(sys.argv[4])
        zstr = sys.argv[5]
        nmesh = int(sys.argv[6])
        if zstr == 'z': zbool = True
        elif zstr == 'real': zbool = False
        hadesHalo_Plk(mneut, nreal, nzbin, zspace=zbool, Lbox=1000., Nmesh=nmesh)
    else: 
        raise ValueError

