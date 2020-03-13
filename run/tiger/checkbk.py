#!/bin/python
'''script to check bispectrum files 
'''
import sys,os

dir_quij = '/projects/QUIJOTE/Galaxies/'

thetas = ['Om_p', 'Ob2_p', 'h_p', 'ns_p', 's8_p', 'Om_m',  'Ob2_m', 'h_m', 'ns_m', 's8_m', 
        'Mnu_p', 'Mnu_pp', 'Mnu_ppp'] 
#        'fiducial_alpha=0.9', 'fiducial_logM1=13.8', 'fiducial_logMmin=13.60', 'fiducial_sigma_logM=0.18',
#        'fiducial_alpha=1.3', 'fiducial_logM1=14.2', 'fiducial_logMmin=13.70', 'fiducial_sigma_logM=0.22', 
#        'fiducial', 'fiducial_ZA']

for theta in thetas: 
    nreal = 500
    if theta == 'fiducial': 
        nreal = 15000 
    
    missing = [] 
    for i in range(nreal): 
        fbk = os.path.join(dir_quij, theta, str(i), 'Bk_RS2_GC_0_z=0.txt') 
        if not os.path.isfile(fbk): 
            missing.append(i)
    print('----------------------------------------') 
    print('%s -- %i realizations missing' % (theta, len(missing))) 
    print(missing)

