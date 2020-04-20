# This script is used to submit jobs to calculate P for HOD catalogs
import numpy as np
import sys,os

# logM0 not included right now  
#thetas = ['Om_p', 'Ob2_p', 'h_p', 'ns_p', 's8_p', 'Om_m',  'Ob2_m', 'h_m', 'ns_m', 's8_m', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp', 
#        'alpha_m', 'logM0_m', 'logM1_m', 'logMmin_m', 'sigma_logM_m',
#        'alpha_p', 'logM0_p', 'logM1_p', 'logMmin_p', 'sigma_logM_p', 
#        'fiducial', 'fiducial_ZA']
#_thetas = ['Om_p', 'Ob2_p', 'h_p', 'ns_p', 's8_p', 'Om_m',  'Ob2_m', 'h_m', 'ns_m', 's8_m', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp', 
#        'fiducial_alpha=0.9', 'fiducial_logM0=13.8', 'fiducial_logM1=13.8', 'fiducial_logMmin=13.60', 'fiducial_sigma_logM=0.18',
#        'fiducial_alpha=1.3', 'fiducial_logM0=14.2', 'fiducial_logM1=14.2', 'fiducial_logMmin=13.70', 'fiducial_sigma_logM=0.22', 
#        'fiducial', 'fiducial_ZA']
################################## INPUT #############################################
#Om_p -- 4 realizations missing
#Ob2_p -- 3 realizations missing
#h_p -- 3 realizations missing
#ns_p -- 5 realizations missing
#s8_p -- 7 realizations missing
#Om_m -- 50 realizations missing
#Ob2_m -- 20 realizations missing
#h_m -- 20 realizations missing
#ns_m -- 20 realizations missing
#s8_m -- 20 realizations missing
#Mnu_p -- 20 realizations missing
#Mnu_pp -- 20 realizations missing
#Mnu_ppp -- 60 realizations missing

#thetas = ['Om_p', 'Ob2_p', 'h_p', 'ns_p', 's8_p', 'Om_m', 'Ob2_m', 'h_m', 'ns_m', 's8_m', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp']
#_thetas = ['Om_p', 'Ob2_p', 'h_p', 'ns_p', 's8_p', 'Om_m', 'Ob2_m', 'h_m', 'ns_m', 's8_m', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp'] 
#thetas = ['fiducial_ZA']
#_thetas = ['fiducial_ZA']
thetas = ['fiducial']
_thetas = ['fiducial']
offset     = 0    #the count will start from offset
snapnum    = 4    #4(z=0), 3(z=0.5), 2(z=1), 1(z=2), 0(z=3)
qos        = 'vvshort' 
######################################################################################
dir_quij = '/projects/QUIJOTE/Galaxies/'

if qos == 'vshort': #number of realizations each cpu will do
    step = 12
    time = 6 
elif qos == 'vvshort': #number of realizations each cpu will do
    step = 2
    time = 1 
elif qos == 'short': 
    step = 40  
    time = 24
elif qos == 'medium': 
    step = 120
    time = 72 
elif qos == 'long': 
    step = 240
    time = 144


# do a loop over the different cosmologies
for i_t, theta in enumerate(thetas): 
    
    folder = theta 
    if theta in ['logMmin_m', 'logMmin_p', 'sigma_logM_m', 'sigma_logM_p', 'logM0_m', 'logM0_p', 'alpha_m', 'alpha_p', 'logM1_m', 'logM1_p']: 
        folder = 'fiducial'

    nreal = 500
    if theta == 'fiducial': 
        nreal = 15000 
    
    missing = [] 
    for i in range(offset,nreal): 
        fbk = os.path.join(dir_quij, _thetas[i_t], str(i), 'Bk_RS2_GC_0_z=0.txt')
        if not os.path.isfile(fbk): 
            missing.append(i)
    missing = np.sort(missing)
    n_missing = len(missing) 
    print('%s missing %i B(k)s' % (theta, n_missing))
    print(missing)

    if n_missing == 0: 
        continue 

    nodes = int(np.ceil(n_missing/step)) 

    logMmin    = 13.65
    sigma_logM = 0.20
    logM0      = 14.0
    alpha      = 1.1
    logM1      = 14.0      
    if theta == 'logMmin_m': 
        logMmin = 13.6
    elif theta == 'logMmin_p': 
        logMmin = 13.7
    elif theta == 'sigma_logM_m': 
        sigma_logM = 0.18
    elif theta == 'sigma_logM_p': 
        sigma_logM = 0.22
    elif theta == 'logM0_m': 
        logM0 = 13.8
    elif theta == 'logM0_p': 
        logM0 = 14.2
    elif theta == 'alpha_m': 
        alpha = 0.9
    elif theta == 'alpha_p': 
        alpha = 1.3
    elif theta == 'logM1_m': 
        logM1 = 13.8
    elif theta == 'logM1_p': 
        logM1 = 14.2
    
    i0 = 0
    for i in range(nodes): # loop over the different realizations
        i_first = step * i 
        i_last  = np.clip(step * (i+1) - 1, None, n_missing-1)
        first = missing[i_first] 
        last  = missing[i_last] + 1
        print(first, last) 
        
        a = '\n'.join(["#!/bin/bash", 
            "#SBATCH -J B_%s%i" % (theta, i),
            "#SBATCH --exclusive",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks-per-node=40",
            "#SBATCH --partition=general",
	    "#SBATCH --time=%s:00:00" % (str(time).zfill(2)),
            "#SBATCH --export=ALL",
            "#SBATCH --output=_bk_%s%i.o" % (theta, i),
            "#SBATCH --mail-type=all",
            "#SBATCH --mail-user=changhoonhahn@lbl.gov",
            "", 
            "module load anaconda3", 
            "conda activate emanu", 
            "",
            "srun -n 1 --mpi=pmi2 python3 create_hodbk.py %d %d %s %d %s %s %s %s %s" % (first, last, folder, snapnum, logMmin, sigma_logM, logM0, alpha, logM1),
            ""]) 
        i0 += step 
        # create the script.sh file, execute it and remove it
        f = open('script.slurm','w')
        f.write(a)
        f.close()
        os.system('sbatch script.slurm')
        os.system('rm script.slurm')
