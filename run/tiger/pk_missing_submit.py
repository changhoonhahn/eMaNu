# This script is used to submit jobs to calculate P for HOD catalogs
import numpy as np
import sys,os

################################## INPUT #############################################
step       = 80  #number of realizations each cpu will do
offset     = 0    #the count will start from offset
snapnum    = 4    #4(z=0), 3(z=0.5), 2(z=1), 1(z=2), 0(z=3)
PorB       = 'pk'
seed       = 1
######################################################################################
dir_quij = '/projects/QUIJOTE/Galaxies/'

# logM0 not included right now  
thetas = ['Om_p', 'Ob2_p', 'h_p', 'ns_p', 's8_p', 'Om_m',  'Ob2_m', 'h_m', 'ns_m', 's8_m', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp', 
        'alpha_m', 'logM0_m', 'logM1_m', 'logMmin_m', 'sigma_logM_m',
        'alpha_p', 'logM0_p', 'logM1_p', 'logMmin_p', 'sigma_logM_p', 
        'fiducial_ZA']

_thetas = ['Om_p', 'Ob2_p', 'h_p', 'ns_p', 's8_p', 'Om_m',  'Ob2_m', 'h_m', 'ns_m', 's8_m', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp', 
        'fiducial_alpha=0.9', 'fiducial_logM0=13.8', 'fiducial_logM1=13.8', 'fiducial_logMmin=13.60', 'fiducial_sigma_logM=0.18',
        'fiducial_alpha=1.3', 'fiducial_logM0=14.2', 'fiducial_logM1=14.2', 'fiducial_logMmin=13.70', 'fiducial_sigma_logM=0.22', 
        'fiducial_ZA']


# do a loop over the different cosmologies
for i_t, theta in enumerate(thetas): 
    
    folder = theta 
    if theta in ['logMmin_m', 'logMmin_p', 'sigma_logM_m', 'sigma_logM_p', 'logM0_m', 'logM0_p', 'alpha_m', 'alpha_p', 'logM1_m', 'logM1_p']: 
        folder = 'fiducial'

    nreal = 500
    if theta == 'fiducial': 
        nreal = 15000 
    
    missing = [] 
    for i in range(nreal): 
        fbk = os.path.join(dir_quij, _thetas[i_t], str(i), 'Pk_RS0_GC_%i_z=0.txt' % seed) 
        if not os.path.isfile(fbk): 
            missing.append(i)
    n_missing = len(missing) 
    print('%s missing %i P(k)s' % (theta, n_missing))
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
        first   = missing[i_first] 
        last    = missing[i_last] + 1
        print(first, last) 
    
        _min, sec   = divmod(np.min([step, float(i_last - i_first + 1)]) * 3. * 90., 60) 
        hr, _min    = divmod(_min, 60) 
        if _min > 0. or sec > 0: 
            hr += 1
        assert hr <= 6

        a = '\n'.join(["#!/bin/bash", 
            "#SBATCH -J P_%s%i" % (theta, i),
            "#SBATCH --exclusive",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks-per-node=40",
            "#SBATCH --partition=general",
	    "#SBATCH --time=%s:59:59" % str(int(hr)-1).zfill(2),
            "#SBATCH --export=ALL",
            "#SBATCH --output=_%s_%s%i.o" % (PorB, theta, i),
            "#SBATCH --mail-type=all",
            "#SBATCH --mail-user=changhoonhahn@lbl.gov",
            "", 
            "module load anaconda3", 
            "conda activate emanu", 
            "",
            "srun -n 1 --mpi=pmi2 python3 create_hodpk.py %d %d %s %d %s %s %s %s %s" % (first, last, folder, snapnum, logMmin, sigma_logM, logM0, alpha, logM1),
            ""]) 
        i0 += step 
        # create the script.sh file, execute it and remove it
        f = open('script.slurm','w')
        f.write(a)
        f.close()
        os.system('sbatch script.slurm')
        os.system('rm script.slurm')
