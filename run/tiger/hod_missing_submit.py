# This script is used to submit jobs to calculate missing HOD catalogs, P and B that fell thrugh the cracks during the original run
import numpy as np
import sys,os

################################## INPUT #############################################
thetas = ['Om_p', 'Ob2_p', 'h_p', 'ns_p', 's8_p', 'Om_m',  'Ob2_m', 'h_m', 'ns_m', 's8_m', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp', 
        'alpha_m', 'logM0_m', 'logM1_m', 'logMmin_m', 'sigma_logM_m',
        'alpha_p', 'logM0_p', 'logM1_p', 'logMmin_p', 'sigma_logM_p', 
        'fiducial_ZA']# fiducial

job        = 'catalog,P,B'
snapnum     = 4    #4(z=0), 3(z=0.5), 2(z=1), 1(z=2), 0(z=3)
qos         = 'vshort' 
reals       = range(100,500) # realizations 
seeds       = range(1,5) # HOD seeds
######################################################################################
dir_quij = '/projects/QUIJOTE/Galaxies/'

if qos == 'vvshort': #number of realizations each cpu will do
    step = 1
    time = 1
elif qos == 'vshort': #number of realizations each cpu will do
    step = 10
    time = 6
elif qos == 'short': 
    step = 40  
    time = 24
elif qos == 'medium': 
    step = 120
    time = 72 
elif qos == 'long': 
    step = 240
    time = 144

# loop over HOD seed 
for seed in seeds: 
    # do a loop over the different cosmologies
    for i_t, theta in enumerate(thetas): 
        
        folder = theta 
        if theta in ['logMmin_m', 'logMmin_p', 'sigma_logM_m', 'sigma_logM_p', 'logM0_m', 'logM0_p', 'alpha_m', 'alpha_p', 'logM1_m', 'logM1_p']: 
            folder = 'fiducial'
            _theta = {
                    'alpha_m': 'fiducial_alpha=0.9', 
                    'alpha_p': 'fiducial_alpha=1.3', 
                    'logM0_m': 'fiducial_logM0=13.8', 
                    'logM0_p': 'fiducial_logM0=14.2', 
                    'logM1_m': 'fiducial_logM1=13.8', 
                    'logM1_p': 'fiducial_logM1=14.2',
                    'logMmin_m': 'fiducial_logMmin=13.60', 
                    'logMmin_p': 'fiducial_logMmin=13.70',
                    'sigma_logM_m': 'fiducial_sigma_logM=0.18',
                    'sigma_logM_p': 'fiducial_sigma_logM=0.22'
                    }[theta]
        else: 
            _theta = theta 


        if isinstance(reals, str) and reals == 'all': 
            if theta == 'fiducial':            
                reals = range(15000) 
            elif theta =='latin_hypercube':    
                reals = range(2000) 
            else:                           
                reals = range(500) 
        
        nreal = len(reals)

        # check which are missing  
        missing = [] 
        for i in reals: 
            fbk0 = os.path.join(dir_quij, _theta, str(i), 
                    'Bk_RS0_GC_%i_z=0.txt' % seed)
            fbk1 = os.path.join(dir_quij, _theta, str(i), 
                    'Bk_RS1_GC_%i_z=0.txt' % seed)
            fbk2 = os.path.join(dir_quij, _theta, str(i), 
                    'Bk_RS2_GC_%i_z=0.txt' % seed)
            if not np.all([os.path.isfile(fbk0), os.path.isfile(fbk1), os.path.isfile(fbk2)]):
                missing.append(i)
        missing = np.sort(missing)
        n_missing = len(missing) 

        if n_missing == 0: 
            continue 
        else: 
            print('%s seed %i missing %i B(k)s' % (theta, seed, n_missing))
            #print(missing)

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
            print('job:%s %s reals:%i-%i seed:%i' % (job, theta, first, last, seed)) 
            
            a = '\n'.join(["#!/bin/bash", 
                "#SBATCH -J B_%s%i" % (theta, i),
                "#SBATCH --exclusive",
                "#SBATCH --nodes=1",
                "#SBATCH --ntasks-per-node=40",
                "#SBATCH --partition=general",
                "#SBATCH --time=%s:59:59" % (str(time-1).zfill(2)),
                "#SBATCH --export=ALL",
                "#SBATCH --output=_%s_%i_%i.o" % (theta, seed, i),
                "#SBATCH --mail-type=all",
                "#SBATCH --mail-user=changhoonhahn@lbl.gov",
                "", 
                "module load anaconda3", 
                "conda activate emanu", 
                "",
                "srun -n 1 --mpi=pmi2 python3 create_hod.py %s %d %d %s %d %s %s %s %s %s %i" % (job, first, last, folder, snapnum, logMmin, sigma_logM, logM0, alpha, logM1, seed),
                ""]) 
            i0 += step 
            # create the script.sh file, execute it and remove it
            f = open('script.slurm','w')
            f.write(a)
            f.close()
            os.system('sbatch script.slurm')
            os.system('rm script.slurm')
