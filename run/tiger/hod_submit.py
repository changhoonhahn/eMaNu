# This script is used to submit jobs to construct HOD catalogs, measure P and B 
import numpy as np
import sys,os

################################## INPUT #############################################
job        = 'catalog,P,B'
qos        = 'vshort'
snapnum    = 4    #4(z=0), 3(z=0.5), 2(z=1), 1(z=2), 0(z=3)
reals      = 'all'    # realization indices 'all' or list specifying indices
seeds      = range(5) # HOD seed 

theta_cosmo = ['Om_p', 'Ob2_p', 'h_p', 'ns_p', 's8_p', 'Om_m',  'Ob2_m', 'h_m', 'ns_m', 's8_m', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp']
#theta_hod = ['logMmin_m', 'logMmin_p', 'sigma_logM_m', 'sigma_logM_p', 'logM0_m', 'logM0_p', 'alpha_m', 'alpha_p', 'logM1_m', 'logM1_p', 'fiducial', 'fiducial_ZA']
theta_hod = ['logMmin_m', 'logMmin_p', 'sigma_logM_m', 'sigma_logM_p', 'logM0_m', 'logM0_p', 'alpha_m', 'alpha_p', 'logM1_m', 'logM1_p', 'fiducial_ZA']
thetas =  theta_cosmo+theta_hod 
#thetas = ['testing'] 
#thetas = ['logM0_m', 'logM0_p']
#thetas = ['Om_m']
thetas = ['sigma_logM_m_HR', 'sigma_logM_p_HR']
######################################################################################

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

# do a loop over the different cosmologies
for seed in seeds: 
    for theta in thetas: 

        folder = theta 
        if theta in ['logMmin_m', 'logMmin_p', 'sigma_logM_m', 'sigma_logM_p', 'logM0_m', 
                'logM0_p', 'alpha_m', 'alpha_p', 'logM1_m', 'logM1_p']: 
            folder = 'fiducial'
        if theta in ['sigma_logM_m_HR', 'sigma_logM_p_HR']: 
            folder = 'fiducial_HR'

        if isinstance(reals, str) and reals == 'all': 
            if theta == 'fiducial':            
                reals = range(15000) 
            elif theta =='latin_hypercube':    
                reals = range(2000) 
            elif 'HR' in theta: 
                # high resolution 
                reals = range(100) 
            else:                           
                reals = range(500) 
        
        nreal = len(reals)
        nodes = int(np.ceil(nreal/step)) 
        
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
        # high resolution tests 
        elif theta == 'sigma_logM_m_HR': 
            sigma_logM = 0.53
        elif theta == 'sigma_logM_p_HR': 
            sigma_logM = 0.57

        for i in range(nodes): # loop over the different realizations
            i_first = step * i 
            i_last  = np.clip(step * (i+1) - 1, None, nreal-1)
            first = reals[i_first] 
            last  = reals[i_last] + 1
            print('job:%s %s reals:%i-%i seed:%i' % (job, theta, first, last, seed)) 

            a = '\n'.join(["#!/bin/bash", 
                "#SBATCH -J HOD",
                "#SBATCH --exclusive",
                "#SBATCH --nodes=1",
                "#SBATCH --ntasks-per-node=40",
                "#SBATCH --partition=general",
                "#SBATCH --time=%s:59:59" % str(time-1).zfill(2),
                "#SBATCH --export=ALL",
                "#SBATCH --output=_%s_%i_%i.o" % (theta, seed, i),
                "#SBATCH --mail-type=all",
                "#SBATCH --mail-user=changhoon.hahn@princeton.edu",
                "", 
                "module load anaconda3", 
                "source activate emanu", 
                "",
                "srun -n 1 --mpi=pmi2 python3 create_hod.py %s %d %d %s %d %s %s %s %s %s %i" % (job, first, last, folder, snapnum, logMmin, sigma_logM, logM0, alpha, logM1, seed),
                ""]) 
            # create the script.sh file, execute it and remove it
            f = open('script.slurm','w')
            f.write(a)
            f.close()
            os.system('sbatch script.slurm')
            os.system('rm script.slurm')
