# This script is used to submit jobs to calculate P and B for HOD catalogs
import numpy as np
import sys,os

################################## INPUT #############################################
step       = 80  #number of realizations each cpu will do
offset     = 0    #the count will start from offset
snapnum    = 4    #4(z=0), 3(z=0.5), 2(z=1), 1(z=2), 0(z=3)
PorB       = 'pk'
######################################################################################

#theta_cosmo = ['Om_p', 'Ob2_p', 'h_p', 'ns_p', 's8_p', 'Om_m',  'Ob2_m', 'h_m', 'ns_m', 's8_m', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp']
#theta_hod = ['logMmin_m', 'logMmin_p', 'sigma_logM_m', 'sigma_logM_p', 'logM0_m', 'logM0_p', 'alpha_m', 'alpha_p', 'logM1_m', 'logM1_p', 'fiducial', 'fiducial_ZA']

#thetas =  theta_cosmo+theta_hod 
#thetas = ['testing'] 
thetas = ['logM0_m', 'logM0_p']

# do a loop over the different cosmologies
for theta in thetas: 

    folder = theta 
    if theta == 'testing': 
        nodes   = 1 
        folder  = 'Om_p'
        step    = 1
    elif theta == 'fiducial':            
        nodes = 15000/step
    elif theta =='latin_hypercube':    
        nodes = 2000/step
    else:                           
        nodes = 500/step
        if theta in ['logMmin_m', 'logMmin_p', 'sigma_logM_m', 'sigma_logM_p', 'logM0_m', 
                'logM0_p', 'alpha_m', 'alpha_p', 'logM1_m', 'logM1_p']: 
            folder = 'fiducial'
    
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

    for i in range(int(nodes)): # loop over the different realizations
        a = '\n'.join(["#!/bin/bash", 
            "#SBATCH -J P_%s%i" % (theta, i),
            "#SBATCH --exclusive",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks-per-node=40",
            "#SBATCH --partition=general",
	    "#SBATCH --time=06:00:00",
            "#SBATCH --export=ALL",
            "#SBATCH --output=_%s_%s%i.o" % (PorB, theta, i),
            "#SBATCH --mail-type=all",
            "#SBATCH --mail-user=changhoonhahn@lbl.gov",
            "", 
            "module load anaconda3", 
            "source activate emanu", 
            "",
            "srun -n 1 --mpi=pmi2 python3 create_hod%s.py %d %d %s %d %s %s %s %s %s" % (PorB, i*step+offset, (i+1)*step+offset, folder, snapnum, logMmin, sigma_logM, logM0, alpha, logM1),
            ""]) 

        # create the script.sh file, execute it and remove it
        f = open('script.slurm','w')
        f.write(a)
        f.close()
        os.system('sbatch script.slurm')
        os.system('rm script.slurm')
