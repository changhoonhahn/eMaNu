# This script is used to submit jobs to measure B of CDM+B
import numpy as np
import sys,os

################################## INPUT #############################################
qos        = 'short'
snapnum    = 4    #4(z=0), 3(z=0.5), 2(z=1), 1(z=2), 0(z=3)
reals      = 'all'    # realization indices 'all' or list specifying indices

#thetas = ['Om_p', 'Ob2_p', 'h_p', 'ns_p', 's8_p', 'Om_m',  'Ob2_m', 'h_m', 'ns_m', 's8_m', 'Mnu_p', 'Mnu_pp', 'Mnu_ppp']
thetas = ['fiducial']
######################################################################################

if qos == 'vvshort': #number of realizations each cpu will do
    time = 1
elif qos == 'vshort': #number of realizations each cpu will do
    time = 6
elif qos == 'short': 
    time = 24
elif qos == 'medium': 
    time = 72 
elif qos == 'long': 
    time = 144
step = int(time * 50.) 

# do a loop over the different cosmologies
for theta in thetas: 
    folder = theta 
    
    if theta == 'fiducial':            
        reals = range(15000) 
    else:                           
        reals = range(500) 

    nreal = len(reals)
    nodes = int(np.ceil(nreal/step)) 
    
    for i in range(nodes): # loop over the different realizations
        i_first = step * i 
        i_last  = np.clip(step * (i+1) - 1, None, nreal-1)
        first = reals[i_first] 
        last  = reals[i_last] + 1
        print('%s reals:%i-%i' % (theta, first, last)) 

        a = '\n'.join(["#!/bin/bash", 
            "#SBATCH -J HOD",
            "#SBATCH --exclusive",
            "#SBATCH --nodes=2",
            "#SBATCH --ntasks-per-node=40",
            "#SBATCH --partition=general",
            "#SBATCH --time=%s:59:59" % str(time-1).zfill(2),
            "#SBATCH --export=ALL",
            "#SBATCH --output=_Bcb_%s_%i.o" % (theta, i),
            "#SBATCH --mail-type=all",
            "#SBATCH --mail-user=changhoon.hahn@princeton.edu",
            "", 
            "module load anaconda3", 
            "source activate emanu", 
            "",
            "srun -n 10 --mpi=pmi2 python3 create_Bcb.py %d %d %s %d" % (first, last, folder, snapnum),
            ""]) 
        # create the script.sh file, execute it and remove it
        f = open('script.slurm','w')
        f.write(a)
        f.close()
        os.system('sbatch script.slurm')
        os.system('rm script.slurm')
