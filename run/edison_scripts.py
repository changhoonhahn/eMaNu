'''

Code for generating and submitting jobs on edison 


'''
import os 
import numpy as np 
import subprocess


def haloPlk_mneut_job(mneut, nreals=[1,100]): 
    ''' calculate P_l(k) for halo catalog with m_nu = mneut (0.0, 0.06, 0.10, 0.15 eV) 
    '''
    # job name 
    fjob = ''.join(['scripts/', 
        'edison_haloplk_', 
        str(mneut), 'eV_', 
        str(nreals[0]), '_', str(nreals[1]), '.slurm']) 

    # each P_l(k) calculation takes ~1 min
    tmin = float(nreals[1] - nreals[0]+1)*1.2
    if tmin < 10.: 
        tmin = 10.
    m = str(int(10.*np.ceil((tmin % 60.)/10.))).zfill(2)
    if m == '60': 
        h = str(int(tmin // 60.)+1).zfill(2) 
        m = '00' 
    else: 
        h = str(int(tmin // 60.)).zfill(2) 
    if tmin <= 30.: 
        queue = 'debug'
    else:
        queue = 'regular'
    
    jname = 'plk_halo_'+str(mneut)+'eV_'+str(nreals[0])+'_'+str(nreals[1])
    ofile = ''.join(['/global/homes/c/chahah/projects/eMaNu/run/scripts/', 
        '_plk_halo_'+str(mneut)+'eV_'+str(nreals[0])+'_'+str(nreals[1]), '.o']) 

    jb = '\n'.join([ 
        '#!/bin/bash',
        '#SBATCH -q '+queue,
        '#SBATCH -N 1',
        '#SBATCH -t '+h+':'+m+':00', 
        '#SBATCH -J '+jname, 
        '#SBATCH -o '+ofile,  
        '', 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        '', 
        'module load python/2.7-anaconda', 
        'source activate nbdkt', 
        '',
        'mneut='+str(mneut), 
        'for ireal in {'+str(nreals[0])+'..'+str(nreals[1])+'}; do', 
        '\techo " --- mneut = "$mneut", nreal= "$ireal" --- "', 
        '\tsrun -n 1 -c 1 python /global/homes/c/chahah/projects/eMaNu/run/plk.py halo $mneut $ireal 4 "real" 360', 
        'done',
        '', 
        'now=$(date +"%T")', 
        'echo "end time ... $now"']) 
    job = open(fjob, 'w') 
    job.write(jb) 
    job.close() 
    return fjob 


def haloPlk_sigma8_job(sig8, nreals=[1,100]): 
    ''' calculate P_l(k) for halo catalog with m_nu = mneut (0.0, 0.06, 0.10, 0.15 eV) 
    '''
    # job name 
    fjob = ''.join(['scripts/', 
        'edison_haloplk_sig8', 
        str(sig8), '_', 
        str(nreals[0]), '_', str(nreals[1]), '.slurm']) 

    # each P_l(k) calculation takes ~1 min
    tmin = float(nreals[1] - nreals[0]+1)*1.2
    if tmin < 10.: 
        tmin = 10.
    m = str(int(10.*np.ceil((tmin % 60.)/10.))).zfill(2)
    if m == '60': 
        h = str(int(tmin // 60.)+1).zfill(2) 
        m = '00' 
    else: 
        h = str(int(tmin // 60.)).zfill(2) 
    if tmin <= 30.: 
        queue = 'debug'
    else:
        queue = 'regular'
    
    jname = 'plk_halo_sig8'+str(sig8)+'_'+str(nreals[0])+'_'+str(nreals[1])
    ofile = ''.join(['/global/homes/c/chahah/projects/eMaNu/run/scripts/', 
        '_plk_halo_sig8'+str(sig8)+'_'+str(nreals[0])+'_'+str(nreals[1]), '.o']) 

    jb = '\n'.join([ 
        '#!/bin/bash',
        '#SBATCH -q '+queue,
        '#SBATCH -N 1',
        '#SBATCH -t '+h+':'+m+':00', 
        '#SBATCH -J '+jname, 
        '#SBATCH -o '+ofile,  
        '', 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        '', 
        'module load python/2.7-anaconda', 
        'source activate nbdkt', 
        '',
        'sig8='+str(sig8), 
        'for ireal in {'+str(nreals[0])+'..'+str(nreals[1])+'}; do', 
        '\techo " --- sigma_8 = "$sig8", nreal= "$ireal" --- "', 
        '\tsrun -n 1 -c 1 python /global/homes/c/chahah/projects/eMaNu/run/plk.py halo_sig8 $sig8 $ireal 4 "real" 360', 
        'done',
        '', 
        'now=$(date +"%T")', 
        'echo "end time ... $now"']) 
    job = open(fjob, 'w') 
    job.write(jb) 
    job.close() 
    return fjob 


def submit_job(fjob):
    ''' run sbatch jobname.slurm 
    '''
    if not os.path.isfile(fjob): raise ValueError
    subprocess.check_output(['sbatch', fjob])
    return None 


if __name__=="__main__": 
    for mneut in [0.0, 0.06, 0.1, 0.15]: 
        fj = haloPlk_mneut_job(mneut, nreals=[1,50])
        submit_job(fj) 
        fj = haloPlk_mneut_job(mneut, nreals=[51,100])
        submit_job(fj) 
    for sig8 in [0.822, 0.818, 0.807, 0.798]: 
        fj = haloPlk_sigma8_job(sig8, nreals=[1,50])
        submit_job(fj) 
        fj = haloPlk_sigma8_job(sig8, nreals=[51,100])
        submit_job(fj) 
