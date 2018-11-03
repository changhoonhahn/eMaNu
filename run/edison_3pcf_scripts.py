'''

Code for generating and submitting 3PCF jobs on edison 


'''
import os 
import numpy as np 
import subprocess


def nnn_job(mneut, i_r, nreals=[1, 100]): 
    ''' write script to generate NNN 
    '''
    fjob = job_name('nnn', mneut, i_r, nreals=nreals)
    print('--- writing %s ---' % fjob) 
    
    # estimate the hours required 
    h = int(np.ceil(((nreals[1] - nreals[0])+1)*9./60.))

    jb = '\n'.join([ 
        '#!/bin/bash', 
        '#SBATCH -p regular', #'#SBATCH -p debug'
        '#SBATCH -n 1', 
        '#SBATCH -t '+str(h)+':00:00', #'#SBATCH -t 00:10:00'
        '#SBATCH -J halo_3PCF_'+str(mneut)+'eV'+'_'+str(nreals[0])+'_'+str(nreals[1])+'_nnn'+str(i_r), 
        '#SBATCH -o /global/homes/c/chahah/projects/eMaNu/run/3pcf/_halo_3PCF_'+str(mneut)+'eV'+'_'+str(nreals[0])+'_'+str(nreals[1])+'_nnn'+str(i_r),
        '', 
        'now=$(date +"%T")',
        'echo "start time ... $now"', 
        '', 
        'mneut='+str(mneut), 
        'i_r='+str(i_r),
        'nbin=20', 
        'nside=20', 
        'cscratch_dir=/global/cscratch1/sd/chahah/emanu/',
        'group_dir=$cscratch_dir"halos/"', 
        'thpcf_dir=$cscratch_dir"3pcf/"',
        '',
        'cmd="srun -n 1 -c 16 /global/homes/c/chahah/projects/eMaNu/emanu/zs_3pcf/grid_multipoles_nbin20 -box 1000. -scale 1 -nside 20"',
        '', 
        'for nreal in {'+str(nreals[0])+'..'+str(nreals[1])+'}; do', 
        '\techo "--- mneut="$mneut" eV nreal="$nreal" ---"',
        '\tin_ddd=$group_dir"groups."$mneut"eV."$nreal".nzbin4.rspace.dat"', 
        '\tout_ddd=$thpcf_dir"3pcf.groups."$mneut"eV."$nreal".nzbin4.nside"$nside".nbin"$nbin".rspace.ddd.dat"',
        '\tmult_ddd=$thpcf_dir"3pcf.groups."$mneut"eV."$nreal".nzbin4.nside"$nside".nbin"$nbin".rspace.ddd.mult"', 
        '', 
        '\ttmp=$group_dir"d_r"$mneut"_"$i_r".tmp"', 
        '\trand=$group_dir"groups.nzbin4.r"$i_r', 
        '\tout_nnn=$thpcf_dir"3pcf.groups."$mneut"eV."$nreal".nzbin4.nside"$nside".nbin"$nbin".rspace.nnn"$i_r".dat"', 
        '', 
        '\t# copy data to tmp', 
        '\tcp $in_ddd $tmp',
        '\t# append random to data in tmp file', 
        '\tcat $rand >> $tmp', 
        '',
        '\techo "--- calculating NNN ---"', 
        '\teval $cmd -in $tmp -load $mult_ddd -balance > $out_nnn', 
        '\trm $tmp', 
        '\techo "NNN (N = D-R) "$i_r" finished"', 
        'done', 
        'now=$(date +"%T")',
        'echo "end time ... $now"'
        ]) 
    
    job = open(fjob, 'w') 
    job.write(jb) 
    job.close() 
    return None 


def submit_job(dorr, mneut, i_r, nreals=[1,100]):
    '''
    '''
    fjob = job_name(dorr, mneut, i_r, nreals=nreals) 
    if not os.path.isfile(fjob): raise ValueError

    subprocess.check_output(['sbatch', fjob])
    return None 


def job_name(dorr, mneut, i_r, nreals=[1,100]): 
    if dorr not in ['nnn']: raise ValueError
    fjob = ''.join(['3pcf/', 
        'edison_3pcf_', str(mneut), 'eV_', dorr, '_', str(i_r), '_', str(nreals[0]), '_', str(nreals[1]), '.slurm']) 
    return fjob


if __name__=='__main__': 
    #nnn_job(0.15, 7, nreals=[99, 100])
    #submit_job('nnn', 0.15, 7, nreals=[99, 100])

    #nnn_job(0.06, 9, nreals=[92, 100])
    #submit_job('nnn', 0.06, 9, nreals=[92, 100])
    for mneut in [0.0, 0.06, 0.1, 0.15, 0.6]: 
        for i_r in range(11,20): 
            nnn_job(mneut, i_r, nreals=[1, 100])
            submit_job('nnn', mneut, i_r, nreals=[1, 100])
