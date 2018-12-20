#!/bin/bash
#SBATCH -p debug 
#SBATCH -n 1
#SBATCH -t 00:10:00 
#SBATCH -J plk_halo_sig8_0822
#SBATCH -o _plk_halo_sig8_0822.o

now=$(date +"%T") 
echo "start time ... $now"

cscratch_dir=/global/cscratch1/sd/chahah/emanu/
group_dir=$cscratch_dir'halos/'

module load python/2.7-anaconda
source activate nbdkt 

for mneut in 0.0; do # 0.10 0.15 0.6; do 
    for ireal in {1..1}; do 
        echo " --- mneut = "$mneut", nreal= "$ireal" --- " 
        srun -n 1 -c 1 python /global/homes/c/chahah/projects/eMaNu/run/plk.py halo $mneut $ireal 4 'real' 360
    done 
done 
