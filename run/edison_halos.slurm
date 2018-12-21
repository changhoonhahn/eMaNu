#!/bin/bash
#SBATCH -p regular 
#SBATCH -n 1
#SBATCH -t 02:00:00 
#SBATCH -J halo_mlim_sig8_0822
#SBATCH -o _halo_mlim_sig8_0822.o

now=$(date +"%T") 
echo "start time ... $now"

cscratch_dir=/global/cscratch1/sd/chahah/emanu/
group_dir=$cscratch_dir'halos/'

module load python/2.7-anaconda
source activate nbdkt 

#for mneut in 0.0 0.10 0.15 0.6; do 
#    for ireal in {1..100}; do 
#        echo " --- mneut = "$mneut", nreal= "$ireal" --- " 
#        srun -n 1 -c 1 python /global/homes/c/chahah/projects/eMaNu/run/halos.py mneut 1 $mneut $ireal 4 'real'
#    done 
#done 

for sig8 in 0.822; do  #0.818 0.807 0.798
    for ireal in {1..100}; do 
        echo " --- sigma8_c="$sig8"; nreal= "$ireal" --- " 
        srun -n 1 -c 1 python /global/homes/c/chahah/projects/eMaNu/run/halos.py sig8 1 $sig8 $ireal 4 'real'
    done 
done 

now=$(date +"%T") 
echo "end time ... $now"
