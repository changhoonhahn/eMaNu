#!/bin/bash
#SBATCH -p regular 
#SBATCH -n 1
#SBATCH -t 00:30:00 
#SBATCH -J Bhalo_sig8_81_90
#SBATCH -o _Bhalo_sig8_81_90.o

now=$(date +"%T") 
echo "start time ... $now"

module load python/2.7-anaconda
source activate myenv0
module load cray-fftw/3.3.8.1

for ireal in {81..90}; do 
    for mneut in 0.0 0.06 0.1 0.15; do 
        now=$(date +"%T") 
        echo "$mneut ... $ireal ... $now"
        srun -n 1 python /global/homes/c/chahah/projects/eMaNu/run/bispectrum.py mneut $mneut $ireal 4 r 
    done 
done 

#for ireal in {91..100}; do 
#    for sig8 in 0.798 0.807 0.818 0.822; do 
#        now=$(date +"%T") 
#        echo "$sig8 ... $ireal ... $now"
#        srun -n 1 python /global/homes/c/chahah/projects/eMaNu/run/bispectrum.py sigma8 $sig8 $ireal 4 r
#    done 
#done 

now=$(date +"%T") 
echo "end time ... $now"
