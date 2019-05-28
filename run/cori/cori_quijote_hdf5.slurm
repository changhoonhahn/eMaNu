#!/bin/bash
#SBATCH -p regular 
#SBATCH -n 1
#SBATCH -t 10:00:00 
#SBATCH -J quijote_hdf5
#SBATCH -o _quijote_hdf5.o
#SBATCH -C haswell

now=$(date +"%T") 
echo "start time ... $now"

module load python/2.7-anaconda
source activate myenv0 

srun -n 1 python /global/homes/c/chahah/projects/eMaNu/run/quijote_hdf5.py

now=$(date +"%T") 
echo "end time ... $now"
