#!/bin/bash
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH -J plk_halo_sig80.822_1_1
#SBATCH -o /global/homes/c/chahah/projects/eMaNu/run/script/_plk_halo_sig80.822_1_1.0

now=$(date +"%T")
echo "start time ... $now"

module load python/2.7-anaconda
source activate nbdkt

sig8=0.822
for ireal in {1..1}; do
	echo " --- mneut = "$mneut", nreal= "$ireal" --- "
	srun -n 1 -c 1 python /global/homes/c/chahah/projects/eMaNu/run/plk.py sig8 $sig8 $ireal 4 "real" 360
done

now=$(date +"%T")
echo "end time ... $now"