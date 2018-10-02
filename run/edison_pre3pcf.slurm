#!/bin/bash
#SBATCH -p regular 
#SBATCH -n 1
#SBATCH -t 01:00:00 
#SBATCH -J halo_pre3PCF
#SBATCH -o _halo_pre3PCF.o

now=$(date +"%T") 
echo "start time ... $now"

nbin=20
nside=20
cscratch_dir=/global/cscratch1/sd/chahah/emanu/
group_dir=$cscratch_dir'halos/'
thpcf_dir=$cscratch_dir'3pcf/'

# pre-processing 
module load python/2.7-anaconda
source activate nbdkt 

for mneut in 0.10 0.15 0.6; do 
    echo " --- mneut = "$mneut" --- " 
    # halo catalogs (in serial)
    #srun -n 1 -c 1 python /global/homes/c/chahah/projects/eMaNu/run/threepcf.py data 1 $mneut 63 4 'real'
    # halo catalogs (in parallel)
    srun -n 1 -c 16 python /global/homes/c/chahah/projects/eMaNu/run/threepcf.py data 16 $mneut 1 100 4 'real'
    # randoms
    srun -n 1 -c 1 python /global/homes/c/chahah/projects/eMaNu/run/threepcf.py randoms $mneut 4
done 

now=$(date +"%T") 
echo "end time ... $now"
