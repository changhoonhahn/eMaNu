#!/bin/bash
#SBATCH -p regular 
#SBATCH -n 1
#SBATCH -t 01:15:00 
#SBATCH -J halo_pre3PCF_mlim_0.06eV
#SBATCH -o _halo_pre3PCF_mlim_0.06eV.o

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

for mneut in 0.15; do #0.10 0.15 0.6; do 
    echo " --- mneut = "$mneut" --- " 
    # halo catalogs (in serial) ~1 min/realization
    for ireal in {1..100}; do 
        srun -n 1 -c 1 python /global/homes/c/chahah/projects/eMaNu/run/threepcf.py data 1 $mneut $ireal 4 'real'
    done 
#    # halo catalogs (in parallel)
#    srun -n 1 -c 16 python /global/homes/c/chahah/projects/eMaNu/run/threepcf.py data 16 $mneut 1 100 4 'real'
done 

# construct 50 randoms
#srun -n 1 -c 1 python /global/homes/c/chahah/projects/eMaNu/run/threepcf.py randoms 4
now=$(date +"%T") 
echo "end time ... $now"
