#!/bin/bash
#SBATCH -p debug 
#SBATCH -n 1
#SBATCH -t 00:30:00 
#SBATCH -J 3PCF
#SBATCH -o _3PCF.o
#SBATCH -C haswell

now=$(date +"%T") 
echo "start time ... $now"

#module load python/2.7-anaconda
#source activate nbdkt 


# pre-processing 
#mneut=0.0
#nreal=1
#srun -n 1 -c 1 python /global/homes/c/chahah/projects/eMaNu/run/threepcf.py 1 $mneut $nreal 4 'real'

nbin=20
nside=20
cscratch_dir=/global/cscratch1/sd/chahah/emanu/

for mneut in 0.0; do
    nreal=1
    #for nreal in {1..1}; do 
    halo_file=$cscratch_dir"halos/groups."$mneut"eV."$nreal".nzbin4.rspace.dat"
    tpcf_file=$cscratch_dir"3pcf/3pcf.groups."$mneut"eV."$nreal".nzbin4.nside"$nside".nbin"$nbin".rspace.dat"
    srun -n 1 -c 16 /global/homes/c/chahah/projects/eMaNu/emanu/zs_3pcf/grid_multipoles_nbin20 \
        -box 1000. -scale 1 -nside $nside -in $halo_file > $tpcf_file
    #done
done

now=$(date +"%T") 
echo "end time ... $now"
