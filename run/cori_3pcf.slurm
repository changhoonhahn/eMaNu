#!/bin/bash
#SBATCH -p debug 
#SBATCH -n 1
#SBATCH -t 00:10:00 
#SBATCH -J 3PCF
#SBATCH -o _3PCF.o
#SBATCH -C haswell

now=$(date +"%T") 
echo "start time ... $now"

#module load python/2.7-anaconda
#source activate nbdkt 


mneut=0.0
nreal=1
# pre-processing 
#srun -n 1 -c 1 python /global/homes/c/chahah/projects/eMaNu/run/threepcf.py 1 $mneut $nreal 4 'real'

nbin=20
nside=20
cscratch_dir=/global/cscratch1/sd/chahah/emanu/
mneut_dir=$cscratch_dir$mneut"eV"
group_dir=$cscratch_dir$mneut"eV"/$nreal

srun -n 1 -c 16 /global/homes/c/chahah/projects/eMaNu/emanu/zs_3pcf/grid_multipoles_nbin20 \
    -box 1000. -scale 1 -nside $nside -in $group_dir/groups.nzbin4.rspace.dat > $mneut_dir/"3pcf.groups."$mneut"eV."$nreal".nzbin4.nside"$nside".nbin"$nbin".rspace.dat"

#for mneut in 0.0eV; do
#    for real in {10..100}; do 
#        srun -n 1 -c 16 /global/homes/c/chahah/projects/specmulator/specmulator/zslepian/grid_multipoles \
#            -box 1000. -scale 1 -nside \$nside -in \$dest_dir/\$mneut/\$real/groups.nzbin4.zspace.dat \
#                > \$dest_dir/\$mneut/"3pcf.groups."\$mneut"."\$real".nzbin4.nside"\$nside".nbin"\$nbin".zspace.dat"
#    done 
#done

now=$(date +"%T") 
echo "end time ... $now"
