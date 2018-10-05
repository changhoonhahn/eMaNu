#!/bin/bash
#SBATCH -p debug 
#SBATCH -n 1
#SBATCH -t 00:05:00 
#SBATCH -J halo_rrr
#SBATCH -o _halo_rrr.o

now=$(date +"%T") 
echo "start time ... $now"

nbin=20
nside=20
cscratch_dir=/global/cscratch1/sd/chahah/emanu/
group_dir=$cscratch_dir'halos/'
thpcf_dir=$cscratch_dir'3pcf/'

echo "--- calculating RRR ---"
cmd='srun -n 1 -c 16 /global/homes/c/chahah/projects/eMaNu/emanu/zs_3pcf/grid_multipoles_nbin20 -box 1000. -scale 1 -nside 20'
# RRR run 
in_rrr=$group_dir'groups.nzbin4.r0'
out_rrr=$thpcf_dir"3pcf.groups.nzbin4.r0.nside"$nside".nbin"$nbin".rrr.dat"
eval $cmd -in $in_rrr -invert > $out_rrr

now=$(date +"%T") 
echo "end time ... $now"
