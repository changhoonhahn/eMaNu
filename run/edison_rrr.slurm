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

for mneut in 0.6; do #0.0 0.06 0.1 0.15 
    echo "--- mneut="$mneut" eV ---"
    cmd='srun -n 1 -c 16 /global/homes/c/chahah/projects/eMaNu/emanu/zs_3pcf/grid_multipoles_nbin20 -box 1000. -scale 1 -nside 20'
    # RRR run 
    in_rrr=$group_dir'groups.'$mneut'eV.nzbin4.r0'
    out_rrr=$thpcf_dir'3pcf.groups.'$mneut"eV.nzbin4.r0.nside"$nside".nbin"$nbin".rspace.rrr.dat"
    eval $cmd -in $in_rrr -invert > $out_rrr
done 
now=$(date +"%T") 
echo "end time ... $now"
