#!/bin/bash
#SBATCH -p debug 
#SBATCH -n 1
#SBATCH -t 00:05:00 
#SBATCH -J halo_3PCF
#SBATCH -o _halo_3PCF.o

module load python/2.7-anaconda
source activate nbdkt 

# pre-processing 
srun -n 1 -c 1 python /global/homes/c/chahah/projects/eMaNu/run/threepcf.py 1 0.0 1 4 'real'

#nbin=20
#nside=20
#dest_dir=/global/homes/c/chahah/projects/specmulator/dat
#for mneut in 0.0eV; do
#    for real in {10..100}; do 
#        srun -n 1 -c 16 /global/homes/c/chahah/projects/specmulator/specmulator/zslepian/grid_multipoles_nbin20 \
#            -box 1000. -scale 1 -nside \$nside -in \$dest_dir/\$mneut/\$real/groups.nzbin4.rspace.dat \
#                > \$dest_dir/\$mneut/"3pcf.groups."\$mneut"."\$real".nzbin4.nside"\$nside".nbin"\$nbin".rspace.dat"
#        srun -n 1 -c 16 /global/homes/c/chahah/projects/specmulator/specmulator/zslepian/grid_multipoles \
#            -box 1000. -scale 1 -nside \$nside -in \$dest_dir/\$mneut/\$real/groups.nzbin4.zspace.dat \
#                > \$dest_dir/\$mneut/"3pcf.groups."\$mneut"."\$real".nzbin4.nside"\$nside".nbin"\$nbin".zspace.dat"
#    done 
#done
