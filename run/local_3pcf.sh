#!/bin/bash

mneut=0.0
# pre-processing 
for nreal in {63..63}; do #{1..100}; do 
    python threepcf.py data 1 $mneut $nreal 4 'real'
    #python threepcf.py 1 $mneut $nreal 4 'z'
done

#nbin=20
#nside=20
#cscratch_dir=/global/cscratch1/sd/chahah/emanu/
#mneut_dir=$cscratch_dir$mneut
#group_dir=$cscratch_dir$mneut/$nreal
#
#srun -n 1 -c 16 /global/homes/c/chahah/projects/eMaNu/emanu/zs_3pcf/grid_multipoles_nbin20 \
#    -box 1000. -scale 1 -nside \$nside -in \$group_dir/groups.nzbin4.rspace.dat > \$mneut_dir/"3pcf.groups."\$mneut"."\$real".nzbin4.nside"\$nside".nbin"\$nbin".rspace.dat"
#
#for mneut in 0.0eV; do
#    for real in {10..100}; do 
#        srun -n 1 -c 16 /global/homes/c/chahah/projects/specmulator/specmulator/zslepian/grid_multipoles \
#            -box 1000. -scale 1 -nside \$nside -in \$dest_dir/\$mneut/\$real/groups.nzbin4.zspace.dat \
#                > \$dest_dir/\$mneut/"3pcf.groups."\$mneut"."\$real".nzbin4.nside"\$nside".nbin"\$nbin".zspace.dat"
#    done 
#done
