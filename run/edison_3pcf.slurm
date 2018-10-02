#!/bin/bash
#SBATCH -p debug 
#SBATCH -n 1
#SBATCH -t 00:30:00 
#SBATCH -J halo_random_3PCF
#SBATCH -o _halo_random_3PCF.o

mneut=0.0

# pre-processing 
module load python/2.7-anaconda
source activate nbdkt 
# halo catalogs 
srun -n 1 -c 16 python /global/homes/c/chahah/projects/eMaNu/run/threepcf.py data 16 $mneut 1 100 4 'real'
# randoms
#srun -n 1 -c 1 python /global/homes/c/chahah/projects/eMaNu/run/threepcf.py randoms $mneut 4

nbin=20
nside=20
cscratch_dir=/global/cscratch1/sd/chahah/emanu/
mneut_dir=$cscratch_dir$mneut
group_dir=$cscratch_dir$mneut/$nreal

## DDD run
#command="srun -n 1 -c 16 /global/homes/c/chahah/projects/eMaNu/emanu/zs_3pcf/grid_multipoles_nbin20 -box 1000. -scale 1 -nside 20"
#in_ddd=$group_dir'/groups.nzbin4.rspace.dat'
#out_ddd=$mneut_dir/"3pcf.groups."$mneut"."$real".nzbin4.nside"$nside".nbin"$nbin".rspace.ddd.dat"
#mult_ddd=$mneut_dir/"3pcf.groups."$mneut"."$real".nzbin4.nside"$nside".nbin"$nbin".rspace.ddd.mult"
#$command -in \$in_ddd -save \$mult_ddd > \$out_ddd
#
## RRR run ? 
#
## D-R 
#for i_r in {0..49}; do 
#    tmp=$group_dir'/d_r.tmp'
#    rand=$group_dir'groups.'$mneut'eV.nzbin4.r'$i_r
#    out_d_r=$mneut_dir/"3pcf.groups."$mneut"."$real".nzbin4.nside"$nside".nbin"$nbin".rspace.d_r.dat"
#    
#    # copy data to tmp 
#    cp $in_ddd $tmp 
#
#    # append random to data in tmp file 
#    cat $rand >> $tmp  
#
#    $command -in $tmp -load $mult_ddd > $out_d_r
#done


#srun -n 1 -c 16 /global/homes/c/chahah/projects/eMaNu/emanu/zs_3pcf/grid_multipoles_nbin20 \
#    -box 1000. -scale 1 -nside \$nside -in \$group_dir/groups.nzbin4.rspace.dat > \$mneut_dir/"3pcf.groups."\$mneut"."\$real".nzbin4.nside"\$nside".nbin"\$nbin".rspace.ddd.dat"
#
#for mneut in 0.0eV; do
#    for real in {10..100}; do 
#        srun -n 1 -c 16 /global/homes/c/chahah/projects/specmulator/specmulator/zslepian/grid_multipoles \
#            -box 1000. -scale 1 -nside \$nside -in \$dest_dir/\$mneut/\$real/groups.nzbin4.zspace.dat \
#                > \$dest_dir/\$mneut/"3pcf.groups."\$mneut"."\$real".nzbin4.nside"\$nside".nbin"\$nbin".zspace.dat"
#    done 
#done
