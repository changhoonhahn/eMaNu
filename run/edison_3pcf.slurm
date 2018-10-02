#!/bin/bash
#SBATCH -p regular 
#SBATCH -n 1
#SBATCH -t 06:00:00 
#SBATCH -J halo_nnn_3PCF
#SBATCH -o _halo_nnn_3PCF.o

now=$(date +"%T") 
echo "start time ... $now"

mneut=0.0
nbin=20
nside=20
cscratch_dir=/global/cscratch1/sd/chahah/emanu/
group_dir=$cscratch_dir'halos/'
thpcf_dir=$cscratch_dir'3pcf/'

# pre-processing 
#module load python/2.7-anaconda
#source activate nbdkt 
# halo catalogs (in serial)
#srun -n 1 -c 1 python /global/homes/c/chahah/projects/eMaNu/run/threepcf.py data 1 $mneut 63 4 'real'
# halo catalogs (in parallel)
#srun -n 1 -c 16 python /global/homes/c/chahah/projects/eMaNu/run/threepcf.py data 16 $mneut 1 100 4 'real'
# randoms
#srun -n 1 -c 1 python /global/homes/c/chahah/projects/eMaNu/run/threepcf.py randoms $mneut 4

for nreal in {1..25}; do 
    echo "--- mneut="$mneut" eV nreal="$nreal" ---"
    ## DDD run (~2 mins) 
    cmd='srun -n 1 -c 16 /global/homes/c/chahah/projects/eMaNu/emanu/zs_3pcf/grid_multipoles_nbin20 -box 1000. -scale 1 -nside 20'
    in_ddd=$group_dir'groups.'$mneut'eV.'$nreal'.nzbin4.rspace.dat'
    out_ddd=$thpcf_dir"3pcf.groups."$mneut"."$nreal".nzbin4.nside"$nside".nbin"$nbin".rspace.ddd.dat"
    mult_ddd=$thpcf_dir"3pcf.groups."$mneut"."$nreal".nzbin4.nside"$nside".nbin"$nbin".rspace.ddd.mult"
    eval $cmd -in $in_ddd -save $mult_ddd > $out_ddd

    now=$(date +"%T") 
    echo " DDD finished ... "$now

    ## RRR run ? 
    
    # D-R (~9 mins each) 
    for i_r in {0..0}; do 
        tmp=$group_dir'd_r.tmp'
        rand=$group_dir'groups.'$mneut'eV.nzbin4.r'$i_r
        out_d_r=$thpcf_dir"3pcf.groups."$mneut"."$nreal".nzbin4.nside"$nside".nbin"$nbin".rspace.d_r"$i_r".dat"
        # copy data to tmp 
        cp $in_ddd $tmp 
        # append random to data in tmp file 
        cat $rand >> $tmp  
    
        eval $cmd -in $tmp -load $mult_ddd > $out_d_r
        rm $tmp
        echo " NNN (N = D-R) #"$i_r" finished"
    done
done 

now=$(date +"%T") 
echo "end time ... $now"

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
