#!/bin/bash
#SBATCH -p debug 
#SBATCH -n 1
#SBATCH -t 00:10:00 
#SBATCH -J halo_rrr_3PCF_0.6eV
#SBATCH -o _halo_rrr_3PCF_0.6eV.o

now=$(date +"%T") 
echo "start time ... $now"

mneut=0.6
nbin=20
nside=20
cscratch_dir=/global/cscratch1/sd/chahah/emanu/
group_dir=$cscratch_dir'halos/'
thpcf_dir=$cscratch_dir'3pcf/'

for nreal in {1..1}; do 
    echo "--- mneut="$mneut" eV nreal="$nreal" ---"
    ## DDD run (~2 mins) 
    cmd='srun -n 1 -c 16 /global/homes/c/chahah/projects/eMaNu/emanu/zs_3pcf/grid_multipoles_nbin20 -box 1000. -scale 1 -nside 20'
    in_ddd=$group_dir'groups.'$mneut'eV.'$nreal'.nzbin4.rspace.dat'
    out_ddd=$thpcf_dir"3pcf.groups."$mneut"eV."$nreal".nzbin4.nside"$nside".nbin"$nbin".rspace.ddd.dat"
    mult_ddd=$thpcf_dir"3pcf.groups."$mneut"eV."$nreal".nzbin4.nside"$nside".nbin"$nbin".rspace.ddd.mult"
    #eval $cmd -in $in_ddd -save $mult_ddd > $out_ddd
    #now=$(date +"%T") 
    #echo " DDD finished ... "$now

    # RRR run 
    in_rrr=$group_dir'groups.'$mneut'eV.nzbin4.r0'
    out_rrr=$thpcf_dir'3pcf.groups.'$mneut'eV.nzbin4.r0.nside"$nside".nbin"$nbin".rspace.ddd.dat'
    eval $cmd -in $in_rrr -invert > $out_rrr
    
    # D-R (~9 mins each) 
    #for i_r in {0..0}; do 
    #    tmp=$group_dir'd_r'$mneut'.tmp'
    #    rand=$group_dir'groups.'$mneut'eV.nzbin4.r'$i_r
    #    out_d_r=$thpcf_dir"3pcf.groups."$mneut"eV."$nreal".nzbin4.nside"$nside".nbin"$nbin".rspace.nnn"$i_r".dat"
    #    # copy data to tmp 
    #    cp $in_ddd $tmp 
    #    # append random to data in tmp file 
    #    cat $rand >> $tmp  
    #
    #    eval $cmd -in $tmp -load $mult_ddd > $out_d_r
    #    rm $tmp
    #    echo " NNN (N = D-R) #"$i_r" finished"
    #done
done 

now=$(date +"%T") 
echo "end time ... $now"
