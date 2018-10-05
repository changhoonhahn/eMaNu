#!/bin/bash
#SBATCH -p regular 
#SBATCH -n 1
#SBATCH -t 01:30:00 
#SBATCH -J halo_3PCF_0.0eV
#SBATCH -o _halo_3PCF_0.0eV.o

now=$(date +"%T") 
echo "start time ... $now"

mneut=0.0
nbin=20
nside=20
cscratch_dir=/global/cscratch1/sd/chahah/emanu/
group_dir=$cscratch_dir'halos/'
thpcf_dir=$cscratch_dir'3pcf/'

for nreal in {1..10}; do 
    echo "--- mneut="$mneut" eV nreal="$nreal" ---"
    ## DDD run (~2 mins) 
    cmd='srun -n 1 -c 16 /global/homes/c/chahah/projects/eMaNu/emanu/zs_3pcf/grid_multipoles_nbin20 -box 1000. -scale 1 -nside 20'
    in_ddd=$group_dir'groups.'$mneut'eV.'$nreal'.nzbin4.rspace.dat'
    out_ddd=$thpcf_dir"3pcf.groups."$mneut"eV."$nreal".nzbin4.nside"$nside".nbin"$nbin".rspace.ddd.dat"
    mult_ddd=$thpcf_dir"3pcf.groups."$mneut"eV."$nreal".nzbin4.nside"$nside".nbin"$nbin".rspace.ddd.mult"
    #echo "--- calculating DDD ---"
    #eval $cmd -in $in_ddd -save $mult_ddd > $out_ddd
    #now=$(date +"%T") 
    #echo " DDD finished ... "$now
    
    # NNN (~9 mins each) 
    for i_r in {0..0}; do 
        tmp=$group_dir'd_r'$mneut'.tmp'
        rand=$group_dir'groups.nzbin4.r'$i_r
        out_nnn=$thpcf_dir"3pcf.groups."$mneut"eV."$nreal".nzbin4.nside"$nside".nbin"$nbin".rspace.nnn"$i_r".dat"

        # copy data to tmp 
        cp $in_ddd $tmp 
        # append random to data in tmp file 
        cat $rand >> $tmp  
    
        echo "--- calculating NNN ---"
        eval $cmd -in $tmp -load $mult_ddd -balance > $out_nnn
        rm $tmp
        echo " NNN (N = D-R) #"$i_r" finished"
    done
done 

now=$(date +"%T") 
echo "end time ... $now"
