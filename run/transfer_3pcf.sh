#!/bin/bash
#echo "password" 
#read -s pwd 
i_r=$1
ireali=1
irealf=100
for mneut in 0.0eV 0.06eV 0.1eV 0.15eV 0.6eV; do 
    #echo "edison:/global/homes/c/chahah/cscratch/emanu/3pcf/3pcf.groups."$mneut".{1..100}.nzbin4.nside20.nbin20.rspace.nnn"$i_r".dat"
    scp edison:"/global/homes/c/chahah/cscratch/emanu/3pcf/3pcf.groups."$mneut".{1..100}.nzbin4.nside20.nbin20.rspace.nnn"$i_r".dat" $EMANU_DIR"3pcf/"
    #for ireal in $(seq $ireali $irealf); do #{ $ireali..$irealf }; do 
    #    name_i=''
    #    if [ $ireal == $irealf ]; then
    #        cmd=$cmd$name_i"}"
    #    else
    #        cmd=$cmd$name_i","
    #    fi
    #done 
done 
#scp $cmd $EMANU_DIR"3pcf/"
