#!/bin/bash/ 
# transfer files from edison to local

local_dir="/Users/ChangHoon/data/emanu/bispectrum/"
nersc_dir="/global/homes/c/chahah/cscratch/emanu/bispectrum/"

for ireal in {81..100}; do 
    for mneut in 0.0  0.06 0.1 0.15; do 
        fbk="groups."$mneut"eV."$ireal".nzbin4.mhmin3200.0.zspace.Ngrid360.Nmax40.Ncut3.step3.pyfftw.dat"
        scp -i ~/.ssh/nersc edison:$nersc_dir$fbk $local_dir
    done 
    for sig8 in 0.798 0.807 0.818 0.822; do 
        fbk="groups.0.0eV.sig8_"$sig8"."$ireal".nzbin4.mhmin3200.0.zspace.Ngrid360.Nmax40.Ncut3.step3.pyfftw.dat"
        scp -i ~/.ssh/nersc edison:$nersc_dir$fbk $local_dir
    done 
done
