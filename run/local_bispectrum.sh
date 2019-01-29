#!/bin/bash/
now=$(date +"%T") 
echo "start time ... $now"

#for ireal in {51..60}; do 
#    for mneut in 0.0 0.06 0.1 0.15; do 
#        now=$(date +"%T") 
#        echo "$mneut ... $ireal ... $now"
#        python /Users/ChangHoon/projects/eMaNu/run/bispectrum.py mneut $mneut $ireal 4 r 
#    done 
#    for sig8 in 0.798 0.807 0.818 0.822; do 
#        now=$(date +"%T") 
#        echo "$sig8 ... $ireal ... $now"
#        python /Users/ChangHoon/projects/eMaNu/run/bispectrum.py sigma8 $sig8 $ireal 4 r 
#    done 
#done 

# RSD bispectrum
for ireal in {20..50}; do 
    for mneut in 0.0  0.06 0.1 0.15; do 
        now=$(date +"%T") 
        echo "$mneut ... $ireal ... $now"
        python /Users/ChangHoon/projects/eMaNu/run/bispectrum.py mneut $mneut $ireal 4 z
    done 
    for sig8 in 0.798 0.807 0.818 0.822; do 
        now=$(date +"%T") 
        echo "$sig8 ... $ireal ... $now"
        python /Users/ChangHoon/projects/eMaNu/run/bispectrum.py sigma8 $sig8 $ireal 4 z 
    done 
done 

now=$(date +"%T") 
echo "end time ... $now"
