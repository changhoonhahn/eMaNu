#!/bin/bash

now=$(date +"%T") 
echo "start time ... $now"

for ireal in {1..100}; do 
    for mneut in 0.0 0.06 0.10 0.15; do 
        echo "--- mneut = "$mneut", nreal= "$ireal" ---" 
        #python /Users/ChangHoon/projects/eMaNu/run/plk.py halo $mneut $ireal 4 'r' 360
        python /Users/ChangHoon/projects/eMaNu/run/plk.py halo $mneut $ireal 4 'z' 360
    done 
    for sig8 in 0.822 0.818 0.807 0.798; do 
        echo "--- sigma8_c="$sig8"; nreal= "$ireal" ---" 
        #python /Users/ChangHoon/projects/eMaNu/run/plk.py halo_sig8 $sig8 $ireal 4 'r' 360
        python /Users/ChangHoon/projects/eMaNu/run/plk.py halo_sig8 $sig8 $ireal 4 'z' 360
    done 
done 


now=$(date +"%T") 
echo "end time ... $now"
