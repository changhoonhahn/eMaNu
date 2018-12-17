#!/bin/bash/
now=$(date +"%T") 
echo "start time ... $now"

for mneut in 0.06 0.1 0.15; do 
    for ireal in {1..1}; do 
        echo "$mneut ... $ireal"
        python /Users/ChangHoon/projects/eMaNu/run/bispectrum.py $mneut $ireal 4
    done 
done 

now=$(date +"%T") 
echo "end time ... $now"
