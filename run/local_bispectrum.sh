#!/bin/bash/
now=$(date +"%T") 
echo "start time ... $now"

for ireal in {6..10}; do 
    for mneut in 0.0 0.06 0.1 0.15; do 
        echo "$mneut ... $ireal"
        python /Users/ChangHoon/projects/eMaNu/run/bispectrum.py $mneut $ireal 4
    done 
done 

now=$(date +"%T") 
echo "end time ... $now"
