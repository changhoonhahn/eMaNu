#!/bin/bash/

for mneut in 0.06 0.1 0.15; do 
    for ireal in {1..1}; do 
        echo "$mneut ... $ireal"
        python /Users/ChangHoon/projects/eMaNu/run/bispectrum.py $mneut $ireal 4
    done 
done 
