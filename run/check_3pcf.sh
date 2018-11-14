#!/bin/bash

for mneut in 0.0eV 0.06eV 0.1eV 0.15eV 0.6eV; do 
    for i_r in {19..19}; do 
        for ireal in {1..100}; do 
            # check if the file is there 
            nnn=$EMANU_DIR"3pcf/3pcf.groups."$mneut"."$ireal".nzbin4.nside20.nbin20.rspace.nnn"$i_r".dat"
            if [ ! -f $nnn ]; then 
                echo "No file: "$nnn
            else 
                # check file size to make sure it didn't crash in the middle 
                fsize=$(wc -c <$nnn)
                if (( $fsize < 36900 )); then
                    echo $fsize" "$nnn
                fi 
            fi 
        done
    done 
done 
