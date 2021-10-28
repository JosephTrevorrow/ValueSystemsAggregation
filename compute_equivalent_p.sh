#!/bin/bash

distr="distributions"

for d in $distr/*
do
    for c in $d/c_*
    do
        e="$d/e_${c##*_}"
        python3 single_norm.py -u -p 3 -i $c | grep "p = " | cut -f 3 -d\  | tail -n 1 > $e
    done
done
