#!/bin/bash

distr="distributions"

for d in $distr/*
do
    for p in $d/p_*
    do
        l="$d/l_${p##*_}"
        c="$d/c_${p##*_}"
        np=`wc -l $p | cut -f 1 -d\ ` # number of norms
        if [ $np -gt 1 ]
        then
            python3 multi_norm.py -u -p $p -l $l -o $c
        else
            python3 single_norm.py -u -p `cat $p` -o $c
        fi
    done
done
